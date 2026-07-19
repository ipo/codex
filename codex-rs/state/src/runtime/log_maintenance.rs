use crate::telemetry::DbTelemetry;
use anyhow::Context;
use chrono::Utc;
use log::LevelFilter;
use sqlx::ConnectOptions;
use sqlx::Connection;
use sqlx::SqliteConnection;
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::sqlite::SqliteSynchronous;
use std::fs::File;
use std::fs::OpenOptions;
use std::io;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;
use tokio::task::JoinHandle;
use tracing::warn;

const LOG_RETENTION_DAYS: i64 = 10;
const DELETE_BATCH_ROWS: u64 = 10_000;
const MAX_DELETED_ROWS: u64 = 100_000;
const MAX_MAINTENANCE_DURATION: Duration = Duration::from_secs(2);
const MAINTENANCE_BUSY_TIMEOUT: Duration = Duration::from_millis(100);
const SUCCESS_INTERVAL_SECONDS: i64 = 60 * 60;
const RETRY_INTERVAL_SECONDS: i64 = 5 * 60;

#[derive(Clone, Copy)]
pub(super) struct MaintenanceLimits {
    delete_batch_rows: u64,
    max_deleted_rows: u64,
    max_duration: Duration,
    busy_timeout: Duration,
    success_interval_seconds: i64,
    retry_interval_seconds: i64,
}

impl MaintenanceLimits {
    const PRODUCTION: Self = Self {
        delete_batch_rows: DELETE_BATCH_ROWS,
        max_deleted_rows: MAX_DELETED_ROWS,
        max_duration: MAX_MAINTENANCE_DURATION,
        busy_timeout: MAINTENANCE_BUSY_TIMEOUT,
        success_interval_seconds: SUCCESS_INTERVAL_SECONDS,
        retry_interval_seconds: RETRY_INTERVAL_SECONDS,
    };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MaintenanceStatus {
    Complete,
    Partial,
    Busy,
    SkippedLocked,
    SkippedRecent,
}

impl MaintenanceStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::Partial => "partial",
            Self::Busy => "busy",
            Self::SkippedLocked => "skipped_locked",
            Self::SkippedRecent => "skipped_recent",
        }
    }

    fn stamp_kind(self) -> StampKind {
        match self {
            Self::Complete => StampKind::Complete,
            Self::Partial | Self::Busy => StampKind::Retry,
            Self::SkippedLocked | Self::SkippedRecent => StampKind::Retry,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CheckpointResult {
    pub busy: i64,
    pub log_frames: i64,
    pub checkpointed_frames: i64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct LogMaintenanceOutcome {
    pub status: MaintenanceStatus,
    pub deleted_rows: u64,
    pub passive_checkpoint: Option<CheckpointResult>,
    pub truncate_checkpoint: Option<CheckpointResult>,
}

#[derive(Clone, Copy)]
enum StampKind {
    Complete,
    Retry,
}

impl StampKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::Retry => "retry",
        }
    }
}

#[derive(Clone, Copy)]
struct MaintenanceStamp {
    kind: StampKind,
    recorded_at: i64,
}

struct MaintenanceLease {
    file: File,
}

pub(super) fn spawn_logs_maintenance(logs_path: PathBuf) -> JoinHandle<()> {
    tokio::spawn(async move {
        if let Err(err) = run_logs_maintenance(
            logs_path.as_path(),
            /*telemetry_override*/ None,
            Utc::now().timestamp(),
            MaintenanceLimits::PRODUCTION,
        )
        .await
        {
            warn!(
                "background maintenance failed for logs db at {}: {err}",
                logs_path.display()
            );
        }
    })
}

pub(super) async fn run_logs_maintenance(
    logs_path: &Path,
    telemetry_override: Option<&dyn DbTelemetry>,
    now: i64,
    limits: MaintenanceLimits,
) -> anyhow::Result<LogMaintenanceOutcome> {
    let started = Instant::now();
    let result = run_logs_maintenance_inner(logs_path, now, limits).await;
    let (status, deleted_rows, checkpoint) = match &result {
        Ok(outcome) => (
            outcome.status.as_str(),
            outcome.deleted_rows,
            outcome.passive_checkpoint,
        ),
        Err(_) => ("failed", 0, None),
    };
    crate::telemetry::record_maintenance_result(
        telemetry_override,
        status,
        started.elapsed(),
        deleted_rows,
        checkpoint.map(|checkpoint| {
            (
                checkpoint.busy,
                checkpoint.log_frames,
                checkpoint.checkpointed_frames,
            )
        }),
    );
    result
}

async fn run_logs_maintenance_inner(
    logs_path: &Path,
    now: i64,
    limits: MaintenanceLimits,
) -> anyhow::Result<LogMaintenanceOutcome> {
    let Some(mut lease) = try_acquire_maintenance_lease(logs_path)? else {
        return Ok(LogMaintenanceOutcome {
            status: MaintenanceStatus::SkippedLocked,
            deleted_rows: 0,
            passive_checkpoint: None,
            truncate_checkpoint: None,
        });
    };
    if lease.is_recent(now, limits)? {
        return Ok(LogMaintenanceOutcome {
            status: MaintenanceStatus::SkippedRecent,
            deleted_rows: 0,
            passive_checkpoint: None,
            truncate_checkpoint: None,
        });
    }

    let result = perform_logs_maintenance(logs_path, now, limits).await;
    let stamp_kind = result
        .as_ref()
        .map_or(StampKind::Retry, |outcome| outcome.status.stamp_kind());
    lease
        .write_stamp(stamp_kind, now)
        .context("failed to update log maintenance stamp")?;
    result
}

async fn perform_logs_maintenance(
    logs_path: &Path,
    now: i64,
    limits: MaintenanceLimits,
) -> anyhow::Result<LogMaintenanceOutcome> {
    let options = SqliteConnectOptions::new()
        .filename(logs_path)
        .create_if_missing(false)
        .synchronous(SqliteSynchronous::Normal)
        .busy_timeout(limits.busy_timeout)
        .log_statements(LevelFilter::Off);
    let mut connection = match SqliteConnection::connect_with(&options).await {
        Ok(connection) => connection,
        Err(err) if sqlite_error_is_lock(&err) => {
            return Ok(LogMaintenanceOutcome {
                status: MaintenanceStatus::Busy,
                deleted_rows: 0,
                passive_checkpoint: None,
                truncate_checkpoint: None,
            });
        }
        Err(err) => return Err(err.into()),
    };
    let result = perform_logs_maintenance_with_connection(&mut connection, now, limits).await;
    let close_result = connection.close().await;
    match (result, close_result) {
        (Err(err), _) => Err(err),
        (Ok(_), Err(err)) => Err(err.into()),
        (Ok(outcome), Ok(())) => Ok(outcome),
    }
}

async fn perform_logs_maintenance_with_connection(
    connection: &mut SqliteConnection,
    now: i64,
    limits: MaintenanceLimits,
) -> anyhow::Result<LogMaintenanceOutcome> {
    let cutoff = now
        .checked_sub(LOG_RETENTION_DAYS * 24 * 60 * 60)
        .context("log retention cutoff is outside the supported timestamp range")?;
    let started = Instant::now();
    let mut deleted_rows = 0_u64;
    loop {
        if deleted_rows >= limits.max_deleted_rows || started.elapsed() >= limits.max_duration {
            return Ok(LogMaintenanceOutcome {
                status: MaintenanceStatus::Partial,
                deleted_rows,
                passive_checkpoint: None,
                truncate_checkpoint: None,
            });
        }
        let batch_rows = limits
            .delete_batch_rows
            .min(limits.max_deleted_rows - deleted_rows);
        let result = sqlx::query(
            "DELETE FROM logs WHERE id IN (SELECT id FROM logs WHERE ts < ? ORDER BY ts, id LIMIT ?)",
        )
        .bind(cutoff)
        .bind(i64::try_from(batch_rows).unwrap_or(i64::MAX))
        .execute(&mut *connection)
        .await;
        let batch_deleted = match result {
            Ok(result) => result.rows_affected(),
            Err(err) if sqlite_error_is_lock(&err) => {
                return Ok(LogMaintenanceOutcome {
                    status: MaintenanceStatus::Busy,
                    deleted_rows,
                    passive_checkpoint: None,
                    truncate_checkpoint: None,
                });
            }
            Err(err) => return Err(err.into()),
        };
        deleted_rows = deleted_rows.saturating_add(batch_deleted);
        if batch_deleted < batch_rows {
            break;
        }
        tokio::task::yield_now().await;
    }

    let passive_checkpoint = match checkpoint_passive(connection).await {
        Ok(checkpoint) => checkpoint,
        Err(err) if sqlite_error_is_lock(&err) => {
            return Ok(LogMaintenanceOutcome {
                status: MaintenanceStatus::Busy,
                deleted_rows,
                passive_checkpoint: None,
                truncate_checkpoint: None,
            });
        }
        Err(err) => return Err(err.into()),
    };
    if passive_checkpoint.busy != 0
        || passive_checkpoint.log_frames != passive_checkpoint.checkpointed_frames
    {
        return Ok(LogMaintenanceOutcome {
            status: MaintenanceStatus::Partial,
            deleted_rows,
            passive_checkpoint: Some(passive_checkpoint),
            truncate_checkpoint: None,
        });
    }

    sqlx::query("PRAGMA busy_timeout = 0")
        .execute(&mut *connection)
        .await?;
    let truncate_checkpoint = match checkpoint_truncate(connection).await {
        Ok(checkpoint) => checkpoint,
        Err(err) if sqlite_error_is_lock(&err) => {
            return Ok(LogMaintenanceOutcome {
                status: MaintenanceStatus::Busy,
                deleted_rows,
                passive_checkpoint: Some(passive_checkpoint),
                truncate_checkpoint: None,
            });
        }
        Err(err) => return Err(err.into()),
    };
    let status = if truncate_checkpoint.busy == 0 {
        MaintenanceStatus::Complete
    } else {
        MaintenanceStatus::Busy
    };
    Ok(LogMaintenanceOutcome {
        status,
        deleted_rows,
        passive_checkpoint: Some(passive_checkpoint),
        truncate_checkpoint: Some(truncate_checkpoint),
    })
}

async fn checkpoint_passive(
    connection: &mut SqliteConnection,
) -> Result<CheckpointResult, sqlx::Error> {
    let (busy, log_frames, checkpointed_frames) =
        sqlx::query_as::<_, (i64, i64, i64)>("PRAGMA wal_checkpoint(PASSIVE)")
            .fetch_one(connection)
            .await?;
    Ok(CheckpointResult {
        busy,
        log_frames,
        checkpointed_frames,
    })
}

async fn checkpoint_truncate(
    connection: &mut SqliteConnection,
) -> Result<CheckpointResult, sqlx::Error> {
    let (busy, log_frames, checkpointed_frames) =
        sqlx::query_as::<_, (i64, i64, i64)>("PRAGMA wal_checkpoint(TRUNCATE)")
            .fetch_one(connection)
            .await?;
    Ok(CheckpointResult {
        busy,
        log_frames,
        checkpointed_frames,
    })
}

fn try_acquire_maintenance_lease(logs_path: &Path) -> anyhow::Result<Option<MaintenanceLease>> {
    let path = maintenance_path(logs_path);
    let file = open_owner_only_file(path.as_path())?;
    match fs2::FileExt::try_lock_exclusive(&file) {
        Ok(()) => Ok(Some(MaintenanceLease { file })),
        Err(err) if lock_is_contended(&err) => Ok(None),
        Err(err) => Err(err.into()),
    }
}

pub(super) fn maintenance_path(logs_path: &Path) -> PathBuf {
    let mut path = logs_path.as_os_str().to_os_string();
    path.push(".maintenance");
    PathBuf::from(path)
}

pub(super) fn open_owner_only_file(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options.open(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(file)
}

impl MaintenanceLease {
    fn is_recent(&mut self, now: i64, limits: MaintenanceLimits) -> io::Result<bool> {
        let Some(stamp) = self.read_stamp()? else {
            return Ok(false);
        };
        let interval = match stamp.kind {
            StampKind::Complete => limits.success_interval_seconds,
            StampKind::Retry => limits.retry_interval_seconds,
        };
        Ok(now.saturating_sub(stamp.recorded_at) < interval)
    }

    fn read_stamp(&mut self) -> io::Result<Option<MaintenanceStamp>> {
        self.file.seek(SeekFrom::Start(0))?;
        let mut contents = String::new();
        self.file.read_to_string(&mut contents)?;
        let mut parts = contents.split_whitespace();
        let kind = match parts.next() {
            Some("complete") => StampKind::Complete,
            Some("retry") => StampKind::Retry,
            _ => return Ok(None),
        };
        let Some(recorded_at) = parts.next().and_then(|value| value.parse::<i64>().ok()) else {
            return Ok(None);
        };
        Ok(Some(MaintenanceStamp { kind, recorded_at }))
    }

    fn write_stamp(&mut self, kind: StampKind, now: i64) -> io::Result<()> {
        self.file.seek(SeekFrom::Start(0))?;
        self.file.set_len(0)?;
        writeln!(self.file, "{} {now}", kind.as_str())?;
        self.file.flush()
    }
}

fn lock_is_contended(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::WouldBlock
        || error.raw_os_error() == fs2::lock_contended_error().raw_os_error()
}

fn sqlite_error_is_lock(error: &sqlx::Error) -> bool {
    let sqlx::Error::Database(database_error) = error else {
        return false;
    };
    if super::sqlite_error_detail_is_lock(database_error.message()) {
        return true;
    }
    let Some(code) = database_error.code() else {
        return false;
    };
    let code = code.to_ascii_lowercase();
    code.parse::<i32>()
        .is_ok_and(|code| matches!(code & 0xff, 5 | 6))
        || matches!(code.as_str(), "sqlite_busy" | "sqlite_locked")
}

#[cfg(test)]
#[path = "log_maintenance_tests.rs"]
mod tests;
