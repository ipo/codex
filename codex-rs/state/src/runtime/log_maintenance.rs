use crate::SqliteConfig;
use crate::sqlite::open_owner_only_lock_file;
use crate::telemetry::DbTelemetry;
use anyhow::Context;
use chrono::Utc;
use sqlx::Connection;
use sqlx::SqliteConnection;
use std::fs::File;
use std::fs::TryLockError;
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
            Self::Partial | Self::Busy | Self::SkippedLocked | Self::SkippedRecent => {
                StampKind::Retry
            }
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

struct MaintenanceLease {
    file: File,
}

pub(super) fn spawn_logs_maintenance(sqlite: SqliteConfig) -> JoinHandle<()> {
    tokio::spawn(async move {
        let logs_path = sqlite.logs_db_path();
        if let Err(err) = run_logs_maintenance(
            &sqlite,
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
    sqlite: &SqliteConfig,
    telemetry_override: Option<&dyn DbTelemetry>,
    now: i64,
    limits: MaintenanceLimits,
) -> anyhow::Result<LogMaintenanceOutcome> {
    let started = Instant::now();
    let result = run_logs_maintenance_inner(sqlite, now, limits).await;
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
        checkpoint.map(|result| (result.busy, result.log_frames, result.checkpointed_frames)),
    );
    result
}

async fn run_logs_maintenance_inner(
    sqlite: &SqliteConfig,
    now: i64,
    limits: MaintenanceLimits,
) -> anyhow::Result<LogMaintenanceOutcome> {
    let Some(mut lease) = try_acquire_maintenance_lease(sqlite.logs_db_path().as_path())? else {
        return Ok(outcome(MaintenanceStatus::SkippedLocked, 0));
    };
    if lease.is_recent(now, limits)? {
        return Ok(outcome(MaintenanceStatus::SkippedRecent, 0));
    }

    let result = perform_logs_maintenance(sqlite, now, limits).await;
    let stamp_kind = result
        .as_ref()
        .map_or(StampKind::Retry, |result| result.status.stamp_kind());
    lease
        .write_stamp(stamp_kind, now)
        .context("failed to update log maintenance stamp")?;
    result
}

async fn perform_logs_maintenance(
    sqlite: &SqliteConfig,
    now: i64,
    limits: MaintenanceLimits,
) -> anyhow::Result<LogMaintenanceOutcome> {
    let mut connection = match sqlite
        .open_log_maintenance_connection(limits.busy_timeout)
        .await
    {
        Ok(connection) => connection,
        Err(err) if sqlite_error_is_lock(&err) => {
            return Ok(outcome(MaintenanceStatus::Busy, 0));
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
            return Ok(outcome(MaintenanceStatus::Partial, deleted_rows));
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
                return Ok(outcome(MaintenanceStatus::Busy, deleted_rows));
            }
            Err(err) => return Err(err.into()),
        };
        deleted_rows = deleted_rows.saturating_add(batch_deleted);
        if batch_deleted < batch_rows {
            break;
        }
        tokio::task::yield_now().await;
    }

    let passive_checkpoint = match checkpoint(connection, "PRAGMA wal_checkpoint(PASSIVE)").await {
        Ok(result) => result,
        Err(err) if sqlite_error_is_lock(&err) => {
            return Ok(outcome(MaintenanceStatus::Busy, deleted_rows));
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
    let truncate_checkpoint = match checkpoint(connection, "PRAGMA wal_checkpoint(TRUNCATE)").await
    {
        Ok(result) => result,
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
    Ok(LogMaintenanceOutcome {
        status: if truncate_checkpoint.busy == 0 {
            MaintenanceStatus::Complete
        } else {
            MaintenanceStatus::Busy
        },
        deleted_rows,
        passive_checkpoint: Some(passive_checkpoint),
        truncate_checkpoint: Some(truncate_checkpoint),
    })
}

fn outcome(status: MaintenanceStatus, deleted_rows: u64) -> LogMaintenanceOutcome {
    LogMaintenanceOutcome {
        status,
        deleted_rows,
        passive_checkpoint: None,
        truncate_checkpoint: None,
    }
}

async fn checkpoint(
    connection: &mut SqliteConnection,
    pragma: &'static str,
) -> Result<CheckpointResult, sqlx::Error> {
    let (busy, log_frames, checkpointed_frames) = sqlx::query_as::<_, (i64, i64, i64)>(pragma)
        .fetch_one(connection)
        .await?;
    Ok(CheckpointResult {
        busy,
        log_frames,
        checkpointed_frames,
    })
}

fn try_acquire_maintenance_lease(logs_path: &Path) -> anyhow::Result<Option<MaintenanceLease>> {
    let file = open_owner_only_lock_file(maintenance_path(logs_path).as_path())?;
    match file.try_lock() {
        Ok(()) => Ok(Some(MaintenanceLease { file })),
        Err(TryLockError::WouldBlock) => Ok(None),
        Err(TryLockError::Error(err)) => Err(err.into()),
    }
}

pub(super) fn maintenance_path(logs_path: &Path) -> PathBuf {
    let mut path = logs_path.as_os_str().to_os_string();
    path.push(".maintenance");
    PathBuf::from(path)
}

impl MaintenanceLease {
    fn is_recent(&mut self, now: i64, limits: MaintenanceLimits) -> io::Result<bool> {
        self.file.seek(SeekFrom::Start(0))?;
        let mut contents = String::new();
        self.file.read_to_string(&mut contents)?;
        let mut parts = contents.split_whitespace();
        let interval = match parts.next() {
            Some("complete") => limits.success_interval_seconds,
            Some("retry") => limits.retry_interval_seconds,
            _ => return Ok(false),
        };
        let Some(recorded_at) = parts.next().and_then(|value| value.parse::<i64>().ok()) else {
            return Ok(false);
        };
        Ok(now.saturating_sub(recorded_at) < interval)
    }

    fn write_stamp(&mut self, kind: StampKind, now: i64) -> io::Result<()> {
        self.file.seek(SeekFrom::Start(0))?;
        self.file.set_len(0)?;
        writeln!(self.file, "{} {now}", kind.as_str())?;
        self.file.flush()
    }
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
