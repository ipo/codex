use super::super::StateRuntime;
use super::super::test_support::unique_temp_dir;
use super::CheckpointResult;
use super::LogMaintenanceOutcome;
use super::MaintenanceLimits;
use super::MaintenanceStatus;
use super::maintenance_path;
use super::open_owner_only_file;
use super::run_logs_maintenance;
use crate::DB_MAINTENANCE_BUSY_METRIC;
use crate::DB_MAINTENANCE_DELETED_ROWS_METRIC;
use crate::DB_MAINTENANCE_DURATION_METRIC;
use crate::DB_MAINTENANCE_METRIC;
use crate::DB_MAINTENANCE_WAL_FRAMES_METRIC;
use crate::DbTelemetry;
use crate::LOGS_DB_FILENAME;
use crate::migrations::LOGS_MIGRATOR;
use chrono::Utc;
use pretty_assertions::assert_eq;
use sqlx::ConnectOptions;
use sqlx::Connection;
use sqlx::Row;
use sqlx::SqliteConnection;
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::sqlite::SqliteJournalMode;
use sqlx::sqlite::SqliteSynchronous;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Duration;

fn test_limits() -> MaintenanceLimits {
    MaintenanceLimits {
        delete_batch_rows: 2,
        max_deleted_rows: 3,
        max_duration: Duration::from_secs(2),
        busy_timeout: Duration::from_millis(25),
        success_interval_seconds: 60 * 60,
        retry_interval_seconds: 5 * 60,
    }
}

async fn create_logs_db() -> (PathBuf, SqliteConnection) {
    let codex_home = unique_temp_dir();
    tokio::fs::create_dir_all(&codex_home)
        .await
        .expect("create Codex home");
    let logs_path = codex_home.join(LOGS_DB_FILENAME);
    let options = SqliteConnectOptions::new()
        .filename(&logs_path)
        .create_if_missing(true)
        .journal_mode(SqliteJournalMode::Wal)
        .synchronous(SqliteSynchronous::Off)
        .log_statements(log::LevelFilter::Off);
    let mut connection = SqliteConnection::connect_with(&options)
        .await
        .expect("create logs database");
    LOGS_MIGRATOR
        .run_direct(/*target*/ None, &mut connection, /*skip*/ false)
        .await
        .expect("apply logs schema");
    (codex_home, connection)
}

async fn insert_log(connection: &mut SqliteConnection, ts: i64, body: &str) {
    sqlx::query(
        "INSERT INTO logs (ts, ts_nanos, level, target, feedback_log_body, estimated_bytes) VALUES (?, 0, 'INFO', 'test', ?, ?)",
    )
    .bind(ts)
    .bind(body)
    .bind(i64::try_from(body.len()).unwrap_or(i64::MAX))
    .execute(connection)
    .await
    .expect("insert log row");
}

async fn log_rows(connection: &mut SqliteConnection) -> Vec<(i64, String)> {
    sqlx::query("SELECT ts, feedback_log_body FROM logs ORDER BY id")
        .fetch_all(connection)
        .await
        .expect("load log rows")
        .into_iter()
        .map(|row| {
            (
                row.get::<i64, _>("ts"),
                row.get::<String, _>("feedback_log_body"),
            )
        })
        .collect()
}

async fn schema_rows(connection: &mut SqliteConnection) -> Vec<(String, String, Option<String>)> {
    sqlx::query("SELECT type, name, sql FROM sqlite_schema ORDER BY type, name")
        .fetch_all(connection)
        .await
        .expect("load schema rows")
        .into_iter()
        .map(|row| {
            (
                row.get::<String, _>("type"),
                row.get::<String, _>("name"),
                row.get::<Option<String>, _>("sql"),
            )
        })
        .collect()
}

async fn migration_rows(connection: &mut SqliteConnection) -> Vec<(i64, bool, Vec<u8>)> {
    sqlx::query("SELECT version, success, checksum FROM _sqlx_migrations ORDER BY version")
        .fetch_all(connection)
        .await
        .expect("load migration rows")
        .into_iter()
        .map(|row| {
            (
                row.get::<i64, _>("version"),
                row.get::<bool, _>("success"),
                row.get::<Vec<u8>, _>("checksum"),
            )
        })
        .collect()
}

#[tokio::test]
async fn maintenance_is_bounded_resumes_after_busy_and_preserves_fresh_rows() {
    let (codex_home, mut connection) = create_logs_db().await;
    let logs_path = codex_home.join(LOGS_DB_FILENAME);
    let now = Utc::now().timestamp();
    let old_ts = now - 11 * 24 * 60 * 60;
    for index in 0..5 {
        insert_log(&mut connection, old_ts, format!("old-{index}").as_str()).await;
    }
    insert_log(&mut connection, now, "fresh-a").await;
    insert_log(&mut connection, now + 1, "fresh-b").await;
    connection.close().await.expect("close setup connection");

    let first = run_logs_maintenance(
        logs_path.as_path(),
        /*telemetry_override*/ None,
        now,
        test_limits(),
    )
    .await
    .expect("run bounded maintenance");
    assert_eq!(
        first,
        LogMaintenanceOutcome {
            status: MaintenanceStatus::Partial,
            deleted_rows: 3,
            passive_checkpoint: None,
            truncate_checkpoint: None,
        }
    );

    let recent = run_logs_maintenance(
        logs_path.as_path(),
        /*telemetry_override*/ None,
        now + 1,
        test_limits(),
    )
    .await
    .expect("honor partial retry interval");
    assert_eq!(recent.status, MaintenanceStatus::SkippedRecent);

    let writer_options = SqliteConnectOptions::new()
        .filename(&logs_path)
        .create_if_missing(false)
        .busy_timeout(Duration::from_millis(25));
    let mut writer = SqliteConnection::connect_with(&writer_options)
        .await
        .expect("open writer");
    sqlx::query("BEGIN IMMEDIATE")
        .execute(&mut writer)
        .await
        .expect("hold writer slot");
    let busy = run_logs_maintenance(
        logs_path.as_path(),
        /*telemetry_override*/ None,
        now + 5 * 60,
        test_limits(),
    )
    .await
    .expect("stop promptly on foreground writer");
    assert_eq!(busy.status, MaintenanceStatus::Busy);
    sqlx::query("ROLLBACK")
        .execute(&mut writer)
        .await
        .expect("release writer slot");
    writer.close().await.expect("close writer");

    let final_outcome = run_logs_maintenance(
        logs_path.as_path(),
        /*telemetry_override*/ None,
        now + 10 * 60,
        test_limits(),
    )
    .await
    .expect("resume maintenance");
    assert_eq!(final_outcome.status, MaintenanceStatus::Complete);
    assert_eq!(final_outcome.deleted_rows, 2);
    assert!(final_outcome.passive_checkpoint.is_some());
    assert!(final_outcome.truncate_checkpoint.is_some());

    let mut connection = SqliteConnection::connect_with(&writer_options)
        .await
        .expect("open inspection connection");
    assert_eq!(
        log_rows(&mut connection).await,
        vec![
            (now, "fresh-a".to_string()),
            (now + 1, "fresh-b".to_string())
        ]
    );
    connection
        .close()
        .await
        .expect("close inspection connection");
    let _ = tokio::fs::remove_dir_all(codex_home).await;
}

#[tokio::test]
async fn maintenance_lock_is_nonblocking_and_stamp_intervals_are_enforced() {
    let (codex_home, connection) = create_logs_db().await;
    let logs_path = codex_home.join(LOGS_DB_FILENAME);
    connection.close().await.expect("close setup connection");
    let lock_path = maintenance_path(logs_path.as_path());
    let lock_file = open_owner_only_file(lock_path.as_path()).expect("open maintenance lock");
    fs2::FileExt::lock_exclusive(&lock_file).expect("hold maintenance lock");

    let locked = tokio::time::timeout(
        Duration::from_millis(100),
        run_logs_maintenance(
            logs_path.as_path(),
            /*telemetry_override*/ None,
            10_000,
            test_limits(),
        ),
    )
    .await
    .expect("maintenance lock acquisition should not wait")
    .expect("report held maintenance lock");
    assert_eq!(locked.status, MaintenanceStatus::SkippedLocked);
    fs2::FileExt::unlock(&lock_file).expect("release maintenance lock");

    let complete = run_logs_maintenance(
        logs_path.as_path(),
        /*telemetry_override*/ None,
        10_000,
        test_limits(),
    )
    .await
    .expect("run maintenance after lock release");
    assert_eq!(complete.status, MaintenanceStatus::Complete);
    let recent = run_logs_maintenance(
        logs_path.as_path(),
        /*telemetry_override*/ None,
        10_000 + 60 * 60 - 1,
        test_limits(),
    )
    .await
    .expect("honor success interval");
    assert_eq!(recent.status, MaintenanceStatus::SkippedRecent);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = tokio::fs::metadata(lock_path)
            .await
            .expect("load maintenance lock metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600);
    }

    let _ = tokio::fs::remove_dir_all(codex_home).await;
}

#[tokio::test]
async fn active_reader_defers_truncate_then_quiet_run_truncates_without_schema_changes() {
    let (codex_home, mut setup) = create_logs_db().await;
    let logs_path = codex_home.join(LOGS_DB_FILENAME);
    insert_log(&mut setup, Utc::now().timestamp(), "baseline").await;
    sqlx::query("PRAGMA wal_checkpoint(TRUNCATE)")
        .execute(&mut setup)
        .await
        .expect("clear initial WAL");
    let before = (
        schema_rows(&mut setup).await,
        migration_rows(&mut setup).await,
    );
    setup.close().await.expect("close setup connection");

    let options = SqliteConnectOptions::new()
        .filename(&logs_path)
        .create_if_missing(false)
        .synchronous(SqliteSynchronous::Off);
    let mut reader = SqliteConnection::connect_with(&options)
        .await
        .expect("open reader");
    sqlx::query("BEGIN")
        .execute(&mut reader)
        .await
        .expect("begin read transaction");
    sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM logs")
        .fetch_one(&mut reader)
        .await
        .expect("establish reader snapshot");

    let mut writer = SqliteConnection::connect_with(&options)
        .await
        .expect("open writer");
    sqlx::query("PRAGMA wal_autocheckpoint = 0")
        .execute(&mut writer)
        .await
        .expect("disable automatic checkpointing");
    sqlx::query("BEGIN IMMEDIATE")
        .execute(&mut writer)
        .await
        .expect("begin WAL writes");
    for index in 0..100 {
        insert_log(
            &mut writer,
            Utc::now().timestamp(),
            format!("wal-{index}").as_str(),
        )
        .await;
    }
    sqlx::query("COMMIT")
        .execute(&mut writer)
        .await
        .expect("commit WAL writes");

    let now = Utc::now().timestamp();
    let blocked = run_logs_maintenance(
        logs_path.as_path(),
        /*telemetry_override*/ None,
        now,
        test_limits(),
    )
    .await
    .expect("run maintenance with active reader");
    assert_eq!(blocked.status, MaintenanceStatus::Partial);
    let checkpoint = blocked
        .passive_checkpoint
        .expect("record passive checkpoint result");
    assert_eq!(checkpoint.busy, 0);
    assert!(checkpoint.checkpointed_frames < checkpoint.log_frames);
    assert_eq!(blocked.truncate_checkpoint, None);

    sqlx::query("ROLLBACK")
        .execute(&mut reader)
        .await
        .expect("release reader snapshot");
    reader.close().await.expect("close reader");
    let quiet = run_logs_maintenance(
        logs_path.as_path(),
        /*telemetry_override*/ None,
        now + 5 * 60,
        test_limits(),
    )
    .await
    .expect("run quiet maintenance");
    assert_eq!(quiet.status, MaintenanceStatus::Complete);
    assert_eq!(
        quiet.truncate_checkpoint,
        Some(CheckpointResult {
            busy: 0,
            log_frames: 0,
            checkpointed_frames: 0,
        })
    );
    let mut wal_path = logs_path.as_os_str().to_os_string();
    wal_path.push("-wal");
    let wal_size = tokio::fs::metadata(PathBuf::from(wal_path))
        .await
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    assert_eq!(wal_size, 0);

    let after = (
        schema_rows(&mut writer).await,
        migration_rows(&mut writer).await,
    );
    assert_eq!(after, before);
    writer.close().await.expect("close writer");
    let _ = tokio::fs::remove_dir_all(codex_home).await;
}

#[tokio::test]
async fn runtime_startup_does_not_wait_for_maintenance_lock_or_log_writer() {
    let codex_home = unique_temp_dir();
    let initial_runtime = StateRuntime::init(codex_home.clone(), "test-provider".to_string())
        .await
        .expect("initialize fixture runtime");
    initial_runtime.close().await;
    let logs_path = codex_home.join(LOGS_DB_FILENAME);
    let lock_file = open_owner_only_file(maintenance_path(logs_path.as_path()).as_path())
        .expect("open maintenance lock");
    fs2::FileExt::lock_exclusive(&lock_file).expect("hold maintenance lock");
    let writer_options = SqliteConnectOptions::new()
        .filename(&logs_path)
        .create_if_missing(false);
    let mut writer = SqliteConnection::connect_with(&writer_options)
        .await
        .expect("open log writer");
    sqlx::query("BEGIN IMMEDIATE")
        .execute(&mut writer)
        .await
        .expect("hold log writer slot");

    let runtime = tokio::time::timeout(
        Duration::from_secs(2),
        StateRuntime::init(codex_home.clone(), "test-provider".to_string()),
    )
    .await
    .expect("runtime startup should not wait for maintenance")
    .expect("initialize runtime while maintenance is blocked");
    let maintenance_task = runtime
        .log_maintenance_task
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
        .expect("maintenance task should be owned by runtime");
    tokio::time::timeout(Duration::from_secs(1), maintenance_task)
        .await
        .expect("background maintenance should skip held lock promptly")
        .expect("background maintenance task should complete");

    sqlx::query("ROLLBACK")
        .execute(&mut writer)
        .await
        .expect("release log writer slot");
    writer.close().await.expect("close log writer");
    fs2::FileExt::unlock(&lock_file).expect("release maintenance lock");
    runtime.close().await;
    let _ = tokio::fs::remove_dir_all(codex_home).await;
}

#[derive(Default)]
struct TestTelemetry {
    counters: Mutex<Vec<RecordedCounter>>,
    durations: Mutex<Vec<String>>,
}

type RecordedCounter = (String, i64, BTreeMap<String, String>);

impl DbTelemetry for TestTelemetry {
    fn counter(&self, name: &str, inc: i64, tags: &[(&str, &str)]) {
        self.counters.lock().expect("counter lock").push((
            name.to_string(),
            inc,
            tags_to_map(tags),
        ));
    }

    fn record_duration(&self, name: &str, _duration: Duration, _tags: &[(&str, &str)]) {
        self.durations
            .lock()
            .expect("duration lock")
            .push(name.to_string());
    }
}

fn tags_to_map(tags: &[(&str, &str)]) -> BTreeMap<String, String> {
    tags.iter()
        .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
        .collect()
}

#[tokio::test]
async fn maintenance_records_outcome_rows_and_checkpoint_values() {
    let (codex_home, mut connection) = create_logs_db().await;
    let logs_path = codex_home.join(LOGS_DB_FILENAME);
    insert_log(
        &mut connection,
        Utc::now().timestamp() - 11 * 24 * 60 * 60,
        "expired",
    )
    .await;
    connection.close().await.expect("close setup connection");
    let telemetry = TestTelemetry::default();

    let outcome = run_logs_maintenance(
        logs_path.as_path(),
        Some(&telemetry),
        Utc::now().timestamp(),
        test_limits(),
    )
    .await
    .expect("run observable maintenance");
    assert_eq!(outcome.status, MaintenanceStatus::Complete);
    let counter_names = telemetry
        .counters
        .lock()
        .expect("counter lock")
        .iter()
        .map(|(name, _, _)| name.clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        counter_names,
        [
            DB_MAINTENANCE_BUSY_METRIC,
            DB_MAINTENANCE_DELETED_ROWS_METRIC,
            DB_MAINTENANCE_METRIC,
            DB_MAINTENANCE_WAL_FRAMES_METRIC,
        ]
        .into_iter()
        .map(str::to_string)
        .collect()
    );
    assert_eq!(
        *telemetry.durations.lock().expect("duration lock"),
        vec![DB_MAINTENANCE_DURATION_METRIC.to_string()]
    );

    let _ = tokio::fs::remove_dir_all(codex_home).await;
}
