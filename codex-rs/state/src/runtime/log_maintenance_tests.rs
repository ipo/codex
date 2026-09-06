use super::super::StateRuntime;
use super::super::test_support::unique_temp_dir;
use super::*;
use crate::DB_MAINTENANCE_DELETED_ROWS_METRIC;
use crate::DB_MAINTENANCE_METRIC;
use crate::migrations::LOGS_MIGRATOR;
use crate::sqlite::open_owner_only_lock_file;
use codex_utils_absolute_path::test_support::PathExt;
use pretty_assertions::assert_eq;
use sqlx::Connection;
use sqlx::Row;
use sqlx::SqliteConnection;
use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;
use std::process::Child;
use std::process::Command;
use std::process::Output;
use std::process::Stdio;
use std::sync::Mutex;
use std::time::Duration;
use std::time::Instant;

const CHILD_HOME_ENV: &str = "CODEX_LOG_MAINTENANCE_CHILD_HOME";
const CHILD_MODE_ENV: &str = "CODEX_LOG_MAINTENANCE_CHILD_MODE";
const CHILD_NOW_ENV: &str = "CODEX_LOG_MAINTENANCE_CHILD_NOW";
const CHILD_READY_PATH_ENV: &str = "CODEX_LOG_MAINTENANCE_CHILD_READY_PATH";
const CHILD_RESULT_PATH_ENV: &str = "CODEX_LOG_MAINTENANCE_CHILD_RESULT_PATH";
const CHILD_START_PATH_ENV: &str = "CODEX_LOG_MAINTENANCE_CHILD_START_PATH";
const CHILD_STOP_PATH_ENV: &str = "CODEX_LOG_MAINTENANCE_CHILD_STOP_PATH";
const CHILD_TEST: &str = "runtime::log_maintenance::tests::log_maintenance_child";
const COMPETING_WORKERS: usize = 4;
static PROCESS_TEST_SEMAPHORE: tokio::sync::Semaphore =
    tokio::sync::Semaphore::const_new(/*permits*/ 1);

fn test_limits() -> MaintenanceLimits {
    MaintenanceLimits {
        delete_batch_rows: 2,
        max_deleted_rows: 3,
        max_duration: Duration::from_secs(30),
        busy_timeout: Duration::from_millis(25),
        success_interval_seconds: 60 * 60,
        retry_interval_seconds: 5 * 60,
    }
}

async fn create_logs_db() -> (SqliteConfig, SqliteConnection) {
    let codex_home = unique_temp_dir();
    tokio::fs::create_dir_all(&codex_home)
        .await
        .expect("create Codex home");
    let sqlite = SqliteConfig::new_for_testing(codex_home.abs());
    let pool = sqlite
        .open_read_write_pool(sqlite.logs_db_path().as_path())
        .await
        .expect("create logs database");
    LOGS_MIGRATOR.run(&pool).await.expect("apply logs schema");
    pool.close().await;
    let connection = open_connection(&sqlite).await;
    (sqlite, connection)
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

async fn open_connection(sqlite: &SqliteConfig) -> SqliteConnection {
    sqlite
        .open_log_maintenance_connection(Duration::from_millis(25))
        .await
        .expect("open logs connection")
}

fn child_command(sqlite: &SqliteConfig, mode: &str) -> Command {
    let mut command = Command::new(std::env::current_exe().expect("locate current test binary"));
    command
        .arg("--exact")
        .arg(CHILD_TEST)
        .arg("--ignored")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(CHILD_HOME_ENV, sqlite.home())
        .env(CHILD_MODE_ENV, mode)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command
}

fn wait_for_path(path: &Path, description: &str) {
    let started = Instant::now();
    while !path.exists() {
        assert!(
            started.elapsed() < Duration::from_secs(/*secs*/ 10),
            "timed out waiting for {description}"
        );
        std::thread::sleep(Duration::from_millis(/*millis*/ 20));
    }
}

fn wait_for_children(mut children: Vec<Child>, timeout: Duration) -> Vec<Output> {
    let started = Instant::now();
    loop {
        let all_finished = children.iter_mut().all(|child| {
            child
                .try_wait()
                .expect("poll log maintenance child")
                .is_some()
        });
        if all_finished {
            break;
        }
        if started.elapsed() > timeout {
            for child in &mut children {
                let _ = child.kill();
                let _ = child.wait();
            }
            panic!("timed out waiting for log maintenance children");
        }
        std::thread::sleep(Duration::from_millis(/*millis*/ 20));
    }
    children
        .into_iter()
        .map(|child| child.wait_with_output().expect("collect child output"))
        .collect()
}

fn assert_children_succeeded(outputs: &[Output]) {
    for (index, output) in outputs.iter().enumerate() {
        assert!(
            output.status.success(),
            "log maintenance child {index} failed\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}

fn read_child_outcome(path: &Path) -> (MaintenanceStatus, u64) {
    let contents = std::fs::read_to_string(path).expect("read maintenance child result");
    let (status, deleted_rows) = contents
        .split_once(' ')
        .expect("maintenance child result fields");
    let status = match status {
        "complete" => MaintenanceStatus::Complete,
        "partial" => MaintenanceStatus::Partial,
        "busy" => MaintenanceStatus::Busy,
        "skipped_locked" => MaintenanceStatus::SkippedLocked,
        "skipped_recent" => MaintenanceStatus::SkippedRecent,
        status => panic!("unknown maintenance child status {status}"),
    };
    (
        status,
        deleted_rows
            .trim()
            .parse()
            .expect("maintenance child deleted rows"),
    )
}

type RecordedCounter = (String, i64, BTreeMap<String, String>);

#[derive(Default)]
struct TestTelemetry(Mutex<Vec<RecordedCounter>>);

impl DbTelemetry for TestTelemetry {
    fn counter(&self, name: &str, inc: i64, tags: &[(&str, &str)]) {
        self.0.lock().expect("telemetry lock").push((
            name.to_string(),
            inc,
            tags.iter()
                .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
                .collect(),
        ));
    }

    fn histogram(&self, _name: &str, _value: i64, _tags: &[(&str, &str)]) {}

    fn record_duration(&self, _name: &str, _duration: Duration, _tags: &[(&str, &str)]) {}
}

#[tokio::test]
async fn maintenance_is_bounded_resumes_after_busy_and_preserves_fresh_rows() {
    let (sqlite, mut connection) = create_logs_db().await;
    let now = chrono::Utc::now().timestamp();
    let old_ts = now - 11 * 24 * 60 * 60;
    for index in 0..5 {
        insert_log(&mut connection, old_ts, format!("old-{index}").as_str()).await;
    }
    insert_log(&mut connection, now, "fresh-a").await;
    insert_log(&mut connection, now + 1, "fresh-b").await;
    connection.close().await.expect("close setup connection");
    let telemetry = TestTelemetry::default();

    let first = run_logs_maintenance(&sqlite, Some(&telemetry), now, test_limits())
        .await
        .expect("run bounded maintenance");
    assert_eq!(first, outcome(MaintenanceStatus::Partial, 3));
    assert_eq!(
        run_logs_maintenance(&sqlite, None, now + 1, test_limits())
            .await
            .expect("honor retry interval")
            .status,
        MaintenanceStatus::SkippedRecent
    );

    let mut writer = open_connection(&sqlite).await;
    sqlx::query("BEGIN IMMEDIATE")
        .execute(&mut writer)
        .await
        .expect("hold writer slot");
    let busy_started = tokio::time::Instant::now();
    let busy = run_logs_maintenance(&sqlite, None, now + 5 * 60, test_limits())
        .await
        .expect("stop promptly on foreground writer");
    assert_eq!(busy, outcome(MaintenanceStatus::Busy, 0));
    assert!(busy_started.elapsed() < Duration::from_secs(2));
    sqlx::query("ROLLBACK")
        .execute(&mut writer)
        .await
        .expect("release writer slot");

    let final_outcome =
        run_logs_maintenance(&sqlite, Some(&telemetry), now + 10 * 60, test_limits())
            .await
            .expect("resume maintenance");
    assert_eq!(final_outcome.status, MaintenanceStatus::Complete);
    assert_eq!(final_outcome.deleted_rows, 2);
    assert!(final_outcome.passive_checkpoint.is_some());
    assert!(final_outcome.truncate_checkpoint.is_some());
    {
        let counters = telemetry.0.lock().expect("telemetry lock");
        assert!(counters.iter().any(|(name, value, tags)| {
            name == DB_MAINTENANCE_DELETED_ROWS_METRIC
                && *value == 3
                && tags.get("status") == Some(&"partial".to_string())
        }));
        assert_eq!(
            counters
                .iter()
                .filter(|(name, _, _)| name == DB_MAINTENANCE_METRIC)
                .count(),
            2
        );
    }

    assert_eq!(
        log_rows(&mut writer).await,
        vec![
            (now, "fresh-a".to_string()),
            (now + 1, "fresh-b".to_string())
        ]
    );
    writer.close().await.expect("close writer");
    let _ = tokio::fs::remove_dir_all(sqlite.home()).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn competing_processes_run_maintenance_once() {
    let _permit = PROCESS_TEST_SEMAPHORE
        .acquire()
        .await
        .expect("process test semaphore should remain open");
    let (sqlite, mut connection) = create_logs_db().await;
    let now = chrono::Utc::now().timestamp();
    for index in 0..12 {
        insert_log(
            &mut connection,
            now - 11 * 24 * 60 * 60,
            format!("old-{index}").as_str(),
        )
        .await;
    }
    insert_log(&mut connection, now, "fresh").await;
    connection.close().await.expect("close setup connection");

    let start_path = sqlite.home().join("maintenance-start");
    let ready_paths = (0..COMPETING_WORKERS)
        .map(|index| sqlite.home().join(format!("maintenance-ready-{index}")))
        .collect::<Vec<_>>();
    let result_paths = (0..COMPETING_WORKERS)
        .map(|index| sqlite.home().join(format!("maintenance-result-{index}")))
        .collect::<Vec<_>>();
    let mut children = Vec::new();
    for (ready_path, result_path) in ready_paths.iter().zip(&result_paths) {
        let mut command = child_command(&sqlite, "maintenance");
        command
            .env(CHILD_NOW_ENV, now.to_string())
            .env(CHILD_READY_PATH_ENV, ready_path)
            .env(CHILD_RESULT_PATH_ENV, result_path)
            .env(CHILD_START_PATH_ENV, &start_path);
        children.push(command.spawn().expect("spawn competing maintenance child"));
    }
    for ready_path in &ready_paths {
        wait_for_path(ready_path, "competing maintenance child");
    }
    std::fs::write(&start_path, []).expect("release competing maintenance children");
    let outputs = wait_for_children(children, Duration::from_secs(/*secs*/ 20));
    assert_children_succeeded(&outputs);
    let outcomes = result_paths
        .iter()
        .map(|path| read_child_outcome(path))
        .collect::<Vec<_>>();
    assert_eq!(
        outcomes
            .iter()
            .filter(|(status, deleted_rows)| {
                *status == MaintenanceStatus::Complete && *deleted_rows == 12
            })
            .count(),
        1
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|(status, deleted_rows)| {
                *deleted_rows == 0
                    && matches!(
                        *status,
                        MaintenanceStatus::SkippedLocked | MaintenanceStatus::SkippedRecent
                    )
            })
            .count(),
        COMPETING_WORKERS - 1
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|(status, deleted_rows)| {
                *deleted_rows > 0
                    && !matches!(
                        *status,
                        MaintenanceStatus::SkippedLocked | MaintenanceStatus::SkippedRecent
                    )
            })
            .count(),
        1
    );
    assert_eq!(
        outcomes
            .iter()
            .map(|(_, deleted_rows)| deleted_rows)
            .sum::<u64>(),
        12
    );

    let mut connection = open_connection(&sqlite).await;
    assert_eq!(
        log_rows(&mut connection).await,
        vec![(now, "fresh".to_string())]
    );
    connection
        .close()
        .await
        .expect("close inspection connection");
    let _ = tokio::fs::remove_dir_all(sqlite.home()).await;
}

#[tokio::test]
async fn maintenance_lock_is_nonblocking_and_stamp_intervals_are_enforced() {
    let (sqlite, connection) = create_logs_db().await;
    connection.close().await.expect("close setup connection");
    let lock_path = maintenance_path(sqlite.logs_db_path().as_path());
    let lock_file = open_owner_only_lock_file(lock_path.as_path()).expect("open maintenance lock");
    lock_file.lock().expect("hold maintenance lock");

    let locked = tokio::time::timeout(
        Duration::from_millis(/*millis*/ 100),
        run_logs_maintenance(&sqlite, None, /*now*/ 10_000, test_limits()),
    )
    .await
    .expect("maintenance lock acquisition should not wait")
    .expect("report held maintenance lock");
    assert_eq!(locked.status, MaintenanceStatus::SkippedLocked);
    lock_file.unlock().expect("release maintenance lock");

    let complete = run_logs_maintenance(&sqlite, None, /*now*/ 10_000, test_limits())
        .await
        .expect("run maintenance after lock release");
    assert_eq!(complete.status, MaintenanceStatus::Complete);
    let recent = run_logs_maintenance(
        &sqlite,
        None,
        /*now*/ 10_000 + 60 * 60 - 1,
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

    let _ = tokio::fs::remove_dir_all(sqlite.home()).await;
}

async fn schema_state(
    connection: &mut SqliteConnection,
) -> (
    Vec<(String, String, Option<String>)>,
    Vec<(i64, bool, Vec<u8>)>,
) {
    let schema = sqlx::query("SELECT type, name, sql FROM sqlite_schema ORDER BY type, name")
        .fetch_all(&mut *connection)
        .await
        .expect("load schema")
        .into_iter()
        .map(|row| (row.get("type"), row.get("name"), row.get("sql")))
        .collect();
    let migrations =
        sqlx::query("SELECT version, success, checksum FROM _sqlx_migrations ORDER BY version")
            .fetch_all(connection)
            .await
            .expect("load migrations")
            .into_iter()
            .map(|row| (row.get("version"), row.get("success"), row.get("checksum")))
            .collect();
    (schema, migrations)
}

#[tokio::test]
async fn active_reader_defers_truncate_then_quiet_run_truncates_without_schema_changes() {
    let (sqlite, mut setup) = create_logs_db().await;
    let logs_path = sqlite.logs_db_path();
    insert_log(&mut setup, chrono::Utc::now().timestamp(), "baseline").await;
    sqlx::query("PRAGMA wal_checkpoint(TRUNCATE)")
        .execute(&mut setup)
        .await
        .expect("clear initial WAL");
    let before = schema_state(&mut setup).await;
    setup.close().await.expect("close setup connection");

    let mut reader = open_connection(&sqlite).await;
    sqlx::query("BEGIN")
        .execute(&mut reader)
        .await
        .expect("begin read transaction");
    sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM logs")
        .fetch_one(&mut reader)
        .await
        .expect("establish reader snapshot");

    let mut writer = open_connection(&sqlite).await;
    sqlx::query("PRAGMA wal_autocheckpoint = 0")
        .execute(&mut writer)
        .await
        .expect("disable automatic checkpointing");
    for index in 0..100 {
        insert_log(
            &mut writer,
            chrono::Utc::now().timestamp(),
            format!("wal-{index}").as_str(),
        )
        .await;
    }

    let now = chrono::Utc::now().timestamp();
    let blocked = run_logs_maintenance(&sqlite, None, now, test_limits())
        .await
        .expect("run maintenance with active reader");
    assert_eq!(blocked.status, MaintenanceStatus::Partial);
    let checkpoint = blocked
        .passive_checkpoint
        .expect("passive checkpoint result");
    assert_eq!(checkpoint.busy, 0);
    assert!(checkpoint.checkpointed_frames < checkpoint.log_frames);
    assert_eq!(blocked.truncate_checkpoint, None);

    sqlx::query("ROLLBACK")
        .execute(&mut reader)
        .await
        .expect("release reader snapshot");
    reader.close().await.expect("close reader");
    let quiet = run_logs_maintenance(&sqlite, None, now + 5 * 60, test_limits())
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
    assert_eq!(schema_state(&mut writer).await, before);

    writer.close().await.expect("close writer");
    let _ = tokio::fs::remove_dir_all(sqlite.home()).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runtime_startup_completes_while_external_process_holds_log_writer() {
    let _permit = PROCESS_TEST_SEMAPHORE
        .acquire()
        .await
        .expect("process test semaphore should remain open");
    let codex_home = unique_temp_dir();
    let sqlite = SqliteConfig::new_for_testing(codex_home.abs());
    tokio::fs::create_dir_all(&codex_home)
        .await
        .expect("create Codex home");
    let initial_runtime = StateRuntime::init(sqlite.clone(), "test-provider".to_string())
        .await
        .expect("initialize fixture runtime");
    initial_runtime.close().await;

    let writer_ready_path = sqlite.home().join("writer-ready");
    let writer_stop_path = sqlite.home().join("writer-stop");
    let mut writer_command = child_command(&sqlite, "hold_writer");
    writer_command
        .env(CHILD_READY_PATH_ENV, &writer_ready_path)
        .env(CHILD_STOP_PATH_ENV, &writer_stop_path);
    let writer = writer_command.spawn().expect("spawn external log writer");
    wait_for_path(&writer_ready_path, "external log writer");

    let startup = child_command(&sqlite, "startup")
        .spawn()
        .expect("spawn runtime startup child");
    let startup_output = wait_for_children(vec![startup], Duration::from_secs(/*secs*/ 8));
    std::fs::write(&writer_stop_path, []).expect("release external log writer");
    let writer_output = wait_for_children(vec![writer], Duration::from_secs(/*secs*/ 5));
    assert_children_succeeded(&startup_output);
    assert_children_succeeded(&writer_output);
    let _ = tokio::fs::remove_dir_all(codex_home).await;
}

#[tokio::test]
async fn background_maintenance_errors_do_not_fail_startup() {
    let codex_home = unique_temp_dir();
    let sqlite = SqliteConfig::new_for_testing(codex_home.abs());
    tokio::fs::create_dir_all(&codex_home)
        .await
        .expect("create Codex home");
    let maintenance_path = maintenance_path(sqlite.logs_db_path().as_path());
    tokio::fs::create_dir(&maintenance_path)
        .await
        .expect("make maintenance sidecar invalid");

    let runtime = StateRuntime::init(sqlite, "test-provider".to_string())
        .await
        .expect("maintenance error must not fail startup");
    let task = runtime
        .log_maintenance_task
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
        .expect("runtime owns maintenance task");
    task.await.expect("maintenance error is isolated in task");
    runtime.close().await;
    let _ = tokio::fs::remove_dir_all(codex_home).await;
}

#[tokio::test]
#[ignore = "subprocess entry point for log maintenance tests"]
async fn log_maintenance_child() {
    let Some(mode) = std::env::var_os(CHILD_MODE_ENV) else {
        return;
    };
    let codex_home = std::env::var_os(CHILD_HOME_ENV)
        .map(PathBuf::from)
        .expect("child Codex home");
    let sqlite = SqliteConfig::new_for_testing(codex_home.abs());
    match mode.to_str().expect("UTF-8 child mode") {
        "hold_writer" => {
            let ready_path = std::env::var_os(CHILD_READY_PATH_ENV)
                .map(PathBuf::from)
                .expect("writer ready path");
            let stop_path = std::env::var_os(CHILD_STOP_PATH_ENV)
                .map(PathBuf::from)
                .expect("writer stop path");
            let mut writer = open_connection(&sqlite).await;
            sqlx::query("BEGIN IMMEDIATE")
                .execute(&mut writer)
                .await
                .expect("hold external log writer slot");
            std::fs::write(ready_path, []).expect("signal external writer ready");
            let started = Instant::now();
            while !stop_path.exists() {
                assert!(
                    started.elapsed() < Duration::from_secs(/*secs*/ 15),
                    "timed out waiting for external writer stop signal"
                );
                tokio::time::sleep(Duration::from_millis(/*millis*/ 10)).await;
            }
            sqlx::query("ROLLBACK")
                .execute(&mut writer)
                .await
                .expect("release external log writer slot");
            writer.close().await.expect("close external log writer");
        }
        "startup" => {
            let runtime = tokio::time::timeout(
                Duration::from_secs(/*secs*/ 7),
                StateRuntime::init(sqlite, "test-provider".to_string()),
            )
            .await
            .expect("runtime startup waited for external log writer")
            .expect("initialize runtime with external log writer");
            runtime.close().await;
        }
        "maintenance" => {
            let ready_path = std::env::var_os(CHILD_READY_PATH_ENV)
                .map(PathBuf::from)
                .expect("maintenance ready path");
            let start_path = std::env::var_os(CHILD_START_PATH_ENV)
                .map(PathBuf::from)
                .expect("maintenance start path");
            let result_path = std::env::var_os(CHILD_RESULT_PATH_ENV)
                .map(PathBuf::from)
                .expect("maintenance result path");
            let now = std::env::var(CHILD_NOW_ENV)
                .expect("maintenance timestamp")
                .parse::<i64>()
                .expect("integer maintenance timestamp");
            std::fs::write(ready_path, []).expect("signal maintenance child ready");
            let started = Instant::now();
            while !start_path.exists() {
                assert!(
                    started.elapsed() < Duration::from_secs(/*secs*/ 10),
                    "timed out waiting for maintenance start signal"
                );
                tokio::time::sleep(Duration::from_millis(/*millis*/ 10)).await;
            }
            let mut limits = test_limits();
            limits.delete_batch_rows = 1;
            limits.max_deleted_rows = 100;
            let result = run_logs_maintenance(&sqlite, None, now, limits)
                .await
                .expect("run child log maintenance");
            std::fs::write(
                result_path,
                format!("{} {}", result.status.as_str(), result.deleted_rows),
            )
            .expect("write maintenance child result");
        }
        mode => panic!("unknown log maintenance child mode {mode}"),
    }
}
