use super::STATE_DB;
use super::acquire_init_lock;
use crate::SqliteConfig;
use crate::migrations::STATE_MIGRATOR;
use crate::migrations::runtime_state_migrator;
use crate::runtime::test_support::unique_temp_dir;
use codex_utils_absolute_path::test_support::PathExt;
use pretty_assertions::assert_eq;
use sqlx::ConnectOptions;
use sqlx::Connection;
use sqlx::Row;
use sqlx::SqliteConnection;
use sqlx::migrate::Migration;
use sqlx::migrate::MigrationType;
use sqlx::migrate::Migrator;
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::sqlite::SqliteJournalMode;
use std::borrow::Cow;
use std::path::Path;
use std::process::Child;
use std::process::Command;
use std::process::Output;
use std::process::Stdio;
use std::time::Duration;
use std::time::Instant;

const CHILD_MODE_ENV: &str = "CODEX_SQLITE_INIT_CHILD_MODE";
const CHILD_PATH_ENV: &str = "CODEX_SQLITE_INIT_CHILD_PATH";
const CHILD_READY_PATH_ENV: &str = "CODEX_SQLITE_INIT_CHILD_READY_PATH";
const CHILD_START_PATH_ENV: &str = "CODEX_SQLITE_INIT_CHILD_START_PATH";
const CHILD_TEST: &str = "sqlite::tests::sqlite_initialization_child";
const CONCURRENT_INITIALIZERS: usize = 8;
const CONCURRENT_WAL_REPAIRERS: usize = 6;
static PROCESS_TEST_SEMAPHORE: tokio::sync::Semaphore =
    tokio::sync::Semaphore::const_new(/*permits*/ 1);

async fn open_raw_connection(path: &Path, create_if_missing: bool) -> SqliteConnection {
    SqliteConnection::connect_with(
        &SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(create_if_missing)
            .log_statements(log::LevelFilter::Off),
    )
    .await
    .expect("open raw SQLite connection")
}

fn spawn_child(path: &Path, mode: &str, start_path: Option<&Path>) -> Child {
    let mut command = Command::new(std::env::current_exe().expect("locate current test binary"));
    command
        .arg("--exact")
        .arg(CHILD_TEST)
        .arg("--ignored")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(CHILD_MODE_ENV, mode)
        .env(CHILD_PATH_ENV, path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(start_path) = start_path {
        command.env(CHILD_START_PATH_ENV, start_path);
    }
    command.spawn().expect("spawn SQLite initializer child")
}

fn wait_for_children(mut children: Vec<Child>) -> Vec<Output> {
    let started = Instant::now();
    loop {
        let all_finished = children.iter_mut().all(|child| {
            child
                .try_wait()
                .expect("poll SQLite initializer child")
                .is_some()
        });
        if all_finished {
            break;
        }
        if started.elapsed() > Duration::from_secs(/*secs*/ 45) {
            for child in &mut children {
                let _ = child.kill();
                let _ = child.wait();
            }
            panic!("timed out waiting for SQLite initializer children");
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
            "SQLite initializer child {index} failed\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}

async fn inspect_database(path: &Path) -> (String, String, Vec<(i64, bool, Vec<u8>)>) {
    let mut connection = open_raw_connection(path, /*create_if_missing*/ false).await;
    let journal_mode = sqlx::query_scalar::<_, String>("PRAGMA journal_mode")
        .fetch_one(&mut connection)
        .await
        .expect("read journal mode");
    let integrity = sqlx::query_scalar::<_, String>("PRAGMA integrity_check")
        .fetch_one(&mut connection)
        .await
        .expect("run integrity check");
    let migrations =
        sqlx::query("SELECT version, success, checksum FROM _sqlx_migrations ORDER BY version")
            .fetch_all(&mut connection)
            .await
            .expect("read migration records")
            .into_iter()
            .map(|row| {
                (
                    row.get::<i64, _>("version"),
                    row.get::<bool, _>("success"),
                    row.get::<Vec<u8>, _>("checksum"),
                )
            })
            .collect();
    connection
        .close()
        .await
        .expect("close inspection connection");
    (journal_mode, integrity, migrations)
}

fn expected_migrations(migrator: &Migrator) -> Vec<(i64, bool, Vec<u8>)> {
    migrator
        .migrations
        .iter()
        .filter(|migration| migration.migration_type.is_up_migration())
        .map(|migration| (migration.version, true, migration.checksum.to_vec()))
        .collect()
}

fn wal_repair_migrator() -> Migrator {
    let mut migrator = runtime_state_migrator();
    migrator.migrations = Cow::Owned(vec![Migration::new(
        /*version*/ 1,
        Cow::Borrowed("create WAL repair fixture"),
        MigrationType::Simple,
        sqlx::SqlStr::from_static(
            "CREATE TABLE wal_repair_fixture (id INTEGER PRIMARY KEY, value TEXT NOT NULL)",
        ),
        /*no_tx*/ false,
    )]);
    migrator
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_processes_initialize_one_clean_database_without_corruption() {
    let _permit = PROCESS_TEST_SEMAPHORE
        .acquire()
        .await
        .expect("process test semaphore should remain open");
    let codex_home = unique_temp_dir();
    std::fs::create_dir_all(&codex_home).expect("create temporary Codex home");
    let path = codex_home.join(super::STATE_DB_FILENAME);
    let start_path = codex_home.join("start");
    let children = (0..CONCURRENT_INITIALIZERS)
        .map(|_| spawn_child(path.as_path(), "initialize", Some(start_path.as_path())))
        .collect();

    std::fs::write(&start_path, []).expect("release initializer children");
    let outputs = wait_for_children(children);
    assert_children_succeeded(&outputs);
    assert_eq!(
        inspect_database(path.as_path()).await,
        (
            "wal".to_string(),
            "ok".to_string(),
            expected_migrations(&STATE_MIGRATOR)
        )
    );

    std::fs::remove_dir_all(codex_home).expect("remove temporary Codex home");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_processes_repair_wal_before_reusing_the_database() {
    let _permit = PROCESS_TEST_SEMAPHORE
        .acquire()
        .await
        .expect("process test semaphore should remain open");
    let codex_home = unique_temp_dir();
    std::fs::create_dir_all(&codex_home).expect("create temporary Codex home");
    let path = codex_home.join(super::STATE_DB_FILENAME);
    let start_path = codex_home.join("start");
    let mut connection = SqliteConnection::connect_with(
        &SqliteConnectOptions::new()
            .filename(&path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .log_statements(log::LevelFilter::Off),
    )
    .await
    .expect("create rollback-journal database");
    let migrator = wal_repair_migrator();
    migrator
        .run_direct(/*target*/ None, &mut connection, /*skip*/ false)
        .await
        .expect("create current schema fixture");
    assert_eq!(
        sqlx::query_scalar::<_, String>("PRAGMA journal_mode = DELETE")
            .fetch_one(&mut connection)
            .await
            .expect("convert fixture to rollback journal mode"),
        "delete"
    );
    connection.close().await.expect("close fixture connection");

    let children = (0..CONCURRENT_WAL_REPAIRERS)
        .map(|_| {
            spawn_child(
                path.as_path(),
                "initialize_wal_fixture",
                Some(start_path.as_path()),
            )
        })
        .collect();
    std::fs::write(&start_path, []).expect("release initializer children");
    let outputs = wait_for_children(children);
    assert_children_succeeded(&outputs);
    assert_eq!(
        inspect_database(path.as_path()).await,
        (
            "wal".to_string(),
            "ok".to_string(),
            expected_migrations(&migrator)
        )
    );

    std::fs::remove_dir_all(codex_home).expect("remove temporary Codex home");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn migration_failure_releases_the_process_lock_for_retry() {
    let _permit = PROCESS_TEST_SEMAPHORE
        .acquire()
        .await
        .expect("process test semaphore should remain open");
    let codex_home = unique_temp_dir();
    std::fs::create_dir_all(&codex_home).expect("create temporary Codex home");
    let path = codex_home.join(super::STATE_DB_FILENAME);
    let failed = wait_for_children(vec![spawn_child(path.as_path(), "fail_migration", None)]);
    assert_children_succeeded(&failed);

    let start_path = codex_home.join("start");
    let children = (0..4)
        .map(|_| spawn_child(path.as_path(), "initialize", Some(start_path.as_path())))
        .collect();
    std::fs::write(&start_path, []).expect("release retry initializer children");
    let retried = wait_for_children(children);
    assert_children_succeeded(&retried);
    assert_eq!(
        inspect_database(path.as_path()).await,
        (
            "wal".to_string(),
            "ok".to_string(),
            expected_migrations(&STATE_MIGRATOR)
        )
    );

    std::fs::remove_dir_all(codex_home).expect("remove temporary Codex home");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn terminating_the_lock_holder_unblocks_another_process() {
    let _permit = PROCESS_TEST_SEMAPHORE
        .acquire()
        .await
        .expect("process test semaphore should remain open");
    let codex_home = unique_temp_dir();
    std::fs::create_dir_all(&codex_home).expect("create temporary Codex home");
    let path = codex_home.join(super::STATE_DB_FILENAME);
    let ready_path = codex_home.join("lock-ready");
    let mut holder = Command::new(std::env::current_exe().expect("locate current test binary"))
        .arg("--exact")
        .arg(CHILD_TEST)
        .arg("--ignored")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(CHILD_MODE_ENV, "hold_lock")
        .env(CHILD_PATH_ENV, &path)
        .env(CHILD_READY_PATH_ENV, &ready_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn initialization lock holder");
    let started = Instant::now();
    while !ready_path.exists() {
        assert!(
            started.elapsed() < Duration::from_secs(/*secs*/ 10),
            "timed out waiting for initialization lock holder"
        );
        std::thread::sleep(Duration::from_millis(/*millis*/ 20));
    }

    let mut waiter = spawn_child(path.as_path(), "initialize", None);
    std::thread::sleep(Duration::from_millis(/*millis*/ 200));
    assert_eq!(waiter.try_wait().expect("poll waiting initializer"), None);
    holder.kill().expect("terminate initialization lock holder");
    let holder_status = holder.wait().expect("wait for terminated lock holder");
    assert!(!holder_status.success());
    let outputs = wait_for_children(vec![waiter]);
    assert_children_succeeded(&outputs);
    assert_eq!(
        inspect_database(path.as_path()).await,
        (
            "wal".to_string(),
            "ok".to_string(),
            expected_migrations(&STATE_MIGRATOR)
        )
    );

    std::fs::remove_dir_all(codex_home).expect("remove temporary Codex home");
}

#[tokio::test]
#[ignore = "subprocess entry point for SQLite initialization tests"]
async fn sqlite_initialization_child() {
    let Some(mode) = std::env::var_os(CHILD_MODE_ENV) else {
        return;
    };
    let path = std::env::var_os(CHILD_PATH_ENV)
        .map(std::path::PathBuf::from)
        .expect("child database path");
    if let Some(start_path) = std::env::var_os(CHILD_START_PATH_ENV) {
        let start_path = std::path::PathBuf::from(start_path);
        let started = Instant::now();
        while !start_path.exists() {
            assert!(
                started.elapsed() < Duration::from_secs(/*secs*/ 10),
                "timed out waiting for parent start signal"
            );
            tokio::time::sleep(Duration::from_millis(/*millis*/ 10)).await;
        }
    }

    match mode.to_str().expect("UTF-8 child mode") {
        "initialize" | "initialize_wal_fixture" => {
            let sqlite =
                SqliteConfig::new_for_testing(path.parent().expect("database parent").abs());
            let migrator = if mode == "initialize" {
                runtime_state_migrator()
            } else {
                wal_repair_migrator()
            };
            let pool = tokio::time::timeout(
                Duration::from_secs(/*secs*/ 40),
                sqlite.open_state_db(&migrator, /*telemetry_override*/ None),
            )
            .await
            .expect("SQLite initialization deadlocked")
            .expect("initialize SQLite database");
            pool.close().await;
            println!("initialized {}", path.display());
        }
        "fail_migration" => {
            let mut migrator = runtime_state_migrator();
            migrator.migrations = Cow::Owned(vec![Migration::new(
                /*version*/ 1,
                Cow::Borrowed("intentional failure"),
                MigrationType::Simple,
                sqlx::SqlStr::from_static("THIS IS NOT VALID SQL"),
                /*no_tx*/ false,
            )]);
            let sqlite =
                SqliteConfig::new_for_testing(path.parent().expect("database parent").abs());
            let error = sqlite
                .open_runtime_db(STATE_DB, &migrator, /*telemetry_override*/ None)
                .await
                .expect_err("invalid migration should fail");
            println!("observed expected migration failure: {error:#}");
        }
        "hold_lock" => {
            let _lock = acquire_init_lock(path.as_path())
                .await
                .expect("acquire initialization lock");
            let ready_path = std::env::var_os(CHILD_READY_PATH_ENV)
                .map(std::path::PathBuf::from)
                .expect("child ready path");
            std::fs::write(ready_path, []).expect("signal acquired initialization lock");
            std::future::pending::<()>().await;
        }
        mode => panic!("unknown SQLite initialization child mode {mode}"),
    }
}
