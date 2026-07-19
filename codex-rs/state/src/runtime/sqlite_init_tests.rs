use super::super::STATE_DB;
use super::super::test_support::unique_temp_dir;
use super::init_lock_path;
use super::open_owner_only_lock_file;
use super::open_sqlite;
use crate::migrations::STATE_MIGRATOR;
use crate::migrations::runtime_state_migrator;
use pretty_assertions::assert_eq;
use sqlx::ConnectOptions;
use sqlx::Connection;
use sqlx::Row;
use sqlx::SqliteConnection;
use sqlx::SqlitePool;
use sqlx::migrate::Migrator;
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::sqlite::SqliteSynchronous;
use std::borrow::Cow;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use tokio::sync::Barrier;

const CONCURRENT_OPEN_COUNT: usize = 16;

async fn create_home() -> std::path::PathBuf {
    let codex_home = unique_temp_dir();
    tokio::fs::create_dir_all(&codex_home)
        .await
        .expect("create codex home");
    codex_home
}

async fn open_raw_connection(path: &Path, create_if_missing: bool) -> SqliteConnection {
    SqliteConnection::connect_with(
        &SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(create_if_missing)
            .synchronous(SqliteSynchronous::Off)
            .log_statements(log::LevelFilter::Off),
    )
    .await
    .expect("open raw SQLite connection")
}

async fn schema_rows(connection: &mut SqliteConnection) -> Vec<(String, String, Option<String>)> {
    sqlx::query("SELECT type, name, sql FROM sqlite_schema ORDER BY type, name")
        .fetch_all(connection)
        .await
        .expect("load SQLite schema")
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
        .expect("load migration history")
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

fn migrator_through(version: i64) -> Migrator {
    Migrator {
        migrations: Cow::Owned(
            STATE_MIGRATOR
                .migrations
                .iter()
                .filter(|migration| migration.version <= version)
                .cloned()
                .collect(),
        ),
        ignore_missing: STATE_MIGRATOR.ignore_missing,
        locking: STATE_MIGRATOR.locking,
        table_name: STATE_MIGRATOR.table_name.clone(),
        create_schemas: STATE_MIGRATOR.create_schemas.clone(),
        no_tx: STATE_MIGRATOR.no_tx,
    }
}

async fn open_state(path: &Path) -> SqlitePool {
    open_sqlite(
        path,
        &runtime_state_migrator(),
        STATE_DB,
        /*telemetry_override*/ None,
    )
    .await
    .expect("open state SQLite database")
}

async fn open_state_concurrently(path: &Path) {
    let barrier = Arc::new(Barrier::new(CONCURRENT_OPEN_COUNT + 1));
    let mut tasks = Vec::with_capacity(CONCURRENT_OPEN_COUNT);
    for _ in 0..CONCURRENT_OPEN_COUNT {
        let barrier = Arc::clone(&barrier);
        let path = path.to_path_buf();
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            let pool = open_state(path.as_path()).await;
            drop(pool);
        }));
    }
    barrier.wait().await;
    for task in tasks {
        task.await.expect("concurrent open task should complete");
    }
}

#[tokio::test]
async fn new_database_uses_wal_and_incremental_auto_vacuum() {
    let codex_home = create_home().await;
    let path = codex_home.join(crate::STATE_DB_FILENAME);

    let pool = open_state(path.as_path()).await;

    let settings = (
        sqlx::query_scalar::<_, String>("PRAGMA journal_mode")
            .fetch_one(&pool)
            .await
            .expect("load journal mode"),
        sqlx::query_scalar::<_, i64>("PRAGMA auto_vacuum")
            .fetch_one(&pool)
            .await
            .expect("load auto-vacuum mode"),
    );
    let mut connection = pool.acquire().await.expect("acquire SQLite connection");
    let applied = migration_rows(&mut connection).await;
    let expected = STATE_MIGRATOR
        .migrations
        .iter()
        .filter(|migration| migration.migration_type.is_up_migration())
        .map(|migration| (migration.version, true, migration.checksum.to_vec()))
        .collect::<Vec<_>>();

    assert_eq!(settings, ("wal".to_string(), 2));
    assert_eq!(applied, expected);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = tokio::fs::metadata(init_lock_path(path.as_path()))
            .await
            .expect("load init lock metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600);
    }

    drop(connection);
    drop(pool);
    let _ = tokio::fs::remove_dir_all(codex_home).await;
}

#[tokio::test]
async fn established_database_keeps_schema_and_auto_vacuum_setting() {
    let codex_home = create_home().await;
    let path = codex_home.join(crate::STATE_DB_FILENAME);
    let mut original_connection = open_raw_connection(path.as_path(), true).await;
    STATE_MIGRATOR
        .run_direct(
            /*target*/ None,
            &mut original_connection,
            /*skip*/ false,
        )
        .await
        .expect("apply current schema");
    let original = (
        schema_rows(&mut original_connection).await,
        migration_rows(&mut original_connection).await,
        sqlx::query_scalar::<_, i64>("PRAGMA auto_vacuum")
            .fetch_one(&mut original_connection)
            .await
            .expect("load original auto-vacuum mode"),
    );
    original_connection
        .close()
        .await
        .expect("close original connection");

    let pool = open_state(path.as_path()).await;
    let mut connection = pool.acquire().await.expect("acquire SQLite connection");
    let reopened = (
        schema_rows(&mut connection).await,
        migration_rows(&mut connection).await,
        sqlx::query_scalar::<_, i64>("PRAGMA auto_vacuum")
            .fetch_one(&mut *connection)
            .await
            .expect("load reopened auto-vacuum mode"),
    );
    let journal_mode = sqlx::query_scalar::<_, String>("PRAGMA journal_mode")
        .fetch_one(&mut *connection)
        .await
        .expect("load journal mode");

    assert_eq!(reopened, original);
    assert_eq!(journal_mode, "wal");

    drop(connection);
    drop(pool);
    let _ = tokio::fs::remove_dir_all(codex_home).await;
}

#[tokio::test]
async fn current_database_does_not_wait_for_init_lock() {
    let codex_home = create_home().await;
    let path = codex_home.join(crate::STATE_DB_FILENAME);
    open_state(path.as_path()).await.close().await;
    let lock_file = open_owner_only_lock_file(init_lock_path(path.as_path()).as_path())
        .expect("open init lock file");
    fs2::FileExt::lock_exclusive(&lock_file).expect("hold init lock");

    let pool = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        open_state(path.as_path()),
    )
    .await
    .expect("current database should bypass held init lock");

    drop(pool);
    fs2::FileExt::unlock(&lock_file).expect("release init lock");
    let _ = tokio::fs::remove_dir_all(codex_home).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_cold_initialization_applies_schema_once() {
    let codex_home = create_home().await;
    let path = codex_home.join(crate::STATE_DB_FILENAME);

    open_state_concurrently(path.as_path()).await;

    let mut connection = open_raw_connection(path.as_path(), false).await;
    let applied = migration_rows(&mut connection).await;
    let expected = STATE_MIGRATOR
        .migrations
        .iter()
        .filter(|migration| migration.migration_type.is_up_migration())
        .map(|migration| (migration.version, true, migration.checksum.to_vec()))
        .collect::<Vec<_>>();
    assert_eq!(applied, expected);

    connection
        .close()
        .await
        .expect("close inspection connection");
    let _ = tokio::fs::remove_dir_all(codex_home).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_upgrade_preserves_existing_data() {
    let codex_home = create_home().await;
    let path = codex_home.join(crate::STATE_DB_FILENAME);
    let mut connection = open_raw_connection(path.as_path(), true).await;
    migrator_through(/*version*/ 37)
        .run_direct(/*target*/ None, &mut connection, /*skip*/ false)
        .await
        .expect("apply old schema");
    sqlx::query("CREATE TABLE preserved_by_upgrade (value TEXT NOT NULL)")
        .execute(&mut connection)
        .await
        .expect("create preserved data table");
    sqlx::query("INSERT INTO preserved_by_upgrade (value) VALUES ('still here')")
        .execute(&mut connection)
        .await
        .expect("insert preserved data");
    connection
        .close()
        .await
        .expect("close old-schema connection");

    open_state_concurrently(path.as_path()).await;

    let mut connection = open_raw_connection(path.as_path(), false).await;
    let value = sqlx::query_scalar::<_, String>("SELECT value FROM preserved_by_upgrade")
        .fetch_one(&mut connection)
        .await
        .expect("load preserved data");
    let applied = migration_rows(&mut connection).await;
    let expected = STATE_MIGRATOR
        .migrations
        .iter()
        .filter(|migration| migration.migration_type.is_up_migration())
        .map(|migration| (migration.version, true, migration.checksum.to_vec()))
        .collect::<Vec<_>>();
    assert_eq!((value, applied), ("still here".to_string(), expected));

    connection
        .close()
        .await
        .expect("close inspection connection");
    let _ = tokio::fs::remove_dir_all(codex_home).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_hot_initialization_succeeds_while_writes_continue() {
    let codex_home = create_home().await;
    let path = codex_home.join(crate::STATE_DB_FILENAME);
    let writer_pool = open_state(path.as_path()).await;
    let stop = Arc::new(AtomicBool::new(false));
    let writer_stop = Arc::clone(&stop);
    let writer = tokio::spawn(async move {
        while !writer_stop.load(Ordering::Relaxed) {
            sqlx::query("UPDATE backfill_state SET updated_at = updated_at WHERE id = 1")
                .execute(&writer_pool)
                .await
                .expect("concurrent write should succeed");
            tokio::task::yield_now().await;
        }
        drop(writer_pool);
    });

    open_state_concurrently(path.as_path()).await;

    stop.store(true, Ordering::Relaxed);
    writer.await.expect("writer task should complete");
    let _ = tokio::fs::remove_dir_all(codex_home).await;
}
