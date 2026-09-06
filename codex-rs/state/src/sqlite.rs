//! Shared SQLite connection configuration.

#![expect(
    clippy::disallowed_methods,
    reason = "this is the centralized SQLite connection shim"
)]

use crate::DbTelemetry;
use crate::migrations::repair_legacy_recency_migration_version;
use crate::runtime::RuntimeDbInitError;
use crate::telemetry;
use crate::telemetry::DbKind;
use codex_utils_absolute_path::AbsolutePathBuf;
use log::LevelFilter;
use sqlx::ConnectOptions;
use sqlx::Connection;
use sqlx::Error;
use sqlx::SqliteConnection;
use sqlx::SqlitePool;
use sqlx::migrate::Migrator;
use sqlx::sqlite::SqliteAutoVacuum;
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::sqlite::SqliteJournalMode;
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::sqlite::SqliteSynchronous;
use std::fs::File;
use std::fs::OpenOptions;
use std::fs::TryLockError;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

const INIT_LOCK_TIMEOUT: Duration = Duration::from_secs(/*secs*/ 60);
const LOGS_DB_FILENAME: &str = "logs_2.sqlite";
const GOALS_DB_FILENAME: &str = "goals_1.sqlite";
const MEMORIES_DB_FILENAME: &str = "memories_1.sqlite";
const QUEUE_DB_FILENAME: &str = "queue_1.sqlite";
const STATE_DB_FILENAME: &str = "state_5.sqlite";
const THREAD_HISTORY_DB_FILENAME: &str = "thread_history_1.sqlite";

#[derive(Clone, Copy)]
struct RuntimeDbSpec {
    label: &'static str,
    filename: &'static str,
    kind: DbKind,
    lock_phase: &'static str,
    open_phase: &'static str,
    migrate_phase: &'static str,
}

impl RuntimeDbSpec {
    fn path(self, codex_home: &Path) -> PathBuf {
        codex_home.join(self.filename)
    }
}

const STATE_DB: RuntimeDbSpec = RuntimeDbSpec {
    label: "state DB",
    filename: STATE_DB_FILENAME,
    kind: DbKind::State,
    lock_phase: "wait_init_lock_state",
    open_phase: "open_state",
    migrate_phase: "migrate_state",
};

const LOGS_DB: RuntimeDbSpec = RuntimeDbSpec {
    label: "log DB",
    filename: LOGS_DB_FILENAME,
    kind: DbKind::Logs,
    lock_phase: "wait_init_lock_logs",
    open_phase: "open_logs",
    migrate_phase: "migrate_logs",
};

const GOALS_DB: RuntimeDbSpec = RuntimeDbSpec {
    label: "goals DB",
    filename: GOALS_DB_FILENAME,
    kind: DbKind::Goals,
    lock_phase: "wait_init_lock_goals",
    open_phase: "open_goals",
    migrate_phase: "migrate_goals",
};

const MEMORIES_DB: RuntimeDbSpec = RuntimeDbSpec {
    label: "memories DB",
    filename: MEMORIES_DB_FILENAME,
    kind: DbKind::Memories,
    lock_phase: "wait_init_lock_memories",
    open_phase: "open_memories",
    migrate_phase: "migrate_memories",
};

const QUEUE_DB: RuntimeDbSpec = RuntimeDbSpec {
    label: "queue DB",
    filename: QUEUE_DB_FILENAME,
    kind: DbKind::Queue,
    lock_phase: "wait_init_lock_queue",
    open_phase: "open_queue",
    migrate_phase: "migrate_queue",
};

const THREAD_HISTORY_DB: RuntimeDbSpec = RuntimeDbSpec {
    label: "thread history DB",
    filename: THREAD_HISTORY_DB_FILENAME,
    kind: DbKind::ThreadHistory,
    lock_phase: "wait_init_lock_thread_history",
    open_phase: "open_thread_history",
    migrate_phase: "migrate_thread_history",
};

const RUNTIME_DBS: [RuntimeDbSpec; 6] = [
    STATE_DB,
    LOGS_DB,
    GOALS_DB,
    MEMORIES_DB,
    QUEUE_DB,
    THREAD_HISTORY_DB,
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeDbPath {
    pub label: &'static str,
    pub path: PathBuf,
}

/// Resolved configuration shared by all Codex SQLite connections.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SqliteConfig {
    sqlite_home: AbsolutePathBuf,
}

impl SqliteConfig {
    pub fn from_sqlite_home(sqlite_home: AbsolutePathBuf) -> Self {
        Self { sqlite_home }
    }

    pub fn new_for_testing(sqlite_home: AbsolutePathBuf) -> Self {
        Self::from_sqlite_home(sqlite_home)
    }

    pub fn home(&self) -> &Path {
        self.sqlite_home.as_path()
    }

    /// Return the path to the primary state database.
    pub fn state_db_path(&self) -> PathBuf {
        STATE_DB.path(self.home())
    }

    /// Return the path to the logs database.
    pub fn logs_db_path(&self) -> PathBuf {
        LOGS_DB.path(self.home())
    }

    /// Return the path to the goals database.
    pub fn goals_db_path(&self) -> PathBuf {
        GOALS_DB.path(self.home())
    }

    /// Return the path to the memories database.
    pub fn memories_db_path(&self) -> PathBuf {
        MEMORIES_DB.path(self.home())
    }

    /// Return the path to the durable user-message queue database.
    pub fn queue_db_path(&self) -> PathBuf {
        QUEUE_DB.path(self.home())
    }

    /// Return the path to the paginated thread-history database.
    pub fn thread_history_db_path(&self) -> PathBuf {
        THREAD_HISTORY_DB.path(self.home())
    }

    /// Return the paths to every database managed by the state runtime.
    pub fn runtime_db_paths(&self) -> Vec<RuntimeDbPath> {
        RUNTIME_DBS
            .iter()
            .map(|spec| RuntimeDbPath {
                label: spec.label,
                path: spec.path(self.home()),
            })
            .collect()
    }

    pub(super) async fn open_state_db(
        &self,
        migrator: &Migrator,
        telemetry_override: Option<&dyn DbTelemetry>,
    ) -> anyhow::Result<SqlitePool> {
        // New state DBs should use incremental auto-vacuum, but retrofitting an
        // existing DB requires a full VACUUM. Do not attempt that during process
        // startup: it is maintenance work that can contend with foreground writers.
        self.open_runtime_db(STATE_DB, migrator, telemetry_override)
            .await
    }

    pub(super) async fn open_logs_db(
        &self,
        migrator: &Migrator,
        telemetry_override: Option<&dyn DbTelemetry>,
    ) -> anyhow::Result<SqlitePool> {
        self.open_runtime_db(LOGS_DB, migrator, telemetry_override)
            .await
    }

    pub(super) async fn open_goals_db(
        &self,
        migrator: &Migrator,
        telemetry_override: Option<&dyn DbTelemetry>,
    ) -> anyhow::Result<SqlitePool> {
        self.open_runtime_db(GOALS_DB, migrator, telemetry_override)
            .await
    }

    pub(super) async fn open_memories_db(
        &self,
        migrator: &Migrator,
        telemetry_override: Option<&dyn DbTelemetry>,
    ) -> anyhow::Result<SqlitePool> {
        self.open_runtime_db(MEMORIES_DB, migrator, telemetry_override)
            .await
    }

    pub(super) async fn open_queue_db(
        &self,
        migrator: &Migrator,
        telemetry_override: Option<&dyn DbTelemetry>,
    ) -> anyhow::Result<SqlitePool> {
        self.open_runtime_db(QUEUE_DB, migrator, telemetry_override)
            .await
    }

    pub(super) async fn open_thread_history_db(
        &self,
        migrator: &Migrator,
        telemetry_override: Option<&dyn DbTelemetry>,
    ) -> anyhow::Result<SqlitePool> {
        self.open_runtime_db(THREAD_HISTORY_DB, migrator, telemetry_override)
            .await
    }

    async fn open_runtime_db(
        &self,
        spec: RuntimeDbSpec,
        migrator: &Migrator,
        telemetry_override: Option<&dyn DbTelemetry>,
    ) -> anyhow::Result<SqlitePool> {
        let path = spec.path(self.home());
        let started = Instant::now();
        let lock_result = acquire_init_lock(path.as_path()).await;
        telemetry::record_init_result(
            telemetry_override,
            spec.kind,
            spec.lock_phase,
            started.elapsed(),
            &lock_result,
        );
        let _init_lock = lock_result.map_err(|source| {
            RuntimeDbInitError::new(spec.label, "lock", path.as_path(), source)
        })?;
        let started = Instant::now();
        let pool_result = if matches!(spec.kind, DbKind::Logs) {
            self.try_open_current_logs_pool(&path, migrator).await
        } else {
            Ok(None)
        };
        let pool_result = match pool_result {
            Ok(Some(pool)) => Ok((pool, false)),
            Ok(None) => self
                .open_read_write_pool(&path)
                .await
                .map(|pool| (pool, true))
                .map_err(anyhow::Error::from),
            Err(err) => Err(err),
        };
        telemetry::record_init_result(
            telemetry_override,
            spec.kind,
            spec.open_phase,
            started.elapsed(),
            &pool_result,
        );
        let (pool, needs_migration) = pool_result.map_err(|source| {
            RuntimeDbInitError::new(spec.label, "open", path.as_path(), source)
        })?;
        let started = Instant::now();
        let migrate_result = async {
            if needs_migration {
                if matches!(spec.kind, DbKind::State) {
                    repair_legacy_recency_migration_version(&pool, migrator).await?;
                }
                migrator.run(&pool).await.map_err(anyhow::Error::from)?;
            }
            Ok::<_, anyhow::Error>(())
        }
        .await;
        telemetry::record_init_result(
            telemetry_override,
            spec.kind,
            spec.migrate_phase,
            started.elapsed(),
            &migrate_result,
        );
        if let Err(source) = migrate_result {
            pool.close().await;
            return Err(
                RuntimeDbInitError::new(spec.label, "migrate", path.as_path(), source).into(),
            );
        }
        Ok(pool)
    }

    async fn try_open_current_logs_pool(
        &self,
        path: &Path,
        migrator: &Migrator,
    ) -> anyhow::Result<Option<SqlitePool>> {
        if !path.exists() {
            return Ok(None);
        }
        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(false)
            .synchronous(SqliteSynchronous::Normal)
            .busy_timeout(Duration::from_secs(5))
            .log_statements(LevelFilter::Off);
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await?;
        let current_result = logs_database_is_current(&pool, migrator).await;
        if matches!(current_result, Ok(true)) {
            Ok(Some(pool))
        } else {
            pool.close().await;
            Ok(None)
        }
    }

    /// Open a writable Codex SQLite database, creating it if necessary.
    pub async fn open_read_write_pool(&self, path: &Path) -> Result<SqlitePool, Error> {
        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal)
            .auto_vacuum(SqliteAutoVacuum::Incremental)
            .busy_timeout(Duration::from_secs(5))
            .log_statements(LevelFilter::Off);
        SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await
    }

    /// Open an existing Codex SQLite database without creating or modifying it.
    pub async fn open_read_only_pool(
        &self,
        path: &Path,
        busy_timeout: Option<Duration>,
    ) -> Result<SqlitePool, Error> {
        let mut options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(false)
            .read_only(true)
            .log_statements(LevelFilter::Off);
        if let Some(busy_timeout) = busy_timeout {
            options = options.busy_timeout(busy_timeout);
        }
        SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await
    }

    pub(crate) async fn open_log_maintenance_connection(
        &self,
        busy_timeout: Duration,
    ) -> Result<SqliteConnection, Error> {
        let options = SqliteConnectOptions::new()
            .filename(self.logs_db_path())
            .create_if_missing(false)
            .synchronous(SqliteSynchronous::Normal)
            .busy_timeout(busy_timeout)
            .log_statements(LevelFilter::Off);
        SqliteConnection::connect_with(&options).await
    }
}

async fn logs_database_is_current(pool: &SqlitePool, migrator: &Migrator) -> anyhow::Result<bool> {
    let journal_mode = sqlx::query_scalar::<_, String>("PRAGMA journal_mode")
        .fetch_one(pool)
        .await?;
    if !journal_mode.eq_ignore_ascii_case("wal") {
        return Ok(false);
    }
    let applied = sqlx::query_as::<_, (i64, bool, Vec<u8>)>(
        "SELECT version, success, checksum FROM _sqlx_migrations ORDER BY version",
    )
    .fetch_all(pool)
    .await?;
    if applied.iter().any(|(_, success, _)| !success) {
        return Ok(false);
    }
    Ok(migrator
        .migrations
        .iter()
        .filter(|migration| migration.migration_type.is_up_migration())
        .all(|migration| {
            applied.iter().any(|(version, _, checksum)| {
                *version == migration.version && checksum.as_slice() == migration.checksum.as_ref()
            })
        }))
}

struct InitLock {
    _file: File,
}

async fn acquire_init_lock(db_path: &Path) -> anyhow::Result<InitLock> {
    let lock_path = init_lock_path(db_path);
    let file = open_owner_only_lock_file(lock_path.as_path())?;
    let started = Instant::now();
    let mut attempt = 0_u32;
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(InitLock { _file: file }),
            Err(TryLockError::WouldBlock) => {
                let delay = init_lock_retry_delay(attempt);
                if started.elapsed().saturating_add(delay) >= INIT_LOCK_TIMEOUT {
                    return Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        format!(
                            "timed out waiting for SQLite initialization lock {}",
                            lock_path.display()
                        ),
                    )
                    .into());
                }
                tokio::time::sleep(delay).await;
                attempt = attempt.saturating_add(1);
            }
            Err(TryLockError::Error(error)) => return Err(error.into()),
        }
    }
}

fn init_lock_path(db_path: &Path) -> PathBuf {
    let mut path = db_path.as_os_str().to_os_string();
    path.push(".init.lock");
    PathBuf::from(path)
}

pub(crate) fn open_owner_only_lock_file(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
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

fn init_lock_retry_delay(attempt: u32) -> Duration {
    let base_millis = 10_u64.saturating_mul(1_u64 << attempt.min(3));
    let jitter_millis = (u64::from(std::process::id()) + u64::from(attempt) * 29) % 31;
    Duration::from_millis(base_millis.saturating_add(jitter_millis))
}

#[cfg(test)]
#[path = "sqlite_tests.rs"]
mod tests;
