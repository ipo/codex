//! Shared SQLite connection configuration.

use crate::DbTelemetry;
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
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

mod init;

const LOGS_DB_FILENAME: &str = "logs_2.sqlite";
const GOALS_DB_FILENAME: &str = "goals_1.sqlite";
const MEMORIES_DB_FILENAME: &str = "memories_1.sqlite";
const STATE_DB_FILENAME: &str = "state_5.sqlite";
const THREAD_HISTORY_DB_FILENAME: &str = "thread_history_1.sqlite";

#[derive(Clone, Copy)]
struct RuntimeDbSpec {
    label: &'static str,
    filename: &'static str,
    kind: DbKind,
    check_phase: &'static str,
    lock_phase: &'static str,
    bootstrap_phase: &'static str,
    open_phase: &'static str,
    migrate_phase: &'static str,
    retry_phase: &'static str,
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
    check_phase: "check_state",
    lock_phase: "wait_init_lock_state",
    bootstrap_phase: "bootstrap_state",
    open_phase: "open_state",
    migrate_phase: "migrate_state",
    retry_phase: "retry_state",
};

const LOGS_DB: RuntimeDbSpec = RuntimeDbSpec {
    label: "log DB",
    filename: LOGS_DB_FILENAME,
    kind: DbKind::Logs,
    check_phase: "check_logs",
    lock_phase: "wait_init_lock_logs",
    bootstrap_phase: "bootstrap_logs",
    open_phase: "open_logs",
    migrate_phase: "migrate_logs",
    retry_phase: "retry_logs",
};

const GOALS_DB: RuntimeDbSpec = RuntimeDbSpec {
    label: "goals DB",
    filename: GOALS_DB_FILENAME,
    kind: DbKind::Goals,
    check_phase: "check_goals",
    lock_phase: "wait_init_lock_goals",
    bootstrap_phase: "bootstrap_goals",
    open_phase: "open_goals",
    migrate_phase: "migrate_goals",
    retry_phase: "retry_goals",
};

const MEMORIES_DB: RuntimeDbSpec = RuntimeDbSpec {
    label: "memories DB",
    filename: MEMORIES_DB_FILENAME,
    kind: DbKind::Memories,
    check_phase: "check_memories",
    lock_phase: "wait_init_lock_memories",
    bootstrap_phase: "bootstrap_memories",
    open_phase: "open_memories",
    migrate_phase: "migrate_memories",
    retry_phase: "retry_memories",
};

const THREAD_HISTORY_DB: RuntimeDbSpec = RuntimeDbSpec {
    label: "thread history DB",
    filename: THREAD_HISTORY_DB_FILENAME,
    kind: DbKind::ThreadHistory,
    check_phase: "check_thread_history",
    lock_phase: "wait_init_lock_thread_history",
    bootstrap_phase: "bootstrap_thread_history",
    open_phase: "open_thread_history",
    migrate_phase: "migrate_thread_history",
    retry_phase: "retry_thread_history",
};

const RUNTIME_DBS: [RuntimeDbSpec; 5] =
    [STATE_DB, LOGS_DB, GOALS_DB, MEMORIES_DB, THREAD_HISTORY_DB];

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
        init::open_sqlite(path.as_path(), migrator, spec, telemetry_override).await
    }

    /// Open a writable Codex SQLite database, creating it if necessary.
    pub async fn open_read_write_pool(&self, path: &Path) -> Result<SqlitePool, Error> {
        let options = init::base_sqlite_options(path)
            .journal_mode(SqliteJournalMode::Wal)
            .auto_vacuum(SqliteAutoVacuum::Incremental)
            .create_if_missing(true);
        SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await
    }

    /// Open an existing Codex SQLite database without creating or modifying it.
    pub async fn open_read_only_pool(&self, path: &Path) -> Result<SqlitePool, Error> {
        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(false)
            .read_only(true)
            .log_statements(LevelFilter::Off);
        SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await
    }

    pub(crate) async fn open_log_maintenance_connection(
        &self,
        busy_timeout: Duration,
    ) -> Result<SqliteConnection, Error> {
        let options = init::base_sqlite_options(&self.logs_db_path())
            .create_if_missing(false)
            .busy_timeout(busy_timeout);
        SqliteConnection::connect_with(&options).await
    }
}
