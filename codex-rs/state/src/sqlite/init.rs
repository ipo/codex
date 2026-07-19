use super::RuntimeDbSpec;
use crate::migrations::repair_legacy_recency_migration_version;
use crate::runtime::RuntimeDbInitError;
use crate::runtime::sqlite_error_detail_is_lock;
use crate::telemetry::DbKind;
use crate::telemetry::DbTelemetry;
use anyhow::anyhow;
use log::LevelFilter;
use sqlx::ConnectOptions;
use sqlx::Connection;
use sqlx::Row;
use sqlx::SqliteConnection;
use sqlx::SqlitePool;
use sqlx::migrate::MigrateError;
use sqlx::migrate::Migrator;
use sqlx::sqlite::SqliteAutoVacuum;
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::sqlite::SqliteJournalMode;
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::sqlite::SqliteSynchronous;
use std::collections::BTreeMap;
use std::fs::File;
use std::fs::OpenOptions;
use std::future::Future;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

const SQLITE_BUSY_TIMEOUT: Duration = Duration::from_secs(5);
const INIT_LOCK_TIMEOUT: Duration = Duration::from_secs(60);
const TRANSIENT_RETRY_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SchemaState {
    Current,
    NeedsInitialization(InitializationPlan),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InitializationPlan {
    is_new: bool,
    needs_wal: bool,
    needs_migration: bool,
}

struct InitLock {
    _file: File,
}

pub(super) fn base_sqlite_options(path: &Path) -> SqliteConnectOptions {
    SqliteConnectOptions::new()
        .filename(path)
        .create_if_missing(true)
        .synchronous(SqliteSynchronous::Normal)
        .busy_timeout(SQLITE_BUSY_TIMEOUT)
        .log_statements(LevelFilter::Off)
}

pub(super) async fn open_sqlite(
    path: &Path,
    migrator: &Migrator,
    spec: RuntimeDbSpec,
    telemetry_override: Option<&dyn DbTelemetry>,
) -> anyhow::Result<SqlitePool> {
    let schema_state = check_database(path, migrator, spec, telemetry_override)
        .await
        .map_err(|source| RuntimeDbInitError::new(spec.label, "check", path, source))?;
    if matches!(schema_state, SchemaState::NeedsInitialization(_)) {
        ensure_initialized(path, migrator, spec, telemetry_override).await?;
    }

    let options = base_sqlite_options(path).create_if_missing(false);
    let started = Instant::now();
    let pool_result = retry_transient(spec, telemetry_override, || {
        let options = options.clone();
        async move {
            SqlitePoolOptions::new()
                .max_connections(5)
                .connect_with(options)
                .await
                .map_err(anyhow::Error::from)
        }
    })
    .await;
    crate::telemetry::record_init_result(
        telemetry_override,
        spec.kind,
        spec.open_phase,
        started.elapsed(),
        &pool_result,
    );
    pool_result.map_err(|source| {
        RuntimeDbInitError::new(spec.label, "open", path, source).into()
    })
}

async fn ensure_initialized(
    path: &Path,
    migrator: &Migrator,
    spec: RuntimeDbSpec,
    telemetry_override: Option<&dyn DbTelemetry>,
) -> anyhow::Result<()> {
    let started = Instant::now();
    let lock_result = acquire_init_lock(path).await;
    crate::telemetry::record_init_result(
        telemetry_override,
        spec.kind,
        spec.lock_phase,
        started.elapsed(),
        &lock_result,
    );
    let _lock = lock_result
        .map_err(|source| RuntimeDbInitError::new(spec.label, "lock", path, source))?;

    let schema_state = check_database(path, migrator, spec, telemetry_override)
        .await
        .map_err(|source| RuntimeDbInitError::new(spec.label, "check", path, source))?;
    let SchemaState::NeedsInitialization(plan) = schema_state else {
        return Ok(());
    };

    let options = initialization_options(path, plan);
    let started = Instant::now();
    let connection_result = retry_transient(spec, telemetry_override, || {
        let options = options.clone();
        async move {
            SqliteConnection::connect_with(&options)
                .await
                .map_err(Into::into)
        }
    })
    .await;
    crate::telemetry::record_init_result(
        telemetry_override,
        spec.kind,
        spec.bootstrap_phase,
        started.elapsed(),
        &connection_result,
    );
    let mut connection = connection_result.map_err(|source| {
        RuntimeDbInitError::new(spec.label, "bootstrap", path, source)
    })?;

    if plan.needs_migration {
        let started = Instant::now();
        let retry_started = Instant::now();
        let mut attempt = 0_u32;
        let migrate_result = loop {
            let result = async {
                if matches!(spec.kind, DbKind::State) {
                    repair_legacy_recency_migration_version(&mut connection, migrator).await?;
                }
                migrator
                    .run_direct(/*target*/ None, &mut connection, /*skip*/ false)
                    .await
                    .map_err(anyhow::Error::from)
            }
            .await;
            match result {
                Ok(()) => break Ok(()),
                Err(error) => {
                    let delay = retry_delay(attempt);
                    if !is_transient_lock_error(&error)
                        || retry_started.elapsed().saturating_add(delay) >= TRANSIENT_RETRY_TIMEOUT
                    {
                        break Err(error);
                    }
                    crate::telemetry::record_init_retry(
                        telemetry_override,
                        spec.kind,
                        spec.retry_phase,
                        retry_started.elapsed(),
                        &error,
                    );
                    tokio::time::sleep(delay).await;
                    attempt = attempt.saturating_add(1);
                }
            }
        };
        crate::telemetry::record_init_result(
            telemetry_override,
            spec.kind,
            spec.migrate_phase,
            started.elapsed(),
            &migrate_result,
        );
        if let Err(source) = migrate_result {
            let _ = connection.close().await;
            return Err(
                RuntimeDbInitError::new(spec.label, "migrate", path, source).into(),
            );
        }
    }
    connection.close().await.map_err(|source| {
        RuntimeDbInitError::new(spec.label, "bootstrap", path, source.into())
    })?;

    match check_database(path, migrator, spec, telemetry_override)
        .await
        .map_err(|source| RuntimeDbInitError::new(spec.label, "check", path, source))?
    {
        SchemaState::Current => Ok(()),
        SchemaState::NeedsInitialization(_) => Err(RuntimeDbInitError::new(
            spec.label,
            "initialize",
            path,
            anyhow!("database still requires initialization after bootstrap"),
        )
        .into()),
    }
}

fn initialization_options(path: &Path, plan: InitializationPlan) -> SqliteConnectOptions {
    let mut options = base_sqlite_options(path).create_if_missing(plan.is_new);
    if plan.is_new {
        options = options
            .auto_vacuum(SqliteAutoVacuum::Incremental)
            .journal_mode(SqliteJournalMode::Wal);
    } else if plan.needs_wal {
        options = options.journal_mode(SqliteJournalMode::Wal);
    }
    options
}

async fn check_database(
    path: &Path,
    migrator: &Migrator,
    spec: RuntimeDbSpec,
    telemetry_override: Option<&dyn DbTelemetry>,
) -> anyhow::Result<SchemaState> {
    let started = Instant::now();
    let result = retry_transient(spec, telemetry_override, || {
        inspect_database(path, migrator, spec)
    })
    .await;
    crate::telemetry::record_init_result(
        telemetry_override,
        spec.kind,
        spec.check_phase,
        started.elapsed(),
        &result,
    );
    result
}

async fn inspect_database(
    path: &Path,
    migrator: &Migrator,
    spec: RuntimeDbSpec,
) -> anyhow::Result<SchemaState> {
    let metadata = match tokio::fs::metadata(path).await {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            return Ok(SchemaState::NeedsInitialization(InitializationPlan {
                is_new: true,
                needs_wal: true,
                needs_migration: true,
            }));
        }
        Err(err) => return Err(err.into()),
    };
    if metadata.is_file() && metadata.len() == 0 {
        return Ok(SchemaState::NeedsInitialization(InitializationPlan {
            is_new: true,
            needs_wal: true,
            needs_migration: true,
        }));
    }

    let options = SqliteConnectOptions::new()
        .filename(path)
        .create_if_missing(false)
        .read_only(true)
        .busy_timeout(SQLITE_BUSY_TIMEOUT)
        .log_statements(LevelFilter::Off);
    let mut connection = SqliteConnection::connect_with(&options).await?;
    let journal_mode = sqlx::query_scalar::<_, String>("PRAGMA journal_mode")
        .fetch_one(&mut connection)
        .await?;
    let needs_wal = !journal_mode.eq_ignore_ascii_case("wal");
    let migrations_table_exists = sqlx::query_scalar::<_, i64>(
        "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = ?",
    )
    .bind(migrator.table_name.as_ref())
    .fetch_optional(&mut connection)
    .await?
    .is_some();
    if !migrations_table_exists {
        connection.close().await?;
        return Ok(SchemaState::NeedsInitialization(InitializationPlan {
            is_new: false,
            needs_wal,
            needs_migration: true,
        }));
    }

    let escaped_table_name = migrator.table_name.replace('"', "\"\"");
    let rows = sqlx::query(sqlx::AssertSqlSafe(format!(
        "SELECT version, success, checksum FROM \"{escaped_table_name}\" ORDER BY version"
    )))
    .fetch_all(&mut connection)
    .await?;
    let mut applied = BTreeMap::new();
    for row in rows {
        let version = row.get::<i64, _>("version");
        if !row.get::<bool, _>("success") {
            return Err(MigrateError::Dirty(version).into());
        }
        applied.insert(version, row.get::<Vec<u8>, _>("checksum"));
    }

    let legacy_recency = if matches!(spec.kind, DbKind::State) {
        migrator
            .iter()
            .find(|migration| migration.version == 39)
            .is_some_and(|migration| {
                applied.get(&38).map(Vec::as_slice) == Some(migration.checksum.as_ref())
                    && !applied.contains_key(&39)
            })
    } else {
        false
    };
    let mut needs_migration = legacy_recency;
    for migration in migrator
        .iter()
        .filter(|migration| migration.migration_type.is_up_migration())
    {
        if legacy_recency && migration.version == 38 {
            continue;
        }
        match applied.get(&migration.version) {
            Some(checksum) if checksum.as_slice() != migration.checksum.as_ref() => {
                return Err(MigrateError::VersionMismatch(migration.version).into());
            }
            Some(_) => {}
            None => needs_migration = true,
        }
    }

    let schema_state = if needs_wal || needs_migration {
        SchemaState::NeedsInitialization(InitializationPlan {
            is_new: false,
            needs_wal,
            needs_migration,
        })
    } else {
        SchemaState::Current
    };
    connection.close().await?;
    Ok(schema_state)
}

async fn acquire_init_lock(db_path: &Path) -> anyhow::Result<InitLock> {
    let lock_path = init_lock_path(db_path);
    let file = open_owner_only_lock_file(lock_path.as_path())?;
    let started = Instant::now();
    let mut attempt = 0_u32;
    loop {
        match fs2::FileExt::try_lock_exclusive(&file) {
            Ok(()) => return Ok(InitLock { _file: file }),
            Err(err) if lock_is_contended(&err) => {
                let delay = retry_delay(attempt);
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
            Err(err) => return Err(err.into()),
        }
    }
}

fn init_lock_path(db_path: &Path) -> PathBuf {
    let mut path = db_path.as_os_str().to_os_string();
    path.push(".init.lock");
    PathBuf::from(path)
}

fn open_owner_only_lock_file(path: &Path) -> io::Result<File> {
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

fn lock_is_contended(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::WouldBlock
        || error.raw_os_error() == fs2::lock_contended_error().raw_os_error()
}

async fn retry_transient<T, Operation, OperationFuture>(
    spec: RuntimeDbSpec,
    telemetry_override: Option<&dyn DbTelemetry>,
    mut operation: Operation,
) -> anyhow::Result<T>
where
    Operation: FnMut() -> OperationFuture,
    OperationFuture: Future<Output = anyhow::Result<T>>,
{
    let started = Instant::now();
    let mut attempt = 0_u32;
    loop {
        match operation().await {
            Ok(value) => return Ok(value),
            Err(error) => {
                let delay = retry_delay(attempt);
                if !is_transient_lock_error(&error)
                    || started.elapsed().saturating_add(delay) >= TRANSIENT_RETRY_TIMEOUT
                {
                    return Err(error);
                }
                crate::telemetry::record_init_retry(
                    telemetry_override,
                    spec.kind,
                    spec.retry_phase,
                    started.elapsed(),
                    &error,
                );
                tokio::time::sleep(delay).await;
                attempt = attempt.saturating_add(1);
            }
        }
    }
}

fn retry_delay(attempt: u32) -> Duration {
    let base_millis = 10_u64.saturating_mul(1_u64 << attempt.min(3));
    let jitter_millis = (u64::from(std::process::id()) + u64::from(attempt) * 29) % 31;
    Duration::from_millis(base_millis.saturating_add(jitter_millis))
}

fn is_transient_lock_error(error: &anyhow::Error) -> bool {
    error.chain().any(|source| {
        let Some(sqlx::Error::Database(database_error)) = source.downcast_ref::<sqlx::Error>()
        else {
            return false;
        };
        if sqlite_error_detail_is_lock(database_error.message()) {
            return true;
        }
        let Some(code) = database_error.code() else {
            return false;
        };
        let code = code.to_ascii_lowercase();
        code.parse::<i32>()
            .is_ok_and(|code| matches!(code & 0xff, 5 | 6))
            || matches!(code.as_str(), "sqlite_busy" | "sqlite_locked")
    })
}

#[cfg(test)]
#[path = "init_tests.rs"]
mod tests;
