/// The current Codex CLI version as embedded at compile time.
pub const CODEX_CLI_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The source-control branch captured when this Codex CLI was built.
#[cfg(not(test))]
pub(crate) const CODEX_BUILD_BRANCH: &str = env!("CODEX_BUILD_BRANCH");

/// The UTC build timestamp captured when this Codex CLI was built.
#[cfg(not(test))]
pub(crate) const CODEX_BUILD_TIMESTAMP: &str = env!("CODEX_BUILD_TIMESTAMP");
