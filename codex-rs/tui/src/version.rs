/// The current Codex CLI version as embedded at compile time.
pub const CODEX_CLI_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Source branch captured at build time.
pub const CODEX_BUILD_BRANCH: &str = env!("CODEX_BUILD_BRANCH", "unknown");
/// UTC build timestamp captured at build time.
pub const CODEX_BUILD_TIMESTAMP: &str = env!("CODEX_BUILD_TIMESTAMP", "unknown");
