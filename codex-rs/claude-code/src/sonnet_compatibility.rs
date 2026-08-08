use crate::ClaudeCodeIdentity;
use crate::RequestTransport;

pub(crate) const SONNET_WIRE_MODEL: &str = "claude-sonnet-5";
const CLAUDE_CODE_VERSION: &str = "2.1.223";
const ANTHROPIC_BETAS: &str = "claude-code-20250219,interleaved-thinking-2025-05-14,redact-thinking-2026-02-12,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,advisor-tool-2026-03-01,effort-2025-11-24";

/// Header-only Claude Code identity for a Sonnet 5 request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SonnetCompatibilityContext {
    pub identity: ClaudeCodeIdentity,
}

impl SonnetCompatibilityContext {
    pub(crate) fn transport(&self) -> RequestTransport {
        self.identity
            .transport(CLAUDE_CODE_VERSION, ANTHROPIC_BETAS, "Linux", "x64")
    }
}
