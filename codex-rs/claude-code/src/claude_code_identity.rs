use std::collections::BTreeMap;

use uuid::Uuid;

use crate::RequestTransport;

const ID_NAMESPACE: Uuid = Uuid::from_u128(0xd73e_49ad_9688_54f9_a068_fbc7_66b5_821a);

/// Identifies whether a Claude Code-compatible request belongs to the root or a subagent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaudeCodeRequestKind {
    Root,
    Subagent,
}

/// Stable session and subagent identities shared by Claude Code-compatible profiles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaudeCodeIdentity {
    pub kind: ClaudeCodeRequestKind,
    pub session_id: String,
    pub thread_id: String,
}

impl ClaudeCodeIdentity {
    pub(crate) fn transport(
        &self,
        version: &str,
        betas: &str,
        stainless_os: &str,
        stainless_arch: &str,
    ) -> RequestTransport {
        let mut headers = BTreeMap::from([
            (
                "User-Agent".to_string(),
                format!("claude-cli/{version} (external, cli)"),
            ),
            ("accept".to_string(), "application/json".to_string()),
            ("anthropic-beta".to_string(), betas.to_string()),
            (
                "anthropic-dangerous-direct-browser-access".to_string(),
                "true".to_string(),
            ),
            ("anthropic-version".to_string(), "2023-06-01".to_string()),
            ("x-app".to_string(), "cli".to_string()),
            (
                "x-claude-code-session-id".to_string(),
                self.session_id.clone(),
            ),
            ("x-stainless-arch".to_string(), stainless_arch.to_string()),
            ("x-stainless-lang".to_string(), "js".to_string()),
            ("x-stainless-os".to_string(), stainless_os.to_string()),
            (
                "x-stainless-package-version".to_string(),
                "0.94.0".to_string(),
            ),
            ("x-stainless-retry-count".to_string(), "0".to_string()),
            ("x-stainless-runtime".to_string(), "node".to_string()),
            (
                "x-stainless-runtime-version".to_string(),
                "v26.3.0".to_string(),
            ),
            ("x-stainless-timeout".to_string(), "600".to_string()),
        ]);
        if let Some(agent_id) = self.agent_id() {
            headers.insert("x-claude-code-agent-id".to_string(), agent_id);
        }
        RequestTransport {
            method: "POST",
            path: "/v1/messages",
            query: BTreeMap::from([("beta".to_string(), "true".to_string())]),
            headers,
        }
    }

    pub(crate) fn agent_id(&self) -> Option<String> {
        match self.kind {
            ClaudeCodeRequestKind::Root => None,
            ClaudeCodeRequestKind::Subagent => {
                let id = stable_uuid("agent", &self.thread_id).simple().to_string();
                Some(format!("a{}", &id[..16]))
            }
        }
    }
}

pub(crate) fn stable_uuid(domain: &str, identity: &str) -> Uuid {
    Uuid::new_v5(&ID_NAMESPACE, format!("{domain}:{identity}").as_bytes())
}
