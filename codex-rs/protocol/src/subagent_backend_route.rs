use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;
use ts_rs::TS;

/// Selects whether a logical subagent sends the inference backend's subagent marker.
/// `MainSession` only suppresses that marker; both routes remain subagents inside Codex.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq, JsonSchema, TS)]
#[serde(rename_all = "snake_case")]
#[ts(rename_all = "snake_case")]
pub enum SubagentBackendRoute {
    #[default]
    ProperSubagent,
    MainSession,
}

impl SubagentBackendRoute {
    pub fn is_proper_subagent(&self) -> bool {
        matches!(self, Self::ProperSubagent)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::ProperSubagent => "proper_subagent",
            Self::MainSession => "main_session",
        }
    }
}
