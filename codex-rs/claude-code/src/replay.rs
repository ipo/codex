use codex_protocol::model_inference::InferenceDialect;
use serde::Deserialize;
use serde::Serialize;
use thiserror::Error;

const PREFIX: &str = "codex:anthropic-thinking:";
const VERSION: u8 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ThinkingReplayBlock {
    Signed { thinking: String, signature: String },
    Redacted { data: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayDecision {
    Native(Vec<ThinkingReplayBlock>),
    UnrelatedOpaqueContent { visible: Vec<String> },
    ModelOrDialectChanged { visible: Vec<String> },
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ReplayError {
    #[error("native model for Anthropic thinking replay is empty")]
    EmptyModel,
    #[error("malformed Anthropic thinking replay envelope: {0}")]
    Malformed(String),
    #[error("unsupported Anthropic thinking replay envelope version {0}")]
    UnknownVersion(u64),
}

#[derive(Serialize, Deserialize)]
struct ReplayEnvelope {
    version: u8,
    dialect: InferenceDialect,
    model: String,
    blocks: Vec<ThinkingReplayBlock>,
}

pub fn encode_thinking_replay(
    dialect: InferenceDialect,
    model: &str,
    blocks: Vec<ThinkingReplayBlock>,
) -> Result<String, ReplayError> {
    if model.is_empty() {
        return Err(ReplayError::EmptyModel);
    }
    let envelope = ReplayEnvelope {
        version: VERSION,
        dialect,
        model: model.to_string(),
        blocks,
    };
    let json = serde_json::to_string(&envelope)
        .map_err(|error| ReplayError::Malformed(error.to_string()))?;
    Ok(format!("{PREFIX}{json}"))
}

pub fn decode_thinking_replay(
    opaque: &str,
    visible: Vec<String>,
    target_dialect: InferenceDialect,
    target_model: &str,
) -> Result<ReplayDecision, ReplayError> {
    let Some(json) = opaque.strip_prefix(PREFIX) else {
        return Ok(ReplayDecision::UnrelatedOpaqueContent { visible });
    };
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|error| ReplayError::Malformed(error.to_string()))?;
    let version = value
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| ReplayError::Malformed("missing numeric version".to_string()))?;
    if version != u64::from(VERSION) {
        return Err(ReplayError::UnknownVersion(version));
    }
    let envelope: ReplayEnvelope =
        serde_json::from_value(value).map_err(|error| ReplayError::Malformed(error.to_string()))?;
    if envelope.dialect == target_dialect && envelope.model == target_model {
        Ok(ReplayDecision::Native(envelope.blocks))
    } else {
        Ok(ReplayDecision::ModelOrDialectChanged { visible })
    }
}
