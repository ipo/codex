use std::collections::BTreeMap;

use codex_api::TerminalOutcome;
use codex_chat_completions::ChunkUsage;
use codex_chat_completions::DialectContext;
use codex_chat_completions::DialectError;
use codex_chat_completions::FinishReason;
use codex_chat_completions::PendingResult;
use codex_chat_completions::ReasoningDelta;
use codex_chat_completions::UsageDetails;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ResponseItem;
use serde_json::Value;
use serde_json::json;
use thiserror::Error;

use crate::KimiDialect;

const REPLAY_PREFIX: &str = "codex:kimi-chat-reasoning:";
const REPLAY_VERSION: u64 = 1;
const MAX_MARKER_LEN: usize = 512;
const REASONING_KEYS: [&str; 3] = ["reasoning_content", "reasoning", "reasoning_details"];

#[derive(Debug, Error, PartialEq, Eq)]
pub enum KimiResponseError {
    #[error("invalid Kimi reasoning provenance `{0}`")]
    InvalidProvenance(String),
    #[error("Kimi replay marker exceeded its size bound")]
    MarkerTooLarge,
}

pub fn response_items(
    dialect: &KimiDialect,
    pending: PendingResult,
) -> Result<Vec<ResponseItem>, KimiResponseError> {
    let key = pending
        .reasoning_provenance
        .as_deref()
        .unwrap_or(REASONING_KEYS[0]);
    if !REASONING_KEYS.contains(&key) {
        return Err(KimiResponseError::InvalidProvenance(key.to_string()));
    }
    let marker = encode_marker(dialect.context(), key)?;
    let mut items = vec![ResponseItem::Reasoning {
        id: None,
        summary: Vec::new(),
        content: Some(vec![ReasoningItemContent::ReasoningText {
            text: pending.reasoning,
        }]),
        encrypted_content: Some(marker),
        internal_chat_message_metadata_passthrough: None,
    }];
    if !pending.content.is_empty() {
        items.push(ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: pending.content,
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        });
    }
    for call in pending.tool_calls {
        items.push(ResponseItem::FunctionCall {
            id: None,
            name: call.function.name,
            namespace: None,
            arguments: call.function.arguments,
            call_id: call.id,
            internal_chat_message_metadata_passthrough: None,
        });
    }
    Ok(items)
}

pub(crate) fn reasoning_delta(
    extensions: &BTreeMap<String, Value>,
) -> Result<Option<ReasoningDelta>, DialectError> {
    for key in REASONING_KEYS {
        if let Some(text) = extensions.get(key).and_then(Value::as_str) {
            return Ok(Some(ReasoningDelta {
                text: text.to_string(),
                provenance: key.to_string(),
            }));
        }
    }
    Ok(None)
}

pub(crate) fn finish_reason(reason: FinishReason) -> TerminalOutcome {
    match reason {
        FinishReason::Stop => TerminalOutcome::Completed,
        FinishReason::ToolCalls | FinishReason::FunctionCall => TerminalOutcome::ToolsReady,
        FinishReason::Length | FinishReason::MaxTokens => TerminalOutcome::OutputExhausted,
        FinishReason::ContentFilter => TerminalOutcome::Refusal,
    }
}

pub(crate) fn usage_details(usage: &ChunkUsage) -> Result<UsageDetails, DialectError> {
    Ok(UsageDetails {
        cached_prompt_tokens: nested_u64(usage, "cached_tokens", "prompt_tokens_details")?,
        reasoning_tokens: nested_u64(usage, "reasoning_tokens", "completion_tokens_details")?,
    })
}

fn nested_u64(
    usage: &ChunkUsage,
    direct_key: &str,
    details_key: &str,
) -> Result<u64, DialectError> {
    let direct = usage.details.get(direct_key);
    let nested = usage
        .details
        .get(details_key)
        .and_then(Value::as_object)
        .and_then(|details| details.get(direct_key));
    let value = direct.or(nested);
    value
        .map(|value| {
            value.as_u64().ok_or_else(|| {
                DialectError::InvalidResponseExtension(format!(
                    "usage {direct_key} must be an unsigned integer"
                ))
            })
        })
        .transpose()
        .map(Option::unwrap_or_default)
}

pub(crate) fn replay_field(
    context: DialectContext<'_>,
    opaque: Option<&str>,
) -> Result<Option<&'static str>, DialectError> {
    let Some(opaque) = opaque else {
        return Ok(Some(REASONING_KEYS[0]));
    };
    let Some(encoded) = opaque.strip_prefix(REPLAY_PREFIX) else {
        return Ok(Some(REASONING_KEYS[0]));
    };
    if opaque.len() > MAX_MARKER_LEN {
        return Err(DialectError::MalformedReplay(
            "Kimi replay marker exceeded its size bound".to_string(),
        ));
    }
    let marker: Value = serde_json::from_str(encoded)
        .map_err(|error| DialectError::MalformedReplay(error.to_string()))?;
    let version = marker.get("version").and_then(Value::as_u64);
    if version != Some(REPLAY_VERSION) {
        return Err(DialectError::MalformedReplay(
            "unsupported Kimi replay marker version".to_string(),
        ));
    }
    let marker_dialect = marker.get("dialect").and_then(Value::as_str);
    if marker_dialect.is_none()
        || marker.get("model").and_then(Value::as_str).is_none()
        || marker.get("key").and_then(Value::as_str).is_none()
    {
        return Err(DialectError::MalformedReplay(
            "invalid Kimi replay marker shape".to_string(),
        ));
    }
    if marker_dialect != Some("kimi_chat")
        || marker["model"] != context.model
        || context.dialect != InferenceDialect::Kimi
    {
        return Ok(None);
    }
    let key = marker["key"].as_str().unwrap_or_default();
    REASONING_KEYS
        .iter()
        .copied()
        .find(|candidate| *candidate == key)
        .map(Some)
        .ok_or_else(|| DialectError::MalformedReplay("unknown Kimi reasoning key".to_string()))
}

pub(crate) fn marker_is_bounded(model: &str) -> bool {
    REASONING_KEYS
        .iter()
        .all(|key| raw_marker(model, key).len() <= MAX_MARKER_LEN)
}

fn encode_marker(context: DialectContext<'_>, key: &str) -> Result<String, KimiResponseError> {
    let marker = raw_marker(context.model, key);
    if marker.len() > MAX_MARKER_LEN {
        return Err(KimiResponseError::MarkerTooLarge);
    }
    Ok(marker)
}

fn raw_marker(model: &str, key: &str) -> String {
    format!(
        "{REPLAY_PREFIX}{}",
        json!({"version":REPLAY_VERSION,"dialect":"kimi_chat","model":model,"key":key})
    )
}
