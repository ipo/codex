use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ReasoningEffort;
use codex_tools::ToolSpec;
use thiserror::Error;

use crate::AssembleError;
use crate::AssembleRequest;
use crate::AssembledRequest;
use crate::SystemBlock;
use crate::history::encode_history;

/// Canonical structured-output state for a Codex prompt.
#[derive(Debug, Clone, Copy)]
pub enum CanonicalOutputSchema<'a> {
    Disabled,
    JsonSchema { schema: &'a serde_json::Value },
}

/// Canonical Codex inputs for one complete native Claude request.
#[derive(Debug)]
pub struct EncodeRequest<'a> {
    pub profile: &'a codex_protocol::model_inference::ModelInferenceConfig,
    pub effort: &'a ReasoningEffort,
    pub system: &'a [SystemBlock],
    pub history: &'a [ResponseItem],
    pub tools: &'a [ToolSpec],
    pub output_schema: CanonicalOutputSchema<'a>,
    pub resumable_session_id: &'a str,
    pub codex_version: &'a str,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum EncodeError {
    #[error(transparent)]
    Assembly(#[from] AssembleError),
    #[error("unsupported history item at index {index}: {kind}")]
    UnsupportedHistoryItem { index: usize, kind: &'static str },
    #[error("unsupported content at history item {item_index}, block {block_index}: {kind}")]
    UnsupportedContent {
        item_index: usize,
        block_index: usize,
        kind: &'static str,
    },
    #[error("tool call `{call_id}` contains invalid JSON arguments: {message}")]
    InvalidToolArguments { call_id: String, message: String },
    #[error("invalid base64 image at history item {item_index}, block {block_index}: {message}")]
    InvalidImage {
        item_index: usize,
        block_index: usize,
        message: String,
    },
    #[error("tool result `{call_id}` has no matching unresolved tool use")]
    UnmatchedToolResult { call_id: String },
    #[error("duplicate unresolved tool use ID `{call_id}`")]
    DuplicateToolUse { call_id: String },
    #[error("structured-output schemas are unsupported by native Claude Messages encoding")]
    UnsupportedStructuredOutput,
    #[error(transparent)]
    Replay(#[from] crate::ReplayError),
}

/// Converts canonical history and assembles one complete native Claude request.
pub fn encode_request(params: EncodeRequest<'_>) -> Result<AssembledRequest, EncodeError> {
    if let CanonicalOutputSchema::JsonSchema { .. } = params.output_schema {
        return Err(EncodeError::UnsupportedStructuredOutput);
    }
    let messages = encode_history(params.history, params.profile)?;
    Ok(crate::assemble_request(AssembleRequest {
        profile: params.profile,
        effort: params.effort,
        messages: &messages,
        system: params.system,
        tools: params.tools,
        resumable_session_id: params.resumable_session_id,
        codex_version: params.codex_version,
    })?)
}
