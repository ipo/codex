use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

use super::request::KimiReasoning;
use super::request::KimiRequestError;
use super::request::KimiToolCall;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KimiTerminal {
    Completed,
    ToolsReady,
    OutputExhausted,
    Refusal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KimiStreamEvent {
    Content {
        response_id: String,
        delta: String,
    },
    Reasoning {
        response_id: String,
        delta: String,
    },
    ToolCall {
        response_id: String,
        index: usize,
        id: String,
        name: String,
        arguments_delta: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KimiUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub cached_prompt_tokens: u64,
    pub reasoning_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KimiPendingResponse {
    pub content: String,
    pub reasoning: KimiReasoning,
    pub tool_calls: Vec<KimiToolCall>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KimiDecodedResponse {
    pub response_id: String,
    pub terminal: KimiTerminal,
    /// Present only when the terminal permits history/tool commit.
    pub pending: Option<KimiPendingResponse>,
    pub usage: Option<KimiUsage>,
    pub trace_id: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct Chunk {
    pub(super) id: String,
    pub(super) choices: Vec<Choice>,
    #[serde(default)]
    pub(super) usage: Option<RawUsage>,
}

#[derive(Deserialize)]
pub(super) struct Choice {
    pub(super) index: usize,
    pub(super) delta: Delta,
    #[serde(default)]
    pub(super) finish_reason: Option<String>,
    #[serde(default)]
    pub(super) usage: Option<RawUsage>,
}

#[derive(Default, Deserialize)]
pub(super) struct Delta {
    #[serde(default)]
    pub(super) role: Option<String>,
    #[serde(default)]
    pub(super) content: Option<String>,
    #[serde(default)]
    pub(super) tool_calls: Vec<ToolFragment>,
    #[serde(flatten)]
    pub(super) extensions: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
pub(super) struct ToolFragment {
    pub(super) index: usize,
    #[serde(default)]
    pub(super) id: Option<String>,
    #[serde(rename = "type", default)]
    pub(super) kind: Option<String>,
    #[serde(default)]
    pub(super) function: Option<FunctionFragment>,
}

#[derive(Deserialize)]
pub(super) struct FunctionFragment {
    #[serde(default)]
    pub(super) name: Option<String>,
    #[serde(default)]
    pub(super) arguments: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct RawUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    #[serde(default)]
    cached_tokens: Option<u64>,
    #[serde(default)]
    reasoning_tokens: Option<u64>,
    #[serde(default)]
    prompt_tokens_details: Option<PromptDetails>,
    #[serde(default)]
    completion_tokens_details: Option<CompletionDetails>,
}

impl RawUsage {
    pub(super) fn into_usage(self) -> KimiUsage {
        let nested_cached = self
            .prompt_tokens_details
            .and_then(|details| details.cached_tokens);
        let nested_reasoning = self
            .completion_tokens_details
            .and_then(|details| details.reasoning_tokens);
        KimiUsage {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            total_tokens: self.total_tokens,
            cached_prompt_tokens: self.cached_tokens.or(nested_cached).unwrap_or_default(),
            reasoning_tokens: self
                .reasoning_tokens
                .or(nested_reasoning)
                .unwrap_or_default(),
        }
    }
}

#[derive(Deserialize)]
struct PromptDetails {
    #[serde(default)]
    cached_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct CompletionDetails {
    #[serde(default)]
    reasoning_tokens: Option<u64>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum KimiStreamError {
    #[error("Kimi Chat stream was empty")]
    EmptyStream,
    #[error("Kimi Chat stream was not UTF-8")]
    InvalidUtf8,
    #[error("malformed Kimi Chat SSE line `{0}`")]
    MalformedFraming(String),
    #[error("malformed Kimi Chat chunk: {0}")]
    MalformedChunk(String),
    #[error("invalid Kimi Chat transition: {0}")]
    InvalidTransition(&'static str),
    #[error("Kimi Chat stream ended before {0}")]
    PrematureEof(&'static str),
    #[error("unknown Kimi Chat finish reason `{0}`")]
    UnknownFinishReason(String),
    #[error("Kimi Chat stream contained a duplicate terminal")]
    DuplicateTerminal,
    #[error("Kimi Chat reasoning keys changed during one response")]
    ConflictingReasoningKeys,
    #[error("Kimi Chat tool contained conflicting {0} fragments")]
    ConflictingToolField(&'static str),
    #[error("Kimi Chat tool {index} was incomplete: {field}")]
    IncompleteTool { index: usize, field: &'static str },
    #[error("Kimi Chat tool {index} contained invalid JSON object: {arguments}")]
    InvalidToolJson { index: usize, arguments: String },
    #[error(transparent)]
    Request(#[from] KimiRequestError),
}
