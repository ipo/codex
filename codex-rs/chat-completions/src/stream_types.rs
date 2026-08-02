use std::collections::BTreeMap;

use codex_api::TerminalOutcome;
use serde::Deserialize;
use serde_json::Value;

use crate::ToolCall;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(default)]
    pub usage: Option<ChunkUsage>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub usage: Option<ChunkUsage>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ChunkDelta {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCallFragment>,
    #[serde(flatten)]
    pub extensions: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ToolCallFragment {
    pub index: usize,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub function: Option<ToolFunctionFragment>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ToolFunctionFragment {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct ChunkUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    #[serde(flatten)]
    pub details: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    ToolCalls,
    FunctionCall,
    Length,
    MaxTokens,
    ContentFilter,
}

impl FinishReason {
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "stop" => Some(Self::Stop),
            "tool_calls" => Some(Self::ToolCalls),
            "function_call" => Some(Self::FunctionCall),
            "length" => Some(Self::Length),
            "max_tokens" => Some(Self::MaxTokens),
            "content_filter" => Some(Self::ContentFilter),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResponseMetadata {
    pub trace_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PendingResult {
    pub content: String,
    pub reasoning: String,
    pub reasoning_provenance: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedStream {
    pub response_id: String,
    pub terminal_outcome: TerminalOutcome,
    pub pending: Option<PendingResult>,
    pub usage: Option<ChunkUsage>,
    pub usage_details: crate::UsageDetails,
    pub metadata: ResponseMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PresentationDelta {
    Content(String),
    Reasoning(String),
    ToolArguments { index: usize, delta: String },
}
