use std::fmt;

use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceDialect {
    ClaudeCode,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireApi {
    AnthropicMessages,
    Responses,
}

impl fmt::Display for WireApi {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AnthropicMessages => formatter.write_str("anthropic_messages"),
            Self::Responses => formatter.write_str("responses"),
        }
    }
}

impl fmt::Display for InferenceDialect {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ClaudeCode => formatter.write_str("claude_code"),
            Self::Other => formatter.write_str("other"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnthropicThinkingPolicy {
    Budgeted { budget_tokens: u32 },
    Adaptive,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaudeRequestProfile {
    pub wire_api: WireApi,
    pub dialect: InferenceDialect,
    pub wire_model: String,
    pub max_output_tokens: u32,
    pub thinking: AnthropicThinkingPolicy,
    pub supports_disabled_thinking: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
    Ultra,
    Persistent,
    Custom(String),
}

impl fmt::Display for ReasoningEffort {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => formatter.write_str("none"),
            Self::Minimal => formatter.write_str("minimal"),
            Self::Low => formatter.write_str("low"),
            Self::Medium => formatter.write_str("medium"),
            Self::High => formatter.write_str("high"),
            Self::XHigh => formatter.write_str("xhigh"),
            Self::Max => formatter.write_str("max"),
            Self::Ultra => formatter.write_str("ultra"),
            Self::Persistent => formatter.write_str("persistent"),
            Self::Custom(value) => formatter.write_str(value),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClaudeFunctionTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ClaudeToolSpec {
    Function(ClaudeFunctionTool),
    Unsupported { kind: &'static str, name: String },
}

impl ClaudeToolSpec {
    pub fn name(&self) -> &str {
        match self {
            Self::Function(tool) => &tool.name,
            Self::Unsupported { name, .. } => name,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalOutcome {
    Completed,
    ToolsReady,
    Continue,
    OutputExhausted,
    Refusal,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TokenUsage {
    pub input_tokens: i64,
    pub cached_input_tokens: i64,
    pub cache_write_input_tokens: i64,
    pub output_tokens: i64,
    pub reasoning_output_tokens: i64,
    pub total_tokens: i64,
}
