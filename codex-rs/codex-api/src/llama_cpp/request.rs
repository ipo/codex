use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ReasoningEffort;
use serde::Serialize;
use serde_json::Value;

use crate::ApiError;
use crate::Reasoning;
use crate::ResponsesApiTools;

use super::LlamaCppCatalogEntry;

/// Provider-neutral input used to build one direct llama.cpp Responses request.
#[derive(Debug, Clone)]
pub struct LlamaCppRequestInput {
    pub instructions: String,
    pub history: Vec<ResponseItem>,
    pub tools: Option<ResponsesApiTools>,
    pub parallel_tool_calls: bool,
    pub reasoning_effort: Option<ReasoningEffort>,
}

/// A request that passed exact llama.cpp input-token preflight.
#[derive(Debug, Clone)]
pub struct LlamaCppPreparedRequest {
    pub model: LlamaCppCatalogEntry,
    pub(super) body: Value,
}

impl LlamaCppPreparedRequest {
    /// Returns the exact inference body that passed llama.cpp token preflight.
    pub fn body(&self) -> &Value {
        &self.body
    }
}

pub(super) fn llama_cpp_request_body(
    model: &LlamaCppCatalogEntry,
    input: LlamaCppRequestInput,
) -> Result<Value, ApiError> {
    let effort = local_effort(input.reasoning_effort)?;
    let thinking = effort != ReasoningEffort::None;
    let (temperature, top_p, presence_penalty) = if thinking {
        (1.0, 0.95, 0.0)
    } else {
        (0.7, 0.8, 1.5)
    };
    serde_json::to_value(LlamaCppResponsesRequest {
        model: model.wire_model.clone(),
        instructions: input.instructions,
        input: input.history,
        tools: input.tools,
        tool_choice: "auto",
        parallel_tool_calls: input.parallel_tool_calls,
        reasoning: Reasoning {
            effort: Some(effort),
            summary: None,
            context: None,
        },
        store: false,
        stream: true,
        include: Vec::new(),
        cache_prompt: true,
        max_output_tokens: model.max_output_tokens,
        chat_template_kwargs: ChatTemplateKwargs {
            enable_thinking: thinking,
            preserve_thinking: thinking,
        },
        temperature,
        top_p,
        top_k: 20,
        min_p: 0.0,
        presence_penalty,
        repeat_penalty: 1.0,
    })
    .map_err(|error| ApiError::InvalidRequest {
        message: format!("failed to encode llama.cpp request: {error}"),
    })
}

fn local_effort(effort: Option<ReasoningEffort>) -> Result<ReasoningEffort, ApiError> {
    match effort.unwrap_or(ReasoningEffort::Low) {
        effort @ (ReasoningEffort::None
        | ReasoningEffort::Low
        | ReasoningEffort::Medium
        | ReasoningEffort::XHigh) => Ok(effort),
        ReasoningEffort::High => Ok(ReasoningEffort::XHigh),
        effort @ (ReasoningEffort::Minimal
        | ReasoningEffort::Max
        | ReasoningEffort::Ultra
        | ReasoningEffort::Persistent
        | ReasoningEffort::Custom(_)) => Err(ApiError::InvalidRequest {
            message: format!("llama.cpp reasoning effort `{effort}` is unsupported"),
        }),
    }
}

#[derive(Debug, Serialize)]
struct LlamaCppResponsesRequest {
    model: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    instructions: String,
    input: Vec<ResponseItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<ResponsesApiTools>,
    tool_choice: &'static str,
    parallel_tool_calls: bool,
    reasoning: Reasoning,
    store: bool,
    stream: bool,
    include: Vec<String>,
    cache_prompt: bool,
    max_output_tokens: u64,
    chat_template_kwargs: ChatTemplateKwargs,
    temperature: f64,
    top_p: f64,
    top_k: u32,
    min_p: f64,
    presence_penalty: f64,
    repeat_penalty: f64,
}

#[derive(Debug, Serialize)]
struct ChatTemplateKwargs {
    enable_thinking: bool,
    preserve_thinking: bool,
}
