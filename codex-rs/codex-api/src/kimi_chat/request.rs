use std::collections::BTreeMap;
use std::collections::BTreeSet;

use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::model_inference::KimiInferenceConfig;
use codex_protocol::model_inference::KimiThinkingPolicy;
use codex_protocol::model_inference::WireApi;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;
use thiserror::Error;

use super::schema::KimiSchemaError;
use super::schema::normalize_kimi_schema;

const MAX_AFFINITY_LEN: usize = 256;
const MAX_REPLAY_MARKER_LEN: usize = 512;
const REPLAY_PREFIX: &str = "codex:kimi-chat-reasoning:";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KimiThinkingEffort {
    Low,
    High,
    Max,
}

impl KimiThinkingEffort {
    fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::High => "high",
            Self::Max => "max",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KimiRequestSettings {
    pub context_window: u64,
    pub input_estimate: KimiInputEstimate,
    pub prompt_cache_key: String,
    pub thinking_effort: Option<KimiThinkingEffort>,
    pub reasoning_key: Option<KimiReasoningKey>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KimiInputEstimate {
    FinalSerialized,
    Fixed(u64),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum KimiContent {
    Text(String),
    Parts(Vec<Value>),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum KimiMessage {
    System {
        content: KimiContent,
    },
    User {
        content: KimiContent,
    },
    Assistant(KimiAssistantMessage),
    Tool {
        tool_call_id: String,
        content: KimiContent,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct KimiAssistantMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<KimiToolCall>,
    #[serde(flatten)]
    reasoning: BTreeMap<String, Value>,
}

impl KimiAssistantMessage {
    pub fn from_input(
        input: KimiAssistantMessageInput<'_>,
        wire_model: &str,
    ) -> Result<Self, KimiRequestError> {
        let mut fields = BTreeMap::new();
        if let Some(reasoning) = input.reasoning {
            if let Some((key, text)) = reasoning.replay_for(wire_model)? {
                fields.insert(key.as_str().to_string(), text.into());
            }
        } else {
            fields.insert(
                KimiReasoningKey::ReasoningContent.as_str().to_string(),
                String::new().into(),
            );
        }
        Ok(Self {
            content: input.content,
            tool_calls: input.tool_calls,
            reasoning: fields,
        })
    }
}

pub struct KimiAssistantMessageInput<'a> {
    pub content: Option<String>,
    pub tool_calls: Vec<KimiToolCall>,
    pub reasoning: Option<&'a KimiReasoning>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KimiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    kind: KimiToolKind,
    pub function: KimiFunction,
}

impl KimiToolCall {
    pub fn function(id: impl Into<String>, name: impl Into<String>, arguments: String) -> Self {
        Self {
            id: id.into(),
            kind: KimiToolKind::Function,
            function: KimiFunction {
                name: name.into(),
                arguments,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum KimiToolKind {
    Function,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KimiFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct KimiFunctionTool {
    #[serde(rename = "type")]
    kind: KimiToolKind,
    pub function: KimiFunctionDefinition,
}

impl KimiFunctionTool {
    pub fn new(function: KimiFunctionDefinition) -> Self {
        Self {
            kind: KimiToolKind::Function,
            function,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct KimiFunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub strict: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct KimiChatRequest {
    pub model: String,
    pub messages: Vec<KimiMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<KimiFunctionTool>,
    pub stream: bool,
    pub stream_options: Value,
    pub max_completion_tokens: u64,
    pub prompt_cache_key: String,
    pub thinking: Value,
}

pub fn build_kimi_chat_request(
    config: &KimiInferenceConfig,
    settings: KimiRequestSettings,
    mut messages: Vec<KimiMessage>,
    mut tools: Vec<KimiFunctionTool>,
) -> Result<KimiChatRequest, KimiRequestError> {
    if (config.wire_api, config.dialect) != (WireApi::ChatCompletions, InferenceDialect::Kimi)
        || config.wire_model.is_empty()
        || config.wire_model.len() > MAX_AFFINITY_LEN
        || [
            KimiReasoningKey::ReasoningContent,
            KimiReasoningKey::Reasoning,
            KimiReasoningKey::ReasoningDetails,
        ]
        .into_iter()
        .any(|key| KimiReasoning::for_model(&config.wire_model, key, String::new()).is_err())
        || config.max_output_tokens == 0
        || settings.context_window == 0
    {
        return Err(KimiRequestError::InvalidProfile);
    }
    if settings.prompt_cache_key.is_empty() || settings.prompt_cache_key.len() > MAX_AFFINITY_LEN {
        return Err(KimiRequestError::InvalidPromptCacheKey);
    }
    match (config.thinking, settings.thinking_effort) {
        (KimiThinkingPolicy::RequiredWithEffort, None) => {
            return Err(KimiRequestError::MissingThinkingEffort);
        }
        (KimiThinkingPolicy::Required, Some(_)) => {
            return Err(KimiRequestError::UnsupportedThinkingEffort);
        }
        (KimiThinkingPolicy::RequiredWithEffort, Some(_))
        | (KimiThinkingPolicy::Required, None) => {}
    }
    normalize_messages(&mut messages, settings.reasoning_key)?;
    for (index, tool) in tools.iter_mut().enumerate() {
        tool.function.parameters = normalize_kimi_schema(&tool.function.parameters)
            .map_err(|source| KimiRequestError::ToolSchema { index, source })?;
    }
    let estimated_input_tokens = match settings.input_estimate {
        KimiInputEstimate::FinalSerialized => estimate_kimi_input_tokens(&messages, &tools),
        KimiInputEstimate::Fixed(estimated_input_tokens) => estimated_input_tokens,
    };
    let remaining = settings
        .context_window
        .saturating_sub(estimated_input_tokens)
        .max(1);
    let mut thinking = json!({"type":"enabled","keep":"all"});
    if let Some(effort) = settings.thinking_effort {
        thinking["effort"] = effort.as_str().into();
    }
    Ok(KimiChatRequest {
        model: config.wire_model.clone(),
        messages,
        tools,
        stream: true,
        stream_options: json!({"include_usage":true}),
        max_completion_tokens: u64::from(config.max_output_tokens).min(remaining),
        prompt_cache_key: settings.prompt_cache_key,
        thinking,
    })
}

pub fn estimate_kimi_input_tokens(messages: &[KimiMessage], tools: &[KimiFunctionTool]) -> u64 {
    serde_json::to_vec(&(messages, tools))
        .map(|input| u64::try_from(input.len()).unwrap_or(u64::MAX).div_ceil(4))
        .unwrap_or(u64::MAX)
}

fn normalize_messages(
    messages: &mut [KimiMessage],
    selected_reasoning_key: Option<KimiReasoningKey>,
) -> Result<(), KimiRequestError> {
    let mut replacements = BTreeMap::new();
    let mut normalized = BTreeSet::new();
    for message in messages.iter() {
        if let KimiMessage::Assistant(assistant) = message {
            for call in &assistant.tool_calls {
                let id = normalize_call_id(&call.id);
                if !normalized.insert(id.clone()) && !replacements.contains_key(&call.id) {
                    return Err(KimiRequestError::DuplicateToolCallId(id));
                }
                replacements.insert(call.id.clone(), id);
            }
        }
    }
    let reasoning_key = selected_reasoning_key
        .map(|key| key.as_str().to_string())
        .or_else(|| {
            messages
                .iter()
                .filter_map(|message| match message {
                    KimiMessage::Assistant(assistant) => assistant.reasoning.keys().next(),
                    KimiMessage::System { .. }
                    | KimiMessage::User { .. }
                    | KimiMessage::Tool { .. } => None,
                })
                .next_back()
                .cloned()
        })
        .unwrap_or_else(|| KimiReasoningKey::ReasoningContent.as_str().to_string());
    for message in messages {
        match message {
            KimiMessage::Assistant(assistant) => {
                let reasoning = assistant
                    .reasoning
                    .pop_first()
                    .map(|(_, value)| value)
                    .unwrap_or_else(|| String::new().into());
                assistant.reasoning.insert(reasoning_key.clone(), reasoning);
                for call in &mut assistant.tool_calls {
                    call.id = replacements[&call.id].clone();
                }
                if !assistant.tool_calls.is_empty() && assistant.content.as_deref() == Some("") {
                    assistant.content = None;
                }
            }
            KimiMessage::Tool { tool_call_id, .. } => {
                if let Some(replacement) = replacements.get(tool_call_id) {
                    tool_call_id.clone_from(replacement);
                }
            }
            KimiMessage::System { .. } | KimiMessage::User { .. } => {}
        }
    }
    Ok(())
}

fn normalize_call_id(call_id: &str) -> String {
    if call_id.len() <= 64 {
        return call_id.to_string();
    }
    let hash = call_id.bytes().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    });
    let prefix_end = (0..=47)
        .rev()
        .find(|index| call_id.is_char_boundary(*index))
        .unwrap_or_default();
    format!("{}-{hash:016x}", &call_id[..prefix_end])
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KimiReasoningKey {
    ReasoningContent,
    Reasoning,
    ReasoningDetails,
}

impl KimiReasoningKey {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::ReasoningContent => "reasoning_content",
            Self::Reasoning => "reasoning",
            Self::ReasoningDetails => "reasoning_details",
        }
    }
}

/// Plaintext Kimi reasoning paired with its bounded, dialect-specific replay marker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KimiReasoning {
    pub text: String,
    marker: String,
}

impl KimiReasoning {
    pub fn from_persisted(text: String, marker: String) -> Result<Self, KimiRequestError> {
        let reasoning = Self { text, marker };
        reasoning.decode_marker()?;
        Ok(reasoning)
    }

    pub fn opaque_marker(&self) -> &str {
        &self.marker
    }

    pub fn key(&self) -> Result<KimiReasoningKey, KimiRequestError> {
        self.decode_marker().map(|marker| marker.key)
    }

    pub fn for_model(
        wire_model: &str,
        key: KimiReasoningKey,
        text: String,
    ) -> Result<Self, KimiRequestError> {
        Self::from_response(wire_model, key, text)
    }

    pub(crate) fn from_response(
        wire_model: &str,
        key: KimiReasoningKey,
        text: String,
    ) -> Result<Self, KimiRequestError> {
        let marker = format!(
            "{REPLAY_PREFIX}{}",
            json!({"version":1,"dialect":"kimi_chat","model":wire_model,"key":key})
        );
        if marker.len() > MAX_REPLAY_MARKER_LEN {
            return Err(KimiRequestError::ReplayMarkerTooLarge);
        }
        Ok(Self { text, marker })
    }

    fn replay_for(
        &self,
        wire_model: &str,
    ) -> Result<Option<(KimiReasoningKey, String)>, KimiRequestError> {
        let marker = self.decode_marker()?;
        if marker.model != wire_model {
            return Ok(None);
        }
        Ok(Some((marker.key, self.text.clone())))
    }

    fn decode_marker(&self) -> Result<ReplayMarker, KimiRequestError> {
        if self.marker.len() > MAX_REPLAY_MARKER_LEN {
            return Err(KimiRequestError::MalformedReplayMarker);
        }
        let encoded = self
            .marker
            .strip_prefix(REPLAY_PREFIX)
            .ok_or(KimiRequestError::MalformedReplayMarker)?;
        let marker: ReplayMarker =
            serde_json::from_str(encoded).map_err(|_| KimiRequestError::MalformedReplayMarker)?;
        if marker.version != 1 || marker.dialect != "kimi_chat" {
            return Err(KimiRequestError::MalformedReplayMarker);
        }
        Ok(marker)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplayMarker {
    version: u64,
    dialect: String,
    model: String,
    key: KimiReasoningKey,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum KimiRequestError {
    #[error("invalid Kimi inference profile")]
    InvalidProfile,
    #[error("invalid Kimi prompt cache key")]
    InvalidPromptCacheKey,
    #[error("this Kimi profile requires a thinking effort")]
    MissingThinkingEffort,
    #[error("this Kimi profile does not accept a thinking effort")]
    UnsupportedThinkingEffort,
    #[error("two tool-call IDs normalize to the same Kimi ID `{0}`")]
    DuplicateToolCallId(String),
    #[error("Kimi reasoning replay marker is malformed")]
    MalformedReplayMarker,
    #[error("Kimi reasoning replay marker exceeds its size bound")]
    ReplayMarkerTooLarge,
    #[error("Kimi tool schema at index {index} is unsupported: {source}")]
    ToolSchema {
        index: usize,
        source: KimiSchemaError,
    },
}
