use std::collections::BTreeMap;
use std::collections::BTreeSet;

use codex_api::TerminalOutcome;
use codex_chat_completions::AssistantReasoningReplay;
use codex_chat_completions::ChatCompletionsRequest;
use codex_chat_completions::ChatMessage;
use codex_chat_completions::ChunkUsage;
use codex_chat_completions::DialectContext;
use codex_chat_completions::DialectError;
use codex_chat_completions::DialectHooks;
use codex_chat_completions::EncodeError;
use codex_chat_completions::FinishReason;
use codex_chat_completions::OpaqueReasoning;
use codex_chat_completions::ReasoningDelta;
use codex_chat_completions::UsageDetails;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::model_inference::KimiInferenceConfig;
use codex_protocol::model_inference::KimiThinkingPolicy;
use codex_protocol::model_inference::WireApi;
use codex_protocol::models::ResponseItem;
use codex_tools::ToolSpec;
use serde_json::Value;
use serde_json::json;
use thiserror::Error;

use crate::SchemaError;
use crate::normalize_schema;

const MAX_PROFILE_FIELD_LEN: usize = 256;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KimiThinking {
    Enabled,
    Effort(KimiThinkingEffort),
    Off,
}

pub struct KimiRequestSettings {
    pub context_window: u64,
    pub estimated_input_tokens: u64,
    pub prompt_cache_key: String,
    pub thinking: KimiThinking,
}

pub struct KimiEncodeRequest<'a> {
    pub system: Option<&'a str>,
    pub history: &'a [ResponseItem],
    pub tools: &'a [ToolSpec],
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum KimiError {
    #[error("incompatible Kimi request profile: {0}")]
    InvalidProfile(&'static str),
    #[error("Kimi thinking cannot be disabled")]
    DisabledThinking,
    #[error("this Kimi profile requires low, high, or max thinking effort")]
    MissingThinkingEffort,
    #[error("this Kimi profile does not accept a thinking effort")]
    UnsupportedThinkingEffort,
    #[error("two tool-call IDs normalize to the same Kimi ID `{0}`")]
    DuplicateToolCallId(String),
    #[error("tool schema at index {index} is unsupported: {source}")]
    ToolSchema { index: usize, source: SchemaError },
    #[error(transparent)]
    Encode(#[from] EncodeError),
}

/// Stateless request-side Kimi dialect selected from resolved typed metadata.
pub struct KimiDialect {
    profile: KimiInferenceConfig,
    settings: KimiRequestSettings,
}

impl KimiDialect {
    pub fn new(
        profile: KimiInferenceConfig,
        settings: KimiRequestSettings,
    ) -> Result<Self, KimiError> {
        if (profile.wire_api, profile.dialect) != (WireApi::ChatCompletions, InferenceDialect::Kimi)
        {
            return Err(KimiError::InvalidProfile(
                "expected chat_completions with the Kimi dialect",
            ));
        }
        if profile.wire_model.is_empty() || profile.wire_model.len() > MAX_PROFILE_FIELD_LEN {
            return Err(KimiError::InvalidProfile("invalid wire model"));
        }
        if !crate::response::marker_is_bounded(&profile.wire_model) {
            return Err(KimiError::InvalidProfile(
                "wire model cannot fit the bounded replay marker",
            ));
        }
        if profile.max_output_tokens == 0 || settings.context_window == 0 {
            return Err(KimiError::InvalidProfile("invalid token budget"));
        }
        if settings.prompt_cache_key.is_empty()
            || settings.prompt_cache_key.len() > MAX_PROFILE_FIELD_LEN
        {
            return Err(KimiError::InvalidProfile("invalid prompt_cache_key"));
        }
        match (profile.thinking, settings.thinking) {
            (_, KimiThinking::Off) => return Err(KimiError::DisabledThinking),
            (KimiThinkingPolicy::RequiredWithEffort, KimiThinking::Enabled) => {
                return Err(KimiError::MissingThinkingEffort);
            }
            (KimiThinkingPolicy::Required, KimiThinking::Effort(_)) => {
                return Err(KimiError::UnsupportedThinkingEffort);
            }
            (KimiThinkingPolicy::RequiredWithEffort, KimiThinking::Effort(_))
            | (KimiThinkingPolicy::Required, KimiThinking::Enabled) => {}
        }
        Ok(Self { profile, settings })
    }

    pub fn context(&self) -> DialectContext<'_> {
        DialectContext {
            dialect: InferenceDialect::Kimi,
            model: &self.profile.wire_model,
        }
    }

    fn validate_context(&self, context: DialectContext<'_>) -> Result<(), DialectError> {
        if context == self.context() {
            Ok(())
        } else {
            Err(DialectError::InvalidRequestExtension(
                "Kimi dialect/model context mismatch".to_string(),
            ))
        }
    }
}

pub fn encode_request(
    dialect: &KimiDialect,
    params: KimiEncodeRequest<'_>,
) -> Result<ChatCompletionsRequest, KimiError> {
    let mut request =
        codex_chat_completions::encode_request(codex_chat_completions::EncodeRequest {
            context: dialect.context(),
            system: params.system,
            history: params.history,
            tools: params.tools,
            dialect,
        })?;
    for (index, tool) in request.tools.iter_mut().enumerate() {
        tool.function.parameters = normalize_schema(&tool.function.parameters)
            .map_err(|source| KimiError::ToolSchema { index, source })?;
    }
    normalize_messages(&mut request.messages)?;
    Ok(request)
}

fn normalize_messages(messages: &mut [ChatMessage]) -> Result<(), KimiError> {
    let mut ids = BTreeMap::new();
    let mut normalized_ids = BTreeSet::new();
    for message in messages.iter() {
        if let ChatMessage::Assistant(assistant) = message {
            for call in &assistant.tool_calls {
                let normalized = normalize_call_id(&call.id);
                if !normalized_ids.insert(normalized.clone()) && !ids.contains_key(&call.id) {
                    return Err(KimiError::DuplicateToolCallId(normalized));
                }
                ids.insert(call.id.clone(), normalized);
            }
        }
    }
    let reasoning_key = messages
        .iter()
        .filter_map(|message| match message {
            ChatMessage::Assistant(assistant) => assistant.reasoning.keys().find(|key| {
                matches!(
                    key.as_str(),
                    "reasoning_content" | "reasoning" | "reasoning_details"
                )
            }),
            ChatMessage::System { .. } | ChatMessage::User { .. } | ChatMessage::Tool { .. } => {
                None
            }
        })
        .next_back()
        .cloned()
        .unwrap_or_else(|| "reasoning_content".to_string());
    for message in messages {
        match message {
            ChatMessage::Assistant(assistant) => {
                let reasoning = ["reasoning_content", "reasoning", "reasoning_details"]
                    .into_iter()
                    .find_map(|key| assistant.reasoning.remove(key))
                    .unwrap_or_else(|| Value::String(String::new()));
                assistant.reasoning.insert(reasoning_key.clone(), reasoning);
                for call in &mut assistant.tool_calls {
                    call.id = ids[&call.id].clone();
                }
                if !assistant.tool_calls.is_empty() && assistant.content.as_deref() == Some("") {
                    assistant.content = None;
                }
            }
            ChatMessage::Tool { tool_call_id, .. } => {
                if let Some(normalized) = ids.get(tool_call_id) {
                    *tool_call_id = normalized.clone();
                }
            }
            ChatMessage::System { .. } | ChatMessage::User { .. } => {}
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

impl DialectHooks for KimiDialect {
    fn request_extensions(
        &self,
        context: DialectContext<'_>,
    ) -> Result<BTreeMap<String, Value>, DialectError> {
        self.validate_context(context)?;
        let remaining = self
            .settings
            .context_window
            .saturating_sub(self.settings.estimated_input_tokens);
        let max_completion_tokens = u64::from(self.profile.max_output_tokens).min(remaining.max(1));
        let mut thinking = json!({"type": "enabled", "keep": "all"});
        if let KimiThinking::Effort(effort) = self.settings.thinking {
            thinking["effort"] = effort.as_str().into();
        }
        Ok(BTreeMap::from([
            ("max_completion_tokens".into(), max_completion_tokens.into()),
            (
                "prompt_cache_key".into(),
                self.settings.prompt_cache_key.clone().into(),
            ),
            ("stream".into(), true.into()),
            ("stream_options".into(), json!({"include_usage": true})),
            ("thinking".into(), thinking),
        ]))
    }

    fn assistant_reasoning(
        &self,
        context: DialectContext<'_>,
        replay: AssistantReasoningReplay<'_>,
    ) -> Result<BTreeMap<String, Value>, DialectError> {
        self.validate_context(context)?;
        let opaque = match replay.opaque {
            OpaqueReasoning::None => None,
            OpaqueReasoning::Other(opaque) => Some(opaque),
        };
        let Some(field) = crate::response::replay_field(context, opaque)? else {
            return Ok(BTreeMap::new());
        };
        Ok(BTreeMap::from([(
            field.to_string(),
            replay.visible.concat().into(),
        )]))
    }

    fn reasoning_delta(
        &self,
        context: DialectContext<'_>,
        extensions: &BTreeMap<String, Value>,
    ) -> Result<Option<ReasoningDelta>, DialectError> {
        self.validate_context(context)?;
        crate::response::reasoning_delta(extensions)
    }

    fn finish_reason(
        &self,
        context: DialectContext<'_>,
        reason: FinishReason,
    ) -> Result<TerminalOutcome, DialectError> {
        self.validate_context(context)?;
        Ok(crate::response::finish_reason(reason))
    }

    fn usage_details(
        &self,
        context: DialectContext<'_>,
        usage: &ChunkUsage,
    ) -> Result<UsageDetails, DialectError> {
        self.validate_context(context)?;
        crate::response::usage_details(usage)
    }
}
