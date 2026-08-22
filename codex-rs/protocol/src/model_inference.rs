//! Model-owned inference metadata shared across catalog, provider, and runtime boundaries.

use std::fmt;

use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;
use ts_rs::TS;

const CHAT_WIRE_API_REMOVED_ERROR: &str = "`wire_api = \"chat\"` is no longer supported.\nHow to fix: set `wire_api = \"responses\"` in your provider config.\nMore info: https://github.com/openai/codex/discussions/7782";

/// The request and response grammar used for model inference.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, JsonSchema, TS)]
#[serde(rename_all = "snake_case")]
pub enum WireApi {
    #[default]
    Responses,
    AnthropicMessages,
    ChatCompletions,
}

impl fmt::Display for WireApi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Responses => "responses",
            Self::AnthropicMessages => "anthropic_messages",
            Self::ChatCompletions => "chat_completions",
        })
    }
}

impl<'de> Deserialize<'de> for WireApi {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        match value.as_str() {
            "responses" => Ok(Self::Responses),
            "anthropic_messages" => Ok(Self::AnthropicMessages),
            "chat_completions" => Ok(Self::ChatCompletions),
            "chat" => Err(serde::de::Error::custom(CHAT_WIRE_API_REMOVED_ERROR)),
            _ => Err(serde::de::Error::unknown_variant(
                &value,
                &["responses", "anthropic_messages", "chat_completions"],
            )),
        }
    }
}

/// Provider-specific semantics layered on top of a wire grammar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TS)]
#[serde(rename_all = "snake_case")]
pub enum InferenceDialect {
    OpenAi,
    ClaudeCode,
    Kimi,
    Grok,
    LlamaCpp,
}

impl fmt::Display for InferenceDialect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::OpenAi => "open_ai",
            Self::ClaudeCode => "claude_code",
            Self::Kimi => "kimi",
            Self::Grok => "grok",
            Self::LlamaCpp => "llama_cpp",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    OpenAi,
    Anthropic,
    Kimi,
    Grok,
    LlamaCpp,
}

impl fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::OpenAi => "open_ai",
            Self::Anthropic => "anthropic",
            Self::Kimi => "kimi",
            Self::Grok => "grok",
            Self::LlamaCpp => "llama_cpp",
        })
    }
}

/// Anthropic thinking behavior selected by a model profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TS)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicThinkingPolicy {
    Budgeted { budget_tokens: u32 },
    Adaptive,
}

/// Required thinking behavior for a managed Kimi model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TS)]
#[serde(rename_all = "snake_case")]
pub enum KimiThinkingPolicy {
    RequiredWithEffort,
    Required,
}

/// Native inference contract for the Kimi model family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TS)]
pub struct KimiInferenceConfig {
    pub wire_api: WireApi,
    pub dialect: InferenceDialect,
    pub route: String,
    pub wire_model: String,
    pub max_output_tokens: u32,
    pub thinking: KimiThinkingPolicy,
}

/// Native Responses contract for the Grok model family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TS)]
pub struct GrokInferenceConfig {
    pub wire_api: WireApi,
    pub dialect: InferenceDialect,
    pub route: String,
    pub wire_model: String,
}

/// Direct Responses contract for a model hosted by the managed llama.cpp server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TS)]
pub struct LlamaCppInferenceConfig {
    pub wire_api: WireApi,
    pub dialect: InferenceDialect,
    pub route: String,
    pub expected_model_basename: String,
    pub context_window: u32,
    pub max_input_tokens: u32,
    pub max_output_tokens: u32,
    pub safety_margin_tokens: u32,
}

/// Model-family specialization and the named provider route it requires.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, TS)]
#[serde(tag = "family", rename_all = "snake_case")]
pub enum ModelInferenceConfig {
    OpenAi {
        wire_api: WireApi,
        dialect: InferenceDialect,
        route: String,
        wire_model: String,
    },
    Anthropic {
        wire_api: WireApi,
        dialect: InferenceDialect,
        route: String,
        wire_model: String,
        max_output_tokens: u32,
        thinking: AnthropicThinkingPolicy,
        supports_disabled_thinking: bool,
    },
    Kimi(KimiInferenceConfig),
    Grok(GrokInferenceConfig),
    LlamaCpp(LlamaCppInferenceConfig),
}

impl ModelInferenceConfig {
    const OPEN_AI_ROUTE_CONTRACTS: &'static [(WireApi, InferenceDialect)] =
        &[(WireApi::Responses, InferenceDialect::OpenAi)];
    const ANTHROPIC_ROUTE_CONTRACTS: &'static [(WireApi, InferenceDialect)] =
        &[(WireApi::AnthropicMessages, InferenceDialect::ClaudeCode)];
    const KIMI_ROUTE_CONTRACTS: &'static [(WireApi, InferenceDialect)] = &[
        (WireApi::ChatCompletions, InferenceDialect::Kimi),
        (WireApi::AnthropicMessages, InferenceDialect::Kimi),
    ];
    const GROK_ROUTE_CONTRACTS: &'static [(WireApi, InferenceDialect)] =
        &[(WireApi::Responses, InferenceDialect::Grok)];
    const LLAMA_CPP_ROUTE_CONTRACTS: &'static [(WireApi, InferenceDialect)] =
        &[(WireApi::Responses, InferenceDialect::LlamaCpp)];

    pub fn family(&self) -> ModelFamily {
        match self {
            Self::OpenAi { .. } => ModelFamily::OpenAi,
            Self::Anthropic { .. } => ModelFamily::Anthropic,
            Self::Kimi(_) => ModelFamily::Kimi,
            Self::Grok(_) => ModelFamily::Grok,
            Self::LlamaCpp(_) => ModelFamily::LlamaCpp,
        }
    }

    pub fn route_contract(&self) -> (WireApi, InferenceDialect, &str) {
        match self {
            Self::OpenAi {
                wire_api,
                dialect,
                route,
                ..
            }
            | Self::Anthropic {
                wire_api,
                dialect,
                route,
                ..
            } => (*wire_api, *dialect, route),
            Self::Kimi(config) => (config.wire_api, config.dialect, &config.route),
            Self::Grok(config) => (config.wire_api, config.dialect, &config.route),
            Self::LlamaCpp(config) => (config.wire_api, config.dialect, &config.route),
        }
    }

    /// Wire and dialect combinations supported by this model family.
    pub fn supported_route_contracts(&self) -> &'static [(WireApi, InferenceDialect)] {
        match self {
            Self::OpenAi { .. } => Self::OPEN_AI_ROUTE_CONTRACTS,
            Self::Anthropic { .. } => Self::ANTHROPIC_ROUTE_CONTRACTS,
            Self::Kimi(_) => Self::KIMI_ROUTE_CONTRACTS,
            Self::Grok(_) => Self::GROK_ROUTE_CONTRACTS,
            Self::LlamaCpp(_) => Self::LLAMA_CPP_ROUTE_CONTRACTS,
        }
    }

    pub fn route_contract_is_supported(&self) -> bool {
        let (wire_api, dialect, _) = self.route_contract();
        self.supported_route_contracts()
            .contains(&(wire_api, dialect))
    }

    pub fn supports_responses_capabilities(&self) -> bool {
        matches!(
            self,
            Self::OpenAi {
                wire_api: WireApi::Responses,
                dialect: InferenceDialect::OpenAi,
                ..
            }
        )
    }
}

#[cfg(test)]
#[path = "model_inference_tests.rs"]
mod tests;
