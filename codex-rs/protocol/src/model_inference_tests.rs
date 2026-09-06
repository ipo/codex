use pretty_assertions::assert_eq;
use serde_json::json;

use super::*;

#[test]
fn typed_inference_families_round_trip_independent_wire_and_dialect_metadata() {
    let configs = [
        ModelInferenceConfig::OpenAi {
            wire_api: WireApi::Responses,
            dialect: InferenceDialect::OpenAi,
            route: "openai".to_string(),
            wire_model: "gpt-5.6-sol".to_string(),
        },
        ModelInferenceConfig::Anthropic {
            wire_api: WireApi::AnthropicMessages,
            dialect: InferenceDialect::ClaudeCode,
            route: "claude_code".to_string(),
            wire_model: "claude-opus-5".to_string(),
            max_output_tokens: 64_000,
            thinking: AnthropicThinkingPolicy::Adaptive,
            supports_disabled_thinking: true,
        },
        ModelInferenceConfig::Kimi(KimiInferenceConfig {
            wire_api: WireApi::ChatCompletions,
            dialect: InferenceDialect::Kimi,
            route: "kimi_code".to_string(),
            wire_model: "k3".to_string(),
            max_output_tokens: 131_072,
            thinking: KimiThinkingPolicy::RequiredWithEffort,
        }),
        ModelInferenceConfig::Grok(GrokInferenceConfig {
            wire_api: WireApi::Responses,
            dialect: InferenceDialect::Grok,
            route: "grok".to_string(),
            wire_model: "grok-4.6".to_string(),
        }),
        ModelInferenceConfig::LlamaCpp(LlamaCppInferenceConfig {
            wire_api: WireApi::Responses,
            dialect: InferenceDialect::LlamaCpp,
            route: "llama_cpp".to_string(),
            expected_model_basename: "Qwen3.8-27B-Uncensored-Q4_K_M.gguf".to_string(),
            context_window: 131_072,
            max_input_tokens: 121_856,
            max_output_tokens: 8_192,
            safety_margin_tokens: 1_024,
        }),
    ];

    for config in configs {
        let value = serde_json::to_value(&config).expect("serialize inference config");
        assert_eq!(
            serde_json::from_value::<ModelInferenceConfig>(value)
                .expect("deserialize inference config"),
            config
        );
    }
}

#[test]
fn wire_grammars_are_not_dialects() {
    assert_eq!(
        serde_json::to_value([
            WireApi::Responses,
            WireApi::AnthropicMessages,
            WireApi::ChatCompletions,
        ])
        .expect("serialize wire APIs"),
        json!(["responses", "anthropic_messages", "chat_completions"])
    );
    assert!(serde_json::from_str::<WireApi>("\"claude_code\"").is_err());
    assert!(serde_json::from_str::<WireApi>("\"kimi\"").is_err());
}

#[test]
fn model_family_authoritatively_controls_route_compatibility_and_capabilities() {
    let contracts = [
        (
            ModelInferenceConfig::OpenAi {
                wire_api: WireApi::Responses,
                dialect: InferenceDialect::OpenAi,
                route: "route".to_string(),
                wire_model: "model".to_string(),
            },
            true,
            true,
        ),
        (
            ModelInferenceConfig::Anthropic {
                wire_api: WireApi::Responses,
                dialect: InferenceDialect::OpenAi,
                route: "route".to_string(),
                wire_model: "model".to_string(),
                max_output_tokens: 1,
                thinking: AnthropicThinkingPolicy::Adaptive,
                supports_disabled_thinking: true,
            },
            false,
            false,
        ),
        (
            ModelInferenceConfig::Kimi(KimiInferenceConfig {
                wire_api: WireApi::AnthropicMessages,
                dialect: InferenceDialect::Kimi,
                route: "route".to_string(),
                wire_model: "model".to_string(),
                max_output_tokens: 1,
                thinking: KimiThinkingPolicy::Required,
            }),
            true,
            false,
        ),
        (
            ModelInferenceConfig::Grok(GrokInferenceConfig {
                wire_api: WireApi::Responses,
                dialect: InferenceDialect::Grok,
                route: "route".to_string(),
                wire_model: "model".to_string(),
            }),
            true,
            false,
        ),
        (
            ModelInferenceConfig::LlamaCpp(LlamaCppInferenceConfig {
                wire_api: WireApi::Responses,
                dialect: InferenceDialect::OpenAi,
                route: "route".to_string(),
                expected_model_basename: "model.gguf".to_string(),
                context_window: 4,
                max_input_tokens: 2,
                max_output_tokens: 1,
                safety_margin_tokens: 1,
            }),
            false,
            false,
        ),
    ];

    assert_eq!(
        contracts.map(|(config, supported, responses_capable)| (
            config.route_contract_is_supported(),
            config.supports_responses_capabilities(),
            supported,
            responses_capable,
        )),
        [
            (true, true, true, true),
            (false, false, false, false),
            (true, false, true, false),
            (true, false, true, false),
            (false, false, false, false),
        ]
    );
}
