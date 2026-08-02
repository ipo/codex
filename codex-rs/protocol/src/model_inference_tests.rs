use pretty_assertions::assert_eq;
use serde_json::json;

use super::*;

#[test]
fn inference_metadata_serializes_family_wire_dialect_and_route_independently() {
    let config = ModelInferenceConfig::Anthropic {
        wire_api: WireApi::AnthropicMessages,
        dialect: InferenceDialect::ClaudeCode,
        route: "claude_code".to_string(),
        wire_model: "claude-haiku-4-5-20251001".to_string(),
        max_output_tokens: 32_000,
        thinking: AnthropicThinkingPolicy::Budgeted {
            budget_tokens: 31_999,
        },
        supports_disabled_thinking: true,
    };

    assert_eq!(
        serde_json::to_value(config).expect("serialize inference config"),
        json!({
            "family": "anthropic",
            "wire_api": "anthropic_messages",
            "dialect": "claude_code",
            "route": "claude_code",
            "wire_model": "claude-haiku-4-5-20251001",
            "max_output_tokens": 32000,
            "thinking": {"type": "budgeted", "budget_tokens": 31999},
            "supports_disabled_thinking": true
        })
    );
}

#[test]
fn all_supported_wire_grammars_round_trip() {
    for wire_api in [
        WireApi::Responses,
        WireApi::AnthropicMessages,
        WireApi::ChatCompletions,
    ] {
        let json = serde_json::to_string(&wire_api).expect("serialize wire api");
        assert_eq!(
            serde_json::from_str::<WireApi>(&json).expect("deserialize wire api"),
            wire_api
        );
    }
}

#[test]
fn model_family_controls_supported_wire_and_dialect_contracts() {
    let open_ai = |wire_api, dialect| ModelInferenceConfig::OpenAi {
        wire_api,
        dialect,
        route: "route".to_string(),
        wire_model: "model".to_string(),
    };
    let anthropic = |wire_api, dialect| ModelInferenceConfig::Anthropic {
        wire_api,
        dialect,
        route: "route".to_string(),
        wire_model: "model".to_string(),
        max_output_tokens: 1,
        thinking: AnthropicThinkingPolicy::Adaptive,
        supports_disabled_thinking: true,
    };
    let kimi = |wire_api| {
        ModelInferenceConfig::Kimi(KimiInferenceConfig {
            wire_api,
            dialect: InferenceDialect::Kimi,
            route: "route".to_string(),
            wire_model: "model".to_string(),
            max_output_tokens: 1,
            thinking: KimiThinkingPolicy::Required,
        })
    };

    assert_eq!(
        [
            open_ai(WireApi::Responses, InferenceDialect::OpenAi).route_contract_is_supported(),
            open_ai(WireApi::ChatCompletions, InferenceDialect::OpenAi)
                .route_contract_is_supported(),
            anthropic(WireApi::AnthropicMessages, InferenceDialect::ClaudeCode,)
                .route_contract_is_supported(),
            anthropic(WireApi::Responses, InferenceDialect::OpenAi).route_contract_is_supported(),
            kimi(WireApi::ChatCompletions).route_contract_is_supported(),
            kimi(WireApi::AnthropicMessages).route_contract_is_supported(),
            kimi(WireApi::Responses).route_contract_is_supported(),
        ],
        [true, false, true, false, true, true, false]
    );
}
