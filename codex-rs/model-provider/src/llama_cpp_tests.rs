use codex_model_provider_info::ModelProviderInfo;
use codex_model_provider_info::ResolvedInferencePlan;
use codex_model_provider_info::ResolvedWireRoute;
use codex_protocol::openai_models::ModelVisibility;
use pretty_assertions::assert_eq;

use super::*;

#[test]
fn discovered_entry_activates_exact_llama_cpp_plan_and_picker_metadata() {
    let base_models = codex_models_manager::bundled_models_response()
        .expect("bundled models")
        .models;
    let entry = LlamaCppCatalogEntry {
        canonical_id: r"local/F:\models\Qwen3.8-IQ2_M.gguf".to_string(),
        wire_model: r"F:\models\Qwen3.8-IQ2_M.gguf".to_string(),
        display_name: "Qwen3.8-IQ2_M.gguf".to_string(),
        aliases: vec!["Qwen3.8-IQ2_M".to_string(), "llama-cpp-local".to_string()],
        context_window: 131_072,
        max_input_tokens: 121_856,
        max_output_tokens: 8_192,
        safety_margin_tokens: 1_024,
    };
    let catalog = LlamaCppCatalog {
        models: vec![entry.clone()],
    };
    let [model]: [ModelInfo; 1] = local_model_infos(&base_models, &catalog)
        .expect("local metadata")
        .try_into()
        .expect("one local model");

    assert_eq!(
        (
            model.slug.as_str(),
            model.aliases.as_slice(),
            model.display_name.as_str(),
            model.visibility,
            model.default_reasoning_level.clone(),
            model
                .supported_reasoning_levels
                .iter()
                .map(|preset| preset.effort.clone())
                .collect::<Vec<_>>(),
            model.context_window,
            model.max_context_window,
            model.auto_compact_token_limit,
            model.effective_context_window_percent,
            model.input_modalities.as_slice(),
            model.tool_mode,
            model.multi_agent_version,
        ),
        (
            entry.canonical_id.as_str(),
            entry.aliases.as_slice(),
            entry.display_name.as_str(),
            ModelVisibility::List,
            Some(ReasoningEffort::Low),
            vec![
                ReasoningEffort::None,
                ReasoningEffort::Low,
                ReasoningEffort::Medium,
                ReasoningEffort::XHigh,
            ],
            Some(131_072),
            Some(131_072),
            Some(121_856),
            100,
            [InputModality::Text].as_slice(),
            Some(ToolMode::Direct),
            Some(MultiAgentVersion::V2),
        )
    );
    assert_eq!(
        (
            model.history_compatibility_group.as_deref(),
            model.requires_nonempty_assistant_messages,
            model.disabled_tools.as_slice(),
        ),
        (
            Some("local_llama_cpp_qwen3_8"),
            false,
            [
                ModelToolCapability::ToolSearch,
                ModelToolCapability::WebSearch,
                ModelToolCapability::ImageGeneration,
                ModelToolCapability::CodexApps,
            ]
            .as_slice(),
        )
    );

    let mut provider = ModelProviderInfo::default();
    provider.install_llama_cpp_route();
    assert_eq!(
        provider
            .resolve_inference_plan(&model)
            .expect("resolved llama.cpp plan"),
        ResolvedInferencePlan::LlamaCpp {
            config: LlamaCppInferenceConfig {
                wire_api: WireApi::Responses,
                dialect: InferenceDialect::LlamaCpp,
                route: LLAMA_CPP_ROUTE_NAME.to_string(),
                expected_model_basename: entry.display_name,
                context_window: 131_072,
                max_input_tokens: 121_856,
                max_output_tokens: 8_192,
                safety_margin_tokens: 1_024,
            },
            route: ResolvedWireRoute {
                name: Some(LLAMA_CPP_ROUTE_NAME.to_string()),
                wire_api: WireApi::Responses,
                dialect: InferenceDialect::LlamaCpp,
                base_url: Some(format!("{LLAMA_CPP_LOCAL_ENDPOINT}/v1")),
                request_path: "responses".to_string(),
                query_params: None,
                request_max_retries: 0,
                stream_max_retries: 5,
                stream_idle_timeout: Duration::from_secs(300),
            },
        }
    );
}
