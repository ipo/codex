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
        )
    );
    assert_eq!(
        (
            model.effective_context_window_percent,
            model.input_modalities.as_slice(),
            model.tool_mode,
            model.multi_agent_version,
        ),
        (
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

fn test_manager(endpoint: &str) -> LlamaCppModelsManager {
    let catalog = codex_models_manager::bundled_models_response().expect("bundled models");
    LlamaCppModelsManager::new_with_endpoint(
        Arc::new(codex_models_manager::manager::StaticModelsManager::new(
            /*auth_manager*/ None, catalog,
        )),
        endpoint.to_string(),
    )
}

fn test_http_factory() -> HttpClientFactory {
    HttpClientFactory::new(codex_http_client::OutboundProxyPolicy::ReqwestDefault)
}

async fn wait_for_discovered_slug(manager: &LlamaCppModelsManager, slug: &str) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let models = manager.get_remote_models().await;
        if models
            .iter()
            .any(|model| model.slug == slug || model.aliases.iter().any(|alias| alias == slug))
        {
            return;
        }
        if tokio::time::Instant::now() >= deadline {
            panic!("timed out waiting for discovered model {slug}");
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn hanging_health_does_not_delay_catalog_listing() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/health"))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(30))
                .set_body_json(serde_json::json!({"status": "ok"})),
        )
        .mount(&server)
        .await;
    let manager = test_manager(&server.uri());
    let started = std::time::Instant::now();
    let catalog = manager
        .raw_model_catalog(RefreshStrategy::OnlineIfUncached, test_http_factory())
        .await;
    assert!(started.elapsed() < Duration::from_millis(750));
    assert!(
        !catalog
            .models
            .iter()
            .any(|model| matches!(model.inference, Some(ModelInferenceConfig::LlamaCpp(_))))
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_list_models_share_one_inflight_probe() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/health"))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(30))
                .set_body_json(serde_json::json!({"status": "ok"})),
        )
        .mount(&server)
        .await;
    let manager = test_manager(&server.uri());
    let factory = test_http_factory();
    let first = manager.list_models(RefreshStrategy::Online, factory.clone());
    let second = manager.list_models(RefreshStrategy::Online, factory);
    let _ = tokio::join!(first, second);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let health_hits = loop {
        let health_hits = server
            .received_requests()
            .await
            .expect("recorded requests")
            .iter()
            .filter(|request| request.url.path() == "/health")
            .count();
        if health_hits >= 1 || tokio::time::Instant::now() >= deadline {
            break health_hits;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert_eq!(health_hits, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn successful_background_discovery_is_visible_on_later_list() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/health"))
        .respond_with(
            wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "status": "ok"
            })),
        )
        .mount(&server)
        .await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/v1/models"))
        .respond_with(
            wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{
                    "id": "Qwen3.8-test.gguf",
                    "meta": {"n_ctx": 32768}
                }]
            })),
        )
        .mount(&server)
        .await;
    let manager = test_manager(&server.uri());
    let _ = manager
        .list_models(RefreshStrategy::OnlineIfUncached, test_http_factory())
        .await;
    wait_for_discovered_slug(&manager, "local/Qwen3.8-test.gguf").await;
    let models = manager.get_remote_models().await;
    let discovered = models
        .iter()
        .find(|model| model.slug == "local/Qwen3.8-test.gguf")
        .expect("discovered local model");
    assert!(matches!(
        discovered.inference,
        Some(ModelInferenceConfig::LlamaCpp(_))
    ));
    assert!(
        discovered
            .aliases
            .iter()
            .any(|alias| alias == "llama-cpp-local")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn get_model_info_synthesizes_llama_cpp_before_discovery_completes() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/health"))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(30))
                .set_body_json(serde_json::json!({"status": "ok"})),
        )
        .mount(&server)
        .await;
    let manager = test_manager(&server.uri());
    let config = ModelsManagerConfig::default();
    for requested in ["llama-cpp-local", r"local/F:\models\Qwen3.8-test.gguf"] {
        let info = manager.get_model_info(requested, &config).await;
        assert!(
            matches!(info.inference, Some(ModelInferenceConfig::LlamaCpp(_))),
            "{requested} should synthesize llama.cpp inference"
        );
    }
}
