use anyhow::Result;
use codex_model_provider_info::ModelProviderWireRoute;
use codex_model_provider_info::WireApi;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::model_inference::ModelInferenceConfig;
use core_test_support::responses;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn missing_native_route_fails_during_thread_start_without_sampling() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = responses::start_mock_server().await;
    let result = test_codex()
        .with_model_info_override("gpt-5.5", |model| {
            model.inference = Some(ModelInferenceConfig::Kimi {
                wire_api: WireApi::ChatCompletions,
                dialect: InferenceDialect::Kimi,
                route: "kimi_code".to_string(),
                wire_model: "k3".to_string(),
            });
        })
        .build_with_auto_env(&server)
        .await;
    let error = match result {
        Ok(_) => panic!("thread start should reject a missing native route"),
        Err(error) => error,
    };

    assert_eq!(
        error.to_string(),
        "model `gpt-5.5` (kimi) requires provider wire route `kimi_code`; configure `model_providers.<provider>.wire_routes.kimi_code` with `wire_api = \"chat_completions\"` and `dialect = \"kimi\"`"
    );
    assert!(
        server
            .received_requests()
            .await
            .expect("response requests")
            .is_empty()
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn family_incompatible_matching_route_fails_before_sampling() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = responses::start_mock_server().await;
    let result = test_codex()
        .with_config(|config| {
            config.model_provider.wire_routes.insert(
                "kimi_code".to_string(),
                ModelProviderWireRoute {
                    wire_api: WireApi::Responses,
                    dialect: InferenceDialect::OpenAi,
                    base_url: "http://127.0.0.1:8080/v1/kimi".to_string(),
                    request_path: "responses".to_string(),
                    query_params: None,
                    request_max_retries: None,
                    stream_max_retries: None,
                    stream_idle_timeout_ms: None,
                },
            );
        })
        .with_model_info_override("gpt-5.5", |model| {
            model.inference = Some(ModelInferenceConfig::Kimi {
                wire_api: WireApi::Responses,
                dialect: InferenceDialect::OpenAi,
                route: "kimi_code".to_string(),
                wire_model: "k3".to_string(),
            });
        })
        .build_with_auto_env(&server)
        .await;
    let error = match result {
        Ok(_) => panic!("thread start should reject a family-incompatible route"),
        Err(error) => error,
    };

    assert_eq!(
        error.to_string(),
        "model `gpt-5.5` declares an incompatible inference contract for family `kimi`: `wire_api = \"responses\"` with `dialect = \"open_ai\"`; supported: `wire_api = \"chat_completions\"` with `dialect = \"kimi\"`"
    );
    assert!(
        server
            .received_requests()
            .await
            .expect("response requests")
            .is_empty()
    );
    Ok(())
}
