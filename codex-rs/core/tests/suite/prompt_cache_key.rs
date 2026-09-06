use std::time::Duration;

use anyhow::Result;
use anyhow::anyhow;
use codex_core::ForkSnapshot;
use codex_core::TurnInputRequest;
use codex_features::Feature;
use codex_login::CodexAuth;
use codex_model_provider_info::ModelProviderWireRoute;
use codex_model_provider_info::WireApi;
use codex_protocol::model_inference::AnthropicThinkingPolicy;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::model_inference::KimiInferenceConfig;
use codex_protocol::model_inference::KimiThinkingPolicy;
use codex_protocol::model_inference::ModelInferenceConfig;
use codex_protocol::protocol::EventMsg;
use codex_protocol::user_input::UserInput;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_function_call_with_namespace;
use core_test_support::responses::ev_response_created;
use core_test_support::responses::mount_sse_once_match;
use core_test_support::responses::mount_sse_sequence;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::test_codex::TestCodexBuilder;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;

const ROOT_PROMPT: &str = "delegate the cache audit";
const CHILD_TASK: &str = "inspect the repository";
const SPAWN_CALL_ID: &str = "spawn-worker";
const COLLABORATION_NAMESPACE: &str = "collaboration";

fn body_contains(request: &wiremock::Request, text: &str) -> bool {
    serde_json::from_slice::<Value>(&request.body).is_ok_and(|body| body.to_string().contains(text))
}

fn request_has_input_type(request: &wiremock::Request, input_type: &str) -> bool {
    serde_json::from_slice::<Value>(&request.body)
        .ok()
        .and_then(|body| body.get("input").and_then(Value::as_array).cloned())
        .is_some_and(|items| {
            items
                .iter()
                .any(|item| item.get("type").and_then(Value::as_str) == Some(input_type))
        })
}

fn native_builder(inference: ModelInferenceConfig, route_base_url: String) -> TestCodexBuilder {
    let (wire_api, dialect, route_name) = inference.route_contract();
    let route_name = route_name.to_string();
    test_codex()
        .with_auth(CodexAuth::from_api_key("dummy"))
        .with_config(move |config| {
            config.model_provider.wire_routes.insert(
                route_name,
                ModelProviderWireRoute {
                    wire_api,
                    dialect,
                    base_url: route_base_url,
                    request_path: match wire_api {
                        WireApi::Responses => "responses",
                        WireApi::AnthropicMessages => "messages",
                        WireApi::ChatCompletions => "chat/completions",
                    }
                    .to_string(),
                    query_params: None,
                    request_max_retries: None,
                    stream_max_retries: None,
                    stream_idle_timeout_ms: None,
                },
            );
        })
        .with_model_info_override("gpt-5.5", move |model| {
            model.inference = Some(inference);
        })
}

async fn assert_native_family_public_fork_identity(inference: ModelInferenceConfig) -> Result<()> {
    let server = start_mock_server().await;
    let route_base_url = format!("{}/native", server.uri());
    let mut root_builder = native_builder(inference.clone(), route_base_url.clone());
    let root = root_builder.build_with_auto_env(&server).await?;
    root.codex.ensure_rollout_materialized().await;
    root.codex.flush_rollout().await?;
    let root_session_id = root.session_configured.session_id;
    let root_thread_id = root.session_configured.thread_id;
    let rollout_path = root.codex.rollout_path().expect("root rollout path");
    let home = root.home.clone();
    root.codex.shutdown_and_wait().await?;

    let mut resume_builder = native_builder(inference.clone(), route_base_url.clone());
    let resumed = resume_builder
        .resume(&server, home, rollout_path.clone())
        .await?;
    assert_eq!(
        (
            resumed.session_configured.thread_id,
            resumed.session_configured.session_id,
        ),
        (root_thread_id, root_session_id)
    );

    let branch = resumed
        .thread_manager
        .fork_thread(
            ForkSnapshot::Interrupted,
            resumed.config.clone(),
            rollout_path,
            /*thread_source*/ None,
            /*parent_trace*/ None,
        )
        .await?;
    assert_ne!(branch.thread_id, root_thread_id);
    assert_eq!(branch.session_configured.session_id, root_session_id);

    let mut distinct_builder = native_builder(inference, route_base_url);
    let distinct = distinct_builder.build_with_auto_env(&server).await?;
    assert_ne!(distinct.session_configured.session_id, root_session_id);
    assert!(
        server
            .received_requests()
            .await
            .expect("native fixture requests")
            .is_empty(),
        "session construction should not activate a native adapter"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_claude_and_kimi_public_forks_preserve_session_identity() -> Result<()> {
    assert_native_family_public_fork_identity(ModelInferenceConfig::Anthropic {
        wire_api: WireApi::AnthropicMessages,
        dialect: InferenceDialect::ClaudeCode,
        route: "claude_code".to_string(),
        wire_model: "claude-haiku-4-5-20251001".to_string(),
        max_output_tokens: 32_000,
        thinking: AnthropicThinkingPolicy::Budgeted {
            budget_tokens: 31_999,
        },
        supports_disabled_thinking: false,
    })
    .await?;
    assert_native_family_public_fork_identity(ModelInferenceConfig::Kimi(KimiInferenceConfig {
        wire_api: WireApi::ChatCompletions,
        dialect: InferenceDialect::Kimi,
        route: "kimi_code".to_string(),
        wire_model: "kimi-for-coding".to_string(),
        max_output_tokens: 32_768,
        thinking: KimiThinkingPolicy::Required,
    }))
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn public_fork_preserves_session_headers_and_prompt_cache_key() -> Result<()> {
    let server = start_mock_server().await;
    let response_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![ev_response_created("root"), ev_completed("root")]),
            sse(vec![ev_response_created("branch"), ev_completed("branch")]),
            sse(vec![
                ev_response_created("distinct"),
                ev_completed("distinct"),
            ]),
        ],
    )
    .await;

    let mut builder = test_codex().with_auth(CodexAuth::from_api_key("dummy"));
    let root = builder.build_with_auto_env(&server).await?;
    root.submit_turn("root turn").await?;
    let root_session_id = root.session_configured.session_id;
    let root_thread_id = root.session_configured.thread_id;
    let rollout_path = root.codex.rollout_path().expect("root rollout path");
    let home = root.home.clone();
    root.codex.shutdown_and_wait().await?;

    let resumed = builder.resume(&server, home, rollout_path.clone()).await?;
    assert_eq!(resumed.session_configured.thread_id, root_thread_id);
    assert_eq!(resumed.session_configured.session_id, root_session_id);

    let branch = resumed
        .thread_manager
        .fork_thread(
            ForkSnapshot::Interrupted,
            resumed.config.clone(),
            rollout_path,
            /*thread_source*/ None,
            /*parent_trace*/ None,
        )
        .await?;
    assert_ne!(branch.thread_id, root_thread_id);
    assert_eq!(branch.session_configured.session_id, root_session_id);
    branch
        .thread
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "branch turn".to_string(),
            text_elements: Vec::new(),
        }]))
        .await?;
    wait_for_event(&branch.thread, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;

    let distinct = builder.build_with_auto_env(&server).await?;
    assert_ne!(distinct.session_configured.session_id, root_session_id);
    distinct.submit_turn("distinct root turn").await?;

    let requests = response_mock.requests();
    assert_eq!(requests.len(), 3);
    assert_eq!(
        json!({
            "root": {
                "sessionId": requests[0].header("session-id"),
                "threadId": requests[0].header("thread-id"),
                "clientRequestId": requests[0].header("x-client-request-id"),
                "promptCacheKey": requests[0].body_json()["prompt_cache_key"].clone(),
            },
            "branch": {
                "sessionId": requests[1].header("session-id"),
                "threadId": requests[1].header("thread-id"),
                "clientRequestId": requests[1].header("x-client-request-id"),
                "promptCacheKey": requests[1].body_json()["prompt_cache_key"].clone(),
            },
            "distinct": {
                "sessionId": requests[2].header("session-id"),
                "threadId": requests[2].header("thread-id"),
                "clientRequestId": requests[2].header("x-client-request-id"),
                "promptCacheKey": requests[2].body_json()["prompt_cache_key"].clone(),
            },
        }),
        json!({
            "root": {
                "sessionId": root_session_id,
                "threadId": root_thread_id,
                "clientRequestId": root_thread_id,
                "promptCacheKey": root_session_id,
            },
            "branch": {
                "sessionId": root_session_id,
                "threadId": branch.thread_id,
                "clientRequestId": branch.thread_id,
                "promptCacheKey": root_session_id,
            },
            "distinct": {
                "sessionId": distinct.session_configured.session_id,
                "threadId": distinct.session_configured.thread_id,
                "clientRequestId": distinct.session_configured.thread_id,
                "promptCacheKey": distinct.session_configured.session_id,
            },
        })
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn api_key_subagent_uses_session_id_as_prompt_cache_key() -> Result<()> {
    let server = start_mock_server().await;
    let spawn_args = serde_json::to_string(&json!({
        "message": CHILD_TASK,
        "task_name": "worker",
    }))?;
    let root_request = mount_sse_once_match(
        &server,
        |request: &wiremock::Request| {
            body_contains(request, ROOT_PROMPT)
                && !request_has_input_type(request, "agent_message")
                && !body_contains(request, SPAWN_CALL_ID)
        },
        sse(vec![
            ev_response_created("root-response-1"),
            ev_function_call_with_namespace(
                SPAWN_CALL_ID,
                COLLABORATION_NAMESPACE,
                "spawn_agent",
                &spawn_args,
            ),
            ev_completed("root-response-1"),
        ]),
    )
    .await;
    let child_request = mount_sse_once_match(
        &server,
        |request: &wiremock::Request| {
            body_contains(request, CHILD_TASK) && !body_contains(request, SPAWN_CALL_ID)
        },
        sse(vec![
            ev_response_created("child-response"),
            ev_assistant_message("child-message", "inspection complete"),
            ev_completed("child-response"),
        ]),
    )
    .await;
    mount_sse_once_match(
        &server,
        |request: &wiremock::Request| body_contains(request, SPAWN_CALL_ID),
        sse(vec![
            ev_response_created("root-response-2"),
            ev_assistant_message("root-message", "worker finished"),
            ev_completed("root-response-2"),
        ]),
    )
    .await;

    let mut builder = test_codex()
        .with_auth(CodexAuth::from_api_key("dummy"))
        .with_config(|config| {
            config
                .features
                .enable(Feature::Collab)
                .expect("test config should allow feature update");
            config
                .features
                .enable(Feature::MultiAgentV2)
                .expect("test config should allow feature update");
        });
    let test = builder.build(&server).await?;
    let expected_session_id = test.session_configured.session_id.to_string();
    test.submit_turn(ROOT_PROMPT).await?;

    let root_request = root_request
        .requests()
        .into_iter()
        .next()
        .expect("root request");
    let root_thread_id = root_request.header("thread-id").expect("root thread ID");
    let child_request = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if let Some(request) = child_request.requests().into_iter().find(|request| {
                request
                    .header("thread-id")
                    .is_some_and(|thread_id| thread_id != root_thread_id)
            }) {
                break request;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .map_err(|_| anyhow!("timed out waiting for the child request"))?;
    let child_thread_id = child_request.header("thread-id").expect("child thread ID");

    assert_eq!(
        json!({
            "differentThreadIds": root_thread_id != child_thread_id,
            "root": {
                "sessionId": root_request.header("session-id"),
                "threadId": &root_thread_id,
                "clientRequestId": root_request.header("x-client-request-id"),
                "promptCacheKey": root_request.body_json()["prompt_cache_key"].clone(),
            },
            "child": {
                "sessionId": child_request.header("session-id"),
                "threadId": &child_thread_id,
                "clientRequestId": child_request.header("x-client-request-id"),
                "promptCacheKey": child_request.body_json()["prompt_cache_key"].clone(),
            },
        }),
        json!({
            "differentThreadIds": true,
            "root": {
                "sessionId": &expected_session_id,
                "threadId": &root_thread_id,
                "clientRequestId": &root_thread_id,
                "promptCacheKey": &expected_session_id,
            },
            "child": {
                "sessionId": &expected_session_id,
                "threadId": &child_thread_id,
                "clientRequestId": &child_thread_id,
                "promptCacheKey": &expected_session_id,
            },
        })
    );

    Ok(())
}
