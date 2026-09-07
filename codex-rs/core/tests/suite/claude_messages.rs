use std::sync::Arc;

use anyhow::Result;
use codex_core::TurnInputRequest;
use codex_features::Feature;
use codex_model_provider_info::CLAUDEFLARE_PROVIDER_ID;
use codex_model_provider_info::built_in_model_providers;
use codex_protocol::openai_models::ConfigShellToolType;
use codex_protocol::openai_models::ModelToolCapability;
use codex_protocol::protocol::EventMsg;
use codex_protocol::user_input::UserInput;
use core_test_support::responses;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use serde_json::json;
use wiremock::ResponseTemplate;

const HAIKU: &str = "anthropic/claude-haiku-4-5-20251001";

fn claude_builder(
    server: &wiremock::MockServer,
) -> core_test_support::test_codex::TestCodexBuilder {
    let base_url = format!("{}/v1/claude-code", server.uri());
    test_codex()
        .with_config(move |config| {
            let mut provider = built_in_model_providers(/*openai_base_url*/ None)
                .remove(CLAUDEFLARE_PROVIDER_ID)
                .expect("managed Claudeflare provider");
            provider.stream_max_retries = Some(10);
            provider
                .wire_routes
                .get_mut("claude_code")
                .expect("Claude route")
                .base_url = base_url;
            provider
                .wire_routes
                .get_mut("claude_code")
                .expect("Claude route")
                .stream_max_retries = Some(10);
            config.model_provider = provider;
            config.model = Some(HAIKU.to_string());
            config.base_instructions = Some("Claude conformance system".to_string());
            config.compact_prompt = Some("Summarize".to_string());
            config.agents_enabled = false;
            config.update_plan_enabled = true;
            config.experimental_request_user_input_enabled = false;
            config.include_skill_instructions = false;
            config.include_permissions_instructions = false;
            config.include_apps_instructions = false;
            config.include_collaboration_mode_instructions = false;
            config.include_environment_context = false;
            config
                .features
                .disable(Feature::ViewImage)
                .expect("test config should disable image tools");
        })
        .with_model_info_override(HAIKU, |model| {
            model.shell_type = ConfigShellToolType::Disabled;
            model.apply_patch_tool_type = None;
            model.experimental_supported_tools.clear();
            model.disabled_tools = vec![
                ModelToolCapability::ApplyPatch,
                ModelToolCapability::ToolSearch,
                ModelToolCapability::WebSearch,
                ModelToolCapability::ImageGeneration,
                ModelToolCapability::CodexApps,
            ];
            model.supports_search_tool = false;
        })
}

fn claude_event(name: &str, data: serde_json::Value) -> String {
    format!("event: {name}\ndata: {data}\n\n")
}

fn claude_text_stream(text: &str, stop_reason: &str) -> String {
    [
        claude_event(
            "message_start",
            json!({"type":"message_start","message":{"id":"msg-1","type":"message","role":"assistant","model":"claude-haiku-4-5-20251001","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}),
        ),
        claude_event(
            "content_block_start",
            json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
        ),
        claude_event(
            "content_block_delta",
            json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":text}}),
        ),
        claude_event(
            "content_block_stop",
            json!({"type":"content_block_stop","index":0}),
        ),
        claude_event(
            "message_delta",
            json!({"type":"message_delta","delta":{"stop_reason":stop_reason},"usage":{"output_tokens":4}}),
        ),
        claude_event("message_stop", json!({"type":"message_stop"})),
    ]
    .concat()
}

fn claude_tool_stream(call_id: &str, name: &str, arguments: &str) -> String {
    [
        claude_event(
            "message_start",
            json!({"type":"message_start","message":{"id":"msg-tool","type":"message","role":"assistant","model":"claude-haiku-4-5-20251001","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}),
        ),
        claude_event(
            "content_block_start",
            json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":call_id,"name":name,"input":{}}}),
        ),
        claude_event(
            "content_block_delta",
            json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":arguments}}),
        ),
        claude_event(
            "content_block_stop",
            json!({"type":"content_block_stop","index":0}),
        ),
        claude_event(
            "message_delta",
            json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":8}}),
        ),
        claude_event("message_stop", json!({"type":"message_stop"})),
    ]
    .concat()
}

fn user_text(body: &serde_json::Value) -> Vec<String> {
    body["messages"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|message| message["role"] == "user")
        .flat_map(|message| message["content"].as_array().into_iter().flatten())
        .filter(|block| block["type"] == "text")
        .filter_map(|block| block["text"].as_str().map(str::to_string))
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn claude_root_request_tool_continuation_and_resume() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = responses::start_mock_server().await;
    let plan_args = json!({
        "explanation": "Claude tool replay",
        "plan": [{"step":"Finish","status":"completed"}],
    })
    .to_string();
    let mock = responses::mount_messages_sse_sequence(
        &server,
        vec![
            claude_tool_stream("call-plan", "update_plan", &plan_args),
            claude_text_stream("Claude completed", "end_turn"),
            claude_text_stream("resumed Claude turn", "end_turn"),
        ],
    )
    .await;
    let mut builder = claude_builder(&server);
    let test = builder.build_with_auto_env(&server).await?;

    test.submit_turn("Use the plan tool once").await?;

    let requests = mock.requests();
    assert_eq!(requests.len(), 2);
    let request = &requests[0];
    assert_eq!(request.path(), "/v1/claude-code/v1/messages");
    assert_eq!(request.query_param("beta").as_deref(), Some("true"));
    assert_eq!(
        request.header("anthropic-version").as_deref(),
        Some("2023-06-01")
    );
    assert!(request.header("x-claude-code-session-id").is_some());
    let body = request.body_json();
    assert_eq!(body["model"], "claude-haiku-4-5-20251001");
    assert_eq!(body["max_tokens"], 32_000);
    assert_eq!(body["stream"], true);
    assert!(
        user_text(&body)
            .iter()
            .any(|text| text.contains("Use the plan tool once")),
        "root request should include the user turn: {body}"
    );
    assert!(
        body["tools"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|tool| tool["name"] == "update_plan"),
        "root request should advertise update_plan: {body}"
    );

    let continuation = requests[1].body_json();
    let continuation_json = continuation.to_string();
    assert!(
        continuation_json.contains("call-plan") && continuation_json.contains("tool_result"),
        "tool continuation should replay the tool use and result: {continuation}"
    );
    assert_eq!(
        requests[1].header("x-claude-code-session-id"),
        request.header("x-claude-code-session-id")
    );

    let home = test.home.clone();
    let rollout_path = test.codex.rollout_path().expect("root rollout path");
    let session_header = request.header("x-claude-code-session-id");
    test.codex.shutdown_and_wait().await?;
    let resumed = builder.resume(&server, home, rollout_path).await?;
    resumed.submit_turn("resume Claude").await?;
    let resumed_request = mock.requests().last().expect("resumed request").clone();
    assert_eq!(mock.requests().len(), 3);
    assert_eq!(
        resumed_request.header("x-claude-code-session-id"),
        session_header
    );
    assert!(
        user_text(&resumed_request.body_json())
            .iter()
            .any(|text| text.contains("resume Claude")),
        "resume should send the new user turn: {}",
        resumed_request.body_json()
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn claude_retry_honors_retry_after_ms() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = responses::start_mock_server().await;
    let mock = responses::mount_messages_response_sequence(
        &server,
        vec![
            ResponseTemplate::new(429)
                .insert_header("retry-after-ms", "25")
                .set_body_string("rate limited"),
            responses::sse_response(claude_text_stream("recovered after retry", "end_turn")),
        ],
    )
    .await;
    let test = claude_builder(&server).build_with_auto_env(&server).await?;
    test.submit_turn("Retry one transient Claude error").await?;
    assert_eq!(mock.requests().len(), 2);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn claude_overflow_compacts_and_retries() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = responses::start_mock_server().await;
    let mock = responses::mount_messages_response_sequence(
        &server,
        vec![
            ResponseTemplate::new(400).set_body_string(
                r#"{"error":{"type":"invalid_request_error","message":"prompt is too long"}}"#,
            ),
            responses::sse_response(claude_text_stream("compacted history", "end_turn")),
            responses::sse_response(claude_text_stream("recovered after compact", "end_turn")),
        ],
    )
    .await;
    let test = claude_builder(&server).build_with_auto_env(&server).await?;
    test.submit_turn("Trigger native Claude overflow recovery")
        .await?;
    let requests = mock.requests();
    assert_eq!(requests.len(), 3);
    assert!(
        user_text(&requests[1].body_json())
            .iter()
            .any(|text| text.contains("Summarize")),
        "overflow recovery should compact before retrying: {}",
        requests[1].body_json()
    );
    assert!(
        user_text(&requests[2].body_json())
            .iter()
            .any(|text| text.contains("Trigger native Claude overflow recovery")),
        "overflow recovery should retry the original turn: {}",
        requests[2].body_json()
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn claude_max_tokens_is_nonretryable() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = responses::start_mock_server().await;
    let mock = responses::mount_messages_sse_once(
        &server,
        claude_text_stream("partial output", "max_tokens"),
    )
    .await;
    let test = claude_builder(&server).build_with_auto_env(&server).await?;
    test.codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "Write until the output budget is exhausted".to_string(),
            text_elements: Vec::new(),
        }]))
        .await?;

    let error_event =
        wait_for_event(&test.codex, |event| matches!(event, EventMsg::Error(_))).await;
    let EventMsg::Error(error) = error_event else {
        unreachable!("predicate guarantees an error event");
    };
    assert_eq!(
        error.message,
        "model output limit reached before the turn completed"
    );
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;

    assert_eq!(mock.requests().len(), 1);
    Ok(())
}
