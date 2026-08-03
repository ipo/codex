use std::time::Duration;

use anyhow::Result;
use codex_features::Feature;
use codex_model_provider_info::CLAUDEFLARE_PROVIDER_ID;
use codex_model_provider_info::built_in_model_providers;
use codex_models_manager::bundled_models_response;
use codex_protocol::protocol::EventMsg;
use core_test_support::responses;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use tokio::time::timeout;
use wiremock::Mock;
use wiremock::Request;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;

const PARENT_PROMPT: &str = "spawn the requested Kimi child";
const CHILD_PROMPT: &str = "complete the Kimi child canary";
const CHILD_REPLY: &str = "Kimi child complete";
const CHILD_SELECTOR: &str = "kimi-2.7";
const CHILD_TASK_NAME: &str = "kimi_worker";
const CHILD_PATH: &str = "/v1/kimi/chat/completions";
const SPAWN_CALL_ID: &str = "spawn-kimi-child";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn public_v2_spawn_delivers_plaintext_agent_message_to_native_kimi() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = responses::start_mock_server().await;
    let spawn_args = serde_json::to_string(&json!({
        "task_name": CHILD_TASK_NAME,
        "fork_turns": "none",
        "model": CHILD_SELECTOR,
        "plaintext_message": CHILD_PROMPT,
    }))?;

    let _spawn = responses::mount_sse_once_match(
        &server,
        |request: &Request| request_body_contains(request, PARENT_PROMPT),
        responses::sse(vec![
            responses::ev_response_created("parent-spawn"),
            responses::ev_function_call_with_namespace(
                SPAWN_CALL_ID,
                "agents",
                "spawn_agent",
                &spawn_args,
            ),
            responses::ev_completed("parent-spawn"),
        ]),
    )
    .await;
    let _parent_completion = responses::mount_sse_once_match(
        &server,
        |request: &Request| request_body_contains(request, SPAWN_CALL_ID),
        responses::sse(vec![
            responses::ev_response_created("parent-complete"),
            responses::ev_assistant_message("parent-message", "parent complete"),
            responses::ev_completed("parent-complete"),
        ]),
    )
    .await;
    Mock::given(method("POST"))
        .and(path(CHILD_PATH))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(kimi_response()),
        )
        .expect(1)
        .mount(&server)
        .await;

    let base_url = server.uri();
    let test = test_codex()
        .with_config(move |config| {
            let mut provider = built_in_model_providers(/*openai_base_url*/ None)
                .remove(CLAUDEFLARE_PROVIDER_ID)
                .expect("managed Claudeflare provider");
            provider.base_url = Some(format!("{base_url}/v1"));
            provider.supports_websockets = false;
            provider
                .wire_routes
                .get_mut("kimi_code")
                .expect("Kimi route")
                .base_url = format!("{base_url}/v1/kimi");
            config.model_provider = provider;
            config.model = Some("gpt-5.6-sol".to_string());
            let mut model_catalog = bundled_models_response().expect("bundled model catalog");
            model_catalog
                .models
                .iter_mut()
                .find(|model| model.slug == "kimi/kimi-for-coding")
                .expect("Kimi for Coding profile")
                .aliases
                .push(CHILD_SELECTOR.to_string());
            config.model_catalog = Some(model_catalog);
            config
                .features
                .enable(Feature::Collab)
                .expect("enable collaboration");
            config
                .features
                .enable(Feature::MultiAgentV2)
                .expect("enable V2 agents");
            config
                .features
                .disable(Feature::EnableRequestCompression)
                .expect("disable request compression");
            config.include_skill_instructions = false;
            config.include_apps_instructions = false;
            config.include_permissions_instructions = false;
            config.include_environment_context = false;
        })
        .build_with_auto_env(&server)
        .await?;
    let mut created_threads = test.thread_manager.subscribe_thread_created();

    test.submit_turn(PARENT_PROMPT).await?;
    let child_thread_id = timeout(Duration::from_secs(10), created_threads.recv()).await??;
    let child_thread = test.thread_manager.get_thread(child_thread_id).await?;
    let mut child_events = Vec::new();
    loop {
        let event = wait_for_event(&child_thread, |_| true).await;
        let complete = matches!(event, EventMsg::TurnComplete(_));
        child_events.push(event);
        if complete {
            break;
        }
    }
    let visible = child_events
        .iter()
        .filter_map(|event| match event {
            EventMsg::AgentMessageContentDelta(message) => Some(message.delta.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        visible,
        vec![CHILD_REPLY],
        "child events: {child_events:#?}"
    );
    let terminal = child_events
        .iter()
        .find_map(|event| match event {
            EventMsg::TurnComplete(terminal) => Some(terminal),
            _ => None,
        })
        .expect("child turn completion");
    assert!(terminal.error.is_none(), "child terminal: {terminal:#?}");
    assert_eq!(terminal.last_agent_message.as_deref(), Some(CHILD_REPLY));

    let requests = server.received_requests().await.expect("recorded requests");
    let request = requests
        .iter()
        .find(|request| request.url.path() == CHILD_PATH)
        .expect("Kimi child request");
    let body: Value = request.body_json()?;
    let rendered = format!("Agent message from /root to /root/{CHILD_TASK_NAME}:\n{CHILD_PROMPT}");
    let delivered = body["messages"]
        .as_array()
        .expect("native messages")
        .iter()
        .find_map(|message| {
            (message["role"] == "user")
                .then(|| message["content"].as_str().map(str::to_string))
                .flatten()
        });
    assert_eq!(delivered, Some(rendered));

    Ok(())
}

fn kimi_response() -> String {
    format!(
        "data: {}\n\ndata: [DONE]\n\n",
        json!({
            "id": "kimi-child",
            "choices": [{
                "index": 0,
                "delta": {"content": CHILD_REPLY},
                "finish_reason": "stop"
            }]
        })
    )
}

fn request_body_contains(request: &Request, text: &str) -> bool {
    String::from_utf8(request.body.clone()).is_ok_and(|body| body.contains(text))
}
