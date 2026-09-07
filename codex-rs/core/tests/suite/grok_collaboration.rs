use std::time::Duration;

use anyhow::Result;
use codex_features::Feature;
use codex_model_provider_info::CLAUDEFLARE_PROVIDER_ID;
use codex_model_provider_info::built_in_model_providers;
use codex_models_manager::bundled_models_response;
use codex_protocol::ThreadId;
use codex_protocol::protocol::AgentStatus;
use core_test_support::responses;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::TestCodexBuilder;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use wiremock::Mock;
use wiremock::Request;
use wiremock::Respond;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;

const SPAWN_CALL_ID: &str = "call-spawn-child";

fn grok_collaboration_builder(server: &wiremock::MockServer) -> TestCodexBuilder {
    let base_url = server.uri();
    test_codex().with_config(move |config| {
        let mut provider = built_in_model_providers(/*openai_base_url*/ None)
            .remove(CLAUDEFLARE_PROVIDER_ID)
            .expect("managed Claudeflare provider");
        provider.base_url = Some(base_url.clone());
        provider
            .wire_routes
            .get_mut("grok")
            .expect("Grok route")
            .base_url = format!("{base_url}/v1/grok");
        config.model_provider = provider;
        config.model = Some("xai/grok-4.6".to_string());
        config.model_catalog = Some(bundled_models_response().expect("bundled model catalog"));
        config.base_instructions = Some("Grok collaboration system".to_string());
        config
            .features
            .enable(Feature::Collab)
            .expect("enable collaboration");
        config
            .features
            .enable(Feature::MultiAgentV2)
            .expect("enable multi-agent V2");
        config
            .features
            .disable(Feature::EnableRequestCompression)
            .expect("disable request compression");
        config.multi_agent_v2.expose_spawn_agent_model_overrides = true;
        config.multi_agent_v2.tool_namespace = None;
        config.include_skill_instructions = false;
        config.include_permissions_instructions = false;
        config.include_apps_instructions = false;
        config.include_collaboration_mode_instructions = false;
        config.include_environment_context = false;
    })
}

fn grok_spawn_response(arguments: &str) -> String {
    responses::sse(vec![
        responses::ev_response_created("grok-root-spawn"),
        json!({
            "type": "response.output_item.done",
            "item": {
                "id": "rs_grok_spawn",
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": "Spawning a worker"}],
                "encrypted_content": format!("gAAAAAB{}", "A".repeat(1_459)),
            }
        }),
        responses::ev_function_call(SPAWN_CALL_ID, "spawn_agent", arguments),
        responses::ev_completed("grok-root-spawn"),
    ])
}

struct GrokRootSequence {
    spawn_arguments: String,
}

impl Respond for GrokRootSequence {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body = String::from_utf8_lossy(&request.body);
        let response = if body.contains("function_call_output") {
            responses::sse(vec![
                responses::ev_response_created("grok-root-complete"),
                responses::ev_assistant_message("grok-root-message", "spawned"),
                responses::ev_completed("grok-root-complete"),
            ])
        } else {
            grok_spawn_response(&self.spawn_arguments)
        };
        ResponseTemplate::new(200)
            .insert_header("content-type", "text/event-stream")
            .set_body_string(response)
    }
}

struct GrokSameFamilySequence {
    spawn_arguments: String,
    child_task: &'static str,
}

impl Respond for GrokSameFamilySequence {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body = String::from_utf8_lossy(&request.body);
        let response = if body.contains("function_call_output") {
            responses::sse(vec![
                responses::ev_response_created("grok-root-complete"),
                responses::ev_assistant_message("grok-root-message", "spawned"),
                responses::ev_completed("grok-root-complete"),
            ])
        } else if body.contains(self.child_task) {
            responses::sse(vec![
                responses::ev_response_created("grok-child"),
                responses::ev_assistant_message("grok-child-message", "grok child complete"),
                responses::ev_completed("grok-child"),
            ])
        } else {
            grok_spawn_response(&self.spawn_arguments)
        };
        ResponseTemplate::new(200)
            .insert_header("content-type", "text/event-stream")
            .set_body_string(response)
    }
}

async fn wait_for_child_completion(
    test: &core_test_support::test_codex::TestCodex,
    child_thread_id: ThreadId,
    expected: &str,
) -> Result<()> {
    let child = test.thread_manager.get_thread(child_thread_id).await?;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        match child.agent_status().await {
            AgentStatus::Completed(message) => {
                assert_eq!(message.as_deref(), Some(expected));
                return Ok(());
            }
            AgentStatus::Errored(error) => anyhow::bail!("child errored: {error}"),
            status if tokio::time::Instant::now() >= deadline => {
                anyhow::bail!("timed out waiting for child completion: {status:?}")
            }
            AgentStatus::PendingInit
            | AgentStatus::Running
            | AgentStatus::Interrupted
            | AgentStatus::Shutdown
            | AgentStatus::NotFound => tokio::time::sleep(Duration::from_millis(10)).await,
        }
    }
}

async fn wait_for_requests(server: &wiremock::MockServer, expected: usize) -> Result<Vec<Request>> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    loop {
        let requests = server.received_requests().await.expect("recorded requests");
        if requests.len() >= expected {
            return Ok(requests);
        }
        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!(
                "timed out waiting for {expected} requests; received {}",
                requests.len()
            );
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grok_root_routes_plaintext_task_to_openai_child() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = responses::start_mock_server().await;
    let child_task = "complete the mixed-family child task";
    let arguments = serde_json::to_string(&json!({
        "plaintext_message": child_task,
        "task_name": "openai_worker",
        "model": "gpt-5.6-sol",
        "fork_turns": "none",
    }))?;
    Mock::given(method("POST"))
        .and(path("/v1/grok/responses"))
        .respond_with(GrokRootSequence {
            spawn_arguments: arguments,
        })
        .up_to_n_times(2)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(responses::sse(vec![
                    responses::ev_response_created("openai-child"),
                    responses::ev_assistant_message("openai-child-message", "child complete"),
                    responses::ev_completed("openai-child"),
                ])),
        )
        .expect(1)
        .mount(&server)
        .await;
    let test = grok_collaboration_builder(&server)
        .build_with_auto_env(&server)
        .await?;
    let mut created_threads = test.thread_manager.subscribe_thread_created();

    test.submit_turn("spawn an OpenAI worker").await?;
    let child_thread_id =
        tokio::time::timeout(Duration::from_secs(10), created_threads.recv()).await??;
    wait_for_child_completion(&test, child_thread_id, "child complete").await?;

    let requests = wait_for_requests(&server, /*expected*/ 3).await?;
    let grok_requests = requests
        .iter()
        .filter(|request| request.url.path() == "/v1/grok/responses")
        .collect::<Vec<_>>();
    assert_eq!(grok_requests.len(), 2);
    let initial: Value = grok_requests[0].body_json()?;
    let spawn = initial["tools"]
        .as_array()
        .expect("Grok tools")
        .iter()
        .find(|tool| tool["name"] == "spawn_agent")
        .expect("plain spawn_agent function");
    assert!(spawn["parameters"]["properties"].get("message").is_none());
    assert!(
        spawn["parameters"]["required"]
            .as_array()
            .expect("required fields")
            .contains(&json!("plaintext_message"))
    );
    assert!(
        spawn["parameters"]["properties"].get("cwd").is_some(),
        "native wires should advertise the spawn_agent cwd parameter"
    );
    let continuation: Value = grok_requests[1].body_json()?;
    assert!(continuation.to_string().contains(SPAWN_CALL_ID));
    assert!(continuation.to_string().contains("function_call_output"));
    let child_request = requests
        .iter()
        .find(|request| request.url.path() == "/responses")
        .expect("OpenAI child request");
    let child_body: Value = child_request.body_json()?;
    assert_eq!(child_body["model"], "gpt-5.6-sol");
    assert!(child_body.to_string().contains(child_task));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grok_root_inherits_history_into_grok_child() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = responses::start_mock_server().await;
    let root_prompt = "spawn a Grok child with inherited history";
    let child_task = "confirm inherited Grok history";
    let arguments = serde_json::to_string(&json!({
        "plaintext_message": child_task,
        "task_name": "grok_worker",
        "fork_turns": "all",
    }))?;
    Mock::given(method("POST"))
        .and(path("/v1/grok/responses"))
        .respond_with(GrokSameFamilySequence {
            spawn_arguments: arguments,
            child_task,
        })
        .up_to_n_times(3)
        .mount(&server)
        .await;
    let test = grok_collaboration_builder(&server)
        .build_with_auto_env(&server)
        .await?;
    let mut created_threads = test.thread_manager.subscribe_thread_created();

    test.submit_turn(root_prompt).await?;
    let child_thread_id =
        tokio::time::timeout(Duration::from_secs(10), created_threads.recv()).await??;
    wait_for_child_completion(&test, child_thread_id, "grok child complete").await?;

    let requests = wait_for_requests(&server, /*expected*/ 3).await?;
    let child_request = requests
        .iter()
        .find(|request| String::from_utf8_lossy(&request.body).contains(child_task))
        .expect("Grok child request");
    let child_body: Value = child_request.body_json()?;
    assert_eq!(child_body["model"], "grok-4.6");
    assert!(child_body.to_string().contains(root_prompt));
    assert!(
        child_body["input"]
            .as_array()
            .expect("Grok child input")
            .iter()
            .all(|item| item["type"] != "agent_message")
    );
    assert!(child_body.to_string().contains(
        "Agent message from /root to /root/grok_worker:\\nconfirm inherited Grok history"
    ));
    let session_id = test.session_configured.session_id.to_string();
    let child_thread_id = child_thread_id.to_string();
    assert_eq!(
        child_request
            .headers
            .get("x-grok-session-id")
            .and_then(|value| value.to_str().ok()),
        Some(session_id.as_str())
    );
    assert_eq!(
        child_request
            .headers
            .get("x-grok-conv-id")
            .and_then(|value| value.to_str().ok()),
        Some(child_thread_id.as_str())
    );
    assert_eq!(child_body["prompt_cache_key"], session_id);
    Ok(())
}
