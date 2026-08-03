use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::Result;
use codex_features::Feature;
use codex_model_provider_info::CLAUDEFLARE_PROVIDER_ID;
use codex_model_provider_info::built_in_model_providers;
use core_test_support::responses;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_response_created;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::TestCodexBuilder;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::Request;
use wiremock::Respond;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;

const CHILD_MODEL: &str = "gpt-5.6-sol";
const CHILD_TASK: &str = "return a brief acknowledgement";
const SPAWN_CALL_ID: &str = "call-spawn-child";

#[derive(Clone, Copy)]
enum NativeParent {
    Claude,
    Kimi,
}

impl NativeParent {
    fn model(self) -> &'static str {
        match self {
            Self::Claude => "anthropic/claude-haiku-4-5-20251001",
            Self::Kimi => "kimi/kimi-for-coding",
        }
    }

    fn path(self) -> &'static str {
        match self {
            Self::Claude => "/v1/messages",
            Self::Kimi => "/v1/kimi/chat/completions",
        }
    }

    fn initial_response(self, arguments: &str) -> String {
        match self {
            Self::Claude => [
                claude_event(
                    "message_start",
                    json!({"type":"message_start","message":{"id":"msg-spawn","type":"message","role":"assistant","model":"claude-haiku-4-5-20251001","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":0}}}),
                ),
                claude_event(
                    "content_block_start",
                    json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":SPAWN_CALL_ID,"name":"spawn_agent","input":{}}}),
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
                    json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":2}}),
                ),
                claude_event("message_stop", json!({"type":"message_stop"})),
            ]
            .concat(),
            Self::Kimi => format!(
                "data: {}\n\ndata: [DONE]\n\n",
                json!({
                    "id": "kimi-spawn",
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": SPAWN_CALL_ID,
                                "type": "function",
                                "function": {"name": "spawn_agent", "arguments": arguments},
                            }],
                        },
                        "finish_reason": "tool_calls",
                    }],
                })
            ),
        }
    }

    fn terminal_response(self) -> String {
        match self {
            Self::Claude => [
                claude_event(
                    "message_start",
                    json!({"type":"message_start","message":{"id":"msg-final","type":"message","role":"assistant","model":"claude-haiku-4-5-20251001","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":0}}}),
                ),
                claude_event(
                    "content_block_start",
                    json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
                ),
                claude_event(
                    "content_block_delta",
                    json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"spawned"}}),
                ),
                claude_event(
                    "content_block_stop",
                    json!({"type":"content_block_stop","index":0}),
                ),
                claude_event(
                    "message_delta",
                    json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}),
                ),
                claude_event("message_stop", json!({"type":"message_stop"})),
            ]
            .concat(),
            Self::Kimi => format!(
                "data: {}\n\ndata: [DONE]\n\n",
                json!({
                    "id": "kimi-final",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": "spawned"},
                        "finish_reason": "stop",
                    }],
                })
            ),
        }
    }
}

fn claude_event(name: &str, data: Value) -> String {
    format!("event: {name}\ndata: {data}\n\n")
}

struct NativeSequence {
    next: AtomicUsize,
    responses: Vec<ResponseTemplate>,
}

impl Respond for NativeSequence {
    fn respond(&self, _: &Request) -> ResponseTemplate {
        self.responses[self.next.fetch_add(1, Ordering::SeqCst)].clone()
    }
}

async fn mount_native_parent(server: &MockServer, parent: NativeParent, arguments: &str) {
    let responses = [
        parent.initial_response(arguments),
        parent.terminal_response(),
    ]
    .into_iter()
    .map(|body| {
        ResponseTemplate::new(200)
            .insert_header("content-type", "text/event-stream")
            .set_body_string(body)
    })
    .collect();
    Mock::given(method("POST"))
        .and(path(parent.path()))
        .respond_with(NativeSequence {
            next: AtomicUsize::new(0),
            responses,
        })
        .up_to_n_times(2)
        .mount(server)
        .await;
}

fn native_parent_builder(server: &MockServer, parent: NativeParent) -> TestCodexBuilder {
    let base_url = server.uri();
    test_codex().with_config(move |config| {
        let mut provider = built_in_model_providers(/*openai_base_url*/ None)
            .remove(CLAUDEFLARE_PROVIDER_ID)
            .expect("managed Claudeflare provider");
        provider.base_url = Some(base_url.clone());
        provider
            .wire_routes
            .get_mut("claude_code")
            .expect("Claude route")
            .base_url = base_url.clone();
        provider
            .wire_routes
            .get_mut("kimi_code")
            .expect("Kimi route")
            .base_url = format!("{base_url}/v1/kimi");
        config.model_provider = provider;
        config.model = Some(parent.model().to_string());
        config.base_instructions = Some("system".to_string());
        config
            .features
            .enable(Feature::Collab)
            .expect("collaboration feature should be enableable");
        config
            .features
            .enable(Feature::MultiAgentV2)
            .expect("multi-agent V2 feature should be enableable");
        config.multi_agent_v2.expose_spawn_agent_model_overrides = true;
        config.include_skill_instructions = false;
        config.include_permissions_instructions = false;
        config.include_apps_instructions = false;
        config.include_environment_context = false;
    })
}

fn plain_spawn_agent_tool(body: &Value, parent: NativeParent) -> Option<&Value> {
    body["tools"].as_array()?.iter().find(|tool| match parent {
        NativeParent::Claude => tool["name"] == "spawn_agent",
        NativeParent::Kimi => tool["function"]["name"] == "spawn_agent",
    })
}

async fn wait_for_request_count(server: &MockServer, expected: usize) -> Result<Vec<Request>> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    loop {
        let requests = server.received_requests().await.expect("requests");
        if requests.len() >= expected {
            return Ok(requests);
        }
        if tokio::time::Instant::now() >= deadline {
            let paths = requests
                .iter()
                .map(|request| request.url.path().to_string())
                .collect::<Vec<_>>();
            anyhow::bail!(
                "timed out waiting for {expected} requests; received {} at {paths:?}",
                requests.len(),
            );
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_parents_route_plain_v2_spawn_agent_to_openai_child() -> Result<()> {
    skip_if_no_network!(Ok(()));
    for parent in [NativeParent::Claude, NativeParent::Kimi] {
        let server = responses::start_mock_server().await;
        let arguments = serde_json::to_string(&json!({
            "plaintext_message": CHILD_TASK,
            "task_name": "worker",
            "model": CHILD_MODEL,
            "fork_turns": "none",
        }))?;
        mount_native_parent(&server, parent, &arguments).await;
        responses::mount_sse_once(
            &server,
            responses::sse(vec![
                ev_response_created("child-response"),
                ev_assistant_message("child-message", "acknowledged"),
                ev_completed("child-response"),
            ]),
        )
        .await;

        let test = native_parent_builder(&server, parent)
            .build_with_auto_env(&server)
            .await?;
        test.submit_turn("spawn a worker").await?;

        let requests = wait_for_request_count(&server, /*expected*/ 3).await?;
        let native_requests = requests
            .iter()
            .filter(|request| request.url.path() == parent.path())
            .collect::<Vec<_>>();
        assert_eq!(native_requests.len(), 2);
        let initial: Value = native_requests[0].body_json()?;
        let initial_text = initial.to_string();
        assert!(initial_text.contains("using the form shown in their tool definitions"));
        assert!(!initial_text.contains("functions.agents.spawn_agent"));
        let spawn_agent = plain_spawn_agent_tool(&initial, parent)
            .expect("native parent should receive plain spawn_agent function");
        assert!(!spawn_agent.to_string().contains("namespace"));
        let continuation: Value = native_requests[1].body_json()?;
        let continuation_messages = continuation["messages"].to_string();
        for expected in [SPAWN_CALL_ID, "spawn_agent", "/root/worker"] {
            assert!(
                continuation_messages.contains(expected),
                "native continuation missing `{expected}`: {continuation_messages}"
            );
        }

        let child_request = requests
            .iter()
            .find(|request| request.url.path() == "/responses")
            .expect("OpenAI Responses child request");
        let child_body: Value = child_request.body_json()?;
        assert_eq!(child_body["model"], CHILD_MODEL);
        assert!(child_body.to_string().contains(CHILD_TASK));
    }
    Ok(())
}
