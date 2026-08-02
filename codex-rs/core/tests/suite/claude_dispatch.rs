use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use anyhow::Result;
use codex_model_provider_info::CLAUDEFLARE_PROVIDER_ID;
use codex_model_provider_info::built_in_model_providers;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::user_input::UserInput;
use core_test_support::responses;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::TestCodexBuilder;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event_match;
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
use wiremock::matchers::query_param;

fn event(name: &str, data: Value) -> String {
    format!("event: {name}\ndata: {data}\n\n")
}

fn start(id: &str, model: &str) -> String {
    event(
        "message_start",
        json!({"type":"message_start","message":{"id":id,"type":"message","role":"assistant","model":model,"content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":0}}}),
    )
}

fn text_terminal(id: &str, model: &str, text: &str, reason: &str) -> String {
    [
        start(id, model),
        event("content_block_start", json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}})),
        event("content_block_delta", json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":text}})),
        event("content_block_stop", json!({"type":"content_block_stop","index":0})),
        event("message_delta", json!({"type":"message_delta","delta":{"stop_reason":reason},"usage":{"output_tokens":2}})),
        event("message_stop", json!({"type":"message_stop"})),
    ]
    .concat()
}

fn parallel_exec_commands(model: &str) -> String {
    [
        start("msg-tools", model),
        event("content_block_start", json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call-one","name":"exec_command","input":{}}})),
        event("content_block_delta", json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"cmd\":\"pwd\",\"yield_time_ms\":1000,\"max_output_tokens\":1000}"}})),
        event("content_block_stop", json!({"type":"content_block_stop","index":0})),
        event("content_block_start", json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"call-two","name":"exec_command","input":{}}})),
        event("content_block_delta", json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"cmd\":\"pwd\",\"yield_time_ms\":1000,\"max_output_tokens\":1000}"}})),
        event("content_block_stop", json!({"type":"content_block_stop","index":1})),
        event("message_delta", json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":2}})),
        event("message_stop", json!({"type":"message_stop"})),
    ]
    .concat()
}

struct Sequence {
    next: AtomicUsize,
    bodies: Vec<String>,
}

impl Respond for Sequence {
    fn respond(&self, _: &Request) -> ResponseTemplate {
        let index = self.next.fetch_add(1, Ordering::SeqCst);
        ResponseTemplate::new(200)
            .insert_header("content-type", "text/event-stream")
            .set_body_string(self.bodies[index].clone())
    }
}

async fn mount_native(server: &MockServer, bodies: Vec<String>) {
    let count = bodies.len() as u64;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(query_param("beta", "true"))
        .respond_with(Sequence {
            next: AtomicUsize::new(0),
            bodies,
        })
        .up_to_n_times(count)
        .mount(server)
        .await;
}

fn native_builder(server: &MockServer, model: &str) -> TestCodexBuilder {
    let base_url = server.uri();
    let model = model.to_string();
    test_codex().with_config(move |config| {
        let mut provider = built_in_model_providers(/*openai_base_url*/ None)
            .remove(CLAUDEFLARE_PROVIDER_ID)
            .expect("managed Claudeflare provider");
        provider.base_url = Some(base_url.clone());
        provider
            .wire_routes
            .get_mut("claude_code")
            .expect("Claude route")
            .base_url = base_url;
        config.model_provider = provider;
        config.model = Some(model);
        config.base_instructions = Some("system".to_string());
        config.agents_enabled = false;
    })
}

async fn submit_and_expect_completion(
    test: &core_test_support::test_codex::TestCodex,
    text: &str,
) -> Result<()> {
    test.codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: text.to_string(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;

    wait_for_completion(test).await
}

async fn wait_for_completion(test: &core_test_support::test_codex::TestCodex) -> Result<()> {
    wait_for_event_match(&test.codex, |event| match event {
        EventMsg::Error(error) => Some(Err(anyhow::anyhow!(error.message.clone()))),
        EventMsg::TurnComplete(event) => Some(match &event.error {
            Some(error) => Err(anyhow::anyhow!(error.message.clone())),
            None => Ok(()),
        }),
        _ => None,
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_haiku_and_adaptive_requests_capture_exact_route_policy_and_identity() -> Result<()>
{
    skip_if_no_network!(Ok(()));
    for (model, wire_model, max_tokens, thinking, output_config) in [
        (
            "anthropic/claude-haiku-4-5-20251001",
            "claude-haiku-4-5-20251001",
            32_000,
            json!({"type":"enabled","budget_tokens":31999,"display":"omitted"}),
            Value::Null,
        ),
        (
            "anthropic/claude-sonnet-5",
            "claude-sonnet-5",
            64_000,
            json!({"type":"adaptive","display":"omitted"}),
            json!({"effort":"high"}),
        ),
    ] {
        let server = responses::start_mock_server().await;
        mount_native(
            &server,
            vec![text_terminal("msg-ok", wire_model, "done", "end_turn")],
        )
        .await;
        let test = native_builder(&server, model)
            .build_with_auto_env(&server)
            .await?;
        submit_and_expect_completion(&test, "hello").await?;
        let requests = server.received_requests().await.expect("requests");
        assert_eq!(requests.len(), 1);
        let request = &requests[0];
        assert_eq!(request.url.path(), "/v1/messages");
        assert_eq!(request.url.query(), Some("beta=true"));
        let body: Value = request.body_json()?;
        assert_eq!(body["model"], wire_model);
        assert_eq!(body["max_tokens"], max_tokens);
        assert_eq!(body["thinking"], thinking);
        assert_eq!(
            body.get("output_config").cloned().unwrap_or(Value::Null),
            output_config
        );
        assert_eq!(body["stream"], true);
        assert_eq!(body["system"][0]["text"], "system");
        assert_eq!(body["messages"][0]["role"], "user");
        assert!(
            body["tools"]
                .as_array()
                .is_some_and(|tools| !tools.is_empty())
        );
        for (name, value) in [
            ("x-app", "codex"),
            ("anthropic-version", "2023-06-01"),
            (
                "user-agent",
                concat!("codex-cli/", env!("CARGO_PKG_VERSION")),
            ),
        ] {
            assert_eq!(
                request
                    .headers
                    .get(name)
                    .and_then(|value| value.to_str().ok()),
                Some(value)
            );
        }
        assert!(request.headers.contains_key("anthropic-beta"));
        assert!(request.headers.contains_key("x-claude-code-session-id"));
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn incompatible_route_and_unsupported_capabilities_fail_before_io() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = responses::start_mock_server().await;
    let test = native_builder(&server, "anthropic/claude-sonnet-5")
        .with_config(|config| {
            config
                .model_provider
                .wire_routes
                .get_mut("claude_code")
                .expect("Claude route")
                .request_path = "v1/responses".to_string();
        })
        .build_with_auto_env(&server)
        .await?;
    let error = submit_and_expect_completion(&test, "wrong route")
        .await
        .expect_err("incompatible route should fail");
    assert!(
        error
            .to_string()
            .contains("must resolve to anthropic_messages/claude_code")
    );
    assert_eq!(server.received_requests().await.expect("requests").len(), 0);

    let server = responses::start_mock_server().await;
    let test = native_builder(&server, "anthropic/claude-sonnet-5")
        .with_config(|config| config.agents_enabled = true)
        .build_with_auto_env(&server)
        .await?;
    let error = submit_and_expect_completion(&test, "unsupported tools")
        .await
        .expect_err("namespace tool should fail");
    assert!(
        error
            .to_string()
            .contains("only JSON-schema function tools are supported")
    );
    assert_eq!(server.received_requests().await.expect("requests").len(), 0);

    let server = responses::start_mock_server().await;
    let test = native_builder(&server, "anthropic/claude-sonnet-5")
        .build_with_auto_env(&server)
        .await?;
    test.codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: "structured output".to_string(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: Some(json!({
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": false
            })),
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    let error = wait_for_completion(&test)
        .await
        .expect_err("structured output should fail");
    assert!(
        error.to_string().contains(
            "structured-output schemas are unsupported by native Claude Messages encoding"
        )
    );
    assert_eq!(server.received_requests().await.expect("requests").len(), 0);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn parallel_exec_commands_pause_and_completion_use_one_native_session() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    let model = "claude-sonnet-5";
    mount_native(
        &server,
        vec![
            parallel_exec_commands(model),
            text_terminal("msg-pause", model, "pause committed", "pause_turn"),
            text_terminal("msg-final", model, "finished", "end_turn"),
        ],
    )
    .await;
    let test = native_builder(&server, "anthropic/claude-sonnet-5")
        .build_with_auto_env(&server)
        .await?;
    submit_and_expect_completion(&test, "run tools").await?;
    let requests = server.received_requests().await.expect("requests");
    assert_eq!(requests.len(), 3);
    let session_ids = requests
        .iter()
        .map(|request| request.headers["x-claude-code-session-id"].clone())
        .collect::<Vec<_>>();
    assert!(session_ids.windows(2).all(|pair| pair[0] == pair[1]));
    let continuation: Value = requests[1].body_json()?;
    let continuation = continuation["messages"].to_string();
    for expected in ["call-one", "call-two", "tool_result"] {
        assert!(continuation.contains(expected), "missing {expected}");
    }
    let initial: Value = requests[0].body_json()?;
    assert!(
        initial["tools"]
            .as_array()
            .is_some_and(|tools| tools.iter().any(|tool| tool["name"] == "exec_command"))
    );
    let after_pause: Value = requests[2].body_json()?;
    assert!(
        after_pause["messages"]
            .to_string()
            .contains("pause committed")
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn discarding_and_protocol_terminals_leave_native_history_stable() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    let model = "claude-sonnet-5";
    let unknown = [start("msg-bad", model), event("message_delta", json!({"type":"message_delta","delta":{"stop_reason":"future_reason"},"usage":{"output_tokens":1}}))].concat();
    mount_native(
        &server,
        vec![
            text_terminal("msg-exhausted", model, "discard exhausted", "max_tokens"),
            text_terminal("msg-after-exhaustion", model, "recovered", "end_turn"),
            text_terminal("msg-refused", model, "discard refusal", "refusal"),
            text_terminal("msg-after-refusal", model, "recovered again", "end_turn"),
            unknown,
            text_terminal("msg-after-protocol", model, "protocol retry", "end_turn"),
        ],
    )
    .await;
    let test = native_builder(&server, "anthropic/claude-sonnet-5")
        .with_config(|config| config.model_provider.stream_max_retries = Some(1))
        .build_with_auto_env(&server)
        .await?;
    for prompt in [
        "exhaust",
        "after exhaustion",
        "refuse",
        "after refusal",
        "protocol",
    ] {
        test.submit_turn_with_environments(prompt, Some(Vec::new()))
            .await?;
    }
    let requests = server.received_requests().await.expect("requests");
    assert_eq!(requests.len(), 6);
    let bodies = requests
        .iter()
        .map(|request| request.body_json::<Value>().expect("body").to_string())
        .collect::<Vec<_>>();
    assert!(!bodies[1].contains("discard exhausted"));
    assert!(!bodies[3].contains("discard refusal"));
    assert!(!bodies[5].contains("future_reason"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn current_kimi_profile_stays_on_responses() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    let response = responses::mount_sse_once(
        &server,
        responses::sse(vec![
            responses::ev_assistant_message("msg", "done"),
            responses::ev_completed("response"),
        ]),
    )
    .await;
    let test = test_codex()
        .with_model("kimi/k3")
        .build_with_auto_env(&server)
        .await?;
    test.submit_turn("hello kimi").await?;
    let request = response.single_request();
    assert_eq!(request.path(), "/v1/responses");
    assert_eq!(request.body_json()["model"], "kimi/k3");
    Ok(())
}
