use std::collections::BTreeMap;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::Result;
use codex_model_provider_info::CLAUDEFLARE_PROVIDER_ID;
use codex_model_provider_info::built_in_model_providers;
use codex_protocol::protocol::AdditionalContextEntry;
use codex_protocol::protocol::AdditionalContextKind;
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
use tokio::net::TcpListener;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::Request;
use wiremock::Respond;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;
use wiremock::matchers::query_param;

pub(super) fn event(name: &str, data: Value) -> String {
    format!("event: {name}\ndata: {data}\n\n")
}

fn native_sse(body: String) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_string(body)
}

pub(super) fn start(id: &str, model: &str) -> String {
    start_with_input_tokens(id, model, /*input_tokens*/ 3)
}

fn start_with_input_tokens(id: &str, model: &str, input_tokens: i64) -> String {
    event(
        "message_start",
        json!({"type":"message_start","message":{"id":id,"type":"message","role":"assistant","model":model,"content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":input_tokens,"output_tokens":0}}}),
    )
}

fn text_terminal(id: &str, model: &str, text: &str, reason: &str) -> String {
    text_terminal_with_input_tokens(id, model, text, reason, /*input_tokens*/ 3)
}

fn text_terminal_with_input_tokens(
    id: &str,
    model: &str,
    text: &str,
    reason: &str,
    input_tokens: i64,
) -> String {
    [
        start_with_input_tokens(id, model, input_tokens),
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
    responses: Vec<ResponseTemplate>,
}

impl Respond for Sequence {
    fn respond(&self, _: &Request) -> ResponseTemplate {
        let index = self.next.fetch_add(1, Ordering::SeqCst);
        self.responses[index].clone()
    }
}

pub(super) async fn mount_native(server: &MockServer, bodies: Vec<String>) {
    let responses = bodies.into_iter().map(native_sse).collect();
    mount_native_responses(server, responses).await;
}

async fn mount_native_responses(server: &MockServer, responses: Vec<ResponseTemplate>) {
    let count = responses.len() as u64;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(query_param("beta", "true"))
        .respond_with(Sequence {
            next: AtomicUsize::new(0),
            responses,
        })
        .up_to_n_times(count)
        .mount(server)
        .await;
}

fn native_retry_limit(builder: TestCodexBuilder, limit: u64) -> TestCodexBuilder {
    builder.with_config(move |config| {
        config.model_provider.stream_max_retries = Some(0);
        config
            .model_provider
            .wire_routes
            .get_mut("claude_code")
            .expect("Claude route")
            .stream_max_retries = Some(limit);
    })
}

pub(super) fn native_builder(server: &MockServer, model: &str) -> TestCodexBuilder {
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
    submit(test, text).await?;
    wait_for_completion(test).await
}

async fn submit(test: &core_test_support::test_codex::TestCodex, text: &str) -> Result<()> {
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
    Ok(())
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

fn cache_control_count(value: &Value) -> usize {
    match value {
        Value::Array(values) => values.iter().map(cache_control_count).sum(),
        Value::Object(values) => {
            usize::from(values.contains_key("cache_control"))
                + values.values().map(cache_control_count).sum::<usize>()
        }
        _ => 0,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_cache_policy_bounds_production_request() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    let model = "claude-opus-4-8";
    mount_native(
        &server,
        vec![text_terminal(
            "cache-policy",
            model,
            "completed",
            "end_turn",
        )],
    )
    .await;
    let test = native_builder(&server, "anthropic/claude-opus-4-8")
        .build_with_auto_env(&server)
        .await?;
    test.codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: "production cache policy".to_string(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: BTreeMap::from([
                (
                    "one".to_string(),
                    AdditionalContextEntry {
                        value: "first application context".to_string(),
                        kind: AdditionalContextKind::Application,
                    },
                ),
                (
                    "two".to_string(),
                    AdditionalContextEntry {
                        value: "second application context".to_string(),
                        kind: AdditionalContextKind::Application,
                    },
                ),
                (
                    "three".to_string(),
                    AdditionalContextEntry {
                        value: "third application context".to_string(),
                        kind: AdditionalContextKind::Application,
                    },
                ),
            ]),
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_completion(&test).await?;

    let requests = server.received_requests().await.expect("native requests");
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert_eq!(request.url.path(), "/v1/messages");
    assert_eq!(request.url.query(), Some("beta=true"));
    let body: Value = request.body_json()?;
    let system = body["system"].as_array().expect("native system blocks");
    assert!(system.len() >= 4);
    assert!(
        system[..system.len() - 1]
            .iter()
            .all(|block| block.get("cache_control").is_none())
    );
    assert_eq!(
        system.last().and_then(|block| block.get("cache_control")),
        Some(&json!({"type": "ephemeral", "ttl": "1h"}))
    );
    assert!(
        body["tools"]
            .as_array()
            .is_some_and(|tools| !tools.is_empty())
    );
    assert!(body["tools"].as_array().is_some_and(|tools| {
        tools
            .iter()
            .any(|tool| tool["name"] == "exec_command" && tool.get("input_schema").is_some())
    }));
    assert!(
        body["tools"]
            .as_array()
            .expect("native tools")
            .iter()
            .all(|tool| tool.get("cache_control").is_none())
    );
    let latest_user = body["messages"]
        .as_array()
        .expect("native messages")
        .iter()
        .rev()
        .find(|message| message["role"] == "user")
        .expect("latest native user message");
    let latest_user_block = latest_user["content"]
        .as_array()
        .expect("latest user content")
        .last()
        .expect("latest user block");
    assert_eq!(
        latest_user_block.get("cache_control"),
        Some(&json!({"type": "ephemeral", "ttl": "1h"}))
    );
    assert_eq!(cache_control_count(&body), 2);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_profiles_capture_route_policy_and_compact_at_reserved_boundary() -> Result<()> {
    skip_if_no_network!(Ok(()));
    for (model, wire_model, max_tokens, prior_input_tokens, thinking, output_config) in [
        (
            "anthropic/claude-haiku-4-5-20251001",
            "claude-haiku-4-5-20251001",
            32_000,
            157_500,
            json!({"type":"enabled","budget_tokens":31999,"display":"omitted"}),
            Value::Null,
        ),
        (
            "anthropic/claude-opus-4-8",
            "claude-opus-4-8",
            64_000,
            885_500,
            json!({"type":"adaptive","display":"omitted"}),
            json!({"effort":"high"}),
        ),
    ] {
        let server = responses::start_mock_server().await;
        mount_native(
            &server,
            vec![
                text_terminal_with_input_tokens(
                    "seed",
                    wire_model,
                    "seeded",
                    "end_turn",
                    prior_input_tokens,
                ),
                text_terminal("compact", wire_model, "boundary summary", "end_turn"),
                text_terminal("success", wire_model, "done", "end_turn"),
            ],
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
        assert!(request.headers.contains_key("originator"));
        assert!(request.headers.contains_key("x-claude-code-session-id"));
        let pending = format!("boundary input {}", "x".repeat(4_000));
        test.submit_turn_with_environments(&pending, Some(Vec::new()))
            .await?;

        let requests = server.received_requests().await.expect("requests");
        assert_eq!(requests.len(), 3);
        let compact: Value = requests[1].body_json()?;
        assert_eq!(compact["max_tokens"], max_tokens);
        assert!(
            compact
                .to_string()
                .contains("CONTEXT CHECKPOINT COMPACTION")
        );
        assert!(compact.to_string().contains(&pending));
        let sampling: Value = requests[2].body_json()?;
        assert!(sampling.to_string().contains("boundary summary"));
        assert!(sampling.to_string().contains(&pending));
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn incompatible_route_and_structured_output_fail_before_io() -> Result<()> {
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
    let model = "claude-opus-4-8";
    let unknown = [start("msg-bad", model), event("message_delta", json!({"type":"message_delta","delta":{"stop_reason":"future_reason"},"usage":{"output_tokens":1}}))].concat();
    mount_native(
        &server,
        vec![
            text_terminal("msg-exhausted", model, "discard exhausted", "max_tokens"),
            text_terminal("msg-after-exhaustion", model, "recovered", "end_turn"),
            text_terminal("msg-refused", model, "discard refusal", "refusal"),
            text_terminal("msg-after-refusal", model, "recovered again", "end_turn"),
            unknown,
        ],
    )
    .await;
    let test = native_retry_limit(native_builder(&server, "anthropic/claude-opus-4-8"), 3)
        .build_with_auto_env(&server)
        .await?;
    for prompt in ["exhaust", "after exhaustion", "refuse", "after refusal"] {
        test.submit_turn_with_environments(prompt, Some(Vec::new()))
            .await?;
    }
    let error = submit_and_expect_completion(&test, "protocol")
        .await
        .expect_err("strict protocol failure should not retry");
    assert!(
        error
            .to_string()
            .contains("unknown native Claude stop reason")
    );
    let requests = server.received_requests().await.expect("requests");
    assert_eq!(requests.len(), 5);
    let bodies = requests
        .iter()
        .map(|request| request.body_json::<Value>().expect("body").to_string())
        .collect::<Vec<_>>();
    assert!(!bodies[1].contains("discard exhausted"));
    assert!(!bodies[3].contains("discard refusal"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_route_retry_limit_covers_http_sse_and_stable_history() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    let model = "claude-sonnet-5";
    let partial = [
        start("failed-message", model),
        event("content_block_start", json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}})),
        event("content_block_delta", json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"failed reasoning"}})),
        event("content_block_delta", json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"failed-signature"}})),
        event("content_block_stop", json!({"type":"content_block_stop","index":0})),
        event("content_block_start", json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"failed-call","name":"exec_command","input":{}}})),
        event("content_block_delta", json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"cmd\":\"printf failed-tool-executed\",\"yield_time_ms\":1000,\"max_output_tokens\":1000}"}})),
        event("content_block_stop", json!({"type":"content_block_stop","index":1})),
    ].concat();
    let mut retry_responses = vec![
        native_sse(partial),
        native_sse(event(
            "error",
            json!({"type":"error","error":{"type":"overloaded_error","message":"busy"}}),
        )),
    ];
    retry_responses.extend(
        [408, 409, 429, 500, 502, 503, 504, 529]
            .map(|status| ResponseTemplate::new(status).insert_header("retry-after-ms", "0")),
    );
    retry_responses.push(native_sse(text_terminal(
        "success", model, "done", "end_turn",
    )));
    mount_native_responses(&server, retry_responses).await;
    let test = native_retry_limit(native_builder(&server, "anthropic/claude-sonnet-5"), 10)
        .build_with_auto_env(&server)
        .await?;
    submit(&test, "retry cleanly").await?;
    wait_for_event_match(&test.codex, |event| match event {
        EventMsg::ExecCommandBegin(event) if event.call_id == "failed-call" => {
            Some(Err(anyhow::anyhow!("failed-attempt tool executed")))
        }
        EventMsg::Error(error) => Some(Err(anyhow::anyhow!(error.message.clone()))),
        EventMsg::TurnComplete(event) => Some(match &event.error {
            Some(error) => Err(anyhow::anyhow!(error.message.clone())),
            None => Ok(()),
        }),
        _ => None,
    })
    .await?;
    let requests = server.received_requests().await.expect("requests");
    assert_eq!(requests.len(), 11);
    for request in &requests[1..] {
        let body = request.body_json::<Value>()?.to_string();
        for failed in [
            "failed-message",
            "failed reasoning",
            "failed-signature",
            "failed-call",
            "failed-tool-executed",
            "tool_result",
        ] {
            assert!(!body.contains(failed), "retry retained {failed}");
        }
    }

    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let base_url = format!("http://{}", listener.local_addr()?);
    let connections = tokio::spawn(async move {
        for _ in 0..2 {
            drop(listener.accept().await?.0);
        }
        std::io::Result::Ok(())
    });
    let server = responses::start_mock_server().await;
    let test = native_retry_limit(native_builder(&server, "anthropic/claude-sonnet-5"), 1)
        .with_config(move |config| {
            config
                .model_provider
                .wire_routes
                .get_mut("claude_code")
                .expect("Claude route")
                .base_url = base_url;
        })
        .build_with_auto_env(&server)
        .await?;
    submit_and_expect_completion(&test, "exhaust connection limit")
        .await
        .expect_err("limit");
    tokio::time::timeout(Duration::from_secs(5), connections).await???;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_invalid_auth_and_quota_fail_without_retry() -> Result<()> {
    skip_if_no_network!(Ok(()));
    for response in [
        ResponseTemplate::new(400).set_body_string("invalid request"),
        ResponseTemplate::new(401).set_body_string("authentication_error"),
        ResponseTemplate::new(429).set_body_string(
            r#"{"error":{"type":"billing_error","message":"credit balance is too low"}}"#,
        ),
    ] {
        let server = responses::start_mock_server().await;
        mount_native_responses(&server, vec![response]).await;
        let test = native_retry_limit(native_builder(&server, "anthropic/claude-sonnet-5"), 3)
            .build_with_auto_env(&server)
            .await?;
        submit_and_expect_completion(&test, "nonretryable")
            .await
            .expect_err("nonretryable native failure");
        assert_eq!(server.received_requests().await.expect("requests").len(), 1);
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_context_overflow_compacts_locally_then_retries() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    let model = "claude-sonnet-5";
    mount_native_responses(&server, vec![
        ResponseTemplate::new(400).set_body_string(r#"{"error":{"type":"invalid_request_error","message":"prompt is too long for the context window"}}"#),
        ResponseTemplate::new(200).insert_header("content-type", "text/event-stream").set_body_string(text_terminal("summary", model, "short stable summary", "end_turn")),
        ResponseTemplate::new(200).insert_header("content-type", "text/event-stream").set_body_string(text_terminal("success", model, "done", "end_turn")),
    ]).await;
    let test = native_retry_limit(native_builder(&server, "anthropic/claude-sonnet-5"), 1)
        .build_with_auto_env(&server)
        .await?;
    submit_and_expect_completion(&test, "overflow me").await?;
    let requests = server.received_requests().await.expect("requests");
    assert_eq!(requests.len(), 3);
    let retry: Value = requests[2].body_json()?;
    let summary_message = retry["messages"]
        .as_array()
        .and_then(|messages| {
            messages
                .iter()
                .find(|message| message.to_string().contains("short stable summary"))
        })
        .expect("plaintext summary in compacted history");
    assert_eq!(summary_message["role"], "user");
    assert!(!retry.to_string().contains("failed-message"));
    Ok(())
}
