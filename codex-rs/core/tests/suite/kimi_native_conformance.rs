use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use anyhow::Result;
use codex_core::ForkSnapshot;
use codex_core::StartThreadOptions;
use codex_model_provider_info::CLAUDEFLARE_PROVIDER_ID;
use codex_model_provider_info::built_in_model_providers;
use codex_protocol::openai_models::ConfigShellToolType;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::protocol::SessionSource;
use codex_protocol::protocol::SubAgentSource;
use codex_protocol::protocol::ThreadSettingsOverrides;
use codex_protocol::user_input::UserInput;
use core_test_support::responses;
use core_test_support::skip_if_no_network;
use core_test_support::submit_thread_settings;
use core_test_support::test_codex::TestCodexBuilder;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use tempfile::TempDir;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::Request;
use wiremock::Respond;
use wiremock::ResponseTemplate;
use wiremock::http::Method;
use wiremock::matchers::method;
use wiremock::matchers::path;

const KIMI_PROFILE: &str = "kimi/k3";
const TRACE_SECRET: &str = "provider-trace-must-not-be-model-visible";

fn chunk(id: &str, delta: Value, finish_reason: Option<&str>) -> String {
    format!(
        "data: {}\n\n",
        json!({"id":id,"choices":[{"index":0,"delta":delta,"finish_reason":finish_reason}]})
    )
}

fn terminal(id: &str, delta: Value, finish_reason: &str) -> String {
    format!("{}data: [DONE]\n\n", chunk(id, delta, Some(finish_reason)))
}

fn text_terminal(id: &str, text: &str) -> String {
    terminal(id, json!({"content":text}), "stop")
}

fn parallel_tools() -> String {
    let usage = json!({
        "prompt_tokens":20,"completion_tokens":9,"total_tokens":29,
        "prompt_tokens_details":{"cached_tokens":7},
        "completion_tokens_details":{"reasoning_tokens":4}
    });
    [
        chunk("tools", json!({"reasoning_content":"think ","tool_calls":[
            {"index":4,"id":"call-four","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":\"pwd\",\"yield_time_ms\":"}}
        ]}), None),
        chunk("tools", json!({"content":"working ","tool_calls":[
            {"index":2,"id":"call-two","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":\"pwd\",\"yield_time_ms\":1000,\"max_output_tokens\":1000}"}},
            {"index":4,"function":{"arguments":"1000,\"max_output_tokens\":1000}"}}
        ]}), None),
        format!("data: {}\n\n", json!({"id":"tools","choices":[{
            "index":0,"delta":{},"finish_reason":"tool_calls","usage":usage
        }],"usage":usage})),
        "data: [DONE]\n\n".to_string(),
    ]
    .concat()
}

fn native_response(body: String) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .insert_header("x-trace-id", TRACE_SECRET)
        .set_body_string(body)
}

struct Sequence {
    next: AtomicUsize,
    responses: Vec<ResponseTemplate>,
}

impl Respond for Sequence {
    fn respond(&self, _: &Request) -> ResponseTemplate {
        self.responses[self.next.fetch_add(1, Ordering::SeqCst)].clone()
    }
}

async fn mount_native(server: &MockServer, responses: Vec<ResponseTemplate>) {
    let count = responses.len() as u64;
    Mock::given(method("POST"))
        .and(path("/v1/kimi/chat/completions"))
        .respond_with(Sequence {
            next: AtomicUsize::new(0),
            responses,
        })
        .up_to_n_times(count)
        .mount(server)
        .await;
}

fn native_builder(server: &MockServer) -> TestCodexBuilder {
    let base_url = format!("{}/v1/kimi", server.uri());
    test_codex().with_config(move |config| {
        let mut provider = built_in_model_providers(/*openai_base_url*/ None)
            .remove(CLAUDEFLARE_PROVIDER_ID)
            .expect("managed Claudeflare provider");
        provider.supports_websockets = true;
        provider
            .wire_routes
            .get_mut("kimi_code")
            .expect("Kimi route")
            .base_url = base_url;
        provider
            .wire_routes
            .get_mut("kimi_code")
            .expect("Kimi route")
            .stream_max_retries = Some(6);
        config.model_provider = provider;
        config.model = Some(KIMI_PROFILE.to_string());
        config.base_instructions = Some("Kimi conformance system".to_string());
        config.agents_enabled = false;
        config.include_skill_instructions = false;
        config.include_permissions_instructions = false;
        config.include_apps_instructions = false;
        config.include_collaboration_mode_instructions = false;
        config.include_environment_context = false;
    })
}

async fn submit(thread: &codex_core::CodexThread, prompt: &str) -> Result<Vec<EventMsg>> {
    thread
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: prompt.to_string(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    let mut events = Vec::new();
    loop {
        let event = wait_for_event(thread, |_| true).await;
        let complete = matches!(event, EventMsg::TurnComplete(_));
        events.push(event);
        if complete {
            return Ok(events);
        }
    }
}

fn body(request: &Request) -> Result<Value> {
    Ok(request.body_json()?)
}

fn error(events: &[EventMsg]) -> Option<&str> {
    events.iter().find_map(|event| match event {
        EventMsg::Error(error) => Some(error.message.as_str()),
        EventMsg::TurnComplete(event) => event.error.as_ref().map(|error| error.message.as_str()),
        _ => None,
    })
}

fn assert_no_private_leak(value: &str) {
    for private in [
        TRACE_SECRET,
        "failed reasoning",
        "failed-call",
        "discarded thinking",
    ] {
        assert!(!value.contains(private), "leaked {private}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ordered_parallel_loop_replays_reasoning_usage_and_matched_results() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    let final_usage = json!({
        "prompt_tokens":5,"completion_tokens":2,"total_tokens":7,
        "prompt_tokens_details":{"cached_tokens":0},
        "completion_tokens_details":{"reasoning_tokens":0}
    });
    mount_native(
        &server,
        vec![
            native_response(parallel_tools()),
            native_response(format!(
                "data: {}\n\ndata: [DONE]\n\n",
                json!({"id":"final","choices":[{"index":0,"delta":{"content":"finished"},"finish_reason":"stop","usage":final_usage}],"usage":final_usage})
            )),
            native_response(text_terminal("prod", "ready again")),
        ],
    )
    .await;
    let test = native_builder(&server).build_with_auto_env(&server).await?;
    let events = submit(&test.codex, "run parallel tools").await?;

    let presented = events
        .iter()
        .filter_map(|event| match event {
            EventMsg::ReasoningRawContentDelta(event) => {
                Some(("reasoning", event.item_id.as_str(), event.delta.as_str()))
            }
            EventMsg::AgentMessageContentDelta(event) => {
                Some(("message", event.item_id.as_str(), event.delta.as_str()))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        presented,
        [
            ("reasoning", "rs_kimi", "think "),
            ("message", "msg_kimi", "working "),
            ("message", "msg_kimi", "finished"),
        ]
    );
    let mut executed = events
        .iter()
        .filter_map(|event| match event {
            EventMsg::ExecCommandBegin(event) => Some(event.call_id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    executed.sort_unstable();
    assert_eq!(executed, ["call-four", "call-two"]);
    let usage = events.iter().rev().find_map(|event| match event {
        EventMsg::TokenCount(event) => event.info.as_ref(),
        _ => None,
    });
    assert_eq!(
        serde_json::to_value(usage)?,
        json!({
            "total_token_usage": {
                "input_tokens":25,"cached_input_tokens":7,"cache_write_input_tokens":0,
                "output_tokens":11,"reasoning_output_tokens":4,"total_tokens":36
            },
            "last_token_usage": {
                "input_tokens":5,"cached_input_tokens":0,"cache_write_input_tokens":0,
                "output_tokens":2,"reasoning_output_tokens":0,"total_tokens":7
            },
            "model_context_window":996147
        })
    );

    let requests = server.received_requests().await.expect("native requests");
    assert_eq!(requests.len(), 2);
    let first = body(&requests[0])?;
    let continuation = body(&requests[1])?;
    assert_eq!(first["prompt_cache_key"], continuation["prompt_cache_key"]);
    let expected_continuation = json!([
        {"role":"assistant","content":"working ","reasoning_content":"think ","tool_calls":[
            {"id":"call-four","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":\"pwd\",\"yield_time_ms\":1000,\"max_output_tokens\":1000}"}},
            {"id":"call-two","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":\"pwd\",\"yield_time_ms\":1000,\"max_output_tokens\":1000}"}}
        ]},
        {"role":"tool","tool_call_id":"call-four","content":continuation["messages"][3]["content"].clone()},
        {"role":"tool","tool_call_id":"call-two","content":continuation["messages"][4]["content"].clone()}
    ]);
    assert_eq!(
        continuation["messages"].as_array().expect("messages")[2..],
        expected_continuation.as_array().expect("expected messages")[..]
    );
    assert_no_private_leak(&continuation.to_string());

    submit(&test.codex, "prod after tools").await?;
    let requests = server.received_requests().await.expect("native requests");
    let third = body(&requests[2])?;
    assert_eq!(third["messages"][5]["reasoning_content"], "");
    assert_no_private_leak(&third.to_string());
    let rollout = std::fs::read_to_string(test.codex.rollout_path().expect("rollout path"))?;
    assert!(!rollout.contains(TRACE_SECRET));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn authoritative_final_usage_executes_and_replays_tool_once() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    let preliminary_usage = json!({
        "prompt_tokens":16646,"completion_tokens":654,"total_tokens":17300
    });
    let final_usage = json!({
        "prompt_tokens":16648,"completion_tokens":654,"total_tokens":17302,
        "prompt_tokens_details":{"cached_tokens":12000},
        "completion_tokens_details":{"reasoning_tokens":378}
    });
    let tool_stream = [
        chunk(
            "authoritative-usage",
            json!({"reasoning_content":"inspect usage ","tool_calls":[{
                "index":0,"id":"usage-call","type":"function",
                "function":{"name":"exec_command","arguments":"{\"cmd\":\"pwd\",\"yield_time_ms\":1000,\"max_output_tokens\":1000}"}
            }]}),
            None,
        ),
        format!("data: {}\n\n", json!({"id":"authoritative-usage","choices":[{
            "index":0,"delta":{},"finish_reason":"tool_calls","usage":preliminary_usage
        }]})),
        format!("data: {}\n\n", json!({
            "id":"authoritative-usage","choices":[],"usage":final_usage
        })),
        "data: [DONE]\n\n".to_string(),
    ]
    .concat();
    mount_native(
        &server,
        vec![
            native_response(tool_stream),
            native_response(text_terminal("continued", "usage accepted")),
        ],
    )
    .await;
    let test = native_builder(&server).build_with_auto_env(&server).await?;
    let events = submit(&test.codex, "run one usage tool").await?;

    assert!(error(&events).is_none());
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, EventMsg::ExecCommandBegin(_)))
            .count(),
        1
    );
    let usage = events.iter().rev().find_map(|event| match event {
        EventMsg::TokenCount(event) => event.info.as_ref(),
        _ => None,
    });
    let expected_usage = codex_protocol::protocol::TokenUsage {
        input_tokens: 16648,
        cached_input_tokens: 12000,
        cache_write_input_tokens: 0,
        output_tokens: 654,
        reasoning_output_tokens: 378,
        total_tokens: 17302,
    };
    assert_eq!(
        usage.map(|usage| &usage.last_token_usage),
        Some(&expected_usage)
    );
    assert_eq!(
        usage.map(|usage| &usage.total_token_usage),
        Some(&expected_usage)
    );

    let requests = server.received_requests().await.expect("native requests");
    assert_eq!(requests.len(), 2);
    let continuation = body(&requests[1])?;
    assert_eq!(
        continuation["messages"][2],
        json!({
            "role":"assistant",
            "reasoning_content":"inspect usage ",
            "tool_calls":[{
                "id":"usage-call","type":"function","function":{
                    "name":"exec_command",
                    "arguments":"{\"cmd\":\"pwd\",\"yield_time_ms\":1000,\"max_output_tokens\":1000}"
                }
            }]
        })
    );
    assert_eq!(continuation["messages"][3]["role"], "tool");
    assert_eq!(continuation["messages"][3]["tool_call_id"], "usage-call");
    assert!(
        continuation["messages"][3]["content"]
            .as_str()
            .is_some_and(|content| !content.is_empty())
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn terminal_and_retry_matrix_preserves_exact_stable_history() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    let malformed = "data: {not-json}\n\n".to_string();
    let premature = chunk(
        "premature",
        json!({"reasoning_content":"failed reasoning"}),
        None,
    );
    let missing_done = chunk(
        "missing-done",
        json!({"content":"failed visible"}),
        Some("stop"),
    );
    let empty_stop = terminal("empty", json!({"content":""}), "stop");
    let thinking_stop = terminal(
        "thinking",
        json!({"reasoning_content":"discarded thinking"}),
        "stop",
    );
    mount_native(
        &server,
        vec![
            native_response(text_terminal("normal", "normal visible")),
            native_response(terminal(
                "function",
                json!({"tool_calls":[{"index":0,"id":"function-call","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":\"pwd\",\"yield_time_ms\":1000,\"max_output_tokens\":1000}"}}]}),
                "function_call",
            )),
            native_response(text_terminal("function-final", "function complete")),
            native_response(terminal("length", json!({"reasoning":"discard length"}), "length")),
            native_response(text_terminal("after-length", "after length")),
            native_response(terminal("max", json!({"content":"discard max"}), "max_tokens")),
            native_response(terminal("filter", json!({"content":"discard filter"}), "content_filter")),
            native_response(malformed),
            native_response(premature),
            native_response(missing_done),
            native_response(empty_stop),
            native_response(thinking_stop),
            native_response(text_terminal("retried", "retry complete")),
            native_response(terminal("unknown", json!({"content":"discard unknown"}), "future")),
            native_response(format!(
                "{}data: [DONE]\n\n",
                chunk("missing", json!({"content":"discard missing"}), None)
            )),
            native_response(text_terminal("reused", "thread reused")),
        ],
    )
    .await;
    let test = native_builder(&server).build_with_auto_env(&server).await?;

    assert!(error(&submit(&test.codex, "normal").await?).is_none());
    let tools = submit(&test.codex, "function call").await?;
    assert_eq!(
        tools
            .iter()
            .filter(|event| matches!(event, EventMsg::ExecCommandBegin(_)))
            .count(),
        1
    );
    let length = submit(&test.codex, "reasoning length").await?;
    assert!(error(&length).is_some_and(|error| error.contains("output limit")));
    assert!(error(&submit(&test.codex, "after length").await?).is_none());
    for (prompt, expected) in [("max tokens", "output limit"), ("filtered", "refused")] {
        assert!(
            error(&submit(&test.codex, prompt).await?)
                .is_some_and(|error| error.contains(expected))
        );
    }
    assert!(error(&submit(&test.codex, "retry all transient streams").await?).is_none());
    for prompt in ["unknown finish", "missing finish"] {
        assert!(error(&submit(&test.codex, prompt).await?).is_some());
    }
    assert!(error(&submit(&test.codex, "reuse after strict failures").await?).is_none());

    let requests = server.received_requests().await.expect("native requests");
    assert_eq!(requests.len(), 16);
    let bodies = requests.iter().map(body).collect::<Result<Vec<_>>>()?;
    assert_eq!(bodies[7..13], vec![bodies[7].clone(); 6]);
    for index in [4, 7, 13, 14, 15] {
        assert_no_private_leak(&bodies[index].to_string());
    }
    for discarded in [
        "discard length",
        "discard max",
        "discard filter",
        "discard unknown",
        "discard missing",
    ] {
        assert!(!bodies[15].to_string().contains(discarded));
    }
    assert!(bodies[4].to_string().contains("reasoning length"));
    assert!(bodies[4].to_string().contains("after length"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pre_native_resume_learns_reasoning_without_opaque_or_cross_model_replay() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let legacy_server = responses::start_mock_server().await;
    responses::mount_sse_sequence(
        &legacy_server,
        vec![
            responses::sse(vec![
                responses::ev_response_created("legacy-tools"),
                responses::ev_reasoning_item("opaque", &["visible legacy"], &[]),
                responses::ev_assistant_message("legacy-message", "compatible text"),
                responses::ev_function_call(
                    "legacy-call",
                    "exec_command",
                    r#"{"cmd":"pwd","yield_time_ms":1000,"max_output_tokens":1000}"#,
                ),
                responses::ev_completed("legacy-tools"),
            ]),
            responses::sse(vec![
                responses::ev_response_created("legacy-final"),
                responses::ev_assistant_message("legacy-final-message", "legacy complete"),
                responses::ev_completed("legacy-final"),
            ]),
        ],
    )
    .await;
    let legacy = test_codex()
        .with_model(KIMI_PROFILE)
        .with_model_info_override(KIMI_PROFILE, |info| info.inference = None)
        .build_with_auto_env(&legacy_server)
        .await?;
    legacy.submit_turn("legacy turn").await?;
    let home = legacy.home.clone();
    let rollout_path = legacy.codex.rollout_path().expect("rollout path");
    let before_native = std::fs::read_to_string(&rollout_path)?;
    assert!(before_native.contains("encrypted_content"));
    assert!(!before_native.contains("codex:kimi-chat-reasoning:"));
    legacy.codex.shutdown_and_wait().await?;

    let native_server = responses::start_mock_server().await;
    mount_native(
        &native_server,
        vec![
            native_response(terminal(
                "native-reasoning",
                json!({"reasoning":"native private thought","content":"native answer"}),
                "stop",
            )),
            native_response(text_terminal("same-model", "same model replayed")),
            native_response(text_terminal("switched-model", "switched safely")),
        ],
    )
    .await;
    let resumed = native_builder(&native_server)
        .resume_with_auto_env(&native_server, home, rollout_path)
        .await?;
    submit(&resumed.codex, "first native").await?;
    submit(&resumed.codex, "replay native").await?;
    submit_thread_settings(
        &resumed.codex,
        ThreadSettingsOverrides {
            model: Some("kimi/k3-256k".to_string()),
            ..Default::default()
        },
    )
    .await?;
    submit(&resumed.codex, "switch exact model").await?;

    let requests = native_server
        .received_requests()
        .await
        .expect("native requests");
    assert_eq!(requests.len(), 3);
    let bodies = requests.iter().map(body).collect::<Result<Vec<_>>>()?;
    let first = bodies[0].to_string();
    for compatible in ["compatible text", "legacy-call", "tool"] {
        assert!(first.contains(compatible), "missing {compatible}");
    }
    for incompatible in [
        "YmJiYmJiYmJi",
        "encrypted_content",
        "codex:kimi-chat-reasoning:",
    ] {
        assert!(!first.contains(incompatible), "leaked {incompatible}");
    }
    assert!(
        bodies[1]
            .to_string()
            .contains("\"reasoning\":\"native private thought\"")
    );
    assert_eq!(bodies[2]["model"], "k3-256k");
    assert!(!bodies[2].to_string().contains("native private thought"));
    assert_eq!(bodies[0]["prompt_cache_key"], bodies[2]["prompt_cache_key"]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn affinity_survives_retry_tools_compaction_resume_and_subagent() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    mount_native(
        &server,
        vec![
            native_response(text_terminal("ordinary", "ordinary answer")),
            ResponseTemplate::new(503).insert_header("retry-after-ms", "0"),
            native_response(text_terminal("retried", "retry answer")),
            native_response(parallel_tools()),
            native_response(text_terminal("tools-done", "tools complete")),
            ResponseTemplate::new(400).set_body_string(
                r#"{"error":{"type":"invalid_request_error","message":"maximum context length exceeded"}}"#,
            ),
            native_response(text_terminal("summary", "stable compacted summary")),
            native_response(text_terminal("compacted", "compaction complete")),
            native_response(text_terminal("resumed", "resume complete")),
            native_response(text_terminal("branch", "branch complete")),
            native_response(text_terminal("subagent", "subagent complete")),
            native_response(text_terminal("distinct", "distinct complete")),
        ],
    )
    .await;
    let root = native_builder(&server).build_with_auto_env(&server).await?;
    for prompt in [
        "ordinary turn",
        "retry turn",
        "parallel tool turn",
        "overflow turn",
    ] {
        submit(&root.codex, prompt).await?;
    }
    let first_requests = server.received_requests().await.expect("native requests");
    assert_eq!(first_requests.len(), 8);
    let expected_key = root.session_configured.session_id.to_string();
    assert!(first_requests.iter().all(|request| {
        body(request).is_ok_and(|body| body["prompt_cache_key"] == expected_key)
    }));
    assert_eq!(body(&first_requests[1])?, body(&first_requests[2])?);
    assert!(
        body(&first_requests[6])?
            .to_string()
            .contains("CONTEXT CHECKPOINT COMPACTION")
    );
    assert!(
        body(&first_requests[7])?
            .to_string()
            .contains("stable compacted summary")
    );

    let home = root.home.clone();
    let rollout_path = root.codex.rollout_path().expect("rollout path");
    root.codex.shutdown_and_wait().await?;
    let resumed = native_builder(&server)
        .resume_with_auto_env(&server, home, rollout_path)
        .await?;
    submit(&resumed.codex, "resumed turn").await?;
    let branch = resumed
        .thread_manager
        .fork_thread(
            ForkSnapshot::Interrupted,
            resumed.config.clone(),
            resumed.codex.rollout_path().expect("resumed rollout path"),
            /*thread_source*/ None,
            /*parent_trace*/ None,
        )
        .await?;
    submit(&branch.thread, "branch turn").await?;
    let parent_thread_id = resumed.session_configured.thread_id;
    let child = resumed
        .thread_manager
        .spawn_subagent(
            parent_thread_id,
            StartThreadOptions {
                session_source: Some(SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
                    parent_thread_id,
                    depth: 1,
                    agent_path: None,
                    agent_nickname: None,
                    agent_role: Some("conformance".to_string()),
                })),
                ..StartThreadOptions::new(resumed.config.clone())
            },
        )
        .await?;
    submit(&child.thread, "subagent turn").await?;

    let distinct = native_builder(&server).build_with_auto_env(&server).await?;
    submit(&distinct.codex, "distinct root").await?;
    let requests = server.received_requests().await.expect("native requests");
    assert_eq!(requests.len(), 12);
    assert!(requests[..11].iter().all(|request| {
        body(request).is_ok_and(|body| body["prompt_cache_key"] == expected_key)
    }));
    assert_ne!(body(&requests[11])?["prompt_cache_key"], expected_key);
    assert_eq!(
        requests
            .iter()
            .filter(|request| request.method == Method::GET)
            .count(),
        0,
        "Kimi must never use Responses WebSockets"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn openai_responses_and_claude_messages_keep_complete_native_bodies() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let openai_server = responses::start_mock_server().await;
    let openai = super::claude_native_conformance::captured_responses_body(
        &openai_server,
        Arc::new(TempDir::new()?),
        "gpt-5.5",
        /*metadata_free*/ false,
    )
    .await?;

    let claude_server = responses::start_mock_server().await;
    let claude_body = [
        super::claude_dispatch::start("fixture", "claude-sonnet-5"),
        super::claude_dispatch::event("message_delta", json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}})),
        super::claude_dispatch::event("message_stop", json!({"type":"message_stop"})),
    ].concat();
    super::claude_dispatch::mount_native(&claude_server, vec![claude_body]).await;
    let claude =
        super::claude_dispatch::native_builder(&claude_server, "anthropic/claude-sonnet-5")
            .with_model_info_override("anthropic/claude-sonnet-5", |info| {
                info.shell_type = ConfigShellToolType::Disabled;
                info.apply_patch_tool_type = None;
                info.experimental_supported_tools.clear();
                info.supports_search_tool = false;
            })
            .with_config(|config| {
                config.update_plan_enabled = false;
                config.experimental_request_user_input_enabled = false;
                config.include_skill_instructions = false;
                config.include_permissions_instructions = false;
                config.include_apps_instructions = false;
                config.include_collaboration_mode_instructions = false;
                config.include_environment_context = false;
            })
            .build_with_auto_env(&claude_server)
            .await?;
    claude
        .submit_turn_with_environments("messages fixture prompt", Some(Vec::new()))
        .await?;
    let requests = claude_server
        .received_requests()
        .await
        .expect("Claude requests");
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].url.path(), "/v1/messages");
    assert_eq!(requests[0].url.query(), Some("beta=true"));
    let claude: Value = requests[0].body_json()?;
    insta::assert_snapshot!(
        "kimi_conformance_established_openai_and_claude_bodies",
        serde_json::to_string_pretty(&json!({"openai":openai,"claude":claude}))?
    );
    Ok(())
}
