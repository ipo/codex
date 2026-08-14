use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use anyhow::Result;
use codex_config::Constrained;
use codex_core::ForkSnapshot;
use codex_core::StartThreadOptions;
use codex_model_provider_info::CLAUDEFLARE_PROVIDER_ID;
use codex_model_provider_info::built_in_model_providers;
use codex_protocol::config_types::WebSearchMode;
use codex_protocol::models::PermissionProfile;
use codex_protocol::openai_models::ConfigShellToolType;
use codex_protocol::protocol::AskForApproval;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::protocol::SessionSource;
use codex_protocol::protocol::SubAgentSource;
use codex_protocol::user_input::UserInput;
use core_test_support::responses;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::TestCodexBuilder;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use core_test_support::wait_for_mcp_server;
use pretty_assertions::assert_eq;
use serde_json::Map;
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
use wiremock::matchers::query_param;

const CLAUDE_MODEL: &str = "claude-sonnet-5";
const CLAUDE_PROFILE: &str = "anthropic/claude-sonnet-5";
const SESSION_HEADER: &str = "x-claude-code-session-id";

fn native_event(name: &str, data: Value) -> String {
    format!("event: {name}\ndata: {data}\n\n")
}

fn native_start(id: &str) -> String {
    native_event(
        "message_start",
        json!({"type":"message_start","message":{"id":id,"type":"message","role":"assistant","model":CLAUDE_MODEL,"content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":0}}}),
    )
}

fn native_text(id: &str, text: &str, stop_reason: &str) -> String {
    [
        native_start(id),
        native_event("content_block_start", json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}})),
        native_event("content_block_delta", json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":text}})),
        native_event("content_block_stop", json!({"type":"content_block_stop","index":0})),
        native_event("message_delta", json!({"type":"message_delta","delta":{"stop_reason":stop_reason},"usage":{"output_tokens":2}})),
        native_event("message_stop", json!({"type":"message_stop"})),
    ]
    .concat()
}

fn native_parallel_tools() -> String {
    [
        native_start("parallel-tools"),
        native_event("content_block_start", json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"parallel-one","name":"exec_command","input":{}}})),
        native_event("content_block_delta", json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"cmd\":\"pwd\",\"yield_time_ms\":1000,\"max_output_tokens\":1000}"}})),
        native_event("content_block_stop", json!({"type":"content_block_stop","index":0})),
        native_event("content_block_start", json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"parallel-two","name":"exec_command","input":{}}})),
        native_event("content_block_delta", json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"cmd\":\"pwd\",\"yield_time_ms\":1000,\"max_output_tokens\":1000}"}})),
        native_event("content_block_stop", json!({"type":"content_block_stop","index":1})),
        native_event("message_delta", json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":2}})),
        native_event("message_stop", json!({"type":"message_stop"})),
    ]
    .concat()
}

fn native_tool_call(id: &str, name: &str, arguments: Value) -> String {
    [
        native_start(id),
        native_event(
            "content_block_start",
            json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":id,"name":name,"input":{}}}),
        ),
        native_event(
            "content_block_delta",
            json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":arguments.to_string()}}),
        ),
        native_event(
            "content_block_stop",
            json!({"type":"content_block_stop","index":0}),
        ),
        native_event(
            "message_delta",
            json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":2}}),
        ),
        native_event("message_stop", json!({"type":"message_stop"})),
    ]
    .concat()
}

fn native_request_tool_names(body: &Value) -> Vec<&str> {
    body["tools"]
        .as_array()
        .expect("Claude tools")
        .iter()
        .filter_map(|tool| tool["name"].as_str())
        .collect()
}

fn native_response(body: String) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_string(body)
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

async fn mount_native(server: &MockServer, responses: Vec<ResponseTemplate>) {
    let count = responses.len() as u64;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(query_param("beta", "true"))
        .respond_with(NativeSequence {
            next: AtomicUsize::new(0),
            responses,
        })
        .up_to_n_times(count)
        .mount(server)
        .await;
}

fn native_builder(server: &MockServer) -> TestCodexBuilder {
    let base_url = server.uri();
    test_codex().with_config(move |config| {
        let mut provider = built_in_model_providers(/*openai_base_url*/ None)
            .remove(CLAUDEFLARE_PROVIDER_ID)
            .expect("managed Claudeflare provider");
        provider.base_url = Some(base_url.clone());
        provider.supports_websockets = true;
        provider
            .wire_routes
            .get_mut("claude_code")
            .expect("Claude route")
            .base_url = base_url;
        provider
            .wire_routes
            .get_mut("claude_code")
            .expect("Claude route")
            .stream_max_retries = Some(1);
        config.model_provider = provider;
        config.model = Some(CLAUDE_PROFILE.to_string());
        config.base_instructions = Some("conformance system".to_string());
        config.agents_enabled = false;
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

fn request_body(request: &Request) -> Result<Value> {
    Ok(request.body_json()?)
}

fn session_id(request: &Request) -> &str {
    request.headers[SESSION_HEADER]
        .to_str()
        .expect("native session header")
}

fn native_identity(request: &Request) -> Result<Value> {
    let body = request_body(request)?;
    Ok(json!({
        "method": request.method.as_str(),
        "path": request.url.path(),
        "query": request.url.query(),
        "session": session_id(request),
        "user_agent": request.headers["user-agent"].to_str()?,
        "x_app": request.headers["x-app"].to_str()?,
        "anthropic_version": request.headers["anthropic-version"].to_str()?,
        "anthropic_beta": request.headers["anthropic-beta"].to_str()?,
        "model": body["model"],
        "max_tokens": body["max_tokens"],
        "thinking": body["thinking"],
        "output_config": body["output_config"],
    }))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_session_lineage_is_exact_through_recovery_resume_and_subagent() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    mount_native(
        &server,
        vec![
            native_response(native_text("ordinary", "ordinary answer", "end_turn")),
            ResponseTemplate::new(503).insert_header("retry-after-ms", "0"),
            native_response(native_text("retried", "retry answer", "end_turn")),
            native_response(native_parallel_tools()),
            native_response(native_text("tools-done", "tools complete", "end_turn")),
            ResponseTemplate::new(400).set_body_string(r#"{"error":{"type":"invalid_request_error","message":"prompt is too long for the context window"}}"#),
            native_response(native_text("summary", "stable compacted summary", "end_turn")),
            native_response(native_text("compacted", "compaction complete", "end_turn")),
            native_response(native_text("resumed", "resume complete", "end_turn")),
            native_response(native_text("subagent", "subagent complete", "end_turn")),
            native_response(native_text("distinct", "distinct complete", "end_turn")),
        ],
    )
    .await;

    let root = native_builder(&server).build_with_auto_env(&server).await?;
    root.submit_turn("ordinary turn").await?;
    root.submit_turn("retry turn").await?;
    root.submit_turn("parallel tool turn").await?;
    root.submit_turn("overflow turn").await?;

    let first_requests = server.received_requests().await.expect("native requests");
    assert_eq!(first_requests.len(), 8);
    let resumable_session = root.session_configured.session_id.to_string();
    let expected_session = session_id(&first_requests[0]).to_string();
    uuid::Uuid::parse_str(&expected_session)?;
    assert_ne!(expected_session, resumable_session);
    assert!(
        first_requests
            .iter()
            .all(|request| session_id(request) == expected_session)
    );
    let identity = native_identity(&first_requests[0])?;
    assert!(first_requests.iter().all(|request| {
        native_identity(request)
            .as_ref()
            .is_ok_and(|value| value == &identity)
    }));
    assert_eq!(
        request_body(&first_requests[1])?,
        request_body(&first_requests[2])?
    );
    let tool_continuation = request_body(&first_requests[4])?.to_string();
    for expected in ["parallel-one", "parallel-two", "tool_result"] {
        assert!(tool_continuation.contains(expected), "missing {expected}");
    }
    let compact = request_body(&first_requests[6])?.to_string();
    assert!(compact.contains("CONTEXT CHECKPOINT COMPACTION"));
    assert!(compact.contains("overflow turn"));
    let overflow = request_body(&first_requests[5])?;
    let overflow_messages = overflow["messages"].as_array().expect("overflow messages");
    let compact_body = request_body(&first_requests[6])?;
    let compact_messages = compact_body["messages"]
        .as_array()
        .expect("compact messages");
    assert!(compact_messages.starts_with(&overflow_messages[..overflow_messages.len() - 1]));
    let compacted_retry = request_body(&first_requests[7])?.to_string();
    assert!(compacted_retry.contains("stable compacted summary"));
    assert!(compacted_retry.contains("overflow turn"));
    for request in &first_requests {
        assert_eq!(request_body(request)?["model"], CLAUDE_MODEL);
    }

    let home = root.home.clone();
    let rollout_path = root.codex.rollout_path().expect("rollout path");
    root.codex.shutdown_and_wait().await?;
    let resumed = native_builder(&server)
        .resume_with_auto_env(&server, home, rollout_path)
        .await?;
    resumed.submit_turn("resumed turn").await?;

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
    assert_eq!(
        child.session_configured.session_id,
        resumed.session_configured.session_id
    );
    submit(&child.thread, "subagent turn").await?;

    let distinct = native_builder(&server).build_with_auto_env(&server).await?;
    distinct.submit_turn("distinct root turn").await?;
    let requests = server.received_requests().await.expect("native requests");
    assert_eq!(requests.len(), 11);
    assert_eq!(session_id(&requests[8]), expected_session);
    assert_eq!(session_id(&requests[9]), expected_session);
    assert_ne!(session_id(&requests[10]), expected_session);
    assert!(requests[..10].iter().all(|request| {
        native_identity(request)
            .as_ref()
            .is_ok_and(|value| value == &identity)
    }));
    assert!(
        request_body(&requests[8])?
            .to_string()
            .contains("compaction complete")
    );
    assert!(
        request_body(&requests[9])?
            .to_string()
            .contains("resumed turn")
    );
    assert_eq!(
        requests
            .iter()
            .filter(|request| request.method == Method::GET)
            .count(),
        0,
        "Claude must not prewarm, attempt, or fall back to Responses WebSockets"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn public_fork_preserves_claude_native_session_header() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    mount_native(
        &server,
        vec![
            native_response(native_text("root", "root complete", "end_turn")),
            native_response(native_text("branch", "branch complete", "end_turn")),
            native_response(native_text("distinct", "distinct complete", "end_turn")),
        ],
    )
    .await;

    let root = native_builder(&server).build_with_auto_env(&server).await?;
    root.submit_turn("root turn").await?;
    let root_session_id = root.session_configured.session_id;
    let root_thread_id = root.session_configured.thread_id;
    let home = root.home.clone();
    let rollout_path = root.codex.rollout_path().expect("root rollout path");
    root.codex.shutdown_and_wait().await?;

    let resumed = native_builder(&server)
        .resume_with_auto_env(&server, home, rollout_path.clone())
        .await?;
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
    submit(&branch.thread, "branch turn").await?;

    let distinct = native_builder(&server).build_with_auto_env(&server).await?;
    assert_ne!(distinct.session_configured.session_id, root_session_id);
    distinct.submit_turn("distinct root turn").await?;

    let requests = server.received_requests().await.expect("native requests");
    assert_eq!(requests.len(), 3);
    let root_native_session = session_id(&requests[0]);
    uuid::Uuid::parse_str(root_native_session)?;
    assert_eq!(session_id(&requests[1]), root_native_session);
    assert_ne!(session_id(&requests[2]), root_native_session);
    Ok(())
}

fn contains_delta(events: &[EventMsg], expected: &str) -> bool {
    events.iter().any(|event| {
        matches!(event, EventMsg::AgentMessageContentDelta(delta) if delta.delta == expected)
    })
}

fn error_message(events: &[EventMsg]) -> Option<&str> {
    events.iter().find_map(|event| match event {
        EventMsg::Error(error) => Some(error.message.as_str()),
        _ => None,
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_terminal_matrix_commits_discards_retries_and_reuses_thread() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    let protocol_failure = [
        native_start("protocol"),
        native_event("content_block_start", json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}})),
        native_event("content_block_delta", json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"discard protocol"}})),
        native_event("content_block_stop", json!({"type":"content_block_stop","index":0})),
        native_event("message_delta", json!({"type":"message_delta","delta":{"stop_reason":"future_reason"},"usage":{"output_tokens":2}})),
    ]
    .concat();
    mount_native(
        &server,
        vec![
            native_response(native_text("normal", "normal visible", "end_turn")),
            native_response(native_text("stop", "stop visible", "stop_sequence")),
            native_response(native_parallel_tools()),
            native_response(native_text("pause", "pause visible", "pause_turn")),
            native_response(native_text("tool-final", "tool final", "end_turn")),
            native_response(native_text("exhausted", "discard exhausted", "max_tokens")),
            native_response(native_text(
                "after-exhaustion",
                "after exhausted",
                "end_turn",
            )),
            native_response(native_text("refused", "discard refusal", "refusal")),
            native_response(native_text("after-refusal", "after refused", "end_turn")),
            ResponseTemplate::new(503).insert_header("retry-after-ms", "0"),
            native_response(native_text("retry-success", "retry visible", "end_turn")),
            native_response(protocol_failure),
            native_response(native_text("after-protocol", "after protocol", "end_turn")),
        ],
    )
    .await;
    let test = native_builder(&server).build_with_auto_env(&server).await?;

    let normal = submit(&test.codex, "normal").await?;
    assert!(contains_delta(&normal, "normal visible"));
    let stopped = submit(&test.codex, "stop sequence").await?;
    assert!(contains_delta(&stopped, "stop visible"));
    let tools = submit(&test.codex, "tools and pause").await?;
    assert!(contains_delta(&tools, "pause visible"));
    assert!(contains_delta(&tools, "tool final"));
    let executed_calls = tools
        .iter()
        .filter_map(|event| match event {
            EventMsg::ExecCommandBegin(event) => Some(event.call_id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(executed_calls, ["parallel-one", "parallel-two"]);

    let exhausted = submit(&test.codex, "exhaust output").await?;
    assert!(contains_delta(&exhausted, "discard exhausted"));
    assert_eq!(
        error_message(&exhausted),
        Some("model output limit reached before the turn completed")
    );
    submit(&test.codex, "after exhaustion").await?;

    let refused = submit(&test.codex, "refuse safely").await?;
    assert!(contains_delta(&refused, "discard refusal"));
    assert_eq!(
        error_message(&refused),
        Some("model refused to complete the turn for safety reasons")
    );
    submit(&test.codex, "after refusal").await?;
    submit(&test.codex, "retry failure").await?;

    let protocol = submit(&test.codex, "protocol failure").await?;
    assert!(contains_delta(&protocol, "discard protocol"));
    assert!(
        error_message(&protocol)
            .is_some_and(|message| message.contains("unknown native Claude stop reason"))
    );
    submit(&test.codex, "after protocol").await?;

    let requests = server.received_requests().await.expect("native requests");
    assert_eq!(requests.len(), 13);
    let bodies = requests
        .iter()
        .map(request_body)
        .collect::<Result<Vec<_>>>()?;
    assert!(bodies[2].to_string().contains("normal visible"));
    assert!(bodies[2].to_string().contains("stop visible"));
    let tool_continuation = bodies[3].to_string();
    for expected in ["parallel-one", "parallel-two", "tool_result"] {
        assert!(tool_continuation.contains(expected));
    }
    assert!(bodies[4].to_string().contains("pause visible"));
    assert!(!bodies[6].to_string().contains("discard exhausted"));
    assert!(!bodies[8].to_string().contains("discard refusal"));
    assert_eq!(bodies[9], bodies[10]);
    assert!(!bodies[12].to_string().contains("discard protocol"));
    assert!(
        requests
            .iter()
            .all(|request| session_id(request) == session_id(&requests[0]))
    );
    Ok(())
}

pub(super) async fn captured_responses_body(
    server: &MockServer,
    home: Arc<TempDir>,
    model: &str,
    metadata_free: bool,
) -> Result<Value> {
    let mock = responses::mount_sse_once(
        server,
        responses::sse(vec![
            responses::ev_response_created("fixture-response"),
            responses::ev_assistant_message("fixture-message", "fixture answer"),
            responses::ev_completed("fixture-response"),
        ]),
    )
    .await;
    let home_path = home.path().display().to_string();
    let builder = test_codex()
        .with_home(home)
        .with_model_info_override(model, move |info| {
            if metadata_free {
                info.inference = None;
            }
            info.shell_type = ConfigShellToolType::Disabled;
            info.apply_patch_tool_type = None;
            info.experimental_supported_tools.clear();
            info.supports_search_tool = false;
        });
    let test = builder
        .with_config(|config| {
            config.base_instructions = Some("responses fixture system".to_string());
            config.agents_enabled = false;
            config.update_plan_enabled = false;
            config.experimental_request_user_input_enabled = false;
            config.include_skill_instructions = false;
            config.include_permissions_instructions = false;
            config.include_apps_instructions = false;
            config.include_collaboration_mode_instructions = false;
            config.include_environment_context = false;
            config
                .web_search_mode
                .set(WebSearchMode::Disabled)
                .expect("disable web search for stable fixture");
        })
        .build_with_auto_env(server)
        .await?;
    test.submit_turn_with_environments("responses fixture prompt", Some(Vec::new()))
        .await?;
    let request = mock.single_request();
    assert_eq!(request.path(), "/v1/responses");
    Ok(normalize_responses_body(request.body_json(), &home_path))
}

fn normalize_responses_body(value: Value, home_path: &str) -> Value {
    match value {
        Value::Array(values) => Value::Array(
            values
                .into_iter()
                .map(|value| normalize_responses_body(value, home_path))
                .collect(),
        ),
        Value::Object(values) => Value::Object(
            values
                .into_iter()
                .map(|(key, value)| {
                    let value = if matches!(
                        key.as_str(),
                        "prompt_cache_key"
                            | "session_id"
                            | "thread_id"
                            | "turn_id"
                            | "x-codex-installation-id"
                            | "x-codex-turn-metadata"
                            | "x-codex-window-id"
                            | "id"
                    ) {
                        Value::String(format!("<{key}>"))
                    } else {
                        normalize_responses_body(value, home_path)
                    };
                    (key, value)
                })
                .collect::<Map<_, _>>(),
        ),
        Value::String(value) => Value::String(value.replace(home_path, "<codex_home>")),
        value => value,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_claude_searches_loads_and_dispatches_flattened_mcp_tool() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    mount_native(
        &server,
        vec![
            native_response(native_tool_call(
                "search-call",
                "tool_search",
                json!({"query":"echo message and environment data","limit":20}),
            )),
            native_response(native_tool_call(
                "echo-call",
                "mcp__rmcp__echo",
                json!({"message":"ping"}),
            )),
            native_response(native_text("done", "done", "end_turn")),
        ],
    )
    .await;
    let command = super::rmcp_client::remote_aware_stdio_server_bin()?;
    let test = native_builder(&server)
        .with_config(move |config| {
            super::rmcp_client::configure_stdio_mcp(config, "rmcp", command);
        })
        .build_with_auto_env(&server)
        .await?;
    wait_for_mcp_server(&test.codex, "rmcp").await?;

    submit(&test.codex, "find and call the echo tool").await?;

    let requests = server.received_requests().await.expect("native requests");
    let bodies = requests
        .iter()
        .map(request_body)
        .collect::<Result<Vec<_>>>()?;
    assert!(native_request_tool_names(&bodies[0]).contains(&"tool_search"));
    assert!(!native_request_tool_names(&bodies[0]).contains(&"mcp__rmcp__echo"));
    assert!(native_request_tool_names(&bodies[1]).contains(&"mcp__rmcp__echo"));
    assert!(bodies[1].to_string().contains("\"tools\""));
    assert!(bodies[2].to_string().contains("ECHOING: ping"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_claude_executes_function_form_apply_patch() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    let patch = "*** Begin Patch\n*** Add File: native-claude.txt\n+patched\n*** End Patch";
    mount_native(
        &server,
        vec![
            native_response(native_tool_call(
                "patch-call",
                "apply_patch",
                json!({"patch":patch}),
            )),
            native_response(native_text("done", "done", "end_turn")),
        ],
    )
    .await;
    let workspace = TempDir::new_in(std::env::current_dir()?)?;
    let cwd = codex_utils_absolute_path::AbsolutePathBuf::try_from(workspace.path().to_path_buf())?;
    let test = native_builder(&server)
        .with_config(move |config| {
            config.cwd = cwd;
            config.permissions.approval_policy = Constrained::allow_any(AskForApproval::Never);
            config
                .permissions
                .set_permission_profile(PermissionProfile::Disabled)
                .expect("test config should allow disabled permissions");
        })
        .build_with_auto_env(&server)
        .await?;

    let events = submit(&test.codex, "apply the patch").await?;
    let requests = server.received_requests().await.expect("native requests");
    let tool_result_request = request_body(&requests[1])?;
    let patched_path = workspace.path().join("native-claude.txt");
    assert!(
        patched_path.exists(),
        "apply_patch did not create {patched_path:?}; events: {events:#?}; follow-up request: {tool_result_request:#}"
    );

    assert_eq!(std::fs::read_to_string(patched_path)?, "patched\n");
    let first = request_body(&requests[0])?;
    let apply_patch = first["tools"]
        .as_array()
        .expect("Claude tools")
        .iter()
        .find(|tool| tool["name"] == "apply_patch")
        .expect("apply_patch function");
    assert_eq!(apply_patch["input_schema"]["required"], json!(["patch"]));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn openai_and_metadata_free_responses_bodies_match_established_fixture() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let home = Arc::new(TempDir::new()?);
    let openai_server = responses::start_mock_server().await;
    let openai = captured_responses_body(&openai_server, home.clone(), "gpt-5.5", false).await?;
    let metadata_free_server = responses::start_mock_server().await;
    let metadata_free =
        captured_responses_body(&metadata_free_server, home, "gpt-5.5", true).await?;
    assert_eq!(metadata_free, openai);
    insta::assert_snapshot!(
        "claude_conformance_openai_responses_body",
        serde_json::to_string_pretty(&openai)?
    );

    Ok(())
}
