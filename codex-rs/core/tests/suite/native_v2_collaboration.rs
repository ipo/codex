use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::Result;
use codex_exec_server::CreateDirectoryOptions;
use codex_features::Feature;
use codex_model_provider_info::CLAUDEFLARE_PROVIDER_ID;
use codex_model_provider_info::built_in_model_providers;
use codex_protocol::ThreadId;
use codex_protocol::openai_models::ReasoningEffort;
use codex_protocol::protocol::AgentStatus;
use codex_utils_path_uri::LegacyAppPathString;
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
const OPUS_MODEL: &str = "claude-opus-5";
const OPUS_PROFILE: &str = "anthropic/claude-opus-5";
const OPUS_BETAS: &str = "claude-code-20250219,context-1m-2025-08-07,interleaved-thinking-2025-05-14,redact-thinking-2026-02-12,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,advisor-tool-2026-03-01,effort-2025-11-24,fallback-credit-2026-06-01";

#[derive(Clone, Copy)]
enum NativeParent {
    Claude,
    Kimi,
    Grok,
}

impl NativeParent {
    fn model(self) -> &'static str {
        match self {
            Self::Claude => "anthropic/claude-haiku-4-5-20251001",
            Self::Kimi => "kimi/kimi-for-coding",
            Self::Grok => "xai/grok-4.6",
        }
    }

    fn path(self) -> &'static str {
        match self {
            Self::Claude => "/v1/messages",
            Self::Kimi => "/v1/kimi/chat/completions",
            Self::Grok => "/v1/grok/responses",
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
            Self::Grok => responses::sse(vec![
                ev_response_created("grok-spawn"),
                responses::ev_function_call(SPAWN_CALL_ID, "spawn_agent", arguments),
                ev_completed("grok-spawn"),
            ]),
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
            Self::Grok => responses::sse(vec![
                ev_response_created("grok-final"),
                ev_assistant_message("grok-final-message", "spawned"),
                ev_completed("grok-final"),
            ]),
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

struct GrokSameFamilySequence {
    spawn_arguments: String,
    child_task: &'static str,
}

impl Respond for GrokSameFamilySequence {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body = String::from_utf8_lossy(&request.body);
        let events = if body.contains("function_call_output") {
            vec![
                ev_response_created("grok-root-complete"),
                ev_assistant_message("grok-root-message", "spawned"),
                ev_completed("grok-root-complete"),
            ]
        } else if body.contains(self.child_task) {
            vec![
                ev_response_created("grok-child"),
                ev_assistant_message("grok-child-message", "grok child complete"),
                ev_completed("grok-child"),
            ]
        } else {
            vec![
                ev_response_created("grok-root-spawn"),
                responses::ev_function_call(SPAWN_CALL_ID, "spawn_agent", &self.spawn_arguments),
                ev_completed("grok-root-spawn"),
            ]
        };
        ResponseTemplate::new(200)
            .insert_header("content-type", "text/event-stream")
            .set_body_string(responses::sse(events))
    }
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
        provider
            .wire_routes
            .get_mut("grok")
            .expect("Grok route")
            .base_url = format!("{base_url}/v1/grok");
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
        config.multi_agent_v2.tool_namespace = None;
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
        NativeParent::Grok => tool["name"] == "spawn_agent",
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

struct OpusV2Sequence {
    root_request_count: AtomicUsize,
    spawn_arguments: String,
}

impl Respond for OpusV2Sequence {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body = if request.headers.contains_key("x-claude-code-agent-id") {
            opus_text("msg-opus-child", "opus child complete")
        } else if self.root_request_count.fetch_add(1, Ordering::SeqCst) == 0 {
            opus_spawn(&self.spawn_arguments)
        } else {
            opus_text("msg-opus-root", "opus root complete")
        };
        ResponseTemplate::new(200)
            .insert_header("content-type", "text/event-stream")
            .set_body_string(body)
    }
}

fn opus_start(id: &str) -> String {
    claude_event(
        "message_start",
        json!({"type":"message_start","message":{"id":id,"type":"message","role":"assistant","model":OPUS_MODEL,"content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":0}}}),
    )
}

fn opus_text(id: &str, text: &str) -> String {
    [
        opus_start(id),
        claude_event("content_block_start", json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}})),
        claude_event("content_block_delta", json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":text}})),
        claude_event("content_block_stop", json!({"type":"content_block_stop","index":0})),
        claude_event("message_delta", json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}})),
        claude_event("message_stop", json!({"type":"message_stop"})),
    ]
    .concat()
}

fn opus_spawn(arguments: &str) -> String {
    [
        opus_start("msg-opus-spawn"),
        claude_event("content_block_start", json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":SPAWN_CALL_ID,"name":"spawn_agent","input":{}}})),
        claude_event("content_block_delta", json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":arguments}})),
        claude_event("content_block_stop", json!({"type":"content_block_stop","index":0})),
        claude_event("message_delta", json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":2}})),
        claude_event("message_stop", json!({"type":"message_stop"})),
    ]
    .concat()
}

fn opus_builder(server: &MockServer) -> TestCodexBuilder {
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
            .base_url = base_url;
        config.model_provider = provider;
        config.model = Some(OPUS_PROFILE.to_string());
        let cwd = config.codex_home.join("opus-cwd");
        std::fs::create_dir_all(&cwd).expect("create isolated Opus cwd");
        config.cwd = cwd;
        config.agent_default_subagent_model = Some(OPUS_PROFILE.to_string());
        config.agent_default_subagent_reasoning_effort = Some(ReasoningEffort::Medium);
        config.base_instructions = Some("Codex root instructions must be replaced".to_string());
        config
            .features
            .enable(Feature::Collab)
            .expect("collaboration feature should be enableable");
        config
            .features
            .enable(Feature::MultiAgentV2)
            .expect("multi-agent V2 feature should be enableable");
        config.include_skill_instructions = false;
        config.include_permissions_instructions = false;
        config.include_apps_instructions = false;
        config.include_environment_context = false;
    })
}

fn assert_opus_identity(
    request: &Request,
    expected_session: &str,
    expected_platform: &str,
    expected_architecture: &str,
    is_subagent: bool,
) -> Result<()> {
    assert_eq!(request.url.path(), "/v1/messages");
    assert_eq!(request.url.query(), Some("beta=true"));
    assert_eq!(
        request.headers["user-agent"],
        "claude-cli/2.1.224 (external, cli)"
    );
    assert_eq!(request.headers["accept"], "application/json");
    assert_eq!(request.headers["x-app"], "cli");
    assert_eq!(request.headers["anthropic-beta"], OPUS_BETAS);
    assert_eq!(request.headers["anthropic-version"], "2023-06-01");
    assert_eq!(
        request.headers["anthropic-dangerous-direct-browser-access"],
        "true"
    );
    assert_eq!(
        request.headers["x-claude-code-session-id"],
        expected_session
    );
    assert_eq!(request.headers["x-stainless-lang"], "js");
    assert_eq!(request.headers["x-stainless-os"], expected_platform);
    assert_eq!(request.headers["x-stainless-arch"], expected_architecture);
    assert_eq!(request.headers["x-stainless-package-version"], "0.94.0");
    assert_eq!(request.headers["x-stainless-runtime"], "node");
    assert_eq!(request.headers["x-stainless-runtime-version"], "v26.3.0");
    assert_eq!(request.headers["x-stainless-timeout"], "600");
    assert_eq!(request.headers["x-stainless-retry-count"], "0");
    assert!(!request.headers.contains_key("originator"));
    assert!(request.headers.contains_key("authorization"));
    assert!(!request.headers.contains_key("chatgpt-account-id"));
    let agent_id = request
        .headers
        .get("x-claude-code-agent-id")
        .map(|value| value.to_str())
        .transpose()?;
    if is_subagent {
        let agent_id = agent_id.expect("subagent identity header");
        assert_eq!(agent_id.len(), 17);
        assert!(agent_id.starts_with('a'));
        assert!(
            agent_id[1..]
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        );
    } else {
        assert_eq!(agent_id, None);
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn opus_root_and_public_v2_subagent_use_claude_code_compatibility_profile() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    let spawn_arguments = serde_json::to_string(&json!({
        "plaintext_message": CHILD_TASK,
        "task_name": "worker",
        "fork_turns": "none",
    }))?;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(OpusV2Sequence {
            root_request_count: AtomicUsize::new(0),
            spawn_arguments,
        })
        .up_to_n_times(3)
        .mount(&server)
        .await;

    let test = opus_builder(&server).build_with_auto_env(&server).await?;
    let selected_cwd = test
        .codex
        .environment_selections()
        .await
        .into_iter()
        .next()
        .expect("selected execution environment")
        .cwd;
    test.fs()
        .create_directory(
            &selected_cwd.join(".git")?,
            CreateDirectoryOptions { recursive: true },
            /*sandbox*/ None,
        )
        .await?;
    let target_info = test.executor_environment().environment().info().await?;
    let target_system = target_info
        .system
        .expect("current exec server should report target system facts");
    let expected_cwd = LegacyAppPathString::from_path_uri(
        &selected_cwd,
        target_system.operating_system.path_convention(),
    )?;
    let expected_shell = if test.executor_environment().environment().is_remote() {
        target_info.shell.path
    } else {
        std::env::var(if cfg!(windows) { "COMSPEC" } else { "SHELL" })
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| if cfg!(windows) { "cmd" } else { "sh" }.to_string())
    };
    test.submit_turn("spawn an Opus worker").await?;
    let requests = wait_for_request_count(&server, /*expected*/ 3).await?;
    let root_requests = requests
        .iter()
        .filter(|request| !request.headers.contains_key("x-claude-code-agent-id"))
        .collect::<Vec<_>>();
    let child_request = requests
        .iter()
        .find(|request| request.headers.contains_key("x-claude-code-agent-id"))
        .expect("public V2 Opus child request");
    assert_eq!(root_requests.len(), 2);
    let expected_session = test.session_configured.session_id.to_string();
    let expected_stainless_platform = match target_system.operating_system.platform() {
        "linux" => "Linux",
        "macos" => "MacOS",
        "windows" => "Windows",
        platform => platform,
    };
    let expected_stainless_architecture = match target_system.architecture.as_str() {
        "x86_64" => "x64",
        "aarch64" => "arm64",
        architecture => architecture,
    };
    assert_opus_identity(
        root_requests[0],
        &expected_session,
        expected_stainless_platform,
        expected_stainless_architecture,
        false,
    )?;
    assert_opus_identity(
        child_request,
        &expected_session,
        expected_stainless_platform,
        expected_stainless_architecture,
        true,
    )?;

    let root_body: Value = root_requests[0].body_json()?;
    let child_body: Value = child_request.body_json()?;
    for body in [&root_body, &child_body] {
        assert_eq!(body["model"], OPUS_MODEL);
        assert_eq!(body["max_tokens"], 64_000);
        assert_eq!(body["thinking"], json!({"type": "adaptive"}));
        assert_eq!(body["output_config"], json!({"effort": "medium"}));
        assert_eq!(
            body["context_management"],
            json!({"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]})
        );
        assert_eq!(
            body["system"][1]["cache_control"],
            json!({"type": "ephemeral"})
        );
        assert_eq!(
            body["system"][2]["cache_control"],
            json!({"type": "ephemeral"})
        );
        let latest_user = body["messages"]
            .as_array()
            .and_then(|messages| {
                messages
                    .iter()
                    .rev()
                    .find(|message| message["role"] == "user")
            })
            .expect("latest user message");
        let latest_user_cache = latest_user["content"]
            .as_array()
            .and_then(|content| {
                content
                    .iter()
                    .rev()
                    .find_map(|block| block.get("cache_control"))
            })
            .expect("latest eligible user cache marker");
        assert_eq!(latest_user_cache, &json!({"type": "ephemeral"}));
        assert!(body["tools"].as_array().is_some_and(|tools| {
            tools
                .iter()
                .all(|tool| tool["name"].is_string() && tool["input_schema"].is_object())
        }));
        let compatibility_prompt = body["system"][2]["text"]
            .as_str()
            .expect("Opus compatibility prompt");
        assert!(compatibility_prompt.contains(expected_cwd.as_str()));
        assert!(compatibility_prompt.contains(&format!(
            "Platform: {}",
            target_system.operating_system.platform()
        )));
        assert!(compatibility_prompt.contains(&format!("Shell: {expected_shell}")));
        assert!(
            compatibility_prompt.contains(&format!("OS Version: {}", target_system.os_version))
        );
        assert!(
            !body
                .to_string()
                .contains("Codex root instructions must be replaced")
        );
        let metadata: Value = serde_json::from_str(
            body["metadata"]["user_id"]
                .as_str()
                .expect("metadata user identity"),
        )?;
        assert_eq!(metadata["device_id"].as_str().map(str::len), Some(64));
        assert_eq!(metadata["account_uuid"], "");
        assert_eq!(metadata["session_id"], expected_session);
    }
    assert!(
        root_body["system"][0]["text"]
            .as_str()
            .is_some_and(|text| !text.contains("cc_is_subagent=true"))
    );
    assert!(root_body["system"][2]["text"].as_str().is_some_and(|text| {
        text.contains("interactive agent") && text.contains("Is a git repository: true")
    }));
    assert!(
        child_body["system"][0]["text"]
            .as_str()
            .is_some_and(|text| text.contains("cc_is_subagent=true"))
    );
    assert!(
        child_body["system"][2]["text"]
            .as_str()
            .is_some_and(|text| text.contains("agent for Claude Code")
                && text.contains("Is directory a git repo: Yes"))
    );
    assert!(child_body["tools"].as_array().is_some_and(|tools| {
        tools
            .iter()
            .any(|tool| matches!(tool["name"].as_str(), Some("exec_command" | "spawn_agent")))
    }));
    assert!(root_body.to_string().contains("spawn an Opus worker"));
    assert!(child_body.to_string().contains(CHILD_TASK));

    let root_thread_id = test.session_configured.thread_id;
    let child_thread_id = test
        .thread_manager
        .list_thread_ids()
        .await
        .into_iter()
        .find(|thread_id| *thread_id != root_thread_id)
        .expect("spawned Opus child thread");
    let child = test.thread_manager.get_thread(child_thread_id).await?;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    loop {
        match child.agent_status().await {
            AgentStatus::Completed(message) => {
                assert_eq!(message.as_deref(), Some("opus child complete"));
                break;
            }
            AgentStatus::Errored(error) => anyhow::bail!("Opus child errored: {error}"),
            status if tokio::time::Instant::now() >= deadline => {
                anyhow::bail!("timed out waiting for Opus child completion: {status:?}")
            }
            AgentStatus::PendingInit
            | AgentStatus::Running
            | AgentStatus::Interrupted
            | AgentStatus::Shutdown
            | AgentStatus::NotFound => tokio::time::sleep(Duration::from_millis(10)).await,
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_parents_route_plain_v2_spawn_agent_to_openai_child() -> Result<()> {
    skip_if_no_network!(Ok(()));
    for parent in [NativeParent::Claude, NativeParent::Kimi, NativeParent::Grok] {
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
        let mut created_threads = test.thread_manager.subscribe_thread_created();
        test.submit_turn("spawn a worker").await?;
        let child_thread_id =
            tokio::time::timeout(Duration::from_secs(10), created_threads.recv()).await??;
        wait_for_child_completion(&test, child_thread_id, "acknowledged").await?;

        let requests = wait_for_request_count(&server, /*expected*/ 3).await?;
        let native_requests = requests
            .iter()
            .filter(|request| request.url.path() == parent.path())
            .collect::<Vec<_>>();
        assert_eq!(native_requests.len(), 2);
        if !matches!(parent, NativeParent::Grok) {
            assert!(native_requests[0].headers.contains_key("authorization"));
        }
        let initial: Value = native_requests[0].body_json()?;
        let initial_text = initial.to_string();
        assert!(initial_text.contains("using the form shown in their tool definitions"));
        assert!(!initial_text.contains("functions.agents.spawn_agent"));
        let spawn_agent = plain_spawn_agent_tool(&initial, parent)
            .expect("native parent should receive plain spawn_agent function");
        assert!(!spawn_agent.to_string().contains("namespace"));
        let continuation: Value = native_requests[1].body_json()?;
        let continuation_messages = continuation
            .get("messages")
            .or_else(|| continuation.get("input"))
            .expect("continuation history")
            .to_string();
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

    let test = native_parent_builder(&server, NativeParent::Grok)
        .build_with_auto_env(&server)
        .await?;
    let mut created_threads = test.thread_manager.subscribe_thread_created();
    test.submit_turn(root_prompt).await?;
    let child_thread_id =
        tokio::time::timeout(Duration::from_secs(10), created_threads.recv()).await??;
    wait_for_child_completion(&test, child_thread_id, "grok child complete").await?;

    let requests = wait_for_request_count(&server, /*expected*/ 3).await?;
    let child_request = requests
        .iter()
        .find(|request| String::from_utf8_lossy(&request.body).contains(child_task))
        .expect("Grok child request");
    let child_body: Value = child_request.body_json()?;
    assert_eq!(child_body["model"], "grok-4.6");
    assert!(child_body.to_string().contains(root_prompt));
    assert!(child_request.headers.contains_key("x-grok-conv-id"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn claude_root_routes_plaintext_task_to_grok_child() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    let arguments = serde_json::to_string(&json!({
        "plaintext_message": CHILD_TASK,
        "task_name": "grok_worker",
        "model": "xai/grok-4.6",
        "fork_turns": "none",
    }))?;
    mount_native_parent(&server, NativeParent::Claude, &arguments).await;
    responses::mount_sse_once(
        &server,
        responses::sse(vec![
            ev_response_created("grok-child"),
            ev_assistant_message("grok-child-message", "grok child complete"),
            ev_completed("grok-child"),
        ]),
    )
    .await;

    let test = native_parent_builder(&server, NativeParent::Claude)
        .build_with_auto_env(&server)
        .await?;
    let mut created_threads = test.thread_manager.subscribe_thread_created();
    test.submit_turn("spawn a Grok worker").await?;
    let child_thread_id =
        tokio::time::timeout(Duration::from_secs(10), created_threads.recv()).await??;
    wait_for_child_completion(&test, child_thread_id, "grok child complete").await?;

    let requests = wait_for_request_count(&server, /*expected*/ 3).await?;
    let child_request = requests
        .iter()
        .find(|request| request.url.path() == "/v1/grok/responses")
        .expect("Grok child request");
    let child_body: Value = child_request.body_json()?;
    assert_eq!(child_body["model"], "grok-4.6");
    assert!(child_body.to_string().contains(CHILD_TASK));
    assert!(child_request.headers.contains_key("x-grok-conv-id"));
    Ok(())
}
