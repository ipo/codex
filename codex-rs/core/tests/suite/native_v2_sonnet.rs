use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::Result;
use codex_exec_server::CreateDirectoryOptions;
use codex_features::Feature;
use codex_model_provider_info::CLAUDEFLARE_PROVIDER_ID;
use codex_model_provider_info::built_in_model_providers;
use codex_protocol::openai_models::ReasoningEffort;
use codex_protocol::protocol::AgentStatus;
use core_test_support::responses;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::TestCodexBuilder;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use uuid::Uuid;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::Request;
use wiremock::Respond;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;
const MODEL: &str = "claude-sonnet-5";
const PROFILE: &str = "anthropic/claude-sonnet-5";
const BETAS: &str = "claude-code-20250219,interleaved-thinking-2025-05-14,redact-thinking-2026-02-12,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,advisor-tool-2026-03-01,effort-2025-11-24";
const CALL_ID: &str = "call-spawn-sonnet";
const CHILD_TASK: &str = "return a Sonnet acknowledgement";

fn event(name: &str, data: Value) -> String {
    format!("event: {name}\ndata: {data}\n\n")
}
fn start(id: &str) -> String {
    event(
        "message_start",
        json!({"type":"message_start","message":{"id":id,"type":"message","role":"assistant","model":MODEL,"content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":0}}}),
    )
}
fn text(id: &str, content: &str) -> String {
    [
        start(id),
        event("content_block_start", json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}})),
        event("content_block_delta", json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":content}})),
        event("content_block_stop", json!({"type":"content_block_stop","index":0})),
        event("message_delta", json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}})),
        event("message_stop", json!({"type":"message_stop"})),
    ]
    .concat()
}
fn spawn(arguments: &str) -> String {
    [
        start("sonnet-spawn"),
        event("content_block_start", json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":CALL_ID,"name":"spawn_agent","input":{}}})),
        event("content_block_delta", json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":arguments}})),
        event("content_block_stop", json!({"type":"content_block_stop","index":0})),
        event("message_delta", json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":2}})),
        event("message_stop", json!({"type":"message_stop"})),
    ]
    .concat()
}

struct SonnetSequence {
    root_requests: AtomicUsize,
    spawn_arguments: String,
}

impl Respond for SonnetSequence {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body = if request.headers.contains_key("x-claude-code-agent-id") {
            text("sonnet-child", "Sonnet child complete")
        } else if self.root_requests.fetch_add(1, Ordering::SeqCst) == 0 {
            spawn(&self.spawn_arguments)
        } else {
            text("sonnet-root", "Sonnet root complete")
        };
        ResponseTemplate::new(200)
            .insert_header("content-type", "text/event-stream")
            .set_body_string(body)
    }
}

fn builder(server: &MockServer) -> TestCodexBuilder {
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
        config.model = Some(PROFILE.to_string());
        let cwd = config.codex_home.join("sonnet-cwd");
        std::fs::create_dir_all(&cwd).expect("create isolated Sonnet cwd");
        config.cwd = cwd;
        config.agent_default_subagent_model = Some(PROFILE.to_string());
        config.agent_default_subagent_reasoning_effort = Some(ReasoningEffort::High);
        config.base_instructions = Some("keep the native Sonnet prompt".to_string());
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

async fn wait_for_requests(server: &MockServer, expected: usize) -> Result<Vec<Request>> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    loop {
        let requests = server.received_requests().await.expect("requests");
        if requests.len() >= expected {
            return Ok(requests);
        }
        if tokio::time::Instant::now() >= deadline {
            let received = requests
                .iter()
                .map(|request| {
                    (
                        request.url.path().to_string(),
                        request.headers.contains_key("x-claude-code-agent-id"),
                    )
                })
                .collect::<Vec<_>>();
            anyhow::bail!("timed out waiting for {expected} Sonnet requests; received {received:?}")
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn assert_sonnet_headers(request: &Request, session_id: &str, subagent: bool) -> Result<()> {
    assert_eq!(request.url.path(), "/v1/messages");
    assert_eq!(request.url.query(), Some("beta=true"));
    for (name, expected) in [
        ("accept", "application/json"),
        ("anthropic-beta", BETAS),
        ("anthropic-dangerous-direct-browser-access", "true"),
        ("anthropic-version", "2023-06-01"),
        ("user-agent", "claude-cli/2.1.223 (external, cli)"),
        ("x-app", "cli"),
        ("x-stainless-arch", "x64"),
        ("x-stainless-lang", "js"),
        ("x-stainless-os", "Linux"),
        ("x-stainless-package-version", "0.94.0"),
        ("x-stainless-retry-count", "0"),
        ("x-stainless-runtime", "node"),
        ("x-stainless-runtime-version", "v26.3.0"),
        ("x-stainless-timeout", "600"),
    ] {
        assert_eq!(request.headers[name], expected, "header {name}");
    }
    assert_eq!(request.headers["x-claude-code-session-id"], session_id);
    assert!(
        Uuid::parse_str(
            request.headers["x-claude-code-session-id"]
                .to_str()
                .expect("valid session header"),
        )
        .is_ok()
    );
    assert!(!request.headers.contains_key("originator"));
    assert!(request.headers.contains_key("authorization"));
    assert!(!request.headers.contains_key("chatgpt-account-id"));
    let agent_id = request
        .headers
        .get("x-claude-code-agent-id")
        .map(|value| value.to_str())
        .transpose()?;
    if subagent {
        let agent_id = agent_id.expect("subagent agent ID");
        assert_eq!(agent_id.len(), 17);
        assert!(agent_id.starts_with('a'));
        assert!(agent_id[1..].bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert!(agent_id[1..].bytes().all(|byte| !byte.is_ascii_uppercase()));
    } else {
        assert_eq!(agent_id, None);
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sonnet_root_and_public_v2_subagent_use_complete_compatibility_profile() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(SonnetSequence {
            root_requests: AtomicUsize::new(0),
            spawn_arguments: serde_json::to_string(&json!({
                "plaintext_message": CHILD_TASK,
                "task_name": "worker",
                "fork_turns": "none",
            }))?,
        })
        .up_to_n_times(3)
        .mount(&server)
        .await;

    let test = builder(&server).build_with_auto_env(&server).await?;
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
    test.submit_turn("spawn a Sonnet worker").await?;
    let requests = wait_for_requests(&server, /*expected*/ 3).await?;
    let root_requests = requests
        .iter()
        .filter(|request| !request.headers.contains_key("x-claude-code-agent-id"))
        .collect::<Vec<_>>();
    let child_request = requests
        .iter()
        .find(|request| request.headers.contains_key("x-claude-code-agent-id"))
        .expect("public V2 Sonnet child request");
    assert_eq!(root_requests.len(), 2);
    let session = test.session_configured.session_id.to_string();
    assert_sonnet_headers(root_requests[0], &session, false)?;
    assert_sonnet_headers(child_request, &session, true)?;

    let root_body: Value = root_requests[0].body_json()?;
    let child_body: Value = child_request.body_json()?;
    assert_eq!(root_body["model"], MODEL);
    assert_eq!(child_body["model"], MODEL);
    for body in [&root_body, &child_body] {
        assert_eq!(body["max_tokens"], 64_000);
        assert_eq!(body["thinking"], json!({"type":"adaptive"}));
        assert_eq!(
            body["context_management"],
            json!({"edits":[{"type":"clear_thinking_20251015","keep":"all"}]})
        );
        assert_eq!(body["system"].as_array().map(Vec::len), Some(3));
        assert_eq!(
            body["system"][1]["cache_control"],
            json!({"type":"ephemeral"})
        );
        assert_eq!(
            body["system"][2]["cache_control"],
            json!({"type":"ephemeral"})
        );
        assert!(body["tools"].as_array().is_some_and(|tools| {
            tools
                .iter()
                .all(|tool| tool["name"].is_string() && tool["input_schema"].is_object())
        }));
        let latest_user_cache = body["messages"]
            .as_array()
            .and_then(|messages| {
                messages
                    .iter()
                    .rev()
                    .find(|message| message["role"] == "user")
            })
            .and_then(|message| message["content"].as_array())
            .and_then(|content| {
                content
                    .iter()
                    .rev()
                    .find_map(|block| block.get("cache_control"))
            })
            .expect("latest eligible user cache marker");
        assert_eq!(latest_user_cache, &json!({"type":"ephemeral"}));
        let metadata: Value = serde_json::from_str(
            body["metadata"]["user_id"]
                .as_str()
                .expect("metadata user identity"),
        )?;
        assert_eq!(metadata["device_id"].as_str().map(str::len), Some(64));
        assert_eq!(metadata["account_uuid"], "");
        assert_eq!(metadata["session_id"], session);
    }
    assert_eq!(root_body["output_config"], json!({"effort":"medium"}));
    assert_eq!(child_body["output_config"], json!({"effort":"high"}));
    assert_eq!(
        root_body["system"][1]["text"],
        "You are Claude Code, Anthropic's official CLI for Claude."
    );
    assert_eq!(
        child_body["system"][1]["text"],
        "You are a Claude agent, built on Anthropic's Claude Agent SDK."
    );
    assert!(root_body["system"][2]["text"].as_str().is_some_and(|text| {
        text.contains("model named Sonnet 5")
            && text.contains("exact model ID is claude-sonnet-5")
            && text.contains("knowledge cutoff is January 2026")
    }));
    assert!(
        child_body["system"][2]["text"]
            .as_str()
            .is_some_and(|text| {
                text.contains("agent for Claude Code")
                    && text.contains("model named Sonnet 5")
                    && text.contains("knowledge cutoff is January 2026")
            })
    );
    assert!(
        !root_body
            .to_string()
            .contains("keep the native Sonnet prompt")
    );
    assert!(child_body.to_string().contains(CHILD_TASK));

    let root_thread = test.session_configured.thread_id;
    let child_thread = test
        .thread_manager
        .list_thread_ids()
        .await
        .into_iter()
        .find(|thread_id| *thread_id != root_thread)
        .expect("spawned Sonnet child thread");
    let child = test.thread_manager.get_thread(child_thread).await?;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    loop {
        match child.agent_status().await {
            AgentStatus::Completed(message) => {
                assert_eq!(message.as_deref(), Some("Sonnet child complete"));
                break;
            }
            AgentStatus::Errored(error) => anyhow::bail!("Sonnet child errored: {error}"),
            status if tokio::time::Instant::now() >= deadline => {
                anyhow::bail!("timed out waiting for Sonnet child completion: {status:?}")
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
