use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::Result;
use codex_core::ForkSnapshot;
use codex_model_provider_info::CLAUDEFLARE_PROVIDER_ID;
use codex_model_provider_info::ModelProviderWireRoute;
use codex_model_provider_info::WireApi;
use codex_model_provider_info::built_in_model_providers;
use codex_protocol::config_types::CollaborationMode;
use codex_protocol::config_types::ModeKind;
use codex_protocol::config_types::Settings;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::protocol::ThreadSettingsOverrides;
use codex_protocol::user_input::UserInput;
use core_test_support::responses;
use core_test_support::skip_if_no_network;
use core_test_support::submit_thread_settings;
use core_test_support::test_codex::TestCodex;
use core_test_support::test_codex::TestCodexBuilder;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
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

fn chunk(id: &str, delta: Value, finish_reason: Option<&str>) -> String {
    format!(
        "data: {}\n\n",
        json!({"id":id,"choices":[{"index":0,"delta":delta,"finish_reason":finish_reason}]})
    )
}

fn text_terminal(id: &str, text: &str, finish_reason: &str) -> String {
    format!(
        "{}data: [DONE]\n\n",
        chunk(id, json!({"content":text}), Some(finish_reason))
    )
}

fn parallel_tools() -> String {
    format!(
        "{}{}data: [DONE]\n\n",
        chunk(
            "kimi-tools",
            json!({"tool_calls":[{"index":0,"id":"call-one","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":\"pwd\",\"yield_time_ms\":1000,\"max_output_tokens\":1000}"}}]}),
            None,
        ),
        chunk(
            "kimi-tools",
            json!({"tool_calls":[{"index":1,"id":"call-two","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":\"pwd\",\"yield_time_ms\":1000,\"max_output_tokens\":1000}"}}]}),
            Some("tool_calls"),
        )
    )
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

async fn mount_native(server: &MockServer, bodies: Vec<String>) {
    let responses = bodies.into_iter().map(native_response).collect::<Vec<_>>();
    mount_native_responses(server, responses).await;
}

fn native_response(body: String) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_string(body)
}

async fn mount_native_responses(server: &MockServer, responses: Vec<ResponseTemplate>) {
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

fn native_retry_limit(builder: TestCodexBuilder, limit: u64) -> TestCodexBuilder {
    builder.with_config(move |config| {
        config.model_provider.stream_max_retries = Some(0);
        config
            .model_provider
            .wire_routes
            .get_mut("kimi_code")
            .expect("Kimi route")
            .stream_max_retries = Some(limit);
    })
}

fn native_builder(server: &MockServer, model: &str) -> TestCodexBuilder {
    let base_url = format!("{}/v1/kimi", server.uri());
    let model = model.to_string();
    test_codex().with_config(move |config| {
        let mut provider = built_in_model_providers(/*openai_base_url*/ None)
            .remove(CLAUDEFLARE_PROVIDER_ID)
            .expect("managed Claudeflare provider");
        provider
            .wire_routes
            .get_mut("kimi_code")
            .expect("Kimi route")
            .base_url = base_url;
        provider
            .http_headers
            .get_or_insert_default()
            .insert("x-test-route".to_string(), "kimi-native".to_string());
        config.model_provider = provider;
        config.model = Some(model);
        config.base_instructions = Some("Kimi system".to_string());
        config.agents_enabled = false;
        config.include_skill_instructions = false;
        config.include_permissions_instructions = false;
        config.include_apps_instructions = false;
        config.include_collaboration_mode_instructions = false;
        config.include_environment_context = false;
    })
}

async fn submit(test: &TestCodex, prompt: &str, output_schema: Option<Value>) -> Result<()> {
    test.codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: prompt.to_string(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: output_schema,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    Ok(())
}

async fn completion(test: &TestCodex) -> Result<()> {
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

async fn submit_thread_and_complete(thread: &codex_core::CodexThread, prompt: &str) -> Result<()> {
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
    loop {
        match wait_for_event(thread, |_| true).await {
            EventMsg::Error(error) => return Err(anyhow::anyhow!(error.message)),
            EventMsg::TurnComplete(event) => {
                return event
                    .error
                    .map_or_else(|| Ok(()), |error| Err(anyhow::anyhow!(error.message)));
            }
            _ => {}
        }
    }
}

async fn failed_turn_completed(test: &TestCodex) -> Result<()> {
    wait_for_event_match(&test.codex, |event| match event {
        EventMsg::TurnComplete(event) => Some(match &event.error {
            Some(_) => Ok(()),
            None => Err(anyhow::anyhow!("failed turn completed successfully")),
        }),
        _ => None,
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn managed_profiles_post_complete_native_requests() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let mut captures = Vec::new();
    for (model, wire_model, max_completion_tokens, thinking) in [
        (
            "kimi/k3",
            "k3",
            131_072,
            json!({"type":"enabled","keep":"all","effort":"high"}),
        ),
        (
            "kimi/k3-256k",
            "k3-256k",
            131_072,
            json!({"type":"enabled","keep":"all","effort":"high"}),
        ),
        (
            "kimi/kimi-for-coding",
            "kimi-for-coding",
            32_768,
            json!({"type":"enabled","keep":"all"}),
        ),
        (
            "kimi/kimi-for-coding-highspeed",
            "kimi-for-coding-highspeed",
            32_768,
            json!({"type":"enabled","keep":"all"}),
        ),
    ] {
        let server = responses::start_mock_server().await;
        mount_native(&server, vec![text_terminal("profile", "done", "stop")]).await;
        let test = native_builder(&server, model)
            .build_with_auto_env(&server)
            .await?;
        submit(&test, "profile prompt", None).await?;
        completion(&test).await?;
        let requests = server.received_requests().await.expect("requests");
        assert_eq!(requests.len(), 1);
        let request = &requests[0];
        assert_eq!(request.url.path(), "/v1/kimi/chat/completions");
        for (name, value) in [
            ("accept", "text/event-stream"),
            ("authorization", "Bearer dummy"),
            ("content-type", "application/json"),
            ("x-test-route", "kimi-native"),
        ] {
            assert_eq!(request.headers[name], value);
        }
        let mut body: Value = request.body_json()?;
        assert_eq!(body["model"], wire_model);
        assert_eq!(body["max_completion_tokens"], max_completion_tokens);
        assert_eq!(body["thinking"], thinking);
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"], json!({"include_usage":true}));
        assert_eq!(
            body["messages"][0],
            json!({"role":"system","content":"Kimi system"})
        );
        assert_eq!(body["messages"][1]["role"], "user");
        assert!(
            body["tools"]
                .as_array()
                .is_some_and(|tools| !tools.is_empty())
        );
        assert!(
            body["prompt_cache_key"]
                .as_str()
                .is_some_and(|key| !key.is_empty())
        );
        body["prompt_cache_key"] = "<session>".into();
        captures.push(body);
    }
    let profiles = captures
        .iter()
        .map(|body| {
            json!({
                "model": body["model"],
                "max_completion_tokens": body["max_completion_tokens"],
                "thinking": body["thinking"],
            })
        })
        .collect::<Vec<_>>();
    let mut common = captures[0].clone();
    for field in ["model", "max_completion_tokens", "thinking"] {
        common
            .as_object_mut()
            .expect("request object")
            .remove(field);
    }
    for body in &captures {
        let mut actual_common = body.clone();
        for field in ["model", "max_completion_tokens", "thinking"] {
            actual_common
                .as_object_mut()
                .expect("request object")
                .remove(field);
        }
        assert_eq!(actual_common, common);
    }
    insta::assert_snapshot!(
        "kimi_managed_profiles_complete_requests",
        serde_json::to_string_pretty(&json!({"profiles":profiles,"common":common}))?
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_kimi_preserves_collaboration_mode_transitions_as_reminders() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    mount_native(
        &server,
        vec![
            text_terminal("default", "first assistant output", "stop"),
            text_terminal("plan", "second assistant output", "stop"),
            text_terminal("default", "third assistant output", "stop"),
        ],
    )
    .await;
    let test = native_builder(&server, "kimi/k3")
        .with_config(|config| {
            config.include_collaboration_mode_instructions = true;
        })
        .build_with_auto_env(&server)
        .await?;

    for (mode, instructions, prompt) in [
        (
            ModeKind::Default,
            "default instructions",
            "first user input",
        ),
        (ModeKind::Plan, "plan instructions", "second user input"),
        (
            ModeKind::Default,
            "default instructions",
            "current user input",
        ),
    ] {
        submit_thread_settings(
            &test.codex,
            ThreadSettingsOverrides {
                collaboration_mode: Some(CollaborationMode {
                    mode,
                    settings: Settings {
                        model: "kimi/k3".to_string(),
                        reasoning_effort: None,
                        developer_instructions: Some(instructions.to_string()),
                    },
                }),
                ..Default::default()
            },
        )
        .await?;
        submit(&test, prompt, None).await?;
        completion(&test).await?;
    }

    let requests = server.received_requests().await.expect("requests");
    assert_eq!(requests.len(), 3);
    let body: Value = requests[2].body_json()?;
    let messages = body["messages"].as_array().expect("Kimi messages");
    let system = messages[0]["content"].as_str().expect("system content");
    assert!(system.contains("Kimi system"));
    assert!(system.contains("Host-generated <system-reminder>"));
    assert!(!system.contains("<collaboration_mode>default instructions</collaboration_mode>"));
    assert!(!system.contains("<collaboration_mode>plan instructions</collaboration_mode>"));
    assert_eq!(
        messages[1..],
        [
            json!({"role":"user","content":"first user input"}),
            json!({"role":"user","content":"<system-reminder><collaboration_mode>default instructions</collaboration_mode></system-reminder>"}),
            json!({"role":"assistant","content":"first assistant output"}),
            json!({"role":"user","content":"second user input"}),
            json!({"role":"user","content":"<system-reminder><collaboration_mode>plan instructions</collaboration_mode></system-reminder>"}),
            json!({"role":"assistant","content":"second assistant output"}),
            json!({"role":"user","content":"current user input"}),
            json!({"role":"user","content":"<system-reminder><collaboration_mode>default instructions</collaboration_mode></system-reminder>"}),
        ]
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn route_and_structured_output_fail_before_io() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    let result = test_codex()
        .with_model("kimi/k3")
        .build_with_auto_env(&server)
        .await;
    let error = match result {
        Ok(_) => panic!("missing route should fail"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("requires provider wire route `kimi_code`")
    );
    assert!(
        server
            .received_requests()
            .await
            .expect("requests")
            .is_empty()
    );

    let server = responses::start_mock_server().await;
    let test = native_builder(&server, "kimi/k3")
        .with_config(|config| {
            config.model_provider.wire_routes.insert(
                "kimi_code".to_string(),
                ModelProviderWireRoute {
                    wire_api: WireApi::ChatCompletions,
                    dialect: InferenceDialect::Kimi,
                    base_url: "http://127.0.0.1:1/v1/kimi".to_string(),
                    request_path: "responses".to_string(),
                    query_params: None,
                    request_max_retries: None,
                    stream_max_retries: None,
                    stream_idle_timeout_ms: None,
                },
            );
        })
        .build_with_auto_env(&server)
        .await?;
    submit(&test, "wrong route", None).await?;
    assert!(
        completion(&test)
            .await
            .expect_err("incompatible route")
            .to_string()
            .contains("chat_completions/kimi")
    );
    assert!(
        server
            .received_requests()
            .await
            .expect("requests")
            .is_empty()
    );

    let server = responses::start_mock_server().await;
    let test = native_builder(&server, "kimi/k3")
        .build_with_auto_env(&server)
        .await?;
    submit(&test, "structured output", Some(json!({"type":"object"}))).await?;
    assert!(
        completion(&test)
            .await
            .expect_err("structured output")
            .to_string()
            .contains("structured output")
    );
    assert!(
        server
            .received_requests()
            .await
            .expect("requests")
            .is_empty()
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn parallel_tools_continue_with_stable_native_identity() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    mount_native(
        &server,
        vec![parallel_tools(), text_terminal("final", "finished", "stop")],
    )
    .await;
    let test = native_builder(&server, "kimi/k3")
        .build_with_auto_env(&server)
        .await?;
    submit(&test, "run tools", None).await?;
    completion(&test).await?;
    let requests = server.received_requests().await.expect("requests");
    assert_eq!(requests.len(), 2);
    let bodies = requests
        .iter()
        .map(Request::body_json::<Value>)
        .collect::<Result<Vec<_>, _>>()?;
    assert!(bodies.iter().all(|body| body["model"] == "k3"));
    assert_eq!(bodies[0]["prompt_cache_key"], bodies[1]["prompt_cache_key"]);
    let continuation = bodies[1]["messages"].to_string();
    for expected in ["call-one", "call-two", "tool"] {
        assert!(continuation.contains(expected), "missing {expected}");
    }
    assert!(
        requests
            .iter()
            .all(|request| request.headers["x-test-route"] == "kimi-native")
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn public_fork_preserves_resumable_session_and_kimi_prompt_cache_key() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    mount_native(
        &server,
        vec![
            text_terminal("root", "root complete", "stop"),
            text_terminal("branch", "branch complete", "stop"),
            text_terminal("distinct", "distinct complete", "stop"),
        ],
    )
    .await;

    let root = native_builder(&server, "kimi/k3")
        .build_with_auto_env(&server)
        .await?;
    submit_thread_and_complete(&root.codex, "root turn").await?;
    let root_session_id = root.session_configured.session_id;
    let root_thread_id = root.session_configured.thread_id;
    let home = root.home.clone();
    let rollout_path = root.codex.rollout_path().expect("root rollout path");
    root.codex.shutdown_and_wait().await?;

    let resumed = native_builder(&server, "kimi/k3")
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
    submit_thread_and_complete(&branch.thread, "branch turn").await?;

    let distinct = native_builder(&server, "kimi/k3")
        .build_with_auto_env(&server)
        .await?;
    assert_ne!(distinct.session_configured.session_id, root_session_id);
    submit_thread_and_complete(&distinct.codex, "distinct root turn").await?;

    let requests = server.received_requests().await.expect("native requests");
    assert_eq!(requests.len(), 3);
    let bodies = requests
        .iter()
        .map(Request::body_json::<Value>)
        .collect::<Result<Vec<_>, _>>()?;
    assert!(requests.iter().all(|request| {
        request.url.path() == "/v1/kimi/chat/completions"
            && request.headers["x-test-route"] == "kimi-native"
    }));
    assert!(bodies.iter().all(|body| body["model"] == "k3"));
    assert_eq!(
        bodies[0]["prompt_cache_key"],
        Value::String(root_session_id.to_string())
    );
    assert_eq!(bodies[1]["prompt_cache_key"], bodies[0]["prompt_cache_key"]);
    assert_eq!(
        bodies[2]["prompt_cache_key"],
        Value::String(distinct.session_configured.session_id.to_string())
    );
    assert_ne!(bodies[2]["prompt_cache_key"], bodies[0]["prompt_cache_key"]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn terminal_gate_and_discarding_outcomes_do_not_commit_or_continue() -> Result<()> {
    skip_if_no_network!(Ok(()));
    for (finish_reason, expected) in [
        ("length", "model output limit reached"),
        ("max_tokens", "model output limit reached"),
        ("content_filter", "model refused"),
    ] {
        let server = responses::start_mock_server().await;
        let mut bodies = vec![text_terminal("discard", "discard me", finish_reason)];
        if finish_reason == "length" {
            bodies.push(text_terminal("recovered", "done", "stop"));
        }
        mount_native(&server, bodies).await;
        let test = native_builder(&server, "kimi/k3")
            .build_with_auto_env(&server)
            .await?;
        submit(&test, finish_reason, None).await?;
        let error = completion(&test).await.expect_err("discarding terminal");
        assert!(error.to_string().contains(expected));
        assert_eq!(server.received_requests().await.expect("requests").len(), 1);
        if finish_reason == "length" {
            failed_turn_completed(&test).await?;
            submit(&test, "after exhaustion", None).await?;
            completion(&test).await?;
            let requests = server.received_requests().await.expect("requests");
            assert_eq!(requests.len(), 2);
            assert!(
                !requests[1]
                    .body_json::<Value>()?
                    .to_string()
                    .contains("discard me")
            );
        }
    }

    let server = responses::start_mock_server().await;
    mount_native(&server, vec![chunk("not-done", json!({"tool_calls":[{"index":0,"id":"call-never","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":\"pwd\",\"yield_time_ms\":1000,\"max_output_tokens\":1000}"}}]}), Some("tool_calls"))]).await;
    let test = native_retry_limit(native_builder(&server, "kimi/k3"), 0)
        .build_with_auto_env(&server)
        .await?;
    submit(&test, "never execute", None).await?;
    assert!(
        completion(&test)
            .await
            .expect_err("missing done")
            .to_string()
            .contains("[DONE]")
    );
    assert_eq!(server.received_requests().await.expect("requests").len(), 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retryable_failures_use_route_limit_and_keep_attempt_history_isolated() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    let partial = [
        chunk("failed-message", json!({"reasoning_content":"failed reasoning"}), None),
        chunk("failed-message", json!({"tool_calls":[{"index":0,"id":"failed-call","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":\"printf failed-tool-executed\",\"yield_time_ms\":1000,\"max_output_tokens\":1000}"}}]}), None),
    ]
    .concat();
    let thinking_only = format!(
        "{}data: [DONE]\n\n",
        chunk(
            "thinking-only",
            json!({"reasoning_content":"discarded thinking"}),
            Some("stop"),
        )
    );
    let mut retry_responses = vec![
        native_response(partial),
        native_response("data: {not-json}\n\n".to_string()),
        native_response(thinking_only),
        native_response(text_terminal("empty", "", "stop")),
    ];
    retry_responses.extend(
        [408, 409, 429, 500, 502, 503, 504, 529]
            .map(|status| ResponseTemplate::new(status).insert_header("retry-after-ms", "0")),
    );
    retry_responses.push(native_response(text_terminal("success", "done", "stop")));
    mount_native_responses(&server, retry_responses).await;
    let test = native_retry_limit(native_builder(&server, "kimi/k3"), 12)
        .build_with_auto_env(&server)
        .await?;
    submit(&test, "retry cleanly", None).await?;
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
    assert_eq!(requests.len(), 13);
    for request in &requests[1..] {
        let body = request.body_json::<Value>()?.to_string();
        for failed in [
            "failed-message",
            "failed reasoning",
            "failed-call",
            "failed-tool-executed",
            "discarded thinking",
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
    let test = native_retry_limit(native_builder(&server, "kimi/k3"), 1)
        .with_config(move |config| {
            config
                .model_provider
                .wire_routes
                .get_mut("kimi_code")
                .expect("Kimi route")
                .base_url = format!("{base_url}/v1/kimi");
        })
        .build_with_auto_env(&server)
        .await?;
    submit(&test, "exhaust connection limit", None).await?;
    completion(&test).await.expect_err("connection limit");
    tokio::time::timeout(Duration::from_secs(5), connections).await???;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nonretryable_failures_and_overflow_preserve_stable_history() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    mount_native_responses(&server, vec![
        ResponseTemplate::new(400).set_body_string("invalid request"),
        ResponseTemplate::new(401).set_body_string("authentication_error"),
        ResponseTemplate::new(429).set_body_string(
            r#"{"error":{"type":"billing_error","message":"credit balance is too low"}}"#,
        ),
        native_response(format!(
            "{}data: [DONE]\n\n",
            chunk("unknown", json!({"content":"discard"}), Some("future_reason"))
        )),
        native_response(format!(
            "{}data: [DONE]\n\n",
            chunk("invalid-tool", json!({"tool_calls":[{"index":0,"id":"bad-call","type":"function","function":{"name":"exec_command","arguments":"{"}}]}), Some("tool_calls"))
        )),
        ResponseTemplate::new(400).set_body_string(
            r#"{"error":{"type":"invalid_request_error","message":"maximum context length exceeded"}}"#,
        ),
        native_response(text_terminal("summary", "short stable summary", "stop")),
        native_response(text_terminal("success", "done", "stop")),
    ]).await;
    let test = native_retry_limit(native_builder(&server, "kimi/k3"), 3)
        .build_with_auto_env(&server)
        .await?;
    for (index, prompt) in ["invalid", "auth", "quota", "unknown", "invalid tool"]
        .into_iter()
        .enumerate()
    {
        submit(&test, prompt, None).await?;
        completion(&test).await.expect_err("nonretryable failure");
        failed_turn_completed(&test).await?;
        assert_eq!(
            server.received_requests().await.expect("requests").len(),
            index + 1
        );
    }
    submit(&test, "overflow me", None).await?;
    completion(&test).await?;
    let requests = server.received_requests().await.expect("requests");
    assert_eq!(requests.len(), 8);
    let compact: Value = requests[6].body_json()?;
    assert!(
        compact
            .to_string()
            .contains("CONTEXT CHECKPOINT COMPACTION")
    );
    let retry: Value = requests[7].body_json()?;
    assert!(retry.to_string().contains("short stable summary"));
    assert!(!retry.to_string().contains("provider summary"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn missing_and_null_finish_execute_tools_and_continue_normally() -> Result<()> {
    for (name, finish_reason) in [("missing", None), ("null", Some(Value::Null))] {
        let server = responses::start_mock_server().await;
        let mut tool_response = json!({
            "id": format!("{name}-finish"),
            "choices": [{
                "index": 0,
                "delta": {
                    "reasoning_content": format!("{name} accepted reasoning"),
                    "tool_calls": [{
                        "index": 0,
                        "id": format!("{name}-call"),
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": "{\"cmd\":\"pwd\",\"yield_time_ms\":1000,\"max_output_tokens\":1000}"
                        }
                    }]
                }
            }]
        });
        if let Some(finish_reason) = finish_reason {
            tool_response["choices"][0]["finish_reason"] = finish_reason;
        }
        mount_native(
            &server,
            vec![
                format!("data: {tool_response}\n\ndata: [DONE]\n\n"),
                text_terminal("continued", "tool completed", "stop"),
            ],
        )
        .await;
        let test = native_retry_limit(native_builder(&server, "kimi/k3"), 3)
            .build_with_auto_env(&server)
            .await?;

        submit(&test, &format!("{name} optional finish"), None).await?;
        completion(&test).await?;
        let requests = server.received_requests().await.expect("requests");
        assert_eq!(requests.len(), 2);
        let continuation = requests[1].body_json::<Value>()?.to_string();
        for expected in [
            format!("{name} optional finish"),
            format!("{name} accepted reasoning"),
            format!("{name}-call"),
            "Process exited with code 0".to_string(),
        ] {
            assert!(
                continuation.contains(&expected),
                "continuation omitted {expected}"
            );
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_does_not_enter_kimi_retry_policy() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    mount_native_responses(
        &server,
        vec![
            native_response(text_terminal("cancelled", "never", "stop"))
                .set_delay(Duration::from_secs(30)),
        ],
    )
    .await;
    let test = native_retry_limit(native_builder(&server, "kimi/k3"), 3)
        .build_with_auto_env(&server)
        .await?;
    submit(&test, "cancel", None).await?;
    wait_for_event_match(&test.codex, |event| {
        matches!(event, EventMsg::TurnStarted(_)).then_some(())
    })
    .await;
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if server
                .received_requests()
                .await
                .is_some_and(|requests| requests.len() == 1)
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await?;
    test.codex.submit(Op::Interrupt).await?;
    wait_for_event_match(&test.codex, |event| {
        matches!(event, EventMsg::TurnAborted(_)).then_some(())
    })
    .await;
    assert_eq!(server.received_requests().await.expect("requests").len(), 1);
    Ok(())
}

fn serialized_input_tokens(body: &Value) -> u64 {
    let input = serde_json::to_vec(&(&body["messages"], &body["tools"]))
        .expect("serializable captured Kimi input");
    u64::try_from(input.len()).unwrap_or(u64::MAX).div_ceil(4)
}

fn assert_completion_budget(body: &Value, context: u64, hard_cap: u64) {
    let expected = hard_cap.min(context.saturating_sub(serialized_input_tokens(body)).max(1));
    assert_eq!(body["max_completion_tokens"], expected);
}

fn set_first_user_prompt(body: &mut Value, prompt: &str) {
    body["messages"][1]["content"] = prompt.into();
}

fn user_prompt_for_exact_tokens(template: &Value, target: u64) -> String {
    let mut body = template.clone();
    set_first_user_prompt(&mut body, "");
    let fixed_bytes = serde_json::to_vec(&(&body["messages"], &body["tools"]))
        .expect("serializable captured Kimi input")
        .len();
    let target_bytes = usize::try_from(target.saturating_mul(4)).expect("bounded target");
    "x".repeat(target_bytes - fixed_bytes)
}

async fn capture_context_requests(
    model: &str,
    prompts: Vec<String>,
    outputs: Vec<&str>,
) -> Result<Vec<Value>> {
    let server = responses::start_mock_server().await;
    let bodies = outputs
        .into_iter()
        .map(|text| text_terminal("context", text, "stop"))
        .collect();
    mount_native(&server, bodies).await;
    let test = native_builder(&server, model)
        .build_with_auto_env(&server)
        .await?;
    for prompt in prompts {
        submit(&test, &prompt, None).await?;
        completion(&test).await?;
    }
    let requests = server.received_requests().await.expect("requests");
    requests
        .iter()
        .map(|request| request.body_json().map_err(Into::into))
        .collect()
}

async fn assert_context_policy(model: &str, context: u64, hard_cap: u64) -> Result<()> {
    let calibration_pending = "pending-stage";
    let requests = capture_context_requests(
        model,
        vec!["seed".to_string(), calibration_pending.to_string()],
        vec!["stable", "done"],
    )
    .await?;
    assert_eq!(requests.len(), 2);
    let near_empty = &requests[0];
    let two_turn_template = &requests[1];
    assert_completion_budget(near_empty, context, hard_cap);
    assert_completion_budget(two_turn_template, context, hard_cap);

    let (neighbor, crossing) = if context == 1_048_576 {
        let exact_85_percent = context.saturating_mul(85).div_ceil(100);
        (exact_85_percent - 1, exact_85_percent)
    } else {
        (context - 50_000, context - 49_999)
    };

    let requests = capture_context_requests(
        model,
        vec![user_prompt_for_exact_tokens(near_empty, neighbor)],
        vec!["done"],
    )
    .await?;
    assert_eq!(requests.len(), 1);
    let neighboring = &requests[0];
    assert_eq!(serialized_input_tokens(neighboring), neighbor);
    assert_completion_budget(neighboring, context, hard_cap);

    let seed = user_prompt_for_exact_tokens(two_turn_template, crossing);
    let mut staged_request = two_turn_template.clone();
    set_first_user_prompt(&mut staged_request, &seed);
    let requests = capture_context_requests(
        model,
        vec![seed, calibration_pending.to_string()],
        vec!["stable", "boundary summary", "done"],
    )
    .await?;
    let [seed_request, compact, sampled] = requests.as_slice() else {
        unreachable!("request count asserted")
    };
    assert_eq!(serialized_input_tokens(&staged_request), crossing);
    staged_request["messages"]
        .as_array_mut()
        .expect("messages")
        .pop();
    assert!(serialized_input_tokens(&staged_request) < crossing);
    assert!(
        compact
            .to_string()
            .contains("CONTEXT CHECKPOINT COMPACTION")
    );
    assert!(compact.to_string().contains(calibration_pending));
    assert!(sampled.to_string().contains("boundary summary"));
    for body in [&seed_request, &compact, &sampled] {
        assert_completion_budget(body, context, hard_cap);
    }

    let requests = capture_context_requests(
        model,
        vec![user_prompt_for_exact_tokens(near_empty, context + 10_000)],
        vec!["near-full summary", "done"],
    )
    .await?;
    assert_eq!(requests.len(), 2);
    let compact = &requests[0];
    let sampled = &requests[1];
    assert_completion_budget(compact, context, hard_cap);
    assert_completion_budget(sampled, context, hard_cap);
    assert_eq!(compact["max_completion_tokens"], 1);
    assert_ne!(
        sampled["max_completion_tokens"],
        compact["max_completion_tokens"]
    );
    Ok(())
}

macro_rules! context_policy_tests {
    ($($name:ident: ($model:literal, $context:literal, $cap:literal)),*) => {$(
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn $name() -> Result<()> {
            skip_if_no_network!(Ok(()));
            assert_context_policy($model, $context, $cap).await
        }
    )*};
}

context_policy_tests! {
    k3_post_staging_context_policy: ("kimi/k3", 1_048_576, 131_072),
    k3_256k_post_staging_context_policy: ("kimi/k3-256k", 262_144, 131_072),
    coding_post_staging_context_policy: ("kimi/kimi-for-coding", 262_144, 32_768)
}
