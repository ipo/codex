use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use anyhow::Result;
use codex_model_provider_info::CLAUDEFLARE_PROVIDER_ID;
use codex_model_provider_info::ModelProviderWireRoute;
use codex_model_provider_info::WireApi;
use codex_model_provider_info::built_in_model_providers;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::user_input::UserInput;
use core_test_support::responses;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::TestCodex;
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
    let responses = bodies
        .into_iter()
        .map(|body| {
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body)
        })
        .collect::<Vec<_>>();
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
async fn route_and_responses_only_capabilities_fail_before_io() -> Result<()> {
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
        .with_config(|config| config.agents_enabled = true)
        .build_with_auto_env(&server)
        .await?;
    submit(&test, "namespace tool", None).await?;
    assert!(
        completion(&test)
            .await
            .expect_err("namespace tool")
            .to_string()
            .contains("unsupported tool")
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
async fn terminal_gate_and_discarding_outcomes_do_not_commit_or_continue() -> Result<()> {
    skip_if_no_network!(Ok(()));
    for (finish_reason, expected) in [
        ("length", "model output limit reached"),
        ("content_filter", "model refused"),
    ] {
        let server = responses::start_mock_server().await;
        mount_native(
            &server,
            vec![text_terminal("discard", "discard me", finish_reason)],
        )
        .await;
        let test = native_builder(&server, "kimi/k3")
            .build_with_auto_env(&server)
            .await?;
        submit(&test, finish_reason, None).await?;
        assert!(
            completion(&test)
                .await
                .expect_err("discarding terminal")
                .to_string()
                .contains(expected)
        );
        assert_eq!(server.received_requests().await.expect("requests").len(), 1);
    }

    let server = responses::start_mock_server().await;
    mount_native(&server, vec![chunk("not-done", json!({"tool_calls":[{"index":0,"id":"call-never","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":\"pwd\",\"yield_time_ms\":1000,\"max_output_tokens\":1000}"}}]}), Some("tool_calls"))]).await;
    let test = native_builder(&server, "kimi/k3")
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
