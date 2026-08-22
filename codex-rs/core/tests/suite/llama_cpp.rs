use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::Result;
use codex_features::Feature;
use codex_model_provider_info::CLAUDEFLARE_PROVIDER_ID;
use codex_model_provider_info::built_in_model_providers;
use codex_protocol::ThreadId;
use codex_protocol::openai_models::ConfigShellToolType;
use codex_protocol::protocol::AgentStatus;
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

const LOCAL_MODEL: &str = "local/qwen3.8-27b";
const WINDOWS_MODEL: &str = r"F:\AI\llama-server\models\Qwen3.8-27B-UD-Q4_K_XL.gguf";

mod collaboration;
mod recovery;
mod turns;

struct Sequence {
    next: AtomicUsize,
    responses: Vec<ResponseTemplate>,
}

impl Respond for Sequence {
    fn respond(&self, _: &Request) -> ResponseTemplate {
        let index = self.next.fetch_add(1, Ordering::SeqCst);
        self.responses
            .get(index)
            .unwrap_or_else(|| panic!("unexpected request at sequence index {index}"))
            .clone()
    }
}

fn local_builder(server: &MockServer) -> TestCodexBuilder {
    let openai_base_url = server.uri();
    let local_base_url = format!("{}/v1", server.uri());
    test_codex()
        .with_config(move |config| {
            let mut provider = built_in_model_providers(/*openai_base_url*/ None)
                .remove(CLAUDEFLARE_PROVIDER_ID)
                .expect("managed Claudeflare provider");
            provider.base_url = Some(openai_base_url.clone());
            provider
                .wire_routes
                .get_mut("llama_cpp")
                .expect("llama.cpp route")
                .base_url = local_base_url;
            config.model_provider = provider;
            config.model = Some(LOCAL_MODEL.to_string());
            config.base_instructions = Some("Local Qwen conformance system".to_string());
            config.update_plan_enabled = true;
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
            config.include_collaboration_mode_instructions = false;
            config.include_environment_context = false;
        })
        .with_model_info_override(LOCAL_MODEL, |info| {
            info.shell_type = ConfigShellToolType::Disabled;
            info.experimental_supported_tools.clear();
            info.supports_search_tool = false;
        })
}

async fn mount_ready(server: &MockServer) {
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"status": "ok"})))
        .mount(server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{"id": WINDOWS_MODEL}]
        })))
        .mount(server)
        .await;
}

async fn mount_token_counts(server: &MockServer, counts: impl IntoIterator<Item = u32>) {
    let responses = counts
        .into_iter()
        .map(|input_tokens| {
            ResponseTemplate::new(200).set_body_json(json!({"input_tokens": input_tokens}))
        })
        .collect::<Vec<_>>();
    let count = responses.len() as u64;
    Mock::given(method("POST"))
        .and(path("/v1/responses/input_tokens"))
        .respond_with(Sequence {
            next: AtomicUsize::new(0),
            responses,
        })
        .up_to_n_times(count)
        .mount(server)
        .await;
}

async fn mount_local_inference(server: &MockServer, responses: Vec<ResponseTemplate>) {
    let count = responses.len() as u64;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(Sequence {
            next: AtomicUsize::new(0),
            responses,
        })
        .up_to_n_times(count)
        .mount(server)
        .await;
}

fn local_sse(events: Vec<Value>) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_string(responses::sse(events))
}

fn text_response(response_id: &str, message_id: &str, text: &str) -> ResponseTemplate {
    local_sse(vec![
        responses::ev_response_created(response_id),
        responses::ev_assistant_message(message_id, text),
        responses::ev_completed(response_id),
    ])
}

async fn wait_for_child_completion(
    test: &TestCodex,
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

async fn submit_turn(test: &TestCodex, prompt: &str) -> Result<()> {
    test.codex
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
    wait_for_event_match(&test.codex, |event| match event {
        EventMsg::TurnComplete(event) => Some(match &event.error {
            Some(error) => Err(anyhow::anyhow!(error.message.clone())),
            None => Ok(()),
        }),
        _ => None,
    })
    .await
}

async fn received(server: &MockServer, request_path: &str) -> Vec<Request> {
    server
        .received_requests()
        .await
        .expect("captured requests")
        .into_iter()
        .filter(|request| request.url.path() == request_path)
        .collect()
}
