use std::time::Duration;

use assert_matches::assert_matches;
use codex_protocol::models::AgentMessageInputContent;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ReasoningItemContent;
use pretty_assertions::assert_eq;

use super::*;

const EXPECTED_MODEL: &str = "Qwen3.8-27B-UD-Q4_K_XL.gguf";
const WINDOWS_MODEL: &str = r"F:\AI\llama-server\models\Qwen3.8-27B-UD-Q4_K_XL.gguf";

fn config() -> LlamaCppInferenceConfig {
    LlamaCppInferenceConfig {
        wire_api: WireApi::Responses,
        dialect: InferenceDialect::LlamaCpp,
        route: "llama_cpp".to_string(),
        expected_model_basename: EXPECTED_MODEL.to_string(),
        context_window: 240_128,
        max_input_tokens: 230_912,
        max_output_tokens: 8_192,
        safety_margin_tokens: 1_024,
    }
}

fn route(base_url: String) -> ResolvedWireRoute {
    ResolvedWireRoute {
        name: Some("llama_cpp".to_string()),
        wire_api: WireApi::Responses,
        dialect: InferenceDialect::LlamaCpp,
        base_url: Some(base_url),
        request_path: "responses".to_string(),
        query_params: None,
        request_max_retries: 0,
        stream_max_retries: 5,
        stream_idle_timeout: Duration::from_secs(300),
    }
}

fn message(role: &str, text: &str) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: role.to_string(),
        content: vec![ContentItem::InputText {
            text: text.to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }
}

fn request(input: Vec<ResponseItem>, effort: ReasoningEffortConfig) -> ResponsesApiRequest {
    ResponsesApiRequest {
        model: WINDOWS_MODEL.to_string(),
        instructions: "coding instructions".to_string(),
        input,
        tools: None,
        tool_choice: "auto".to_string(),
        parallel_tool_calls: true,
        reasoning: Some(Reasoning {
            effort: Some(effort),
            summary: None,
            context: None,
        }),
        store: false,
        stream: true,
        stream_options: None,
        include: Vec::new(),
        service_tier: None,
        prompt_cache_key: None,
        text: None,
        client_metadata: None,
    }
}

#[test]
fn validates_exact_route_token_contract_and_efforts() {
    validate_route(&route("http://localhost:8080/v1".to_string())).expect("canonical route");
    validate_config(&config()).expect("canonical token contract");
    assert_eq!(
        [
            local_effort(None).expect("default effort"),
            local_effort(Some(ReasoningEffort::High)).expect("high effort alias"),
            local_effort(Some(ReasoningEffort::None)).expect("disabled thinking"),
        ],
        [
            ReasoningEffort::Low,
            ReasoningEffort::XHigh,
            ReasoningEffort::None,
        ]
    );

    let mut invalid_route = route("http://localhost:8080/v1".to_string());
    invalid_route.dialect = InferenceDialect::OpenAi;
    assert!(validate_route(&invalid_route).is_err());
    let mut invalid_config = config();
    invalid_config.max_input_tokens -= 1;
    assert!(validate_config(&invalid_config).is_err());
    assert!(local_effort(Some(ReasoningEffort::Max)).is_err());
}

#[test]
fn request_body_preserves_typed_history_and_exact_sampling_profiles() {
    let reasoning = ResponseItem::Reasoning {
        id: None,
        summary: Vec::new(),
        content: Some(vec![ReasoningItemContent::ReasoningText {
            text: "raw thought".to_string(),
        }]),
        encrypted_content: Some(String::new()),
        internal_chat_message_metadata_passthrough: None,
    };
    let thinking = llama_cpp_request_body(
        &request(
            vec![message("user", "hello"), reasoning],
            ReasoningEffort::Low,
        ),
        &config(),
        &ReasoningEffort::Low,
    )
    .expect("thinking request");
    assert_eq!(thinking["model"], WINDOWS_MODEL);
    assert_eq!(thinking["stream"], true);
    assert_eq!(thinking["cache_prompt"], true);
    assert_eq!(thinking["max_output_tokens"], 8_192);
    assert_eq!(
        thinking["chat_template_kwargs"],
        json!({"enable_thinking": true, "preserve_thinking": true})
    );
    assert_eq!(thinking["temperature"], 1.0);
    assert_eq!(thinking["top_p"], 0.95);
    assert_eq!(thinking["top_k"], 20);
    assert_eq!(thinking["min_p"], 0.0);
    assert_eq!(thinking["presence_penalty"], 0.0);
    assert_eq!(thinking["repeat_penalty"], 1.0);
    assert_eq!(thinking["input"][1]["content"][0]["text"], "raw thought");
    assert!(thinking.get("previous_response_id").is_none());

    let non_thinking = llama_cpp_request_body(
        &request(vec![message("user", "hello")], ReasoningEffort::None),
        &config(),
        &ReasoningEffort::None,
    )
    .expect("non-thinking request");
    assert_eq!(
        non_thinking["chat_template_kwargs"],
        json!({"enable_thinking": false, "preserve_thinking": false})
    );
    assert_eq!(non_thinking["temperature"], 0.7);
    assert_eq!(non_thinking["top_p"], 0.8);
    assert_eq!(non_thinking["presence_penalty"], 1.5);
}

#[test]
fn normalizes_late_instructions_and_plaintext_agent_messages_without_rewriting_history() {
    let prompt = Prompt {
        base_instructions: BaseInstructions {
            text: "base".to_string(),
        },
        input: vec![
            message("developer", "initial guidance"),
            message("user", "question"),
            message("developer", "late guidance"),
            ResponseItem::AgentMessage {
                id: None,
                author: "child".to_string(),
                recipient: "parent".to_string(),
                content: vec![AgentMessageInputContent::InputText {
                    text: "plain result".to_string(),
                }],
                internal_chat_message_metadata_passthrough: None,
            },
        ],
        ..Default::default()
    };

    let normalized = normalize_input(&prompt).expect("normalize replayable history");
    assert_eq!(normalized[0], message("developer", "initial guidance"));
    assert_eq!(normalized[1], message("user", "question"));
    assert_matches!(
        &normalized[2],
        ResponseItem::Message { role, content, .. }
            if role == "user"
                && matches!(&content[..], [ContentItem::InputText { text }] if text.contains("late guidance"))
    );
    assert_eq!(
        normalized[3],
        message("user", "Agent message from child to parent:\nplain result")
    );
}

#[test]
fn deterministic_template_failures_are_not_retryable() {
    assert_matches!(
        classify_transport_error(codex_api::TransportError::Http {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            url: None,
            headers: None,
            body: Some(
                json!({"error":{"message":"System message must be at the beginning"}}).to_string(),
            ),
        }),
        ApiError::InvalidRequest { .. }
    );
    assert_matches!(
        classify_transport_error(codex_api::TransportError::Http {
            status: StatusCode::SERVICE_UNAVAILABLE,
            url: None,
            headers: None,
            body: Some("loading".to_string()),
        }),
        ApiError::Retryable { .. }
    );
}
