use super::*;
use codex_api::OpenAiVerbosity;
use codex_api::ReasoningContext;
use codex_protocol::config_types::ReasoningSummary;
use codex_protocol::models::AgentMessageInputContent;
use codex_protocol::openai_models::ReasoningEffort;
use pretty_assertions::assert_eq;
use serde_json::json;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::time::Duration;

fn route() -> ResolvedWireRoute {
    ResolvedWireRoute {
        name: Some("grok".to_string()),
        wire_api: WireApi::Responses,
        dialect: InferenceDialect::Grok,
        base_url: Some("http://127.0.0.1:8080/v1/grok".to_string()),
        request_path: "responses".to_string(),
        query_params: None,
        request_max_retries: 0,
        stream_max_retries: 10,
        stream_idle_timeout: Duration::from_secs(300),
    }
}

fn openai_shaped_request() -> ResponsesApiRequest {
    ResponsesApiRequest {
        model: "catalog-slug".to_string(),
        instructions: "Keep the response short.".to_string(),
        input: vec![
            serde_json::from_value(json!({
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "Checked"}],
                "content": null,
                "encrypted_content": "encrypted-reasoning",
                "internal_chat_message_metadata_passthrough": {"turn_id": "turn-internal"}
            }))
            .expect("reasoning fixture"),
        ],
        tools: None,
        tool_choice: "auto".to_string(),
        parallel_tool_calls: true,
        reasoning: Some(Reasoning {
            effort: Some(ReasoningEffort::High),
            summary: Some(ReasoningSummary::Concise),
            context: Some(ReasoningContext::AllTurns),
        }),
        store: true,
        stream: false,
        stream_options: Some(StreamOptions {
            reasoning_summary_delivery: codex_api::ReasoningSummaryDelivery::SequentialCutoff,
        }),
        include: vec!["openai-only.include".to_string()],
        service_tier: Some("priority".to_string()),
        prompt_cache_key: Some("stable-cache-key".to_string()),
        text: Some(codex_api::TextControls {
            verbosity: Some(OpenAiVerbosity::High),
            format: None,
        }),
        client_metadata: Some(std::collections::HashMap::from([(
            "openai".to_string(),
            "metadata".to_string(),
        )])),
        access_programs: Some(codex_protocol::turn_input::CyberAccessProgram::Standard.into()),
    }
}

#[test]
fn request_profile_omits_openai_fields_and_preserves_encrypted_reasoning() {
    let request = adapt_request(openai_shaped_request(), "grok-4.6");
    assert_eq!(
        serde_json::to_value(request).expect("serialize Grok request"),
        json!({
            "model": "grok-4.6",
            "instructions": "Keep the response short.",
            "input": [{
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "Checked"}],
                "encrypted_content": "encrypted-reasoning"
            }],
            "tool_choice": "auto",
            "parallel_tool_calls": true,
            "reasoning": {"effort": "high", "summary": "concise"},
            "store": false,
            "stream": true,
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "stable-cache-key"
        })
    );
}

#[test]
fn request_profile_preserves_structured_output_format() {
    let mut request = openai_shaped_request();
    request.text = create_text_param_for_request(
        Some(codex_protocol::config_types::Verbosity::High),
        &Some(json!({"type": "object"})),
        /*output_schema_strict*/ true,
    );
    let request = adapt_request(request, "grok-4.6");
    let body = serde_json::to_value(request).expect("serialize Grok request");
    assert_eq!(
        body["text"],
        json!({
            "format": {
                "type": "json_schema",
                "strict": true,
                "schema": {"type": "object"},
                "name": "codex_output_schema"
            }
        })
    );
}

#[test]
fn history_projection_renders_plaintext_agent_messages() {
    let mut input = vec![ResponseItem::AgentMessage {
        id: Some(ResponseItemId::new("amsg_1")),
        author: "/root".to_string(),
        recipient: "/root/worker".to_string(),
        content: vec![AgentMessageInputContent::InputText {
            text: "continue".to_string(),
        }],
        internal_chat_message_metadata_passthrough: None,
    }];

    project_history(&mut input).expect("plaintext history should project");

    assert_eq!(
        input,
        vec![ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "Agent message from /root to /root/worker:\ncontinue".to_string(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        }]
    );
}

#[test]
fn history_projection_rejects_encrypted_agent_messages() {
    let mut input = vec![ResponseItem::AgentMessage {
        id: None,
        author: "/root".to_string(),
        recipient: "/root/worker".to_string(),
        content: vec![AgentMessageInputContent::EncryptedContent {
            encrypted_content: "opaque".to_string(),
        }],
        internal_chat_message_metadata_passthrough: None,
    }];

    assert_eq!(
        project_history(&mut input)
            .expect_err("encrypted history should fail")
            .to_string(),
        "Grok history item at index 0 contains a non-plaintext structured agent message"
    );
}

fn headers_as_map(headers: &ApiHeaderMap) -> BTreeMap<String, String> {
    headers
        .iter()
        .map(|(name, value)| {
            (
                name.as_str().to_string(),
                value.to_str().expect("ASCII Grok header").to_string(),
            )
        })
        .collect()
}

#[test]
fn lineage_headers_use_turn_id_and_stable_thread_fallback() {
    let mut metadata = CodexResponsesMetadata::new(
        "installation-1".to_string(),
        "session-1".to_string(),
        "thread-1".to_string(),
        "window-1".to_string(),
    );
    let fallback = lineage_headers(&metadata, "grok-4.6").expect("fallback headers");
    assert_eq!(fallback[GROK_REQUEST_ID_HEADER], "thread-1");

    metadata.turn_id = Some("turn-1".to_string());
    let headers = lineage_headers(&metadata, "grok-4.6").expect("turn headers");
    assert_eq!(
        headers_as_map(&headers),
        BTreeMap::from([
            (
                GROK_AGENT_ID_HEADER.to_string(),
                "installation-1".to_string()
            ),
            (
                GROK_CONVERSATION_ID_HEADER.to_string(),
                "thread-1".to_string(),
            ),
            (
                GROK_MODEL_OVERRIDE_HEADER.to_string(),
                "grok-4.6".to_string(),
            ),
            (GROK_REQUEST_ID_HEADER.to_string(), "turn-1".to_string()),
            (GROK_SESSION_ID_HEADER.to_string(), "session-1".to_string()),
        ])
    );
}

#[test]
fn invalid_lineage_value_is_rejected() {
    let metadata = CodexResponsesMetadata::new(
        "installation-1".to_string(),
        "session-1".to_string(),
        "thread\ninvalid".to_string(),
        "window-1".to_string(),
    );
    assert!(
        lineage_headers(&metadata, "grok-4.6")
            .expect_err("invalid header")
            .to_string()
            .contains("x-grok-conv-id")
    );
}

#[test]
fn grok_dispatch_rejects_noncanonical_routes() {
    validate_route(&route()).expect("canonical Grok route");

    let mut wrong_path = route();
    wrong_path.request_path = "chat/completions".to_string();
    let mut query = route();
    query.query_params = Some(HashMap::from([("beta".to_string(), "true".into())]));
    let mut wrong_dialect = route();
    wrong_dialect.dialect = InferenceDialect::OpenAi;

    for invalid in [wrong_path, query, wrong_dialect] {
        assert!(
            validate_route(&invalid)
                .expect_err("noncanonical Grok route")
                .to_string()
                .contains("must resolve to responses/grok")
        );
    }
}

#[test]
fn route_provider_disables_nested_retries_and_uses_route_timeout() {
    let mut provider = codex_model_provider_info::ModelProviderInfo::default()
        .to_api_provider(/*auth_mode*/ None)
        .expect("API provider");
    provider.query_params = Some(HashMap::from([("openai".to_string(), "true".to_string())]));
    provider.retry.max_attempts = 4;
    provider.stream_idle_timeout = Duration::from_secs(1);

    let provider = provider_for_route(provider, &route()).expect("Grok provider");
    assert_eq!(
        (
            provider.base_url,
            provider.query_params,
            provider.retry.max_attempts,
            provider.stream_idle_timeout,
        ),
        (
            "http://127.0.0.1:8080/v1/grok".to_string(),
            None,
            0,
            Duration::from_secs(300),
        )
    );
}
