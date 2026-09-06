use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::model_inference::KimiInferenceConfig;
use codex_protocol::model_inference::KimiThinkingPolicy;
use codex_protocol::model_inference::WireApi;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;

use super::*;

fn settings(input: u64, effort: Option<KimiThinkingEffort>) -> KimiRequestSettings {
    KimiRequestSettings {
        context_window: 262_144,
        input_estimate: KimiInputEstimate::Fixed(input),
        prompt_cache_key: "stable-session-affinity".to_string(),
        thinking_effort: effort,
        reasoning_key: None,
    }
}

fn profile(
    wire_model: &str,
    max_output_tokens: u32,
    thinking: KimiThinkingPolicy,
) -> KimiInferenceConfig {
    KimiInferenceConfig {
        wire_api: WireApi::ChatCompletions,
        dialect: InferenceDialect::Kimi,
        route: "kimi_code".to_string(),
        wire_model: wire_model.to_string(),
        max_output_tokens,
        thinking,
    }
}

fn message(role: &str, content: &str) -> KimiMessage {
    match role {
        "system" => KimiMessage::System {
            content: KimiContent::Text(content.to_string()),
        },
        "user" => KimiMessage::User {
            content: KimiContent::Text(content.to_string()),
        },
        _ => panic!("unsupported test role"),
    }
}

fn decode(stream: &str) -> Result<KimiDecodedResponse, KimiStreamError> {
    let mut decoder = KimiStreamDecoder::new("k3").with_trace_id("trace-1");
    for fragment in stream.as_bytes().chunks(17) {
        decoder.feed(fragment)?;
    }
    decoder.finish()
}

#[test]
fn exact_requests_cover_profiles_thinking_and_budget_edges() {
    let cases = [
        (
            profile("k3", 131_072, KimiThinkingPolicy::RequiredWithEffort),
            1_048_576,
            0,
            Some(KimiThinkingEffort::Low),
        ),
        (
            profile("k3-256k", 131_072, KimiThinkingPolicy::RequiredWithEffort),
            262_144,
            131_073,
            Some(KimiThinkingEffort::High),
        ),
        (
            profile("k3-256k", 131_072, KimiThinkingPolicy::RequiredWithEffort),
            262_144,
            262_144,
            Some(KimiThinkingEffort::Max),
        ),
        (
            profile("kimi-for-coding", 32_768, KimiThinkingPolicy::Required),
            262_144,
            0,
            None,
        ),
        (
            profile(
                "kimi-for-coding-highspeed",
                32_768,
                KimiThinkingPolicy::Required,
            ),
            262_144,
            229_377,
            None,
        ),
    ];
    let requests = cases
        .into_iter()
        .map(|(profile, context_window, input, effort)| {
            let mut request_settings = settings(input, effort);
            request_settings.context_window = context_window;
            let request = build_kimi_chat_request(
                &profile,
                request_settings,
                vec![message("system", "You are Codex.")],
                Vec::new(),
            )
            .expect("valid Kimi request");
            serde_json::to_value(request).expect("request JSON")
        })
        .collect::<Vec<_>>();
    assert_eq!(
        requests,
        vec![
            json!({"model":"k3","messages":[{"role":"system","content":"You are Codex."}],"stream":true,"stream_options":{"include_usage":true},"max_completion_tokens":131072,"prompt_cache_key":"stable-session-affinity","thinking":{"type":"enabled","keep":"all","effort":"low"}}),
            json!({"model":"k3-256k","messages":[{"role":"system","content":"You are Codex."}],"stream":true,"stream_options":{"include_usage":true},"max_completion_tokens":131071,"prompt_cache_key":"stable-session-affinity","thinking":{"type":"enabled","keep":"all","effort":"high"}}),
            json!({"model":"k3-256k","messages":[{"role":"system","content":"You are Codex."}],"stream":true,"stream_options":{"include_usage":true},"max_completion_tokens":1,"prompt_cache_key":"stable-session-affinity","thinking":{"type":"enabled","keep":"all","effort":"max"}}),
            json!({"model":"kimi-for-coding","messages":[{"role":"system","content":"You are Codex."}],"stream":true,"stream_options":{"include_usage":true},"max_completion_tokens":32768,"prompt_cache_key":"stable-session-affinity","thinking":{"type":"enabled","keep":"all"}}),
            json!({"model":"kimi-for-coding-highspeed","messages":[{"role":"system","content":"You are Codex."}],"stream":true,"stream_options":{"include_usage":true},"max_completion_tokens":32767,"prompt_cache_key":"stable-session-affinity","thinking":{"type":"enabled","keep":"all"}}),
        ]
    );
    assert_eq!(
        [
            build_kimi_chat_request(
                &profile("k3", 131_072, KimiThinkingPolicy::RequiredWithEffort),
                settings(/*input*/ 0, /*effort*/ None),
                Vec::new(),
                Vec::new()
            ),
            build_kimi_chat_request(
                &profile("kimi-for-coding", 32_768, KimiThinkingPolicy::Required),
                settings(/*input*/ 0, Some(KimiThinkingEffort::High)),
                Vec::new(),
                Vec::new()
            ),
        ],
        [
            Err(KimiRequestError::MissingThinkingEffort),
            Err(KimiRequestError::UnsupportedThinkingEffort),
        ]
    );
    assert_eq!(
        build_kimi_chat_request(
            &profile(
                &"\\".repeat(256),
                131_072,
                KimiThinkingPolicy::RequiredWithEffort,
            ),
            settings(/*input*/ 0, Some(KimiThinkingEffort::High)),
            Vec::new(),
            Vec::new(),
        ),
        Err(KimiRequestError::InvalidProfile)
    );
}

#[test]
fn interleaved_tools_reasoning_and_final_usage_decode_transactionally() {
    let stream = concat!(
        "data: {\"id\":\"chat-1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning\":\"think \"}}]}\n\n",
        "data: {\"id\":\"chat-1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-0\",\"type\":\"function\",\"function\":{\"name\":\"first\",\"arguments\":\"{\\\"a\\\":\"}},{\"index\":1,\"id\":\"call-1\",\"type\":\"function\",\"function\":{\"name\":\"second\",\"arguments\":\"{\"}}]}}]}\n\n",
        "data: {\"id\":\"chat-1\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\"then\",\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"}\\n\"}},{\"index\":0,\"function\":{\"arguments\":\"1}\"}}]},\"finish_reason\":\"tool_calls\",\"usage\":{\"prompt_tokens\":16646,\"completion_tokens\":654,\"total_tokens\":17300}}]}\n\n",
        "data: {\"id\":\"chat-1\",\"choices\":[],\"usage\":{\"prompt_tokens\":16648,\"completion_tokens\":654,\"total_tokens\":17302,\"prompt_tokens_details\":{\"cached_tokens\":4000},\"completion_tokens_details\":{\"reasoning_tokens\":378}}}\n\n",
        "data: [DONE]\n\n"
    );
    let mut decoder = KimiStreamDecoder::new("k3").with_trace_id("trace-1");
    let mut events = Vec::new();
    for fragment in stream.as_bytes().chunks(13) {
        events.extend(decoder.feed(fragment).expect("valid stream fragment"));
    }
    let decoded = decoder.finish().expect("valid Kimi stream");
    assert_eq!(decoded.terminal, KimiTerminal::ToolsReady);
    assert_eq!(decoded.trace_id.as_deref(), Some("trace-1"));
    assert_eq!(
        decoded.usage,
        Some(KimiUsage {
            prompt_tokens: 16_648,
            completion_tokens: 654,
            total_tokens: 17_302,
            cached_prompt_tokens: 4_000,
            reasoning_tokens: 378,
        })
    );
    let pending = decoded.pending.expect("committable output");
    assert_eq!(pending.reasoning.text, "think then");
    assert_eq!(
        pending.tool_calls,
        vec![
            KimiToolCall::function("call-0", "first", "{\"a\":1}".to_string()),
            KimiToolCall::function("call-1", "second", "{}\n".to_string()),
        ]
    );
    assert_eq!(events.len(), 6);
    assert!(events.iter().all(|event| match event {
        KimiStreamEvent::Content { response_id, .. }
        | KimiStreamEvent::Reasoning { response_id, .. }
        | KimiStreamEvent::ToolCall { response_id, .. } => response_id == "chat-1",
    }));
}

#[test]
fn learned_reasoning_key_replays_only_to_the_exact_wire_model() {
    let decoded = decode(concat!(
        "data: {\"id\":\"chat-2\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_details\":\"private\",\"content\":\"answer\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chat-2\",\"choices\":[]}\n\n",
        "data: [DONE]\n\n"
    ))
    .expect("optional finish reason is valid");
    let reasoning = &decoded.pending.expect("committable output").reasoning;
    let persisted = KimiReasoning::from_persisted(
        reasoning.text.clone(),
        reasoning.opaque_marker().to_string(),
    )
    .expect("bounded replay marker round trip");
    assert_eq!(
        KimiReasoning::from_persisted("unsafe".to_string(), "responses-encrypted".to_string()),
        Err(KimiRequestError::MalformedReplayMarker)
    );
    let exact = KimiAssistantMessage::from_input(
        KimiAssistantMessageInput {
            content: Some("answer".to_string()),
            tool_calls: Vec::new(),
            reasoning: Some(&persisted),
        },
        "k3",
    )
    .expect("matching replay");
    let switched = KimiAssistantMessage::from_input(
        KimiAssistantMessageInput {
            content: Some("answer".to_string()),
            tool_calls: Vec::new(),
            reasoning: Some(&persisted),
        },
        "k3-256k",
    )
    .expect("model switch drops replay");
    assert_eq!(
        serde_json::to_value(&exact).expect("assistant JSON"),
        json!({"content":"answer","reasoning_details":"private"})
    );
    assert_eq!(
        serde_json::to_value(&switched).expect("assistant JSON"),
        json!({"content":"answer"})
    );
    let earlier = KimiAssistantMessage::from_input(
        KimiAssistantMessageInput {
            content: Some("earlier".to_string()),
            tool_calls: Vec::new(),
            reasoning: None,
        },
        "k3",
    )
    .expect("default reasoning key");
    let mut request_settings = settings(/*input*/ 0, Some(KimiThinkingEffort::High));
    request_settings.reasoning_key = Some(persisted.key().expect("learned reasoning key"));
    let request = build_kimi_chat_request(
        &profile("k3", 131_072, KimiThinkingPolicy::RequiredWithEffort),
        request_settings,
        vec![
            KimiMessage::Assistant(earlier),
            KimiMessage::Assistant(exact),
        ],
        Vec::new(),
    )
    .expect("learned key applies to every assistant message");
    let messages = serde_json::to_value(request).expect("request JSON")["messages"]
        .as_array()
        .expect("messages")
        .clone();
    assert_eq!(messages[0]["reasoning_details"], "");
    assert_eq!(messages[1]["reasoning_details"], "private");
}

#[test]
fn normalized_parallel_calls_keep_results_matched_and_omit_empty_content() {
    let long_id = "call-that-is-deliberately-longer-than-the-kimi-sixty-four-byte-limit-aaaa";
    let assistant = KimiAssistantMessage::from_input(
        KimiAssistantMessageInput {
            content: Some(String::new()),
            tool_calls: vec![KimiToolCall::function(long_id, "inspect", "{}".to_string())],
            reasoning: None,
        },
        "k3",
    )
    .expect("assistant message");
    let request = build_kimi_chat_request(
        &profile("k3", 131_072, KimiThinkingPolicy::RequiredWithEffort),
        settings(/*input*/ 0, Some(KimiThinkingEffort::High)),
        vec![
            KimiMessage::Assistant(assistant),
            KimiMessage::Tool {
                tool_call_id: long_id.to_string(),
                content: KimiContent::Text("done".to_string()),
            },
        ],
        Vec::new(),
    )
    .expect("valid Kimi request");
    let value = serde_json::to_value(request).expect("request JSON");
    let messages = value["messages"].as_array().expect("message array");
    let call_id = messages[0]["tool_calls"][0]["id"]
        .as_str()
        .expect("normalized call ID");
    assert!(call_id.len() <= 64);
    assert_eq!(messages[1]["tool_call_id"], call_id);
    assert_eq!(messages[0].get("content"), None);
    assert_eq!(messages[0]["reasoning_content"], "");
}

#[test]
fn invalid_or_noncommittable_streams_never_return_pending_output() {
    let exhausted = decode(concat!(
        "data: {\"id\":\"chat-3\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"unfinished\"},\"finish_reason\":\"length\"}]}\n\n",
        "data: [DONE]\n\n"
    ))
    .expect("valid exhaustion terminal");
    assert_eq!(exhausted.terminal, KimiTerminal::OutputExhausted);
    assert_eq!(exhausted.pending, None);

    let failures = [
        decode(""),
        decode("data: not-json\n\n"),
        decode("data: {\"id\":\"x\",\"choices\":[]}\n\n"),
        decode(concat!(
            "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        )),
        decode(concat!(
            "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"thinking only\"}}]}\n\n",
            "data: [DONE]\n\n"
        )),
        decode(concat!(
            "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call\",\"type\":\"function\",\"function\":{\"name\":\"bad\",\"arguments\":\"[1]\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n"
        )),
    ];
    assert!(failures.into_iter().all(|failure| failure.is_err()));
}

#[test]
fn non_string_reasoning_details_are_ignored_for_compatibility() {
    let decoded = decode(concat!(
        "data: {\"id\":\"chat-4\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_details\":[{\"type\":\"summary\"}],\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n"
    ))
    .expect("array-shaped compatibility details are ignored");
    assert_eq!(decoded.pending.expect("committable response").content, "ok");
}

#[test]
fn request_shape_never_contains_legacy_max_tokens() {
    let request = build_kimi_chat_request(
        &profile("kimi-for-coding", 32_768, KimiThinkingPolicy::Required),
        settings(/*input*/ 0, /*effort*/ None),
        vec![message("user", "hello")],
        vec![KimiFunctionTool::new(KimiFunctionDefinition {
            name: "inspect".to_string(),
            description: "Inspect".to_string(),
            parameters: json!({"type":"object","properties":{}}),
            strict: true,
        })],
    )
    .expect("valid request");
    let value: Value = serde_json::to_value(request).expect("request JSON");
    assert_eq!(value.get("max_tokens"), None);
    assert_eq!(value["tools"][0]["type"], "function");
}

#[test]
fn tool_schemas_resolve_local_definitions_and_complete_nested_types() {
    let normalized = normalize_kimi_schema(&json!({
        "type": "object",
        "$defs": {
            "entry": {
                "type": "object",
                "properties": {
                    "label": {"minLength": 1},
                    "enabled": {"const": true}
                },
                "required": ["label"]
            }
        },
        "properties": {
            "entry": {"$ref": "#/$defs/entry", "description": "Selected entry"}
        }
    }))
    .expect("supported Kimi schema");
    assert_eq!(
        normalized,
        json!({
            "type": "object",
            "properties": {
                "entry": {
                    "type": "object",
                    "description": "Selected entry",
                    "properties": {
                        "label": {"type": "string", "minLength": 1},
                        "enabled": {"type": "boolean", "const": true}
                    },
                    "required": ["label"]
                }
            }
        })
    );
    assert_eq!(
        normalize_kimi_schema(&json!({
            "type": "object",
            "properties": {"entry": {"$ref": "https://example.test/schema"}}
        })),
        Err(KimiSchemaError::UnsupportedReference(
            "https://example.test/schema".to_string()
        ))
    );
}
