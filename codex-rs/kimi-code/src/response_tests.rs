use codex_chat_completions::DecodeError;
use codex_chat_completions::DecodeStream;
use codex_chat_completions::DecodedStream;
use codex_chat_completions::PendingResult;
use codex_chat_completions::ResponseMetadata;
use codex_chat_completions::ToolCall;
use codex_chat_completions::ToolCallFunction;
use codex_chat_completions::ToolCallKind;
use codex_chat_completions::UsageDetails;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::model_inference::KimiInferenceConfig;
use codex_protocol::model_inference::KimiThinkingPolicy;
use codex_protocol::model_inference::WireApi;
use codex_protocol::models::ResponseItem;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;

use super::*;

fn dialect(model: &str) -> KimiDialect {
    dialect_result(model).expect("valid Kimi dialect")
}

fn dialect_result(model: &str) -> Result<KimiDialect, KimiError> {
    KimiDialect::new(
        KimiInferenceConfig {
            wire_api: WireApi::ChatCompletions,
            dialect: InferenceDialect::Kimi,
            route: "unactivated-native-route".to_string(),
            wire_model: model.to_string(),
            max_output_tokens: 131_072,
            thinking: KimiThinkingPolicy::RequiredWithEffort,
        },
        KimiRequestSettings {
            context_window: 262_144,
            estimated_input_tokens: 1,
            prompt_cache_key: "stable-affinity".to_string(),
            thinking: KimiThinking::Effort(KimiThinkingEffort::High),
        },
    )
}

fn frame(delta: Value, finish_reason: Value, usage: Option<Value>) -> String {
    let mut choice = json!({"index":0,"delta":delta,"finish_reason":finish_reason});
    if let Some(usage) = usage {
        choice["usage"] = usage;
    }
    format!("data: {}\n\n", json!({"id":"chat-1","choices":[choice]}))
}

fn decode(
    dialect: &KimiDialect,
    body: &str,
    trace_id: Option<&str>,
) -> Result<DecodedStream, DecodeError> {
    codex_chat_completions::decode_stream(
        DecodeStream {
            context: dialect.context(),
            dialect,
            metadata: ResponseMetadata {
                trace_id: trace_id.map(str::to_string),
            },
        },
        [body],
        |_| {},
    )
}

fn request(dialect: &KimiDialect, history: &[ResponseItem]) -> Value {
    serde_json::to_value(
        encode_request(
            dialect,
            KimiEncodeRequest {
                system: None,
                history,
                tools: &[],
            },
        )
        .expect("Kimi request"),
    )
    .expect("request JSON")
}

fn append_results(history: &mut Vec<ResponseItem>, calls: &[ToolCall]) {
    history.extend(calls.iter().map(|call| {
        serde_json::from_value(json!({
            "type":"function_call_output",
            "call_id":call.id,
            "output":format!("result-{}", call.id)
        }))
        .expect("tool result")
    }));
}

#[test]
fn learns_and_replays_each_supported_reasoning_representation_including_empty() {
    for key in ["reasoning_content", "reasoning", "reasoning_details"] {
        let dialect = dialect("k3");
        let body = frame(json!({(key):"observed"}), "stop".into(), None) + "data: [DONE]\n\n";
        let pending = decode(&dialect, &body, None)
            .expect("valid stream")
            .pending
            .expect("committed pending result");
        assert_eq!(pending.reasoning_provenance.as_deref(), Some(key));
        let history = response_items(&dialect, pending).expect("canonical response");
        assert_eq!(
            request(&dialect, &history)["messages"],
            json!([
                {"role":"assistant", (key):"observed"}
            ])
        );

        let empty = response_items(
            &dialect,
            PendingResult {
                content: "text only".to_string(),
                reasoning: String::new(),
                reasoning_provenance: Some(key.to_string()),
                tool_calls: Vec::new(),
            },
        )
        .expect("empty reasoning response");
        assert_eq!(
            request(&dialect, &empty)["messages"],
            json!([
                {"role":"assistant", "content":"text only", (key):""}
            ])
        );
    }

    let dialect = dialect("k3");
    let body = frame(
        json!({"reasoning_content":null,"reasoning":"fallback"}),
        "stop".into(),
        None,
    ) + "data: [DONE]\n\n";
    assert_eq!(
        decode(&dialect, &body, None)
            .unwrap()
            .pending
            .unwrap()
            .reasoning_provenance,
        Some("reasoning".to_string())
    );
}

#[test]
fn exact_model_and_dialect_boundaries_discard_native_reasoning_marker() {
    let source = dialect("k3");
    let calls = vec![ToolCall {
        id: "kept-call".to_string(),
        kind: ToolCallKind::Function,
        function: ToolCallFunction {
            name: "inspect".to_string(),
            arguments: "{}".to_string(),
        },
    }];
    let mut native = response_items(
        &source,
        PendingResult {
            content: "compatible text".to_string(),
            reasoning: "private thought".to_string(),
            reasoning_provenance: Some("reasoning".to_string()),
            tool_calls: calls.clone(),
        },
    )
    .unwrap();
    append_results(&mut native, &calls);
    assert_eq!(
        request(&source, &native)["messages"],
        json!([
            {"role":"assistant","content":"compatible text","reasoning":"private thought","tool_calls":[
                {"id":"kept-call","type":"function","function":{"name":"inspect","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"kept-call","content":"result-kept-call"}
        ])
    );
    assert_eq!(
        request(&dialect("k3-256k"), &native)["messages"],
        json!([
            {"role":"assistant","content":"compatible text","reasoning_content":"","tool_calls":[
                {"id":"kept-call","type":"function","function":{"name":"inspect","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"kept-call","content":"result-kept-call"}
        ])
    );

    let mut wrong_dialect = serde_json::to_value(&native).unwrap();
    let marker = wrong_dialect[0]["encrypted_content"]
        .as_str()
        .unwrap()
        .replace("kimi_chat", "kimi_messages");
    wrong_dialect[0]["encrypted_content"] = marker.into();
    let wrong_dialect: Vec<ResponseItem> = serde_json::from_value(wrong_dialect).unwrap();
    assert_eq!(
        request(&source, &wrong_dialect)["messages"],
        json!([
            {"role":"assistant","content":"compatible text","reasoning_content":"","tool_calls":[
                {"id":"kept-call","type":"function","function":{"name":"inspect","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"kept-call","content":"result-kept-call"}
        ])
    );

    for (wire_api, inference_dialect) in [
        (WireApi::AnthropicMessages, InferenceDialect::Kimi),
        (WireApi::Responses, InferenceDialect::Kimi),
        (WireApi::ChatCompletions, InferenceDialect::OpenAi),
    ] {
        assert!(matches!(
            KimiDialect::new(
                KimiInferenceConfig {
                    wire_api,
                    dialect: inference_dialect,
                    route: "other-boundary".into(),
                    wire_model: "k3".into(),
                    max_output_tokens: 1,
                    thinking: KimiThinkingPolicy::RequiredWithEffort,
                },
                KimiRequestSettings {
                    context_window: 1,
                    estimated_input_tokens: 0,
                    prompt_cache_key: "key".into(),
                    thinking: KimiThinking::Effort(KimiThinkingEffort::High)
                }
            ),
            Err(KimiError::InvalidProfile(_))
        ));
    }
}

#[test]
fn pre_native_encrypted_history_stays_opaque_then_native_result_starts_marker() {
    let dialect = dialect("k3");
    let mut history: Vec<ResponseItem> = serde_json::from_value(json!([
        {"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"visible legacy"}],"encrypted_content":"responses-encrypted-secret"},
        {"type":"message","role":"assistant","content":[{"type":"output_text","text":"kept text"}]},
        {"type":"function_call","name":"old","arguments":"{}","call_id":"old-call"},
        {"type":"function_call_output","call_id":"old-call","output":"kept result"}
    ])).unwrap();
    let first = request(&dialect, &history);
    assert_eq!(
        first["messages"],
        json!([
            {"role":"assistant","content":"kept text","reasoning_content":"visible legacy","tool_calls":[
                {"id":"old-call","type":"function","function":{"name":"old","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"old-call","content":"kept result"}
        ])
    );
    assert!(!first.to_string().contains("responses-encrypted-secret"));

    let body = frame(
        json!({"reasoning":"native","content":"next"}),
        "stop".into(),
        None,
    ) + "data: [DONE]\n\n";
    let pending = decode(&dialect, &body, None).unwrap().pending.unwrap();
    history.extend(response_items(&dialect, pending).unwrap());
    let second = request(&dialect, &history);
    assert_eq!(second["messages"][0]["reasoning"], "visible legacy");
    assert_eq!(
        second["messages"][2],
        json!({
            "role":"assistant","content":"next","reasoning":"native"
        })
    );
    assert!(!second.to_string().contains("encrypted"));
}

#[test]
fn anthropic_signed_and_redacted_thinking_never_becomes_kimi_plaintext() {
    let dialect = dialect("k3");
    let signed: Vec<ResponseItem> = serde_json::from_value(json!([
        {"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"claude private thought"}],
         "encrypted_content":"codex:anthropic-thinking:{\"version\":1,\"dialect\":\"claude_code\",\"model\":\"claude\",\"blocks\":[{\"type\":\"signed\",\"thinking\":\"claude private thought\",\"signature\":\"claude-signature\"}]}"},
        {"type":"message","role":"assistant","content":[{"type":"output_text","text":"compatible text"}]},
        {"type":"function_call","name":"inspect","arguments":"{}","call_id":"kept-call"},
        {"type":"function_call_output","call_id":"kept-call","output":"compatible result"}
    ])).unwrap();
    let redacted: Vec<ResponseItem> = serde_json::from_value(json!([
        {"type":"reasoning","summary":[],
         "encrypted_content":"codex:anthropic-thinking:{\"version\":1,\"dialect\":\"claude_code\",\"model\":\"claude\",\"blocks\":[{\"type\":\"redacted\",\"data\":\"claude-redacted-secret\"}]}"},
        {"type":"message","role":"assistant","content":[{"type":"output_text","text":"compatible text"}]},
        {"type":"function_call","name":"inspect","arguments":"{}","call_id":"kept-call"},
        {"type":"function_call_output","call_id":"kept-call","output":"compatible result"}
    ])).unwrap();
    for history in [&signed, &redacted] {
        let encoded = request(&dialect, history);
        assert_eq!(
            encoded["messages"],
            json!([
                {"role":"assistant","content":"compatible text","reasoning_content":"","tool_calls":[
                    {"id":"kept-call","type":"function","function":{"name":"inspect","arguments":"{}"}}
                ]},
                {"role":"tool","tool_call_id":"kept-call","content":"compatible result"}
            ])
        );
        let wire = encoded.to_string();
        for secret in [
            "claude private thought",
            "claude-signature",
            "claude-redacted-secret",
        ] {
            assert!(!wire.contains(secret));
        }
        assert!(!wire.contains("codex:anthropic-thinking:"));
    }
}

#[test]
fn accepted_escaped_models_always_produce_bounded_self_decodable_markers() {
    let mut last_accepted = None;
    for length in 1..=256 {
        let model = "\0".repeat(length);
        match dialect_result(&model) {
            Ok(dialect) => last_accepted = Some((length, dialect)),
            Err(KimiError::InvalidProfile(_)) => break,
            Err(error) => panic!("unexpected constructor error: {error}"),
        }
    }
    let (accepted_length, dialect) = last_accepted.expect("one escaped model must fit");
    for key in ["reasoning_content", "reasoning", "reasoning_details"] {
        let history = response_items(
            &dialect,
            PendingResult {
                content: "visible".into(),
                reasoning: "bounded".into(),
                reasoning_provenance: Some(key.into()),
                tool_calls: Vec::new(),
            },
        )
        .expect("accepted models must produce bounded markers");
        let marker = match &history[0] {
            ResponseItem::Reasoning {
                encrypted_content: Some(marker),
                ..
            } => marker,
            item => panic!("unexpected marker item: {item:?}"),
        };
        assert!(marker.len() <= 512);
        assert_eq!(
            request(&dialect, &history)["messages"],
            json!([
                {"role":"assistant","content":"visible",(key):"bounded"}
            ])
        );
    }
    assert!(matches!(
        dialect_result(&"\0".repeat(accepted_length + 1)),
        Err(KimiError::InvalidProfile(
            "wire model cannot fit the bounded replay marker"
        ))
    ));
    assert!(matches!(
        dialect_result(&"\\".repeat(256)),
        Err(KimiError::InvalidProfile(
            "wire model cannot fit the bounded replay marker"
        ))
    ));
}

#[test]
fn reasoning_parallel_calls_usage_and_trace_round_trip_without_metadata_leakage() {
    let dialect = dialect("k3");
    let usage = json!({
        "prompt_tokens":20,"completion_tokens":9,"total_tokens":29,
        "prompt_tokens_details":{"cached_tokens":7},
        "completion_tokens_details":{"reasoning_tokens":4}
    });
    let body = [
        frame(json!({"reasoning_content":"think","tool_calls":[
            {"index":1,"id":"call-b","type":"function","function":{"name":"beta","arguments":"{\"b\":"}}
        ]}), Value::Null, None),
        frame(json!({"tool_calls":[
            {"index":0,"id":"call-a","type":"function","function":{"name":"alpha","arguments":"{\"a\":"}},
            {"index":1,"function":{"arguments":"2}"}}
        ]}), Value::Null, None),
        format!("data: {}\n\n", json!({"id":"chat-1","choices":[{
            "index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},
            "finish_reason":"tool_calls","usage":usage
        }],"usage":usage})),
        "data: [DONE]\n\n".to_string(),
    ].concat();
    let decoded = decode(&dialect, &body, Some("trace-42")).unwrap();
    assert_eq!(
        decoded.usage_details,
        UsageDetails {
            cached_prompt_tokens: 7,
            reasoning_tokens: 4
        }
    );
    assert_eq!(
        decoded.metadata,
        ResponseMetadata {
            trace_id: Some("trace-42".into())
        }
    );
    let pending = decoded.pending.unwrap();
    let calls = pending.tool_calls.clone();
    let mut history = response_items(&dialect, pending).unwrap();
    append_results(&mut history, &calls);
    let next = request(&dialect, &history);
    assert_eq!(
        next["messages"],
        json!([
            {"role":"assistant","reasoning_content":"think","tool_calls":[
                {"id":"call-b","type":"function","function":{"name":"beta","arguments":"{\"b\":2}"}},
                {"id":"call-a","type":"function","function":{"name":"alpha","arguments":"{\"a\":1}"}}
            ]},
            {"role":"tool","tool_call_id":"call-b","content":"result-call-b"},
            {"role":"tool","tool_call_id":"call-a","content":"result-call-a"}
        ])
    );
    assert!(!next.to_string().contains("trace-42"));
}

#[test]
fn noncommittable_terminals_and_protocol_failures_produce_no_marker_or_output() {
    let dialect = dialect("k3");
    let cases = [
        (
            frame(json!({"reasoning":"partial"}), "length".into(), None) + "data: [DONE]\n\n",
            None,
        ),
        (
            frame(
                json!({"reasoning":"partial"}),
                "content_filter".into(),
                None,
            ) + "data: [DONE]\n\n",
            None,
        ),
        (
            frame(json!({"reasoning":"partial"}), "future".into(), None) + "data: [DONE]\n\n",
            Some(DecodeError::UnknownFinishReason("future".into())),
        ),
        (
            frame(json!({"reasoning":"partial"}), "stop".into(), None),
            Some(DecodeError::PrematureEof {
                expected: "[DONE]".into(),
            }),
        ),
        (
            "data: {bad}\n\n".to_string(),
            Some(DecodeError::MalformedChunk(
                "key must be a string at line 1 column 2".into(),
            )),
        ),
    ];
    for (body, expected_error) in cases {
        match (decode(&dialect, &body, None), expected_error) {
            (Ok(decoded), None) => assert_eq!(decoded.pending, None),
            (Err(error), Some(expected)) => assert_eq!(error, expected),
            (result, expected) => panic!("unexpected result {result:?}, expected {expected:?}"),
        }
    }
    assert_eq!(request(&dialect, &[])["messages"], json!([]));
}
