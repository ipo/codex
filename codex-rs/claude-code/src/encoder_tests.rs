use codex_protocol::model_inference::AnthropicThinkingPolicy;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::model_inference::ModelInferenceConfig;
use codex_protocol::model_inference::WireApi;
use codex_protocol::models::AgentMessageInputContent;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ReasoningEffort;
use pretty_assertions::assert_eq;
use serde_json::json;

use super::*;

const SESSION: &str = "019fbf00-0000-7000-8000-000000000044";

fn profile(model: &str) -> ModelInferenceConfig {
    profile_with_dialect(model, InferenceDialect::ClaudeCode)
}

fn profile_with_dialect(model: &str, dialect: InferenceDialect) -> ModelInferenceConfig {
    ModelInferenceConfig::Anthropic {
        wire_api: WireApi::AnthropicMessages,
        dialect,
        route: "claude_code".to_string(),
        wire_model: model.to_string(),
        max_output_tokens: 64_000,
        thinking: AnthropicThinkingPolicy::Adaptive,
        supports_disabled_thinking: true,
    }
}

fn message(role: &str, content: Vec<ContentItem>) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: role.to_string(),
        content,
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }
}

fn call(call_id: &str, name: &str, arguments: &str) -> ResponseItem {
    ResponseItem::FunctionCall {
        id: None,
        name: name.to_string(),
        namespace: None,
        arguments: arguments.to_string(),
        call_id: call_id.to_string(),
        internal_chat_message_metadata_passthrough: None,
    }
}

fn result(call_id: &str, body: FunctionCallOutputBody, success: bool) -> ResponseItem {
    ResponseItem::FunctionCallOutput {
        id: None,
        call_id: call_id.to_string(),
        output: FunctionCallOutputPayload {
            body,
            success: Some(success),
        },
        internal_chat_message_metadata_passthrough: None,
    }
}

fn encode_for(
    profile: &ModelInferenceConfig,
    history: &[ResponseItem],
) -> Result<AssembledRequest, EncodeError> {
    let opus_compatibility = match profile {
        ModelInferenceConfig::Anthropic { wire_model, .. } if wire_model == "claude-opus-5" => {
            Some(OpusCompatibilityContext {
                kind: OpusRequestKind::Root,
                session_id: SESSION.to_string(),
                thread_id: "019fbf00-0000-7000-8000-000000000045".to_string(),
                installation_id: "019fbf00-0000-7000-8000-000000000046".to_string(),
                environment: OpusEnvironment {
                    cwd: "/workspace/project".to_string(),
                    is_git_repository: true,
                    platform: "linux".to_string(),
                    architecture: "x86_64".to_string(),
                    shell: "bash".to_string(),
                    os_version: "Linux 6.17.0-test".to_string(),
                },
            })
        }
        ModelInferenceConfig::Anthropic { .. }
        | ModelInferenceConfig::OpenAi { .. }
        | ModelInferenceConfig::Kimi(_) => None,
    };
    encode_request(EncodeRequest {
        profile,
        effort: &ReasoningEffort::High,
        system: &[],
        history,
        tools: &[],
        output_schema: CanonicalOutputSchema::Disabled,
        resumable_session_id: SESSION,
        codex_version: "0.146.0",
        opus_compatibility: opus_compatibility.as_ref(),
        sonnet_compatibility: None,
    })
}

#[test]
fn structured_output_schema_is_rejected_before_request_assembly() {
    let model = profile("claude-sonnet-5");
    let schema = json!({"type": "object"});
    let result = encode_request(EncodeRequest {
        profile: &model,
        effort: &ReasoningEffort::High,
        system: &[],
        history: &[],
        tools: &[],
        output_schema: CanonicalOutputSchema::JsonSchema { schema: &schema },
        resumable_session_id: "invalid-if-assembly-runs",
        codex_version: "0.146.0",
        opus_compatibility: None,
        sonnet_compatibility: None,
    });

    assert_eq!(result, Err(EncodeError::UnsupportedStructuredOutput));
}

#[test]
fn encodes_plaintext_agent_messages_as_user_input_and_rejects_encrypted_content() {
    let model = profile("claude-sonnet-5");
    let plaintext = ResponseItem::AgentMessage {
        id: None,
        author: "/root".to_string(),
        recipient: "/root/worker".to_string(),
        content: vec![AgentMessageInputContent::InputText {
            text: "inspect the request".to_string(),
        }],
        internal_chat_message_metadata_passthrough: None,
    };

    let ordered = [
        message(
            "user",
            vec![ContentItem::InputText {
                text: "before".to_string(),
            }],
        ),
        plaintext,
        message(
            "assistant",
            vec![ContentItem::OutputText {
                text: "after".to_string(),
            }],
        ),
    ];
    assert_eq!(
        encode_for(&model, &ordered)
            .expect("encode plaintext agent message")
            .body
            .messages,
        vec![
            Message {
                role: Role::User,
                content: vec![
                    ContentBlock::Text {
                        text: "before".to_string(),
                        cache_control: None,
                    },
                    ContentBlock::Text {
                        text: "Agent message from /root to /root/worker:\ninspect the request"
                            .to_string(),
                        cache_control: Some(CacheControl::Ephemeral {
                            ttl: Some(CacheTtl::OneHour),
                        }),
                    },
                ],
            },
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: "after".to_string(),
                    cache_control: None,
                }],
            },
        ]
    );

    let encrypted = ResponseItem::AgentMessage {
        id: None,
        author: "/root".to_string(),
        recipient: "/root/worker".to_string(),
        content: vec![AgentMessageInputContent::EncryptedContent {
            encrypted_content: "must-not-leak".to_string(),
        }],
        internal_chat_message_metadata_passthrough: None,
    };
    assert_eq!(
        encode_for(&model, &[encrypted]),
        Err(EncodeError::UnsupportedHistoryItem {
            index: 0,
            kind: "non-plaintext structured agent message",
        })
    );
}

#[test]
fn snapshots_ordered_transcript_replay_images_results_and_orphans() {
    let native_model = profile("claude-sonnet-5");
    let replay = encode_thinking_replay(
        InferenceDialect::ClaudeCode,
        "claude-sonnet-5",
        vec![
            ThinkingReplayBlock::Signed {
                thinking: "think\0\nexact".to_string(),
                signature: "sig+/=\0".to_string(),
            },
            ThinkingReplayBlock::Redacted {
                data: "redacted+/=\0".to_string(),
            },
        ],
    )
    .expect("encode replay");
    let history = vec![
        message(
            "user",
            vec![
                ContentItem::InputText {
                    text: "inspect these".to_string(),
                },
                ContentItem::InputImage {
                    image_url: "data:image/png;base64,aGVsbG8=".to_string(),
                    detail: None,
                },
                ContentItem::InputText {
                    text: "in order".to_string(),
                },
            ],
        ),
        ResponseItem::Reasoning {
            id: None,
            summary: vec![],
            content: Some(vec![ReasoningItemContent::ReasoningText {
                text: "visible thought".to_string(),
            }]),
            encrypted_content: Some(replay),
            internal_chat_message_metadata_passthrough: None,
        },
        message(
            "assistant",
            vec![ContentItem::OutputText {
                text: "Checking both.".to_string(),
            }],
        ),
        call("call-a", "read", r#"{"path":"a"}"#),
        call("call-b", "read", r#"{"path":"b"}"#),
        result(
            "call-a",
            FunctionCallOutputBody::ContentItems(vec![
                FunctionCallOutputContentItem::InputText {
                    text: "first".to_string(),
                },
                FunctionCallOutputContentItem::InputImage {
                    image_url: "data:image/jpeg;base64,d29ybGQ=".to_string(),
                    detail: None,
                },
                FunctionCallOutputContentItem::InputText {
                    text: "after image".to_string(),
                },
            ]),
            true,
        ),
        result(
            "call-b",
            FunctionCallOutputBody::Text("failed".to_string()),
            false,
        ),
        call("call-c", "read", r#"{"path":"c"}"#),
        call("call-d", "read", r#"{"path":"d"}"#),
        result(
            "call-c",
            FunctionCallOutputBody::Text("completed".to_string()),
            true,
        ),
    ];

    let request = encode_for(&native_model, &history).expect("encode complete transcript");

    insta::assert_json_snapshot!("canonical_transcript", request.body.messages);
}

#[test]
fn replay_is_native_only_for_the_exact_model_and_dialect() {
    let replay = encode_thinking_replay(
        InferenceDialect::ClaudeCode,
        "claude-sonnet-5",
        vec![
            ThinkingReplayBlock::Signed {
                thinking: "exact thinking".to_string(),
                signature: "exact signature".to_string(),
            },
            ThinkingReplayBlock::Redacted {
                data: "must disappear after switch".to_string(),
            },
        ],
    )
    .expect("encode replay");
    let history = [ResponseItem::Reasoning {
        id: None,
        summary: vec![],
        content: Some(vec![ReasoningItemContent::ReasoningText {
            text: "visible fallback".to_string(),
        }]),
        encrypted_content: Some(replay),
        internal_chat_message_metadata_passthrough: None,
    }];

    let same = encode_for(&profile("claude-sonnet-5"), &history).expect("same model replay");
    assert_eq!(
        same.body.messages[0].content,
        vec![
            ContentBlock::Thinking {
                thinking: "exact thinking".to_string(),
                signature: "exact signature".to_string(),
            },
            ContentBlock::RedactedThinking {
                data: "must disappear after switch".to_string(),
            },
        ]
    );

    let switched = encode_for(&profile("claude-opus-5"), &history).expect("model downgrade");
    assert_eq!(
        switched.body.messages[0].content,
        vec![ContentBlock::Text {
            text: "visible fallback".to_string(),
            cache_control: None,
        }]
    );

    let dialect_switched = crate::history::encode_history(
        &history,
        &profile_with_dialect("claude-sonnet-5", InferenceDialect::OpenAi),
    )
    .expect("dialect downgrade");
    assert_eq!(dialect_switched, switched.body.messages);

    let malformed = [ResponseItem::Reasoning {
        id: None,
        summary: vec![],
        content: None,
        encrypted_content: Some("codex:anthropic-thinking:not-json".to_string()),
        internal_chat_message_metadata_passthrough: None,
    }];
    assert!(matches!(
        encode_for(&profile("claude-sonnet-5"), &malformed),
        Err(EncodeError::Replay(_))
    ));
}

#[test]
fn every_unsupported_canonical_item_and_content_kind_is_rejected() {
    let model = profile("claude-sonnet-5");
    let unsupported = [
        json!({"type":"additional_tools","role":"user","tools":[]}),
        json!({"type":"local_shell_call","call_id":"shell","status":"completed","action":{"type":"exec","command":["pwd"],"timeout_ms":null,"working_directory":null,"env":null,"user":null}}),
        json!({"type":"custom_tool_call","call_id":"free","name":"free","input":"raw"}),
        json!({"type":"custom_tool_call_output","call_id":"free","output":"raw"}),
        json!({"type":"tool_search_call","call_id":"search","execution":"server","arguments":{}}),
        json!({"type":"tool_search_output","call_id":"search","status":"completed","execution":"server","tools":[]}),
        json!({"type":"web_search_call","status":"completed","action":{"type":"search","query":"q"}}),
        json!({"type":"image_generation_call","status":"completed","result":"image"}),
        json!({"type":"compaction","encrypted_content":"opaque"}),
        json!({"type":"compaction_trigger"}),
        json!({"type":"context_compaction","encrypted_content":"opaque"}),
        json!({"type":"future_unknown_item"}),
    ];
    for value in unsupported {
        let item: ResponseItem = serde_json::from_value(value).expect("canonical item fixture");
        assert!(matches!(
            encode_for(&model, &[item]),
            Err(EncodeError::UnsupportedHistoryItem { index: 0, .. })
        ));
    }

    let unsupported_content = [
        json!({"type":"message","role":"user","content":[{"type":"input_audio","audio_url":"data:audio/wav;base64,AA=="}]}),
        json!({"type":"message","role":"user","content":[{"type":"output_text","text":"structured output"}]}),
        json!({"type":"message","role":"assistant","content":[{"type":"input_text","text":"wrong role"}]}),
    ];
    for value in unsupported_content {
        let item: ResponseItem = serde_json::from_value(value).expect("canonical content fixture");
        assert!(matches!(
            encode_for(&model, &[item]),
            Err(EncodeError::UnsupportedContent { .. })
        ));
    }
    let encrypted_output = serde_json::from_value(json!({
        "type":"function_call_output",
        "call_id":"call",
        "output":[{"type":"encrypted_content","encrypted_content":"opaque"}]
    }))
    .expect("canonical encrypted output fixture");
    let encrypted_result = [call("call", "read", "{}"), encrypted_output];
    assert!(matches!(
        encode_for(&model, &encrypted_result),
        Err(EncodeError::UnsupportedContent { .. })
    ));

    let namespaced = [ResponseItem::FunctionCall {
        id: None,
        name: "read".to_string(),
        namespace: Some("apps".to_string()),
        arguments: "{}".to_string(),
        call_id: "namespaced".to_string(),
        internal_chat_message_metadata_passthrough: None,
    }];
    assert!(matches!(
        encode_for(&model, &namespaced),
        Err(EncodeError::UnsupportedHistoryItem {
            kind: "namespace function call",
            ..
        })
    ));
}
