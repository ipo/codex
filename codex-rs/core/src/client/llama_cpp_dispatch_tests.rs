use std::sync::Arc;

use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::model_inference::LlamaCppInferenceConfig;
use codex_protocol::model_inference::ModelInferenceConfig;
use codex_protocol::model_inference::WireApi;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::InputModality;
use codex_tools::JsonSchema;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolSpec;
use pretty_assertions::assert_eq;
use serde_json::json;

use crate::client_common::Prompt;
use crate::context_manager::ContextManager;

use super::request::normalize_input;
use super::request::validate_tools;

#[test]
fn normalizes_text_history_for_plain_llama_cpp_responses() {
    let prompt = Prompt {
        input: response_items(json!([
            {
                "id": "msg_system",
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": "initial policy"}],
                "phase": "commentary",
                "internal_chat_message_metadata_passthrough": {"turn_id": "turn-1"}
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "start"}]
            },
            {
                "type": "message",
                "role": "developer",
                "content": [
                    {"type": "input_text", "text": "late"},
                    {"type": "output_text", "text": "policy"}
                ],
                "internal_chat_message_metadata_passthrough": {"turn_id": "turn-2"}
            },
            {
                "type": "agent_message",
                "author": "/root",
                "recipient": "/root/worker",
                "content": [
                    {"type": "input_text", "text": "plain"},
                    {"type": "input_text", "text": "handoff"}
                ]
            },
            {
                "id": "rs_reasoning",
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": "use the tool"}],
                "encrypted_content": "",
                "internal_chat_message_metadata_passthrough": {"turn_id": "turn-3"}
            },
            {
                "id": "fc_call",
                "type": "function_call",
                "name": "read_file",
                "arguments": "{\"path\":\"Cargo.toml\"}",
                "call_id": "call-1"
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": [
                    {"type": "input_text", "text": "workspace"},
                    {"type": "input_text", "text": "contents"}
                ],
                "internal_chat_message_metadata_passthrough": {"turn_id": "turn-4"}
            }
        ])),
        ..Default::default()
    };

    assert_eq!(
        normalize_input(&prompt).expect("supported history should normalize"),
        response_items(json!([
            {
                "id": "msg_system",
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": "initial policy"}]
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "start"}]
            },
            {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": "<external_local_developer_update>late\npolicy</external_local_developer_update>"
                }]
            },
            {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": "Agent message from /root to /root/worker:\nplain\nhandoff"
                }]
            },
            {
                "id": "rs_reasoning",
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": "use the tool"}],
                "encrypted_content": ""
            },
            {
                "id": "fc_call",
                "type": "function_call",
                "name": "read_file",
                "arguments": "{\"path\":\"Cargo.toml\"}",
                "call_id": "call-1"
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "workspace\ncontents"
            }
        ]))
    );
}

#[test]
fn rejects_history_that_llama_cpp_cannot_represent() {
    let cases = [
        (
            json!({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA=="}]
            }),
            "unsupported image or audio input",
        ),
        (
            json!({
                "type": "agent_message",
                "author": "/root",
                "recipient": "/root/worker",
                "content": [{"type": "encrypted_content", "encrypted_content": "enc_opaque"}]
            }),
            "encrypted cross-agent message",
        ),
        (
            json!({
                "type": "reasoning",
                "summary": [],
                "encrypted_content": "enc_opaque"
            }),
            "unsupported encrypted reasoning",
        ),
        (
            json!({
                "type": "function_call",
                "namespace": "tools",
                "name": "read_file",
                "arguments": "{}",
                "call_id": "call-1"
            }),
            "unsupported namespace tool call",
        ),
        (
            json!({
                "type": "function_call_output",
                "call_id": "call-1",
                "output": [{"type": "encrypted_content", "encrypted_content": "enc_opaque"}]
            }),
            "unsupported media or encrypted content",
        ),
        (
            json!({"type": "unknown_item"}),
            "unsupported Responses item type",
        ),
    ];

    for (item, expected) in cases {
        let prompt = Prompt {
            input: response_items(json!([item])),
            ..Default::default()
        };
        let error = normalize_input(&prompt).expect_err("unsupported history should fail");
        assert!(
            error.to_string().contains(expected),
            "expected `{expected}` in `{error}`"
        );
    }
}

#[test]
fn production_projection_preserves_media_for_llama_cpp_rejection() {
    let mut model = codex_models_manager::model_info::model_info_from_slug("gpt-5.6-sol");
    model.input_modalities = vec![InputModality::Text];
    model.inference = Some(ModelInferenceConfig::LlamaCpp(LlamaCppInferenceConfig {
        wire_api: WireApi::Responses,
        dialect: InferenceDialect::LlamaCpp,
        route: "llama_cpp".to_string(),
        expected_model_basename: "qwen-test.gguf".to_string(),
        context_window: 32_768,
        max_input_tokens: 23_552,
        max_output_tokens: 8_192,
        safety_margin_tokens: 1_024,
    }));
    let cases = [
        (
            json!([{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA=="}]
            }]),
            "unsupported image or audio input",
        ),
        (
            json!([{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_audio", "audio_url": "data:audio/wav;base64,AA=="}]
            }]),
            "unsupported image or audio input",
        ),
        (
            json!([
                {
                    "type": "function_call",
                    "name": "inspect_media",
                    "arguments": "{}",
                    "call_id": "call-image"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-image",
                    "output": [{"type": "input_image", "image_url": "data:image/png;base64,AA=="}]
                }
            ]),
            "unsupported media or encrypted content",
        ),
        (
            json!([
                {
                    "type": "function_call",
                    "name": "inspect_media",
                    "arguments": "{}",
                    "call_id": "call-audio"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-audio",
                    "output": [{"type": "input_audio", "audio_url": "data:audio/wav;base64,AA=="}]
                }
            ]),
            "unsupported media or encrypted content",
        ),
    ];

    for (items, expected) in cases {
        let mut history = ContextManager::new();
        let items = response_items(items);
        history.record_items(
            items.iter(),
            codex_utils_output_truncation::TruncationPolicy::Tokens(10_000),
        );
        let prompt = Prompt {
            input: history.for_model_prompt(&model),
            ..Default::default()
        };

        let error = normalize_input(&prompt)
            .expect_err("production-projected unsupported media should fail");
        assert!(
            error.to_string().contains(expected),
            "expected `{expected}` in `{error}`"
        );
    }
}

#[test]
fn rejects_non_function_tool_specs() {
    let valid = Prompt {
        tools: Arc::from([function_spec("read_file")]),
        ..Default::default()
    };
    validate_tools(&valid).expect("plain function tool should be accepted");

    let invalid = Prompt {
        tools: Arc::from([ToolSpec::ToolSearch {
            execution: "client".to_string(),
            description: "search".to_string(),
            parameters: JsonSchema::default(),
        }]),
        ..Default::default()
    };
    assert_eq!(
        validate_tools(&invalid)
            .expect_err("tool search should be rejected")
            .to_string(),
        "direct llama.cpp Responses supports only plain function tools"
    );
}

fn response_items(value: serde_json::Value) -> Vec<ResponseItem> {
    serde_json::from_value(value).expect("valid response items")
}

fn function_spec(name: &str) -> ToolSpec {
    ToolSpec::Function(ResponsesApiTool {
        name: name.to_string(),
        description: String::new(),
        strict: false,
        parameters: JsonSchema::default(),
        output_schema: None,
        defer_loading: None,
    })
}
