use codex_protocol::mcp::CallToolResult;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseInputItem;
use codex_tools::JsonSchema;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolSpec;
use pretty_assertions::assert_eq;
use serde_json::json;

use super::response_input_contains_encrypted_tool_content;
use super::strip_encrypted_parameters;
use super::validate_model_visible_function_names;

#[test]
fn strips_encrypted_properties_and_required_entries_recursively() {
    let mut schema: JsonSchema = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "secret": {"type": "string", "encrypted": true},
            "safe": {"type": "string"},
            "nested": {
                "type": "object",
                "properties": {
                    "secret": {"type": "string", "encrypted": true},
                    "safe": {"type": "string"}
                },
                "required": ["secret", "safe"]
            }
        },
        "required": ["secret", "safe", "nested"]
    }))
    .expect("schema should deserialize");

    strip_encrypted_parameters(&mut schema);

    assert_eq!(
        serde_json::to_value(schema).expect("schema should serialize"),
        json!({
            "type": "object",
            "properties": {
                "safe": {"type": "string"},
                "nested": {
                    "type": "object",
                    "properties": {"safe": {"type": "string"}},
                    "required": ["safe"]
                }
            },
            "required": ["safe", "nested"]
        })
    );
}

#[test]
fn detects_encrypted_model_facing_tool_results() {
    let encrypted_function_output = ResponseInputItem::FunctionCallOutput {
        call_id: "call-1".to_string(),
        output: FunctionCallOutputPayload::from_content_items(vec![
            FunctionCallOutputContentItem::EncryptedContent {
                encrypted_content: "enc_opaque".to_string(),
            },
        ]),
    };
    let encrypted_mcp_output = ResponseInputItem::McpToolCallOutput {
        call_id: "call-2".to_string(),
        output: CallToolResult {
            content: vec![json!({
                "type": "text",
                "text": "enc_opaque",
                "_meta": {"codex/encryptedContent": true}
            })],
            structured_content: None,
            is_error: None,
            meta: None,
        },
    };
    let plaintext_output = ResponseInputItem::FunctionCallOutput {
        call_id: "call-3".to_string(),
        output: FunctionCallOutputPayload::from_text("plain".to_string()),
    };

    assert!(response_input_contains_encrypted_tool_content(
        &encrypted_function_output
    ));
    assert!(response_input_contains_encrypted_tool_content(
        &encrypted_mcp_output
    ));
    assert!(!response_input_contains_encrypted_tool_content(
        &plaintext_output
    ));
}

#[test]
#[should_panic(expected = "already exposed to the model")]
fn rejects_duplicate_flat_function_names() {
    validate_model_visible_function_names(vec![
        function_spec("duplicate"),
        function_spec("duplicate"),
    ]);
}

#[test]
#[should_panic(expected = "exceeds the 64-byte wire limit")]
fn rejects_function_names_over_the_wire_limit() {
    validate_model_visible_function_names(vec![function_spec(&"x".repeat(65))]);
}

fn function_spec(name: &str) -> ToolSpec {
    ToolSpec::Function(ResponsesApiTool {
        name: name.to_string(),
        description: String::new(),
        strict: false,
        defer_loading: None,
        parameters: JsonSchema::default(),
        output_schema: None,
    })
}
