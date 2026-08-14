use codex_tools::JsonSchema;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolSpec;
use pretty_assertions::assert_eq;
use serde_json::json;

use super::strip_encrypted_parameters;
use super::validate_model_visible_function_names;

#[test]
fn strips_encrypted_properties_and_required_entries_recursively() {
    let mut schema: JsonSchema = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "message": {"type": "string", "encrypted": true},
            "plaintext_message": {"type": "string"},
            "nested": {
                "type": "object",
                "properties": {
                    "secret": {"type": "string", "encrypted": true},
                    "safe": {"type": "string"}
                },
                "required": ["secret", "safe"]
            }
        },
        "required": ["message", "plaintext_message", "nested"]
    }))
    .expect("schema should deserialize");

    strip_encrypted_parameters(&mut schema);

    assert_eq!(
        serde_json::to_value(schema).expect("schema should serialize"),
        json!({
            "type": "object",
            "properties": {
                "plaintext_message": {"type": "string"},
                "nested": {
                    "type": "object",
                    "properties": {"safe": {"type": "string"}},
                    "required": ["safe"]
                }
            },
            "required": ["plaintext_message", "nested"]
        })
    );
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
        local_result_schema: None,
    })
}
