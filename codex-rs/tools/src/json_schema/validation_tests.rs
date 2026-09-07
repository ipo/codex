use super::validate_tool_input;
use crate::parse_tool_input_schema;
use serde_json::json;

fn advertised_schema() -> crate::JsonSchema {
    parse_tool_input_schema(&json!({
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": false
    }))
    .expect("schema should parse")
}

#[test]
fn accepts_advertised_schema_valid_object() {
    validate_tool_input(&advertised_schema(), &json!({"path": "Cargo.toml"}))
        .expect("schema-valid arguments should pass");
}

#[test]
fn rejects_type_mismatch_missing_required_and_unknown_properties() {
    let schema = advertised_schema();
    assert!(validate_tool_input(&schema, &json!({"path": 7})).is_err());
    assert!(validate_tool_input(&schema, &json!({})).is_err());
    assert!(validate_tool_input(&schema, &json!({"path": "Cargo.toml", "extra": true})).is_err());
}
