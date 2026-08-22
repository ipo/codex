use std::collections::BTreeMap;

use pretty_assertions::assert_eq;
use serde_json::json;

use super::*;

fn schema() -> JsonSchema {
    JsonSchema::object(
        BTreeMap::from([
            (
                "path".to_string(),
                JsonSchema::string(Some("Project-relative path".to_string())),
            ),
            (
                "attempts".to_string(),
                JsonSchema::integer(Some("Retry count".to_string())),
            ),
        ]),
        Some(vec!["path".to_string()]),
        Some(false.into()),
    )
}

#[test]
fn validates_required_types_and_unknown_properties() {
    let schema = schema();
    assert_eq!(
        validate_tool_input(&schema, &json!({"path":"src/lib.rs"})),
        Ok(())
    );
    assert_eq!(
        validate_tool_input(&schema, &json!({"attempts":1})),
        Err("$: missing required property `path`".to_string())
    );
    assert_eq!(
        validate_tool_input(&schema, &json!({"path":7})),
        Err("$.path: expected string, got number".to_string())
    );
    assert_eq!(
        validate_tool_input(&schema, &json!({"path":"src/lib.rs","extra":true})),
        Err("$: unknown property `extra`".to_string())
    );
}

#[test]
fn validates_compositions_and_local_refs() {
    let schema: JsonSchema = serde_json::from_value(json!({
        "type": "object",
        "properties": {"mode": {"$ref": "#/$defs/mode"}},
        "required": ["mode"],
        "additionalProperties": false,
        "$defs": {
            "mode": {"oneOf": [
                {"type":"string","enum":["fast"]},
                {"type":"integer","minimum":1,"maximum":3}
            ]}
        }
    }))
    .expect("schema");

    assert_eq!(
        validate_tool_input(&schema, &json!({"mode":"fast"})),
        Ok(())
    );
    assert_eq!(validate_tool_input(&schema, &json!({"mode":2})), Ok(()));
    assert!(validate_tool_input(&schema, &json!({"mode":4})).is_err());
}

#[test]
fn validates_numeric_extensions_and_fails_closed_for_unsupported_keywords() {
    let constrained: JsonSchema = serde_json::from_value(json!({
        "type": "number",
        "minimum": 1,
        "maximum": 5,
        "multipleOf": 0.5
    }))
    .expect("schema");
    assert_eq!(validate_tool_input(&constrained, &json!(2.5)), Ok(()));
    assert!(validate_tool_input(&constrained, &json!(2.25)).is_err());

    let unsupported: JsonSchema = serde_json::from_value(json!({
        "type": "string",
        "pattern": "^[a-z]+$"
    }))
    .expect("schema");
    assert_eq!(
        validate_tool_input(&unsupported, &json!("safe")),
        Err("$: advertised tool schema uses unsupported validation keyword `pattern`".to_string())
    );
}

#[test]
fn enum_and_unique_items_use_mathematical_number_equality() {
    let schema: JsonSchema = serde_json::from_value(json!({
        "enum": [{"value": 1}]
    }))
    .expect("schema");
    let equivalent = serde_json::from_str(r#"{"value":1.0}"#).expect("input");
    assert_eq!(validate_tool_input(&schema, &equivalent), Ok(()));

    let unique: JsonSchema = serde_json::from_value(json!({
        "type": "array",
        "uniqueItems": true
    }))
    .expect("schema");
    let duplicate = serde_json::from_str("[1,1.0]").expect("input");
    assert_eq!(
        validate_tool_input(&unique, &duplicate),
        Err("$: array items must be unique".to_string())
    );
}

#[test]
fn large_integer_bounds_do_not_round_through_f64() {
    let bounded: JsonSchema = serde_json::from_value(json!({
        "type": "integer",
        "maximum": 9_007_199_254_740_992_u64
    }))
    .expect("schema");
    assert_eq!(
        validate_tool_input(&bounded, &json!(9_007_199_254_740_992_u64)),
        Ok(())
    );
    assert_eq!(
        validate_tool_input(&bounded, &json!(9_007_199_254_740_993_u64)),
        Err("$: number violates `maximum`".to_string())
    );

    let multiple: JsonSchema = serde_json::from_value(json!({
        "type": "integer",
        "multipleOf": 9_007_199_254_740_992_u64
    }))
    .expect("schema");
    assert_eq!(
        validate_tool_input(&multiple, &json!(9_007_199_254_740_993_u64)),
        Err("$: number violates `multipleOf`".to_string())
    );
}

#[test]
fn fractional_and_exponent_constraints_are_exact() {
    let schema: JsonSchema = serde_json::from_str(
        r#"{
            "type":"number",
            "minimum":1e-3,
            "maximum":3e-2,
            "multipleOf":1e-3
        }"#,
    )
    .expect("schema");
    assert_eq!(
        validate_tool_input(&schema, &serde_json::from_str("3e-2").expect("input")),
        Ok(())
    );
    assert_eq!(
        validate_tool_input(&schema, &serde_json::from_str("3.1e-2").expect("input")),
        Err("$: number violates `maximum`".to_string())
    );
    assert_eq!(
        validate_tool_input(&schema, &serde_json::from_str("2.55e-2").expect("input")),
        Err("$: number violates `multipleOf`".to_string())
    );
}
