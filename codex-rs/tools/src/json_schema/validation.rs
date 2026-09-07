//! Validate model-visible function-call arguments against advertised tool schemas.

use super::AdditionalProperties;
use super::JsonSchema;
use super::JsonSchemaPrimitiveType;
use super::JsonSchemaType;
use serde_json::Value as JsonValue;

const MAX_SCHEMA_REF_DEPTH: usize = 32;

/// Validate JSON arguments against a tool's advertised input schema.
pub fn validate_tool_input(schema: &JsonSchema, input: &JsonValue) -> Result<(), String> {
    validate_against(schema, input, schema, 0)
}

fn validate_against(
    schema: &JsonSchema,
    value: &JsonValue,
    root: &JsonSchema,
    depth: usize,
) -> Result<(), String> {
    if depth > MAX_SCHEMA_REF_DEPTH {
        return Err("schema validation exceeded maximum $ref depth".to_string());
    }

    if let Some(schema_ref) = &schema.schema_ref {
        let resolved = resolve_ref(root, schema_ref)?;
        validate_against(resolved, value, root, depth + 1)?;
    }

    if let Some(any_of) = &schema.any_of
        && !any_of
            .iter()
            .any(|variant| validate_against(variant, value, root, depth + 1).is_ok())
    {
        return Err("value did not match any anyOf schema".to_string());
    }
    if let Some(one_of) = &schema.one_of {
        let matches = one_of
            .iter()
            .filter(|variant| validate_against(variant, value, root, depth + 1).is_ok())
            .count();
        if matches != 1 {
            return Err("value did not match exactly one oneOf schema".to_string());
        }
    }
    if let Some(all_of) = &schema.all_of {
        for variant in all_of {
            validate_against(variant, value, root, depth + 1)?;
        }
    }

    if let Some(enum_values) = &schema.enum_values
        && !enum_values.contains(value)
    {
        return Err("value is not one of the allowed enum values".to_string());
    }

    if let Some(schema_type) = &schema.schema_type {
        match schema_type {
            JsonSchemaType::Single(expected) => check_type(*expected, value)?,
            JsonSchemaType::Multiple(expected) => {
                if !expected
                    .iter()
                    .any(|candidate| check_type(*candidate, value).is_ok())
                {
                    return Err("value does not match any allowed type".to_string());
                }
            }
        }
    }

    match value {
        JsonValue::Object(map) => {
            if let Some(required) = &schema.required {
                for key in required {
                    if !map.contains_key(key) {
                        return Err(format!("missing required property `{key}`"));
                    }
                }
            }
            for (key, property_value) in map {
                if let Some(property_schema) = schema
                    .properties
                    .as_ref()
                    .and_then(|properties| properties.get(key))
                {
                    validate_against(property_schema, property_value, root, depth + 1)?;
                    continue;
                }
                match &schema.additional_properties {
                    Some(AdditionalProperties::Boolean(false)) => {
                        return Err(format!("unexpected property `{key}`"));
                    }
                    Some(AdditionalProperties::Schema(additional_schema)) => {
                        validate_against(additional_schema, property_value, root, depth + 1)?;
                    }
                    Some(AdditionalProperties::Boolean(true)) | None => {}
                }
            }
        }
        JsonValue::Array(items) => {
            if let Some(min_items) = schema.min_items
                && items.len() < min_items
            {
                return Err(format!("array shorter than minItems {min_items}"));
            }
            if let Some(item_schema) = &schema.items {
                for item in items {
                    validate_against(item_schema, item, root, depth + 1)?;
                }
            }
        }
        JsonValue::Null | JsonValue::Bool(_) | JsonValue::Number(_) | JsonValue::String(_) => {}
    }

    Ok(())
}

fn check_type(expected: JsonSchemaPrimitiveType, value: &JsonValue) -> Result<(), String> {
    let matches = match expected {
        JsonSchemaPrimitiveType::String => value.is_string(),
        JsonSchemaPrimitiveType::Number => value.is_number(),
        JsonSchemaPrimitiveType::Integer => match value {
            JsonValue::Number(number) => {
                number.is_i64()
                    || number.is_u64()
                    || number.as_f64().is_some_and(|number| number.fract() == 0.0)
            }
            _ => false,
        },
        JsonSchemaPrimitiveType::Boolean => value.is_boolean(),
        JsonSchemaPrimitiveType::Object => value.is_object(),
        JsonSchemaPrimitiveType::Array => value.is_array(),
        JsonSchemaPrimitiveType::Null => value.is_null(),
    };
    if matches {
        Ok(())
    } else {
        Err(format!("expected {expected:?}"))
    }
}

fn resolve_ref<'a>(root: &'a JsonSchema, schema_ref: &str) -> Result<&'a JsonSchema, String> {
    let fragment = schema_ref.strip_prefix('#').unwrap_or(schema_ref);
    let decoded = urlencoding::decode(fragment).map_err(|error| error.to_string())?;
    let mut parts = decoded.trim_start_matches('/').split('/');
    let table = parts.next().unwrap_or_default();
    let Some(name) = parts.next() else {
        return Err(format!("unresolved $ref `{schema_ref}`"));
    };
    let table = match table {
        "$defs" => root.defs.as_ref(),
        "definitions" => root.definitions.as_ref(),
        _ => None,
    };
    let Some(schema) = table.and_then(|table| table.get(name)) else {
        return Err(format!("unresolved $ref `{schema_ref}`"));
    };
    if parts.next().is_some() {
        return Err(format!("unsupported nested $ref `{schema_ref}`"));
    }
    Ok(schema)
}

#[cfg(test)]
#[path = "validation_tests.rs"]
mod tests;
