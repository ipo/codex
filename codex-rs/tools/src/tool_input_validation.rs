use serde_json::Value;

use crate::AdditionalProperties;
use crate::JsonSchema;
use crate::JsonSchemaPrimitiveType;
use crate::JsonSchemaType;

mod exact_number;

use self::exact_number::ExactJsonNumber;
use self::exact_number::contains_equivalent;

const MAX_VALIDATION_DEPTH: usize = 64;

/// Validates a function-tool input against the JSON Schema advertised to the model.
///
/// The validator covers the structural subset retained by [`JsonSchema`], including local refs,
/// type unions, enums, object/array constraints, and schema compositions. Unsupported refs or
/// validation keywords fail closed instead of allowing an unvalidated invocation through.
pub fn validate_tool_input(schema: &JsonSchema, input: &Value) -> Result<(), String> {
    let root = serde_json::to_value(schema)
        .map_err(|error| format!("failed to serialize advertised tool schema: {error}"))?;
    validate(schema, input, schema, &root, "$", /*depth*/ 0)
}

fn validate(
    schema: &JsonSchema,
    input: &Value,
    root_schema: &JsonSchema,
    root_value: &Value,
    path: &str,
    depth: usize,
) -> Result<(), String> {
    if depth > MAX_VALIDATION_DEPTH {
        return Err(format!("{path}: tool input exceeds maximum schema depth"));
    }
    if let Some(schema_ref) = schema.schema_ref.as_deref() {
        let referenced = resolve_local_ref(schema_ref, root_schema, root_value)?;
        validate(&referenced, input, root_schema, root_value, path, depth + 1)?;
    }
    validate_supported_extensions(schema, path)?;
    if let Some(enum_values) = &schema.enum_values
        && !contains_equivalent(enum_values, input)?
    {
        return Err(format!(
            "{path}: value is not one of the allowed enum values"
        ));
    }
    if let Some(schema_type) = &schema.schema_type
        && !type_matches(schema_type, input)
    {
        return Err(format!(
            "{path}: expected {}, got {}",
            type_description(schema_type),
            value_type(input)
        ));
    }
    validate_compositions(schema, input, root_schema, root_value, path, depth)?;
    if let Value::Object(object) = input {
        validate_object(schema, object, root_schema, root_value, path, depth)?;
    }
    if let Value::Array(items) = input {
        validate_array(schema, items, root_schema, root_value, path, depth)?;
    }
    validate_scalar_extensions(schema, input, path)
}

fn validate_compositions(
    schema: &JsonSchema,
    input: &Value,
    root_schema: &JsonSchema,
    root_value: &Value,
    path: &str,
    depth: usize,
) -> Result<(), String> {
    if let Some(variants) = &schema.all_of {
        for variant in variants {
            validate(variant, input, root_schema, root_value, path, depth + 1)?;
        }
    }
    if let Some(variants) = &schema.any_of
        && !variants.iter().any(|variant| {
            validate(variant, input, root_schema, root_value, path, depth + 1).is_ok()
        })
    {
        return Err(format!("{path}: value does not match any allowed schema"));
    }
    if let Some(variants) = &schema.one_of {
        let matches = variants
            .iter()
            .filter(|variant| {
                validate(variant, input, root_schema, root_value, path, depth + 1).is_ok()
            })
            .count();
        if matches != 1 {
            return Err(format!(
                "{path}: value must match exactly one allowed schema"
            ));
        }
    }
    Ok(())
}

fn validate_object(
    schema: &JsonSchema,
    object: &serde_json::Map<String, Value>,
    root_schema: &JsonSchema,
    root_value: &Value,
    path: &str,
    depth: usize,
) -> Result<(), String> {
    if let Some(required) = &schema.required {
        for name in required {
            if !object.contains_key(name) {
                return Err(format!("{path}: missing required property `{name}`"));
            }
        }
    }
    let properties = schema.properties.as_ref();
    for (name, value) in object {
        let property_path = format!("{path}.{name}");
        if let Some(property_schema) = properties.and_then(|properties| properties.get(name)) {
            validate(
                property_schema,
                value,
                root_schema,
                root_value,
                &property_path,
                depth + 1,
            )?;
            continue;
        }
        match &schema.additional_properties {
            Some(AdditionalProperties::Boolean(false)) => {
                return Err(format!("{path}: unknown property `{name}`"));
            }
            Some(AdditionalProperties::Schema(additional_schema)) => validate(
                additional_schema,
                value,
                root_schema,
                root_value,
                &property_path,
                depth + 1,
            )?,
            Some(AdditionalProperties::Boolean(true)) | None => {}
        }
    }
    validate_size_extension(schema, object.len(), "minProperties", "maxProperties", path)
}

fn validate_array(
    schema: &JsonSchema,
    items: &[Value],
    root_schema: &JsonSchema,
    root_value: &Value,
    path: &str,
    depth: usize,
) -> Result<(), String> {
    if let Some(item_schema) = &schema.items {
        for (index, item) in items.iter().enumerate() {
            validate(
                item_schema,
                item,
                root_schema,
                root_value,
                &format!("{path}[{index}]"),
                depth + 1,
            )?;
        }
    }
    validate_size_extension(schema, items.len(), "minItems", "maxItems", path)?;
    let unique_items = schema
        .extensions
        .get("uniqueItems")
        .map(|value| {
            value.as_bool().ok_or_else(|| {
                format!("{path}: advertised schema keyword `uniqueItems` must be boolean")
            })
        })
        .transpose()?
        .unwrap_or(false);
    if unique_items {
        for (index, item) in items.iter().enumerate() {
            if contains_equivalent(&items[..index], item)? {
                return Err(format!("{path}: array items must be unique"));
            }
        }
    }
    Ok(())
}

fn validate_scalar_extensions(
    schema: &JsonSchema,
    input: &Value,
    path: &str,
) -> Result<(), String> {
    if let Some(value) = input.as_str() {
        validate_size_extension(
            schema,
            value.chars().count(),
            "minLength",
            "maxLength",
            path,
        )?;
    }
    if let Value::Number(number) = input {
        let value = ExactJsonNumber::parse(number)?;
        for keyword in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"] {
            if let Some(limit) = extension_number(schema, keyword)?
                && match keyword {
                    "minimum" => value.cmp(&limit).is_lt(),
                    "maximum" => value.cmp(&limit).is_gt(),
                    "exclusiveMinimum" => !value.cmp(&limit).is_gt(),
                    "exclusiveMaximum" => !value.cmp(&limit).is_lt(),
                    _ => unreachable!("numeric constraint list is exhaustive"),
                }
            {
                return Err(format!("{path}: number violates `{keyword}`"));
            }
        }
        if let Some(multiple) = extension_number(schema, "multipleOf")? {
            if !multiple.is_positive() {
                return Err(format!(
                    "{path}: advertised schema keyword `multipleOf` must be positive"
                ));
            }
            if !value.is_multiple_of(&multiple)? {
                return Err(format!("{path}: number violates `multipleOf`"));
            }
        }
    }
    Ok(())
}

fn validate_size_extension(
    schema: &JsonSchema,
    size: usize,
    minimum: &str,
    maximum: &str,
    path: &str,
) -> Result<(), String> {
    if let Some(limit) = extension_u64(schema, minimum, path)?
        && u64::try_from(size).unwrap_or(u64::MAX) < limit
    {
        return Err(format!("{path}: size is below `{minimum}`"));
    }
    if let Some(limit) = extension_u64(schema, maximum, path)?
        && u64::try_from(size).unwrap_or(u64::MAX) > limit
    {
        return Err(format!("{path}: size exceeds `{maximum}`"));
    }
    Ok(())
}

fn extension_number(schema: &JsonSchema, keyword: &str) -> Result<Option<ExactJsonNumber>, String> {
    schema
        .extensions
        .get(keyword)
        .map(|value| {
            let Value::Number(number) = value else {
                return Err(format!(
                    "advertised tool schema keyword `{keyword}` must be numeric"
                ));
            };
            ExactJsonNumber::parse(number)
        })
        .transpose()
}

fn extension_u64(schema: &JsonSchema, keyword: &str, path: &str) -> Result<Option<u64>, String> {
    schema
        .extensions
        .get(keyword)
        .map(|value| {
            value.as_u64().ok_or_else(|| {
                format!(
                    "{path}: advertised schema keyword `{keyword}` must be a nonnegative integer"
                )
            })
        })
        .transpose()
}

fn validate_supported_extensions(schema: &JsonSchema, path: &str) -> Result<(), String> {
    for keyword in schema.extensions.keys() {
        if !matches!(
            keyword.as_str(),
            "uniqueItems"
                | "minItems"
                | "maxItems"
                | "minLength"
                | "maxLength"
                | "minProperties"
                | "maxProperties"
                | "minimum"
                | "maximum"
                | "exclusiveMinimum"
                | "exclusiveMaximum"
                | "multipleOf"
        ) {
            return Err(format!(
                "{path}: advertised tool schema uses unsupported validation keyword `{keyword}`"
            ));
        }
    }
    Ok(())
}

fn resolve_local_ref(
    schema_ref: &str,
    root_schema: &JsonSchema,
    root_value: &Value,
) -> Result<JsonSchema, String> {
    if schema_ref == "#" {
        return Ok(root_schema.clone());
    }
    let pointer = schema_ref
        .strip_prefix('#')
        .ok_or_else(|| format!("unsupported external tool-schema reference `{schema_ref}`"))?;
    let value = root_value
        .pointer(pointer)
        .ok_or_else(|| format!("unresolved tool-schema reference `{schema_ref}`"))?;
    serde_json::from_value(value.clone())
        .map_err(|error| format!("invalid referenced tool schema `{schema_ref}`: {error}"))
}

fn type_matches(schema_type: &JsonSchemaType, input: &Value) -> bool {
    match schema_type {
        JsonSchemaType::Single(expected) => primitive_matches(*expected, input),
        JsonSchemaType::Multiple(expected) => expected
            .iter()
            .any(|expected| primitive_matches(*expected, input)),
    }
}

fn primitive_matches(expected: JsonSchemaPrimitiveType, input: &Value) -> bool {
    match expected {
        JsonSchemaPrimitiveType::String => input.is_string(),
        JsonSchemaPrimitiveType::Number => input.is_number(),
        JsonSchemaPrimitiveType::Boolean => input.is_boolean(),
        JsonSchemaPrimitiveType::Integer => {
            input.as_i64().is_some()
                || input.as_u64().is_some()
                || input
                    .as_f64()
                    .is_some_and(|value| value.is_finite() && value.fract() == 0.0)
        }
        JsonSchemaPrimitiveType::Object => input.is_object(),
        JsonSchemaPrimitiveType::Array => input.is_array(),
        JsonSchemaPrimitiveType::Null => input.is_null(),
    }
}

fn type_description(schema_type: &JsonSchemaType) -> String {
    match schema_type {
        JsonSchemaType::Single(value) => primitive_name(*value).to_string(),
        JsonSchemaType::Multiple(values) => values
            .iter()
            .map(|value| primitive_name(*value))
            .collect::<Vec<_>>()
            .join(" or "),
    }
}

fn primitive_name(value: JsonSchemaPrimitiveType) -> &'static str {
    match value {
        JsonSchemaPrimitiveType::String => "string",
        JsonSchemaPrimitiveType::Number => "number",
        JsonSchemaPrimitiveType::Boolean => "boolean",
        JsonSchemaPrimitiveType::Integer => "integer",
        JsonSchemaPrimitiveType::Object => "object",
        JsonSchemaPrimitiveType::Array => "array",
        JsonSchemaPrimitiveType::Null => "null",
    }
}

fn value_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
#[path = "tool_input_validation_tests.rs"]
mod tests;
