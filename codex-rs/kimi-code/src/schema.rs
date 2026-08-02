use std::collections::BTreeMap;
use std::collections::BTreeSet;

use serde_json::Value;
use thiserror::Error;

const MAP_SLOTS: &str = "dependencies dependentSchemas patternProperties properties";
const SINGLE_SLOTS: &str = "additionalItems additionalProperties contains contentSchema else if not propertyNames then unevaluatedItems unevaluatedProperties";
const ARRAY_SLOTS: &str = "allOf anyOf oneOf prefixItems";
const OBJECT_KEYS: &str = "additionalProperties dependencies dependentRequired dependentSchemas maxProperties minProperties patternProperties properties propertyNames required unevaluatedProperties";
const ARRAY_KEYS: &str = "additionalItems contains items maxContains maxItems minContains minItems prefixItems unevaluatedItems uniqueItems";
const STRING_KEYS: &str =
    "contentEncoding contentMediaType contentSchema format maxLength minLength pattern";
const NUMERIC_KEYS: &str = "exclusiveMaximum exclusiveMinimum maximum minimum multipleOf";

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SchemaError {
    #[error("tool schema must be a JSON object")]
    NotObject,
    #[error("unsupported schema reference `{0}`")]
    UnsupportedReference(String),
    #[error("unresolved local schema reference `{0}`")]
    UnresolvedReference(String),
    #[error("recursive local schema reference `{0}` is unsupported")]
    RecursiveReference(String),
    #[error("property `{0}` is required but is not defined")]
    MissingRequiredProperty(String),
    #[error("schema enum or const values do not imply one type")]
    AmbiguousValueType,
}

pub fn normalize_schema(schema: &Value) -> Result<Value, SchemaError> {
    let root = schema.as_object().ok_or(SchemaError::NotObject)?;
    let mut definitions = BTreeMap::new();
    for table in ["$defs", "definitions"]
        .into_iter()
        .filter_map(|key| root.get(key))
    {
        for (name, value) in table.as_object().ok_or(SchemaError::NotObject)? {
            if definitions.insert(name.clone(), value.clone()).is_some() {
                return Err(SchemaError::UnsupportedReference(format!(
                    "duplicate definition {name}"
                )));
            }
        }
    }
    let mut normalized = schema.clone();
    visit(&mut normalized, &definitions, &mut Vec::new(), false)?;
    Ok(normalized)
}

fn visit(
    schema: &mut Value,
    definitions: &BTreeMap<String, Value>,
    stack: &mut Vec<String>,
    complete_type: bool,
) -> Result<(), SchemaError> {
    let object = schema.as_object_mut().ok_or(SchemaError::NotObject)?;
    if let Some(reference) = object.remove("$ref") {
        let reference = reference
            .as_str()
            .ok_or_else(|| SchemaError::UnsupportedReference(reference.to_string()))?;
        let name = local_reference_name(reference)?;
        if stack.contains(&name) {
            return Err(SchemaError::RecursiveReference(reference.to_string()));
        }
        let mut resolved = definitions
            .get(&name)
            .cloned()
            .ok_or_else(|| SchemaError::UnresolvedReference(reference.to_string()))?;
        resolved
            .as_object_mut()
            .ok_or(SchemaError::NotObject)?
            .extend(std::mem::take(object));
        stack.push(name);
        let result = visit(&mut resolved, definitions, stack, complete_type);
        stack.pop();
        result?;
        *schema = resolved;
        return Ok(());
    }

    let object = schema.as_object_mut().ok_or(SchemaError::NotObject)?;
    object.remove("$defs");
    object.remove("definitions");
    let required = object
        .get("required")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for key in MAP_SLOTS.split_ascii_whitespace() {
        let Some(children) = object.get_mut(key).and_then(Value::as_object_mut) else {
            continue;
        };
        for child in children.values_mut().filter(|child| child.is_object()) {
            visit(child, definitions, stack, true)?;
        }
        if key == "properties"
            && let Some(name) = required
                .iter()
                .filter_map(Value::as_str)
                .find(|name| !children.contains_key(*name))
        {
            return Err(SchemaError::MissingRequiredProperty(name.to_string()));
        }
    }
    for key in SINGLE_SLOTS.split_ascii_whitespace() {
        if let Some(child) = object.get_mut(key).filter(|child| child.is_object()) {
            visit(child, definitions, stack, true)?;
        }
    }
    for key in ARRAY_SLOTS.split_ascii_whitespace() {
        if let Some(children) = object.get_mut(key).and_then(Value::as_array_mut) {
            for child in children.iter_mut().filter(|child| child.is_object()) {
                visit(child, definitions, stack, true)?;
            }
        }
    }
    if let Some(items) = object.get_mut("items") {
        if items.is_object() {
            visit(items, definitions, stack, true)?;
        } else if let Some(items) = items.as_array_mut() {
            for item in items.iter_mut().filter(|item| item.is_object()) {
                visit(item, definitions, stack, true)?;
            }
        }
    }

    if complete_type
        && !object.contains_key("type")
        && !has_any(object, "allOf anyOf else if not oneOf then")
    {
        let inferred = if let Some(values) = object.get("enum").and_then(Value::as_array)
            && !values.is_empty()
        {
            infer_value_type(values)?
        } else if let Some(value) = object.get("const") {
            infer_value_type(std::slice::from_ref(value))?
        } else if has_any(object, OBJECT_KEYS) {
            "object"
        } else if has_any(object, ARRAY_KEYS) {
            "array"
        } else if has_any(object, STRING_KEYS) {
            "string"
        } else if has_any(object, NUMERIC_KEYS) {
            "number"
        } else {
            "string"
        };
        object.insert("type".to_string(), inferred.into());
    }
    Ok(())
}

fn has_any(object: &serde_json::Map<String, Value>, keys: &str) -> bool {
    keys.split_ascii_whitespace()
        .any(|key| object.contains_key(key))
}

fn local_reference_name(reference: &str) -> Result<String, SchemaError> {
    reference
        .strip_prefix("#/$defs/")
        .or_else(|| reference.strip_prefix("#/definitions/"))
        .map(|name| name.replace("~1", "/").replace("~0", "~"))
        .ok_or_else(|| SchemaError::UnsupportedReference(reference.to_string()))
}

fn infer_value_type(values: &[Value]) -> Result<&'static str, SchemaError> {
    let mut inferred = BTreeSet::new();
    for value in values {
        inferred.insert(match value {
            Value::Null => "null",
            Value::Bool(_) => "boolean",
            Value::Number(number) if number.is_i64() || number.is_u64() => "integer",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        });
    }
    if inferred.contains("number") {
        inferred.remove("integer");
    }
    if inferred.len() != 1 {
        return Err(SchemaError::AmbiguousValueType);
    }
    inferred.pop_first().ok_or(SchemaError::AmbiguousValueType)
}
