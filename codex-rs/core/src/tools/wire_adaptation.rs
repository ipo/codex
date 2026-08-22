use std::collections::HashSet;

use codex_protocol::model_inference::ModelInferenceConfig;
use codex_protocol::model_inference::WireApi;
use codex_tools::AdditionalProperties;
use codex_tools::JsonSchema;
use codex_tools::NamespaceToolSpecMode;
use codex_tools::ResponsesApiNamespaceTool;
use codex_tools::ToolSpec;

use crate::session::turn_context::TurnContext;
use crate::util::error_or_panic;

const MAX_WIRE_TOOL_NAME_BYTES: usize = 64;

pub(crate) fn adapt_spec_for_wire(turn_context: &TurnContext, mut spec: ToolSpec) -> ToolSpec {
    if wire_supports_encrypted_tool_content(turn_context) {
        return spec;
    }
    match &mut spec {
        ToolSpec::Function(tool) => strip_encrypted_parameters(&mut tool.parameters),
        ToolSpec::Namespace(namespace) => {
            for tool in &mut namespace.tools {
                let ResponsesApiNamespaceTool::Function(tool) = tool;
                strip_encrypted_parameters(&mut tool.parameters);
            }
        }
        ToolSpec::ToolSearch { parameters, .. } => strip_encrypted_parameters(parameters),
        ToolSpec::WebSearch { .. } | ToolSpec::Freeform(_) => {}
    }
    spec
}

fn strip_encrypted_parameters(schema: &mut JsonSchema) {
    if let Some(properties) = schema.properties.as_mut() {
        properties.retain(|_, property| property.encrypted != Some(true));
        if let Some(required) = schema.required.as_mut() {
            required.retain(|name| properties.contains_key(name));
        }
        for property in properties.values_mut() {
            strip_encrypted_parameters(property);
        }
    }
    if let Some(items) = schema.items.as_mut() {
        strip_encrypted_parameters(items);
    }
    for schemas in [
        schema.any_of.as_mut(),
        schema.one_of.as_mut(),
        schema.all_of.as_mut(),
    ]
    .into_iter()
    .flatten()
    {
        for schema in schemas {
            strip_encrypted_parameters(schema);
        }
    }
    for definitions in [schema.defs.as_mut(), schema.definitions.as_mut()]
        .into_iter()
        .flatten()
    {
        for schema in definitions.values_mut() {
            strip_encrypted_parameters(schema);
        }
    }
    if let Some(AdditionalProperties::Schema(schema)) = schema.additional_properties.as_mut() {
        strip_encrypted_parameters(schema);
    }
}

pub(crate) fn validate_model_visible_function_names(specs: Vec<ToolSpec>) -> Vec<ToolSpec> {
    let mut seen_function_names = HashSet::new();
    specs
        .into_iter()
        .filter(|spec| {
            let ToolSpec::Function(tool) = spec else {
                return true;
            };
            if tool.name.len() > MAX_WIRE_TOOL_NAME_BYTES {
                error_or_panic(format!(
                    "tool name `{}` exceeds the {MAX_WIRE_TOOL_NAME_BYTES}-byte wire limit",
                    tool.name
                ));
                return false;
            }
            if seen_function_names.insert(tool.name.clone()) {
                return true;
            }
            error_or_panic(format!("tool {} already exposed to the model", tool.name));
            false
        })
        .collect()
}

pub(crate) fn namespace_tool_spec_mode(turn_context: &TurnContext) -> NamespaceToolSpecMode {
    if super::spec_plan::namespace_tools_enabled(turn_context) {
        NamespaceToolSpecMode::Preserve
    } else {
        NamespaceToolSpecMode::Flatten
    }
}

pub(crate) fn native_wire(turn_context: &TurnContext) -> bool {
    turn_context
        .model_info
        .wire_api(turn_context.provider.info().wire_api)
        != WireApi::Responses
}

pub(crate) fn requires_function_tool_specs(turn_context: &TurnContext) -> bool {
    native_wire(turn_context)
        || matches!(
            turn_context.model_info.inference,
            Some(ModelInferenceConfig::Grok(_) | ModelInferenceConfig::LlamaCpp(_))
        )
}

pub(crate) fn wire_supports_encrypted_tool_content(turn_context: &TurnContext) -> bool {
    turn_context
        .model_info
        .supports_responses_capabilities(turn_context.provider.info().wire_api)
}

#[cfg(test)]
#[path = "wire_adaptation_tests.rs"]
mod tests;
