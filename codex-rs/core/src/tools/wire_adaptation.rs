use std::collections::HashSet;

use codex_protocol::model_inference::ModelInferenceConfig;
use codex_protocol::model_inference::WireApi;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::ResponseInputItem;
use codex_protocol::openai_models::ModelInfo;
use codex_tools::AdditionalProperties;
use codex_tools::JsonSchema;
use codex_tools::NamespaceToolSpecMode;
use codex_tools::ResponsesApiNamespaceTool;
use codex_tools::ToolSpec;

use crate::session::turn_context::TurnContext;
use crate::util::error_or_panic;

const MAX_WIRE_TOOL_NAME_BYTES: usize = 64;

pub(crate) fn adapt_spec_for_wire(
    turn_context: &TurnContext,
    model_info: &ModelInfo,
    spec: ToolSpec,
) -> ToolSpec {
    adapt_spec_for_wire_kind(turn_context, model_info, spec, WireSpecKind::General)
}

pub(crate) fn adapt_collaboration_spec_for_wire(
    turn_context: &TurnContext,
    model_info: &ModelInfo,
    spec: ToolSpec,
) -> ToolSpec {
    adapt_spec_for_wire_kind(
        turn_context,
        model_info,
        spec,
        WireSpecKind::CollaborationMessage,
    )
}

#[derive(Clone, Copy)]
enum WireSpecKind {
    General,
    CollaborationMessage,
}

fn adapt_spec_for_wire_kind(
    turn_context: &TurnContext,
    model_info: &ModelInfo,
    mut spec: ToolSpec,
    kind: WireSpecKind,
) -> ToolSpec {
    if wire_supports_encrypted_tool_content(turn_context, model_info) {
        return spec;
    }
    match &mut spec {
        ToolSpec::Function(tool) => {
            if matches!(kind, WireSpecKind::CollaborationMessage) {
                adapt_collaboration_message_parameter(&tool.name, &mut tool.parameters);
            }
            strip_encrypted_parameters(&mut tool.parameters);
        }
        ToolSpec::Namespace(namespace) => {
            for tool in &mut namespace.tools {
                if let ResponsesApiNamespaceTool::Function(tool) = tool {
                    if matches!(kind, WireSpecKind::CollaborationMessage) {
                        adapt_collaboration_message_parameter(&tool.name, &mut tool.parameters);
                    }
                    strip_encrypted_parameters(&mut tool.parameters);
                }
            }
        }
        ToolSpec::ToolSearch { parameters, .. } => strip_encrypted_parameters(parameters),
        ToolSpec::WebSearch { .. } | ToolSpec::Freeform(_) => {}
    }
    spec
}

fn adapt_collaboration_message_parameter(tool_name: &str, schema: &mut JsonSchema) {
    if !matches!(tool_name, "spawn_agent" | "send_message" | "followup_task") {
        return;
    }
    let Some(properties) = schema.properties.as_mut() else {
        return;
    };
    let Some(message) = properties.get("message") else {
        return;
    };
    if message.encrypted != Some(true) {
        return;
    }
    if !properties.contains_key("plaintext_message") {
        let mut plaintext_message = message.clone();
        plaintext_message.encrypted = None;
        plaintext_message.description = Some(match tool_name {
            "spawn_agent" => "Plaintext initial task for the new agent.".to_string(),
            "send_message" => "Plaintext message text to queue on the target agent.".to_string(),
            "followup_task" => "Plaintext message text to send to the target agent.".to_string(),
            _ => unreachable!("collaboration message tool was checked above"),
        });
        properties.insert("plaintext_message".to_string(), plaintext_message);
    }

    let required = schema.required.get_or_insert_default();
    if !required
        .iter()
        .any(|property| property == "plaintext_message")
    {
        required.push("plaintext_message".to_string());
    }
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

pub(crate) fn namespace_tool_spec_mode(
    turn_context: &TurnContext,
    model_info: &ModelInfo,
) -> NamespaceToolSpecMode {
    if super::spec_plan::namespace_tools_enabled(turn_context, model_info) {
        NamespaceToolSpecMode::Preserve
    } else {
        NamespaceToolSpecMode::Flatten
    }
}

pub(crate) fn native_wire(turn_context: &TurnContext, model_info: &ModelInfo) -> bool {
    model_info.wire_api(turn_context.provider.info().wire_api) != WireApi::Responses
        || matches!(
            model_info.inference.as_ref(),
            Some(ModelInferenceConfig::Grok(_))
        )
}

pub(crate) fn wire_supports_encrypted_tool_content(
    turn_context: &TurnContext,
    model_info: &ModelInfo,
) -> bool {
    !native_wire(turn_context, model_info)
        && !matches!(
            model_info.inference.as_ref(),
            Some(ModelInferenceConfig::LlamaCpp(_))
        )
}

pub(crate) fn response_input_contains_encrypted_tool_content(response: &ResponseInputItem) -> bool {
    match response {
        ResponseInputItem::FunctionCallOutput { output, .. }
        | ResponseInputItem::CustomToolCallOutput { output, .. } => {
            matches!(
                &output.body,
                FunctionCallOutputBody::ContentItems(items)
                    if items.iter().any(|item| matches!(
                        item,
                        FunctionCallOutputContentItem::EncryptedContent { .. }
                    ))
            )
        }
        ResponseInputItem::McpToolCallOutput { output, .. } => output.contains_encrypted_content(),
        ResponseInputItem::Message { .. } | ResponseInputItem::ToolSearchOutput { .. } => false,
    }
}

#[cfg(test)]
#[path = "wire_adaptation_tests.rs"]
mod tests;
