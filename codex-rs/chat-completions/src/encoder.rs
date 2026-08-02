use std::collections::BTreeMap;

use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ReasoningItemReasoningSummary;
use codex_protocol::models::ResponseItem;
use codex_tools::ToolSpec;
use serde_json::Value;
use thiserror::Error;

use crate::AssistantMessage;
use crate::AssistantReasoningReplay;
use crate::ChatCompletionsRequest;
use crate::ChatMessage;
use crate::ContentPart;
use crate::DialectContext;
use crate::DialectError;
use crate::DialectHooks;
use crate::FunctionDefinition;
use crate::FunctionTool;
use crate::ImageUrl;
use crate::MessageContent;
use crate::OpaqueReasoning;
use crate::ToolCall;
use crate::ToolCallFunction;
use crate::types::FunctionToolKind;
use crate::types::ToolCallKind;

pub struct EncodeRequest<'a> {
    pub context: DialectContext<'a>,
    pub system: Option<&'a str>,
    pub history: &'a [ResponseItem],
    pub tools: &'a [ToolSpec],
    pub dialect: &'a dyn DialectHooks,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum EncodeError {
    #[error("native Chat Completions model is empty")]
    EmptyModel,
    #[error("unsupported history item at index {index}: {kind}")]
    UnsupportedHistoryItem { index: usize, kind: &'static str },
    #[error("unsupported content at history item {item_index}, block {block_index}: {kind}")]
    UnsupportedContent {
        item_index: usize,
        block_index: usize,
        kind: &'static str,
    },
    #[error("unsupported tool at index {index}: {kind}")]
    UnsupportedTool { index: usize, kind: &'static str },
    #[error("function tool schema at index {index} could not be serialized: {message}")]
    ToolSchema { index: usize, message: String },
    #[error("tool call `{call_id}` contains invalid JSON-object arguments: {message}")]
    InvalidToolArguments { call_id: String, message: String },
    #[error("tool result `{call_id}` has no matching unresolved tool call")]
    UnmatchedToolResult { call_id: String },
    #[error("duplicate unresolved tool call ID `{call_id}`")]
    DuplicateToolCall { call_id: String },
    #[error("dialect extension `{field}` conflicts with a reserved {target} field")]
    ReservedExtension { target: &'static str, field: String },
    #[error(transparent)]
    Dialect(#[from] DialectError),
}

pub fn encode_request(params: EncodeRequest<'_>) -> Result<ChatCompletionsRequest, EncodeError> {
    if params.context.model.is_empty() {
        return Err(EncodeError::EmptyModel);
    }
    let extensions = params.dialect.request_extensions(params.context)?;
    reject_reserved(&extensions, &["model", "messages", "tools"], "request")?;
    let tools = encode_tools(params.tools)?;
    let mut messages = params
        .system
        .map(|content| ChatMessage::System {
            content: MessageContent::Text(content.to_string()),
        })
        .into_iter()
        .collect();
    encode_history(&mut messages, &params)?;
    Ok(ChatCompletionsRequest {
        model: params.context.model.to_string(),
        messages,
        tools,
        extensions,
    })
}

fn encode_tools(tools: &[ToolSpec]) -> Result<Vec<FunctionTool>, EncodeError> {
    tools
        .iter()
        .enumerate()
        .map(|(index, tool)| match tool {
            ToolSpec::Function(function)
                if function.defer_loading.is_none() && function.output_schema.is_none() =>
            {
                Ok(FunctionTool {
                    kind: FunctionToolKind::Function,
                    function: FunctionDefinition {
                        name: function.name.clone(),
                        description: function.description.clone(),
                        parameters: serde_json::to_value(&function.parameters).map_err(
                            |error| EncodeError::ToolSchema {
                                index,
                                message: error.to_string(),
                            },
                        )?,
                        strict: function.strict,
                    },
                })
            }
            ToolSpec::Function(_) => Err(EncodeError::UnsupportedTool {
                index,
                kind: "deferred function or function output schema",
            }),
            ToolSpec::Namespace(_) => Err(EncodeError::UnsupportedTool {
                index,
                kind: "namespace",
            }),
            ToolSpec::ToolSearch { .. } => Err(EncodeError::UnsupportedTool {
                index,
                kind: "tool search",
            }),
            ToolSpec::WebSearch { .. } => Err(EncodeError::UnsupportedTool {
                index,
                kind: "web search",
            }),
            ToolSpec::Freeform(_) => Err(EncodeError::UnsupportedTool {
                index,
                kind: "freeform",
            }),
        })
        .collect()
}

fn encode_history(
    messages: &mut Vec<ChatMessage>,
    params: &EncodeRequest<'_>,
) -> Result<(), EncodeError> {
    let mut pending = Vec::<String>::new();
    for (index, item) in params.history.iter().enumerate() {
        match item {
            ResponseItem::Message { role, content, .. } if role == "user" => {
                messages.push(ChatMessage::User {
                    content: message_content(content, index)?,
                });
            }
            ResponseItem::Message { role, content, .. } if role == "assistant" => {
                let text = assistant_text(content, index)?;
                assistant(messages)
                    .content
                    .get_or_insert_default()
                    .push_str(&text);
            }
            ResponseItem::Message { .. } => return Err(unsupported(index, "message role")),
            ResponseItem::Reasoning {
                summary,
                content,
                encrypted_content,
                ..
            } => {
                let visible = visible_reasoning(summary, content.as_deref());
                let reasoning = params.dialect.assistant_reasoning(
                    params.context,
                    AssistantReasoningReplay {
                        visible: &visible,
                        opaque: classify_opaque_reasoning(encrypted_content.as_deref()),
                    },
                )?;
                reject_reserved(&reasoning, &["role", "content", "tool_calls"], "assistant")?;
                let assistant = assistant(messages);
                for (field, value) in reasoning {
                    if assistant.reasoning.insert(field.clone(), value).is_some() {
                        return Err(EncodeError::ReservedExtension {
                            target: "assistant reasoning",
                            field,
                        });
                    }
                }
            }
            ResponseItem::FunctionCall {
                name,
                namespace,
                arguments,
                call_id,
                ..
            } => {
                if namespace.is_some() {
                    return Err(unsupported(index, "namespace function call"));
                }
                validate_arguments(call_id, arguments)?;
                if pending.contains(call_id) {
                    return Err(EncodeError::DuplicateToolCall {
                        call_id: call_id.clone(),
                    });
                }
                pending.push(call_id.clone());
                assistant(messages).tool_calls.push(ToolCall {
                    id: call_id.clone(),
                    kind: ToolCallKind::Function,
                    function: ToolCallFunction {
                        name: name.clone(),
                        arguments: arguments.clone(),
                    },
                });
            }
            ResponseItem::FunctionCallOutput {
                call_id, output, ..
            } => {
                let Some(position) = pending.iter().position(|pending| pending == call_id) else {
                    return Err(EncodeError::UnmatchedToolResult {
                        call_id: call_id.clone(),
                    });
                };
                pending.remove(position);
                messages.push(ChatMessage::Tool {
                    tool_call_id: call_id.clone(),
                    content: tool_output(&output.body, index)?,
                });
            }
            other => return Err(unsupported(index, history_kind(other))),
        }
    }
    Ok(())
}

fn classify_opaque_reasoning(opaque: Option<&str>) -> OpaqueReasoning<'_> {
    match opaque {
        None => OpaqueReasoning::None,
        Some(opaque) => OpaqueReasoning::Other(opaque),
    }
}

fn assistant(messages: &mut Vec<ChatMessage>) -> &mut AssistantMessage {
    if !matches!(messages.last(), Some(ChatMessage::Assistant(_))) {
        messages.push(ChatMessage::Assistant(AssistantMessage {
            content: None,
            tool_calls: Vec::new(),
            reasoning: BTreeMap::new(),
        }));
    }
    let Some(ChatMessage::Assistant(message)) = messages.last_mut() else {
        unreachable!("assistant message was just inserted")
    };
    message
}

fn message_content(
    content: &[ContentItem],
    item_index: usize,
) -> Result<MessageContent, EncodeError> {
    let parts = content
        .iter()
        .enumerate()
        .map(|(block_index, item)| match item {
            ContentItem::InputText { text } => Ok(ContentPart::Text { text: text.clone() }),
            ContentItem::InputImage { image_url, detail } => Ok(ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: image_url.clone(),
                    detail: *detail,
                },
            }),
            ContentItem::InputAudio { .. } => Err(EncodeError::UnsupportedContent {
                item_index,
                block_index,
                kind: "audio",
            }),
            _ => Err(EncodeError::UnsupportedContent {
                item_index,
                block_index,
                kind: "content role/type combination",
            }),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if let [ContentPart::Text { text }] = parts.as_slice() {
        Ok(MessageContent::Text(text.clone()))
    } else {
        Ok(MessageContent::Parts(parts))
    }
}

fn assistant_text(content: &[ContentItem], item_index: usize) -> Result<String, EncodeError> {
    content
        .iter()
        .enumerate()
        .map(|(block_index, item)| match item {
            ContentItem::OutputText { text } => Ok(text.as_str()),
            ContentItem::InputAudio { .. } => Err(EncodeError::UnsupportedContent {
                item_index,
                block_index,
                kind: "audio",
            }),
            _ => Err(EncodeError::UnsupportedContent {
                item_index,
                block_index,
                kind: "content role/type combination",
            }),
        })
        .collect()
}

fn tool_output(
    body: &FunctionCallOutputBody,
    item_index: usize,
) -> Result<MessageContent, EncodeError> {
    match body {
        FunctionCallOutputBody::Text(text) => Ok(MessageContent::Text(text.clone())),
        FunctionCallOutputBody::ContentItems(items) => items
            .iter()
            .enumerate()
            .map(|(block_index, item)| match item {
                FunctionCallOutputContentItem::InputText { text } => {
                    Ok(ContentPart::Text { text: text.clone() })
                }
                FunctionCallOutputContentItem::InputImage { image_url, detail } => {
                    Ok(ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: image_url.clone(),
                            detail: *detail,
                        },
                    })
                }
                FunctionCallOutputContentItem::InputAudio { .. } => {
                    Err(EncodeError::UnsupportedContent {
                        item_index,
                        block_index,
                        kind: "audio tool result",
                    })
                }
                FunctionCallOutputContentItem::EncryptedContent { .. } => {
                    Err(EncodeError::UnsupportedContent {
                        item_index,
                        block_index,
                        kind: "encrypted tool result",
                    })
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map(MessageContent::Parts),
    }
}

fn validate_arguments(call_id: &str, arguments: &str) -> Result<(), EncodeError> {
    let value: Value =
        serde_json::from_str(arguments).map_err(|error| EncodeError::InvalidToolArguments {
            call_id: call_id.to_string(),
            message: error.to_string(),
        })?;
    if value.is_object() {
        Ok(())
    } else {
        Err(EncodeError::InvalidToolArguments {
            call_id: call_id.to_string(),
            message: "expected a JSON object".to_string(),
        })
    }
}

fn reject_reserved(
    extensions: &BTreeMap<String, Value>,
    reserved: &[&str],
    target: &'static str,
) -> Result<(), EncodeError> {
    if let Some(field) = reserved
        .iter()
        .find(|field| extensions.contains_key(**field))
    {
        Err(EncodeError::ReservedExtension {
            target,
            field: (*field).to_string(),
        })
    } else {
        Ok(())
    }
}

fn visible_reasoning(
    summary: &[ReasoningItemReasoningSummary],
    content: Option<&[ReasoningItemContent]>,
) -> Vec<String> {
    let content = content
        .unwrap_or_default()
        .iter()
        .map(|item| match item {
            ReasoningItemContent::ReasoningText { text } | ReasoningItemContent::Text { text } => {
                text.clone()
            }
        })
        .collect::<Vec<_>>();
    if content.is_empty() {
        summary
            .iter()
            .map(|ReasoningItemReasoningSummary::SummaryText { text }| text.clone())
            .collect()
    } else {
        content
    }
}

fn unsupported(index: usize, kind: &'static str) -> EncodeError {
    EncodeError::UnsupportedHistoryItem { index, kind }
}

fn history_kind(item: &ResponseItem) -> &'static str {
    match item {
        ResponseItem::AdditionalTools { .. } => "additional tools",
        ResponseItem::AgentMessage { .. } => "structured agent message",
        ResponseItem::LocalShellCall { .. } => "local shell call",
        ResponseItem::CustomToolCall { .. } => "freeform tool call",
        ResponseItem::CustomToolCallOutput { .. } => "freeform tool result",
        ResponseItem::ToolSearchCall { .. } | ResponseItem::ToolSearchOutput { .. } => {
            "hosted tool search"
        }
        ResponseItem::WebSearchCall { .. } => "hosted web search",
        ResponseItem::ImageGenerationCall { .. } => "image generation",
        ResponseItem::Compaction { .. } | ResponseItem::ContextCompaction { .. } => {
            "encrypted Responses compaction"
        }
        ResponseItem::CompactionTrigger { .. } => "compaction trigger",
        ResponseItem::Other => "unknown input",
        ResponseItem::Message { .. }
        | ResponseItem::Reasoning { .. }
        | ResponseItem::FunctionCall { .. }
        | ResponseItem::FunctionCallOutput { .. } => "invalid supported item",
    }
}
