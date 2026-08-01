use codex_protocol::model_inference::ModelInferenceConfig;
use codex_protocol::models::ResponseItem;

use crate::ContentBlock;
use crate::EncodeError;
use crate::Message;
use crate::ReplayDecision;
use crate::Role;
use crate::ThinkingReplayBlock;
use crate::ToolResultContent;
use crate::decode_thinking_replay;
use crate::history_content::encode_message_content;
use crate::history_content::encode_tool_output;
use crate::history_content::history_kind;
use crate::history_content::visible_reasoning;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Group {
    User,
    Assistant,
    ToolResults,
}

pub(crate) fn encode_history(
    history: &[ResponseItem],
    profile: &ModelInferenceConfig,
) -> Result<Vec<Message>, EncodeError> {
    let ModelInferenceConfig::Anthropic {
        dialect,
        wire_model,
        ..
    } = profile
    else {
        return Err(crate::AssembleError::NotAnthropicModel.into());
    };
    let mut messages = Vec::new();
    let mut pending = Vec::new();
    for (index, item) in history.iter().enumerate() {
        match item {
            ResponseItem::Message { role, content, .. } if role == "user" => {
                resolve_orphans(&mut messages, &mut pending);
                push_blocks(
                    &mut messages,
                    Group::User,
                    encode_message_content(content, index, Role::User)?,
                );
            }
            ResponseItem::Message { role, content, .. } if role == "assistant" => {
                resolve_before_assistant(&mut messages, &mut pending);
                push_blocks(
                    &mut messages,
                    Group::Assistant,
                    encode_message_content(content, index, Role::Assistant)?,
                );
            }
            ResponseItem::Message { .. } => {
                return Err(EncodeError::UnsupportedHistoryItem {
                    index,
                    kind: "message role",
                });
            }
            ResponseItem::Reasoning {
                summary,
                content,
                encrypted_content,
                ..
            } => {
                resolve_before_assistant(&mut messages, &mut pending);
                let visible = visible_reasoning(summary, content.as_deref());
                let replay = match encrypted_content {
                    Some(opaque) => decode_thinking_replay(opaque, visible, *dialect, wire_model)?,
                    None => ReplayDecision::UnrelatedOpaqueContent { visible },
                };
                push_blocks(&mut messages, Group::Assistant, replay_blocks(replay));
            }
            ResponseItem::FunctionCall {
                name,
                namespace,
                arguments,
                call_id,
                ..
            } => {
                resolve_before_assistant(&mut messages, &mut pending);
                if namespace.is_some() {
                    return Err(EncodeError::UnsupportedHistoryItem {
                        index,
                        kind: "namespace function call",
                    });
                }
                if pending.contains(call_id) {
                    return Err(EncodeError::DuplicateToolUse {
                        call_id: call_id.clone(),
                    });
                }
                let input: serde_json::Value =
                    serde_json::from_str(arguments).map_err(|error| {
                        EncodeError::InvalidToolArguments {
                            call_id: call_id.clone(),
                            message: error.to_string(),
                        }
                    })?;
                if !input.is_object() {
                    return Err(EncodeError::InvalidToolArguments {
                        call_id: call_id.clone(),
                        message: "expected a JSON object".to_string(),
                    });
                }
                pending.push(call_id.clone());
                push_blocks(
                    &mut messages,
                    Group::Assistant,
                    vec![ContentBlock::ToolUse {
                        id: call_id.clone(),
                        name: name.clone(),
                        input,
                    }],
                );
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
                push_blocks(
                    &mut messages,
                    Group::ToolResults,
                    vec![ContentBlock::ToolResult {
                        tool_use_id: call_id.clone(),
                        content: encode_tool_output(&output.body, index)?,
                        is_error: output.success == Some(false),
                        cache_control: None,
                    }],
                );
            }
            other => {
                return Err(EncodeError::UnsupportedHistoryItem {
                    index,
                    kind: history_kind(other),
                });
            }
        }
    }
    resolve_orphans(&mut messages, &mut pending);
    Ok(messages.into_iter().map(|(_, message)| message).collect())
}

fn replay_blocks(replay: ReplayDecision) -> Vec<ContentBlock> {
    match replay {
        ReplayDecision::Native(blocks) => blocks
            .into_iter()
            .map(|block| match block {
                ThinkingReplayBlock::Signed {
                    thinking,
                    signature,
                } => ContentBlock::Thinking {
                    thinking,
                    signature,
                },
                ThinkingReplayBlock::Redacted { data } => ContentBlock::RedactedThinking { data },
            })
            .collect(),
        ReplayDecision::UnrelatedOpaqueContent { visible }
        | ReplayDecision::ModelOrDialectChanged { visible } => visible
            .into_iter()
            .map(|text| ContentBlock::Text {
                text,
                cache_control: None,
            })
            .collect(),
    }
}

fn resolve_before_assistant(messages: &mut Vec<(Group, Message)>, pending: &mut Vec<String>) {
    if messages
        .last()
        .is_some_and(|(group, _)| *group == Group::ToolResults)
    {
        resolve_orphans(messages, pending);
    }
}

fn resolve_orphans(messages: &mut Vec<(Group, Message)>, pending: &mut Vec<String>) {
    if pending.is_empty() {
        return;
    }
    let blocks = pending
        .drain(..)
        .map(|call_id| ContentBlock::ToolResult {
            tool_use_id: call_id,
            content: ToolResultContent::Text(
                "Tool execution result was not recorded by Codex.".to_string(),
            ),
            is_error: true,
            cache_control: None,
        })
        .collect();
    push_blocks(messages, Group::ToolResults, blocks);
}

fn push_blocks(messages: &mut Vec<(Group, Message)>, group: Group, blocks: Vec<ContentBlock>) {
    let role = if group == Group::Assistant {
        Role::Assistant
    } else {
        Role::User
    };
    if let Some((last_group, message)) = messages.last_mut()
        && *last_group == group
    {
        message.content.extend(blocks);
    } else {
        messages.push((
            group,
            Message {
                role,
                content: blocks,
            },
        ));
    }
}
