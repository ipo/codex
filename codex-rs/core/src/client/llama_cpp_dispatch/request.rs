use codex_protocol::models::AgentMessageInputContent;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::models::plaintext_agent_message_content;
use codex_tools::ToolSpec;

use crate::client_common::Prompt;
use crate::context::AdditionalContextUserFragment;
use crate::context::ContextualUserFragment;

use super::*;

pub(super) fn validate_tools(prompt: &Prompt) -> Result<()> {
    if prompt
        .tools
        .iter()
        .any(|tool| !matches!(tool, ToolSpec::Function(_)))
    {
        return Err(CodexErr::InvalidRequest(
            "direct llama.cpp Responses supports only plain function tools".to_string(),
        ));
    }
    Ok(())
}

pub(super) fn normalize_input(prompt: &Prompt) -> Result<Vec<ResponseItem>> {
    let mut input = Vec::with_capacity(prompt.input.len());
    let mut conversation_started = false;
    for (index, mut item) in prompt
        .get_formatted_input_for_request(/*use_responses_lite*/ false)
        .into_iter()
        .enumerate()
    {
        match &mut item {
            ResponseItem::Message {
                role,
                content,
                phase,
                ..
            } => {
                let is_instruction_role = matches!(role.as_str(), "system" | "developer");
                if content.iter().any(|content| {
                    matches!(
                        content,
                        ContentItem::InputImage { .. } | ContentItem::InputAudio { .. }
                    )
                }) {
                    return Err(CodexErr::InvalidRequest(format!(
                        "llama.cpp history item at index {index} contains unsupported image or audio input"
                    )));
                }
                if is_instruction_role && conversation_started {
                    let text = content
                        .iter()
                        .map(|content| match content {
                            ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                                Ok(text.as_str())
                            }
                            ContentItem::InputImage { .. } | ContentItem::InputAudio { .. } => {
                                Err(CodexErr::InvalidRequest(
                                    "late llama.cpp context update must contain only text"
                                        .to_string(),
                                ))
                            }
                        })
                        .collect::<Result<Vec<_>>>()?
                        .join("\n");
                    item = ContextualUserFragment::into(AdditionalContextUserFragment::new(
                        format!("local_{role}_update"),
                        text,
                    ));
                } else if !matches!(role.as_str(), "system" | "developer" | "user" | "assistant") {
                    return Err(CodexErr::InvalidRequest(format!(
                        "llama.cpp history item at index {index} has unsupported message role `{role}`"
                    )));
                } else {
                    *phase = None;
                }
                if !is_instruction_role {
                    conversation_started = true;
                }
            }
            ResponseItem::AgentMessage {
                author,
                recipient,
                content,
                ..
            } => {
                let text = render_plaintext_agent_message(author, recipient, content).ok_or_else(
                    || {
                        CodexErr::InvalidRequest(format!(
                            "llama.cpp history item at index {index} contains an encrypted cross-agent message"
                        ))
                    },
                )?;
                item = ResponseItem::Message {
                    id: None,
                    role: "user".to_string(),
                    content: vec![ContentItem::InputText { text }],
                    phase: None,
                    internal_chat_message_metadata_passthrough: None,
                };
                conversation_started = true;
            }
            ResponseItem::Reasoning {
                encrypted_content, ..
            } => {
                if encrypted_content
                    .as_deref()
                    .is_some_and(|content| !content.is_empty())
                {
                    return Err(CodexErr::InvalidRequest(format!(
                        "llama.cpp history item at index {index} contains unsupported encrypted reasoning"
                    )));
                }
                conversation_started = true;
            }
            ResponseItem::FunctionCall { namespace, .. } => {
                if namespace.is_some() {
                    return Err(CodexErr::InvalidRequest(format!(
                        "llama.cpp history item at index {index} contains an unsupported namespace tool call"
                    )));
                }
                conversation_started = true;
            }
            ResponseItem::FunctionCallOutput { output, .. } => {
                if let Some(items) = output.content_items() {
                    let mut text = Vec::with_capacity(items.len());
                    for item in items {
                        match item {
                            FunctionCallOutputContentItem::InputText { text: value } => {
                                text.push(value.as_str());
                            }
                            FunctionCallOutputContentItem::InputImage { .. }
                            | FunctionCallOutputContentItem::InputAudio { .. }
                            | FunctionCallOutputContentItem::EncryptedContent { .. } => {
                                return Err(CodexErr::InvalidRequest(format!(
                                    "llama.cpp tool result at history index {index} contains unsupported media or encrypted content"
                                )));
                            }
                        }
                    }
                    *output = codex_protocol::models::FunctionCallOutputPayload::from_text(
                        text.join("\n"),
                    );
                }
                conversation_started = true;
            }
            ResponseItem::AdditionalTools { .. }
            | ResponseItem::LocalShellCall { .. }
            | ResponseItem::ToolSearchCall { .. }
            | ResponseItem::CustomToolCall { .. }
            | ResponseItem::CustomToolCallOutput { .. }
            | ResponseItem::ToolSearchOutput { .. }
            | ResponseItem::WebSearchCall { .. }
            | ResponseItem::ImageGenerationCall { .. }
            | ResponseItem::Compaction { .. }
            | ResponseItem::CompactionTrigger { .. }
            | ResponseItem::ContextCompaction { .. }
            | ResponseItem::Other => {
                return Err(CodexErr::InvalidRequest(format!(
                    "llama.cpp history item at index {index} uses an unsupported Responses item type"
                )));
            }
        }
        item.clear_internal_chat_message_metadata_passthrough();
        input.push(item);
    }
    Ok(input)
}

fn render_plaintext_agent_message(
    author: &str,
    recipient: &str,
    content: &[AgentMessageInputContent],
) -> Option<String> {
    plaintext_agent_message_content(content)
        .map(|text| format!("Agent message from {author} to {recipient}:\n{text}"))
}
