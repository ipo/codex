use base64::Engine;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ReasoningItemReasoningSummary;
use codex_protocol::models::ResponseItem;

use crate::ContentBlock;
use crate::EncodeError;
use crate::ImageSource;
use crate::Role;
use crate::ToolResultBlock;
use crate::ToolResultContent;

pub(crate) fn encode_message_content(
    content: &[ContentItem],
    item_index: usize,
    role: Role,
) -> Result<Vec<ContentBlock>, EncodeError> {
    content
        .iter()
        .enumerate()
        .map(|(block_index, block)| match (role, block) {
            (Role::User, ContentItem::InputText { text })
            | (Role::Assistant, ContentItem::OutputText { text }) => Ok(ContentBlock::Text {
                text: text.clone(),
                cache_control: None,
            }),
            (Role::User, ContentItem::InputImage { image_url, .. }) => Ok(ContentBlock::Image {
                source: image_source(image_url, item_index, block_index)?,
                cache_control: None,
            }),
            (_, ContentItem::InputAudio { .. }) => Err(EncodeError::UnsupportedContent {
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

pub(crate) fn encode_tool_output(
    body: &FunctionCallOutputBody,
    item_index: usize,
) -> Result<ToolResultContent, EncodeError> {
    match body {
        FunctionCallOutputBody::Text(text) => Ok(ToolResultContent::Text(text.clone())),
        FunctionCallOutputBody::ContentItems(items) => items
            .iter()
            .enumerate()
            .map(|(block_index, item)| match item {
                FunctionCallOutputContentItem::InputText { text } => {
                    Ok(ToolResultBlock::Text { text: text.clone() })
                }
                FunctionCallOutputContentItem::InputImage { image_url, .. } => {
                    Ok(ToolResultBlock::Image {
                        source: image_source(image_url, item_index, block_index)?,
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
            .map(ToolResultContent::Blocks),
    }
}

fn image_source(
    url: &str,
    item_index: usize,
    block_index: usize,
) -> Result<ImageSource, EncodeError> {
    let (metadata, data) = url
        .strip_prefix("data:")
        .and_then(|value| value.split_once(','))
        .ok_or_else(|| invalid_image(item_index, block_index, "expected a data URL"))?;
    let media_type = metadata
        .strip_suffix(";base64")
        .filter(|media_type| media_type.starts_with("image/"))
        .ok_or_else(|| {
            invalid_image(
                item_index,
                block_index,
                "expected an image/* base64 data URL",
            )
        })?;
    base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|error| invalid_image(item_index, block_index, &error.to_string()))?;
    Ok(ImageSource::Base64 {
        media_type: media_type.to_string(),
        data: data.to_string(),
    })
}

fn invalid_image(item_index: usize, block_index: usize, message: &str) -> EncodeError {
    EncodeError::InvalidImage {
        item_index,
        block_index,
        message: message.to_string(),
    }
}

pub(crate) fn visible_reasoning(
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

pub(crate) fn history_kind(item: &ResponseItem) -> &'static str {
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
