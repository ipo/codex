//! Model-specific outbound projection for canonical conversation history.

use super::history::ContextManager;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ModelInfo;

impl ContextManager {
    /// Returns model-projected history without changing the stored transcript.
    pub(crate) fn for_model_prompt(self, model_info: &ModelInfo) -> Vec<ResponseItem> {
        let mut items = self.for_prompt(&model_info.input_modalities);
        if model_info.requires_nonempty_assistant_messages {
            items.retain(|item| !is_empty_assistant_message(item));
        }
        items
    }
}

fn is_empty_assistant_message(item: &ResponseItem) -> bool {
    let ResponseItem::Message { role, content, .. } = item else {
        return false;
    };
    role == "assistant"
        && content.iter().all(|item| match item {
            ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                text.trim().is_empty()
            }
            ContentItem::InputImage { .. } | ContentItem::InputAudio { .. } => false,
        })
}

#[cfg(test)]
#[path = "history_projection_tests.rs"]
mod tests;
