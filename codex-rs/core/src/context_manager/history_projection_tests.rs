use super::*;
use crate::context_manager::history::ContextManager;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseItem;
use codex_utils_output_truncation::TruncationPolicy;
use pretty_assertions::assert_eq;

fn message(role: &str, text: &str) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: role.to_string(),
        content: vec![ContentItem::OutputText {
            text: text.to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }
}

fn history(items: &[ResponseItem]) -> ContextManager {
    let mut history = ContextManager::new();
    history.record_items(items.iter(), TruncationPolicy::Tokens(10_000));
    history
}

#[test]
fn marked_model_projection_omits_only_empty_assistant_messages() {
    let function_call = ResponseItem::FunctionCall {
        id: None,
        name: "tool".to_string(),
        namespace: None,
        arguments: "{}".to_string(),
        encrypted_function_args: None,
        call_id: "call-1".to_string(),
        internal_chat_message_metadata_passthrough: None,
    };
    let function_output = ResponseItem::FunctionCallOutput {
        id: None,
        call_id: Some("call-1".to_string()),
        name: None,
        namespace: None,
        output: FunctionCallOutputPayload::from_text("result".to_string()),
        internal_chat_message_metadata_passthrough: None,
    };
    let items = vec![
        message("user", "question"),
        message("assistant", " \n"),
        function_call.clone(),
        function_output.clone(),
        message("assistant", "answer"),
        message("user", ""),
    ];
    let history = history(&items);
    let mut model = codex_models_manager::model_info::model_info_from_slug("kimi/k3");
    model.requires_nonempty_assistant_messages = true;

    assert_eq!(
        history.clone().for_model_prompt(&model),
        vec![
            items[0].clone(),
            function_call,
            function_output,
            items[4].clone(),
            items[5].clone(),
        ]
    );
    assert_eq!(history.raw_items().cloned().collect::<Vec<_>>(), items);
}

#[test]
fn unmarked_model_projection_preserves_empty_assistant_messages() {
    let items = vec![message("assistant", " \n")];
    let model = codex_models_manager::model_info::model_info_from_slug("unknown");

    assert_eq!(history(&items).for_model_prompt(&model), items);
}
