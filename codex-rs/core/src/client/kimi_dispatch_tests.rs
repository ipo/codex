use super::*;
use codex_protocol::models::BaseInstructions;

fn message(role: &str, text: &str) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: role.to_string(),
        content: vec![ContentItem::InputText {
            text: text.to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }
}

fn prompt(input: Vec<ResponseItem>) -> Prompt {
    Prompt {
        input,
        base_instructions: BaseInstructions {
            text: "base instructions".to_string(),
        },
        ..Default::default()
    }
}

#[test]
fn collaboration_mode_transitions_are_reminders_in_chronological_history() {
    let default = "<collaboration_mode>default</collaboration_mode>";
    let plan = "<collaboration_mode>plan</collaboration_mode>";
    let prompt = prompt(vec![
        message("developer", "developer guidance"),
        message("user", "first user input"),
        message("developer", default),
        message("assistant", "first assistant output"),
        message("user", "second user input"),
        message("developer", plan),
        message("assistant", "second assistant output"),
        message("user", "current user input"),
        message("developer", default),
    ]);

    let (system, history) = native_system_and_history(&prompt).expect("projection succeeds");

    assert_eq!(
        system,
        Some(format!(
            "base instructions\n\ndeveloper guidance\n\n{KIMI_COLLABORATION_REMINDER_INSTRUCTIONS}"
        ))
    );
    assert_eq!(
        history,
        vec![
            message("user", "first user input"),
            message(
                "user",
                &format!("<system-reminder>{default}</system-reminder>")
            ),
            message("assistant", "first assistant output"),
            message("user", "second user input"),
            message(
                "user",
                &format!("<system-reminder>{plan}</system-reminder>")
            ),
            message("assistant", "second assistant output"),
            message("user", "current user input"),
            message(
                "user",
                &format!("<system-reminder>{default}</system-reminder>")
            ),
        ]
    );
}

#[test]
fn collaboration_transitions_collapse_without_separating_tool_results() {
    let prompt = prompt(vec![
        message("user", "user input"),
        message("developer", "<collaboration_mode>plan</collaboration_mode>"),
        message(
            "developer",
            "<collaboration_mode>default</collaboration_mode>",
        ),
        ResponseItem::FunctionCall {
            id: None,
            call_id: "call-1".to_string(),
            name: "exec_command".to_string(),
            namespace: None,
            arguments: "{}".to_string(),
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::FunctionCallOutput {
            id: None,
            call_id: "call-1".to_string(),
            output: Default::default(),
            internal_chat_message_metadata_passthrough: None,
        },
    ]);

    let (_, history) = native_system_and_history(&prompt).expect("projection succeeds");

    assert_eq!(
        history,
        vec![
            message("user", "user input"),
            message(
                "user",
                "<system-reminder><collaboration_mode>default</collaboration_mode></system-reminder>",
            ),
            ResponseItem::FunctionCall {
                id: None,
                call_id: "call-1".to_string(),
                name: "exec_command".to_string(),
                namespace: None,
                arguments: "{}".to_string(),
                internal_chat_message_metadata_passthrough: None,
            },
            ResponseItem::FunctionCallOutput {
                id: None,
                call_id: "call-1".to_string(),
                output: Default::default(),
                internal_chat_message_metadata_passthrough: None,
            },
        ]
    );
}

#[test]
fn partitioning_retains_unrelated_developer_text() {
    assert_eq!(
        partition_collaboration_mode_blocks(
            "before<collaboration_mode>plan</collaboration_mode>after",
        ),
        (
            "beforeafter".to_string(),
            vec!["<collaboration_mode>plan</collaboration_mode>".to_string()],
        )
    );
}
