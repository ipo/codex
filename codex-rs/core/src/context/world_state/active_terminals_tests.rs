use super::*;
use crate::context::world_state::WorldState;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_utils_path_uri::PathUri;
use pretty_assertions::assert_eq;
use serde_json::json;

#[test]
fn renders_active_shared_terminal_metadata_without_output() {
    let state = ActiveTerminalsState::from_background_terminals(vec![terminal(
        "1000",
        Some("default"),
        "/bin/bash -l",
        BackgroundTerminalSource::SharedTerminal,
    )]);
    let mut world_state = WorldState::default();
    world_state.add_section(state);

    assert_eq!(
        vec![user_message(
            r#"<active_terminals>
  <terminal label="default" process_id="1000" source="sharedTerminal" status="running">
    <cwd>/repo</cwd>
    <command>/bin/bash -l</command>
  </terminal>
</active_terminals>"#,
        )],
        render_fragments(world_state.render_full()),
    );
    assert_eq!(
        serde_json::to_value(world_state.snapshot()).expect("serialize world-state snapshot"),
        json!({
            "active_terminals": {
                "terminals": {
                    "1000": {
                        "label": "default",
                        "process_id": "1000",
                        "cwd": "/repo",
                        "command": "/bin/bash -l",
                        "source": "sharedTerminal",
                        "status": "running",
                    }
                }
            }
        }),
    );
    assert_eq!(
        Vec::<ResponseItem>::new(),
        render_fragments(world_state.render_diff(&world_state.snapshot()))
    );
}

#[test]
fn filters_non_shared_terminals_and_caps_metadata() {
    let mut terminals = (0..20)
        .map(|idx| {
            terminal(
                &format!("{}", 1000 + idx),
                Some("shared"),
                &"x".repeat(MAX_CONTEXT_FIELD_CHARS + 40),
                BackgroundTerminalSource::SharedTerminal,
            )
        })
        .collect::<Vec<_>>();
    terminals.push(terminal(
        "2000",
        None,
        "agent command",
        BackgroundTerminalSource::Agent,
    ));

    let state = ActiveTerminalsState::from_background_terminals(terminals);
    let snapshot = state.snapshot();

    assert_eq!(snapshot.terminals.len(), MAX_ACTIVE_TERMINALS);
    assert!(!snapshot.terminals.contains_key("2000"));
    let command = &snapshot
        .terminals
        .get("1000")
        .expect("first shared terminal")
        .command;
    assert_eq!(command.chars().count(), MAX_CONTEXT_FIELD_CHARS);
    assert!(command.ends_with("..."));
}

#[test]
fn renders_removed_terminal_as_unavailable() {
    let mut previous = WorldState::default();
    previous.add_section(ActiveTerminalsState::from_background_terminals(vec![
        terminal(
            "1000",
            Some("default"),
            "/bin/bash -l",
            BackgroundTerminalSource::SharedTerminal,
        ),
    ]));
    let mut current = WorldState::default();
    current.add_section(ActiveTerminalsState::default());

    assert_eq!(
        vec![user_message(
            r#"<active_terminals>
  <terminal label="default" process_id="1000" source="sharedTerminal" status="unavailable" />
</active_terminals>"#,
        )],
        render_fragments(current.render_diff(&previous.snapshot())),
    );
}

fn terminal(
    process_id: &str,
    label: Option<&str>,
    command: &str,
    source: BackgroundTerminalSource,
) -> BackgroundTerminalInfo {
    BackgroundTerminalInfo {
        item_id: format!("item-{process_id}"),
        process_id: process_id.to_string(),
        command: command.to_string(),
        cwd: PathUri::parse("file:///repo").expect("valid cwd"),
        source,
        label: label.map(str::to_string),
        tty: true,
        terminal_size: None,
        status: BackgroundTerminalStatus::Running,
    }
}

fn render_fragments(fragments: Vec<Box<dyn ContextualUserFragment>>) -> Vec<ResponseItem> {
    fragments
        .into_iter()
        .map(ContextualUserFragment::into_boxed_response_item)
        .collect()
}

fn user_message(text: &str) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: text.to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }
}
