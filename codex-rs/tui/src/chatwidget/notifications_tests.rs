use super::*;
use pretty_assertions::assert_eq;
use std::time::Duration;

#[test]
fn background_terminal_completion_contract() {
    let notification = Notification::BackgroundTerminalComplete {
        command: format!("printf done\u{7} {}", "x".repeat(100)),
        duration: Duration::from_millis(600_001),
        exit_code: 42,
    };

    assert_eq!(
        (
            notification.display(),
            notification.type_name(),
            notification.priority(),
            notification.allowed_for(&Notifications::Enabled(true)),
            notification.allowed_for(&Notifications::Enabled(false)),
            notification.allowed_for(&Notifications::Custom(vec![
                "background-terminal-complete".to_string(),
            ])),
            notification.allowed_for(&Notifications::Custom(vec![
                "agent-turn-complete".to_string(),
            ])),
        ),
        (
            format!(
                "Background terminal completed: printf done {}... • 10m 00s • exit 42",
                "x".repeat(65)
            ),
            "background-terminal-complete",
            0,
            true,
            false,
            true,
            false,
        )
    );
}

#[test]
fn safety_alert_contract() {
    let notification = Notification::SafetyAlert;
    let plan = Notification::PlanModePrompt {
        title: "Choose a path".to_string(),
    };
    let approval = Notification::ExecApprovalRequested {
        command: "cargo test".to_string(),
    };

    assert_eq!(
        (
            notification.display(),
            notification.type_name(),
            notification.priority(),
            notification.allowed_for(&Notifications::Enabled(true)),
            notification.allowed_for(&Notifications::Enabled(false)),
            notification.allowed_for(&Notifications::Custom(vec!["safety-alert".to_string()])),
            notification.allowed_for(&Notifications::Custom(vec![
                "agent-turn-complete".to_string(),
            ])),
            plan.priority(),
            approval.priority(),
        ),
        (
            "Codex stopped: This content can't be shown".to_string(),
            "safety-alert",
            1,
            true,
            false,
            true,
            false,
            1,
            1,
        )
    );
    insta::assert_snapshot!(notification.display());
}
