use super::*;
use pretty_assertions::assert_eq;

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
