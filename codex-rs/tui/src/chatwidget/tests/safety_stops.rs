use super::*;
use pretty_assertions::assert_eq;

const CYBER_HISTORY: &str = concat!(
    "ⓘ This content can't be shown\n",
    "  We take extra caution with cybersecurity requests. If you’re a security\n",
    "  professional, you may be able to apply for Trusted Access.\n",
    "  Trusted Access: https://openai.com/form/enterprise-trusted-access-for-cyber/\n",
    "  Learn more: https://help.openai.com/en/articles/20001326\n",
);

#[derive(Debug, Eq, PartialEq)]
struct SafetyOutcome {
    history: Vec<String>,
    pending_notification: Option<Notification>,
    task_running: bool,
}

#[tokio::test]
async fn live_failed_turn_safety_stop_notifies() {
    let (mut chat, mut rx, _op_rx) = make_chatwidget_manual(/*model_override*/ None).await;

    chat.handle_server_notification(
        failed_turn(
            "turn-1",
            "server fallback message",
            Some(CodexErrorInfo::CyberPolicy),
        ),
        /*replay_kind*/ None,
    );

    assert_eq!(
        take_safety_outcome(&mut chat, &mut rx),
        stopped_with(Some(Notification::SafetyAlert))
    );
}

#[tokio::test]
async fn queued_user_turn_suppresses_safety_alert() {
    let (mut chat, mut rx, mut op_rx) = make_chatwidget_manual(/*model_override*/ None).await;
    chat.thread_id = Some(ThreadId::new());
    handle_turn_started(&mut chat, "turn-1");
    drain_insert_history(&mut rx);
    chat.queue_user_message("Continue".into());

    chat.handle_server_notification(
        safety_error("turn-1", "blocked", Some(CodexErrorInfo::CyberPolicy)),
        /*replay_kind*/ None,
    );

    assert_eq!(chat.pending_notification, None);
    assert!(chat.input_queue.queued_user_messages.is_empty());
    assert_matches!(next_submit_op(&mut op_rx), Op::UserTurn { .. });
}

#[tokio::test]
async fn queued_action_that_leaves_tui_idle_keeps_safety_alert() {
    let (mut chat, mut rx, mut op_rx) = make_chatwidget_manual(Some("gpt-5.2")).await;
    chat.thread_id = Some(ThreadId::new());
    handle_turn_started(&mut chat, "turn-1");
    drain_insert_history(&mut rx);
    chat.queue_user_message_with_options(
        "/model".into(),
        QueuedInputAction::ParseSlash,
        Vec::new(),
    );

    chat.handle_server_notification(
        safety_error("turn-1", "blocked", Some(CodexErrorInfo::CyberPolicy)),
        /*replay_kind*/ None,
    );

    assert_eq!(chat.pending_notification, Some(Notification::SafetyAlert));
    assert!(render_bottom_popup(&chat, /*width*/ 80).contains("Select Model"));
    assert_no_submit_op(&mut op_rx);
}

#[tokio::test]
async fn safety_stops_from_both_replay_kinds_do_not_notify() {
    for replay_kind in [
        ReplayKind::ResumeInitialMessages,
        ReplayKind::ThreadSnapshot,
    ] {
        let (mut chat, mut rx, _op_rx) = make_chatwidget_manual(/*model_override*/ None).await;

        chat.replay_thread_turns(
            vec![app_server_turn(
                "turn-1",
                AppServerTurnStatus::Failed,
                /*duration_ms*/ None,
                Some(AppServerTurnError {
                    message: "server fallback message".to_string(),
                    codex_error_info: Some(CodexErrorInfo::CyberPolicy),
                    additional_details: None,
                }),
            )],
            replay_kind,
        );

        assert_eq!(
            take_safety_outcome(&mut chat, &mut rx),
            stopped_with(None),
            "unexpected replay result for {replay_kind:?}"
        );
    }
}

#[tokio::test]
async fn replayed_non_retry_error_does_not_notify() {
    let (mut chat, mut rx, _op_rx) = make_chatwidget_manual(/*model_override*/ None).await;

    chat.handle_server_notification(
        safety_error(
            "turn-1",
            "server fallback message",
            Some(CodexErrorInfo::CyberPolicy),
        ),
        Some(ReplayKind::ResumeInitialMessages),
    );

    assert_eq!(take_safety_outcome(&mut chat, &mut rx), stopped_with(None));
}

#[tokio::test]
async fn paired_live_error_and_failed_turn_emit_once() {
    let (mut chat, mut rx, _op_rx) = make_chatwidget_manual(/*model_override*/ None).await;
    handle_turn_started(&mut chat, "turn-1");
    drain_insert_history(&mut rx);

    chat.handle_server_notification(
        safety_error(
            "turn-1",
            "server fallback message",
            Some(CodexErrorInfo::CyberPolicy),
        ),
        /*replay_kind*/ None,
    );
    chat.handle_server_notification(
        failed_turn(
            "turn-1",
            "server fallback message",
            Some(CodexErrorInfo::CyberPolicy),
        ),
        /*replay_kind*/ None,
    );

    assert_eq!(
        take_safety_outcome(&mut chat, &mut rx),
        stopped_with(Some(Notification::SafetyAlert))
    );
}

#[tokio::test]
async fn unrelated_errors_and_cyber_verification_warning_do_not_notify() {
    for (message, codex_error_info) in [
        ("permission denied", None),
        (
            "server overloaded",
            Some(CodexErrorInfo::ServerOverloadedBeforeInput),
        ),
        ("usage limited", Some(CodexErrorInfo::UsageLimitExceeded)),
    ] {
        let (mut chat, _rx, _op_rx) = make_chatwidget_manual(/*model_override*/ None).await;
        chat.handle_server_notification(
            safety_error("turn-1", message, codex_error_info),
            /*replay_kind*/ None,
        );
        assert_eq!(
            chat.pending_notification, None,
            "unexpected notification for {message}"
        );
    }

    let (mut retryable, _rx, _op_rx) = make_chatwidget_manual(/*model_override*/ None).await;
    retryable.handle_server_notification(
        ServerNotification::Error(ErrorNotification {
            error: AppServerTurnError {
                message: "try again".to_string(),
                codex_error_info: Some(CodexErrorInfo::CyberPolicy),
                additional_details: None,
            },
            will_retry: true,
            thread_id: "thread-1".to_string(),
            turn_id: "turn-1".to_string(),
        }),
        /*replay_kind*/ None,
    );
    assert_eq!(retryable.pending_notification, None);

    let (mut warning, _rx, _op_rx) = make_chatwidget_manual(/*model_override*/ None).await;
    warning.handle_server_notification(
        ServerNotification::ModelVerification(ModelVerificationNotification {
            thread_id: "thread-1".to_string(),
            turn_id: "turn-1".to_string(),
            verifications: vec![AppServerModelVerification::TrustedAccessForCyber],
        }),
        /*replay_kind*/ None,
    );
    assert_eq!(warning.pending_notification, None);
}

#[tokio::test]
async fn safety_alert_coalesces_at_action_priority() {
    let (mut chat, _rx, _op_rx) = make_chatwidget_manual(/*model_override*/ None).await;
    chat.notify(Notification::AgentTurnComplete {
        response: "done".to_string(),
    });
    chat.notify(Notification::SafetyAlert);
    assert_eq!(chat.pending_notification, Some(Notification::SafetyAlert));

    chat.notify(Notification::ExecApprovalRequested {
        command: "cargo test".to_string(),
    });
    assert_eq!(
        chat.pending_notification,
        Some(Notification::ExecApprovalRequested {
            command: "cargo test".to_string(),
        })
    );

    chat.notify(Notification::PlanModePrompt {
        title: "Choose a path".to_string(),
    });
    assert_eq!(
        chat.pending_notification,
        Some(Notification::PlanModePrompt {
            title: "Choose a path".to_string(),
        })
    );

    chat.notify(Notification::SafetyAlert);
    assert_eq!(chat.pending_notification, Some(Notification::SafetyAlert));
}

fn safety_error(
    turn_id: &str,
    message: &str,
    codex_error_info: Option<CodexErrorInfo>,
) -> ServerNotification {
    ServerNotification::Error(ErrorNotification {
        error: AppServerTurnError {
            message: message.to_string(),
            codex_error_info,
            additional_details: None,
        },
        will_retry: false,
        thread_id: "thread-1".to_string(),
        turn_id: turn_id.to_string(),
    })
}

fn failed_turn(
    turn_id: &str,
    message: &str,
    codex_error_info: Option<CodexErrorInfo>,
) -> ServerNotification {
    ServerNotification::TurnCompleted(TurnCompletedNotification {
        thread_id: "thread-1".to_string(),
        turn: app_server_turn(
            turn_id,
            AppServerTurnStatus::Failed,
            /*duration_ms*/ None,
            Some(AppServerTurnError {
                message: message.to_string(),
                codex_error_info,
                additional_details: None,
            }),
        ),
    })
}

fn take_safety_outcome(
    chat: &mut ChatWidget,
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<AppEvent>,
) -> SafetyOutcome {
    SafetyOutcome {
        history: drain_insert_history(rx)
            .iter()
            .map(|lines| lines_to_single_string(lines))
            .collect(),
        pending_notification: chat.pending_notification.take(),
        task_running: chat.bottom_pane.is_task_running(),
    }
}

fn stopped_with(pending_notification: Option<Notification>) -> SafetyOutcome {
    SafetyOutcome {
        history: vec![CYBER_HISTORY.to_string()],
        pending_notification,
        task_running: false,
    }
}
