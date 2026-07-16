use super::*;
use crate::render::renderable::Renderable;
use pretty_assertions::assert_eq;

#[tokio::test]
async fn long_exec_and_guardian_review_clear_after_completion() {
    const WIDTH: u16 = 181;

    let (mut chat, mut rx, _op_rx) = make_chatwidget_manual(/*model_override*/ None).await;
    chat.show_welcome_banner = false;
    chat.on_terminal_resize(WIDTH);
    chat.on_task_started();

    let token = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".repeat(55);
    let command = format!(
        "tmux send-keys -t remote-admin-mail 'printf %s {token} | base64 -d > /tmp/script.sh' Enter"
    );
    let begin = begin_exec(&mut chat, "long-command", &command);
    let action = GuardianAssessmentAction::Command {
        source: GuardianCommandSource::Shell,
        command: command.clone(),
        cwd: test_path_buf("/tmp/project").abs(),
    };
    chat.on_guardian_assessment(GuardianAssessmentEvent {
        id: "long-command-review".into(),
        target_item_id: Some("long-command".into()),
        turn_id: "turn-1".into(),
        started_at_ms: 0,
        completed_at_ms: None,
        status: GuardianAssessmentStatus::InProgress,
        risk_level: None,
        user_authorization: None,
        rationale: None,
        decision_source: None,
        action: action.clone(),
    });

    let pending_status = chat
        .bottom_pane
        .status_widget()
        .expect("guardian review status should be visible");
    assert_eq!(pending_status.header(), "Reviewing approval request");

    let pending_height = chat.desired_height(WIDTH);
    let mut pending_terminal =
        ratatui::Terminal::new(ratatui::backend::TestBackend::new(WIDTH, pending_height))
            .expect("create pending terminal");
    pending_terminal
        .draw(|frame| chat.render(frame.area(), frame.buffer_mut()))
        .expect("render pending long command and guardian review");
    assert_chatwidget_snapshot!(
        "long_exec_and_guardian_review_pending_181_columns",
        normalized_backend_snapshot(pending_terminal.backend())
    );

    chat.on_guardian_assessment(GuardianAssessmentEvent {
        id: "long-command-review".into(),
        target_item_id: Some("long-command".into()),
        turn_id: "turn-1".into(),
        started_at_ms: 0,
        completed_at_ms: Some(1),
        status: GuardianAssessmentStatus::Approved,
        risk_level: Some(GuardianRiskLevel::Low),
        user_authorization: Some(GuardianUserAuthorization::High),
        rationale: Some("Command is confined to the requested target.".into()),
        decision_source: Some(GuardianAssessmentDecisionSource::Agent),
        action,
    });
    end_exec(&mut chat, begin, "", "", /*exit_code*/ 0);
    handle_turn_completed(&mut chat, "turn-1", /*duration_ms*/ None);

    let mut completed_exec = None;
    while let Ok(event) = rx.try_recv() {
        if let AppEvent::InsertHistoryCell(cell) = event {
            let lines = cell.display_lines(WIDTH);
            if lines_to_single_string(&lines).contains("• Ran") {
                completed_exec = Some(cell);
            }
        }
    }
    let completed_exec = completed_exec.expect("completed exec history cell");
    assert_eq!(
        (
            chat.transcript.active_cell.is_none(),
            chat.bottom_pane.status_widget().is_none(),
        ),
        (true, true),
    );

    let completed_height = completed_exec.desired_height(WIDTH);
    let completed_area = Rect::new(0, 0, WIDTH, completed_height);
    let mut completed_buffer = ratatui::buffer::Buffer::empty(completed_area);
    completed_exec.render(completed_area, &mut completed_buffer);
    assert_chatwidget_snapshot!(
        "long_exec_after_guardian_and_command_completion_181_columns",
        format!("{completed_buffer:?}")
    );
}
