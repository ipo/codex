use super::*;
use codex_app_server_protocol::ThreadGoalStatus as Status;
use ratatui::Terminal;
use ratatui::backend::TestBackend;

#[tokio::test]
async fn status_line_goal_lifecycle_provenance_footer_snapshot() {
    let (mut chat, _rx, _op_rx) = make_chatwidget_manual(Some("gpt-5.4")).await;
    chat.set_feature_enabled(Feature::Goals, /*enabled*/ true);
    chat.show_welcome_banner = false;
    chat.config.tui_status_line = Some(vec!["model-name".to_string()]);
    chat.refresh_status_line();

    let cases = [
        ("active budget", Status::Active, Some(50_000), 40_000, 1_800),
        ("active elapsed", Status::Active, None, 0, 1_800),
        ("paused", Status::Paused, None, 0, 1_800),
        ("blocked", Status::Blocked, None, 0, 1_800),
        ("usage limited", Status::UsageLimited, None, 0, 1_800),
        (
            "budget exceeded",
            Status::BudgetLimited,
            Some(50_000),
            51_000,
            1_800,
        ),
        ("budget absent", Status::BudgetLimited, None, 0, 1_800),
        (
            "complete budget",
            Status::Complete,
            Some(50_000),
            40_000,
            1_800,
        ),
        ("complete elapsed", Status::Complete, None, 40_000, 36_720),
    ];
    let mut states = Vec::new();

    for (name, status, token_budget, tokens_used, time_used_seconds) in cases {
        let mut goal = thread_goal(status, token_budget, tokens_used);
        goal.time_used_seconds = time_used_seconds;
        chat.handle_server_notification(
            ServerNotification::ThreadGoalUpdated(
                codex_app_server_protocol::ThreadGoalUpdatedNotification {
                    thread_id: "thread-1".to_string(),
                    turn_id: None,
                    goal,
                },
            ),
            /*replay_kind*/ None,
        );

        let width = 120;
        let height = chat.desired_height(width);
        let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("create terminal");
        terminal
            .draw(|f| chat.render(f.area(), f.buffer_mut()))
            .expect("draw goal lifecycle footer");
        let rendered = normalized_backend_snapshot(terminal.backend());
        let footer = rendered.lines().last().expect("rendered footer line");
        states.push(format!("{name}: {footer}"));
    }

    assert_chatwidget_snapshot!(
        "status_line_goal_lifecycle_provenance_footer",
        states.join("\n---\n")
    );
}

#[tokio::test]
async fn status_line_goal_provenance_context_priority_snapshot() {
    let (mut chat, _rx, _op_rx) = make_chatwidget_manual(Some("gpt-5.4")).await;
    chat.set_feature_enabled(Feature::Goals, /*enabled*/ true);
    chat.show_welcome_banner = false;
    chat.config.tui_status_line = Some(vec!["model-name".to_string()]);
    chat.refresh_status_line();
    chat.handle_server_notification(
        ServerNotification::ThreadGoalUpdated(
            codex_app_server_protocol::ThreadGoalUpdatedNotification {
                thread_id: "thread-1".to_string(),
                turn_id: None,
                goal: thread_goal(
                    Status::Active,
                    /*token_budget*/ Some(50_000),
                    /*tokens_used*/ 40_000,
                ),
            },
        ),
        /*replay_kind*/ None,
    );

    let cases = [
        ("status", false, false, 80),
        ("status + IDE", true, false, 120),
        ("status + Vim", false, true, 120),
        ("all constrained", true, true, 50),
    ];
    let mut states = Vec::new();

    for (name, ide_context_active, vim_enabled, width) in cases {
        chat.bottom_pane.set_ide_context_active(ide_context_active);
        chat.bottom_pane.set_vim_enabled(vim_enabled);
        let height = chat.desired_height(width);
        let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("create terminal");
        terminal
            .draw(|f| chat.render(f.area(), f.buffer_mut()))
            .expect("draw constrained goal footer");
        let rendered = normalized_backend_snapshot(terminal.backend());
        let footer = rendered.lines().last().expect("rendered footer line");
        states.push(format!("{name} ({width} columns): {footer}"));
    }

    assert_chatwidget_snapshot!(
        "status_line_goal_provenance_context_priority",
        states.join("\n---\n")
    );
}

fn thread_goal(
    status: Status,
    token_budget: Option<i64>,
    tokens_used: i64,
) -> codex_app_server_protocol::ThreadGoal {
    codex_app_server_protocol::ThreadGoal {
        thread_id: "thread-1".to_string(),
        objective: "Keep improving the benchmark".to_string(),
        status,
        token_budget,
        tokens_used,
        time_used_seconds: 30 * 60,
        created_at: 0,
        updated_at: 0,
    }
}
