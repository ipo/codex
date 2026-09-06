use super::*;
use codex_features::Feature;
use pretty_assertions::assert_eq;

fn replace_test_chat_widget_with_initial_message(
    app: &mut App,
    initial_user_message: Option<crate::chatwidget::UserMessage>,
) {
    let config = app.config.clone();
    let model = get_model_offline_for_tests(config.model.as_deref());
    app.chat_widget = ChatWidget::new_with_app_event(ChatWidgetInit {
        config,
        frame_requester: crate::tui::FrameRequester::test_dummy(),
        app_event_tx: app.app_event_tx.clone(),
        workspace_command_runner: None,
        initial_user_message,
        enhanced_keys_supported: false,
        has_chatgpt_account: false,
        has_codex_backend_auth: false,
        model_catalog: app.model_catalog.clone(),
        feedback: codex_feedback::CodexFeedback::new(),
        is_first_run: false,
        status_account_display: None,
        runtime_model_provider_base_url: None,
        initial_plan_type: None,
        model: Some(model),
        startup_tooltip_override: None,
        status_line_invalid_items_warned: app.status_line_invalid_items_warned.clone(),
        terminal_title_invalid_items_warned: app.terminal_title_invalid_items_warned.clone(),
        session_telemetry: app.session_telemetry.clone(),
    });
}

#[tokio::test]
async fn startup_goal_handoff_is_shared_by_fresh_resume_and_fork_routes() -> Result<()> {
    #[derive(Clone, Copy, Debug)]
    enum StartupRoute {
        Fresh,
        Resume,
        Fork,
    }

    for route in [
        StartupRoute::Fresh,
        StartupRoute::Resume,
        StartupRoute::Fork,
    ] {
        let (mut app, mut app_event_rx, mut op_rx) = make_test_app_with_channels().await;
        app.chat_widget
            .set_feature_enabled(Feature::Goals, /*enabled*/ true);
        replace_test_chat_widget_with_initial_message(
            &mut app,
            create_initial_user_message(
                Some("/goal investigate the flaky test".to_string()),
                Vec::new(),
                Vec::new(),
            ),
        );
        let thread_id = ThreadId::new();
        let mut session = test_thread_session(thread_id, test_path_buf("/tmp/project"));
        let turns = match route {
            StartupRoute::Fresh | StartupRoute::Fork => Vec::new(),
            StartupRoute::Resume => {
                vec![test_turn("resumed-turn", TurnStatus::Completed, Vec::new())]
            }
        };
        if matches!(route, StartupRoute::Fork) {
            session.forked_from_id = Some(ThreadId::new());
        }

        assert_eq!(
            app.enqueue_primary_thread_session(session, turns).await?,
            crate::chatwidget::InitialUserMessageSubmission::Stop,
            "unexpected handoff outcome for {route:?}"
        );

        let events = std::iter::from_fn(|| app_event_rx.try_recv().ok()).collect::<Vec<_>>();
        assert!(
            events.iter().any(|event| matches!(
                event,
                AppEvent::SetThreadGoalDraft {
                    thread_id: event_thread_id,
                    draft,
                    mode: crate::app_event::ThreadGoalSetMode::ConfirmIfExists,
                } if *event_thread_id == thread_id
                    && draft == &crate::goal_files::GoalDraft {
                        objective: "investigate the flaky test".to_string(),
                        ..Default::default()
                    }
            )),
            "expected startup goal event for {route:?}; events: {events:?}"
        );
        assert!(events.iter().all(|event| !matches!(
            event,
            AppEvent::CodexOp(Op::UserTurn { items, .. })
                | AppEvent::SubmitThreadOp {
                    op: Op::UserTurn { items, .. },
                    ..
                }
                if items.iter().any(|item| matches!(
                    item,
                    UserInput::Text { text, .. } if text.starts_with("/goal")
                ))
        )));
        while let Ok(op) = op_rx.try_recv() {
            assert!(
                !matches!(op, Op::UserTurn { ref items, .. } if items.iter().any(|item| matches!(
                    item,
                    UserInput::Text { text, .. } if text.starts_with("/goal")
                ))),
                "startup goal reached the model for {route:?}: {op:?}"
            );
        }
    }

    Ok(())
}

#[tokio::test]
async fn startup_goal_is_persisted_active_and_starts_canonical_continuation() -> Result<()> {
    use wiremock::Mock;
    use wiremock::ResponseTemplate;
    use wiremock::matchers::method;
    use wiremock::matchers::path;

    let server = core_test_support::responses::start_mock_server().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_delay(std::time::Duration::from_secs(/*secs*/ 2))
                .set_body_string(core_test_support::responses::sse(vec![
                    core_test_support::responses::ev_response_created(
                        "resp-startup-goal-continuation",
                    ),
                    core_test_support::responses::ev_completed("resp-startup-goal-continuation"),
                ])),
        )
        .mount(&server)
        .await;

    let (mut app, mut app_event_rx, mut op_rx) = make_test_app_with_channels().await;
    app.config.model_provider.base_url = Some(format!("{}/v1", server.uri()));
    app.config.model_provider.supports_websockets = false;
    app.config
        .features
        .set_enabled(Feature::Goals, /*enabled*/ true)?;
    replace_test_chat_widget_with_initial_message(
        &mut app,
        create_initial_user_message(
            Some("/goal investigate the flaky test".to_string()),
            Vec::new(),
            Vec::new(),
        ),
    );
    let mut app_server = crate::start_embedded_app_server_for_picker(&app.config).await?;
    let started = app_server.start_thread(&app.config).await?;
    let thread_id = started.session.thread_id;

    assert_eq!(
        app.enqueue_primary_thread_session(started.session, started.turns)
            .await?,
        crate::chatwidget::InitialUserMessageSubmission::Stop
    );
    let draft = std::iter::from_fn(|| app_event_rx.try_recv().ok())
        .find_map(|event| match event {
            AppEvent::SetThreadGoalDraft {
                thread_id: event_thread_id,
                draft,
                mode: crate::app_event::ThreadGoalSetMode::ConfirmIfExists,
            } if event_thread_id == thread_id => Some(draft),
            _ => None,
        })
        .expect("startup /goal should request a goal set");
    assert_eq!(
        draft,
        crate::goal_files::GoalDraft {
            objective: "investigate the flaky test".to_string(),
            ..Default::default()
        }
    );

    app.set_thread_goal_draft(
        &mut app_server,
        thread_id,
        draft,
        crate::app_event::ThreadGoalSetMode::ConfirmIfExists,
    )
    .await;
    let goal = app_server
        .thread_goal_get(thread_id)
        .await?
        .goal
        .expect("startup goal should be persisted");
    assert_eq!(goal.objective, "investigate the flaky test");
    assert_eq!(
        goal.status,
        codex_app_server_protocol::ThreadGoalStatus::Active
    );

    let mut continuation_turn = None;
    for _ in 0..20 {
        let event = time::timeout(
            std::time::Duration::from_secs(/*secs*/ 2),
            app_server.next_event(),
        )
        .await
        .expect("app-server should emit the goal continuation")
        .expect("app-server event stream should remain open");
        if let codex_app_server_client::AppServerEvent::ServerNotification(notification) = &event
            && let ServerNotification::TurnStarted(notification) = notification.as_ref()
            && notification.thread_id == thread_id.to_string()
        {
            continuation_turn = Some(notification.turn.clone());
            break;
        }
    }
    assert!(
        continuation_turn.is_some(),
        "goal continuation did not start"
    );
    while let Ok(op) = op_rx.try_recv() {
        assert!(!matches!(op, Op::UserTurn { .. }));
    }

    app_server.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn unknown_startup_command_immediately_drains_queued_input() -> Result<()> {
    let (mut app, mut app_event_rx, mut op_rx) = make_test_app_with_channels().await;
    replace_test_chat_widget_with_initial_message(
        &mut app,
        create_initial_user_message(Some("/does-not-exist".to_string()), Vec::new(), Vec::new()),
    );
    app.chat_widget
        .set_queue_submissions_until_session_configured(/*queue*/ true);
    app.chat_widget
        .apply_external_edit("queued after startup command".to_string());
    app.chat_widget
        .handle_key_event(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

    let thread_id = ThreadId::new();
    assert_eq!(
        app.enqueue_primary_thread_session(
            test_thread_session(thread_id, test_path_buf("/tmp/project")),
            Vec::new(),
        )
        .await?,
        crate::chatwidget::InitialUserMessageSubmission::Continue
    );

    let events = std::iter::from_fn(|| app_event_rx.try_recv().ok()).collect::<Vec<_>>();
    let rendered = events
        .iter()
        .filter_map(|event| match event {
            AppEvent::InsertHistoryCell(cell) => {
                Some(lines_to_single_string(&cell.display_lines(/*width*/ 80)))
            }
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(rendered.contains("Unrecognized command '/does-not-exist'"));
    assert!(
        events.iter().any(|event| matches!(
            event,
            AppEvent::CodexOp(Op::UserTurn { items, .. })
                | AppEvent::SubmitThreadOp {
                    op: Op::UserTurn { items, .. },
                    ..
                }
                if items == &vec![UserInput::Text {
                    text: "queued after startup command".to_string(),
                    text_elements: Vec::new(),
                }]
        )),
        "expected queued prompt submission; events: {events:?}"
    );
    while let Ok(op) = op_rx.try_recv() {
        assert!(
            !matches!(op, Op::UserTurn { .. }),
            "app-event-backed widget unexpectedly used direct op channel: {op:?}"
        );
    }
    assert!(app.chat_widget.queued_user_message_texts().is_empty());

    Ok(())
}
