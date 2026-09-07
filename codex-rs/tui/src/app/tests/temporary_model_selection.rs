use super::*;
use crate::chatwidget::UserMessage;
use pretty_assertions::assert_eq;

fn temporary_user_turn(app: &App) -> Op {
    let temporary_mode = app.chat_widget.effective_collaboration_mode();
    let config = app.chat_widget.config_ref();
    let text = "temporary selection turn".to_string();
    Op::UserTurn {
        items: vec![UserInput::Text {
            text: text.clone(),
            text_elements: Vec::new(),
        }],
        cwd: config.cwd.to_path_buf(),
        approval_policy: AskForApproval::from(config.permissions.approval_policy.value()),
        approvals_reviewer: None,
        active_permission_profile: config.permissions.active_permission_profile(),
        model: temporary_mode.model().to_string(),
        effort: temporary_mode.reasoning_effort(),
        summary: None,
        service_tier: None,
        final_output_json_schema: None,
        collaboration_mode: Some(temporary_mode),
        personality: None,
        submitted_user_message: Some(UserMessage::from(text)),
    }
}

fn thread_settings_updated(
    app: &App,
    thread_id: ThreadId,
    collaboration_mode: CollaborationMode,
) -> ThreadSettingsUpdatedNotification {
    ThreadSettingsUpdatedNotification {
        thread_id: thread_id.to_string(),
        thread_settings: ThreadSettings {
            cwd: app.chat_widget.config_ref().cwd.clone(),
            approval_policy: AskForApproval::Never,
            approvals_reviewer: codex_app_server_protocol::ApprovalsReviewer::User,
            sandbox_policy: codex_app_server_protocol::SandboxPolicy::ReadOnly {
                network_access: false,
            },
            active_permission_profile: None,
            model: collaboration_mode.model().to_string(),
            model_provider: "test-provider".to_string(),
            service_tier: None,
            effort: collaboration_mode.reasoning_effort(),
            summary: None,
            collaboration_mode,
            multi_agent_mode: Default::default(),
            personality: None,
        },
    }
}

#[tokio::test]
async fn temporary_model_selection_uses_live_turn_then_restores_before_resume() {
    let mut app = make_test_app().await;
    let config_path = app.config.codex_home.join("config.toml");
    let original_config = "model = \"gpt-5.4\"\nmodel_reasoning_effort = \"low\"\n";
    std::fs::write(&config_path, original_config).expect("write config");

    let mut app_server = crate::start_embedded_app_server_for_picker(app.chat_widget.config_ref())
        .await
        .expect("embedded app server");
    let started = app_server
        .start_thread(app.chat_widget.config_ref())
        .await
        .expect("thread/start should succeed");
    let thread_id = started.session.thread_id;
    app.enqueue_primary_thread_session(started.session, started.turns)
        .await
        .expect("primary thread should be registered");

    let saved_mode = app.chat_widget.persisted_effective_collaboration_mode();
    app.chat_widget
        .apply_temporary_model_selection("gpt-5.2", Some(ReasoningEffortConfig::High));
    let turn = temporary_user_turn(&app);

    assert!(
        app.try_submit_active_thread_op_via_app_server(&mut app_server, thread_id, &turn)
            .await
            .expect("temporary turn should submit")
    );
    assert!(app.chat_widget.has_pending_temporary_settings_restore());

    let temporary_update = next_thread_settings_updated(&mut app_server, thread_id).await;
    assert_eq!(temporary_update.thread_settings.model, "gpt-5.2");
    assert_eq!(
        temporary_update.thread_settings.effort,
        Some(ReasoningEffortConfig::High)
    );
    app.chat_widget.on_thread_settings_updated(temporary_update);
    assert!(app.chat_widget.has_pending_temporary_settings_restore());
    let restored_update = next_thread_settings_updated(&mut app_server, thread_id).await;
    assert_eq!(
        restored_update.thread_settings.collaboration_mode,
        saved_mode
    );
    app.chat_widget.on_thread_settings_updated(restored_update);
    assert!(!app.chat_widget.has_pending_temporary_settings_restore());

    app_server.shutdown().await.expect("shutdown app server");
    let mut resumed_server =
        crate::start_embedded_app_server_for_picker(app.chat_widget.config_ref())
            .await
            .expect("fresh embedded app server");
    let resumed = resumed_server
        .resume_thread(
            app.config.clone(),
            thread_id,
            crate::app_server_session::ResumeModelSettings::RestoreFromThread,
        )
        .await
        .expect("thread should resume with restored settings");
    assert_eq!(resumed.session.model, saved_mode.model());
    assert_eq!(
        resumed.session.reasoning_effort,
        saved_mode.reasoning_effort()
    );
    assert_eq!(
        std::fs::read_to_string(config_path).expect("read config"),
        original_config
    );
    resumed_server
        .shutdown()
        .await
        .expect("shutdown resumed server");
}

#[tokio::test]
async fn temporary_model_selection_requires_thread_settings_update_support() {
    let mut app = make_test_app().await;
    let mut app_server = crate::start_embedded_app_server_for_picker(app.chat_widget.config_ref())
        .await
        .expect("embedded app server");
    let started = app_server
        .start_thread(app.chat_widget.config_ref())
        .await
        .expect("thread/start should succeed");
    let thread_id = started.session.thread_id;
    app.enqueue_primary_thread_session(started.session, started.turns)
        .await
        .expect("primary thread should be registered");
    app.chat_widget
        .apply_temporary_model_selection("gpt-5.2", Some(ReasoningEffortConfig::High));
    let turn = temporary_user_turn(&app);
    app_server.disable_thread_settings_update_for_test();

    assert!(
        app.try_submit_active_thread_op_via_app_server(&mut app_server, thread_id, &turn)
            .await
            .expect("unsupported temporary turn should be handled")
    );
    assert_eq!(
        app.chat_widget.composer_text_with_pending(),
        "temporary selection turn"
    );
    assert!(!app.chat_widget.has_pending_temporary_settings_restore());

    app_server.shutdown().await.expect("shutdown app server");
}

#[tokio::test]
async fn failed_temporary_settings_restore_blocks_clean_shutdown() {
    let mut app = make_test_app().await;
    let mut app_server = crate::start_embedded_app_server_for_picker(app.chat_widget.config_ref())
        .await
        .expect("embedded app server");
    let started = app_server
        .start_thread(app.chat_widget.config_ref())
        .await
        .expect("thread/start should succeed");
    let thread_id = started.session.thread_id;
    app.enqueue_primary_thread_session(started.session, started.turns)
        .await
        .expect("primary thread should be registered");
    let saved_mode = app.chat_widget.persisted_effective_collaboration_mode();
    app.chat_widget
        .apply_temporary_model_selection("gpt-5.2", Some(ReasoningEffortConfig::High));
    let temporary_mode = app.chat_widget.effective_collaboration_mode();
    app.chat_widget
        .mark_temporary_settings_restore_pending(thread_id, saved_mode, temporary_mode);
    app_server.disable_thread_settings_update_for_test();

    let control = app
        .handle_exit_mode(&mut app_server, ExitMode::ShutdownFirst)
        .await;
    assert!(matches!(control, AppRunControl::Continue));
    assert!(app.chat_widget.has_pending_temporary_settings_restore());

    app_server.shutdown().await.expect("shutdown app server");
}

#[tokio::test]
async fn temporary_settings_restores_remain_pending_per_thread() {
    let mut app = make_test_app().await;
    let mut app_server = crate::start_embedded_app_server_for_picker(app.chat_widget.config_ref())
        .await
        .expect("embedded app server");
    let first = app_server
        .start_thread(app.chat_widget.config_ref())
        .await
        .expect("first thread/start should succeed");
    let first_thread_id = first.session.thread_id;
    let second = app_server
        .start_thread(app.chat_widget.config_ref())
        .await
        .expect("second thread/start should succeed");
    let second_thread_id = second.session.thread_id;
    app.enqueue_primary_thread_session(second.session, second.turns)
        .await
        .expect("second thread should be registered");

    let saved_mode = app.chat_widget.persisted_effective_collaboration_mode();
    let temporary_mode = saved_mode.with_updates(
        Some("gpt-5.2".to_string()),
        Some(Some(ReasoningEffortConfig::High)),
        /*developer_instructions*/ None,
    );
    let first_saved_mode = saved_mode.clone();
    app.chat_widget.mark_temporary_settings_restore_pending(
        first_thread_id,
        saved_mode.clone(),
        temporary_mode.clone(),
    );
    app.chat_widget.mark_temporary_settings_restore_pending(
        second_thread_id,
        saved_mode.clone(),
        temporary_mode.clone(),
    );

    app.chat_widget
        .on_thread_settings_updated(thread_settings_updated(
            &app,
            second_thread_id,
            temporary_mode,
        ));
    app.chat_widget
        .on_thread_settings_updated(thread_settings_updated(&app, second_thread_id, saved_mode));

    assert_eq!(
        app.chat_widget.pending_temporary_settings_restores(),
        vec![(first_thread_id, first_saved_mode)]
    );
    app_server.disable_thread_settings_update_for_test();
    let control = app
        .handle_exit_mode(&mut app_server, ExitMode::ShutdownFirst)
        .await;
    assert!(matches!(control, AppRunControl::Continue));
    assert_eq!(
        app.chat_widget.pending_temporary_settings_restores().len(),
        1
    );

    app_server.shutdown().await.expect("shutdown app server");
}
