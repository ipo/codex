use super::*;
use pretty_assertions::assert_eq;

fn configure_mcp_servers(app: &mut App) {
    let config: codex_config::types::McpServerConfig =
        toml::from_str::<toml::Value>("command = 'true'")
            .expect("test MCP config should parse")
            .try_into()
            .expect("test MCP config should deserialize");
    app.config
        .mcp_servers
        .set(HashMap::from([
            ("eager".to_string(), config.clone()),
            ("deferred".to_string(), config),
        ]))
        .expect("test MCP servers should accept any configuration");
}

#[tokio::test]
async fn app_server_lag_keeps_mcp_startup_visible_until_terminal_status() {
    let (mut app, mut app_event_rx, _op_rx) = make_test_app_with_channels().await;
    while app_event_rx.try_recv().is_ok() {}
    let config: codex_config::types::McpServerConfig =
        toml::from_str::<toml::Value>("command = 'true'")
            .expect("test MCP config should parse")
            .try_into()
            .expect("test MCP config should deserialize");
    app.config
        .mcp_servers
        .set(HashMap::from([("sentry".to_string(), config)]))
        .expect("test MCP servers should accept any configuration");
    let app_server = crate::start_embedded_app_server_for_picker(app.chat_widget.config_ref())
        .await
        .expect("embedded app server");
    let thread_id = ThreadId::new();
    app.primary_thread_id = Some(thread_id);
    app.ensure_thread_channel(thread_id);
    app.activate_thread_channel(thread_id).await;

    app.handle_app_server_event(
        &app_server,
        codex_app_server_client::AppServerEvent::ServerNotification(Box::new(
            ServerNotification::McpServerStatusUpdated(McpServerStatusUpdatedNotification {
                thread_id: Some(thread_id.to_string()),
                name: "sentry".to_string(),
                status: McpServerStartupState::Starting,
                error: None,
                failure_reason: None,
            }),
        )),
    )
    .await;
    let event = app
        .active_thread_rx
        .as_mut()
        .expect("active thread receiver")
        .try_recv()
        .expect("MCP startup update should be queued for the active thread");
    app.handle_thread_event_now(event);

    app.handle_app_server_event(
        &app_server,
        codex_app_server_client::AppServerEvent::Lagged { skipped: 7 },
    )
    .await;

    assert!(app.chat_widget.is_task_running_for_test());
    insta::assert_snapshot!(
        "app_server_lag_keeps_mcp_startup_visible_until_terminal_status",
        render_bottom_popup(&app.chat_widget, /*width*/ 80)
    );

    app.handle_app_server_event(
        &app_server,
        codex_app_server_client::AppServerEvent::ServerNotification(Box::new(
            ServerNotification::McpServerStatusUpdated(McpServerStatusUpdatedNotification {
                thread_id: Some(thread_id.to_string()),
                name: "sentry".to_string(),
                status: McpServerStartupState::Ready,
                error: None,
                failure_reason: None,
            }),
        )),
    )
    .await;
    let event = app
        .active_thread_rx
        .as_mut()
        .expect("active thread receiver")
        .try_recv()
        .expect("MCP terminal update should be queued for the active thread");
    app.handle_thread_event_now(event);

    assert!(!app.chat_widget.is_task_running_for_test());
    let warnings = std::iter::from_fn(|| app_event_rx.try_recv().ok())
        .filter_map(|event| match event {
            AppEvent::InsertHistoryCell(cell) => {
                Some(lines_to_single_string(&cell.display_lines(/*width*/ 80)))
            }
            _ => None,
        })
        .collect::<String>();
    assert!(!warnings.contains("MCP startup interrupted"));
}

#[tokio::test]
async fn late_receiver_replays_terminal_mcp_status_from_bounded_reconnect_store() {
    let (mut app, mut app_event_rx, _op_rx) = make_test_app_with_channels().await;
    while app_event_rx.try_recv().is_ok() {}
    let config: codex_config::types::McpServerConfig =
        toml::from_str::<toml::Value>("command = 'true'")
            .expect("test MCP config should parse")
            .try_into()
            .expect("test MCP config should deserialize");
    app.config
        .mcp_servers
        .set(HashMap::from([("sentry".to_string(), config)]))
        .expect("test MCP servers should accept any configuration");
    let thread_id = ThreadId::new();
    app.primary_thread_id = Some(thread_id);
    app.thread_event_channels.insert(
        thread_id,
        ThreadEventChannel::new_with_session(
            /*capacity*/ 1,
            test_thread_session(thread_id, test_path_buf("/tmp/late-mcp")),
            Vec::new(),
        ),
    );

    for status in [
        McpServerStartupState::Starting,
        McpServerStartupState::Ready,
    ] {
        app.enqueue_primary_thread_notification(ServerNotification::McpServerStatusUpdated(
            McpServerStatusUpdatedNotification {
                thread_id: Some(thread_id.to_string()),
                name: "sentry".to_string(),
                status,
                error: None,
                failure_reason: None,
            },
        ))
        .await
        .expect("inactive thread should retain MCP status");
    }

    let snapshot = {
        let channel = app
            .thread_event_channels
            .get(&thread_id)
            .expect("thread channel");
        let mut store = channel.store.lock().await;
        assert_eq!(store.buffer.len(), 1);
        store.rebase_buffer_after_session_refresh();
        store.snapshot()
    };
    app.replay_thread_snapshot(snapshot, /*resume_restored_queue*/ false);

    assert!(!app.chat_widget.is_task_running_for_test());
    let warnings = std::iter::from_fn(|| app_event_rx.try_recv().ok())
        .filter_map(|event| match event {
            AppEvent::InsertHistoryCell(cell) => {
                Some(lines_to_single_string(&cell.display_lines(/*width*/ 80)))
            }
            _ => None,
        })
        .collect::<String>();
    assert!(!warnings.contains("MCP startup interrupted"));
}

#[tokio::test]
async fn subagent_mcp_startup_settles_while_cached_servers_remain_deferred() {
    let (mut app, _app_event_rx, _op_rx) = make_test_app_with_channels().await;
    configure_mcp_servers(&mut app);
    let app_server = crate::start_embedded_app_server_for_picker(app.chat_widget.config_ref())
        .await
        .expect("embedded app server");
    let root_thread_id = ThreadId::new();
    let subagent_thread_id = ThreadId::new();
    app.primary_thread_id = Some(root_thread_id);
    app.upsert_agent_picker_thread(
        subagent_thread_id,
        /*agent_nickname*/ None,
        /*agent_role*/ None,
        /*is_closed*/ false,
    );
    app.ensure_thread_channel(subagent_thread_id);
    app.activate_thread_channel(subagent_thread_id).await;
    app.replay_thread_snapshot(
        ThreadEventSnapshot {
            session: Some(test_thread_session(
                subagent_thread_id,
                test_path_buf("/tmp/subagent"),
            )),
            turns: Vec::new(),
            events: Vec::new(),
            input_state: None,
        },
        /*resume_restored_queue*/ false,
    );

    let mut visible_startup_states = Vec::new();
    for (name, status, task_running) in [
        ("eager", McpServerStartupState::Starting, true),
        ("eager", McpServerStartupState::Ready, false),
        ("deferred", McpServerStartupState::Starting, true),
        ("deferred", McpServerStartupState::Ready, false),
    ] {
        app.handle_app_server_event(
            &app_server,
            codex_app_server_client::AppServerEvent::ServerNotification(Box::new(
                ServerNotification::McpServerStatusUpdated(McpServerStatusUpdatedNotification {
                    thread_id: Some(subagent_thread_id.to_string()),
                    name: name.to_string(),
                    status,
                    error: None,
                    failure_reason: None,
                }),
            )),
        )
        .await;
        let event = app
            .active_thread_rx
            .as_mut()
            .expect("subagent receiver should be active")
            .try_recv()
            .expect("MCP startup update should reach the active subagent");
        app.handle_thread_event_now(event);

        assert_eq!(app.chat_widget.is_task_running_for_test(), task_running);
        let rendered = render_bottom_popup(&app.chat_widget, /*width*/ 80);
        let visible_status = rendered
            .lines()
            .find(|line| line.contains("Booting MCP server:"))
            .and_then(|line| line.split_once(" ("))
            .map_or("idle", |(status, _)| status);
        visible_startup_states.push(format!("{name}: {visible_status}"));
    }

    insta::assert_snapshot!(visible_startup_states.join("\n"), @r"
    eager: • Booting MCP server: eager
    eager: idle
    deferred: • Booting MCP server: deferred
    deferred: idle
    ");
}

#[tokio::test]
async fn resumed_subagent_mcp_startup_settles_while_cached_servers_remain_deferred() {
    let mut app = make_test_app().await;
    configure_mcp_servers(&mut app);
    let subagent_thread_id = ThreadId::new();
    app.primary_thread_id = Some(subagent_thread_id);
    app.active_thread_id = Some(subagent_thread_id);
    app.agent_navigation.mark_parent_owned(subagent_thread_id);
    app.refresh_mcp_startup_expected_servers_from_config();

    for status in [
        McpServerStartupState::Starting,
        McpServerStartupState::Ready,
    ] {
        app.chat_widget.handle_server_notification(
            ServerNotification::McpServerStatusUpdated(McpServerStatusUpdatedNotification {
                thread_id: Some(subagent_thread_id.to_string()),
                name: "eager".to_string(),
                status,
                error: None,
                failure_reason: None,
            }),
            /*replay_kind*/ None,
        );
    }

    assert!(!app.chat_widget.is_task_running_for_test());
}

#[tokio::test]
async fn side_conversations_wait_for_every_configured_mcp_server() {
    let mut app = make_test_app().await;
    configure_mcp_servers(&mut app);
    let root_thread_id = ThreadId::new();
    let side_thread_id = ThreadId::new();
    app.primary_thread_id = Some(root_thread_id);
    app.side_threads
        .insert(side_thread_id, SideThreadState::new(root_thread_id));
    app.active_thread_id = Some(side_thread_id);
    app.refresh_mcp_startup_expected_servers_from_config();

    for status in [
        McpServerStartupState::Starting,
        McpServerStartupState::Ready,
    ] {
        app.chat_widget.handle_server_notification(
            ServerNotification::McpServerStatusUpdated(McpServerStatusUpdatedNotification {
                thread_id: Some(side_thread_id.to_string()),
                name: "eager".to_string(),
                status,
                error: None,
                failure_reason: None,
            }),
            /*replay_kind*/ None,
        );
    }

    assert!(app.chat_widget.is_task_running_for_test());
}
