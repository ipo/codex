use super::*;
use crate::user_terminal::TerminalFocusMode;
use crate::user_terminal::TerminalLayoutMode;
use assert_matches::assert_matches;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use codex_app_server_protocol::ProcessExitedNotification;
use codex_app_server_protocol::ProcessOutputStream;
use codex_app_server_protocol::ThreadTerminalOpenResponse;
use codex_app_server_protocol::ThreadTerminalResizeResponse;
use codex_app_server_protocol::ThreadTerminalSource;
use color_eyre::eyre::bail;
use pretty_assertions::assert_eq;
use pretty_assertions::assert_ne;
use ratatui::buffer::Buffer;
use std::collections::VecDeque;

#[test]
fn normalizes_empty_label_to_default() {
    assert_eq!(
        normalize_user_terminal_label("   ").expect("default label"),
        "default"
    );
}

#[test]
fn trims_and_accepts_supported_label_characters() {
    assert_eq!(
        normalize_user_terminal_label("  build.shell_1  ").expect("valid label"),
        "build.shell_1"
    );
}

#[test]
fn rejects_unsupported_label_characters() {
    assert_eq!(
        normalize_user_terminal_label("build shell").expect_err("invalid label"),
        "Invalid /sh label. Use 1-64 characters from A-Z, a-z, 0-9, '.', '_' or '-'."
    );
}

#[test]
fn rejects_overlong_labels() {
    let label = "a".repeat(65);

    assert_eq!(
        normalize_user_terminal_label(&label).expect_err("invalid label"),
        "Invalid /sh label. Use 1-64 characters from A-Z, a-z, 0-9, '.', '_' or '-'."
    );
}

#[tokio::test]
async fn opens_named_terminals_and_preserves_state_across_dismiss() -> Result<()> {
    let (mut app, mut app_event_rx, _op_rx) =
        crate::app::test_support::make_test_app_with_channels().await;
    let thread_id = ThreadId::new();
    app.active_thread_id = Some(thread_id);
    let mut tui = crate::tui::test_support::make_test_tui()?;
    let default_terminal = fake_terminal(&app, thread_id, "default", "41");
    let work_terminal = fake_terminal(&app, thread_id, "work", "42");
    let fresh_default_terminal = fake_terminal(&app, thread_id, "default", "43");
    let mut terminal_rpc = RecordingTerminalRpc::new(vec![
        default_terminal.clone(),
        work_terminal,
        default_terminal.clone(),
        default_terminal,
        fresh_default_terminal,
    ]);

    app.open_user_terminal(&mut tui, &mut terminal_rpc, String::new())
        .await;
    let default_key = app
        .user_terminals
        .active_key()
        .expect("default terminal should be active");
    assert_eq!(default_key.thread_id, thread_id);
    assert_eq!(default_key.label, "default");
    let default_process_id = active_process_id(&app);
    assert_eq!(
        active_content_size(&app),
        Some(TerminalContentSize { rows: 10, cols: 80 })
    );
    assert_eq!(
        terminal_rpc.open_calls,
        vec![OpenCall {
            thread_id,
            label: "default".to_string(),
            size: Some(ProcessTerminalSize { rows: 10, cols: 80 }),
        }]
    );

    app.resize_active_user_terminal_if_needed(
        &mut terminal_rpc,
        TerminalContentSize { rows: 10, cols: 80 },
        "test duplicate resize",
    )
    .await;
    assert_eq!(
        terminal_rpc.resize_calls,
        Vec::<ResizeCall>::new(),
        "same-size host resize should not issue a duplicate resize RPC"
    );

    app.open_user_terminal(&mut tui, &mut terminal_rpc, "work".to_string())
        .await;
    let work_key = app
        .user_terminals
        .active_key()
        .expect("work terminal should be active");
    assert_eq!(work_key.thread_id, thread_id);
    assert_eq!(work_key.label, "work");
    assert_ne!(active_process_id(&app), default_process_id);

    app.open_user_terminal(&mut tui, &mut terminal_rpc, "default".to_string())
        .await;
    assert_eq!(active_process_id(&app), default_process_id);

    app.handle_user_terminal_key(
        &mut tui,
        &mut terminal_rpc,
        KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE),
    )
    .await;
    assert_eq!(
        terminal_rpc.write_calls,
        vec![WriteCall {
            thread_id,
            process_id: default_process_id.clone(),
            input: b"p".to_vec(),
        }]
    );

    app.handle_user_terminal_key(
        &mut tui,
        &mut terminal_rpc,
        KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL),
    )
    .await;
    assert_eq!(
        active_focus_mode(&app),
        Some(TerminalFocusMode::FrameControls)
    );
    assert_eq!(
        terminal_rpc.write_calls.len(),
        1,
        "open-controls key must not be sent to the terminal process"
    );

    app.handle_user_terminal_key(
        &mut tui,
        &mut terminal_rpc,
        KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE),
    )
    .await;
    assert_eq!(
        active_layout_mode(&app),
        Some(TerminalLayoutMode::Fullscreen)
    );
    assert_eq!(
        active_content_size(&app),
        Some(TerminalContentSize { rows: 22, cols: 80 })
    );

    app.handle_user_terminal_key(
        &mut tui,
        &mut terminal_rpc,
        KeyEvent::new(KeyCode::Char('h'), KeyModifiers::NONE),
    )
    .await;
    assert_eq!(active_layout_mode(&app), Some(TerminalLayoutMode::Half));
    assert_eq!(
        active_content_size(&app),
        Some(TerminalContentSize { rows: 10, cols: 80 })
    );

    app.handle_user_terminal_key(
        &mut tui,
        &mut terminal_rpc,
        KeyEvent::new(KeyCode::Char('+'), KeyModifiers::NONE),
    )
    .await;
    assert_eq!(
        active_layout_mode(&app),
        Some(TerminalLayoutMode::CustomHeight(14))
    );
    assert_eq!(
        active_content_size(&app),
        Some(TerminalContentSize { rows: 12, cols: 80 })
    );

    app.handle_user_terminal_key(
        &mut tui,
        &mut terminal_rpc,
        KeyEvent::new(KeyCode::Char('-'), KeyModifiers::NONE),
    )
    .await;
    assert_eq!(
        active_layout_mode(&app),
        Some(TerminalLayoutMode::CustomHeight(12))
    );
    assert_eq!(
        active_content_size(&app),
        Some(TerminalContentSize { rows: 10, cols: 80 })
    );
    assert_eq!(
        terminal_rpc.resize_calls,
        vec![
            ResizeCall {
                thread_id,
                process_id: default_process_id.clone(),
                size: ProcessTerminalSize { rows: 22, cols: 80 },
            },
            ResizeCall {
                thread_id,
                process_id: default_process_id.clone(),
                size: ProcessTerminalSize { rows: 10, cols: 80 },
            },
            ResizeCall {
                thread_id,
                process_id: default_process_id.clone(),
                size: ProcessTerminalSize { rows: 12, cols: 80 },
            },
            ResizeCall {
                thread_id,
                process_id: default_process_id.clone(),
                size: ProcessTerminalSize { rows: 10, cols: 80 },
            },
        ]
    );

    while app_event_rx.try_recv().is_ok() {}
    let marker = "codex-tui-user-terminal-marker";
    assert!(
        app.handle_user_terminal_process_output(&ProcessOutputDeltaNotification {
            process_handle: default_process_id.clone(),
            stream: ProcessOutputStream::Stdout,
            delta_base64: STANDARD.encode(format!("{marker}\r\n")),
            cap_reached: false,
        })
    );
    assert!(render_active_terminal(&mut app).contains(marker));
    assert_matches!(app_event_rx.try_recv(), Ok(AppEvent::RequestRedraw));

    app.handle_user_terminal_key(
        &mut tui,
        &mut terminal_rpc,
        KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE),
    )
    .await;
    assert_eq!(app.user_terminals.active_key(), None);
    assert!(
        app.user_terminals
            .process_keys
            .contains_key(&default_process_id)
    );

    app.open_user_terminal(&mut tui, &mut terminal_rpc, "default".to_string())
        .await;
    assert_eq!(active_process_id(&app), default_process_id);
    assert_eq!(
        active_layout_mode(&app),
        Some(TerminalLayoutMode::CustomHeight(12))
    );
    assert!(render_active_terminal(&mut app).contains(marker));

    assert!(
        app.handle_user_terminal_process_exit(&ProcessExitedNotification {
            process_handle: default_process_id.clone(),
            exit_code: 0,
            stdout: "old process exit\r\n".to_string(),
            stdout_cap_reached: false,
            stderr: String::new(),
            stderr_cap_reached: false,
        })
    );
    assert!(render_active_terminal(&mut app).contains("old process exit"));

    app.open_user_terminal(&mut tui, &mut terminal_rpc, "default".to_string())
        .await;
    let fresh_process_id = active_process_id(&app);
    assert_ne!(fresh_process_id, default_process_id);
    assert_eq!(
        active_layout_mode(&app),
        Some(TerminalLayoutMode::CustomHeight(12))
    );
    let rendered = render_active_terminal(&mut app);
    assert!(!rendered.contains(marker));
    assert!(!rendered.contains("old process exit"));

    Ok(())
}

#[tokio::test]
async fn paste_and_frame_control_escape_stay_terminal_local() -> Result<()> {
    let (mut app, _app_event_rx, _op_rx) =
        crate::app::test_support::make_test_app_with_channels().await;
    let thread_id = ThreadId::new();
    app.active_thread_id = Some(thread_id);
    let mut tui = crate::tui::test_support::make_test_tui()?;
    let terminal = fake_terminal(&app, thread_id, "default", "41");
    let mut terminal_rpc = RecordingTerminalRpc::new(vec![terminal]);

    app.open_user_terminal(&mut tui, &mut terminal_rpc, String::new())
        .await;
    let process_id = active_process_id(&app);

    let paste = "printf one\\nprintf two\\n".to_string();
    app.handle_user_terminal_tui_event(&mut tui, &mut terminal_rpc, TuiEvent::Paste(paste.clone()))
        .await?;
    assert_eq!(
        terminal_rpc.write_calls,
        vec![WriteCall {
            thread_id,
            process_id: process_id.clone(),
            input: paste.into_bytes(),
        }]
    );

    app.handle_user_terminal_key(
        &mut tui,
        &mut terminal_rpc,
        KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL),
    )
    .await;
    assert_eq!(
        active_focus_mode(&app),
        Some(TerminalFocusMode::FrameControls)
    );
    assert_eq!(
        terminal_rpc.write_calls.len(),
        1,
        "open-controls key must not be sent to the terminal process"
    );

    app.handle_user_terminal_key(
        &mut tui,
        &mut terminal_rpc,
        KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE),
    )
    .await;
    assert_eq!(active_focus_mode(&app), Some(TerminalFocusMode::Input));

    app.handle_user_terminal_key(
        &mut tui,
        &mut terminal_rpc,
        KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE),
    )
    .await;
    assert_eq!(
        terminal_rpc.write_calls,
        vec![
            WriteCall {
                thread_id,
                process_id: process_id.clone(),
                input: b"printf one\\nprintf two\\n".to_vec(),
            },
            WriteCall {
                thread_id,
                process_id,
                input: b"x".to_vec(),
            },
        ]
    );
    assert!(app.user_terminals.active_key().is_some());

    Ok(())
}

#[tokio::test]
async fn invalid_label_reports_error_without_opening_terminal() -> Result<()> {
    let (mut app, mut app_event_rx, _op_rx) =
        crate::app::test_support::make_test_app_with_channels().await;
    app.active_thread_id = Some(ThreadId::new());
    let mut tui = crate::tui::test_support::make_test_tui()?;
    let mut terminal_rpc = RecordingTerminalRpc::new(Vec::new());

    app.open_user_terminal(&mut tui, &mut terminal_rpc, "bad label".to_string())
        .await;

    assert!(terminal_rpc.open_calls.is_empty());
    assert_eq!(app.user_terminals.active_key(), None);
    assert!(
        drain_history_text(&mut app_event_rx).contains(
            "Invalid /sh label. Use 1-64 characters from A-Z, a-z, 0-9, '.', '_' or '-'."
        )
    );
    Ok(())
}

#[tokio::test]
async fn terminal_rpc_failures_are_reported() -> Result<()> {
    let (mut app, mut app_event_rx, _op_rx) =
        crate::app::test_support::make_test_app_with_channels().await;
    let thread_id = ThreadId::new();
    app.active_thread_id = Some(thread_id);
    let mut tui = crate::tui::test_support::make_test_tui()?;

    let mut open_failure_rpc = RecordingTerminalRpc::new(Vec::new());
    open_failure_rpc.open_error = Some("open boom".to_string());
    app.open_user_terminal(&mut tui, &mut open_failure_rpc, String::new())
        .await;
    assert_eq!(app.user_terminals.active_key(), None);
    assert!(
        drain_history_text(&mut app_event_rx).contains("Failed to open /sh 'default': open boom")
    );

    let terminal = fake_terminal(&app, thread_id, "default", "41");
    let mut write_failure_rpc = RecordingTerminalRpc::new(vec![terminal]);
    write_failure_rpc.write_error = Some("write boom".to_string());
    app.open_user_terminal(&mut tui, &mut write_failure_rpc, String::new())
        .await;
    while app_event_rx.try_recv().is_ok() {}

    let process_id = active_process_id(&app);
    app.handle_user_terminal_key(
        &mut tui,
        &mut write_failure_rpc,
        KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE),
    )
    .await;
    let drained = drain_app_events(&mut app_event_rx);
    assert_eq!(
        write_failure_rpc.write_calls,
        vec![WriteCall {
            thread_id,
            process_id: process_id.clone(),
            input: b"p".to_vec(),
        }]
    );
    assert!(
        drained
            .history_text
            .contains("Failed to write to /sh 'default': write boom")
    );
    assert_eq!(drained.request_redraws, 1);

    let mut resize_failure_rpc = RecordingTerminalRpc::new(Vec::new());
    resize_failure_rpc.resize_error = Some("resize boom".to_string());
    app.resize_active_user_terminal_if_needed(
        &mut resize_failure_rpc,
        TerminalContentSize { rows: 11, cols: 80 },
        "resize /sh terminal",
    )
    .await;
    let drained = drain_app_events(&mut app_event_rx);
    assert_eq!(
        resize_failure_rpc.resize_calls,
        vec![ResizeCall {
            thread_id,
            process_id,
            size: ProcessTerminalSize { rows: 11, cols: 80 },
        }]
    );
    assert!(
        drained
            .history_text
            .contains("Failed to resize /sh terminal 'default': resize boom")
    );
    assert_eq!(drained.request_redraws, 1);

    Ok(())
}

fn active_process_id(app: &App) -> String {
    let key = app.user_terminals.active_key().expect("active terminal");
    app.user_terminals
        .sessions
        .get(&key)
        .and_then(|session| session.process_id.clone())
        .expect("active process id")
}

fn active_content_size(app: &App) -> Option<TerminalContentSize> {
    let key = app.user_terminals.active_key()?;
    app.user_terminals
        .sessions
        .get(&key)
        .and_then(|session| session.last_content_size)
}

fn active_focus_mode(app: &App) -> Option<TerminalFocusMode> {
    let key = app.user_terminals.active_key()?;
    app.user_terminals
        .sessions
        .get(&key)
        .map(|session| session.surface.focus_mode())
}

fn active_layout_mode(app: &App) -> Option<TerminalLayoutMode> {
    let key = app.user_terminals.active_key()?;
    app.user_terminals
        .sessions
        .get(&key)
        .map(|session| session.surface.layout_mode())
}

fn render_active_terminal(app: &mut App) -> String {
    let key = app.user_terminals.active_key().expect("active terminal");
    let session = app
        .user_terminals
        .sessions
        .get_mut(&key)
        .expect("active terminal session");
    let mut buffer = Buffer::empty(Rect::new(0, 0, 80, 24));
    session.surface.render(Rect::new(0, 0, 80, 24), &mut buffer);
    let area = buffer.area();
    (0..area.height)
        .map(|row| {
            let mut line = String::new();
            for col in 0..area.width {
                line.push_str(buffer[(col, row)].symbol());
            }
            line.trim_end().to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn fake_terminal(
    app: &App,
    thread_id: ThreadId,
    label: &str,
    process_id: &str,
) -> ThreadTerminalInfo {
    ThreadTerminalInfo {
        thread_id: thread_id.to_string(),
        label: label.to_string(),
        process_id: process_id.to_string(),
        command: "test-shell".to_string(),
        cwd: app.config.cwd.clone(),
        source: ThreadTerminalSource::SharedTerminal,
        tty: true,
        size: None,
        status: ThreadTerminalStatus {
            kind: ThreadTerminalStatusKind::Running,
            exit_code: None,
        },
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct OpenCall {
    thread_id: ThreadId,
    label: String,
    size: Option<ProcessTerminalSize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct WriteCall {
    thread_id: ThreadId,
    process_id: String,
    input: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ResizeCall {
    thread_id: ThreadId,
    process_id: String,
    size: ProcessTerminalSize,
}

struct RecordingTerminalRpc {
    open_responses: VecDeque<ThreadTerminalInfo>,
    open_calls: Vec<OpenCall>,
    write_calls: Vec<WriteCall>,
    resize_calls: Vec<ResizeCall>,
    open_error: Option<String>,
    write_error: Option<String>,
    resize_error: Option<String>,
}

impl RecordingTerminalRpc {
    fn new(open_responses: Vec<ThreadTerminalInfo>) -> Self {
        Self {
            open_responses: open_responses.into(),
            open_calls: Vec::new(),
            write_calls: Vec::new(),
            resize_calls: Vec::new(),
            open_error: None,
            write_error: None,
            resize_error: None,
        }
    }
}

impl UserTerminalRpc for RecordingTerminalRpc {
    async fn open_terminal(
        &mut self,
        thread_id: ThreadId,
        label: String,
        size: Option<ProcessTerminalSize>,
    ) -> Result<ThreadTerminalOpenResponse> {
        self.open_calls.push(OpenCall {
            thread_id,
            label,
            size,
        });
        if let Some(error) = self.open_error.take() {
            bail!("{error}");
        }
        Ok(ThreadTerminalOpenResponse {
            terminal: self
                .open_responses
                .pop_front()
                .expect("test should provide an open response"),
        })
    }

    async fn write_terminal(
        &mut self,
        thread_id: ThreadId,
        process_id: String,
        input: &[u8],
    ) -> Result<()> {
        self.write_calls.push(WriteCall {
            thread_id,
            process_id,
            input: input.to_vec(),
        });
        if let Some(error) = self.write_error.take() {
            bail!("{error}");
        }
        Ok(())
    }

    async fn resize_terminal(
        &mut self,
        thread_id: ThreadId,
        process_id: String,
        size: ProcessTerminalSize,
    ) -> Result<ThreadTerminalResizeResponse> {
        self.resize_calls.push(ResizeCall {
            thread_id,
            process_id: process_id.clone(),
            size,
        });
        if let Some(error) = self.resize_error.take() {
            bail!("{error}");
        }
        Ok(ThreadTerminalResizeResponse {
            terminal: ThreadTerminalInfo {
                thread_id: thread_id.to_string(),
                label: "default".to_string(),
                process_id,
                command: "test-shell".to_string(),
                cwd: crate::test_support::test_path_buf("/tmp/project").abs(),
                source: ThreadTerminalSource::SharedTerminal,
                tty: true,
                size: Some(size),
                status: ThreadTerminalStatus {
                    kind: ThreadTerminalStatusKind::Running,
                    exit_code: None,
                },
            },
        })
    }
}

struct DrainedAppEvents {
    history_text: String,
    request_redraws: usize,
}

fn drain_app_events(
    app_event_rx: &mut tokio::sync::mpsc::UnboundedReceiver<AppEvent>,
) -> DrainedAppEvents {
    let mut rendered = Vec::new();
    let mut request_redraws = 0;
    while let Ok(event) = app_event_rx.try_recv() {
        match event {
            AppEvent::InsertHistoryCell(cell) => {
                rendered.push(
                    cell.display_lines(/*width*/ 80)
                        .into_iter()
                        .map(|line| line.to_string())
                        .collect::<Vec<_>>()
                        .join("\n"),
                );
            }
            AppEvent::RequestRedraw => {
                request_redraws += 1;
            }
            _ => {}
        }
    }
    DrainedAppEvents {
        history_text: rendered.join("\n"),
        request_redraws,
    }
}

fn drain_history_text(app_event_rx: &mut tokio::sync::mpsc::UnboundedReceiver<AppEvent>) -> String {
    drain_app_events(app_event_rx).history_text
}
