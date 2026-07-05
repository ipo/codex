//! App-level wiring for the focused `/sh` terminal overlay.

use super::*;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use codex_app_server_protocol::ProcessExitedNotification;
use codex_app_server_protocol::ProcessOutputDeltaNotification;
use codex_app_server_protocol::ProcessTerminalSize;
use codex_app_server_protocol::ThreadTerminalInfo;
use codex_app_server_protocol::ThreadTerminalOpenResponse;
use codex_app_server_protocol::ThreadTerminalResizeResponse;
use codex_app_server_protocol::ThreadTerminalStatus;
use codex_app_server_protocol::ThreadTerminalStatusKind;
use std::collections::hash_map::Entry;

use crate::user_terminal::TerminalContentSize;
use crate::user_terminal::TerminalFrameAction;
use crate::user_terminal::TerminalMetadata;
use crate::user_terminal::TerminalRenderOutcome;
use crate::user_terminal::UserTerminalSurface;

/// Thread-scoped terminal RPCs used by the `/sh` TUI surface.
///
/// Production calls go through [`AppServerSession`]. Tests use a local recorder so app-level input,
/// resize, and open routing can be validated without requiring a live active model turn.
pub(super) trait UserTerminalRpc {
    fn open_terminal(
        &mut self,
        thread_id: ThreadId,
        label: String,
        size: Option<ProcessTerminalSize>,
    ) -> impl std::future::Future<Output = Result<ThreadTerminalOpenResponse>> + Send;

    fn write_terminal(
        &mut self,
        thread_id: ThreadId,
        process_id: String,
        input: &[u8],
    ) -> impl std::future::Future<Output = Result<()>> + Send;

    fn resize_terminal(
        &mut self,
        thread_id: ThreadId,
        process_id: String,
        size: ProcessTerminalSize,
    ) -> impl std::future::Future<Output = Result<ThreadTerminalResizeResponse>> + Send;
}

impl UserTerminalRpc for AppServerSession {
    async fn open_terminal(
        &mut self,
        thread_id: ThreadId,
        label: String,
        size: Option<ProcessTerminalSize>,
    ) -> Result<ThreadTerminalOpenResponse> {
        AppServerSession::thread_terminal_open(self, thread_id, label, size).await
    }

    async fn write_terminal(
        &mut self,
        thread_id: ThreadId,
        process_id: String,
        input: &[u8],
    ) -> Result<()> {
        AppServerSession::thread_terminal_write(self, thread_id, process_id, input).await
    }

    async fn resize_terminal(
        &mut self,
        thread_id: ThreadId,
        process_id: String,
        size: ProcessTerminalSize,
    ) -> Result<ThreadTerminalResizeResponse> {
        AppServerSession::thread_terminal_resize(self, thread_id, process_id, size).await
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct UserTerminalKey {
    thread_id: ThreadId,
    label: String,
}

impl UserTerminalKey {
    fn new(thread_id: ThreadId, label: String) -> Self {
        Self { thread_id, label }
    }
}

pub(super) struct UserTerminalSession {
    surface: UserTerminalSurface,
    process_id: Option<String>,
    last_content_size: Option<TerminalContentSize>,
}

#[derive(Default)]
pub(super) struct UserTerminalState {
    sessions: HashMap<UserTerminalKey, UserTerminalSession>,
    process_keys: HashMap<String, UserTerminalKey>,
    active_key: Option<UserTerminalKey>,
}

impl UserTerminalState {
    pub(super) fn has_active(&self) -> bool {
        self.active_key.is_some()
    }

    fn active_key(&self) -> Option<UserTerminalKey> {
        self.active_key.clone()
    }

    fn active_session_mut(&mut self) -> Option<&mut UserTerminalSession> {
        let key = self.active_key.as_ref()?;
        self.sessions.get_mut(key)
    }

    fn ensure_session(
        &mut self,
        key: UserTerminalKey,
        metadata: TerminalMetadata,
        open_controls: Vec<crate::key_hint::KeyBinding>,
    ) -> &mut UserTerminalSession {
        match self.sessions.entry(key) {
            Entry::Occupied(entry) => {
                let session = entry.into_mut();
                session.surface.set_open_controls(open_controls);
                session
            }
            Entry::Vacant(entry) => entry.insert(UserTerminalSession {
                surface: UserTerminalSurface::new(metadata, open_controls),
                process_id: None,
                last_content_size: None,
            }),
        }
    }

    fn attach_terminal(&mut self, key: UserTerminalKey, terminal: &ThreadTerminalInfo) {
        let old_process_id = self
            .sessions
            .get(&key)
            .and_then(|session| session.process_id.clone());
        if let Some(old_process_id) = old_process_id {
            self.process_keys.remove(&old_process_id);
        }

        let metadata = metadata_from_terminal(terminal);
        if let Some(session) = self.sessions.get_mut(&key) {
            session.surface.set_metadata(metadata);
            session.process_id = Some(terminal.process_id.clone());
        }
        self.process_keys
            .insert(terminal.process_id.clone(), key.clone());
        self.active_key = Some(key);
    }

    fn process_session_mut(&mut self, process_id: &str) -> Option<&mut UserTerminalSession> {
        let key = self.process_keys.get(process_id)?;
        self.sessions.get_mut(key)
    }

    fn dismiss_active(&mut self) {
        self.active_key = None;
    }
}

impl App {
    pub(super) async fn open_user_terminal(
        &mut self,
        tui: &mut tui::Tui,
        app_server: &mut impl UserTerminalRpc,
        raw_label: String,
    ) {
        let label = match normalize_user_terminal_label(&raw_label) {
            Ok(label) => label,
            Err(message) => {
                self.chat_widget.add_error_message(message);
                tui.frame_requester().schedule_frame();
                return;
            }
        };
        let Some(thread_id) = self.current_displayed_thread_id() else {
            self.chat_widget.add_error_message(
                "Session is still starting; try /sh again in a moment.".to_string(),
            );
            tui.frame_requester().schedule_frame();
            return;
        };

        let key = UserTerminalKey::new(thread_id, label.clone());
        let viewport = match terminal_viewport_rect(tui) {
            Ok(viewport) => viewport,
            Err(err) => {
                self.chat_widget
                    .add_error_message(format!("Failed to inspect terminal size for /sh: {err}"));
                tui.frame_requester().schedule_frame();
                return;
            }
        };
        let initial_size = {
            let session = self.user_terminals.ensure_session(
                key.clone(),
                TerminalMetadata::new(
                    label.clone(),
                    Some(self.config.cwd.as_path().display().to_string()),
                    "opening",
                ),
                self.keymap.terminal.open_controls.clone(),
            );
            terminal_content_size_to_protocol(session.surface.content_size_for_area(viewport))
        };

        match app_server
            .open_terminal(thread_id, label.clone(), Some(initial_size))
            .await
        {
            Ok(response) => {
                let content_size = protocol_size_to_terminal_content_size(initial_size);
                let session = self.user_terminals.ensure_session(
                    key.clone(),
                    metadata_from_terminal(&response.terminal),
                    self.keymap.terminal.open_controls.clone(),
                );
                session.last_content_size = Some(content_size);
                self.user_terminals.attach_terminal(key, &response.terminal);
                tui.frame_requester().schedule_frame();
            }
            Err(err) => {
                self.chat_widget
                    .add_error_message(format!("Failed to open /sh '{label}': {err}"));
                tui.frame_requester().schedule_frame();
            }
        }
    }

    pub(super) async fn handle_user_terminal_tui_event(
        &mut self,
        tui: &mut tui::Tui,
        app_server: &mut impl UserTerminalRpc,
        event: TuiEvent,
    ) -> Result<()> {
        match event {
            TuiEvent::Key(key_event) => {
                self.handle_user_terminal_key(tui, app_server, key_event)
                    .await;
            }
            TuiEvent::Paste(pasted) => {
                self.handle_user_terminal_paste(tui, app_server, pasted)
                    .await;
            }
            TuiEvent::Draw | TuiEvent::Resize => {
                if self.backtrack_render_pending {
                    self.rebuild_transcript_after_backtrack(tui)?;
                    self.backtrack_render_pending = false;
                }
                self.chat_widget.maybe_post_pending_notification(tui);
                self.chat_widget.pre_draw_tick();
                if let Some(outcome) = self.render_user_terminal_frame(tui)? {
                    self.resize_active_user_terminal_if_needed(
                        app_server,
                        outcome.content_size,
                        "resize /sh terminal",
                    )
                    .await;
                }
            }
        }
        Ok(())
    }

    async fn handle_user_terminal_key(
        &mut self,
        tui: &mut tui::Tui,
        app_server: &mut impl UserTerminalRpc,
        key_event: KeyEvent,
    ) {
        let Some(session) = self.user_terminals.active_session_mut() else {
            return;
        };
        let outcome = session.surface.handle_key_event(key_event);
        if matches!(outcome.frame_action, Some(TerminalFrameAction::Dismiss)) {
            self.user_terminals.dismiss_active();
            tui.frame_requester().schedule_frame();
            return;
        }
        if outcome.needs_resize {
            match terminal_viewport_rect(tui) {
                Ok(viewport) => {
                    let content_size = self
                        .user_terminals
                        .active_session_mut()
                        .map(|session| session.surface.content_size_for_area(viewport));
                    if let Some(content_size) = content_size {
                        self.resize_active_user_terminal_if_needed(
                            app_server,
                            content_size,
                            "resize /sh terminal",
                        )
                        .await;
                    }
                }
                Err(err) => {
                    self.chat_widget
                        .add_error_message(format!("Failed to inspect terminal size: {err}"));
                }
            }
            tui.frame_requester().schedule_frame();
        }
        if !outcome.pty_input.is_empty() {
            self.write_active_user_terminal_input(app_server, &outcome.pty_input)
                .await;
        }
    }

    async fn handle_user_terminal_paste(
        &mut self,
        tui: &mut tui::Tui,
        app_server: &mut impl UserTerminalRpc,
        pasted: String,
    ) {
        let Some(session) = self.user_terminals.active_session_mut() else {
            return;
        };
        let outcome = session.surface.handle_paste(&pasted);
        if !outcome.pty_input.is_empty() {
            self.write_active_user_terminal_input(app_server, &outcome.pty_input)
                .await;
            tui.frame_requester().schedule_frame();
        }
    }

    async fn write_active_user_terminal_input(
        &mut self,
        app_server: &mut impl UserTerminalRpc,
        input: &[u8],
    ) {
        let Some((thread_id, process_id, label)) = self.active_user_terminal_target() else {
            self.chat_widget
                .add_error_message("Cannot write to /sh: terminal process is unknown.".to_string());
            self.app_event_tx.send(AppEvent::RequestRedraw);
            return;
        };
        if let Err(err) = app_server
            .write_terminal(thread_id, process_id, input)
            .await
        {
            self.chat_widget
                .add_error_message(format!("Failed to write to /sh '{label}': {err}"));
            self.app_event_tx.send(AppEvent::RequestRedraw);
        }
    }

    async fn resize_active_user_terminal_if_needed(
        &mut self,
        app_server: &mut impl UserTerminalRpc,
        content_size: TerminalContentSize,
        action: &str,
    ) {
        let Some(active_key) = self.user_terminals.active_key() else {
            return;
        };
        let Some(session) = self.user_terminals.sessions.get_mut(&active_key) else {
            return;
        };
        if session.last_content_size == Some(content_size) {
            return;
        }
        let Some(process_id) = session.process_id.clone() else {
            return;
        };
        session.last_content_size = Some(content_size);
        match app_server
            .resize_terminal(
                active_key.thread_id,
                process_id,
                terminal_content_size_to_protocol(content_size),
            )
            .await
        {
            Ok(response) => {
                if let Some(session) = self.user_terminals.sessions.get_mut(&active_key) {
                    session
                        .surface
                        .set_metadata(metadata_from_terminal(&response.terminal));
                }
            }
            Err(err) => {
                self.chat_widget
                    .add_error_message(format!("Failed to {action} '{}': {err}", active_key.label));
                self.app_event_tx.send(AppEvent::RequestRedraw);
            }
        }
    }

    fn render_user_terminal_frame(
        &mut self,
        tui: &mut tui::Tui,
    ) -> Result<Option<TerminalRenderOutcome>> {
        let Some(active_key) = self.user_terminals.active_key() else {
            return Ok(None);
        };
        let desired_height = self.chat_widget.desired_height(tui.terminal.size()?.width);
        let mut outcome = None;
        tui.draw_with_resize_reflow(desired_height, |frame| {
            let area = frame.area();
            self.chat_widget.render(area, frame.buffer);
            if let Some(session) = self.user_terminals.sessions.get_mut(&active_key) {
                let rendered = session.surface.render(area, frame.buffer);
                if let Some(cursor) = rendered.cursor_position {
                    frame.set_cursor_position(cursor);
                }
                outcome = Some(rendered);
            } else if let Some((x, y)) = self.chat_widget.cursor_pos(area) {
                frame.set_cursor_style(self.chat_widget.cursor_style(area));
                frame.set_cursor_position((x, y));
            }
        })?;
        Ok(outcome)
    }

    pub(super) fn handle_user_terminal_process_output(
        &mut self,
        notification: &ProcessOutputDeltaNotification,
    ) -> bool {
        let Some(session) = self
            .user_terminals
            .process_session_mut(&notification.process_handle)
        else {
            return false;
        };
        match STANDARD.decode(&notification.delta_base64) {
            Ok(bytes) => {
                session.surface.process_output(&bytes);
            }
            Err(err) => {
                self.chat_widget.add_error_message(format!(
                    "Failed to decode /sh output for process '{}': {err}",
                    notification.process_handle
                ));
            }
        }
        self.app_event_tx.send(AppEvent::RequestRedraw);
        true
    }

    pub(super) fn handle_user_terminal_process_exit(
        &mut self,
        notification: &ProcessExitedNotification,
    ) -> bool {
        let Some(session) = self
            .user_terminals
            .process_session_mut(&notification.process_handle)
        else {
            return false;
        };
        if !notification.stdout.is_empty() {
            session
                .surface
                .process_output(notification.stdout.as_bytes());
        }
        if !notification.stderr.is_empty() {
            session
                .surface
                .process_output(notification.stderr.as_bytes());
        }
        let mut metadata = session.surface_metadata();
        metadata.status = format!("exited {}", notification.exit_code);
        session.surface.set_metadata(metadata);
        self.app_event_tx.send(AppEvent::RequestRedraw);
        true
    }

    fn active_user_terminal_target(&self) -> Option<(ThreadId, String, String)> {
        let key = self.user_terminals.active_key.as_ref()?;
        let session = self.user_terminals.sessions.get(key)?;
        let process_id = session.process_id.clone()?;
        Some((key.thread_id, process_id, key.label.clone()))
    }
}

impl UserTerminalSession {
    fn surface_metadata(&self) -> TerminalMetadata {
        self.surface.metadata().clone()
    }
}

fn normalize_user_terminal_label(raw_label: &str) -> std::result::Result<String, String> {
    let label = raw_label.trim();
    if label.is_empty() {
        return Ok("default".to_string());
    }
    let valid = label.len() <= 64
        && label
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'));
    if valid {
        Ok(label.to_string())
    } else {
        Err(
            "Invalid /sh label. Use 1-64 characters from A-Z, a-z, 0-9, '.', '_' or '-'."
                .to_string(),
        )
    }
}

fn metadata_from_terminal(terminal: &ThreadTerminalInfo) -> TerminalMetadata {
    TerminalMetadata::new(
        terminal.label.clone(),
        Some(terminal.cwd.as_path().display().to_string()),
        terminal_status_label(&terminal.status),
    )
}

fn terminal_status_label(status: &ThreadTerminalStatus) -> String {
    match status.kind {
        ThreadTerminalStatusKind::Running => "running".to_string(),
        ThreadTerminalStatusKind::Exited => match status.exit_code {
            Some(code) => format!("exited {code}"),
            None => "exited".to_string(),
        },
    }
}

fn terminal_viewport_rect(tui: &tui::Tui) -> Result<Rect> {
    let size = tui.terminal.size()?;
    Ok(Rect::new(0, 0, size.width, size.height))
}

fn terminal_content_size_to_protocol(size: TerminalContentSize) -> ProcessTerminalSize {
    ProcessTerminalSize {
        rows: size.rows,
        cols: size.cols,
    }
}

fn protocol_size_to_terminal_content_size(size: ProcessTerminalSize) -> TerminalContentSize {
    TerminalContentSize {
        rows: size.rows,
        cols: size.cols,
    }
}

#[cfg(test)]
#[path = "user_terminal_tests.rs"]
mod tests;
