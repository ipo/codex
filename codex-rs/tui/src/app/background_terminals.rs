//! App-level background terminal list actions.

use super::*;

impl App {
    pub(super) async fn show_background_terminals(
        &mut self,
        tui: &mut tui::Tui,
        app_server: &mut AppServerSession,
    ) {
        let Some(thread_id) = self.current_displayed_thread_id() else {
            self.chat_widget.add_error_message(
                "Session is still starting; try /ps again in a moment.".to_string(),
            );
            tui.frame_requester().schedule_frame();
            return;
        };

        match app_server.thread_background_terminals_list(thread_id).await {
            Ok(response) => self
                .chat_widget
                .add_background_terminals_output(response.data),
            Err(err) => self
                .chat_widget
                .add_error_message(format!("Failed to list background terminals: {err}")),
        }
        tui.frame_requester().schedule_frame();
    }
}
