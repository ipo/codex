//! Safety-access error classification and terminal turn handling.

use super::*;

const LEGACY_SAFETY_ACCESS_BLOCK_PREFIX: &str =
    "Invalid prompt: we've limited access to this content for safety reasons.";
const BIO_POLICY_SAFETY_ACCESS_BLOCK_PREFIX: &str =
    "This content was flagged for possible biological risk.";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SafetyStopSource {
    Live,
    Replay,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SafetyStopKind {
    Cyber,
    Biology,
}

impl ChatWidget {
    pub(super) fn try_handle_safety_stop(
        &mut self,
        message: &str,
        codex_error_info: Option<&AppServerCodexErrorInfo>,
        source: SafetyStopSource,
    ) -> bool {
        let kind = if codex_error_info.is_some_and(is_app_server_cyber_policy_error) {
            SafetyStopKind::Cyber
        } else if is_biology_safety_access_block(message) {
            SafetyStopKind::Biology
        } else {
            return false;
        };

        self.on_safety_stop(kind, source);
        true
    }

    fn on_safety_stop(&mut self, kind: SafetyStopKind, source: SafetyStopSource) {
        self.input_queue.submit_pending_steers_after_interrupt = false;
        self.finalize_turn();
        let cell = match kind {
            SafetyStopKind::Cyber => history_cell::new_cyber_policy_error_event(),
            SafetyStopKind::Biology => history_cell::new_safety_access_block_event(),
        };
        self.add_to_history(cell);
        self.request_redraw();

        let follow_up_started = self.maybe_send_next_queued_input();
        if source == SafetyStopSource::Live && !follow_up_started {
            self.notify(Notification::SafetyAlert);
        }
    }
}

fn is_biology_safety_access_block(message: &str) -> bool {
    if is_safety_access_block_message(message) {
        return true;
    }
    serde_json::from_str::<serde_json::Value>(message).is_ok_and(|response| {
        response["error"]["code"].as_str() == Some("bio_policy")
            || response["error"]["message"]
                .as_str()
                .is_some_and(is_safety_access_block_message)
    })
}

fn is_safety_access_block_message(message: &str) -> bool {
    message.starts_with(LEGACY_SAFETY_ACCESS_BLOCK_PREFIX)
        || message.starts_with(BIO_POLICY_SAFETY_ACCESS_BLOCK_PREFIX)
}
