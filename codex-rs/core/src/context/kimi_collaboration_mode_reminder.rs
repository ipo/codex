use codex_protocol::protocol::COLLABORATION_MODE_CLOSE_TAG;
use codex_protocol::protocol::COLLABORATION_MODE_OPEN_TAG;

use super::ContextualUserFragment;

const SYSTEM_REMINDER_OPEN_TAG: &str = "<system-reminder>";
const SYSTEM_REMINDER_CLOSE_TAG: &str = "</system-reminder>";

/// A native Kimi reminder that preserves the authority of a collaboration-mode transition.
#[derive(Debug, Clone)]
pub(crate) struct KimiCollaborationModeReminder {
    collaboration_mode: String,
}

impl KimiCollaborationModeReminder {
    pub(crate) fn new(collaboration_mode: String) -> Self {
        Self { collaboration_mode }
    }
}

impl ContextualUserFragment for KimiCollaborationModeReminder {
    fn role(&self) -> &'static str {
        "user"
    }

    fn markers(&self) -> (&'static str, &'static str) {
        Self::type_markers()
    }

    fn type_markers() -> (&'static str, &'static str) {
        (SYSTEM_REMINDER_OPEN_TAG, SYSTEM_REMINDER_CLOSE_TAG)
    }

    fn matches_text(text: &str) -> bool {
        let Some(body) = text
            .trim()
            .strip_prefix(SYSTEM_REMINDER_OPEN_TAG)
            .and_then(|body| body.strip_suffix(SYSTEM_REMINDER_CLOSE_TAG))
        else {
            return false;
        };

        body.starts_with(COLLABORATION_MODE_OPEN_TAG)
            && body.ends_with(COLLABORATION_MODE_CLOSE_TAG)
    }

    fn body(&self) -> String {
        self.collaboration_mode.clone()
    }
}
