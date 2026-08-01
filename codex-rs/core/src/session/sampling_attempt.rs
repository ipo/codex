use codex_api::TerminalOutcome;
use codex_protocol::error::CodexErr;
use codex_protocol::items::TurnItem;
use codex_protocol::models::ResponseItem;

#[derive(Debug)]
pub(super) struct PendingOutputItem {
    pub item: ResponseItem,
    pub previously_streamed_item: Option<TurnItem>,
}

#[derive(Default)]
pub(super) struct SamplingAttempt {
    pending_items: Vec<PendingOutputItem>,
    preempt_for_mailbox_mail: bool,
}

#[derive(Debug)]
pub(super) struct CommittedAttempt {
    pub pending_items: Vec<PendingOutputItem>,
    pub needs_follow_up: bool,
    pub preempt_for_mailbox_mail: bool,
}

impl SamplingAttempt {
    pub fn push(
        &mut self,
        item: ResponseItem,
        previously_streamed_item: Option<TurnItem>,
        preempt_for_mailbox_mail: bool,
    ) {
        self.pending_items.push(PendingOutputItem {
            item,
            previously_streamed_item,
        });
        self.preempt_for_mailbox_mail |= preempt_for_mailbox_mail;
    }

    pub fn finish(self, outcome: TerminalOutcome) -> Result<CommittedAttempt, CodexErr> {
        let needs_follow_up = match outcome {
            TerminalOutcome::Completed => false,
            TerminalOutcome::ToolsReady | TerminalOutcome::Continue => true,
            TerminalOutcome::OutputExhausted => {
                return Err(CodexErr::InvalidRequest(
                    "model output limit reached before the turn completed".to_string(),
                ));
            }
            TerminalOutcome::Refusal => {
                return Err(CodexErr::InvalidRequest(
                    "model refused to complete the turn for safety reasons".to_string(),
                ));
            }
        };
        Ok(CommittedAttempt {
            pending_items: self.pending_items,
            needs_follow_up,
            preempt_for_mailbox_mail: self.preempt_for_mailbox_mail,
        })
    }
}

#[cfg(test)]
#[path = "sampling_attempt_tests.rs"]
mod tests;
