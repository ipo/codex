use std::collections::HashSet;
use std::time::Duration;
use std::time::Instant;

use codex_protocol::openai_models::ModelReasoningDisplay;

use super::*;

const FLUSH_INTERVAL: Duration = Duration::from_millis(50);

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct RawReasoningOccurrence {
    turn_id: String,
    item_id: String,
    replay_ordinal: Option<usize>,
}

impl RawReasoningOccurrence {
    pub(super) fn live(turn_id: String, item_id: String) -> Self {
        Self {
            turn_id,
            item_id,
            replay_ordinal: None,
        }
    }

    pub(super) fn replay(turn_id: String, item_id: String, replay_ordinal: usize) -> Self {
        Self {
            turn_id,
            item_id,
            replay_ordinal: Some(replay_ordinal),
        }
    }
}

#[derive(Default)]
pub(super) struct RawReasoningState {
    occurrence: Option<RawReasoningOccurrence>,
    draft: String,
    last_flush: Option<Instant>,
    inserted: HashSet<RawReasoningOccurrence>,
}

impl ChatWidget {
    fn raw_reasoning_display(&self) -> Option<ModelReasoningDisplay> {
        self.model_catalog
            .try_list_models()
            .ok()
            .and_then(|models| {
                models
                    .into_iter()
                    .find(|preset| preset.model == self.current_model())
            })
            .map(|preset| preset.reasoning_display)
    }

    pub(super) fn raw_reasoning_model(&self) -> bool {
        matches!(
            self.raw_reasoning_display(),
            Some(ModelReasoningDisplay::Raw | ModelReasoningDisplay::KimiRaw)
        )
    }

    pub(super) fn raw_reasoning_visible(&self) -> bool {
        if self.config.hide_agent_reasoning {
            return false;
        }
        match self.raw_reasoning_display() {
            Some(ModelReasoningDisplay::Raw) => true,
            Some(ModelReasoningDisplay::KimiRaw) => !self.config.kimi_hide_agent_reasoning,
            Some(ModelReasoningDisplay::Summary) | None => false,
        }
    }

    pub(super) fn start_raw_reasoning(&mut self, occurrence: RawReasoningOccurrence) {
        if !self.raw_reasoning_visible() {
            return;
        }
        self.discard_raw_reasoning_draft();
        self.raw_reasoning.occurrence = Some(occurrence);
        self.raw_reasoning.last_flush = Some(Instant::now());
        self.flush_active_cell();
        self.transcript.active_cell = Some(Box::new(history_cell::RawReasoningCell::live(
            String::new(),
            self.config.animations,
        )));
        self.bump_active_cell_revision();
    }

    pub(super) fn append_raw_reasoning(&mut self, item_id: &str, delta: String) {
        if self
            .raw_reasoning
            .occurrence
            .as_ref()
            .map(|occurrence| occurrence.item_id.as_str())
            != Some(item_id)
        {
            return;
        }
        self.raw_reasoning.draft.push_str(&delta);
        let elapsed = self
            .raw_reasoning
            .last_flush
            .map(|last_flush| last_flush.elapsed())
            .unwrap_or(FLUSH_INTERVAL);
        if elapsed >= FLUSH_INTERVAL {
            self.publish_raw_reasoning_draft();
        } else {
            self.frame_requester
                .schedule_frame_in(FLUSH_INTERVAL.saturating_sub(elapsed));
        }
    }

    pub(super) fn publish_raw_reasoning_draft(&mut self) {
        if self.raw_reasoning.occurrence.is_none() {
            return;
        }
        if self
            .raw_reasoning
            .last_flush
            .is_some_and(|last_flush| last_flush.elapsed() < FLUSH_INTERVAL)
        {
            return;
        }
        if let Some(cell) = self.transcript.active_cell.as_mut().and_then(|cell| {
            cell.as_any_mut()
                .downcast_mut::<history_cell::RawReasoningCell>()
        }) {
            cell.update(self.raw_reasoning.draft.clone());
            self.raw_reasoning.last_flush = Some(Instant::now());
            self.bump_active_cell_revision();
            self.request_redraw();
        }
    }

    pub(super) fn finish_raw_reasoning(
        &mut self,
        occurrence: RawReasoningOccurrence,
        content: Vec<String>,
    ) {
        if !self.raw_reasoning_visible() {
            if self.raw_reasoning.occurrence.as_ref() == Some(&occurrence) {
                self.discard_raw_reasoning_draft();
            }
            return;
        }
        let text = content.concat();
        if text.is_empty() {
            if self.raw_reasoning.occurrence.as_ref() == Some(&occurrence) {
                self.discard_raw_reasoning_draft();
            }
            return;
        }
        if !self.raw_reasoning.inserted.insert(occurrence.clone()) {
            if self.raw_reasoning.occurrence.as_ref() == Some(&occurrence) {
                self.discard_raw_reasoning_draft();
            }
            return;
        }
        if self.raw_reasoning.occurrence.as_ref() == Some(&occurrence)
            && let Some(cell) = self.transcript.active_cell.as_mut().and_then(|cell| {
                cell.as_any_mut()
                    .downcast_mut::<history_cell::RawReasoningCell>()
            })
        {
            cell.finalize(text);
            self.raw_reasoning.occurrence = None;
            self.raw_reasoning.draft.clear();
            self.flush_active_cell();
        } else {
            self.add_to_history(history_cell::RawReasoningCell::completed(text));
        }
        self.request_redraw();
    }

    pub(super) fn retain_partial_raw_reasoning(&mut self) {
        if self.raw_reasoning.occurrence.is_some() {
            if self.raw_reasoning.draft.is_empty() {
                self.discard_raw_reasoning_draft();
                return;
            }
            if let Some(cell) = self.transcript.active_cell.as_mut().and_then(|cell| {
                cell.as_any_mut()
                    .downcast_mut::<history_cell::RawReasoningCell>()
            }) {
                cell.finalize(std::mem::take(&mut self.raw_reasoning.draft));
            }
            if let Some(occurrence) = self.raw_reasoning.occurrence.take() {
                self.raw_reasoning.inserted.insert(occurrence);
            }
            self.flush_active_cell();
        }
    }

    pub(super) fn discard_raw_reasoning_draft(&mut self) {
        if self.raw_reasoning.occurrence.take().is_some()
            && self
                .transcript
                .active_cell
                .as_ref()
                .is_some_and(|cell| cell.as_any().is::<history_cell::RawReasoningCell>())
        {
            self.transcript.active_cell = None;
            self.bump_active_cell_revision();
        }
        self.raw_reasoning.draft.clear();
        self.raw_reasoning.last_flush = None;
    }

    pub(super) fn reset_raw_reasoning_for_thread(&mut self) {
        self.discard_raw_reasoning_draft();
        self.raw_reasoning.inserted.clear();
    }
}
