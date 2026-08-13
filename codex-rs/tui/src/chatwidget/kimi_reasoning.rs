use std::collections::HashSet;
use std::time::Duration;
use std::time::Instant;

use codex_protocol::openai_models::ModelReasoningDisplay;

use super::*;

const FLUSH_INTERVAL: Duration = Duration::from_millis(50);

#[derive(Default)]
pub(super) struct KimiReasoningState {
    item_id: Option<String>,
    draft: String,
    last_flush: Option<Instant>,
    inserted: HashSet<String>,
}

impl ChatWidget {
    pub(super) fn kimi_reasoning_model(&self) -> bool {
        self.model_catalog
            .try_list_models()
            .ok()
            .and_then(|models| {
                models
                    .into_iter()
                    .find(|preset| preset.model == self.current_model())
            })
            .is_some_and(|preset| preset.reasoning_display == ModelReasoningDisplay::KimiRaw)
    }

    pub(super) fn kimi_reasoning_visible(&self) -> bool {
        self.kimi_reasoning_model()
            && !self.config.hide_agent_reasoning
            && !self.config.kimi_hide_agent_reasoning
    }

    pub(super) fn start_kimi_reasoning(&mut self, item_id: String) {
        if !self.kimi_reasoning_visible() {
            return;
        }
        self.discard_kimi_reasoning_draft();
        self.kimi_reasoning.item_id = Some(item_id);
        self.kimi_reasoning.last_flush = Some(Instant::now());
        self.flush_active_cell();
        self.transcript.active_cell = Some(Box::new(history_cell::KimiReasoningCell::live(
            String::new(),
            self.config.animations,
        )));
        self.bump_active_cell_revision();
    }

    pub(super) fn append_kimi_reasoning(&mut self, item_id: &str, delta: String) {
        if self.kimi_reasoning.item_id.as_deref() != Some(item_id) {
            return;
        }
        self.kimi_reasoning.draft.push_str(&delta);
        let elapsed = self
            .kimi_reasoning
            .last_flush
            .map(|last_flush| last_flush.elapsed())
            .unwrap_or(FLUSH_INTERVAL);
        if elapsed >= FLUSH_INTERVAL {
            self.publish_kimi_reasoning_draft();
        } else {
            self.frame_requester
                .schedule_frame_in(FLUSH_INTERVAL.saturating_sub(elapsed));
        }
    }

    pub(super) fn publish_kimi_reasoning_draft(&mut self) {
        if self.kimi_reasoning.item_id.is_none() {
            return;
        }
        if self
            .kimi_reasoning
            .last_flush
            .is_some_and(|last_flush| last_flush.elapsed() < FLUSH_INTERVAL)
        {
            return;
        }
        if let Some(cell) = self.transcript.active_cell.as_mut().and_then(|cell| {
            cell.as_any_mut()
                .downcast_mut::<history_cell::KimiReasoningCell>()
        }) {
            cell.update(self.kimi_reasoning.draft.clone());
            self.kimi_reasoning.last_flush = Some(Instant::now());
            self.bump_active_cell_revision();
            self.request_redraw();
        }
    }

    pub(super) fn finish_kimi_reasoning(&mut self, item_id: String, content: Vec<String>) {
        if !self.kimi_reasoning_visible() {
            if self.kimi_reasoning.item_id.as_deref() == Some(item_id.as_str()) {
                self.discard_kimi_reasoning_draft();
            }
            return;
        }
        let text = content.concat();
        if text.is_empty() {
            if self.kimi_reasoning.item_id.as_deref() == Some(item_id.as_str()) {
                self.discard_kimi_reasoning_draft();
            }
            return;
        }
        if !self.kimi_reasoning.inserted.insert(item_id.clone()) {
            if self.kimi_reasoning.item_id.as_deref() == Some(item_id.as_str()) {
                self.discard_kimi_reasoning_draft();
            }
            return;
        }
        if self.kimi_reasoning.item_id.as_deref() == Some(item_id.as_str())
            && let Some(cell) = self.transcript.active_cell.as_mut().and_then(|cell| {
                cell.as_any_mut()
                    .downcast_mut::<history_cell::KimiReasoningCell>()
            })
        {
            cell.finalize(text);
            self.kimi_reasoning.item_id = None;
            self.kimi_reasoning.draft.clear();
            self.flush_active_cell();
        } else {
            self.add_to_history(history_cell::KimiReasoningCell::completed(text));
        }
        self.request_redraw();
    }

    pub(super) fn retain_partial_kimi_reasoning(&mut self) {
        if self.kimi_reasoning.item_id.is_some() {
            if self.kimi_reasoning.draft.is_empty() {
                self.discard_kimi_reasoning_draft();
                return;
            }
            if let Some(cell) = self.transcript.active_cell.as_mut().and_then(|cell| {
                cell.as_any_mut()
                    .downcast_mut::<history_cell::KimiReasoningCell>()
            }) {
                cell.finalize(std::mem::take(&mut self.kimi_reasoning.draft));
            }
            self.kimi_reasoning.item_id = None;
            self.flush_active_cell();
        }
    }

    pub(super) fn discard_kimi_reasoning_draft(&mut self) {
        if self.kimi_reasoning.item_id.take().is_some()
            && self
                .transcript
                .active_cell
                .as_ref()
                .is_some_and(|cell| cell.as_any().is::<history_cell::KimiReasoningCell>())
        {
            self.transcript.active_cell = None;
            self.bump_active_cell_revision();
        }
        self.kimi_reasoning.draft.clear();
        self.kimi_reasoning.last_flush = None;
    }

    pub(super) fn reset_kimi_reasoning_for_thread(&mut self) {
        self.discard_kimi_reasoning_draft();
        self.kimi_reasoning.inserted.clear();
    }
}
