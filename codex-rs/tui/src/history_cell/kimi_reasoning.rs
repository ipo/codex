use std::time::Instant;

use ratatui::style::Stylize;
use ratatui::text::Line;
use textwrap::Options;

use super::HistoryCell;
use super::plain_lines;
use crate::motion::MotionMode;
use crate::motion::ReducedMotionIndicator;
use crate::motion::activity_indicator;

const PREVIEW_LINES: usize = 2;

#[derive(Debug)]
pub(crate) struct KimiReasoningCell {
    text: String,
    live: bool,
    motion_mode: MotionMode,
    started_at: Instant,
}

impl KimiReasoningCell {
    pub(crate) fn live(text: String, animations_enabled: bool) -> Self {
        Self {
            text,
            live: true,
            motion_mode: MotionMode::from_animations_enabled(animations_enabled),
            started_at: Instant::now(),
        }
    }

    pub(crate) fn completed(text: String) -> Self {
        Self {
            text,
            live: false,
            motion_mode: MotionMode::Reduced,
            started_at: Instant::now(),
        }
    }

    pub(crate) fn update(&mut self, text: String) {
        self.text = text;
    }

    pub(crate) fn finalize(&mut self, text: String) {
        self.text = text;
        self.live = false;
    }

    fn final_lines(&self, width: u16) -> Vec<Line<'static>> {
        let width = usize::from(width.saturating_sub(2)).max(1);
        textwrap::wrap(&self.text, Options::new(width))
            .into_iter()
            .enumerate()
            .map(|(index, text)| {
                let prefix = if index == 0 { "• " } else { "  " };
                Line::from(vec![
                    prefix.dim().italic(),
                    text.into_owned().dim().italic(),
                ])
            })
            .collect()
    }

    fn live_lines(&self, width: u16, transcript: bool) -> Vec<Line<'static>> {
        let width = usize::from(width.saturating_sub(2)).max(1);
        let wrapped = textwrap::wrap(&self.text, Options::new(width));
        let preview = if transcript {
            wrapped.as_slice()
        } else {
            &wrapped[wrapped.len().saturating_sub(PREVIEW_LINES)..]
        };
        let indicator = activity_indicator(
            Some(self.started_at),
            self.motion_mode,
            ReducedMotionIndicator::StaticBullet,
        )
        .unwrap_or_else(|| "•".dim());
        std::iter::once(Line::from(vec![indicator, " thinking…".dim()]))
            .chain(
                preview
                    .iter()
                    .map(|text| Line::from(vec!["  ".into(), text.to_string().dim().italic()])),
            )
            .collect()
    }
}

impl HistoryCell for KimiReasoningCell {
    fn display_lines(&self, width: u16) -> Vec<Line<'static>> {
        if self.live {
            self.live_lines(width, /*transcript*/ false)
        } else {
            self.final_lines(width)
        }
    }

    fn transcript_lines(&self, width: u16) -> Vec<Line<'static>> {
        if self.live {
            self.live_lines(width, /*transcript*/ true)
        } else {
            self.final_lines(width)
        }
    }

    fn raw_lines(&self) -> Vec<Line<'static>> {
        if self.live {
            plain_lines(self.live_lines(u16::MAX, /*transcript*/ true))
        } else {
            plain_lines(self.final_lines(u16::MAX))
        }
    }

    fn transcript_animation_tick(&self) -> Option<u64> {
        self.live
            .then(|| self.started_at.elapsed().as_millis() as u64 / 600)
    }
}
