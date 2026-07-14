use super::build_provenance;
use ratatui::style::Stylize;
use ratatui::text::Line;
use ratatui::text::Span;

pub(super) const MODE_CYCLE_HINT: &str = "shift+tab to cycle";

const MODE_LABEL: &str = "Plan mode";

pub(super) fn line(show_cycle_hint: bool) -> Line<'static> {
    build_provenance::line(mode_line(show_cycle_hint))
}

pub(super) fn line_fitting_width(show_cycle_hint: bool, max_width: usize) -> Option<Line<'static>> {
    if show_cycle_hint {
        let full = line(/*show_cycle_hint*/ true);
        if full.width() <= max_width {
            return Some(full);
        }
    }

    build_provenance::line_fitting_width(mode_only_line(), max_width)
}

fn mode_line(show_cycle_hint: bool) -> Line<'static> {
    let label = if show_cycle_hint {
        format!("{MODE_LABEL} ({MODE_CYCLE_HINT})")
    } else {
        MODE_LABEL.to_string()
    };
    Line::from(vec![label.magenta()])
}

fn mode_only_line() -> Line<'static> {
    Line::from(vec![Span::from(MODE_LABEL).magenta()])
}

#[cfg(test)]
#[path = "plan_mode_indicator_tests.rs"]
mod tests;
