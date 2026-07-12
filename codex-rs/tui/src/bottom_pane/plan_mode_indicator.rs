use crate::line_truncation::truncate_line_with_ellipsis_if_overflow;
#[cfg(not(test))]
use crate::version::CODEX_BUILD_BRANCH;
#[cfg(not(test))]
use crate::version::CODEX_CLI_VERSION;
use ratatui::style::Stylize;
use ratatui::text::Line;
use ratatui::text::Span;
use unicode_width::UnicodeWidthStr;

pub(super) const MODE_CYCLE_HINT: &str = "shift+tab to cycle";

const MODE_LABEL: &str = "Plan mode";
const BUILD_SEPARATOR: &str = " · ";
const MIN_TRUNCATED_BRANCH_WIDTH: usize = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BuildIdentity<'a> {
    branch: &'a str,
    version: &'a str,
}

impl<'a> BuildIdentity<'a> {
    const fn new(branch: &'a str, version: &'a str) -> Self {
        Self { branch, version }
    }
}

pub(super) fn line(show_cycle_hint: bool) -> Line<'static> {
    full_line_for(current_build_identity(), show_cycle_hint)
}

pub(super) fn line_fitting_width(show_cycle_hint: bool, max_width: usize) -> Option<Line<'static>> {
    line_fitting_width_for(current_build_identity(), show_cycle_hint, max_width)
}

fn current_build_identity() -> BuildIdentity<'static> {
    #[cfg(test)]
    {
        BuildIdentity::new("feature/robustness", "0.144.1")
    }
    #[cfg(not(test))]
    {
        BuildIdentity::new(CODEX_BUILD_BRANCH, CODEX_CLI_VERSION)
    }
}

fn line_fitting_width_for(
    identity: BuildIdentity<'_>,
    show_cycle_hint: bool,
    max_width: usize,
) -> Option<Line<'static>> {
    let full = full_line_for(identity, show_cycle_hint);
    if full.width() <= max_width {
        return Some(full);
    }

    if show_cycle_hint {
        let without_cycle_hint = full_line_for(identity, /*show_cycle_hint*/ false);
        if without_cycle_hint.width() <= max_width {
            return Some(without_cycle_hint);
        }
    }

    let fixed_width = UnicodeWidthStr::width(MODE_LABEL)
        + UnicodeWidthStr::width(BUILD_SEPARATOR)
        + 1
        + UnicodeWidthStr::width(identity.version);
    let branch_width = max_width.saturating_sub(fixed_width);
    if branch_width >= MIN_TRUNCATED_BRANCH_WIDTH
        && UnicodeWidthStr::width(identity.branch) > branch_width
    {
        let truncated = truncated_branch_line_for(identity, branch_width);
        if truncated.width() <= max_width {
            return Some(truncated);
        }
    }

    let version_only = version_only_line_for(identity);
    if version_only.width() <= max_width {
        return Some(version_only);
    }

    let mode_only = mode_only_line();
    (mode_only.width() <= max_width).then_some(mode_only)
}

fn full_line_for(identity: BuildIdentity<'_>, show_cycle_hint: bool) -> Line<'static> {
    let mut line = mode_line(show_cycle_hint);
    line.push_span(BUILD_SEPARATOR.dim());
    line.push_span(format!("{}@{}", identity.branch, identity.version).dim());
    line
}

fn truncated_branch_line_for(
    identity: BuildIdentity<'_>,
    max_branch_width: usize,
) -> Line<'static> {
    let mut line = mode_only_line();
    line.push_span(BUILD_SEPARATOR.dim());
    let truncated_branch = truncate_line_with_ellipsis_if_overflow(
        Line::from(vec![identity.branch.to_string().dim()]),
        max_branch_width,
    );
    for span in truncated_branch.spans {
        line.push_span(span);
    }
    line.push_span(format!("@{}", identity.version).dim());
    line
}

fn version_only_line_for(identity: BuildIdentity<'_>) -> Line<'static> {
    let mut line = mode_only_line();
    line.push_span(BUILD_SEPARATOR.dim());
    line.push_span(identity.version.to_string().dim());
    line
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
