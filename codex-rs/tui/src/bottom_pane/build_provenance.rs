use crate::line_truncation::truncate_line_with_ellipsis_if_overflow;
#[cfg(not(test))]
use crate::version::CODEX_BUILD_BRANCH;
#[cfg(not(test))]
use crate::version::CODEX_BUILD_TIMESTAMP;
#[cfg(not(test))]
use crate::version::CODEX_CLI_VERSION;
use ratatui::style::Stylize;
use ratatui::text::Line;
use unicode_width::UnicodeWidthStr;

const BUILD_SEPARATOR: &str = " · ";
const MIN_TRUNCATED_BRANCH_WIDTH: usize = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct BuildIdentity<'a> {
    branch: &'a str,
    version: &'a str,
    timestamp: &'a str,
}

impl<'a> BuildIdentity<'a> {
    pub(super) const fn new(branch: &'a str, version: &'a str, timestamp: &'a str) -> Self {
        Self {
            branch,
            version,
            timestamp,
        }
    }
}

pub(super) fn line(label: Line<'static>) -> Line<'static> {
    line_for(label, current_build_identity())
}

pub(super) fn line_fitting_width(label: Line<'static>, max_width: usize) -> Option<Line<'static>> {
    line_fitting_width_for(label, current_build_identity(), max_width)
}

fn current_build_identity() -> BuildIdentity<'static> {
    #[cfg(test)]
    {
        BuildIdentity::new("feature/robustness", "0.144.1", "2026-07-12T13:31:30Z")
    }
    #[cfg(not(test))]
    {
        BuildIdentity::new(CODEX_BUILD_BRANCH, CODEX_CLI_VERSION, CODEX_BUILD_TIMESTAMP)
    }
}

pub(super) fn line_for(mut label: Line<'static>, identity: BuildIdentity<'_>) -> Line<'static> {
    label.push_span(BUILD_SEPARATOR.dim());
    label.push_span(format!("{}@{}", identity.branch, identity.version).dim());
    label.push_span(BUILD_SEPARATOR.dim());
    label.push_span(format!("built {}", identity.timestamp).dim());
    label
}

pub(super) fn line_fitting_width_for(
    label: Line<'static>,
    identity: BuildIdentity<'_>,
    max_width: usize,
) -> Option<Line<'static>> {
    let full = line_for(label.clone(), identity);
    if full.width() <= max_width {
        return Some(full);
    }

    let without_timestamp = branch_version_line_for(label.clone(), identity);
    if without_timestamp.width() <= max_width {
        return Some(without_timestamp);
    }

    let fixed_width = label.width()
        + UnicodeWidthStr::width(BUILD_SEPARATOR)
        + 1
        + UnicodeWidthStr::width(identity.version);
    let branch_width = max_width.saturating_sub(fixed_width);
    if branch_width >= MIN_TRUNCATED_BRANCH_WIDTH
        && UnicodeWidthStr::width(identity.branch) > branch_width
    {
        let truncated = truncated_branch_line_for(label.clone(), identity, branch_width);
        if truncated.width() <= max_width {
            return Some(truncated);
        }
    }

    let version_only = version_only_line_for(label.clone(), identity);
    if version_only.width() <= max_width {
        return Some(version_only);
    }

    (label.width() <= max_width).then_some(label)
}

fn branch_version_line_for(mut label: Line<'static>, identity: BuildIdentity<'_>) -> Line<'static> {
    label.push_span(BUILD_SEPARATOR.dim());
    label.push_span(format!("{}@{}", identity.branch, identity.version).dim());
    label
}

fn truncated_branch_line_for(
    mut label: Line<'static>,
    identity: BuildIdentity<'_>,
    max_branch_width: usize,
) -> Line<'static> {
    label.push_span(BUILD_SEPARATOR.dim());
    let truncated_branch = truncate_line_with_ellipsis_if_overflow(
        Line::from(vec![identity.branch.to_string().dim()]),
        max_branch_width,
    );
    label.spans.extend(truncated_branch.spans);
    label.push_span(format!("@{}", identity.version).dim());
    label
}

fn version_only_line_for(mut label: Line<'static>, identity: BuildIdentity<'_>) -> Line<'static> {
    label.push_span(BUILD_SEPARATOR.dim());
    label.push_span(identity.version.to_string().dim());
    label
}

#[cfg(test)]
#[path = "build_provenance_tests.rs"]
mod tests;
