use super::build_provenance;
use ratatui::style::Stylize;
use ratatui::text::Line;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum GoalStatusIndicator {
    Active { usage: Option<String> },
    Paused,
    Blocked,
    UsageLimited,
    BudgetLimited { usage: Option<String> },
    Complete { usage: Option<String> },
}

pub(crate) fn line(indicator: Option<&GoalStatusIndicator>) -> Option<Line<'static>> {
    indicator.map(|indicator| build_provenance::line(label_line(indicator)))
}

pub(super) fn line_fitting_width(
    indicator: Option<&GoalStatusIndicator>,
    max_width: usize,
) -> Option<Line<'static>> {
    build_provenance::line_fitting_width(label_line(indicator?), max_width)
}

fn label_line(indicator: &GoalStatusIndicator) -> Line<'static> {
    let label = match indicator {
        GoalStatusIndicator::Active { usage } => usage.as_ref().map_or_else(
            || "Pursuing goal".to_string(),
            |usage| format!("Pursuing goal ({usage})"),
        ),
        GoalStatusIndicator::Paused => "Goal paused (/goal resume)".to_string(),
        GoalStatusIndicator::Blocked => "Goal blocked (/goal resume)".to_string(),
        GoalStatusIndicator::UsageLimited => "Goal hit usage limits (/goal resume)".to_string(),
        GoalStatusIndicator::BudgetLimited { usage } => usage.as_ref().map_or_else(
            || "Goal abandoned".to_string(),
            |usage| format!("Goal unmet ({usage})"),
        ),
        GoalStatusIndicator::Complete { usage } => usage.as_ref().map_or_else(
            || "Goal achieved".to_string(),
            |usage| format!("Goal achieved ({usage})"),
        ),
    };
    Line::from(vec![label.magenta()])
}

#[cfg(test)]
#[path = "goal_status_indicator_tests.rs"]
mod tests;
