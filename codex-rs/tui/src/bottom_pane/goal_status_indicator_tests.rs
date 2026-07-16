use super::*;
use pretty_assertions::assert_eq;

#[test]
fn every_goal_variant_keeps_its_label_and_gains_provenance() {
    let cases = [
        (
            GoalStatusIndicator::Active {
                usage: Some("4K / 5K".to_string()),
            },
            "Pursuing goal (4K / 5K)",
        ),
        (GoalStatusIndicator::Active { usage: None }, "Pursuing goal"),
        (GoalStatusIndicator::Paused, "Goal paused (/goal resume)"),
        (GoalStatusIndicator::Blocked, "Goal blocked (/goal resume)"),
        (
            GoalStatusIndicator::UsageLimited,
            "Goal hit usage limits (/goal resume)",
        ),
        (
            GoalStatusIndicator::BudgetLimited {
                usage: Some("4K / 5K tokens".to_string()),
            },
            "Goal unmet (4K / 5K tokens)",
        ),
        (
            GoalStatusIndicator::BudgetLimited { usage: None },
            "Goal abandoned",
        ),
        (
            GoalStatusIndicator::Complete {
                usage: Some("10h 12m".to_string()),
            },
            "Goal achieved (10h 12m)",
        ),
        (
            GoalStatusIndicator::Complete { usage: None },
            "Goal achieved",
        ),
    ];

    for (indicator, label) in cases {
        assert_eq!(
            line(Some(&indicator)),
            Some(Line::from(vec![
                label.to_string().magenta(),
                " · ".dim(),
                "feature/robustness@0.144.1".dim(),
                " · ".dim(),
                "built 2026-07-12T13:31:30Z".dim(),
            ]))
        );
    }
}

#[test]
fn label_only_tier_does_not_rewrite_the_goal_label() {
    let indicator = GoalStatusIndicator::UsageLimited;
    let expected = label_line(&indicator);

    assert_eq!(
        line_fitting_width(Some(&indicator), expected.width()),
        Some(expected)
    );
}
