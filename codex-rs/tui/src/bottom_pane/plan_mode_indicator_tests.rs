use super::*;
use pretty_assertions::assert_eq;

fn text(line: Option<Line<'static>>) -> Option<String> {
    line.map(|line| {
        line.spans
            .into_iter()
            .fold(String::new(), |mut text, span| {
                text.push_str(span.content.as_ref());
                text
            })
    })
}

#[test]
fn full_indicator_styles_build_identity_separately() {
    let identity = BuildIdentity::new("feature/robustness", "0.144.1");

    assert_eq!(
        full_line_for(identity, /*show_cycle_hint*/ true),
        Line::from(vec![
            "Plan mode (shift+tab to cycle)".magenta(),
            " · ".dim(),
            "feature/robustness@0.144.1".dim(),
        ])
    );
}

#[test]
fn indicator_compacts_through_each_identity_tier() {
    let identity = BuildIdentity::new("feature/robustness", "0.144.1");

    assert_eq!(
        [38, 32, 19, 9, 8].map(|max_width| {
            text(line_fitting_width_for(
                identity, /*show_cycle_hint*/ false, max_width,
            ))
        }),
        [
            Some("Plan mode · feature/robustness@0.144.1".to_string()),
            Some("Plan mode · feature/rob…@0.144.1".to_string()),
            Some("Plan mode · 0.144.1".to_string()),
            Some("Plan mode".to_string()),
            None,
        ]
    );
}

#[test]
fn branch_truncation_uses_unicode_display_width() {
    let identity = BuildIdentity::new("功能/robustness", "9.8.7");
    let line = line_fitting_width_for(identity, /*show_cycle_hint*/ false, 26)
        .expect("indicator should fit");

    assert_eq!(
        (line.width(), text(Some(line))),
        (26, Some("Plan mode · 功能/ro…@9.8.7".to_string()),)
    );
}
