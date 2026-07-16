use super::*;
use pretty_assertions::assert_eq;
use ratatui::text::Span;

fn text(line: Option<Line<'static>>) -> Option<String> {
    line.map(|line| {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect()
    })
}

#[test]
fn styles_build_identity_separately_from_the_label() {
    let identity = BuildIdentity::new("feature/robustness", "0.144.1", "2026-07-12T13:31:30Z");

    assert_eq!(
        line_for(Line::from(vec!["Pursuing goal".magenta()]), identity),
        Line::from(vec![
            "Pursuing goal".magenta(),
            " · ".dim(),
            "feature/robustness@0.144.1".dim(),
            " · ".dim(),
            "built 2026-07-12T13:31:30Z".dim(),
        ])
    );
}

#[test]
fn compacts_through_each_identity_tier() {
    let identity = BuildIdentity::new("feature/robustness", "0.144.1", "2026-07-12T13:31:30Z");
    let label = || Line::from(vec![Span::from("Pursuing goal").magenta()]);

    let tiers = [71, 42, 36, 23, 13, 12]
        .map(|max_width| text(line_fitting_width_for(label(), identity, max_width)));

    assert_eq!(
        tiers,
        [
            Some(
                "Pursuing goal · feature/robustness@0.144.1 · built 2026-07-12T13:31:30Z"
                    .to_string()
            ),
            Some("Pursuing goal · feature/robustness@0.144.1".to_string()),
            Some("Pursuing goal · feature/rob…@0.144.1".to_string()),
            Some("Pursuing goal · 0.144.1".to_string()),
            Some("Pursuing goal".to_string()),
            None,
        ]
    );
    insta::assert_debug_snapshot!("goal_build_provenance_ascii_width_tiers", tiers);
}

#[test]
fn branch_truncation_uses_unicode_display_width() {
    let identity = BuildIdentity::new("功能/robustness", "9.8.7", "2026-07-12T13:31:30Z");
    let line = line_fitting_width_for(
        Line::from(vec!["Pursuing goal".magenta()]),
        identity,
        /*max_width*/ 30,
    )
    .expect("indicator should fit");
    let rendered = (line.width(), text(Some(line)));

    assert_eq!(
        rendered,
        (30, Some("Pursuing goal · 功能/ro…@9.8.7".to_string()))
    );
    insta::assert_debug_snapshot!("goal_build_provenance_unicode_width", rendered);
}
