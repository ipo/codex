use super::*;
use pretty_assertions::assert_eq;

#[test]
fn full_indicator_styles_build_identity_separately() {
    assert_eq!(
        line(/*show_cycle_hint*/ true),
        Line::from(vec![
            "Plan mode (shift+tab to cycle)".magenta(),
            " · ".dim(),
            "feature/robustness@0.144.1".dim(),
            " · ".dim(),
            "built 2026-07-12T13:31:30Z".dim(),
        ])
    );
}

#[test]
fn indicator_drops_the_optional_cycle_hint_before_provenance() {
    let compact = line(/*show_cycle_hint*/ false);
    assert_eq!(
        line_fitting_width(/*show_cycle_hint*/ true, compact.width()),
        Some(compact)
    );
}
