use super::full_context_window_limit;
use codex_models_manager::bundled_models_response;
use pretty_assertions::assert_eq;

fn bundled_model(slug: &str) -> codex_protocol::openai_models::ModelInfo {
    bundled_models_response()
        .expect("bundled catalog should parse")
        .models
        .into_iter()
        .find(|model| model.slug == slug)
        .unwrap_or_else(|| panic!("missing bundled model {slug}"))
}

#[test]
fn bundled_haiku_reserves_native_output_from_usable_window() {
    let model = bundled_model("anthropic/claude-haiku-4-5-20251001");
    assert_eq!(model.resolved_context_window(), Some(200_000));
    assert_eq!(model.effective_context_window_percent, 95);
    assert_eq!(full_context_window_limit(&model), Some(158_000));
}

#[test]
fn bundled_adaptive_claude_profiles_reserve_64k_output() {
    for slug in [
        "anthropic/claude-fable-5",
        "anthropic/claude-opus-5",
        "anthropic/claude-opus-4-8",
        "anthropic/claude-sonnet-5",
    ] {
        let model = bundled_model(slug);
        assert_eq!(model.resolved_context_window(), Some(1_000_000), "{slug}");
        assert_eq!(model.effective_context_window_percent, 95, "{slug}");
        assert_eq!(full_context_window_limit(&model), Some(886_000), "{slug}");
    }
}

#[test]
fn non_anthropic_models_keep_percentage_only_hard_limit() {
    let model = bundled_model("gpt-5.4");
    let expected = model.resolved_context_window().map(|context_window| {
        context_window.saturating_mul(model.effective_context_window_percent) / 100
    });
    assert_eq!(full_context_window_limit(&model), expected);
}
