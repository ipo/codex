use super::*;
use codex_protocol::protocol::RateLimitReachedType;
use codex_protocol::protocol::RateLimitWindow;
use pretty_assertions::assert_eq;

fn rate_limits(
    primary: Option<f64>,
    secondary: Option<f64>,
    rate_limit_reached_type: Option<RateLimitReachedType>,
) -> RateLimitSnapshot {
    let window = |used_percent| RateLimitWindow {
        used_percent,
        window_minutes: None,
        resets_at: None,
    };
    RateLimitSnapshot {
        limit_id: Some("codex".to_string()),
        limit_name: None,
        primary: primary.map(window),
        secondary: secondary.map(window),
        credits: None,
        individual_limit: None,
        plan_type: None,
        rate_limit_reached_type,
    }
}

fn decision(
    rate_limits: Option<&RateLimitSnapshot>,
    provider_id: &str,
    child_model: Option<&str>,
    has_explicit_override: bool,
) -> (SubagentBackendRoute, bool) {
    let decision = select_spawn_agent_backend_route(SpawnAgentRoutingInput {
        child_provider_id: provider_id,
        parent_model: "parent",
        parent_reasoning_effort: None,
        child_model,
        child_reasoning_effort: None,
        has_explicit_model_or_effort: has_explicit_override,
        codex_rate_limits: rate_limits,
    });
    (decision.backend_route, decision.quota_fallback)
}

#[test]
fn selects_route_from_override_eligibility_and_quota_state() {
    const EXPLICIT: bool = true;
    const INHERITED: bool = false;
    let below_threshold = rate_limits(Some(94.9), Some(10.0), None);
    let at_threshold = rate_limits(Some(95.0), Some(10.0), None);
    let secondary_exhausted = rate_limits(Some(10.0), Some(99.0), None);
    let reached = rate_limits(
        Some(10.0),
        Some(10.0),
        Some(RateLimitReachedType::RateLimitReached),
    );
    assert_eq!(
        [
            decision(None, OPENAI_PROVIDER_ID, Some("child"), EXPLICIT),
            decision(
                Some(&below_threshold),
                OPENAI_PROVIDER_ID,
                Some("child"),
                EXPLICIT,
            ),
        ],
        [(SubagentBackendRoute::MainSession, false); 2]
    );
    for exhausted in [&at_threshold, &secondary_exhausted, &reached] {
        assert_eq!(
            decision(Some(exhausted), OPENAI_PROVIDER_ID, Some("child"), EXPLICIT),
            (SubagentBackendRoute::ProperSubagent, true)
        );
    }
    assert_eq!(
        [
            decision(None, "custom", Some("child"), EXPLICIT),
            decision(None, OPENAI_PROVIDER_ID, Some("child"), INHERITED),
            decision(None, OPENAI_PROVIDER_ID, None, EXPLICIT),
        ],
        [(SubagentBackendRoute::ProperSubagent, false); 3]
    );
}
