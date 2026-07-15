use codex_model_provider_info::OPENAI_PROVIDER_ID;
use codex_protocol::openai_models::ReasoningEffort;
use codex_protocol::protocol::RateLimitSnapshot;
use codex_protocol::protocol::SubagentBackendRoute;

use crate::config::Config;
use crate::session::session::Session;
use crate::session::turn_context::TurnContext;

const MAIN_SESSION_ROUTE_QUOTA_FALLBACK_PERCENT: f64 = 95.0;

pub(crate) struct SpawnAgentRoutingInput<'a> {
    pub(crate) child_provider_id: &'a str,
    pub(crate) parent_model: &'a str,
    pub(crate) parent_reasoning_effort: Option<&'a ReasoningEffort>,
    pub(crate) child_model: Option<&'a str>,
    pub(crate) child_reasoning_effort: Option<&'a ReasoningEffort>,
    pub(crate) has_explicit_model_or_effort: bool,
    pub(crate) codex_rate_limits: Option<&'a RateLimitSnapshot>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SpawnAgentRoutingDecision {
    pub(crate) backend_route: SubagentBackendRoute,
    pub(crate) quota_fallback: bool,
}

pub(crate) fn select_spawn_agent_backend_route(
    input: SpawnAgentRoutingInput<'_>,
) -> SpawnAgentRoutingDecision {
    if !is_main_session_route_candidate(&input) {
        return SpawnAgentRoutingDecision {
            backend_route: SubagentBackendRoute::ProperSubagent,
            quota_fallback: false,
        };
    }

    if input
        .codex_rate_limits
        .is_some_and(quota_requires_proper_subagent_route)
    {
        return SpawnAgentRoutingDecision {
            backend_route: SubagentBackendRoute::ProperSubagent,
            quota_fallback: true,
        };
    }

    SpawnAgentRoutingDecision {
        backend_route: SubagentBackendRoute::MainSession,
        quota_fallback: false,
    }
}

pub(crate) async fn route_spawn_agent_config(
    session: &Session,
    turn: &TurnContext,
    config: &mut Config,
    has_explicit_model_or_effort: bool,
) -> SpawnAgentRoutingDecision {
    let parent_reasoning_effort = turn
        .reasoning_effort
        .as_ref()
        .or(turn.model_info.default_reasoning_level.as_ref());
    let mut input = SpawnAgentRoutingInput {
        child_provider_id: config.model_provider_id.as_str(),
        parent_model: turn.model_info.slug.as_str(),
        parent_reasoning_effort,
        child_model: config.model.as_deref(),
        child_reasoning_effort: config.model_reasoning_effort.as_ref(),
        has_explicit_model_or_effort,
        codex_rate_limits: None,
    };
    let codex_rate_limits = if is_main_session_route_candidate(&input) {
        session.codex_rate_limits().await
    } else {
        None
    };
    input.codex_rate_limits = codex_rate_limits.as_ref();
    let decision = select_spawn_agent_backend_route(input);
    if decision.quota_fallback {
        config.model = Some(turn.model_info.slug.clone());
        config.model_reasoning_effort = parent_reasoning_effort.cloned();
    }
    decision
}

fn is_main_session_route_candidate(input: &SpawnAgentRoutingInput<'_>) -> bool {
    let effective_settings_differ = input.child_model.unwrap_or(input.parent_model)
        != input.parent_model
        || input.child_reasoning_effort != input.parent_reasoning_effort;
    input.child_provider_id == OPENAI_PROVIDER_ID
        && input.has_explicit_model_or_effort
        && effective_settings_differ
}

fn quota_requires_proper_subagent_route(snapshot: &RateLimitSnapshot) -> bool {
    snapshot.rate_limit_reached_type.is_some()
        || snapshot
            .primary
            .as_ref()
            .is_some_and(|window| window.used_percent >= MAIN_SESSION_ROUTE_QUOTA_FALLBACK_PERCENT)
        || snapshot
            .secondary
            .as_ref()
            .is_some_and(|window| window.used_percent >= MAIN_SESSION_ROUTE_QUOTA_FALLBACK_PERCENT)
}

#[cfg(test)]
#[path = "multi_agent_spawn_routing_tests.rs"]
mod tests;
