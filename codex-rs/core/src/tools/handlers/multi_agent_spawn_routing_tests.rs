use super::*;
use codex_otel::MetricsClient;
use codex_otel::MetricsConfig;
use codex_otel::SessionTelemetry;
use codex_protocol::ThreadId;
use codex_protocol::protocol::RateLimitReachedType;
use codex_protocol::protocol::RateLimitWindow;
use codex_protocol::protocol::SessionSource;
use opentelemetry_sdk::metrics::InMemoryMetricExporter;
use opentelemetry_sdk::metrics::data::AggregatedMetrics;
use opentelemetry_sdk::metrics::data::MetricData;
use pretty_assertions::assert_eq;
use std::collections::BTreeMap;

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
    parent_reasoning_effort: Option<&ReasoningEffort>,
    child_reasoning_effort: Option<&ReasoningEffort>,
    has_explicit_override: bool,
) -> (SubagentBackendRoute, bool) {
    let decision = select_spawn_agent_backend_route(SpawnAgentRoutingInput {
        child_provider_id: provider_id,
        parent_model: "parent",
        parent_reasoning_effort,
        child_model,
        child_reasoning_effort,
        has_explicit_model_or_effort: has_explicit_override,
        codex_rate_limits: rate_limits,
    });
    (decision.backend_route, decision.quota_fallback)
}

#[test]
fn selects_route_from_override_eligibility_and_quota_state() {
    const EXPLICIT: bool = true;
    let below_primary_threshold = rate_limits(Some(94.999), Some(10.0), None);
    let below_secondary_threshold = rate_limits(Some(10.0), Some(94.999), None);
    let at_primary_threshold = rate_limits(Some(95.0), Some(10.0), None);
    let at_secondary_threshold = rate_limits(Some(10.0), Some(95.0), None);
    assert_eq!(
        [
            decision(
                None,
                OPENAI_PROVIDER_ID,
                Some("child"),
                None,
                None,
                EXPLICIT,
            ),
            decision(
                Some(&below_primary_threshold),
                OPENAI_PROVIDER_ID,
                Some("child"),
                None,
                None,
                EXPLICIT,
            ),
            decision(
                Some(&below_secondary_threshold),
                OPENAI_PROVIDER_ID,
                Some("child"),
                None,
                None,
                EXPLICIT,
            ),
        ],
        [(SubagentBackendRoute::MainSession, false); 3]
    );
    for exhausted in [&at_primary_threshold, &at_secondary_threshold] {
        assert_eq!(
            decision(
                Some(exhausted),
                OPENAI_PROVIDER_ID,
                Some("child"),
                None,
                None,
                EXPLICIT,
            ),
            (SubagentBackendRoute::ProperSubagent, true)
        );
    }
    for reached_type in [
        RateLimitReachedType::RateLimitReached,
        RateLimitReachedType::WorkspaceOwnerCreditsDepleted,
        RateLimitReachedType::WorkspaceMemberCreditsDepleted,
        RateLimitReachedType::WorkspaceOwnerUsageLimitReached,
        RateLimitReachedType::WorkspaceMemberUsageLimitReached,
    ] {
        let reached = rate_limits(Some(10.0), Some(10.0), Some(reached_type));
        assert_eq!(
            decision(
                Some(&reached),
                OPENAI_PROVIDER_ID,
                Some("child"),
                None,
                None,
                EXPLICIT,
            ),
            (SubagentBackendRoute::ProperSubagent, true)
        );
    }
}

#[test]
fn selects_main_route_only_for_an_explicit_effective_openai_change() {
    const EXPLICIT: bool = true;
    const INHERITED: bool = false;
    let medium = ReasoningEffort::Medium;
    let low = ReasoningEffort::Low;

    assert_eq!(
        decision(
            None,
            OPENAI_PROVIDER_ID,
            Some("parent"),
            Some(&medium),
            Some(&low),
            EXPLICIT,
        ),
        (SubagentBackendRoute::MainSession, false)
    );
    assert_eq!(
        [
            decision(
                None,
                "custom",
                Some("child"),
                Some(&medium),
                Some(&low),
                EXPLICIT,
            ),
            decision(
                None,
                OPENAI_PROVIDER_ID,
                Some("child"),
                Some(&medium),
                Some(&low),
                INHERITED,
            ),
            decision(
                None,
                OPENAI_PROVIDER_ID,
                Some("parent"),
                Some(&medium),
                Some(&medium),
                EXPLICIT,
            ),
        ],
        [(SubagentBackendRoute::ProperSubagent, false); 3]
    );
}

#[test]
fn spawn_routing_telemetry_records_route_and_quota_fallback() {
    let metrics = MetricsClient::new(
        MetricsConfig::in_memory(
            "test",
            "codex-core",
            env!("CARGO_PKG_VERSION"),
            InMemoryMetricExporter::default(),
        )
        .with_runtime_reader(),
    )
    .expect("in-memory metrics client");
    let telemetry = SessionTelemetry::new(
        ThreadId::new(),
        "gpt-5.4",
        "gpt-5.4",
        /*account_id*/ None,
        /*account_email*/ None,
        /*auth_mode*/ None,
        "test_originator".to_string(),
        /*log_user_prompts*/ false,
        "test".to_string(),
        SessionSource::Cli,
    )
    .with_metrics_without_metadata_tags(metrics);

    record_spawn_agent_routing_telemetry(
        &telemetry,
        "worker",
        "v2",
        SpawnAgentRoutingDecision {
            backend_route: SubagentBackendRoute::ProperSubagent,
            quota_fallback: true,
        },
    );

    let snapshot = telemetry
        .snapshot_metrics()
        .expect("runtime metrics snapshot");
    let metric = snapshot
        .scope_metrics()
        .flat_map(opentelemetry_sdk::metrics::data::ScopeMetrics::metrics)
        .find(|metric| metric.name() == "codex.multi_agent.spawn")
        .expect("spawn metric");
    let AggregatedMetrics::U64(MetricData::Sum(sum)) = metric.data() else {
        panic!("spawn metric should be a u64 counter");
    };
    let points = sum.data_points().collect::<Vec<_>>();
    assert_eq!(points.len(), 1);
    assert_eq!(points[0].value(), 1);
    assert_eq!(
        points[0]
            .attributes()
            .map(|attribute| {
                (
                    attribute.key.as_str().to_string(),
                    attribute.value.as_str().to_string(),
                )
            })
            .collect::<BTreeMap<_, _>>(),
        BTreeMap::from([
            ("backend_route".to_string(), "proper_subagent".to_string()),
            ("quota_fallback".to_string(), "true".to_string()),
            ("role".to_string(), "worker".to_string()),
            ("version".to_string(), "v2".to_string()),
        ])
    );
}
