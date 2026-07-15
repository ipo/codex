use super::*;
use crate::session::tests::make_session_configuration_for_tests;
use crate::state::AutoCompactWindowSnapshot;
use codex_protocol::protocol::CreditsSnapshot;
use codex_protocol::protocol::RateLimitWindow;
use codex_protocol::protocol::SpendControlLimitSnapshot;
use pretty_assertions::assert_eq;

#[tokio::test]
// Verifies connector merging deduplicates repeated IDs.
async fn merge_connector_selection_deduplicates_entries() {
    let session_configuration = make_session_configuration_for_tests().await;
    let mut state = SessionState::new(session_configuration);
    let merged = state.merge_connector_selection([
        "calendar".to_string(),
        "calendar".to_string(),
        "drive".to_string(),
    ]);

    assert_eq!(
        merged,
        HashSet::from(["calendar".to_string(), "drive".to_string()])
    );
}

#[tokio::test]
// Verifies clearing connector selection removes all saved IDs.
async fn clear_connector_selection_removes_entries() {
    let session_configuration = make_session_configuration_for_tests().await;
    let mut state = SessionState::new(session_configuration);
    state.merge_connector_selection(["calendar".to_string()]);

    state.clear_connector_selection();

    assert_eq!(state.get_connector_selection(), HashSet::new());
}

#[tokio::test]
async fn set_rate_limits_defaults_limit_id_to_codex_when_missing() {
    let session_configuration = make_session_configuration_for_tests().await;
    let mut state = SessionState::new(session_configuration);

    state.set_rate_limits(RateLimitSnapshot {
        limit_id: None,
        limit_name: None,
        primary: Some(RateLimitWindow {
            used_percent: 12.0,
            window_minutes: Some(60),
            resets_at: Some(100),
        }),
        secondary: None,
        credits: None,
        individual_limit: None,
        plan_type: None,
        rate_limit_reached_type: None,
    });

    assert_eq!(
        state
            .latest_rate_limits
            .as_ref()
            .and_then(|v| v.limit_id.clone()),
        Some("codex".to_string())
    );
}

#[tokio::test]
async fn set_rate_limits_normalizes_nonempty_limit_id() {
    let session_configuration = make_session_configuration_for_tests().await;
    let mut state = SessionState::new(session_configuration);

    state.set_rate_limits(RateLimitSnapshot {
        limit_id: Some(" CoDeX-Other ".to_string()),
        limit_name: None,
        primary: None,
        secondary: None,
        credits: None,
        individual_limit: None,
        plan_type: None,
        rate_limit_reached_type: None,
    });

    assert!(state.rate_limits_by_id.contains_key("codex_other"));
    assert_eq!(
        state
            .latest_rate_limits
            .as_ref()
            .and_then(|snapshot| snapshot.limit_id.as_deref()),
        Some("codex_other")
    );
}

#[tokio::test]
async fn replace_history_clears_auto_compact_window_prefill() {
    let session_configuration = make_session_configuration_for_tests().await;
    let mut state = SessionState::new(session_configuration);

    state.set_auto_compact_window_estimated_prefill(/*tokens*/ 100);
    state.replace_history(Vec::new(), /*reference_context_item*/ None);

    assert_eq!(
        state.auto_compact_window_snapshot(),
        AutoCompactWindowSnapshot {
            prefill_input_tokens: None,
        }
    );
}

#[tokio::test]
async fn set_rate_limits_defaults_to_codex_when_limit_id_missing_after_other_bucket() {
    let session_configuration = make_session_configuration_for_tests().await;
    let mut state = SessionState::new(session_configuration);

    state.set_rate_limits(RateLimitSnapshot {
        limit_id: Some("codex_other".to_string()),
        limit_name: Some("codex_other".to_string()),
        primary: Some(RateLimitWindow {
            used_percent: 20.0,
            window_minutes: Some(60),
            resets_at: Some(200),
        }),
        secondary: None,
        credits: None,
        individual_limit: None,
        plan_type: None,
        rate_limit_reached_type: None,
    });
    state.set_rate_limits(RateLimitSnapshot {
        limit_id: None,
        limit_name: None,
        primary: Some(RateLimitWindow {
            used_percent: 30.0,
            window_minutes: Some(60),
            resets_at: Some(300),
        }),
        secondary: None,
        credits: None,
        individual_limit: None,
        plan_type: None,
        rate_limit_reached_type: None,
    });

    assert_eq!(
        state
            .latest_rate_limits
            .as_ref()
            .and_then(|v| v.limit_id.clone()),
        Some("codex".to_string())
    );
    assert_eq!(
        state.codex_rate_limits(),
        Some(RateLimitSnapshot {
            limit_id: Some("codex".to_string()),
            limit_name: None,
            primary: Some(RateLimitWindow {
                used_percent: 30.0,
                window_minutes: Some(60),
                resets_at: Some(300),
            }),
            secondary: None,
            credits: None,
            individual_limit: None,
            plan_type: None,
            rate_limit_reached_type: None,
        })
    );
}

#[tokio::test]
async fn set_rate_limits_carries_account_metadata_from_codex_to_codex_other() {
    let session_configuration = make_session_configuration_for_tests().await;
    let mut state = SessionState::new(session_configuration);

    let initial = RateLimitSnapshot {
        limit_id: Some("codex".to_string()),
        limit_name: Some("codex".to_string()),
        primary: Some(RateLimitWindow {
            used_percent: 10.0,
            window_minutes: Some(60),
            resets_at: Some(100),
        }),
        secondary: None,
        credits: Some(CreditsSnapshot {
            has_credits: true,
            unlimited: false,
            balance: Some("50".to_string()),
        }),
        individual_limit: Some(SpendControlLimitSnapshot {
            limit: "25000".to_string(),
            used: "8000".to_string(),
            remaining_percent: 68,
            resets_at: 300,
        }),
        plan_type: Some(codex_protocol::account::PlanType::Plus),
        rate_limit_reached_type: None,
    };
    state.set_rate_limits(initial.clone());

    state.set_rate_limits(RateLimitSnapshot {
        limit_id: Some("codex_other".to_string()),
        limit_name: None,
        primary: Some(RateLimitWindow {
            used_percent: 30.0,
            window_minutes: Some(120),
            resets_at: Some(200),
        }),
        secondary: None,
        credits: None,
        individual_limit: None,
        plan_type: None,
        rate_limit_reached_type: None,
    });

    assert_eq!(
        state.latest_rate_limits,
        Some(RateLimitSnapshot {
            limit_id: Some("codex_other".to_string()),
            limit_name: None,
            primary: Some(RateLimitWindow {
                used_percent: 30.0,
                window_minutes: Some(120),
                resets_at: Some(200),
            }),
            secondary: None,
            credits: Some(CreditsSnapshot {
                has_credits: true,
                unlimited: false,
                balance: Some("50".to_string()),
            }),
            individual_limit: Some(SpendControlLimitSnapshot {
                limit: "25000".to_string(),
                used: "8000".to_string(),
                remaining_percent: 68,
                resets_at: 300,
            }),
            plan_type: Some(codex_protocol::account::PlanType::Plus),
            rate_limit_reached_type: None,
        })
    );
    assert_eq!(state.codex_rate_limits(), Some(initial));
}

#[tokio::test]
async fn sparse_bucket_update_preserves_latest_metadata_from_the_latest_bucket() {
    let session_configuration = make_session_configuration_for_tests().await;
    let mut state = SessionState::new(session_configuration);
    let snapshot = |limit_id: &str, used_percent: f64, balance: Option<&str>| RateLimitSnapshot {
        limit_id: Some(limit_id.to_string()),
        limit_name: Some(limit_id.to_string()),
        primary: Some(RateLimitWindow {
            used_percent,
            window_minutes: Some(60),
            resets_at: Some(100),
        }),
        secondary: None,
        credits: balance.map(|balance| CreditsSnapshot {
            has_credits: true,
            unlimited: false,
            balance: Some(balance.to_string()),
        }),
        individual_limit: None,
        plan_type: None,
        rate_limit_reached_type: None,
    };
    let initial_codex = snapshot("codex", 10.0, Some("50"));
    let latest_other = snapshot("codex_other", 20.0, Some("40"));
    let sparse_codex = snapshot("codex", 30.0, None);

    state.set_rate_limits(initial_codex);
    state.set_rate_limits(latest_other.clone());
    state.set_rate_limits(sparse_codex.clone());

    assert_eq!(
        state.latest_rate_limits,
        Some(RateLimitSnapshot {
            credits: latest_other.credits,
            ..sparse_codex.clone()
        })
    );
    assert_eq!(
        state.codex_rate_limits(),
        Some(RateLimitSnapshot {
            credits: Some(CreditsSnapshot {
                has_credits: true,
                unlimited: false,
                balance: Some("50".to_string()),
            }),
            ..sparse_codex
        })
    );
}

#[tokio::test]
async fn empty_default_bucket_update_retains_quota_for_routing_only() {
    let session_configuration = make_session_configuration_for_tests().await;
    let mut state = SessionState::new(session_configuration);
    let initial = RateLimitSnapshot {
        limit_id: Some("codex".to_string()),
        limit_name: Some("Codex".to_string()),
        primary: Some(RateLimitWindow {
            used_percent: 95.0,
            window_minutes: Some(300),
            resets_at: Some(100),
        }),
        secondary: Some(RateLimitWindow {
            used_percent: 80.0,
            window_minutes: Some(10_080),
            resets_at: Some(200),
        }),
        credits: None,
        individual_limit: None,
        plan_type: None,
        rate_limit_reached_type: Some(
            codex_protocol::protocol::RateLimitReachedType::RateLimitReached,
        ),
    };
    state.set_rate_limits(initial.clone());

    let empty_update = RateLimitSnapshot {
        limit_id: Some("codex".to_string()),
        limit_name: None,
        primary: None,
        secondary: None,
        credits: None,
        individual_limit: None,
        plan_type: None,
        rate_limit_reached_type: None,
    };
    state.set_rate_limits(empty_update.clone());

    assert_eq!(state.codex_rate_limits(), Some(initial));
    assert_eq!(state.latest_rate_limits, Some(empty_update));
}
