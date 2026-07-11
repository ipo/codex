use super::*;
use codex_core::config::Config;
use codex_core::config::ConfigBuilder;
use codex_features::Feature;
use codex_protocol::protocol::MAX_THREAD_GOAL_OBJECTIVE_CHARS;
use pretty_assertions::assert_eq;

async fn goal_config() -> Config {
    let codex_home = tempfile::tempdir().expect("temporary codex home");
    let mut config = ConfigBuilder::default()
        .codex_home(codex_home.path().to_path_buf())
        .build()
        .await
        .expect("test config");
    config
        .features
        .set_enabled(Feature::Goals, /*enabled*/ true)
        .expect("enable Goals");
    config
}

fn fresh_preflight() -> GoalPreflight {
    GoalPreflight {
        invocation: GoalInvocation::Fresh,
        image_count: 0,
        output_schema_present: false,
    }
}

#[tokio::test]
async fn classifies_only_canonical_leading_goal_prompt() {
    let config = goal_config().await;
    for (prompt, expected) in [
        ("/goal finish the migration", Some("finish the migration")),
        ("/goal\nfinish the migration", Some("finish the migration")),
        (" /goal finish the migration", None),
        ("/goalkeeper finish the migration", None),
        ("/other finish the migration", None),
    ] {
        assert_eq!(
            classify_goal_prompt(prompt, &config, fresh_preflight())
                .expect("classification should succeed")
                .as_deref(),
            expected,
            "unexpected classification for {prompt:?}"
        );
    }
}

#[tokio::test]
async fn goal_preflight_rejects_every_unsupported_configuration() {
    let config = goal_config().await;
    assert_eq!(
        classify_goal_prompt("/goal   ", &config, fresh_preflight()),
        Err(GoalPreflightError::MissingObjective)
    );
    assert_eq!(
        classify_goal_prompt(
            "/goal objective",
            &config,
            GoalPreflight {
                invocation: GoalInvocation::Resume,
                ..fresh_preflight()
            },
        ),
        Err(GoalPreflightError::ResumeUnsupported)
    );

    let mut ephemeral_config = config.clone();
    ephemeral_config.ephemeral = true;
    assert_eq!(
        classify_goal_prompt("/goal objective", &ephemeral_config, fresh_preflight()),
        Err(GoalPreflightError::EphemeralUnsupported)
    );

    let mut disabled_config = config.clone();
    disabled_config
        .features
        .set_enabled(Feature::Goals, /*enabled*/ false)
        .expect("disable Goals");
    assert_eq!(
        classify_goal_prompt("/goal objective", &disabled_config, fresh_preflight()),
        Err(GoalPreflightError::GoalsDisabled)
    );
    assert_eq!(
        classify_goal_prompt(
            "/goal objective",
            &config,
            GoalPreflight {
                image_count: 1,
                ..fresh_preflight()
            },
        ),
        Err(GoalPreflightError::ImagesUnsupported)
    );
    assert_eq!(
        classify_goal_prompt(
            "/goal objective",
            &config,
            GoalPreflight {
                output_schema_present: true,
                ..fresh_preflight()
            },
        ),
        Err(GoalPreflightError::OutputSchemaUnsupported)
    );

    let objective = "x".repeat(MAX_THREAD_GOAL_OBJECTIVE_CHARS + 1);
    assert_eq!(
        classify_goal_prompt(&format!("/goal {objective}"), &config, fresh_preflight()),
        Err(GoalPreflightError::ObjectiveTooLong {
            actual: MAX_THREAD_GOAL_OBJECTIVE_CHARS + 1,
            limit: MAX_THREAD_GOAL_OBJECTIVE_CHARS,
        })
    );
}

#[test]
fn every_goal_status_has_an_explicit_disposition() {
    for (status, expected) in [
        (ThreadGoalStatus::Active, GoalDisposition::Continue),
        (ThreadGoalStatus::Complete, GoalDisposition::Complete),
        (ThreadGoalStatus::Paused, GoalDisposition::Incomplete),
        (ThreadGoalStatus::Blocked, GoalDisposition::Incomplete),
        (ThreadGoalStatus::UsageLimited, GoalDisposition::Incomplete),
        (ThreadGoalStatus::BudgetLimited, GoalDisposition::Incomplete),
    ] {
        assert_eq!(disposition_for_status(status), expected);
    }
}

#[test]
fn goal_run_tracks_dynamic_turns_and_wait_windows() {
    let mut run = GoalRun::new("thread-1".to_string());
    assert!(run.next_turn_deadline().is_some());
    assert_eq!(run.active_turn_id(), None);
    assert!(!run.observe_turn_started("other-thread", "turn-1"));
    assert!(run.observe_turn_started("thread-1", "turn-1"));
    assert_eq!(run.active_turn_id(), Some("turn-1"));
    assert_eq!(run.next_turn_deadline(), None);
    assert_eq!(
        run.observe_turn_completed("thread-1", "turn-1", ThreadGoalStatus::Active),
        Some(GoalDisposition::Continue)
    );
    assert_eq!(run.active_turn_id(), None);
    assert!(run.next_turn_deadline().is_some());
    assert!(run.observe_turn_started("thread-1", "turn-2"));
    assert_eq!(
        run.observe_turn_completed("thread-1", "turn-2", ThreadGoalStatus::Complete),
        Some(GoalDisposition::Complete)
    );
    assert_eq!(run.next_turn_deadline(), None);
}

#[test]
fn cancellation_pauses_before_interrupting_and_handles_idle_waits() {
    let mut active = GoalRun::new("thread-1".to_string());
    assert!(active.observe_turn_started("thread-1", "turn-1"));
    assert_eq!(
        cancellation_actions(&active),
        vec![
            GoalCancellationAction::PauseGoal,
            GoalCancellationAction::InterruptTurn {
                turn_id: "turn-1".to_string(),
            },
        ]
    );

    let idle = GoalRun::new("thread-2".to_string());
    assert_eq!(
        cancellation_actions(&idle),
        vec![GoalCancellationAction::PauseGoal]
    );
}
