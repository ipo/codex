use codex_api::TerminalOutcome;
use pretty_assertions::assert_eq;

use super::SamplingAttempt;

#[test]
fn terminal_outcome_table_controls_commit_and_continuation() {
    for (outcome, expected_follow_up) in [
        (TerminalOutcome::Completed, false),
        (TerminalOutcome::ToolsReady, true),
        (TerminalOutcome::Continue, true),
    ] {
        let committed = SamplingAttempt::default()
            .finish(outcome)
            .expect("committable terminal outcome");
        assert_eq!(committed.needs_follow_up, expected_follow_up);
    }

    for outcome in [TerminalOutcome::OutputExhausted, TerminalOutcome::Refusal] {
        let error = SamplingAttempt::default()
            .finish(outcome)
            .expect_err("discarding terminal outcome");
        assert!(!error.is_retryable());
    }
}
