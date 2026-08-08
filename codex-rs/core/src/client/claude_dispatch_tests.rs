use super::opus_environment;
use crate::responses_metadata::CodexResponsesMetadata;

#[test]
fn opus_environment_fails_closed_without_authoritative_target_facts() {
    let metadata = CodexResponsesMetadata::new(
        "installation".to_string(),
        "session".to_string(),
        "thread".to_string(),
        "window".to_string(),
    );

    let error = opus_environment(&metadata).expect_err("legacy target facts must be rejected");

    assert_eq!(
        error.to_string(),
        "Opus 5 compatibility requires authoritative execution environment facts"
    );
}
