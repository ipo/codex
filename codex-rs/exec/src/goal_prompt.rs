use std::fmt;

use codex_core::config::Config;
use codex_features::Feature;
use codex_protocol::protocol::MAX_THREAD_GOAL_OBJECTIVE_CHARS;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GoalInvocation {
    Fresh,
    Resume,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GoalPreflight {
    pub(crate) invocation: GoalInvocation,
    pub(crate) image_count: usize,
    pub(crate) output_schema_present: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum GoalPreflightError {
    MissingObjective,
    ResumeUnsupported,
    EphemeralUnsupported,
    GoalsDisabled,
    ImagesUnsupported,
    OutputSchemaUnsupported,
    ObjectiveTooLong { actual: usize, limit: usize },
}

impl fmt::Display for GoalPreflightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingObjective => write!(f, "`/goal` requires a non-empty objective."),
            Self::ResumeUnsupported => write!(
                f,
                "`/goal` is not supported with `codex exec resume`; exec will not replace or mutate an existing thread goal."
            ),
            Self::EphemeralUnsupported => write!(
                f,
                "`/goal` requires a persisted exec thread and cannot be used with `--ephemeral`."
            ),
            Self::GoalsDisabled => {
                write!(
                    f,
                    "`/goal` is unavailable because the Goals feature is disabled."
                )
            }
            Self::ImagesUnsupported => write!(
                f,
                "`/goal` does not support `--image`; remove image attachments and include paths in the objective instead."
            ),
            Self::OutputSchemaUnsupported => {
                write!(f, "`/goal` cannot be used with `--output-schema`.")
            }
            Self::ObjectiveTooLong { actual, limit } => write!(
                f,
                "Goal objective is {actual} characters; the maximum is {limit}."
            ),
        }
    }
}

impl std::error::Error for GoalPreflightError {}

pub(crate) fn classify_goal_prompt(
    prompt: &str,
    config: &Config,
    preflight: GoalPreflight,
) -> Result<Option<String>, GoalPreflightError> {
    let Some(rest) = prompt.strip_prefix("/goal") else {
        return Ok(None);
    };
    if rest
        .chars()
        .next()
        .is_some_and(|character| !character.is_whitespace())
    {
        return Ok(None);
    }

    let objective = rest.trim();
    if objective.is_empty() {
        return Err(GoalPreflightError::MissingObjective);
    }
    if preflight.invocation == GoalInvocation::Resume {
        return Err(GoalPreflightError::ResumeUnsupported);
    }
    if config.ephemeral {
        return Err(GoalPreflightError::EphemeralUnsupported);
    }
    if !config.features.enabled(Feature::Goals) {
        return Err(GoalPreflightError::GoalsDisabled);
    }
    if preflight.image_count != 0 {
        return Err(GoalPreflightError::ImagesUnsupported);
    }
    if preflight.output_schema_present {
        return Err(GoalPreflightError::OutputSchemaUnsupported);
    }
    let actual = objective.chars().count();
    if actual > MAX_THREAD_GOAL_OBJECTIVE_CHARS {
        return Err(GoalPreflightError::ObjectiveTooLong {
            actual,
            limit: MAX_THREAD_GOAL_OBJECTIVE_CHARS,
        });
    }

    Ok(Some(objective.to_string()))
}
