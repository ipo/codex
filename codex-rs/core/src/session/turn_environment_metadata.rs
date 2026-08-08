use codex_file_system::FindUpErrorPolicy;
use codex_file_system::find_nearest_ancestor_with_markers;
use codex_protocol::model_inference::ModelInferenceConfig;
use codex_protocol::openai_models::ModelInfo;

use super::turn_context::TurnEnvironment;
use crate::responses_metadata::TurnExecutionEnvironment;
use crate::turn_metadata::local_model_visible_shell;

pub(super) async fn collect_opus_turn_environment(
    model_info: &ModelInfo,
    primary: Option<&TurnEnvironment>,
) -> Option<TurnExecutionEnvironment> {
    if !matches!(
        model_info.inference.as_ref(),
        Some(ModelInferenceConfig::Anthropic { wire_model, .. })
            if wire_model == "claude-opus-5"
    ) {
        return None;
    }
    let primary = primary?;
    let (system, target_shell) = primary.info().and_then(|info| {
        info.system
            .clone()
            .map(|system| (system, info.shell.path.clone()))
    })?;
    let repo_root = match find_nearest_ancestor_with_markers(
        primary.environment.get_filesystem().as_ref(),
        primary.cwd(),
        vec![".git".to_string()],
        FindUpErrorPolicy::Propagate,
        /*sandbox*/ None,
    )
    .await
    {
        Ok(repo_root) => repo_root,
        Err(err) => {
            tracing::warn!(
                environment_id = primary.environment_id,
                "failed to detect Opus 5 repository state in selected environment: {err}"
            );
            return None;
        }
    };

    Some(TurnExecutionEnvironment {
        cwd: primary.cwd().clone(),
        is_git_repository: repo_root.is_some(),
        shell: if primary.environment.is_remote() {
            target_shell
        } else {
            local_model_visible_shell()
        },
        system,
    })
}
