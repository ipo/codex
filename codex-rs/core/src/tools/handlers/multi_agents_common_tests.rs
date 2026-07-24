use super::*;
use crate::environment_selection::TurnEnvironmentState;
use crate::session::tests::make_session_and_context;
use crate::session::turn_context::TurnEnvironment;
use codex_protocol::models::PermissionProfile;
use codex_utils_path_uri::PathUri;
use pretty_assertions::assert_eq;
use std::sync::Arc;

#[tokio::test]
async fn spawn_agent_cwd_updates_only_the_primary_environment() {
    let (_session, mut turn) = make_session_and_context().await;
    turn.permission_profile = PermissionProfile::Disabled;
    Arc::make_mut(&mut turn.config)
        .permissions
        .set_permission_profile(PermissionProfile::Disabled)
        .expect("disable sandboxing");
    let primary = turn
        .environments
        .primary()
        .cloned()
        .expect("primary environment");
    let primary_cwd = primary.cwd().to_abs_path().expect("host cwd");
    let child_cwd = tempfile::Builder::new()
        .prefix("spawn-agent-child-cwd-")
        .tempdir_in(primary_cwd.as_path())
        .expect("create child cwd");
    let child_cwd_name = child_cwd
        .path()
        .file_name()
        .expect("child cwd basename")
        .to_string_lossy()
        .into_owned();
    let secondary = tempfile::tempdir().expect("secondary cwd");
    turn.environments
        .environments
        .push(TurnEnvironmentState::Ready(TurnEnvironment::new(
            "secondary".to_string(),
            Arc::clone(&primary.environment),
            PathUri::from_host_native_path(secondary.path()).expect("secondary cwd URI"),
            Vec::new(),
            primary.shell.clone(),
        )));
    let original = turn.environments.to_selections();
    let original_workspace_roots = turn.config.effective_workspace_roots();
    let expected_primary_cwd = primary
        .cwd()
        .join(&child_cwd_name)
        .expect("resolved child cwd");

    let resolved = resolve_spawn_agent_environments(&turn, Some(&child_cwd_name))
        .await
        .expect("relative cwd should resolve");
    let mut expected = original.clone();
    expected[0].cwd = expected_primary_cwd;

    assert_eq!(resolved, expected);
    assert_eq!(turn.environments.to_selections(), original);
    assert_eq!(
        turn.config.effective_workspace_roots(),
        original_workspace_roots
    );
}

#[tokio::test]
async fn spawn_agent_cwd_accepts_a_readable_absolute_directory_outside_workspace_roots() {
    let (_session, mut turn) = make_session_and_context().await;
    turn.permission_profile = PermissionProfile::Disabled;
    Arc::make_mut(&mut turn.config)
        .permissions
        .set_permission_profile(PermissionProfile::Disabled)
        .expect("disable sandboxing");
    let target = tempfile::tempdir().expect("target cwd");
    let target_uri = PathUri::from_host_native_path(target.path()).expect("target cwd URI");
    assert!(
        !turn
            .config
            .effective_workspace_roots()
            .iter()
            .any(|root| target.path().starts_with(root.as_path())),
        "test target must be outside inherited workspace roots"
    );

    let resolved =
        resolve_spawn_agent_environments(&turn, Some(&target_uri.inferred_native_path_string()))
            .await
            .expect("readable absolute cwd should resolve");
    let mut expected = turn.environments.to_selections();
    expected[0].cwd = target_uri;

    assert_eq!(resolved, expected);
}

#[tokio::test]
async fn spawn_agent_cwd_rejects_invalid_targets_without_changing_the_parent() {
    let (_session, mut turn) = make_session_and_context().await;
    turn.permission_profile = PermissionProfile::Disabled;
    Arc::make_mut(&mut turn.config)
        .permissions
        .set_permission_profile(PermissionProfile::Disabled)
        .expect("disable sandboxing");
    let primary_cwd = turn
        .environments
        .primary()
        .expect("primary environment")
        .cwd()
        .to_abs_path()
        .expect("host cwd");
    let file = tempfile::Builder::new()
        .prefix("spawn-agent-cwd-file-")
        .tempfile_in(primary_cwd.as_path())
        .expect("write non-directory target");
    let file_name = file
        .path()
        .file_name()
        .expect("file target basename")
        .to_string_lossy()
        .into_owned();
    let missing = tempfile::Builder::new()
        .prefix("missing-spawn-agent-cwd-")
        .tempdir_in(primary_cwd.as_path())
        .expect("temporary missing cwd");
    let missing_name = missing
        .path()
        .file_name()
        .expect("missing cwd basename")
        .to_string_lossy()
        .into_owned();
    drop(missing);
    let original = turn.environments.to_selections();

    for (requested_cwd, expected_message) in [
        ("".to_string(), "spawn_agent cwd must not be empty"),
        (missing_name, "cannot access cwd"),
        (file_name, "is not a directory"),
        (
            r"C:\foreign".to_string(),
            "uses the Windows path convention",
        ),
    ] {
        let error = resolve_spawn_agent_environments(&turn, Some(&requested_cwd))
            .await
            .expect_err("cwd should be rejected");
        let FunctionCallError::RespondToModel(message) = error else {
            panic!("expected model-visible cwd error, got {error:?}");
        };
        assert!(
            message.contains(expected_message),
            "unexpected error for {requested_cwd:?}: {message}"
        );
        assert_eq!(turn.environments.to_selections(), original);
    }
}

#[tokio::test]
async fn spawn_agent_without_cwd_preserves_environment_selections() {
    let (_session, turn) = make_session_and_context().await;
    let expected = turn.environments.to_selections();

    assert_eq!(
        resolve_spawn_agent_environments(&turn, None).await,
        Ok(expected)
    );
}
