use anyhow::Result;
use anyhow::anyhow;
use codex_exec_server::CreateDirectoryOptions;
use codex_exec_server::WriteFileOptions;
use codex_features::Feature;
use codex_protocol::config_types::ApprovalsReviewer;
use codex_protocol::models::PermissionProfile;
use codex_utils_path_uri::PathUri;
use core_test_support::responses;
use core_test_support::responses::ResponseMock;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;
use serde_json::json;
use std::time::Duration;
use test_case::test_case;

const PARENT_PROMPT: &str = "spawn a child in the selected workspace";
const CHILD_PROMPT: &str = "inspect the selected working directory";
const CHILD_INSTRUCTIONS: &str = "CHILD_CWD_AGENTS_INSTRUCTIONS";
const SPAWN_CALL_ID: &str = "spawn-child-cwd";

#[derive(Clone, Copy)]
enum SpawnToolVersion {
    V1,
    V2,
}

#[derive(Clone, Copy)]
enum CwdInput {
    Relative,
    Absolute,
    Omitted,
}

#[derive(Clone, Copy)]
enum ForkMode {
    Fresh,
    Truncated,
    Full,
}

#[test_case(SpawnToolVersion::V1, CwdInput::Relative, ForkMode::Fresh; "v1 relative fresh child cwd")]
#[test_case(SpawnToolVersion::V1, CwdInput::Omitted, ForkMode::Fresh; "v1 omitted fresh child cwd")]
#[test_case(SpawnToolVersion::V1, CwdInput::Relative, ForkMode::Full; "v1 relative full history child cwd")]
// `build_with_auto_env` runs this as a valid Windows absolute path against the Wine executor in
// remote CI, while still exercising the local executor's absolute-path handling elsewhere.
#[test_case(SpawnToolVersion::V2, CwdInput::Absolute, ForkMode::Fresh; "v2 absolute fresh child cwd")]
#[test_case(SpawnToolVersion::V2, CwdInput::Relative, ForkMode::Truncated; "v2 relative truncated child cwd")]
#[test_case(SpawnToolVersion::V2, CwdInput::Relative, ForkMode::Full; "v2 relative full history child cwd")]
#[test_case(SpawnToolVersion::V2, CwdInput::Omitted, ForkMode::Fresh; "v2 omitted fresh child cwd")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spawn_agent_cwd_selects_or_inherits_child_environment(
    version: SpawnToolVersion,
    cwd_input: CwdInput,
    fork_mode: ForkMode,
) -> Result<()> {
    let server = responses::start_mock_server().await;
    let mut builder = test_codex()
        .with_workspace_setup(|cwd, filesystem| async move {
            let nested = PathUri::from_abs_path(&cwd).join("nested")?;
            filesystem
                .create_directory(
                    &nested,
                    CreateDirectoryOptions {
                        recursive: true,
                        follow_symlinks: true,
                    },
                    /*sandbox*/ None,
                )
                .await?;
            filesystem
                .write_file(
                    &nested.join("AGENTS.md")?,
                    CHILD_INSTRUCTIONS.as_bytes().to_vec(),
                    WriteFileOptions::default(),
                    /*sandbox*/ None,
                )
                .await?;
            Ok(())
        })
        .with_config(move |config| {
            config
                .features
                .enable(Feature::Collab)
                .expect("test config should allow feature update");
            if matches!(version, SpawnToolVersion::V2) {
                config
                    .features
                    .enable(Feature::MultiAgentV2)
                    .expect("test config should allow feature update");
            }
            config
                .features
                .disable(Feature::EnableRequestCompression)
                .expect("test config should allow feature update");
            config.approvals_reviewer = ApprovalsReviewer::AutoReview;
            config
                .permissions
                .set_permission_profile(PermissionProfile::read_only())
                .expect("test config should allow permission profile update");
        });
    let test = builder.build_with_auto_env(&server).await?;
    let parent_snapshot = test.codex.config_snapshot().await;
    let parent_selections = test.codex.environment_selections().await;
    let nested = test.workspace_path_uri("nested")?;
    let mut expected_selections = parent_selections.clone();
    if !matches!(cwd_input, CwdInput::Omitted) {
        expected_selections[0].cwd = nested.clone();
    }
    let mut spawn_args = json!({ "message": CHILD_PROMPT });
    match version {
        SpawnToolVersion::V1 => {
            if matches!(fork_mode, ForkMode::Full) {
                spawn_args["fork_context"] = json!(true);
            }
        }
        SpawnToolVersion::V2 => {
            spawn_args["task_name"] = json!("cwd_child");
            spawn_args["fork_turns"] = match fork_mode {
                ForkMode::Fresh => json!("none"),
                ForkMode::Truncated => json!("1"),
                ForkMode::Full => json!("all"),
            };
        }
    }
    match cwd_input {
        CwdInput::Relative => spawn_args["cwd"] = json!("nested"),
        CwdInput::Absolute => spawn_args["cwd"] = json!(nested.inferred_native_path_string()),
        CwdInput::Omitted => {}
    }
    let spawn_args = serde_json::to_string(&spawn_args)?;
    let spawn_event = match version {
        SpawnToolVersion::V1 => responses::ev_function_call_with_namespace(
            SPAWN_CALL_ID,
            "multi_agent_v1",
            "spawn_agent",
            &spawn_args,
        ),
        SpawnToolVersion::V2 => responses::ev_function_call_with_namespace(
            SPAWN_CALL_ID,
            "collaboration",
            "spawn_agent",
            &spawn_args,
        ),
    };
    responses::mount_sse_once_match(
        &server,
        |request: &wiremock::Request| body_contains(request, PARENT_PROMPT),
        responses::sse(vec![
            responses::ev_response_created("parent-spawn-response"),
            spawn_event,
            responses::ev_completed("parent-spawn-response"),
        ]),
    )
    .await;
    let child_uses_nested_instructions = !matches!(cwd_input, CwdInput::Omitted);
    let child_mock = responses::mount_sse_once_match(
        &server,
        move |request: &wiremock::Request| {
            body_contains(request, CHILD_PROMPT)
                && (!child_uses_nested_instructions || body_contains(request, CHILD_INSTRUCTIONS))
                && !body_contains(request, SPAWN_CALL_ID)
        },
        responses::sse(vec![
            responses::ev_response_created("child-response"),
            responses::ev_assistant_message("child-message", "child done"),
            responses::ev_completed("child-response"),
        ]),
    )
    .await;
    responses::mount_sse_once_match(
        &server,
        |request: &wiremock::Request| body_contains(request, SPAWN_CALL_ID),
        responses::sse(vec![
            responses::ev_response_created("parent-follow-up-response"),
            responses::ev_assistant_message("parent-follow-up-message", "spawned"),
            responses::ev_completed("parent-follow-up-response"),
        ]),
    )
    .await;

    let mut created_threads = test.thread_manager.subscribe_thread_created();
    test.submit_turn_with_approval_and_permission_profile(
        PARENT_PROMPT,
        parent_snapshot.approval_policy,
        parent_snapshot.permission_profile.clone(),
    )
    .await?;
    let child_thread_id = tokio::time::timeout(Duration::from_secs(10), created_threads.recv())
        .await
        .map_err(|_| anyhow!("timed out waiting for spawned child"))??;
    let child_thread = test.thread_manager.get_thread(child_thread_id).await?;
    let child_snapshot = child_thread.config_snapshot().await;
    let child_request = wait_for_request(&child_mock).await?;

    assert_eq!(
        child_snapshot.environment_selections(),
        expected_selections.as_slice()
    );
    assert_eq!(
        child_snapshot.workspace_roots,
        parent_snapshot.workspace_roots
    );
    assert_eq!(
        child_snapshot.profile_workspace_roots,
        parent_snapshot.profile_workspace_roots
    );
    assert_eq!(
        child_snapshot.permission_profile,
        parent_snapshot.permission_profile
    );
    assert_eq!(
        child_snapshot.approval_policy,
        parent_snapshot.approval_policy
    );
    assert_eq!(
        child_snapshot.approvals_reviewer,
        parent_snapshot.approvals_reviewer
    );
    assert_eq!(
        child_snapshot.approvals_reviewer,
        ApprovalsReviewer::AutoReview
    );
    assert_eq!(child_snapshot.full_access, parent_snapshot.full_access);
    assert!(!child_snapshot.full_access);
    assert_eq!(
        child_snapshot.active_permission_profile,
        parent_snapshot.active_permission_profile
    );
    let expected_cwd = expected_selections[0].cwd.inferred_native_path_string();
    assert!(child_request.body_contains_text(&expected_cwd));
    let expected_config_cwd = match cwd_input {
        CwdInput::Omitted => parent_snapshot.cwd().clone(),
        CwdInput::Relative | CwdInput::Absolute => test.config.cwd.join("nested"),
    };
    assert_eq!(child_snapshot.cwd(), &expected_config_cwd);
    if !matches!(cwd_input, CwdInput::Omitted) {
        assert_eq!(
            child_thread.instruction_sources().await,
            vec![nested.join("AGENTS.md")?]
        );
    }

    Ok(())
}

#[test_case("missing", "cannot access cwd"; "missing cwd")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spawn_agent_cwd_rejects_invalid_paths_without_creating_a_child(
    cwd: &str,
    expected_error: &str,
) -> Result<()> {
    let server = responses::start_mock_server().await;
    let spawn_args = serde_json::to_string(&json!({
        "message": CHILD_PROMPT,
        "task_name": "invalid_cwd_child",
        "cwd": cwd,
    }))?;
    responses::mount_sse_once(
        &server,
        responses::sse(vec![
            responses::ev_response_created("parent-spawn-response"),
            responses::ev_function_call_with_namespace(
                SPAWN_CALL_ID,
                "collaboration",
                "spawn_agent",
                &spawn_args,
            ),
            responses::ev_completed("parent-spawn-response"),
        ]),
    )
    .await;
    let result_mock = responses::mount_sse_once_match(
        &server,
        |request: &wiremock::Request| body_contains(request, SPAWN_CALL_ID),
        responses::sse(vec![
            responses::ev_response_created("parent-result-response"),
            responses::ev_assistant_message("parent-result-message", "handled"),
            responses::ev_completed("parent-result-response"),
        ]),
    )
    .await;
    let test = test_codex()
        .with_config(|config| {
            for feature in [Feature::Collab, Feature::MultiAgentV2] {
                config
                    .features
                    .enable(feature)
                    .expect("test config should allow feature update");
            }
        })
        .build_with_auto_env(&server)
        .await?;
    let thread_count = test.thread_manager.list_thread_ids().await.len();

    test.submit_turn(PARENT_PROMPT).await?;

    let output = result_mock
        .single_request()
        .function_call_output_text(SPAWN_CALL_ID)
        .expect("spawn failure should be returned to the model");
    assert!(
        output.contains(expected_error),
        "unexpected spawn error: {output}"
    );
    assert_eq!(
        test.thread_manager.list_thread_ids().await.len(),
        thread_count
    );

    Ok(())
}

async fn wait_for_request(mock: &ResponseMock) -> Result<responses::ResponsesRequest> {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if let Some(request) = mock.requests().into_iter().find(|request| {
                request.body_contains_text(CHILD_PROMPT)
                    && !request.body_contains_text(SPAWN_CALL_ID)
            }) {
                return request;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .map_err(|_| anyhow!("timed out waiting for child request"))
}

fn body_contains(request: &wiremock::Request, text: &str) -> bool {
    String::from_utf8(request.body.clone()).is_ok_and(|body| body.contains(text))
}
