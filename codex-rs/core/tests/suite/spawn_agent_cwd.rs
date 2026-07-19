use anyhow::Result;
use anyhow::anyhow;
use codex_exec_server::CreateDirectoryOptions;
use codex_features::Feature;
use codex_utils_path_uri::PathUri;
use core_test_support::responses;
use core_test_support::responses::ResponseMock;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;
use serde_json::json;
use std::time::Duration;
use test_case::test_case;

const PARENT_PROMPT: &str = "spawn a child in the nested workspace";
const CHILD_PROMPT: &str = "inspect the selected working directory";
const CHILD_INSTRUCTIONS: &str = "CHILD_CWD_AGENTS_INSTRUCTIONS";
const SPAWN_CALL_ID: &str = "spawn-child-cwd";

#[derive(Clone, Copy, Debug)]
enum SpawnToolVersion {
    V1,
    V2,
}

#[test_case(SpawnToolVersion::V1; "v1 fresh child")]
#[test_case(SpawnToolVersion::V2; "v2 full history child")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spawn_agent_cwd_selects_child_directory_without_widening_permissions(
    version: SpawnToolVersion,
) -> Result<()> {
    let server = responses::start_mock_server().await;
    let mut spawn_args = json!({
        "message": CHILD_PROMPT,
        "cwd": "nested",
    });
    if matches!(version, SpawnToolVersion::V2) {
        spawn_args["task_name"] = json!("cwd_child");
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
        |request: &wiremock::Request| request_body_contains(request, PARENT_PROMPT),
        responses::sse(vec![
            responses::ev_response_created("parent-spawn-response"),
            spawn_event,
            responses::ev_completed("parent-spawn-response"),
        ]),
    )
    .await;
    let child_mock = responses::mount_sse_once_match(
        &server,
        |request: &wiremock::Request| {
            request_body_contains(request, CHILD_PROMPT)
                && request_body_contains(request, CHILD_INSTRUCTIONS)
                && !request_body_contains(request, SPAWN_CALL_ID)
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
        |request: &wiremock::Request| request_body_contains(request, SPAWN_CALL_ID),
        responses::sse(vec![
            responses::ev_response_created("parent-follow-up-response"),
            responses::ev_assistant_message("parent-follow-up-message", "spawned"),
            responses::ev_completed("parent-follow-up-response"),
        ]),
    )
    .await;

    let mut builder = test_codex()
        .with_workspace_setup(|cwd, filesystem| async move {
            let root = PathUri::from_abs_path(&cwd);
            let git_dir = root.join(".git")?;
            let nested = root.join("nested")?;
            filesystem
                .create_directory(
                    &git_dir,
                    CreateDirectoryOptions { recursive: true },
                    /*sandbox*/ None,
                )
                .await?;
            filesystem
                .create_directory(
                    &nested,
                    CreateDirectoryOptions { recursive: true },
                    /*sandbox*/ None,
                )
                .await?;
            filesystem
                .write_file(
                    &nested.join("AGENTS.md")?,
                    CHILD_INSTRUCTIONS.as_bytes().to_vec(),
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
        });
    let test = builder.build_with_auto_env(&server).await?;
    let parent_snapshot = test.codex.config_snapshot().await;
    let parent_selections = test.codex.environment_selections().await;
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
    let mut expected_selections = parent_selections;
    expected_selections[0].cwd = expected_selections[0].cwd.join("nested")?;

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
        child_thread.instruction_sources().await,
        vec![expected_selections[0].cwd.join("AGENTS.md")?]
    );
    let expected_cwd = expected_selections[0].cwd.inferred_native_path_string();
    assert!(
        child_request.body_contains_text(&expected_cwd),
        "child request should advertise {expected_cwd}; user inputs: {:?}",
        child_request.message_input_texts("user")
    );

    Ok(())
}

async fn wait_for_request(mock: &ResponseMock) -> Result<responses::ResponsesRequest> {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if let Some(request) = mock.requests().into_iter().find(|request| {
                request.body_contains_text(CHILD_PROMPT)
                    && request.body_contains_text(CHILD_INSTRUCTIONS)
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

fn request_body_contains(request: &wiremock::Request, text: &str) -> bool {
    String::from_utf8(request.body.clone()).is_ok_and(|body| body.contains(text))
}
