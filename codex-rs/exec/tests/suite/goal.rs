use std::collections::BTreeSet;
use std::path::Path;
#[cfg(unix)]
use std::process::Stdio;
#[cfg(unix)]
use std::time::Duration;

use codex_protocol::ThreadId;
use codex_state::ThreadGoalStatus as StoredThreadGoalStatus;
use core_test_support::responses;
use core_test_support::test_codex_exec::TestCodexExecBuilder;
use core_test_support::test_codex_exec::test_codex_exec;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;

fn configure_goal_command(
    test: &TestCodexExecBuilder,
    server: &wiremock::MockServer,
) -> assert_cmd::Command {
    let mut command = test.cmd();
    let base_url = format!("{}/v1", server.uri());
    command
        .arg("--skip-git-repo-check")
        .arg("-c")
        .arg("features.goals=true")
        .arg("-c")
        .arg("model_provider=\"test\"")
        .arg("-c")
        .arg("model_providers.test.name=\"Test\"")
        .arg("-c")
        .arg(format!("model_providers.test.base_url={}", json!(base_url)));
    command
}

fn completion_responses(final_message: &str) -> Vec<String> {
    vec![
        responses::sse(vec![
            responses::ev_response_created("resp-goal-tool"),
            responses::ev_function_call(
                "call-complete-goal",
                "update_goal",
                r#"{"status":"complete"}"#,
            ),
            responses::ev_completed("resp-goal-tool"),
        ]),
        responses::sse(vec![
            responses::ev_response_created("resp-goal-final"),
            responses::ev_assistant_message("msg-goal-final", final_message),
            responses::ev_completed("resp-goal-final"),
        ]),
    ]
}

fn stdout_json_lines(output: &assert_cmd::assert::Assert) -> Vec<Value> {
    json_lines(&output.get_output().stdout)
}

fn json_lines(stdout: &[u8]) -> Vec<Value> {
    String::from_utf8_lossy(stdout)
        .lines()
        .map(|line| serde_json::from_str(line).expect("exec stdout must be JSONL"))
        .collect()
}

fn assert_no_literal_goal_prompt(response_mock: &responses::ResponseMock, literal_prompt: &str) {
    let requests = response_mock.requests();
    assert!(
        !requests.is_empty(),
        "goal should issue a Responses request"
    );
    assert!(
        requests
            .iter()
            .all(|request| !request.body_contains_text(literal_prompt)),
        "literal startup slash text must not reach the model: {requests:?}"
    );
}

async fn stored_goal(home: &Path, thread_id: &str) -> anyhow::Result<codex_state::ThreadGoal> {
    let runtime = codex_state::StateRuntime::init(home.to_path_buf(), "openai".to_string()).await?;
    runtime
        .thread_goals()
        .get_thread_goal(ThreadId::from_string(thread_id)?)
        .await?
        .ok_or_else(|| anyhow::anyhow!("stored goal is missing"))
}

async fn assert_ordinary_exec_prompt_is_unchanged(prompt: &str) -> anyhow::Result<()> {
    let test = test_codex_exec();
    let server = responses::start_mock_server().await;
    let response_mock = responses::mount_sse_once(
        &server,
        responses::sse(vec![
            responses::ev_response_created("resp-ordinary"),
            responses::ev_assistant_message("msg-ordinary", "ordinary response"),
            responses::ev_completed("resp-ordinary"),
        ]),
    )
    .await;

    let mut command = configure_goal_command(&test, &server);
    let output = command.arg("--json").arg(prompt).assert().success();
    let events = stdout_json_lines(&output);
    let thread_id = events
        .first()
        .and_then(|event| event["thread_id"].as_str())
        .expect("thread.started thread id");
    let normalized_stdout =
        String::from_utf8_lossy(&output.get_output().stdout).replace(thread_id, "<thread-id>");
    assert_eq!(
        normalized_stdout,
        concat!(
            "{\"type\":\"thread.started\",\"thread_id\":\"<thread-id>\"}\n",
            "{\"type\":\"turn.started\"}\n",
            "{\"type\":\"item.completed\",\"item\":{\"id\":\"item_0\",\"type\":\"agent_message\",\"text\":\"ordinary response\"}}\n",
            "{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":0,\"cached_input_tokens\":0,\"output_tokens\":0,\"reasoning_output_tokens\":0}}\n",
        )
    );

    let request = response_mock.single_request();
    assert_eq!(
        request.input().last(),
        Some(&json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}],
        }))
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ordinary_text_retains_single_turn_request_and_jsonl() -> anyhow::Result<()> {
    assert_ordinary_exec_prompt_is_unchanged("ordinary prompt").await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn goalkeeper_text_retains_single_turn_request_and_jsonl() -> anyhow::Result<()> {
    assert_ordinary_exec_prompt_is_unchanged("/goalkeeper defend this input").await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn other_slash_text_retains_single_turn_request_and_jsonl() -> anyhow::Result<()> {
    assert_ordinary_exec_prompt_is_unchanged("/status keep this literal").await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn leading_space_goal_retains_single_turn_request_and_jsonl() -> anyhow::Result<()> {
    assert_ordinary_exec_prompt_is_unchanged(" /goal keep this literal").await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exec_goal_follows_multiple_turns_and_emits_ordered_json() -> anyhow::Result<()> {
    let test = test_codex_exec();
    let server = responses::start_mock_server().await;
    let response_mock = responses::mount_sse_sequence(
        &server,
        vec![
            responses::sse(vec![
                responses::ev_response_created("resp-goal-first"),
                responses::ev_assistant_message("msg-goal-first", "first continuation"),
                responses::ev_completed("resp-goal-first"),
            ]),
            completion_responses("migration finished").remove(0),
            completion_responses("migration finished").remove(1),
        ],
    )
    .await;
    let last_message_path = test.cwd_path().join("last-message.txt");
    let literal_prompt = "/goal finish the migration";

    let mut command = configure_goal_command(&test, &server);
    let output = command
        .arg("--json")
        .arg("--output-last-message")
        .arg(&last_message_path)
        .arg(literal_prompt)
        .assert()
        .success();
    let events = stdout_json_lines(&output);
    let lifecycle = events
        .iter()
        .filter(|event| {
            matches!(
                event.get("type").and_then(Value::as_str),
                Some("thread.started" | "turn.started" | "turn.completed" | "goal.updated")
            )
        })
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(
        lifecycle
            .iter()
            .map(|event| event["type"].as_str().expect("event type"))
            .collect::<Vec<_>>(),
        vec![
            "thread.started",
            "goal.updated",
            "turn.started",
            "turn.completed",
            "goal.updated",
            "turn.started",
            "turn.completed",
            "goal.updated",
        ]
    );
    let thread_id = lifecycle[0]["thread_id"]
        .as_str()
        .expect("thread.started thread id");
    let goal_events = lifecycle
        .iter()
        .filter(|event| event["type"] == "goal.updated")
        .collect::<Vec<_>>();
    assert_eq!(goal_events.len(), 3);
    assert_eq!(goal_events[0]["goal"]["status"], "active");
    assert_eq!(goal_events[1]["goal"]["status"], "active");
    assert_eq!(goal_events[2]["goal"]["status"], "complete");
    for event in goal_events {
        assert_eq!(event["goal"]["thread_id"], thread_id);
        assert_eq!(event["goal"]["objective"], "finish the migration");
        assert_eq!(
            event["goal"]
                .as_object()
                .expect("goal object")
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([
                "created_at",
                "objective",
                "status",
                "thread_id",
                "time_used_seconds",
                "token_budget",
                "tokens_used",
                "updated_at",
            ])
        );
    }
    assert_eq!(
        std::fs::read_to_string(&last_message_path)?,
        "migration finished"
    );

    let requests = response_mock.requests();
    assert_eq!(requests.len(), 3);
    assert!(requests[0].body_contains_text("finish the migration"));
    assert_no_literal_goal_prompt(&response_mock, literal_prompt);
    assert!(
        response_mock
            .function_call_output_text("call-complete-goal")
            .is_some_and(|output| output.contains("complete"))
    );

    let goal = stored_goal(test.home_path(), thread_id).await?;
    assert_eq!(goal.objective, "finish the migration");
    assert_eq!(goal.status, StoredThreadGoalStatus::Complete);
    Ok(())
}

#[derive(Clone, Copy)]
enum StdinGoalSource {
    ExplicitDash,
    Implicit,
}

async fn assert_exec_goal_from_stdin(source: StdinGoalSource) -> anyhow::Result<()> {
    let test = test_codex_exec();
    let server = responses::start_mock_server().await;
    let response_mock =
        responses::mount_sse_sequence(&server, completion_responses("stdin goal complete")).await;
    let literal_prompt = "/goal finish from stdin\n";
    let mut command = configure_goal_command(&test, &server);
    match source {
        StdinGoalSource::ExplicitDash => {
            command.arg("-");
        }
        StdinGoalSource::Implicit => {}
    }
    command
        .write_stdin(literal_prompt)
        .assert()
        .success()
        .stderr(predicates::str::contains("goal: complete"));

    assert_eq!(response_mock.requests().len(), 2);
    assert_no_literal_goal_prompt(&response_mock, literal_prompt.trim_end());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exec_goal_accepts_explicit_stdin() -> anyhow::Result<()> {
    assert_exec_goal_from_stdin(StdinGoalSource::ExplicitDash).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exec_goal_accepts_implicit_stdin() -> anyhow::Result<()> {
    assert_exec_goal_from_stdin(StdinGoalSource::Implicit).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn positional_goal_preserves_appended_piped_stdin_in_objective() -> anyhow::Result<()> {
    let test = test_codex_exec();
    let server = responses::start_mock_server().await;
    let response_mock =
        responses::mount_sse_sequence(&server, completion_responses("combined goal complete"))
            .await;
    let mut command = configure_goal_command(&test, &server);
    let output = command
        .arg("--json")
        .arg("/goal summarize the input")
        .write_stdin("piped context\n")
        .assert()
        .success();
    let events = stdout_json_lines(&output);
    let objective = events
        .iter()
        .find(|event| event["type"] == "goal.updated")
        .and_then(|event| event["goal"]["objective"].as_str())
        .expect("goal objective event");
    assert_eq!(
        objective,
        "summarize the input\n\n<stdin>\npiped context\n</stdin>"
    );
    assert_no_literal_goal_prompt(&response_mock, "/goal summarize the input");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn blocked_goal_exits_nonzero_without_clearing_goal() -> anyhow::Result<()> {
    let test = test_codex_exec();
    let server = responses::start_mock_server().await;
    let response_mock = responses::mount_sse_sequence(
        &server,
        vec![
            responses::sse(vec![
                responses::ev_response_created("resp-goal-block"),
                responses::ev_function_call(
                    "call-block-goal",
                    "update_goal",
                    r#"{"status":"blocked"}"#,
                ),
                responses::ev_completed("resp-goal-block"),
            ]),
            responses::sse(vec![
                responses::ev_response_created("resp-goal-block-final"),
                responses::ev_assistant_message("msg-goal-block", "goal blocked"),
                responses::ev_completed("resp-goal-block-final"),
            ]),
        ],
    )
    .await;
    let mut command = configure_goal_command(&test, &server);
    let output = command
        .arg("--json")
        .arg("/goal wait for an external dependency")
        .assert()
        .failure();
    let events = stdout_json_lines(&output);
    let thread_id = events
        .iter()
        .find(|event| event["type"] == "thread.started")
        .and_then(|event| event["thread_id"].as_str())
        .expect("thread id");
    assert_eq!(
        events
            .iter()
            .rfind(|event| event["type"] == "goal.updated")
            .expect("terminal goal update")["goal"]["status"],
        "blocked"
    );
    assert_eq!(response_mock.requests().len(), 2);
    let goal = stored_goal(test.home_path(), thread_id).await?;
    assert_eq!(goal.status, StoredThreadGoalStatus::Blocked);
    Ok(())
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ctrl_c_pauses_goal_and_prevents_another_continuation() -> anyhow::Result<()> {
    use wiremock::Mock;
    use wiremock::ResponseTemplate;
    use wiremock::matchers::method;
    use wiremock::matchers::path;

    let test = test_codex_exec();
    let server = responses::start_mock_server().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_delay(Duration::from_secs(/*secs*/ 5))
                .set_body_string(responses::sse(vec![
                    responses::ev_response_created("resp-interrupted-goal"),
                    responses::ev_completed("resp-interrupted-goal"),
                ])),
        )
        .mount(&server)
        .await;

    let mut command = configure_goal_command(&test, &server);
    command
        .arg("--json")
        .arg("/goal keep running until interrupted");
    let mut process = std::process::Command::new(command.get_program());
    process.args(command.get_args());
    if let Some(current_dir) = command.get_current_dir() {
        process.current_dir(current_dir);
    }
    for (name, value) in command.get_envs() {
        if let Some(value) = value {
            process.env(name, value);
        } else {
            process.env_remove(name);
        }
    }
    let child = process
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut saw_responses_request = false;
    for _ in 0..400 {
        let requests = server.received_requests().await.unwrap_or_default();
        if requests.iter().any(|request| {
            request.method.as_str() == "POST" && request.url.path() == "/v1/responses"
        }) {
            saw_responses_request = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(/*millis*/ 100)).await;
    }
    assert!(
        saw_responses_request,
        "goal turn never reached Responses API"
    );

    // SAFETY: `child.id()` is the live child process created immediately above, and SIGINT does
    // not require ownership of memory in that process.
    let signal_result = unsafe { libc::kill(child.id() as i32, libc::SIGINT) };
    assert_eq!(signal_result, 0);
    let output = tokio::task::spawn_blocking(move || child.wait_with_output()).await??;
    assert!(!output.status.success());

    let events = json_lines(&output.stdout);
    let thread_id = events
        .iter()
        .find(|event| event["type"] == "thread.started")
        .and_then(|event| event["thread_id"].as_str())
        .unwrap_or_else(|| {
            panic!(
                "thread id missing after Ctrl-C; stdout: {}; stderr: {}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
        });
    assert_eq!(
        events
            .iter()
            .rfind(|event| event["type"] == "goal.updated")
            .expect("paused goal event")["goal"]["status"],
        "paused"
    );
    let goal = stored_goal(test.home_path(), thread_id).await?;
    assert_eq!(goal.status, StoredThreadGoalStatus::Paused);
    let responses_requests = server
        .received_requests()
        .await
        .unwrap_or_default()
        .into_iter()
        .filter(|request| {
            request.method.as_str() == "POST" && request.url.path() == "/v1/responses"
        })
        .count();
    assert_eq!(responses_requests, 1);
    Ok(())
}

struct RejectionCase {
    args: Vec<String>,
    expected_error: &'static str,
    goals_enabled: bool,
    create_image: bool,
    create_schema: bool,
}

fn assert_goal_preflight_rejection(case: RejectionCase) -> anyhow::Result<()> {
    let test = test_codex_exec();
    let image_path = test.cwd_path().join("image.png");
    let schema_path = test.cwd_path().join("schema.json");
    if case.create_image {
        std::fs::write(&image_path, b"png")?;
    }
    if case.create_schema {
        std::fs::write(&schema_path, b"not valid json")?;
    }
    let mut command = test.cmd();
    command
        .arg("--skip-git-repo-check")
        .arg("-c")
        .arg(format!("features.goals={}", case.goals_enabled));
    for arg in case.args {
        command.arg(match arg.as_str() {
            "IMAGE" => image_path.as_os_str(),
            "SCHEMA" => schema_path.as_os_str(),
            _ => arg.as_ref(),
        });
    }
    command
        .assert()
        .failure()
        .stderr(predicates::str::contains(case.expected_error));
    assert!(
        !test.home_path().join("sessions").exists(),
        "rejected goal invocation created a session"
    );
    Ok(())
}

macro_rules! goal_preflight_rejection_test {
    (
        $name:ident,
        args: $args:expr,
        expected_error: $expected_error:literal,
        goals_enabled: $goals_enabled:expr,
        create_image: $create_image:expr,
        create_schema: $create_schema:expr $(,)?
    ) => {
        #[test]
        fn $name() -> anyhow::Result<()> {
            assert_goal_preflight_rejection(RejectionCase {
                args: $args,
                expected_error: $expected_error,
                goals_enabled: $goals_enabled,
                create_image: $create_image,
                create_schema: $create_schema,
            })
        }
    };
}

goal_preflight_rejection_test!(
    empty_goal_preflight_rejection_creates_no_session,
    args: vec!["/goal".to_string()],
    expected_error: "requires a non-empty objective",
    goals_enabled: true,
    create_image: false,
    create_schema: false,
);
goal_preflight_rejection_test!(
    disabled_goal_preflight_rejection_creates_no_session,
    args: vec!["/goal disabled".to_string()],
    expected_error: "Goals feature is disabled",
    goals_enabled: false,
    create_image: false,
    create_schema: false,
);
goal_preflight_rejection_test!(
    ephemeral_goal_preflight_rejection_creates_no_session,
    args: vec!["--ephemeral".to_string(), "/goal persisted".to_string()],
    expected_error: "cannot be used with `--ephemeral`",
    goals_enabled: true,
    create_image: false,
    create_schema: false,
);
goal_preflight_rejection_test!(
    image_goal_preflight_rejection_creates_no_session,
    args: vec![
        "/goal image".to_string(),
        "--image".to_string(),
        "IMAGE".to_string(),
    ],
    expected_error: "does not support `--image`",
    goals_enabled: true,
    create_image: true,
    create_schema: false,
);
goal_preflight_rejection_test!(
    output_schema_goal_preflight_rejection_creates_no_session,
    args: vec![
        "--output-schema".to_string(),
        "SCHEMA".to_string(),
        "/goal schema".to_string(),
    ],
    expected_error: "cannot be used with `--output-schema`",
    goals_enabled: true,
    create_image: false,
    create_schema: true,
);
goal_preflight_rejection_test!(
    oversized_goal_preflight_rejection_creates_no_session,
    args: vec![format!("/goal {}", "x".repeat(4_001))],
    expected_error: "the maximum is 4000",
    goals_enabled: true,
    create_image: false,
    create_schema: false,
);
goal_preflight_rejection_test!(
    resume_goal_preflight_rejection_creates_no_session,
    args: vec![
        "resume".to_string(),
        "--last".to_string(),
        "/goal replace existing".to_string(),
    ],
    expected_error: "not supported with `codex exec resume`",
    goals_enabled: true,
    create_image: false,
    create_schema: false,
);
