use super::*;
use pretty_assertions::assert_eq;

const OPENAI_MODEL: &str = "gpt-5.6-sol";
const ROOT_SPAWN_CALL: &str = "root-spawn";
const CHILD_PLAN_CALL: &str = "child-plan";
const CHILD_TASK_REMINDER: &str = "The most recent agent message after inherited history contains your assigned child task. Treat that task as authoritative and inherited parent prompts as context only. Do not continue or repeat the parent's collaboration actions unless the assigned child task explicitly requests them.";

struct CollaborationResponder {
    root_prompt: &'static str,
    child_task: &'static str,
    spawn_arguments: String,
    child_final: &'static str,
}

impl Respond for CollaborationResponder {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = request.body_json().expect("collaboration request JSON");
        let events = if message_contains(&body, self.child_task) {
            if has_function_output(&body, CHILD_PLAN_CALL) {
                vec![
                    responses::ev_response_created("child-final-response"),
                    responses::ev_assistant_message("child-final-message", self.child_final),
                    responses::ev_completed("child-final-response"),
                ]
            } else {
                vec![
                    responses::ev_response_created("child-tool-response"),
                    responses::ev_function_call(
                        CHILD_PLAN_CALL,
                        "update_plan",
                        r#"{"explanation":"child tool round trip","plan":[{"step":"Finish","status":"completed"}]}"#,
                    ),
                    responses::ev_completed("child-tool-response"),
                ]
            }
        } else {
            assert!(message_contains(&body, self.root_prompt));
            if has_function_output(&body, ROOT_SPAWN_CALL) {
                vec![
                    responses::ev_response_created("root-final-response"),
                    responses::ev_assistant_message("root-final-message", "root complete"),
                    responses::ev_completed("root-final-response"),
                ]
            } else {
                vec![
                    responses::ev_response_created("root-spawn-response"),
                    responses::ev_function_call(
                        ROOT_SPAWN_CALL,
                        "spawn_agent",
                        &self.spawn_arguments,
                    ),
                    responses::ev_completed("root-spawn-response"),
                ]
            }
        };
        local_sse(events)
    }
}

fn message_contains(body: &Value, needle: &str) -> bool {
    body["input"].as_array().is_some_and(|input| {
        input
            .iter()
            .any(|item| item["type"] != "function_call" && item.to_string().contains(needle))
    })
}

fn message_index(body: &Value, needle: &str) -> usize {
    body["input"]
        .as_array()
        .and_then(|input| {
            input.iter().position(|item| {
                item["type"] != "function_call" && item.to_string().contains(needle)
            })
        })
        .unwrap_or_else(|| panic!("no input message contained {needle:?}"))
}

fn has_function_output(body: &Value, call_id: &str) -> bool {
    body["input"].as_array().is_some_and(|input| {
        input
            .iter()
            .any(|item| item["type"] == "function_call_output" && item["call_id"] == call_id)
    })
}

async fn mount_collaboration_endpoint(
    server: &MockServer,
    request_path: &str,
    root_prompt: &'static str,
    child_task: &'static str,
    spawn_arguments: String,
    child_final: &'static str,
    expected_requests: u64,
) {
    Mock::given(method("POST"))
        .and(path(request_path))
        .respond_with(CollaborationResponder {
            root_prompt,
            child_task,
            spawn_arguments,
            child_final,
        })
        .up_to_n_times(expected_requests)
        .mount(server)
        .await;
}

fn message_request<'a>(requests: &'a [Request], needle: &str) -> &'a Request {
    requests
        .iter()
        .find(|request| {
            request
                .body_json::<Value>()
                .is_ok_and(|body| message_contains(&body, needle))
        })
        .unwrap_or_else(|| panic!("no request contained message {needle:?}"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn local_child_obeys_task_reminder_after_inherited_root_spawn() -> Result<()> {
    skip_if_no_network!(Ok(()));
    const ROOT_PROMPT: &str =
        "Spawn a same-local Qwen child with fork_turns=all and have it create the assigned marker";
    const CHILD_TASK: &str = "Create LOCAL_QWEN_CHILD_TOOL=JADE-518 with update_plan, do not spawn another agent, and return the exact marker";
    const CHILD_FINAL: &str = "LOCAL_QWEN_CHILD_TOOL=JADE-518";
    let server = MockServer::start().await;
    mount_ready(&server).await;
    Mock::given(method("POST"))
        .and(path("/v1/responses/input_tokens"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"input_tokens": 50})))
        .mount(&server)
        .await;
    mount_collaboration_endpoint(
        &server,
        "/v1/responses",
        ROOT_PROMPT,
        CHILD_TASK,
        json!({
            "plaintext_message": CHILD_TASK,
            "task_name": "local_worker",
            "model": LOCAL_MODEL,
            "fork_turns": "all",
        })
        .to_string(),
        CHILD_FINAL,
        4,
    )
    .await;
    let test = local_builder(&server).build_with_auto_env(&server).await?;
    let mut created_threads = test.thread_manager.subscribe_thread_created();

    submit_turn(&test, ROOT_PROMPT).await?;
    let child_thread_id =
        tokio::time::timeout(Duration::from_secs(10), created_threads.recv()).await??;
    wait_for_child_completion(&test, child_thread_id, CHILD_FINAL).await?;

    let requests = received(&server, "/v1/responses").await;
    let child: Value = message_request(&requests, CHILD_TASK).body_json()?;
    assert!(message_contains(&child, ROOT_PROMPT));
    assert!(message_contains(&child, CHILD_TASK_REMINDER));
    let root_prompt_index = message_index(&child, ROOT_PROMPT);
    let reminder_index = message_index(&child, CHILD_TASK_REMINDER);
    let child_task_index = message_index(&child, CHILD_TASK);
    assert!(root_prompt_index < reminder_index);
    assert!(reminder_index < child_task_index);
    let child_input = child["input"].as_array().expect("child input array");
    assert_eq!(child_input[reminder_index]["role"], "user");
    assert!(
        child_input[reminder_index]
            .to_string()
            .contains("<external_local_developer_update>")
    );
    assert!(requests.iter().any(|request| {
        request.body_json::<Value>().is_ok_and(|body| {
            message_contains(&body, CHILD_TASK) && has_function_output(&body, CHILD_PLAN_CALL)
        })
    }));
    assert_eq!(
        requests
            .iter()
            .filter(|request| {
                request
                    .body_json::<Value>()
                    .is_ok_and(|body| message_contains(&body, CHILD_TASK))
            })
            .count(),
        2,
        "the child should make only its assigned tool turn and final turn"
    );
    assert_eq!(
        requests
            .iter()
            .filter_map(|request| request.body_json::<Value>().ok())
            .filter(|body| message_contains(body, CHILD_TASK))
            .flat_map(|body| {
                body["input"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|item| item["type"] == "function_call")
                    .filter_map(|item| item["name"].as_str())
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
        vec!["update_plan".to_string()],
        "the inherited root spawn action must not be replayed by the child"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn local_root_spawns_history_isolated_openai_tool_using_child() -> Result<()> {
    skip_if_no_network!(Ok(()));
    const ROOT_PROMPT: &str = "Spawn an OpenAI worker";
    const CHILD_TASK: &str = "OpenAI child task";
    const CHILD_FINAL: &str = "OpenAI child complete";
    let server = MockServer::start().await;
    mount_ready(&server).await;
    Mock::given(method("POST"))
        .and(path("/v1/responses/input_tokens"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"input_tokens": 50})))
        .mount(&server)
        .await;
    mount_collaboration_endpoint(
        &server,
        "/v1/responses",
        ROOT_PROMPT,
        "unused local child",
        json!({
            "plaintext_message": CHILD_TASK,
            "task_name": "openai_worker",
            "model": OPENAI_MODEL,
            "fork_turns": "none",
        })
        .to_string(),
        "unused",
        2,
    )
    .await;
    mount_collaboration_endpoint(
        &server,
        "/responses",
        "unused OpenAI root",
        CHILD_TASK,
        "{}".to_string(),
        CHILD_FINAL,
        2,
    )
    .await;
    let test = local_builder(&server).build_with_auto_env(&server).await?;
    let mut created_threads = test.thread_manager.subscribe_thread_created();

    submit_turn(&test, ROOT_PROMPT).await?;
    let child_thread_id =
        tokio::time::timeout(Duration::from_secs(10), created_threads.recv()).await??;
    wait_for_child_completion(&test, child_thread_id, CHILD_FINAL).await?;

    let requests = received(&server, "/responses").await;
    let child: Value = message_request(&requests, CHILD_TASK).body_json()?;
    assert!(!message_contains(&child, ROOT_PROMPT));
    assert!(requests.iter().any(|request| {
        request
            .body_json::<Value>()
            .is_ok_and(|body| has_function_output(&body, CHILD_PLAN_CALL))
    }));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn openai_root_spawns_history_isolated_local_tool_using_child() -> Result<()> {
    skip_if_no_network!(Ok(()));
    const ROOT_PROMPT: &str = "OpenAI root spawning local";
    const CHILD_TASK: &str = "Local child from OpenAI task";
    const CHILD_FINAL: &str = "mixed local child complete";
    let server = MockServer::start().await;
    mount_ready(&server).await;
    Mock::given(method("POST"))
        .and(path("/v1/responses/input_tokens"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"input_tokens": 50})))
        .mount(&server)
        .await;
    mount_collaboration_endpoint(
        &server,
        "/responses",
        ROOT_PROMPT,
        "unused OpenAI child",
        json!({
            "plaintext_message": CHILD_TASK,
            "task_name": "local_worker",
            "model": LOCAL_MODEL,
            "fork_turns": "none",
        })
        .to_string(),
        "unused",
        2,
    )
    .await;
    mount_collaboration_endpoint(
        &server,
        "/v1/responses",
        "unused local root",
        CHILD_TASK,
        "{}".to_string(),
        CHILD_FINAL,
        2,
    )
    .await;
    let test = local_builder(&server)
        .with_config(|config| config.model = Some(OPENAI_MODEL.to_string()))
        .build_with_auto_env(&server)
        .await?;
    let mut created_threads = test.thread_manager.subscribe_thread_created();

    submit_turn(&test, ROOT_PROMPT).await?;
    let child_thread_id =
        tokio::time::timeout(Duration::from_secs(10), created_threads.recv()).await??;
    wait_for_child_completion(&test, child_thread_id, CHILD_FINAL).await?;

    let requests = received(&server, "/v1/responses").await;
    let child: Value = message_request(&requests, CHILD_TASK).body_json()?;
    assert!(!message_contains(&child, ROOT_PROMPT));
    assert!(requests.iter().any(|request| {
        request
            .body_json::<Value>()
            .is_ok_and(|body| has_function_output(&body, CHILD_PLAN_CALL))
    }));
    Ok(())
}
