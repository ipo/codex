use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::user_input::UserInput;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_function_call;
use core_test_support::responses::ev_message_item_added;
use core_test_support::responses::ev_output_text_delta;
use core_test_support::responses::ev_response_created;
use core_test_support::responses::mount_sse_sequence;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::TestCodex;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;

async fn submit_turn(test: &TestCodex, text: &str) -> anyhow::Result<()> {
    test.codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: text.to_string(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    Ok(())
}

fn incomplete(id: &str, reason: &str) -> Value {
    json!({
        "type": "response.incomplete",
        "response": {
            "id": id,
            "incomplete_details": { "reason": reason }
        }
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_terminal_cannot_rehabilitate_attempt_or_start_its_tool() -> anyhow::Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let failed_arguments = json!({"sleep_after_ms": 1, "secret": "failed-tool-args"}).to_string();
    let failed = sse(vec![
        ev_response_created("failed-response"),
        ev_message_item_added("failed-message", ""),
        ev_output_text_delta("failed-visible-delta"),
        ev_assistant_message("failed-message", "failed-history-text"),
        ev_function_call("failed-call-id", "test_sync_tool", &failed_arguments),
        incomplete("failed-response", "future_reason"),
        ev_completed("must-not-rehabilitate"),
    ]);
    let succeeded = sse(vec![
        ev_response_created("successful-response"),
        ev_assistant_message("successful-message", "retry succeeded"),
        ev_completed("successful-response"),
    ]);
    let response_mock = mount_sse_sequence(&server, vec![failed, succeeded]).await;
    let mut builder = test_codex().with_model("test-gpt-5.1-codex");
    let test = builder.build_with_auto_env(&server).await?;

    submit_turn(&test, "stable user input").await?;
    wait_for_event(&test.codex, |event| {
        matches!(
            event,
            EventMsg::AgentMessageContentDelta(delta)
                if delta.delta == "failed-visible-delta"
        )
    })
    .await;
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;

    let requests = response_mock.requests();
    assert_eq!(requests.len(), 2);
    let retry = &requests[1];
    assert!(retry.body_contains_text("stable user input"));
    for discarded in [
        "failed-history-text",
        "failed-call-id",
        "failed-tool-args",
        "failed-visible-delta",
    ] {
        assert!(!retry.body_contains_text(discarded), "leaked {discarded}");
    }
    assert!(!retry.has_function_call("failed-call-id"));
    assert_eq!(
        response_mock.function_call_output_text("failed-call-id"),
        None
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn valid_terminal_commits_parallel_tools_once_in_order() -> anyhow::Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let first = sse(vec![
        ev_response_created("tool-response"),
        ev_function_call("call-one", "test_sync_tool", "{}"),
        ev_function_call("call-two", "test_sync_tool", "{}"),
        ev_completed("tool-response"),
    ]);
    let second = sse(vec![
        ev_assistant_message("final-message", "tools completed"),
        ev_completed("final-response"),
    ]);
    let response_mock = mount_sse_sequence(&server, vec![first, second]).await;
    let mut builder = test_codex().with_model("test-gpt-5.1-codex");
    let test = builder.build_with_auto_env(&server).await?;

    submit_turn(&test, "run two tools").await?;
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;

    let requests = response_mock.requests();
    assert_eq!(requests.len(), 2);
    let ordered_types_and_calls = requests[1]
        .input()
        .into_iter()
        .filter_map(|item| {
            let kind = item.get("type")?.as_str()?;
            let call_id = item.get("call_id")?.as_str()?;
            Some((kind.to_string(), call_id.to_string()))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        ordered_types_and_calls,
        vec![
            ("function_call".to_string(), "call-one".to_string()),
            ("function_call".to_string(), "call-two".to_string()),
            ("function_call_output".to_string(), "call-one".to_string()),
            ("function_call_output".to_string(), "call-two".to_string()),
        ]
    );
    assert_eq!(
        requests[1].function_call_output_content_and_success("call-one"),
        Some((Some("ok".to_string()), None))
    );
    assert_eq!(
        requests[1].function_call_output_content_and_success("call-two"),
        Some((Some("ok".to_string()), None))
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn discarding_terminals_are_visible_nonretryable_and_leave_thread_reusable()
-> anyhow::Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let output_exhausted = sse(vec![
        ev_message_item_added("exhausted-message", ""),
        ev_output_text_delta("discarded exhausted delta"),
        ev_assistant_message("exhausted-message", "discarded exhausted history"),
        incomplete("exhausted-response", "max_output_tokens"),
    ]);
    let after_exhaustion = sse(vec![
        ev_assistant_message("after-exhaustion", "thread still works"),
        ev_completed("after-exhaustion-response"),
    ]);
    let refused = sse(vec![
        ev_assistant_message("refused-message", "discarded refusal history"),
        incomplete("refused-response", "content_filter"),
    ]);
    let after_refusal = sse(vec![
        ev_assistant_message("after-refusal", "thread works again"),
        ev_completed("after-refusal-response"),
    ]);
    let response_mock = mount_sse_sequence(
        &server,
        vec![output_exhausted, after_exhaustion, refused, after_refusal],
    )
    .await;
    let mut builder = test_codex()
        .with_model("test-gpt-5.1-codex")
        .with_config(|config| config.model_provider.stream_max_retries = Some(2));
    let test = builder.build_with_auto_env(&server).await?;

    submit_turn(&test, "exhaust output").await?;
    let exhausted_error =
        wait_for_event(&test.codex, |event| matches!(event, EventMsg::Error(_))).await;
    assert!(matches!(
        exhausted_error,
        EventMsg::Error(error)
            if error.message == "model output limit reached before the turn completed"
    ));
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;

    submit_turn(&test, "new input after exhaustion").await?;
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;

    submit_turn(&test, "trigger safety refusal").await?;
    let refusal_error =
        wait_for_event(&test.codex, |event| matches!(event, EventMsg::Error(_))).await;
    assert!(matches!(
        refusal_error,
        EventMsg::Error(error)
            if error.message == "model refused to complete the turn for safety reasons"
    ));
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;

    submit_turn(&test, "new input after refusal").await?;
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;

    let requests = response_mock.requests();
    assert_eq!(requests.len(), 4, "discarding outcomes must not retry");
    assert!(requests[1].body_contains_text("new input after exhaustion"));
    assert!(!requests[1].body_contains_text("discarded exhausted history"));
    assert!(!requests[1].body_contains_text("discarded exhausted delta"));
    assert!(requests[3].body_contains_text("new input after refusal"));
    assert!(!requests[3].body_contains_text("discarded refusal history"));
    Ok(())
}
