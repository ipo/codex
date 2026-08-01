use codex_protocol::items::TurnItem;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::user_input::UserInput;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_message_item_added;
use core_test_support::responses::ev_reasoning_item;
use core_test_support::responses::ev_reasoning_item_added;
use core_test_support::responses::ev_response_created;
use core_test_support::responses::mount_sse_sequence;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::TestCodex;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
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

fn text_delta(item_id: &str, delta: &str) -> serde_json::Value {
    json!({
        "type": "response.output_text.delta",
        "item_id": item_id,
        "delta": delta,
    })
}

fn reasoning_delta(item_id: &str, delta: &str) -> serde_json::Value {
    json!({
        "type": "response.reasoning_summary_text.delta",
        "item_id": item_id,
        "summary_index": 0,
        "delta": delta,
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn interleaved_items_keep_addressed_deltas_and_exact_lifecycles() -> anyhow::Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let first = sse(vec![
        ev_response_created("interleaved-response"),
        ev_message_item_added("message-1", ""),
        text_delta("message-1", "text-one"),
        ev_reasoning_item_added("reasoning-1", &[""]),
        reasoning_delta("reasoning-1", "reason-one"),
        json!({
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "id": "tool-item-1",
                "call_id": "tool-call-1",
                "name": "test_sync_tool",
                "arguments": "",
            }
        }),
        text_delta("message-1", "text-two"),
        reasoning_delta("reasoning-1", "reason-two"),
        ev_assistant_message("message-1", "text-onetext-two"),
        ev_reasoning_item("reasoning-1", &["reason-onereason-two"], &[]),
        json!({
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "id": "tool-item-1",
                "call_id": "tool-call-1",
                "name": "test_sync_tool",
                "arguments": "{}",
            }
        }),
        ev_completed("interleaved-response"),
    ]);
    let second = sse(vec![
        ev_assistant_message("final-message", "done"),
        ev_completed("final-response"),
    ]);
    let response_mock = mount_sse_sequence(&server, vec![first, second]).await;
    let mut builder = test_codex().with_model("test-gpt-5.1-codex");
    let test = builder.build_with_auto_env(&server).await?;

    submit_turn(&test, "interleave items").await?;
    let mut presentation = Vec::new();
    loop {
        match wait_for_event(&test.codex, |_| true).await {
            EventMsg::ItemStarted(event) => match event.item {
                TurnItem::AgentMessage(item) => presentation.push(format!("start:{}", item.id)),
                TurnItem::Reasoning(item) => presentation.push(format!("start:{}", item.id)),
                _ => {}
            },
            EventMsg::AgentMessageContentDelta(event) => {
                presentation.push(format!("text:{}:{}", event.item_id, event.delta));
            }
            EventMsg::ReasoningContentDelta(event) => {
                presentation.push(format!("reason:{}:{}", event.item_id, event.delta));
            }
            EventMsg::ItemCompleted(event) => match event.item {
                TurnItem::AgentMessage(item) => presentation.push(format!("done:{}", item.id)),
                TurnItem::Reasoning(item) => presentation.push(format!("done:{}", item.id)),
                _ => {}
            },
            EventMsg::TurnComplete(_) => break,
            _ => {}
        }
    }

    assert_eq!(
        presentation,
        vec![
            "start:message-1",
            "text:message-1:text-one",
            "start:reasoning-1",
            "reason:reasoning-1:reason-one",
            "text:message-1:text-two",
            "reason:reasoning-1:reason-two",
            "done:message-1",
            "done:reasoning-1",
            "start:final-message",
            "done:final-message",
        ]
    );
    assert_eq!(
        response_mock.function_call_output_text("tool-call-1"),
        Some("ok".to_string())
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tool_delta_is_visible_but_uncommitted_without_terminal() -> anyhow::Result<()> {
    skip_if_no_network!(Ok(()));
    let server = start_mock_server().await;
    let patch = "*** Begin Patch\n*** Add File: must-not-exist.txt\n+pending\n*** End Patch";
    let failed = sse(vec![
        ev_response_created("failed-response"),
        json!({
            "type": "response.output_item.added",
            "item": {
                "type": "custom_tool_call",
                "id": "tool-item-1",
                "call_id": "tool-call-1",
                "name": "apply_patch",
                "input": "",
            }
        }),
        json!({
            "type": "response.custom_tool_call_input.delta",
            "item_id": "tool-item-1",
            "call_id": "tool-call-1",
            "delta": patch,
        }),
        json!({
            "type": "response.output_item.done",
            "item": {
                "type": "custom_tool_call",
                "id": "tool-item-1",
                "call_id": "tool-call-1",
                "name": "apply_patch",
                "input": patch,
            }
        }),
    ]);
    let succeeded = sse(vec![
        ev_assistant_message("message-2", "still stable"),
        ev_completed("successful-response"),
    ]);
    let response_mock = mount_sse_sequence(&server, vec![failed, succeeded]).await;
    let mut builder = test_codex().with_config(|config| {
        config.model_provider.stream_max_retries = Some(0);
        config
            .features
            .enable(codex_features::Feature::ApplyPatchStreamingEvents)
            .expect("enable apply patch streaming events");
    });
    let test = builder.build_with_auto_env(&server).await?;

    submit_turn(&test, "present an uncommitted tool").await?;
    let mut saw_delta = false;
    let mut saw_tool_start = false;
    loop {
        match wait_for_event(&test.codex, |_| true).await {
            EventMsg::PatchApplyUpdated(event) if event.call_id == "tool-call-1" => {
                saw_delta = true;
            }
            EventMsg::PatchApplyBegin(event) if event.call_id == "tool-call-1" => {
                saw_tool_start = true;
            }
            EventMsg::TurnComplete(_) => break,
            _ => {}
        }
    }
    assert!(
        saw_delta,
        "addressed tool delta was not presented before EOF"
    );
    assert!(!saw_tool_start, "tool started without a valid terminal");

    submit_turn(&test, "continue from stable history").await?;
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    let requests = response_mock.requests();
    assert_eq!(requests.len(), 2);
    assert!(!requests[1].body_contains_text("tool-call-1"));
    assert!(!requests[1].body_contains_text("must-not-exist.txt"));
    Ok(())
}
