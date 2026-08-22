use core_test_support::wait_for_mcp_server;
use pretty_assertions::assert_eq;

use super::*;

fn function_response(
    response_id: &str,
    call_id: &str,
    name: &str,
    arguments: &str,
) -> ResponseTemplate {
    local_sse(vec![
        responses::ev_response_created(response_id),
        responses::ev_function_call(call_id, name, arguments),
        responses::ev_completed(response_id),
    ])
}

fn function_output(body: &Value, call_id: &str) -> Option<String> {
    body["input"].as_array()?.iter().find_map(|item| {
        (item["type"] == "function_call_output" && item["call_id"] == call_id)
            .then(|| item["output"].to_string())
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn full_local_turn_uses_exact_wire_sequence_and_authoritative_tool_call() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = MockServer::start().await;
    mount_ready(&server).await;
    mount_token_counts(&server, [91, 137]).await;
    let valid_arguments = json!({
        "explanation": "local plan",
        "plan": [{"step": "Finish", "status": "completed"}],
    })
    .to_string();
    mount_local_inference(
        &server,
        vec![
            local_sse(vec![
                responses::ev_response_created("local-tool"),
                responses::ev_reasoning_item_added("rs-local", &[]),
                responses::ev_reasoning_text_delta("inspect then update"),
                json!({
                    "type": "response.output_item.done",
                    "item": {
                        "id": "rs-local",
                        "type": "reasoning",
                        "status": "completed",
                        "summary": [],
                        "content": [{"type": "reasoning_text", "text": "inspect then update"}],
                        "encrypted_content": "",
                    }
                }),
                json!({
                    "type": "response.function_call_arguments.delta",
                    "delta": "{\"discarded\":true}",
                }),
                responses::ev_function_call("call-plan", "update_plan", &valid_arguments),
                responses::ev_completed("local-tool"),
            ]),
            text_response("local-final", "msg-local", "Local turn complete"),
        ],
    )
    .await;
    let test = local_builder(&server).build_with_auto_env(&server).await?;

    submit_turn(&test, "Use update_plan exactly once").await?;

    let requests = server.received_requests().await.expect("captured requests");
    assert_eq!(
        requests
            .iter()
            .map(|request| request.url.path())
            .collect::<Vec<_>>(),
        [
            "/health",
            "/v1/models",
            "/v1/responses/input_tokens",
            "/v1/responses",
            "/v1/responses/input_tokens",
            "/v1/responses",
        ]
    );
    assert!(
        requests
            .iter()
            .all(|request| !request.headers.contains_key(http::header::AUTHORIZATION))
    );
    for pair in [(&requests[2], &requests[3]), (&requests[4], &requests[5])] {
        assert_eq!(pair.0.body, pair.1.body);
    }

    let first: Value = requests[3].body_json()?;
    assert_eq!(first["model"], WINDOWS_MODEL);
    assert_eq!(first["stream"], true);
    assert_eq!(first["cache_prompt"], true);
    assert_eq!(first["max_output_tokens"], 8_192);
    assert_eq!(
        first["chat_template_kwargs"],
        json!({"enable_thinking": true, "preserve_thinking": true})
    );
    assert_eq!(first["temperature"], 1.0);
    assert_eq!(first["top_p"], 0.95);
    assert_eq!(first["top_k"], 20);
    assert_eq!(first["min_p"], 0.0);
    assert_eq!(first["presence_penalty"], 0.0);
    assert_eq!(first["repeat_penalty"], 1.0);
    assert!(first.get("previous_response_id").is_none());

    let continuation: Value = requests[5].body_json()?;
    let input = continuation["input"].as_array().expect("typed input");
    assert_eq!(
        input
            .iter()
            .filter(|item| item["type"] == "reasoning")
            .count(),
        1
    );
    assert_eq!(
        input
            .iter()
            .find(|item| item["type"] == "reasoning")
            .expect("raw reasoning")["content"][0]["text"],
        "inspect then update"
    );
    let function_call = input
        .iter()
        .find(|item| item["type"] == "function_call")
        .expect("function call replay");
    assert_eq!(function_call["arguments"], valid_arguments);
    assert_eq!(
        input
            .iter()
            .filter(|item| item["type"] == "function_call_output")
            .count(),
        1
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn local_plain_function_calls_are_schema_validated_before_dispatch() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = MockServer::start().await;
    mount_ready(&server).await;
    mount_token_counts(&server, [10; 6]).await;
    let valid = json!({
        "explanation": "only valid invocation",
        "plan": [{"step": "Finish", "status": "completed"}],
    })
    .to_string();
    mount_local_inference(
        &server,
        vec![
            function_response("r1", "malformed", "update_plan", "{"),
            function_response("r2", "unknown", "not_advertised", "{}"),
            function_response("r3", "missing", "update_plan", "{}"),
            function_response(
                "r4",
                "wrong",
                "update_plan",
                r#"{"explanation":7,"plan":"bad","extra":true}"#,
            ),
            function_response("r5", "valid", "update_plan", &valid),
            text_response("r6", "m6", "validation repaired"),
        ],
    )
    .await;
    let test = local_builder(&server).build_with_auto_env(&server).await?;

    submit_turn(&test, "Repair invalid plain function calls").await?;

    let requests = received(&server, "/v1/responses").await;
    assert_eq!(requests.len(), 6);
    for (index, call_id, expected) in [
        (1, "malformed", "malformed JSON arguments"),
        (2, "unknown", "unknown or unadvertised tool"),
        (3, "missing", "missing required property"),
        (4, "wrong", "failed schema validation"),
    ] {
        let body: Value = requests[index].body_json()?;
        let output = function_output(&body, call_id).expect("validation output");
        assert!(
            output.contains(expected),
            "output for {call_id} did not contain {expected:?}: {output}"
        );
    }
    let final_body: Value = requests[5].body_json()?;
    let valid_output = function_output(&final_body, "valid").expect("valid tool output");
    assert!(!valid_output.contains("schema validation"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn local_mcp_function_schema_is_enforced_and_valid_call_executes() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = MockServer::start().await;
    mount_ready(&server).await;
    mount_token_counts(&server, [10; 3]).await;
    mount_local_inference(
        &server,
        vec![
            function_response("mcp-r1", "mcp-invalid", "mcp__rmcp__echo", "{}"),
            function_response(
                "mcp-r2",
                "mcp-valid",
                "mcp__rmcp__echo",
                r#"{"message":"ping"}"#,
            ),
            text_response("mcp-r3", "mcp-final", "MCP complete"),
        ],
    )
    .await;
    let command = super::super::rmcp_client::remote_aware_stdio_server_bin()?;
    let test = local_builder(&server)
        .with_config(move |config| {
            super::super::rmcp_client::configure_stdio_mcp(config, "rmcp", command);
        })
        .build_with_auto_env(&server)
        .await?;
    wait_for_mcp_server(&test.codex, "rmcp").await?;

    submit_turn(&test, "Validate then call the MCP echo tool").await?;

    let requests = received(&server, "/v1/responses").await;
    assert_eq!(requests.len(), 3);
    let invalid: Value = requests[1].body_json()?;
    assert!(
        function_output(&invalid, "mcp-invalid")
            .expect("invalid MCP output")
            .contains("missing required property `message`")
    );
    let valid: Value = requests[2].body_json()?;
    assert!(
        function_output(&valid, "mcp-valid")
            .expect("valid MCP output")
            .contains("ECHOING: ping")
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn normal_local_turn_rejects_image_and_audio_before_network() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = MockServer::start().await;
    let test = local_builder(&server).build_with_auto_env(&server).await?;
    test.codex
        .submit(Op::UserInput {
            items: vec![
                UserInput::Image {
                    image_url: "data:image/png;base64,iVBORw0KGgo=".to_string(),
                    detail: None,
                },
                UserInput::Audio {
                    audio_url: "data:audio/wav;base64,UklGRg==".to_string(),
                },
            ],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;

    let error = wait_for_event_match(&test.codex, |event| match event {
        EventMsg::Error(error) => Some(error.message.clone()),
        EventMsg::TurnComplete(event) => event.error.as_ref().map(|error| error.message.clone()),
        _ => None,
    })
    .await;
    assert!(error.contains("unsupported image or audio input"));
    assert!(
        server
            .received_requests()
            .await
            .expect("captured requests")
            .is_empty()
    );
    Ok(())
}
