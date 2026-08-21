use anyhow::Result;
use codex_core::resolve_installation_id;
use codex_model_provider_info::CLAUDEFLARE_PROVIDER_ID;
use codex_model_provider_info::built_in_model_providers;
use codex_protocol::openai_models::ConfigShellToolType;
use core_test_support::responses;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;
use serde_json::json;

fn grok_builder(server: &wiremock::MockServer) -> core_test_support::test_codex::TestCodexBuilder {
    let base_url = format!("{}/v1/grok", server.uri());
    test_codex()
        .with_config(move |config| {
            let mut provider = built_in_model_providers(/*openai_base_url*/ None)
                .remove(CLAUDEFLARE_PROVIDER_ID)
                .expect("managed Claudeflare provider");
            provider
                .wire_routes
                .get_mut("grok")
                .expect("Grok route")
                .base_url = base_url;
            config.model_provider = provider;
            config.model = Some("xai/grok-4.6".to_string());
            config.base_instructions = Some("Grok conformance system".to_string());
            config.agents_enabled = false;
            config.update_plan_enabled = true;
            config.experimental_request_user_input_enabled = false;
            config.include_skill_instructions = false;
            config.include_permissions_instructions = false;
            config.include_apps_instructions = false;
            config.include_collaboration_mode_instructions = false;
            config.include_environment_context = false;
        })
        .with_model_info_override("xai/grok-4.6", |info| {
            info.shell_type = ConfigShellToolType::Disabled;
            info.experimental_supported_tools.clear();
            info.supports_search_tool = false;
        })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grok_responses_conformance_and_tool_continuation() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    let plan_args = json!({
        "explanation": "Grok tool replay",
        "plan": [{"step":"Finish","status":"completed"}],
    })
    .to_string();
    let encrypted_reasoning = format!("gAAAAAB{}", "A".repeat(1_459));
    let reasoning = json!({
        "type": "response.output_item.done",
        "item": {
            "id": "rs_grok_replay",
            "type": "reasoning",
            "status": "completed",
            "summary": [{"type": "summary_text", "text": "Checked the plan"}],
            "encrypted_content": encrypted_reasoning,
        }
    });
    let function_call = json!({
        "type": "response.output_item.done",
        "item": {
            "id": "fc_grok_replay",
            "type": "function_call",
            "status": "completed",
            "call_id": "call-plan",
            "name": "update_plan",
            "arguments": plan_args,
        }
    });
    let mock = responses::mount_sse_sequence(
        &server,
        vec![
            responses::sse(vec![
                responses::ev_response_created("resp-grok-1"),
                reasoning,
                function_call,
                responses::ev_completed("resp-grok-1"),
            ]),
            responses::sse(vec![
                responses::ev_response_created("resp-grok-2"),
                responses::ev_assistant_message("msg-grok", "Grok completed"),
                responses::ev_completed("resp-grok-2"),
            ]),
        ],
    )
    .await;
    let test = grok_builder(&server).build_with_auto_env(&server).await?;

    test.submit_turn("Use the plan tool once").await?;

    let requests = mock.requests();
    assert_eq!(requests.len(), 2);
    let first = &requests[0];
    let session_id = test.session_configured.session_id.to_string();
    let thread_id = test.session_configured.thread_id.to_string();
    let installation_id = resolve_installation_id(&test.config.codex_home).await?;
    assert_eq!(first.path(), "/v1/grok/responses");
    assert_eq!(
        first.header("x-grok-conv-id").as_deref(),
        Some(thread_id.as_str())
    );
    assert_eq!(
        first.header("x-grok-session-id").as_deref(),
        Some(session_id.as_str())
    );
    assert_eq!(
        first.header("x-grok-agent-id").as_deref(),
        Some(installation_id.as_str())
    );
    assert_eq!(
        first.header("x-grok-model-override").as_deref(),
        Some("grok-4.6")
    );
    assert!(first.header("x-grok-req-id").is_some());
    let tools = first.body_json()["tools"]
        .as_array()
        .expect("Grok tools should be an array")
        .clone();
    assert!(
        tools
            .iter()
            .all(|tool| tool["type"].as_str() != Some("custom"))
    );
    assert_eq!(
        tools.iter().find(|tool| tool["name"] == "apply_patch"),
        Some(&json!({
            "type": "function",
            "name": "apply_patch",
            "description": "Apply a patch to files in the workspace.",
            "strict": false,
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "string",
                        "description": "The complete apply_patch patch text."
                    }
                },
                "required": ["patch"],
                "additionalProperties": false
            }
        }))
    );
    for header in [
        "openai-beta",
        "x-openai-subagent",
        "x-oai-attestation",
        "x-client-request-id",
        "session_id",
        "conversation_id",
        "x-codex-turn-metadata",
    ] {
        assert_eq!(first.header(header), None, "unexpected header {header}");
    }

    let mut snapshot = first.body_json();
    snapshot["prompt_cache_key"] = json!("<session-id>");
    for item in snapshot["input"].as_array_mut().expect("snapshot input") {
        if item.get("id").is_some() {
            item["id"] = json!("<item-id>");
        }
    }
    let header_snapshot = json!({
        "x-grok-conv-id": "<thread-id>",
        "x-grok-req-id": "<turn-id>",
        "x-grok-session-id": "<session-id>",
        "x-grok-agent-id": "<installation-id>",
        "x-grok-model-override": first.header("x-grok-model-override"),
    });
    insta::assert_snapshot!(
        "grok_responses_request_profile",
        serde_json::to_string_pretty(&json!({"headers":header_snapshot,"body":snapshot}))?
    );

    let second = requests[1].body_json();
    for header in [
        "x-grok-conv-id",
        "x-grok-req-id",
        "x-grok-session-id",
        "x-grok-agent-id",
        "x-grok-model-override",
    ] {
        assert_eq!(requests[1].header(header), first.header(header));
    }
    assert_eq!(
        second["prompt_cache_key"],
        first.body_json()["prompt_cache_key"]
    );
    let replay = second["input"].as_array().expect("second input");
    let reasoning_index = replay
        .iter()
        .position(|item| item["type"] == "reasoning")
        .expect("replayed reasoning");
    assert_eq!(
        replay[reasoning_index],
        json!({
            "id": "rs_grok_replay",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Checked the plan"}],
            "encrypted_content": encrypted_reasoning,
        })
    );
    assert_eq!(
        replay[reasoning_index + 1],
        json!({
            "id": "fc_grok_replay",
            "type": "function_call",
            "call_id": "call-plan",
            "name": "update_plan",
            "arguments": plan_args,
        })
    );
    assert_eq!(replay[reasoning_index + 2]["type"], "function_call_output");
    assert_eq!(replay[reasoning_index + 2]["call_id"], "call-plan");
    Ok(())
}
