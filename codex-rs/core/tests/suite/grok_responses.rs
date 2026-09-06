use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use codex_core::ForkSnapshot;
use codex_core::TurnInputRequest;
use codex_core::resolve_installation_id;
use codex_features::Feature;
use codex_model_provider_info::CLAUDEFLARE_PROVIDER_ID;
use codex_model_provider_info::built_in_model_providers;
use codex_protocol::openai_models::ConfigShellToolType;
use codex_protocol::openai_models::ModelToolCapability;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::user_input::UserInput;
use core_test_support::responses;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use serde_json::json;

fn grok_builder(server: &wiremock::MockServer) -> core_test_support::test_codex::TestCodexBuilder {
    let base_url = format!("{}/v1/grok", server.uri());
    test_codex()
        .with_config(move |config| {
            let mut provider = built_in_model_providers(/*openai_base_url*/ None)
                .remove(CLAUDEFLARE_PROVIDER_ID)
                .expect("managed Claudeflare provider");
            provider.stream_max_retries = Some(0);
            provider
                .wire_routes
                .get_mut("grok")
                .expect("Grok route")
                .base_url = base_url;
            provider
                .wire_routes
                .get_mut("grok")
                .expect("Grok route")
                .stream_max_retries = Some(1);
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
            config
                .features
                .disable(Feature::ViewImage)
                .expect("test config should disable image tools");
        })
        .with_model_info_override("xai/grok-4.6", |model| {
            model.shell_type = ConfigShellToolType::Disabled;
            model.apply_patch_tool_type = None;
            model.experimental_supported_tools.clear();
            model.disabled_tools = vec![
                ModelToolCapability::ApplyPatch,
                ModelToolCapability::ToolSearch,
                ModelToolCapability::WebSearch,
                ModelToolCapability::ImageGeneration,
                ModelToolCapability::CodexApps,
            ];
            model.supports_search_tool = false;
        })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grok_standalone_error_uses_route_retry_budget() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = responses::start_mock_server().await;
    let mock = responses::mount_sse_sequence(
        &server,
        vec![
            responses::sse(vec![json!({
                "type": "error",
                "code": "temporary_grok_error",
                "message": "retry me"
            })]),
            responses::sse(vec![
                responses::ev_response_created("resp-grok-retry"),
                responses::ev_assistant_message("msg-grok-retry", "recovered"),
                responses::ev_completed("resp-grok-retry"),
            ]),
        ],
    )
    .await;
    let test = grok_builder(&server).build_with_auto_env(&server).await?;

    test.submit_turn("Retry one transient Grok error").await?;

    assert_eq!(mock.requests().len(), 2);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn interrupting_grok_sampling_does_not_retry() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = responses::start_mock_server().await;
    let mock = responses::mount_response_once(
        &server,
        responses::sse_response(responses::sse(vec![
            responses::ev_response_created("resp-grok-interrupt"),
            responses::ev_completed("resp-grok-interrupt"),
        ]))
        .set_delay(Duration::from_secs(60)),
    )
    .await;
    let test = grok_builder(&server).build_with_auto_env(&server).await?;
    let codex = Arc::clone(&test.codex);
    codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "Interrupt this Grok request".to_string(),
            text_elements: Vec::new(),
        }]))
        .await?;
    tokio::time::timeout(Duration::from_secs(10), async {
        while mock.requests().is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await?;

    codex.submit(Op::Interrupt).await?;
    wait_for_event(&codex, |event| matches!(event, EventMsg::TurnAborted(_))).await;

    assert_eq!(mock.requests().len(), 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grok_resume_and_public_fork_preserve_wire_lineage() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = responses::start_mock_server().await;
    let mock = responses::mount_sse_sequence(
        &server,
        ["root", "branch", "distinct"]
            .into_iter()
            .map(|id| {
                responses::sse(vec![
                    responses::ev_response_created(id),
                    responses::ev_completed(id),
                ])
            })
            .collect(),
    )
    .await;
    let mut builder = grok_builder(&server);
    let root = builder.build_with_auto_env(&server).await?;
    root.submit_turn("root Grok turn").await?;
    let root_session_id = root.session_configured.session_id;
    let root_thread_id = root.session_configured.thread_id;
    let rollout_path = root.codex.rollout_path().expect("root rollout path");
    let home = root.home.clone();
    root.codex.shutdown_and_wait().await?;

    let resumed = builder.resume(&server, home, rollout_path.clone()).await?;
    assert_eq!(
        (
            resumed.session_configured.thread_id,
            resumed.session_configured.session_id,
        ),
        (root_thread_id, root_session_id)
    );
    let branch = resumed
        .thread_manager
        .fork_thread(
            ForkSnapshot::Interrupted,
            resumed.config.clone(),
            rollout_path,
            /*thread_source*/ None,
            /*parent_trace*/ None,
        )
        .await?;
    assert_ne!(branch.thread_id, root_thread_id);
    assert_eq!(branch.session_configured.session_id, root_session_id);
    branch
        .thread
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "branch Grok turn".to_string(),
            text_elements: Vec::new(),
        }]))
        .await?;
    wait_for_event(&branch.thread, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;

    let distinct = builder.build_with_auto_env(&server).await?;
    distinct.submit_turn("distinct Grok root").await?;
    assert_ne!(distinct.session_configured.session_id, root_session_id);

    let requests = mock.requests();
    assert_eq!(requests.len(), 3);
    assert_eq!(
        json!({
            "root": {
                "session": requests[0].header("x-grok-session-id"),
                "conversation": requests[0].header("x-grok-conv-id"),
                "cache": requests[0].body_json()["prompt_cache_key"].clone(),
            },
            "branch": {
                "session": requests[1].header("x-grok-session-id"),
                "conversation": requests[1].header("x-grok-conv-id"),
                "cache": requests[1].body_json()["prompt_cache_key"].clone(),
            },
            "distinct": {
                "session": requests[2].header("x-grok-session-id"),
                "conversation": requests[2].header("x-grok-conv-id"),
                "cache": requests[2].body_json()["prompt_cache_key"].clone(),
            },
        }),
        json!({
            "root": {
                "session": root_session_id,
                "conversation": root_thread_id,
                "cache": root_session_id,
            },
            "branch": {
                "session": root_session_id,
                "conversation": branch.thread_id,
                "cache": root_session_id,
            },
            "distinct": {
                "session": distinct.session_configured.session_id,
                "conversation": distinct.session_configured.thread_id,
                "cache": distinct.session_configured.session_id,
            },
        })
    );
    Ok(())
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
    let request = &requests[0];
    let session_id = test.session_configured.session_id.to_string();
    let thread_id = test.session_configured.thread_id.to_string();
    let installation_id = resolve_installation_id(&test.config.codex_home).await?;
    assert_eq!(request.path(), "/v1/grok/responses");
    assert_eq!(
        request.header("x-grok-conv-id").as_deref(),
        Some(thread_id.as_str())
    );
    assert_eq!(
        request.header("x-grok-session-id").as_deref(),
        Some(session_id.as_str())
    );
    assert_eq!(
        request.header("x-grok-agent-id").as_deref(),
        Some(installation_id.as_str())
    );
    assert_eq!(
        request.header("x-grok-model-override").as_deref(),
        Some("grok-4.6")
    );
    assert!(request.header("x-grok-req-id").is_some());
    for header in [
        "openai-beta",
        "x-openai-subagent",
        "x-oai-attestation",
        "x-client-request-id",
        "session_id",
        "conversation_id",
        "x-codex-turn-metadata",
    ] {
        assert_eq!(request.header(header), None, "unexpected header {header}");
    }

    let mut body = request.body_json();
    body["prompt_cache_key"] = json!("<session-id>");
    for item in body["input"].as_array_mut().expect("request input") {
        if item.get("id").is_some() {
            item["id"] = json!("<item-id>");
        }
    }
    assert_eq!(
        body,
        json!({
            "model": "grok-4.6",
            "instructions": "Grok conformance system",
            "input": [{
                "type": "message",
                "role": "user",
                "id": "<item-id>",
                "content": [{"type": "input_text", "text": "Use the plan tool once"}]
            }],
            "tools": [{
                "type": "function",
                "name": "update_plan",
                "description": "Updates the task plan.\nProvide an optional explanation and a list of plan items, each with a step and status.\nAt most one step can be in_progress at a time.\n",
                "strict": false,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "explanation": {
                            "type": "string",
                            "description": "Optional explanation for this plan update."
                        },
                        "plan": {
                            "type": "array",
                            "description": "The list of steps",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "step": {"type": "string", "description": "Task step text."},
                                    "status": {
                                        "type": "string",
                                        "enum": ["pending", "in_progress", "completed"],
                                        "description": "Step status."
                                    }
                                },
                                "required": ["step", "status"],
                                "additionalProperties": false
                            }
                        }
                    },
                    "required": ["plan"],
                    "additionalProperties": false
                }
            }],
            "tool_choice": "auto",
            "parallel_tool_calls": true,
            "reasoning": {"effort": "high", "summary": "concise"},
            "store": false,
            "stream": true,
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "<session-id>"
        })
    );

    for header in [
        "x-grok-conv-id",
        "x-grok-req-id",
        "x-grok-session-id",
        "x-grok-agent-id",
        "x-grok-model-override",
    ] {
        assert_eq!(requests[1].header(header), request.header(header));
    }
    assert_eq!(
        requests[1].body_json()["prompt_cache_key"],
        request.body_json()["prompt_cache_key"]
    );
    let replay = requests[1].body_json()["input"]
        .as_array()
        .expect("continuation input")
        .clone();
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
