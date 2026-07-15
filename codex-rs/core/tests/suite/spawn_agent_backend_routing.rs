use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use codex_core::ThreadConfigSnapshot;
use codex_core::config::AgentRoleConfig;
use codex_features::Feature;
use codex_protocol::SessionId;
use codex_protocol::openai_models::ReasoningEffort;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::protocol::RolloutItem;
use codex_protocol::protocol::SessionMeta;
use codex_protocol::protocol::SessionSource;
use codex_protocol::protocol::SubAgentSource;
use codex_protocol::protocol::SubagentBackendRoute;
use codex_protocol::protocol::ThreadSource;
use codex_protocol::user_input::UserInput;
use core_test_support::responses::ResponseMock;
use core_test_support::responses::ResponsesRequest;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_function_call_with_namespace;
use core_test_support::responses::ev_response_created;
use core_test_support::responses::mount_response_once_match;
use core_test_support::responses::mount_sse_once;
use core_test_support::responses::mount_sse_once_match;
use core_test_support::responses::sse;
use core_test_support::responses::sse_response;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::TestCodex;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use std::time::Duration;

const PARENT_PROMPT: &str = "spawn the requested routing-test child";
const CHILD_PROMPT: &str = "routing-test child prompt";
const LIST_PARENT_PROMPT: &str = "list the routing-test child";
const RELOAD_PARENT_PROMPT: &str = "reload the routing-test child";
const RELOAD_CHILD_PROMPT: &str = "continue after AgentControl reload";
const COMPLETION_PARENT_PROMPT: &str = "collect the reloaded child's completion";
const SPAWN_CALL_ID: &str = "routing-spawn-call";
const RELOAD_LIST_CALL_ID: &str = "routing-reload-list-call";
const RELOAD_CALL_ID: &str = "routing-reload-call";
const COMPLETION_WAIT_CALL_ID: &str = "routing-completion-wait-call";
const PARENT_MODEL: &str = "gpt-5.4";
const PARENT_EFFORT: ReasoningEffort = ReasoningEffort::Medium;
const CHILD_MODEL: &str = "gpt-5.2";
const CHILD_EFFORT: ReasoningEffort = ReasoningEffort::Low;

#[derive(Clone, Copy)]
enum ToolVersion {
    V1,
    V2,
}

struct SpawnObservation {
    request: ResponsesRequest,
    tool_output: String,
    session_meta: SessionMeta,
    config_snapshot: ThreadConfigSnapshot,
    root_thread_id: codex_protocol::ThreadId,
    child_thread_id: codex_protocol::ThreadId,
    server: wiremock::MockServer,
    test: TestCodex,
}

async fn run_spawn(
    version: ToolVersion,
    spawn_args: Value,
    primary_used_percent: Option<f64>,
    custom_provider: bool,
    custom_role: bool,
) -> Result<SpawnObservation> {
    let server = start_mock_server().await;
    let namespace = match version {
        ToolVersion::V1 => "multi_agent_v1",
        ToolVersion::V2 => "collaboration",
    };
    let spawn_args = serde_json::to_string(&spawn_args)?;
    let parent_sse = sse(vec![
        ev_response_created("resp-parent-1"),
        ev_function_call_with_namespace(SPAWN_CALL_ID, namespace, "spawn_agent", &spawn_args),
        ev_completed("resp-parent-1"),
    ]);
    let mut parent_response = sse_response(parent_sse);
    if let Some(primary_used_percent) = primary_used_percent {
        parent_response = parent_response.insert_header(
            "x-codex-primary-used-percent",
            primary_used_percent.to_string(),
        );
    }
    mount_response_once_match(
        &server,
        |request: &wiremock::Request| {
            request_body_contains(request, PARENT_PROMPT) && !wire_request_is_spawned_child(request)
        },
        parent_response,
    )
    .await;
    let child_mock = mount_sse_once_match(
        &server,
        |request: &wiremock::Request| {
            request_body_contains(request, CHILD_PROMPT) && wire_request_is_spawned_child(request)
        },
        sse(vec![
            ev_response_created("resp-child-1"),
            ev_assistant_message("msg-child-1", "child complete"),
            ev_completed("resp-child-1"),
        ]),
    )
    .await;
    let parent_followup = mount_sse_once_match(
        &server,
        |request: &wiremock::Request| {
            request_body_contains(request, SPAWN_CALL_ID) && !wire_request_is_spawned_child(request)
        },
        sse(vec![
            ev_response_created("resp-parent-2"),
            ev_assistant_message("msg-parent-2", "parent complete"),
            ev_completed("resp-parent-2"),
        ]),
    )
    .await;

    let mut builder = test_codex()
        .with_model(PARENT_MODEL)
        .with_config(move |config| {
            config
                .features
                .enable(Feature::Collab)
                .expect("test config should allow feature update");
            if matches!(version, ToolVersion::V2) {
                config
                    .features
                    .enable(Feature::MultiAgentV2)
                    .expect("test config should allow feature update");
            }
            config.model_reasoning_effort = Some(PARENT_EFFORT);
            config.model_provider.supports_websockets = false;
            config
                .features
                .disable(Feature::EnableRequestCompression)
                .expect("test config should allow feature update");
            if custom_provider {
                let custom_provider = config.model_provider.clone();
                config
                    .model_providers
                    .insert("custom".to_string(), custom_provider.clone());
                config.model_provider_id = "custom".to_string();
                config.model_provider = custom_provider;
            }
            if custom_role {
                let role_path = config.codex_home.join("routing-test-role.toml");
                let role_provider = if custom_provider { "model_provider = \"openai\"\n" } else { Default::default() };
                std::fs::write(
                    &role_path,
                    format!(
                        "{role_provider}model = \"{CHILD_MODEL}\"\nmodel_reasoning_effort = \"{CHILD_EFFORT}\"\n"
                    ),
                )
                .expect("write routing-test role");
                config.agent_roles.insert(
                    "routing_test".to_string(),
                    AgentRoleConfig {
                        description: Some("Routing test role".to_string()),
                        config_file: Some(role_path.to_path_buf()),
                        nickname_candidates: None,
                    },
                );
            }
        });
    if custom_role {
        let openai_base_url = format!("{}/v1", server.uri());
        builder = builder.with_pre_build_hook(move |home| {
            std::fs::write(
                home.join("config.toml"),
                format!("openai_base_url = {openai_base_url:?}\n"),
            )
            .expect("write test OpenAI base URL");
        });
    }
    let test = builder.build_with_auto_env(&server).await?;
    test.submit_turn(PARENT_PROMPT).await?;

    let parent_followup =
        wait_for_request(&parent_followup, "parent follow-up request", |request| {
            request.function_call_output_text(SPAWN_CALL_ID).is_some()
        })
        .await?;
    let tool_output = parent_followup
        .function_call_output_text(SPAWN_CALL_ID)
        .ok_or_else(|| anyhow!("parent follow-up is missing spawn tool output"))?;
    let request = wait_for_request(&child_mock, "child request", |request| {
        request.body_contains_text(CHILD_PROMPT) && responses_request_is_spawned_child(request)
    })
    .await
    .map_err(|error| anyhow!("{error}; spawn output: {tool_output}"))?;
    let root_thread_id = test.session_configured.thread_id;
    let child_thread_id = test
        .thread_manager
        .list_thread_ids()
        .await
        .into_iter()
        .find(|thread_id| *thread_id != root_thread_id)
        .ok_or_else(|| anyhow!("spawned child thread is missing"))?;
    let child_thread = test.thread_manager.get_thread(child_thread_id).await?;
    let config_snapshot = child_thread.config_snapshot().await;
    child_thread.ensure_rollout_materialized().await;
    child_thread.flush_rollout().await?;
    let stored_thread = child_thread
        .read_thread(
            /*include_archived*/ true, /*include_history*/ true,
        )
        .await?;
    let session_meta = stored_thread
        .history
        .ok_or_else(|| anyhow!("spawned child history is missing"))?
        .items
        .into_iter()
        .find_map(|item| match item {
            RolloutItem::SessionMeta(meta) => Some(meta.meta),
            _ => None,
        })
        .ok_or_else(|| anyhow!("spawned child session metadata is missing"))?;

    Ok(SpawnObservation {
        request,
        tool_output,
        session_meta,
        config_snapshot,
        root_thread_id,
        child_thread_id,
        server,
        test,
    })
}

fn assert_spawn_route(
    observation: &SpawnObservation,
    expected_route: SubagentBackendRoute,
    expected_model: &str,
    expected_effort: ReasoningEffort,
    expected_agent_path: Option<&str>,
) -> Result<()> {
    let expected_subagent_header = expected_route
        .is_proper_subagent()
        .then_some("collab_spawn");
    let body = observation.request.body_json();
    let root_thread_id = observation.root_thread_id.to_string();
    let turn_metadata: Value = serde_json::from_str(
        observation
            .request
            .header("x-codex-turn-metadata")
            .as_deref()
            .ok_or_else(|| anyhow!("child request is missing logical turn metadata"))?,
    )?;

    assert_eq!(
        (
            body["model"].as_str(),
            body.pointer("/reasoning/effort").and_then(Value::as_str),
            observation.request.header("x-openai-subagent").as_deref(),
            body.pointer("/client_metadata/x-openai-subagent")
                .and_then(Value::as_str),
        ),
        (
            Some(expected_model),
            Some(expected_effort.as_str()),
            expected_subagent_header,
            expected_subagent_header,
        )
    );
    assert_eq!(
        (
            turn_metadata["subagent_kind"].as_str(),
            turn_metadata["parent_thread_id"].as_str(),
            observation.session_meta.subagent_backend_route,
            observation.session_meta.parent_thread_id,
            observation.session_meta.thread_source.as_ref(),
            observation.session_meta.session_id,
            observation.config_snapshot.model.as_str(),
            observation.config_snapshot.reasoning_effort.as_ref(),
            observation.session_meta.agent_path.as_deref(),
        ),
        (
            Some("thread_spawn"),
            Some(root_thread_id.as_str()),
            expected_route,
            Some(observation.root_thread_id),
            Some(&ThreadSource::Subagent),
            SessionId::from(observation.root_thread_id),
            expected_model,
            Some(&expected_effort),
            expected_agent_path,
        )
    );
    assert!(matches!(
        observation.session_meta.source,
        SessionSource::SubAgent(SubAgentSource::ThreadSpawn { .. })
    ));
    assert!(matches!(
        observation.config_snapshot.session_source,
        SessionSource::SubAgent(SubAgentSource::ThreadSpawn { .. })
    ));
    assert!(!observation.tool_output.to_lowercase().contains("quota"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v1_and_v2_overrides_use_main_route_for_new_and_partial_forks() -> Result<()> {
    skip_if_no_network!(Ok(()));
    for (version, spawn_args, expected_model, expected_effort, expected_path) in [
        (
            ToolVersion::V1,
            json!({
                "message": CHILD_PROMPT,
                "model": CHILD_MODEL,
                "reasoning_effort": CHILD_EFFORT,
            }),
            CHILD_MODEL,
            CHILD_EFFORT,
            None,
        ),
        (
            ToolVersion::V2,
            json!({
                "message": CHILD_PROMPT,
                "task_name": "new_child",
                "fork_turns": "none",
                "model": CHILD_MODEL,
                "reasoning_effort": CHILD_EFFORT,
            }),
            CHILD_MODEL,
            CHILD_EFFORT,
            Some("/root/new_child"),
        ),
        (
            ToolVersion::V2,
            json!({
                "message": CHILD_PROMPT,
                "task_name": "partial_child",
                "fork_turns": "1",
                "model": CHILD_MODEL,
                "reasoning_effort": CHILD_EFFORT,
            }),
            CHILD_MODEL,
            CHILD_EFFORT,
            Some("/root/partial_child"),
        ),
        (
            ToolVersion::V1,
            json!({
                "message": CHILD_PROMPT,
                "model": CHILD_MODEL,
            }),
            CHILD_MODEL,
            ReasoningEffort::Medium,
            None,
        ),
        (
            ToolVersion::V2,
            json!({
                "message": CHILD_PROMPT,
                "task_name": "effort_child",
                "fork_turns": "none",
                "reasoning_effort": CHILD_EFFORT,
            }),
            PARENT_MODEL,
            CHILD_EFFORT,
            Some("/root/effort_child"),
        ),
    ] {
        let observation = run_spawn(
            version, spawn_args, /*primary_used_percent*/ None,
            /*custom_provider*/ false, /*custom_role*/ false,
        )
        .await?;
        assert_spawn_route(
            &observation,
            SubagentBackendRoute::MainSession,
            expected_model,
            expected_effort,
            expected_path,
        )?;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn response_header_quota_boundary_restores_both_parent_settings() -> Result<()> {
    skip_if_no_network!(Ok(()));
    for (used_percent, route, model, effort) in [
        (
            94.999,
            SubagentBackendRoute::MainSession,
            CHILD_MODEL,
            CHILD_EFFORT,
        ),
        (
            95.0,
            SubagentBackendRoute::ProperSubagent,
            PARENT_MODEL,
            PARENT_EFFORT,
        ),
    ] {
        let observation = run_spawn(
            ToolVersion::V1,
            json!({
                "message": CHILD_PROMPT,
                "model": CHILD_MODEL,
                "reasoning_effort": CHILD_EFFORT,
            }),
            Some(used_percent),
            /*custom_provider*/ false,
            /*custom_role*/ false,
        )
        .await?;
        assert_spawn_route(
            &observation,
            route,
            model,
            effort,
            /*expected_agent_path*/ None,
        )?;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resumed_children_keep_their_persisted_backend_route() -> Result<()> {
    skip_if_no_network!(Ok(()));
    for (route, spawn_args, expected_model, expected_effort) in [
        (
            SubagentBackendRoute::MainSession,
            json!({
                "message": CHILD_PROMPT,
                "model": CHILD_MODEL,
                "reasoning_effort": CHILD_EFFORT,
            }),
            CHILD_MODEL,
            CHILD_EFFORT,
        ),
        (
            SubagentBackendRoute::ProperSubagent,
            json!({"message": CHILD_PROMPT}),
            PARENT_MODEL,
            PARENT_EFFORT,
        ),
    ] {
        let observation = run_spawn(
            ToolVersion::V1,
            spawn_args,
            /*primary_used_percent*/ None,
            /*custom_provider*/ false,
            /*custom_role*/ false,
        )
        .await?;
        let child = observation
            .test
            .thread_manager
            .get_thread(observation.child_thread_id)
            .await?;
        let rollout_path = child
            .rollout_path()
            .ok_or_else(|| anyhow!("spawned child rollout path is missing"))?;
        child.submit(Op::Shutdown).await?;
        wait_for_event(&child, |event| matches!(event, EventMsg::ShutdownComplete)).await;
        observation
            .test
            .thread_manager
            .remove_thread(&observation.child_thread_id)
            .await;
        drop(child);

        let rollout = std::fs::read_to_string(&rollout_path)?;
        assert_eq!(
            rollout.contains("\"subagent_backend_route\":\"main_session\""),
            route == SubagentBackendRoute::MainSession
        );

        let server = start_mock_server().await;
        let resumed_request = mount_sse_once(
            &server,
            sse(vec![
                ev_response_created("resp-resumed-1"),
                ev_assistant_message("msg-resumed-1", "resumed child complete"),
                ev_completed("resp-resumed-1"),
            ]),
        )
        .await;
        let configured_effort = expected_effort.clone();
        let mut builder = test_codex()
            .with_model(expected_model)
            .with_config(move |config| {
                config.model_reasoning_effort = Some(configured_effort);
                config.model_provider.supports_websockets = false;
                config
                    .features
                    .disable(Feature::EnableRequestCompression)
                    .expect("test config should allow feature update");
            });
        let resumed = builder
            .resume(&server, observation.test.home.clone(), rollout_path)
            .await?;
        resumed
            .codex
            .submit(Op::UserInput {
                items: vec![UserInput::Text {
                    text: "continue after unload".to_string(),
                    text_elements: Vec::new(),
                }],
                final_output_json_schema: None,
                responsesapi_client_metadata: None,
                additional_context: Default::default(),
                thread_settings: Default::default(),
            })
            .await?;
        wait_for_event(&resumed.codex, |event| {
            matches!(event, EventMsg::TurnComplete(_))
        })
        .await;

        let request = resumed_request.single_request();
        let body = request.body_json();
        let turn_metadata: Value = serde_json::from_str(
            request
                .header("x-codex-turn-metadata")
                .as_deref()
                .ok_or_else(|| anyhow!("resumed request is missing turn metadata"))?,
        )?;
        let expected_subagent = route.is_proper_subagent().then_some("collab_spawn");
        let root_thread_id = observation.root_thread_id.to_string();
        let resumed_snapshot = resumed.codex.config_snapshot().await;
        assert_eq!(
            (
                resumed.session_configured.thread_id,
                resumed.session_configured.session_id,
                resumed.session_configured.thread_source.as_ref(),
                body["model"].as_str(),
                body.pointer("/reasoning/effort").and_then(Value::as_str),
                request.header("x-openai-subagent").as_deref(),
                turn_metadata["subagent_kind"].as_str(),
                turn_metadata["parent_thread_id"].as_str(),
                resumed_snapshot.session_source,
            ),
            (
                observation.child_thread_id,
                SessionId::from(observation.root_thread_id),
                Some(&ThreadSource::Subagent),
                Some(expected_model),
                Some(expected_effort.as_str()),
                expected_subagent,
                Some("thread_spawn"),
                Some(root_thread_id.as_str()),
                observation.config_snapshot.session_source,
            )
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_agent_control_reload_preserves_child_model_effort_and_route() -> Result<()> {
    skip_if_no_network!(Ok(()));
    for (
        route,
        spawn_args,
        expected_model,
        expected_effort,
        expected_provider,
        task_name,
        custom_provider,
        custom_role,
    ) in [
        (
            SubagentBackendRoute::MainSession,
            json!({
                "message": CHILD_PROMPT,
                "task_name": "main_reload_child",
                "fork_turns": "none",
                "model": CHILD_MODEL,
                "reasoning_effort": CHILD_EFFORT,
            }),
            CHILD_MODEL,
            CHILD_EFFORT,
            "openai",
            "main_reload_child",
            false,
            false,
        ),
        (
            SubagentBackendRoute::ProperSubagent,
            json!({
                "message": CHILD_PROMPT,
                "task_name": "proper_reload_child",
                "fork_turns": "none",
                "agent_type": "routing_test",
            }),
            PARENT_MODEL,
            PARENT_EFFORT,
            "openai",
            "proper_reload_child",
            false,
            true,
        ),
        (
            SubagentBackendRoute::MainSession,
            json!({
                "message": CHILD_PROMPT,
                "task_name": "provider_reload_child",
                "fork_turns": "none",
                "agent_type": "routing_test",
                "reasoning_effort": CHILD_EFFORT,
            }),
            CHILD_MODEL,
            CHILD_EFFORT,
            "openai",
            "provider_reload_child",
            true,
            true,
        ),
    ] {
        let observation = run_spawn(
            ToolVersion::V2,
            spawn_args,
            /*primary_used_percent*/ None,
            custom_provider,
            custom_role,
        )
        .await?;
        let child = observation
            .test
            .thread_manager
            .get_thread(observation.child_thread_id)
            .await?;

        if task_name == "main_reload_child" {
            mount_sse_once_match(
                &observation.server,
                |request: &wiremock::Request| {
                    request_body_contains(request, LIST_PARENT_PROMPT)
                        && !wire_request_is_spawned_child(request)
                },
                sse(vec![
                    ev_response_created("resp-list-parent-1"),
                    ev_function_call_with_namespace(
                        RELOAD_LIST_CALL_ID,
                        "collaboration",
                        "list_agents",
                        "{}",
                    ),
                    ev_completed("resp-list-parent-1"),
                ]),
            )
            .await;
            let listed_agent = mount_sse_once_match(
                &observation.server,
                |request: &wiremock::Request| {
                    request_body_contains(request, RELOAD_LIST_CALL_ID)
                        && !wire_request_is_spawned_child(request)
                },
                sse(vec![
                    ev_response_created("resp-list-parent-2"),
                    ev_assistant_message("msg-list-parent-2", "child listed"),
                    ev_completed("resp-list-parent-2"),
                ]),
            )
            .await;

            observation.test.submit_turn(LIST_PARENT_PROMPT).await?;
            let list_request = wait_for_request(&listed_agent, "list_agents output", |request| {
                request
                    .function_call_output_text(RELOAD_LIST_CALL_ID)
                    .is_some()
            })
            .await?;
            let list_output = list_request
                .function_call_output_text(RELOAD_LIST_CALL_ID)
                .ok_or_else(|| anyhow!("parent follow-up is missing list_agents output"))?;
            assert!(list_output.contains(&format!("/root/{task_name}")));
        }

        child.ensure_rollout_materialized().await;
        child.flush_rollout().await?;
        child.submit(Op::Shutdown).await?;
        wait_for_event(&child, |event| matches!(event, EventMsg::ShutdownComplete)).await;
        observation
            .test
            .thread_manager
            .remove_thread(&observation.child_thread_id)
            .await;
        drop(child);

        let reload_args = serde_json::to_string(&json!({
            "target": format!("/root/{task_name}"),
            "message": RELOAD_CHILD_PROMPT,
        }))?;
        mount_sse_once_match(
            &observation.server,
            |request: &wiremock::Request| {
                request_body_contains(request, RELOAD_PARENT_PROMPT)
                    && !wire_request_is_spawned_child(request)
            },
            sse(vec![
                ev_response_created("resp-reload-parent-1"),
                ev_function_call_with_namespace(
                    RELOAD_CALL_ID,
                    "collaboration",
                    "followup_task",
                    &reload_args,
                ),
                ev_completed("resp-reload-parent-1"),
            ]),
        )
        .await;
        let reloaded_child = mount_sse_once_match(
            &observation.server,
            |request: &wiremock::Request| {
                request_body_contains(request, RELOAD_CHILD_PROMPT)
                    && wire_request_is_spawned_child(request)
            },
            sse(vec![
                ev_response_created("resp-reloaded-child-1"),
                ev_assistant_message("msg-reloaded-child-1", "reloaded child complete"),
                ev_completed("resp-reloaded-child-1"),
            ]),
        )
        .await;
        mount_sse_once_match(
            &observation.server,
            |request: &wiremock::Request| {
                request_body_contains(request, RELOAD_CALL_ID)
                    && !wire_request_is_spawned_child(request)
            },
            sse(vec![
                ev_response_created("resp-reload-parent-3"),
                ev_assistant_message("msg-reload-parent-3", "reload requested"),
                ev_completed("resp-reload-parent-3"),
            ]),
        )
        .await;

        observation.test.submit_turn(RELOAD_PARENT_PROMPT).await?;
        let request = wait_for_request(&reloaded_child, "reloaded child request", |request| {
            request.body_contains_text(RELOAD_CHILD_PROMPT)
                && responses_request_is_spawned_child(request)
        })
        .await?;
        let expected_subagent = route.is_proper_subagent().then_some("collab_spawn");
        let body = request.body_json();
        assert_eq!(
            (
                body["model"].as_str(),
                body.pointer("/reasoning/effort").and_then(Value::as_str),
                request.header("x-openai-subagent").as_deref(),
            ),
            (
                Some(expected_model),
                Some(expected_effort.as_str()),
                expected_subagent,
            )
        );
        let reloaded_snapshot = observation
            .test
            .thread_manager
            .get_thread(observation.child_thread_id)
            .await?
            .config_snapshot()
            .await;
        assert_eq!(
            (
                reloaded_snapshot.model.as_str(),
                reloaded_snapshot.model_provider_id.as_str(),
                reloaded_snapshot.reasoning_effort.as_ref(),
                reloaded_snapshot.session_source,
            ),
            (
                expected_model,
                expected_provider,
                Some(&expected_effort),
                observation.config_snapshot.session_source,
            )
        );

        if task_name == "main_reload_child" {
            mount_sse_once_match(
                &observation.server,
                |request: &wiremock::Request| {
                    request_body_contains(request, COMPLETION_PARENT_PROMPT)
                        && !request_body_contains(request, "Message Type: FINAL_ANSWER")
                        && !wire_request_is_spawned_child(request)
                },
                sse(vec![
                    ev_response_created("resp-completion-parent-1"),
                    ev_function_call_with_namespace(
                        COMPLETION_WAIT_CALL_ID,
                        "collaboration",
                        "wait_agent",
                        "{}",
                    ),
                    ev_completed("resp-completion-parent-1"),
                ]),
            )
            .await;
            let completion_delivery = mount_sse_once_match(
                &observation.server,
                |request: &wiremock::Request| {
                    request_body_contains(request, COMPLETION_PARENT_PROMPT)
                        && request_body_contains(request, "Message Type: FINAL_ANSWER")
                        && request_body_contains(request, "reloaded child complete")
                        && !wire_request_is_spawned_child(request)
                },
                sse(vec![
                    ev_response_created("resp-completion-parent-2"),
                    ev_assistant_message("msg-completion-parent-2", "completion received"),
                    ev_completed("resp-completion-parent-2"),
                ]),
            )
            .await;

            observation
                .test
                .submit_turn(COMPLETION_PARENT_PROMPT)
                .await?;
            wait_for_request(
                &completion_delivery,
                "parent completion delivery",
                |request| {
                    request.body_contains_text("Message Type: FINAL_ANSWER")
                        && request.body_contains_text("reloaded child complete")
                },
            )
            .await?;
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_candidates_keep_proper_subagent_routing() -> Result<()> {
    skip_if_no_network!(Ok(()));
    for (label, spawn_args, custom_provider, custom_role, model, effort) in [
        (
            "ordinary spawn",
            json!({"message": CHILD_PROMPT}),
            false,
            false,
            PARENT_MODEL,
            PARENT_EFFORT,
        ),
        (
            "identical explicit settings",
            json!({
                "message": CHILD_PROMPT,
                "model": PARENT_MODEL,
                "reasoning_effort": PARENT_EFFORT,
            }),
            false,
            false,
            PARENT_MODEL,
            PARENT_EFFORT,
        ),
        (
            "custom provider",
            json!({
                "message": CHILD_PROMPT,
                "model": CHILD_MODEL,
                "reasoning_effort": CHILD_EFFORT,
            }),
            true,
            false,
            CHILD_MODEL,
            CHILD_EFFORT,
        ),
        (
            "role override",
            json!({
                "message": CHILD_PROMPT,
                "agent_type": "routing_test",
            }),
            false,
            true,
            CHILD_MODEL,
            CHILD_EFFORT,
        ),
        (
            "service-tier-only override",
            json!({
                "message": CHILD_PROMPT,
                "service_tier": "priority",
            }),
            false,
            false,
            PARENT_MODEL,
            PARENT_EFFORT,
        ),
    ] {
        let observation = run_spawn(
            ToolVersion::V1,
            spawn_args,
            /*primary_used_percent*/ None,
            custom_provider,
            custom_role,
        )
        .await
        .with_context(|| label)?;
        assert_spawn_route(
            &observation,
            SubagentBackendRoute::ProperSubagent,
            model,
            effort,
            /*expected_agent_path*/ None,
        )?;
    }
    Ok(())
}

async fn wait_for_request(
    mock: &ResponseMock,
    label: &str,
    mut predicate: impl FnMut(&ResponsesRequest) -> bool,
) -> Result<ResponsesRequest> {
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            if let Some(request) = mock.requests().into_iter().find(&mut predicate) {
                return request;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .map_err(|_| anyhow!("timed out waiting for {label}"))
}

fn request_body_contains(request: &wiremock::Request, text: &str) -> bool {
    std::str::from_utf8(&request.body).is_ok_and(|body| body.contains(text))
}

fn wire_request_is_spawned_child(request: &wiremock::Request) -> bool {
    request
        .headers
        .get("x-codex-turn-metadata")
        .and_then(|value| value.to_str().ok())
        .is_some_and(turn_metadata_is_spawned_child)
}

fn responses_request_is_spawned_child(request: &ResponsesRequest) -> bool {
    request
        .header("x-codex-turn-metadata")
        .as_deref()
        .is_some_and(turn_metadata_is_spawned_child)
}

fn turn_metadata_is_spawned_child(metadata: &str) -> bool {
    serde_json::from_str::<Value>(metadata)
        .is_ok_and(|metadata| metadata["subagent_kind"].as_str() == Some("thread_spawn"))
}
