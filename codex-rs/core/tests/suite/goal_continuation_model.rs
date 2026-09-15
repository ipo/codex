use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use codex_analytics::AnalyticsEventsClient;
use codex_core::TurnInputRequest;
use codex_core::config::Config;
use codex_extension_api::ExtensionRegistryBuilder;
use codex_features::Feature;
use codex_goal_extension::GoalExtensionConfig;
use codex_goal_extension::GoalService;
use codex_goal_extension::install_with_backend;
use codex_model_provider_info::ModelProviderWireRoute;
use codex_model_provider_info::WireApi;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::models::PermissionProfile;
use codex_protocol::protocol::AskForApproval;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::protocol::ThreadSettingsOverrides;
use codex_protocol::user_input::UserInput;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_custom_tool_call;
use core_test_support::responses::ev_response_created;
use core_test_support::responses::mount_sse_once;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::skip_if_wine_exec;
use core_test_support::test_codex::test_codex;
use core_test_support::test_codex::turn_permission_fields;
use core_test_support::wait_for_event_with_timeout;
use pretty_assertions::assert_eq;
use tokio::time::timeout;

const KIMI: &str = "kimi/k3-256k";
const TERRA: &str = "gpt-5.6-terra";
const GOAL_OBJECTIVE: &str = "preserve the terra exec history";

#[cfg_attr(windows, ignore = "no exec_command on Windows")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn active_goal_continuation_keeps_effective_terra_model() -> Result<()> {
    skip_if_no_network!(Ok(()));
    skip_if_wine_exec!(Ok(()), "code-mode exec is unavailable under Wine-exec");

    let server = start_mock_server().await;
    let first = mount_sse_once(
        &server,
        sse(vec![
            ev_response_created("resp-1"),
            ev_custom_tool_call("call-1", "exec", "text('ok');"),
            ev_completed("resp-1"),
        ]),
    )
    .await;
    let second = mount_sse_once(
        &server,
        sse(vec![
            ev_assistant_message("msg-1", "terra turn done"),
            ev_completed("resp-2"),
        ]),
    )
    .await;
    let continuation_mock = mount_sse_once(
        &server,
        sse(vec![
            ev_assistant_message("msg-2", "continuation done"),
            ev_completed("resp-3"),
        ]),
    )
    .await;

    let kimi_base_url = format!("{}/v1", server.uri());
    let builder = test_codex().with_model(KIMI).with_config(move |config| {
        config
            .features
            .enable(Feature::CodeMode)
            .expect("test config should allow CodeMode");
        config
            .features
            .enable(Feature::Goals)
            .expect("test config should allow Goals");
        config.model_provider.supports_websockets = false;
        config.model_provider.wire_routes.insert(
            "kimi_code".to_string(),
            ModelProviderWireRoute {
                wire_api: WireApi::ChatCompletions,
                dialect: InferenceDialect::Kimi,
                base_url: kimi_base_url,
                request_path: "chat/completions".to_string(),
                query_params: None,
                request_max_retries: None,
                stream_max_retries: None,
                stream_idle_timeout_ms: None,
            },
        );
    });
    let test = builder
        .with_extension_factory(|thread_manager, state_db| {
            let mut registry = ExtensionRegistryBuilder::<Config>::new();
            if let Some(state_db) = state_db {
                install_with_backend(
                    &mut registry,
                    state_db,
                    AnalyticsEventsClient::disabled(),
                    /*metrics_client*/ None,
                    thread_manager,
                    Arc::new(GoalService::new()),
                    |config: &Config| GoalExtensionConfig {
                        enabled: config.features.enabled(Feature::Goals),
                        max_goal_token_budget: config.max_goal_token_budget,
                    },
                );
            }
            Arc::new(registry.build())
        })
        .build(&server)
        .await?;

    assert_eq!(test.session_configured.model, KIMI);
    let state_db = test
        .codex
        .state_db()
        .expect("goal continuation requires sqlite state");
    state_db
        .thread_goals()
        .replace_thread_goal(
            test.session_configured.thread_id,
            GOAL_OBJECTIVE,
            codex_state::ThreadGoalStatus::Active,
            /*token_budget*/ None,
        )
        .await?;

    let (sandbox_policy, permission_profile) =
        turn_permission_fields(PermissionProfile::Disabled, test.cwd_path());
    let submission = test
        .codex
        .start_or_steer_turn(
            TurnInputRequest::user_input(vec![UserInput::Text {
                text: "use exec".into(),
                text_elements: Vec::new(),
            }])
            .with_thread_settings(ThreadSettingsOverrides {
                approval_policy: Some(AskForApproval::Never),
                sandbox_policy: Some(sandbox_policy),
                permission_profile,
                model: Some(TERRA.to_string()),
                ..Default::default()
            }),
        )
        .await?;
    assert!(
        matches!(submission, codex_core::TurnInputSubmission::Started { .. }),
        "terra turn should start: {submission:?}"
    );

    wait_for_event_with_timeout(
        &test.codex,
        |event| matches!(event, EventMsg::TurnStarted(_)),
        Duration::from_secs(/*secs*/ 10),
    )
    .await;
    timeout(Duration::from_secs(/*secs*/ 10), async {
        loop {
            if !first.requests().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(/*millis*/ 50)).await;
        }
    })
    .await
    .expect("terra turn should sample before restoring kimi");

    test.codex
        .submit(Op::ThreadSettings {
            thread_settings: ThreadSettingsOverrides {
                model: Some(KIMI.to_string()),
                ..Default::default()
            },
        })
        .await?;

    let mut applied_models = Vec::new();
    let mut turn_completes = 0;
    let mut errors = Vec::new();
    timeout(Duration::from_secs(/*secs*/ 60), async {
        loop {
            let event = test
                .codex
                .next_event()
                .await
                .expect("event stream should stay open");
            match event.msg {
                EventMsg::ThreadSettingsApplied(applied) => {
                    applied_models.push(applied.thread_settings.model.clone());
                }
                EventMsg::Error(error) => errors.push(error.message),
                EventMsg::TurnComplete(_) => {
                    turn_completes += 1;
                    if turn_completes == 2 {
                        break;
                    }
                }
                _ => {}
            }
        }
    })
    .await
    .expect("timed out waiting for terra turn and goal continuation");

    assert!(
        errors.iter().all(|message| {
            !message.contains("unsupported native Kimi history item")
                && !message.contains("freeform tool call")
        }),
        "continuation must not replay exec history as Kimi: {errors:?}"
    );
    assert_eq!(test.codex.thread_settings_snapshot().await.model, KIMI);
    assert_eq!(applied_models, vec![KIMI.to_string()]);

    assert!(
        !first.requests().is_empty(),
        "terra turn should sample: {errors:?}"
    );
    assert_eq!(first.single_request().body_json()["model"], TERRA);
    assert!(
        !second.requests().is_empty(),
        "terra exec follow-up should sample: {errors:?}"
    );
    let continuation = continuation_mock.single_request();
    assert_eq!(continuation.body_json()["model"], TERRA);
    assert!(continuation.body_contains_text("text('ok');"));
    assert!(
        continuation.body_contains_text("Continue working toward the active thread goal"),
        "continuation request should include the goal continuation prompt: {}",
        continuation.body_json()
    );
    assert!(
        continuation.body_contains_text(GOAL_OBJECTIVE),
        "continuation request should include the active goal objective"
    );
    Ok(())
}
