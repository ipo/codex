#![cfg(unix)]

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use app_test_support::TestAppServer;
use app_test_support::create_mock_responses_server_sequence_unchecked;
use app_test_support::create_shell_command_sse_response;
use app_test_support::to_response;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use codex_app_server_protocol::ItemStartedNotification;
use codex_app_server_protocol::JSONRPCNotification;
use codex_app_server_protocol::JSONRPCResponse;
use codex_app_server_protocol::ProcessOutputDeltaNotification;
use codex_app_server_protocol::ProcessOutputStream;
use codex_app_server_protocol::ProcessTerminalSize;
use codex_app_server_protocol::RequestId;
use codex_app_server_protocol::ThreadBackgroundTerminalsListParams;
use codex_app_server_protocol::ThreadBackgroundTerminalsListResponse;
use codex_app_server_protocol::ThreadItem;
use codex_app_server_protocol::ThreadStartParams;
use codex_app_server_protocol::ThreadStartResponse;
use codex_app_server_protocol::ThreadTerminalOpenParams;
use codex_app_server_protocol::ThreadTerminalOpenResponse;
use codex_app_server_protocol::ThreadTerminalResizeParams;
use codex_app_server_protocol::ThreadTerminalResizeResponse;
use codex_app_server_protocol::ThreadTerminalSource;
use codex_app_server_protocol::ThreadTerminalStatusKind;
use codex_app_server_protocol::ThreadTerminalWriteParams;
use codex_app_server_protocol::ThreadTerminalWriteResponse;
use codex_app_server_protocol::TurnStartParams;
use codex_app_server_protocol::TurnStartResponse;
use codex_app_server_protocol::UserInput as V2UserInput;
use pretty_assertions::assert_eq;
use std::path::Path;
use tempfile::TempDir;
use tokio::time::Duration;
use tokio::time::Instant;
use tokio::time::sleep;
use tokio::time::timeout;

const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(10);

#[tokio::test]
async fn thread_terminal_facade_opens_writes_resizes_and_reuses_labeled_shell() -> Result<()> {
    let tmp = TempDir::new()?;
    let codex_home = tmp.path().join("codex_home");
    std::fs::create_dir(&codex_home)?;
    let working_directory = tmp.path().join("workdir");
    std::fs::create_dir(&working_directory)?;

    let server =
        create_mock_responses_server_sequence_unchecked(vec![create_shell_command_sse_response(
            vec!["sleep".to_string(), "30".to_string()],
            Some(&working_directory),
            Some(30_000),
            "call_sleep",
        )?])
        .await;
    create_config_toml(&codex_home, &server.uri())?;

    let mut mcp = TestAppServer::new(&codex_home).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let thread_req = mcp
        .send_thread_start_request(ThreadStartParams {
            model: Some("mock-model".to_string()),
            ..Default::default()
        })
        .await?;
    let thread_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(thread_req)),
    )
    .await??;
    let ThreadStartResponse { thread, .. } = to_response::<ThreadStartResponse>(thread_resp)?;

    let turn_req = mcp
        .send_turn_start_request(TurnStartParams {
            thread_id: thread.id.clone(),
            client_user_message_id: None,
            input: vec![V2UserInput::Text {
                text: "run sleep".to_string(),
                text_elements: Vec::new(),
            }],
            cwd: Some(working_directory.clone()),
            ..Default::default()
        })
        .await?;
    let turn_resp: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(turn_req)),
    )
    .await??;
    let TurnStartResponse { turn } = to_response::<TurnStartResponse>(turn_resp)?;
    wait_for_command_execution_started(&mut mcp).await?;

    let initial_size = ProcessTerminalSize { rows: 20, cols: 80 };
    let open = open_terminal(&mut mcp, &thread.id, "default", initial_size).await?;
    assert_eq!(open.label, "default");
    assert_eq!(open.cwd.as_path(), working_directory.as_path());
    assert_eq!(open.source, ThreadTerminalSource::SharedTerminal);
    assert_eq!(open.tty, true);
    assert_eq!(open.size, Some(initial_size));
    assert_eq!(open.status.kind, ThreadTerminalStatusKind::Running);
    assert_eq!(open.status.exit_code, None);
    assert!(!open.command.is_empty());
    let process_id = open.process_id.clone();

    let marker = "codex-thread-terminal-marker";
    let output = write_command_and_wait_for_output(
        &mut mcp,
        &thread.id,
        &process_id,
        &format!("printf '{marker}\\n'\n"),
        marker,
    )
    .await?;
    assert!(
        output.contains(marker),
        "expected terminal output to contain marker, got {output:?}"
    );

    let first_resize = resize_terminal(
        &mut mcp,
        &thread.id,
        &process_id,
        ProcessTerminalSize {
            rows: 24,
            cols: 100,
        },
    )
    .await?;
    assert_eq!(
        first_resize.size,
        Some(ProcessTerminalSize {
            rows: 24,
            cols: 100,
        })
    );

    let second_resize = resize_terminal(
        &mut mcp,
        &thread.id,
        &process_id,
        ProcessTerminalSize {
            rows: 30,
            cols: 120,
        },
    )
    .await?;
    assert_eq!(
        second_resize.size,
        Some(ProcessTerminalSize {
            rows: 30,
            cols: 120,
        })
    );

    let reopened = open_terminal(&mut mcp, &thread.id, "default", initial_size).await?;
    assert_eq!(reopened.process_id, process_id);

    let list = list_background_terminals(&mut mcp, &thread.id).await?;
    let shared = list
        .data
        .iter()
        .find(|terminal| terminal.process_id == process_id)
        .context("shared terminal should appear in background terminal list")?;
    assert_eq!(shared.source, ThreadTerminalSource::SharedTerminal);
    assert_eq!(shared.label.as_deref(), Some("default"));
    assert_eq!(shared.tty, true);
    assert_eq!(
        shared.size,
        Some(ProcessTerminalSize {
            rows: 30,
            cols: 120,
        })
    );
    assert_eq!(shared.status.kind, ThreadTerminalStatusKind::Running);

    mcp.interrupt_turn_and_wait_for_aborted(
        thread.id.clone(),
        turn.id.clone(),
        DEFAULT_READ_TIMEOUT,
    )
    .await?;

    Ok(())
}

async fn open_terminal(
    mcp: &mut TestAppServer,
    thread_id: &str,
    label: &str,
    size: ProcessTerminalSize,
) -> Result<codex_app_server_protocol::ThreadTerminalInfo> {
    let request_id = mcp
        .send_thread_terminal_open_request(ThreadTerminalOpenParams {
            thread_id: thread_id.to_string(),
            label: label.to_string(),
            size: Some(size),
        })
        .await?;
    let response = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(request_id)),
    )
    .await??;
    let ThreadTerminalOpenResponse { terminal } =
        to_response::<ThreadTerminalOpenResponse>(response)?;
    Ok(terminal)
}

async fn resize_terminal(
    mcp: &mut TestAppServer,
    thread_id: &str,
    process_id: &str,
    size: ProcessTerminalSize,
) -> Result<codex_app_server_protocol::ThreadTerminalInfo> {
    let request_id = mcp
        .send_thread_terminal_resize_request(ThreadTerminalResizeParams {
            thread_id: thread_id.to_string(),
            process_id: process_id.to_string(),
            size,
        })
        .await?;
    let response = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(request_id)),
    )
    .await??;
    let ThreadTerminalResizeResponse { terminal } =
        to_response::<ThreadTerminalResizeResponse>(response)?;
    Ok(terminal)
}

async fn write_command_and_wait_for_output(
    mcp: &mut TestAppServer,
    thread_id: &str,
    process_id: &str,
    command: &str,
    expected: &str,
) -> Result<String> {
    let deadline = Instant::now() + DEFAULT_READ_TIMEOUT;
    let mut next_input = Some(command.to_string());
    let mut output = String::new();

    while Instant::now() < deadline {
        write_terminal(mcp, thread_id, process_id, next_input.take()).await?;
        while let Some(delta) = read_terminal_output_delta(mcp, process_id).await? {
            output.push_str(&delta);
            if output.contains(expected) {
                return Ok(output);
            }
        }
        sleep(Duration::from_millis(50)).await;
    }

    bail!("timed out waiting for terminal output {expected:?}; collected {output:?}");
}

async fn write_terminal(
    mcp: &mut TestAppServer,
    thread_id: &str,
    process_id: &str,
    input: Option<String>,
) -> Result<()> {
    let request_id = mcp
        .send_thread_terminal_write_request(ThreadTerminalWriteParams {
            thread_id: thread_id.to_string(),
            process_id: process_id.to_string(),
            delta_base64: input.map(|input| STANDARD.encode(input)),
        })
        .await?;
    let response = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(request_id)),
    )
    .await??;
    let _: ThreadTerminalWriteResponse = to_response::<ThreadTerminalWriteResponse>(response)?;
    Ok(())
}

async fn read_terminal_output_delta(
    mcp: &mut TestAppServer,
    process_id: &str,
) -> Result<Option<String>> {
    let notification = match timeout(
        Duration::from_millis(100),
        mcp.read_stream_until_matching_notification("process/outputDelta", |notification| {
            notification.method == "process/outputDelta"
        }),
    )
    .await
    {
        Ok(result) => result?,
        Err(_) => return Ok(None),
    };
    let delta = parse_process_output_delta(notification)?;
    if delta.process_handle != process_id {
        return Ok(None);
    }
    assert_eq!(delta.stream, ProcessOutputStream::Stdout);
    let bytes = STANDARD
        .decode(delta.delta_base64)
        .context("decode terminal output delta")?;
    Ok(Some(String::from_utf8_lossy(&bytes).into_owned()))
}

fn parse_process_output_delta(
    notification: JSONRPCNotification,
) -> Result<ProcessOutputDeltaNotification> {
    let params = notification
        .params
        .context("process/outputDelta notification should include params")?;
    serde_json::from_value(params).context("deserialize process/outputDelta notification")
}

async fn list_background_terminals(
    mcp: &mut TestAppServer,
    thread_id: &str,
) -> Result<ThreadBackgroundTerminalsListResponse> {
    let request_id = mcp
        .send_thread_background_terminals_list_request(ThreadBackgroundTerminalsListParams {
            thread_id: thread_id.to_string(),
            cursor: None,
            limit: None,
        })
        .await?;
    let response = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(request_id)),
    )
    .await??;
    to_response::<ThreadBackgroundTerminalsListResponse>(response)
}

async fn wait_for_command_execution_started(mcp: &mut TestAppServer) -> Result<()> {
    loop {
        let notification = timeout(
            DEFAULT_READ_TIMEOUT,
            mcp.read_stream_until_notification_message("item/started"),
        )
        .await??;
        let params = notification
            .params
            .context("item/started notification should include params")?;
        let started: ItemStartedNotification =
            serde_json::from_value(params).context("deserialize item/started notification")?;
        if matches!(started.item, ThreadItem::CommandExecution { .. }) {
            return Ok(());
        }
    }
}

fn create_config_toml(codex_home: &Path, server_uri: &str) -> std::io::Result<()> {
    std::fs::write(
        codex_home.join("config.toml"),
        format!(
            r#"
model = "mock-model"
approval_policy = "never"
sandbox_mode = "workspace-write"

model_provider = "mock_provider"

[model_providers.mock_provider]
name = "Mock provider for test"
base_url = "{server_uri}/v1"
wire_api = "responses"
request_max_retries = 0
stream_max_retries = 0
"#
        ),
    )
}
