//! Shared retry and transport fallback decisions for Responses requests.

use std::time::Duration;

use crate::client::ModelClientSession;
use crate::session::session::Session;
use crate::session::turn_context::TurnContext;
use crate::util::backoff;
use crate::util::positive_jitter;
use codex_protocol::error::CodexErr;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::WarningEvent;
use tokio_util::sync::CancellationToken;
use tracing::warn;

// Keep capacity recovery bounded while giving a continuously running turn time to reach a healthy
// backend. Jitter is added separately and never shortens these base delays.
const SERVER_OVERLOADED_RETRY_DELAYS: [Duration; 5] = [
    Duration::from_secs(10),
    Duration::from_secs(20),
    Duration::from_secs(40),
    Duration::from_secs(80),
    Duration::from_secs(160),
];
const SERVER_OVERLOADED_MAX_RETRIES: u64 = SERVER_OVERLOADED_RETRY_DELAYS.len() as u64;

#[derive(Debug, Clone, Copy)]
pub(crate) enum ResponsesStreamRequest {
    Sampling,
    RemoteCompactionV2,
}

pub(crate) fn should_retry_server_overload_in_core(
    err: &CodexErr,
    turn_context: &TurnContext,
) -> bool {
    matches!(err, CodexErr::ServerOverloaded)
        && !crate::guardian::is_guardian_reviewer_source(&turn_context.session_source)
}

/// Handles a retryable stream error and returns `Ok(())` when the caller should
/// retry the request loop.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn handle_retryable_response_stream_error(
    retries: &mut u64,
    max_retries: u64,
    err: CodexErr,
    client_session: &mut ModelClientSession,
    sess: &Session,
    turn_context: &TurnContext,
    request: ResponsesStreamRequest,
    cancellation_token: CancellationToken,
) -> Result<(), CodexErr> {
    let is_server_overloaded = matches!(&err, CodexErr::ServerOverloaded);
    let max_retries = if is_server_overloaded {
        SERVER_OVERLOADED_MAX_RETRIES
    } else {
        max_retries
    };
    if !is_server_overloaded
        && *retries >= max_retries
        && client_session.try_switch_fallback_transport(
            &turn_context.session_telemetry,
            &turn_context.model_info,
        )
    {
        sess.send_event(
            turn_context,
            EventMsg::Warning(WarningEvent {
                message: format!("Falling back from WebSockets to HTTPS transport. {err:#}"),
            }),
        )
        .await;
        *retries = 0;
        return Ok(());
    }

    if *retries < max_retries {
        *retries += 1;
        let retry_count = *retries;
        let delay = match &err {
            CodexErr::ServerOverloaded => positive_jitter(
                SERVER_OVERLOADED_RETRY_DELAYS[retry_count.saturating_sub(1) as usize],
            ),
            CodexErr::Stream(_, requested_delay) => {
                requested_delay.unwrap_or_else(|| backoff(retry_count))
            }
            _ => backoff(retry_count),
        };
        log_retry(request, turn_context, &err, retry_count, max_retries, delay);

        // In release builds, hide the first websocket retry notification to reduce noisy
        // transient reconnect messages. In debug builds, keep full visibility for diagnosis.
        let report_error = is_server_overloaded
            || retry_count > 1
            || cfg!(debug_assertions)
            || !sess.services.model_client.responses_websocket_enabled();
        if report_error {
            // Surface retry information to any UI/front-end so the user understands what is
            // happening instead of staring at a seemingly frozen screen.
            sess.notify_stream_error(
                turn_context,
                format!("Reconnecting... {retry_count}/{max_retries}"),
                err,
            )
            .await;
        }
        return tokio::select! {
            () = tokio::time::sleep(delay) => Ok(()),
            () = cancellation_token.cancelled() => Err(CodexErr::TurnAborted),
        };
    }

    Err(err)
}

fn log_retry(
    request: ResponsesStreamRequest,
    turn_context: &TurnContext,
    err: &CodexErr,
    retries: u64,
    max_retries: u64,
    delay: Duration,
) {
    match request {
        ResponsesStreamRequest::Sampling => {
            warn!(
                turn_id = %turn_context.sub_id,
                retries,
                max_retries,
                sampling_error = %err,
                "stream disconnected - retrying sampling request ({retries}/{max_retries} in {delay:?})...",
            );
        }
        ResponsesStreamRequest::RemoteCompactionV2 => {
            warn!(
                turn_id = %turn_context.sub_id,
                retries,
                max_retries,
                compact_error = %err,
                "remote compaction v2 stream failed; retrying request after delay"
            );
        }
    }
}

#[cfg(test)]
#[path = "responses_retry_tests.rs"]
mod tests;
