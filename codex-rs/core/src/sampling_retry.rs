//! Retry scheduling shared by native model transports.

use std::time::Duration;

use codex_protocol::error::CodexErr;
use futures::future::BoxFuture;
use rand::Rng;
use tracing::warn;

use crate::session::session::Session;
use crate::session::turn_context::TurnContext;

const NATIVE_INITIAL_DELAY: Duration = Duration::from_millis(500);
const NATIVE_MAX_DELAY: Duration = Duration::from_secs(32);

pub(crate) struct RetryScheduler {
    jitter: Box<dyn Fn() -> f64 + Send + Sync>,
    sleep: Box<dyn Fn(Duration) -> BoxFuture<'static, ()> + Send + Sync>,
}

impl RetryScheduler {
    pub(crate) fn production() -> Self {
        Self {
            jitter: Box::new(|| rand::rng().random_range(0.9..1.1)),
            sleep: Box::new(|delay| Box::pin(tokio::time::sleep(delay))),
        }
    }

    pub(crate) fn delay(&self, error: &CodexErr, retry_count: u64) -> Duration {
        error
            .retry_delay()
            .unwrap_or_else(|| native_backoff(retry_count, (self.jitter)()))
    }
}

pub(crate) async fn handle_native_sampling_retry(
    max_retries: u64,
    retries: &mut u64,
    error: CodexErr,
    sess: &Session,
    turn_context: &TurnContext,
    scheduler: &RetryScheduler,
) -> Result<(), CodexErr> {
    if *retries >= max_retries {
        return Err(error);
    }
    *retries += 1;
    let retry_count = *retries;
    let delay = scheduler.delay(&error, retry_count);
    warn!(
        turn_id = %turn_context.sub_id,
        retries = retry_count,
        max_retries,
        sampling_error = %error,
        "stream disconnected - retrying sampling request ({retry_count}/{max_retries} in {delay:?})...",
    );
    let report_error = retry_count > 1
        || cfg!(debug_assertions)
        || !sess
            .services
            .model_client
            .responses_websocket_enabled(&turn_context.model_info);
    if report_error {
        sess.notify_stream_error(
            turn_context,
            format!("Reconnecting... {retry_count}/{max_retries}"),
            error,
        )
        .await;
    }
    (scheduler.sleep)(delay).await;
    Ok(())
}

fn native_backoff(retry_count: u64, jitter: f64) -> Duration {
    let exponent = retry_count.saturating_sub(1).min(63) as u32;
    let base = NATIVE_INITIAL_DELAY
        .saturating_mul(2u32.saturating_pow(exponent))
        .min(NATIVE_MAX_DELAY);
    base.mul_f64(jitter).min(NATIVE_MAX_DELAY)
}

#[cfg(test)]
#[path = "sampling_retry_tests.rs"]
mod tests;
