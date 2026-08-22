//! Provider-aware retry decisions for model sampling and local compaction requests.

use std::time::Duration;

use chrono::DateTime;
use chrono::Utc;
use codex_api::ApiError;
use codex_api::TransportError;
use codex_claude_code::DecodeError;
use codex_claude_code::NativeStreamError;
use codex_kimi_code::KimiStreamError;
use codex_model_provider_info::ModelProviderInfo;
use codex_model_provider_info::ResolvedInferencePlan;
use codex_protocol::error::CodexErr;
use codex_protocol::error::CodexErrorDetails;
use codex_protocol::openai_models::ModelInfo;
use futures::future::BoxFuture;
use http::HeaderMap;
use rand::Rng;
use tracing::warn;

use crate::session::session::Session;
use crate::session::turn_context::TurnContext;

const NATIVE_INITIAL_DELAY: Duration = Duration::from_millis(500);
const NATIVE_MAX_DELAY: Duration = Duration::from_secs(32);

pub(crate) struct RetryScheduler {
    now: Box<dyn Fn() -> DateTime<Utc> + Send + Sync>,
    jitter: Box<dyn Fn() -> f64 + Send + Sync>,
    sleep: Box<dyn Fn(Duration) -> BoxFuture<'static, ()> + Send + Sync>,
}

impl RetryScheduler {
    pub(crate) fn production() -> Self {
        Self {
            now: Box::new(Utc::now),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SamplingRetryPolicy {
    Responses { max_retries: u64 },
    NativeClaude { max_retries: u64 },
    NativeKimi { max_retries: u64 },
    LocalLlamaCpp { max_retries: u64 },
}

impl SamplingRetryPolicy {
    pub(crate) fn resolve(
        provider: &ModelProviderInfo,
        model: &ModelInfo,
    ) -> Result<Self, CodexErr> {
        let plan = provider
            .resolve_inference_plan(model)
            .map_err(|error| CodexErr::InvalidRequest(error.to_string()))?;
        Ok(match plan {
            ResolvedInferencePlan::Anthropic { route, .. } => Self::NativeClaude {
                max_retries: route.stream_max_retries,
            },
            ResolvedInferencePlan::Kimi { route, .. } => Self::NativeKimi {
                max_retries: route.stream_max_retries,
            },
            ResolvedInferencePlan::Grok { route, .. } => Self::Responses {
                max_retries: route.stream_max_retries,
            },
            ResolvedInferencePlan::LlamaCpp { route, .. } => Self::LocalLlamaCpp {
                max_retries: route.stream_max_retries,
            },
            ResolvedInferencePlan::Legacy { .. } | ResolvedInferencePlan::OpenAi { .. } => {
                Self::Responses {
                    max_retries: provider.stream_max_retries(),
                }
            }
        })
    }

    pub(crate) fn is_retryable(self, error: &CodexErr) -> bool {
        match self {
            Self::Responses { .. } | Self::LocalLlamaCpp { .. } => error.is_retryable(),
            Self::NativeClaude { .. } | Self::NativeKimi { .. } => {
                matches!(error.details(), CodexErrorDetails::Stream(_))
            }
        }
    }
}

pub(crate) fn classify_kimi_error(error: KimiStreamError) -> ApiError {
    classify_kimi_error_with_scheduler(error, &RetryScheduler::production())
}

fn classify_kimi_error_with_scheduler(
    error: KimiStreamError,
    scheduler: &RetryScheduler,
) -> ApiError {
    match error {
        KimiStreamError::Request(ApiError::Transport(transport)) => {
            classify_kimi_transport_error(transport, (scheduler.now)())
        }
        KimiStreamError::Request(ApiError::ServerOverloaded) => {
            retryable("server overloaded", None)
        }
        KimiStreamError::Request(ApiError::RateLimit(message)) => retryable(message, None),
        KimiStreamError::Request(error) => error,
        KimiStreamError::IdleTimeout => retryable("native Kimi stream idle timeout", None),
        KimiStreamError::Transport(message) => retryable(message, None),
        KimiStreamError::ThinkingOnlyStop => {
            retryable("native Kimi stop contained only thinking", None)
        }
        KimiStreamError::RetryableStream(message) => retryable(message, None),
        KimiStreamError::Decode(error) => ApiError::InvalidRequest {
            message: error.to_string(),
        },
        KimiStreamError::Cancelled => ApiError::InvalidRequest {
            message: "native Kimi stream was cancelled".to_string(),
        },
        KimiStreamError::Response(error) => ApiError::InvalidRequest {
            message: error.to_string(),
        },
        KimiStreamError::InvalidRequest(message) => ApiError::InvalidRequest { message },
        KimiStreamError::UsageOverflow => ApiError::InvalidRequest {
            message: "native Kimi usage exceeded canonical integer bounds".to_string(),
        },
    }
}

fn classify_kimi_transport_error(error: TransportError, now: DateTime<Utc>) -> ApiError {
    match error {
        TransportError::Http {
            status,
            url,
            headers,
            body,
        } => {
            let body_text = body.as_deref().unwrap_or_default();
            if is_context_overflow(body_text) {
                ApiError::ContextWindowExceeded
            } else if is_quota_exhaustion(body_text) {
                ApiError::QuotaExceeded
            } else if matches!(status.as_u16(), 408 | 409 | 429 | 529) || status.is_server_error() {
                retryable(
                    format!("native Kimi request failed with HTTP {status}"),
                    headers
                        .as_ref()
                        .and_then(|headers| retry_header_delay(headers, now)),
                )
            } else {
                ApiError::Transport(TransportError::Http {
                    status,
                    url,
                    headers,
                    body,
                })
            }
        }
        TransportError::Timeout | TransportError::Network(_) => retryable(error.to_string(), None),
        TransportError::RetryLimit | TransportError::Build(_) => ApiError::Transport(error),
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

pub(crate) fn classify_native_error(error: NativeStreamError) -> ApiError {
    classify_native_error_with_scheduler(error, &RetryScheduler::production())
}

fn classify_native_error_with_scheduler(
    error: NativeStreamError,
    scheduler: &RetryScheduler,
) -> ApiError {
    match error {
        NativeStreamError::Request(ApiError::Transport(transport)) => {
            classify_transport_error(transport, (scheduler.now)())
        }
        NativeStreamError::Request(ApiError::ServerOverloaded) => {
            retryable("server overloaded", None)
        }
        NativeStreamError::Request(error) => error,
        NativeStreamError::IdleTimeout => retryable("native Claude stream idle timeout", None),
        NativeStreamError::Transport(message) => retryable(message, None),
        NativeStreamError::Decode(DecodeError::EmptyStream) => {
            retryable("native Claude stream was empty", None)
        }
        NativeStreamError::Decode(DecodeError::PrematureEof { expected }) => retryable(
            format!("native Claude stream ended before {expected}"),
            None,
        ),
        NativeStreamError::Decode(DecodeError::ProviderError {
            error_type,
            message,
        }) => classify_provider_error(&error_type, message),
        NativeStreamError::Decode(error) => ApiError::InvalidRequest {
            message: error.to_string(),
        },
        NativeStreamError::Cancelled => ApiError::InvalidRequest {
            message: "native Claude stream was cancelled".to_string(),
        },
        NativeStreamError::InvalidRequest(message) => ApiError::InvalidRequest { message },
    }
}

fn classify_transport_error(error: TransportError, now: DateTime<Utc>) -> ApiError {
    match error {
        TransportError::Http {
            status,
            url,
            headers,
            body,
        } => {
            let body_text = body.as_deref().unwrap_or_default();
            if is_context_overflow(body_text) {
                ApiError::ContextWindowExceeded
            } else if is_quota_exhaustion(body_text) {
                ApiError::QuotaExceeded
            } else if matches!(status.as_u16(), 408 | 409 | 429) || status.is_server_error() {
                retryable(
                    format!("native Claude request failed with HTTP {status}"),
                    headers
                        .as_ref()
                        .and_then(|headers| retry_header_delay(headers, now)),
                )
            } else {
                ApiError::Transport(TransportError::Http {
                    status,
                    url,
                    headers,
                    body,
                })
            }
        }
        TransportError::Timeout | TransportError::Network(_) => retryable(error.to_string(), None),
        TransportError::RetryLimit | TransportError::Build(_) => ApiError::Transport(error),
    }
}

fn classify_provider_error(error_type: &str, message: String) -> ApiError {
    if is_context_overflow(&message) {
        ApiError::ContextWindowExceeded
    } else if matches!(error_type, "billing_error" | "quota_exceeded")
        || is_quota_exhaustion(&message)
    {
        ApiError::QuotaExceeded
    } else if matches!(
        error_type,
        "overloaded_error" | "rate_limit_error" | "api_error" | "timeout_error"
    ) {
        retryable(format!("native Claude {error_type}: {message}"), None)
    } else {
        ApiError::InvalidRequest {
            message: format!("native Claude {error_type}: {message}"),
        }
    }
}

fn retryable(message: impl Into<String>, delay: Option<Duration>) -> ApiError {
    ApiError::Retryable {
        message: message.into(),
        delay,
    }
}

fn retry_header_delay(headers: &HeaderMap, now: DateTime<Utc>) -> Option<Duration> {
    if let Some(milliseconds) = headers
        .get("retry-after-ms")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
    {
        return Some(Duration::from_millis(milliseconds));
    }
    let value = headers.get(http::header::RETRY_AFTER)?.to_str().ok()?;
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    let retry_at = DateTime::parse_from_rfc2822(value)
        .ok()?
        .with_timezone(&Utc);
    (retry_at > now)
        .then(|| (retry_at - now).to_std().ok())
        .flatten()
}

fn is_context_overflow(message: &str) -> bool {
    contains_any(
        message,
        "context window|context limit|maximum context length|prompt is too long|too many input tokens",
    )
}

fn is_quota_exhaustion(message: &str) -> bool {
    contains_any(
        message,
        "quota_exceeded|insufficient_quota|billing_error|credit balance is too low|exceeded your current quota|usage_limit",
    )
}

fn contains_any(message: &str, needles: &str) -> bool {
    let message = message.to_ascii_lowercase();
    needles.split('|').any(|needle| message.contains(needle))
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
