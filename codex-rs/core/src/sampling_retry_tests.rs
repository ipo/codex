use super::*;
use crate::session::tests::make_session_and_context;
use chrono::TimeZone;
use http::HeaderValue;
use http::StatusCode;
use pretty_assertions::assert_eq;
use std::sync::Arc;
use std::sync::Mutex;

fn http_error(status: StatusCode, body: &str, headers: Option<HeaderMap>) -> NativeStreamError {
    NativeStreamError::Request(ApiError::Transport(TransportError::Http {
        status,
        url: None,
        headers,
        body: Some(body.to_string()),
    }))
}

#[tokio::test]
async fn native_retry_handler_uses_headers_jittered_backoff_and_cap() {
    let now = Utc.with_ymd_and_hms(2026, 8, 2, 0, 0, 0).unwrap();
    let delays = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&delays);
    let scheduler = RetryScheduler {
        now: Box::new(move || now),
        jitter: Box::new(|| 1.0),
        sleep: Box::new(move |delay| {
            recorded.lock().expect("delays").push(delay);
            Box::pin(async {})
        }),
    };
    let (session, turn_context) = make_session_and_context().await;

    for (name, value) in [
        ("retry-after-ms", "1250"),
        ("retry-after", "3"),
        ("retry-after", "Sun, 02 Aug 2026 00:00:05 GMT"),
    ] {
        let mut headers = HeaderMap::new();
        headers.insert(name, HeaderValue::from_static(value));
        let error = codex_api::map_api_error(classify_native_error_with_scheduler(
            http_error(StatusCode::TOO_MANY_REQUESTS, "rate limited", Some(headers)),
            &scheduler,
        ));
        let mut retries = 0;
        handle_native_sampling_retry(20, &mut retries, error, &session, &turn_context, &scheduler)
            .await
            .expect("retry");
    }
    let mut retries = 0;
    for _ in 0..20 {
        handle_native_sampling_retry(
            20,
            &mut retries,
            CodexErr::Stream("temporary".into()),
            &session,
            &turn_context,
            &scheduler,
        )
        .await
        .expect("retry");
    }
    let delays = delays.lock().expect("delays");
    assert_eq!(
        delays[..3],
        [1_250, 3_000, 5_000].map(Duration::from_millis)
    );
    assert_eq!(
        delays[3..9],
        [500, 1_000, 2_000, 4_000, 8_000, 16_000].map(Duration::from_millis)
    );
    assert_eq!(delays[9..], [Duration::from_secs(32); 14]);
}

#[test]
fn exhausted_and_refused_terminals_are_nonretryable() {
    for (outcome, message) in [
        (
            TerminalOutcome::OutputExhausted,
            "model output limit reached before the turn completed",
        ),
        (
            TerminalOutcome::Refusal,
            "model refused to complete the turn for safety reasons",
        ),
    ] {
        let error = codex_api::map_api_error(classify_native_error(
            NativeStreamError::UnsuccessfulTerminal(outcome),
        ));
        assert!(!error.is_retryable());
        assert_eq!(error.to_string(), message);
    }
}

#[test]
fn context_overflow_is_nonretryable_and_distinct_from_rate_limits() {
    let overflow = codex_api::map_api_error(classify_native_error(http_error(
        StatusCode::BAD_REQUEST,
        "prompt is too long: too many input tokens",
        None,
    )));
    assert!(!overflow.is_retryable());
    assert!(matches!(
        overflow.details(),
        codex_protocol::error::CodexErrorDetails::ContextWindowExceeded
    ));

    let mut headers = HeaderMap::new();
    headers.insert("retry-after-ms", HeaderValue::from_static("250"));
    let retryable = codex_api::map_api_error(classify_native_error(http_error(
        StatusCode::TOO_MANY_REQUESTS,
        "rate limited",
        Some(headers),
    )));
    assert!(retryable.is_retryable());
    assert_eq!(retryable.retry_delay(), Some(Duration::from_millis(250)));
}
