use super::*;
use crate::session::tests::make_session_and_context;
use chrono::TimeZone;
use codex_api::TransportError;
use codex_kimi_code::KimiStreamError;
use http::HeaderValue;
use http::StatusCode;
use pretty_assertions::assert_eq;
use std::sync::Arc;
use std::sync::Mutex;

#[tokio::test]
async fn native_retry_handler_uses_kimi_headers_backoff_and_injected_sleep() {
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

    let mut headers = HeaderMap::new();
    headers.insert("retry-after-ms", HeaderValue::from_static("2750"));
    let kimi_error = codex_api::map_api_error(classify_kimi_error_with_scheduler(
        KimiStreamError::Request(ApiError::Transport(TransportError::Http {
            status: StatusCode::TOO_MANY_REQUESTS,
            url: None,
            headers: Some(headers),
            body: Some("rate limited".to_string()),
        })),
        &scheduler,
    ));
    handle_native_sampling_retry(20, &mut 0, kimi_error, &session, &turn_context, &scheduler)
        .await
        .expect("retry");

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
    assert_eq!(delays[0], Duration::from_millis(2_750));
    assert_eq!(
        delays[1..7],
        [500, 1_000, 2_000, 4_000, 8_000, 16_000].map(Duration::from_millis)
    );
    assert_eq!(delays[7..], [Duration::from_secs(32); 14]);
}
