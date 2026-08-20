use super::*;
use crate::session::tests::make_session_and_context;
use chrono::TimeZone;
use codex_api::TransportError;
use codex_kimi_code::KimiStreamError;
use codex_model_provider_info::CLAUDEFLARE_PROVIDER_ID;
use codex_model_provider_info::built_in_model_providers;
use codex_models_manager::bundled_models_response;
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
async fn live_native_retry_handler_uses_injected_headers_backoff_and_sleep() {
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
    macro_rules! retry {
        ($retries:expr, $error:expr) => {
            handle_native_sampling_retry(20, $retries, $error, &session, &turn_context, &scheduler)
                .await
                .expect("retry")
        };
    }

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
        retry!(&mut 0, error);
    }
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
    retry!(&mut 0, kimi_error);
    let mut retries = 0;
    for _ in 0..20 {
        retry!(&mut retries, CodexErr::Stream("temporary".into()));
    }
    let delays = delays.lock().expect("delays");
    assert_eq!(
        delays[..4],
        [1_250, 3_000, 5_000, 2_750].map(Duration::from_millis)
    );
    assert_eq!(
        delays[4..10],
        [500, 1_000, 2_000, 4_000, 8_000, 16_000].map(Duration::from_millis)
    );
    assert_eq!(delays[10..], [Duration::from_secs(32); 14]);
}

#[test]
fn grok_uses_route_scoped_responses_retry_policy_without_retrying_cancellation() {
    let mut provider = built_in_model_providers(/*openai_base_url*/ None)
        .remove(CLAUDEFLARE_PROVIDER_ID)
        .expect("Claudeflare provider");
    provider.stream_max_retries = Some(2);
    provider
        .wire_routes
        .get_mut("grok")
        .expect("Grok route")
        .stream_max_retries = Some(7);
    let model = bundled_models_response()
        .expect("bundled models")
        .models
        .into_iter()
        .find(|model| model.slug == "xai/grok-4.6")
        .expect("Grok model");

    let policy = SamplingRetryPolicy::resolve(&provider, &model).expect("Grok retry policy");
    assert_eq!(policy, SamplingRetryPolicy::Responses { max_retries: 7 });
    assert!(policy.is_retryable(&CodexErr::Stream("premature closure".to_string())));
    assert!(!policy.is_retryable(&CodexErr::Interrupted));
}
