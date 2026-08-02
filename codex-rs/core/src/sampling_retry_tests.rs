use super::*;
use crate::session::tests::make_session_and_context;
use pretty_assertions::assert_eq;
use std::sync::Arc;
use std::sync::Mutex;

#[tokio::test]
async fn native_retry_handler_uses_exponential_backoff_and_injected_sleep() {
    let delays = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&delays);
    let scheduler = RetryScheduler {
        jitter: Box::new(|| 1.0),
        sleep: Box::new(move |delay| {
            recorded.lock().expect("delays").push(delay);
            Box::pin(async {})
        }),
    };
    let (session, turn_context) = make_session_and_context().await;
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
        delays[..6],
        [500, 1_000, 2_000, 4_000, 8_000, 16_000].map(Duration::from_millis)
    );
    assert_eq!(delays[6..], [Duration::from_secs(32); 14]);
}
