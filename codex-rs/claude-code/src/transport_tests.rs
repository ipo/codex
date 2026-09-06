use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use bytes::Bytes;
use codex_api::AuthProvider;
use codex_api::HttpTransport;
use codex_api::Provider;
use codex_api::ResponseEvent;
use codex_api::RetryConfig;
use codex_client::Request;
use codex_client::RequestBody;
use codex_client::Response;
use codex_client::StreamResponse;
use codex_client::TransportError;
use codex_protocol::models::ResponseItem;
use futures::StreamExt;
use http::HeaderMap;
use http::HeaderValue;
use http::StatusCode;
use pretty_assertions::assert_eq;
use serde_json::json;
use tokio_util::sync::CancellationToken;

use super::*;

const SESSION: &str = "019fbf00-0000-7000-8000-000000000047";
type StreamChunks = Vec<Result<Bytes, String>>;

#[derive(Clone)]
struct CaptureTransport {
    request: Arc<Mutex<Option<Request>>>,
    chunks: Arc<Mutex<Option<StreamChunks>>>,
    hang_after_chunks: bool,
}

impl CaptureTransport {
    fn chunks(chunks: StreamChunks) -> Self {
        Self {
            request: Arc::new(Mutex::new(None)),
            chunks: Arc::new(Mutex::new(Some(chunks))),
            hang_after_chunks: false,
        }
    }

    fn pending() -> Self {
        Self {
            request: Arc::new(Mutex::new(None)),
            chunks: Arc::new(Mutex::new(None)),
            hang_after_chunks: true,
        }
    }

    fn request(&self) -> Request {
        self.request
            .lock()
            .expect("request mutex")
            .clone()
            .expect("captured request")
    }
}

impl HttpTransport for CaptureTransport {
    async fn execute(&self, _request: Request) -> Result<Response, TransportError> {
        Err(TransportError::Build("execute should not run".to_string()))
    }

    async fn stream(&self, request: Request) -> Result<StreamResponse, TransportError> {
        *self.request.lock().expect("request mutex") = Some(request);
        let chunks = self.chunks.lock().expect("chunks mutex").take();
        let bytes = match chunks {
            Some(chunks) => {
                let chunks = futures::stream::iter(
                    chunks
                        .into_iter()
                        .map(|chunk| chunk.map_err(TransportError::Network)),
                );
                if self.hang_after_chunks {
                    chunks.chain(futures::stream::pending()).boxed()
                } else {
                    chunks.boxed()
                }
            }
            None => futures::stream::pending().boxed(),
        };
        Ok(StreamResponse {
            status: StatusCode::OK,
            headers: HeaderMap::from_iter([(
                http::header::HeaderName::from_static("x-request-id"),
                HeaderValue::from_static("request-47"),
            )]),
            bytes,
        })
    }
}

#[derive(Default)]
struct TestAuth;

impl AuthProvider for TestAuth {
    fn add_auth_headers(&self, headers: &mut HeaderMap) {
        headers.insert(
            http::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer generic-auth"),
        );
    }
}

fn provider(idle: Duration) -> Provider {
    Provider {
        name: "claudeflare-claude".to_string(),
        base_url: "http://127.0.0.1:8080/v1/claude-code".to_string(),
        query_params: None,
        headers: HeaderMap::new(),
        retry: RetryConfig {
            max_attempts: 1,
            base_delay: Duration::from_millis(1),
            retry_429: false,
            retry_5xx: false,
            retry_transport: false,
        },
        stream_idle_timeout: idle,
    }
}

fn profile() -> ModelInferenceConfig {
    ModelInferenceConfig::Anthropic {
        wire_api: WireApi::AnthropicMessages,
        dialect: InferenceDialect::ClaudeCode,
        route: "claude_code".to_string(),
        wire_model: "claude-haiku-4-5-20251001".to_string(),
        max_output_tokens: 32_000,
        thinking: AnthropicThinkingPolicy::Budgeted {
            budget_tokens: 31_999,
        },
        supports_disabled_thinking: true,
    }
}

fn request() -> AssembledRequest {
    assemble_request(AssembleRequest {
        profile: &profile(),
        effort: &ReasoningEffort::High,
        messages: &[],
        system: &[],
        tools: &[],
        resumable_session_id: SESSION,
        codex_version: "0.153.4",
        opus_compatibility: None,
        sonnet_compatibility: None,
    })
    .expect("request assembles")
}

fn event(name: &str, data: serde_json::Value) -> String {
    format!("event: {name}\ndata: {data}\n\n")
}

fn successful_fixture(reason: &str) -> String {
    [
        event("message_start", json!({"type":"message_start","message":{"id":"msg-47","type":"message","role":"assistant","model":"claude-haiku-4-5-20251001","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":0}}})),
        event("content_block_start", json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}})),
        event("content_block_delta", json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}})),
        event("content_block_stop", json!({"type":"content_block_stop","index":0})),
        event("message_delta", json!({"type":"message_delta","delta":{"stop_reason":reason},"usage":{"output_tokens":2}})),
        event("message_stop", json!({"type":"message_stop"})),
    ]
    .concat()
}

#[tokio::test]
async fn exact_request_and_successful_stream_use_generic_transport_contracts() {
    let expected = request();
    let transport = CaptureTransport::chunks(vec![Ok(Bytes::from(successful_fixture("end_turn")))]);
    let stream = ClaudeHttpAdapter::new(
        transport.clone(),
        provider(Duration::from_secs(1)),
        Arc::new(TestAuth),
    )
    .stream_request(expected.clone(), CancellationToken::new())
    .await
    .expect("stream starts");
    assert_eq!(stream.upstream_request_id.as_deref(), Some("request-47"));

    let captured = transport.request();
    assert_eq!(
        captured.url,
        "http://127.0.0.1:8080/v1/claude-code/v1/messages?beta=true"
    );
    assert_eq!(captured.method, http::Method::POST);
    assert_eq!(
        captured.headers[http::header::AUTHORIZATION],
        "Bearer generic-auth"
    );
    assert_eq!(captured.headers[http::header::ACCEPT], "text/event-stream");
    let Some(RequestBody::EncodedJson(body)) = captured.body else {
        panic!("expected encoded request body");
    };
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(body.as_bytes()).unwrap(),
        serde_json::to_value(&expected.body).unwrap()
    );
}

#[tokio::test]
async fn terminal_streams_commit_only_success_and_continue_outcomes() {
    for (reason, end_turn) in [("end_turn", true), ("pause_turn", false)] {
        let transport = CaptureTransport::chunks(vec![Ok(Bytes::from(successful_fixture(reason)))]);
        let stream = ClaudeHttpAdapter::new(
            transport,
            provider(Duration::from_secs(1)),
            Arc::new(TestAuth),
        )
        .stream_request(request(), CancellationToken::new())
        .await
        .expect("stream starts");
        let events = stream.collect::<Vec<_>>().await;
        assert!(matches!(
            events.last(),
            Some(Ok(ResponseEvent::Completed {
                end_turn: Some(actual),
                ..
            })) if *actual == end_turn
        ));
        assert!(events.iter().any(|event| matches!(
            event,
            Ok(ResponseEvent::OutputItemDone(ResponseItem::Message { .. }))
        )));
    }

    for (reason, outcome) in [
        ("max_tokens", TerminalOutcome::OutputExhausted),
        ("refusal", TerminalOutcome::Refusal),
    ] {
        let transport = CaptureTransport::chunks(vec![Ok(Bytes::from(successful_fixture(reason)))]);
        let stream = ClaudeHttpAdapter::new(
            transport,
            provider(Duration::from_secs(1)),
            Arc::new(TestAuth),
        )
        .stream_request(request(), CancellationToken::new())
        .await
        .expect("stream starts");
        let events = stream.collect::<Vec<_>>().await;
        assert!(matches!(
            events.last(),
            Some(Err(NativeStreamError::UnsuccessfulTerminal(actual))) if *actual == outcome
        ));
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, Ok(ResponseEvent::OutputItemDone(_))))
        );
    }
}

#[tokio::test]
async fn cancellation_and_idle_are_distinct_typed_failures() {
    let cancellation = CancellationToken::new();
    let mut cancelled = ClaudeHttpAdapter::new(
        CaptureTransport::pending(),
        provider(Duration::from_secs(1)),
        Arc::new(TestAuth),
    )
    .stream_request(request(), cancellation.clone())
    .await
    .expect("stream starts");
    cancellation.cancel();
    assert!(matches!(
        cancelled.next().await,
        Some(Err(NativeStreamError::Cancelled))
    ));

    let mut idle = ClaudeHttpAdapter::new(
        CaptureTransport::pending(),
        provider(Duration::from_millis(1)),
        Arc::new(TestAuth),
    )
    .stream_request(request(), CancellationToken::new())
    .await
    .expect("stream starts");
    assert!(matches!(
        idle.next().await,
        Some(Err(NativeStreamError::IdleTimeout))
    ));
}
