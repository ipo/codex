use super::*;
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
use codex_protocol::model_inference::AnthropicThinkingPolicy;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::model_inference::ModelInferenceConfig;
use codex_protocol::model_inference::WireApi;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ReasoningEffort;
use futures::StreamExt;
use http::HeaderMap;
use http::HeaderValue;
use http::StatusCode;
use pretty_assertions::assert_eq;
use serde_json::json;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

const SESSION: &str = "019fbf00-0000-7000-8000-000000000047";
type StreamChunks = Vec<Result<Bytes, String>>;

#[derive(Clone)]
struct CaptureTransport {
    request: Arc<Mutex<Option<Request>>>,
    chunks: Arc<Mutex<Option<StreamChunks>>>,
    hang_after_chunks: bool,
}

#[rustfmt::skip]
impl CaptureTransport {
    fn new(chunks: Option<StreamChunks>, hang_after_chunks: bool) -> Self {
        Self { request: Arc::new(Mutex::new(None)), chunks: Arc::new(Mutex::new(chunks)), hang_after_chunks }
    }
    fn chunks(chunks: StreamChunks) -> Self { Self::new(Some(chunks), false) }
    fn hanging(chunks: StreamChunks) -> Self { Self::new(Some(chunks), true) }
    fn pending() -> Self { Self::new(None, true) }
    fn request(&self) -> Request { self.request.lock().expect("request mutex").clone().expect("captured request") }
}

#[rustfmt::skip]
impl HttpTransport for CaptureTransport {
    async fn execute(&self, _request: Request) -> Result<Response, TransportError> { Err(TransportError::Build("execute should not run".to_string())) }

    async fn stream(&self, request: Request) -> Result<StreamResponse, TransportError> {
        *self.request.lock().expect("request mutex") = Some(request);
        let chunks = self.chunks.lock().expect("chunks mutex").take();
        let bytes = match chunks {
            Some(chunks) => {
                let chunks = futures::stream::iter(chunks.into_iter().map(|chunk| chunk.map_err(TransportError::Network)));
                if self.hang_after_chunks { chunks.chain(futures::stream::pending()).boxed() } else { chunks.boxed() }
            }
            None => futures::stream::pending().boxed(),
        };
        Ok(StreamResponse { status: StatusCode::OK, headers: HeaderMap::from_iter([(
            http::header::HeaderName::from_static("x-request-id"), HeaderValue::from_static("request-47"),
        )]), bytes })
    }
}

#[derive(Default)]
struct TestAuth;

#[rustfmt::skip]
impl AuthProvider for TestAuth {
    fn add_auth_headers(&self, headers: &mut HeaderMap) {
        headers.insert(http::header::AUTHORIZATION, HeaderValue::from_static("Bearer generic-auth"));
    }
}

#[rustfmt::skip]
fn provider(idle: Duration) -> Provider {
    Provider {
        name: "claudeflare-claude".to_string(),
        base_url: "http://127.0.0.1:8080/v1/claude-code".to_string(),
        query_params: None, headers: HeaderMap::new(),
        retry: RetryConfig { max_attempts: 1, base_delay: Duration::from_millis(1), retry_429: false, retry_5xx: false, retry_transport: false },
        stream_idle_timeout: idle,
    }
}

#[rustfmt::skip]
fn profile(model: &str, max_output_tokens: u32, thinking: AnthropicThinkingPolicy) -> ModelInferenceConfig {
    ModelInferenceConfig::Anthropic {
        wire_api: WireApi::AnthropicMessages, dialect: InferenceDialect::ClaudeCode,
        route: "claude_code".to_string(), wire_model: model.to_string(), max_output_tokens, thinking,
        supports_disabled_thinking: true,
    }
}

#[rustfmt::skip]
fn request(profile: &ModelInferenceConfig, effort: ReasoningEffort) -> AssembledRequest {
    assemble_request(AssembleRequest {
        profile, effort: &effort, messages: &[], system: &[], tools: &[],
        resumable_session_id: SESSION, codex_version: "0.146.0",
        opus_compatibility: None,
        sonnet_compatibility: None,
    }).expect("request assembles")
}

#[rustfmt::skip]
async fn start_stream(transport: CaptureTransport, idle: Duration, cancellation: CancellationToken) -> ClaudeResponseStream {
    let request = request(&profile("m", 64_000, AnthropicThinkingPolicy::Adaptive), ReasoningEffort::High);
    ClaudeHttpAdapter::new(transport, provider(idle), Arc::new(TestAuth))
        .stream_request(request, cancellation).await.expect("stream starts")
}

#[rustfmt::skip]
async fn chunk_stream(chunks: StreamChunks) -> ClaudeResponseStream {
    start_stream(CaptureTransport::chunks(chunks), Duration::from_secs(1), CancellationToken::new()).await
}

fn event(name: &str, data: serde_json::Value) -> String {
    format!("event: {name}\ndata: {data}\n\n")
}

fn successful_fixture(text: &str) -> String {
    [
        event("message_start", json!({"type":"message_start","message":{"id":"msg-47","type":"message","role":"assistant","model":"claude-haiku-4-5-20251001","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":0}}})),
        event("content_block_start", json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}})),
        event("content_block_delta", json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":text}})),
        event("content_block_stop", json!({"type":"content_block_stop","index":0})),
        event("message_delta", json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}})),
        event("message_stop", json!({"type":"message_stop"})),
    ]
    .concat()
}

#[rustfmt::skip]
fn all_delta_fixture() -> String {
    [
        event("message_start", json!({"type":"message_start","message":{"id":"msg-47","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":0}}})),
        event("content_block_start", json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}})), event("content_block_start", json!({"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}})),
        event("content_block_start", json!({"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"call-a","name":"alpha","input":{}}})), event("content_block_start", json!({"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"call-b","name":"beta","input":{}}})),
        event("content_block_delta", json!({"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}})),
        event("content_block_delta", json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think "}})),
        event("content_block_delta", json!({"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"b\":"}})),
        event("content_block_delta", json!({"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"hello "}})),
        event("content_block_delta", json!({"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"1}"}})),
        event("content_block_delta", json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"again"}})),
        event("content_block_delta", json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}})),
        event("content_block_delta", json!({"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"2}"}})),
        event("content_block_delta", json!({"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"world"}})),
        event("content_block_stop", json!({"type":"content_block_stop","index":0})), event("content_block_stop", json!({"type":"content_block_stop","index":1})),
        event("content_block_stop", json!({"type":"content_block_stop","index":2})), event("content_block_stop", json!({"type":"content_block_stop","index":3})),
        event("message_delta", json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":2}})),
        event("message_stop", json!({"type":"message_stop"})),
    ].concat() }

#[rustfmt::skip]
fn event_trace(event: ResponseEvent) -> String {
    match event {
        ResponseEvent::OutputItemAdded(ResponseItem::FunctionCall { id: Some(id), call_id, name, .. }) => format!("added:{id}:{call_id}:{name}"),
        ResponseEvent::OutputItemAdded(item) => format!("added:{}", item.id().expect("added item ID")),
        ResponseEvent::OutputTextDelta { item_id: Some(id), delta } => format!("text:{id}:{delta}"),
        ResponseEvent::ReasoningContentDelta { item_id: Some(id), delta, .. } => format!("reasoning:{id}:{delta}"),
        ResponseEvent::ToolCallInputDelta { item_id, call_id: Some(call_id), delta } => format!("tool:{item_id}:{call_id}:{delta}"),
        ResponseEvent::OutputItemDone(item) => format!("done:{}", item.id().expect("done item ID")),
        ResponseEvent::Completed { .. } => "completed".to_string(),
        event => panic!("unexpected event: {event:?}"),
    }
}

#[tokio::test]
#[rustfmt::skip]
async fn exact_haiku_and_adaptive_requests_use_generic_auth_and_telemetry() {
    let cases = [
        request(&profile("claude-haiku-4-5-20251001", 32_000, AnthropicThinkingPolicy::Budgeted { budget_tokens: 31_999 }), ReasoningEffort::High),
        request(&profile("claude-sonnet-5", 64_000, AnthropicThinkingPolicy::Adaptive), ReasoningEffort::Minimal),
    ];
    for expected in cases {
        let transport = CaptureTransport::chunks(vec![Ok(Bytes::from(successful_fixture("ok")))]);
        let adapter = ClaudeHttpAdapter::new(transport.clone(), provider(Duration::from_secs(1)), Arc::new(TestAuth));
        let stream = adapter.stream_request(expected.clone(), CancellationToken::new()).await.expect("stream starts");
        assert_eq!(stream.upstream_request_id.as_deref(), Some("request-47"));
        let captured = transport.request();
        assert_eq!(captured.url, "http://127.0.0.1:8080/v1/claude-code/v1/messages?beta=true");
        assert_eq!(captured.method, http::Method::POST);
        let mut expected_headers = HeaderMap::from_iter([
            (http::header::AUTHORIZATION, HeaderValue::from_static("Bearer generic-auth")),
            (http::header::ACCEPT, HeaderValue::from_static("text/event-stream")),
            (http::header::CONTENT_TYPE, HeaderValue::from_static("application/json")),
        ]);
        for (name, value) in &expected.transport.headers {
            expected_headers.insert(http::header::HeaderName::try_from(name).unwrap(), HeaderValue::from_str(value).unwrap());
        }
        assert_eq!(captured.headers, expected_headers);
        let Some(RequestBody::EncodedJson(body)) = captured.body else { panic!("encoded body") };
        assert_eq!(serde_json::from_slice::<serde_json::Value>(body.as_bytes()).unwrap(), serde_json::to_value(&expected.body).unwrap());
    }
}

#[test]
fn every_byte_boundary_matches_strict_decoder_with_utf8_and_multiline_data() {
    let fixture = successful_fixture("héllo 🦀").replace(
        "data: {\"type\":\"message_stop\"}",
        "data: {\"type\":\ndata: \"message_stop\"}",
    );
    let strict = decode_stream(fixture.as_bytes(), |_| {}).expect("strict decode");
    for split in 1..fixture.len() {
        let mut decoder = IncrementalDecoder::new(|_| {});
        decoder.feed(&fixture.as_bytes()[..split]).unwrap();
        decoder.feed(&fixture.as_bytes()[split..]).unwrap();
        assert_eq!(decoder.finish(), Ok(strict.clone()), "split {split}");
    }
}

#[tokio::test]
#[rustfmt::skip]
async fn interleaved_deltas_retain_item_identity_until_terminal_done() {
    let fixture = all_delta_fixture();
    let split = fixture.find("content_block_stop").expect("stop marker");
    let transport = CaptureTransport::hanging(vec![Ok(Bytes::copy_from_slice(&fixture.as_bytes()[..split])), Ok(Bytes::copy_from_slice(&fixture.as_bytes()[split..]))]);
    let stream = start_stream(transport, Duration::from_secs(1), CancellationToken::new()).await;
    let actual = stream.map(|event| event_trace(event.expect("valid event"))).collect::<Vec<_>>().await;
    assert_eq!(actual, vec![
        "added:fc_claude_2:call-a:alpha", "tool:fc_claude_2:call-a:{\"a\":", "added:rs_claude_0", "reasoning:rs_claude_0:think ",
        "added:fc_claude_3:call-b:beta", "tool:fc_claude_3:call-b:{\"b\":", "added:msg_claude_1", "text:msg_claude_1:hello ",
        "tool:fc_claude_2:call-a:1}", "reasoning:rs_claude_0:again", "tool:fc_claude_3:call-b:2}", "text:msg_claude_1:world",
        "done:rs_claude_0", "done:msg_claude_1", "done:fc_claude_2", "done:fc_claude_3", "completed",
    ]);
}

#[tokio::test]
#[rustfmt::skip]
async fn real_adapter_paths_return_typed_terminal_failures() {
    let native_error = event("error", json!({"type":"error","error":{"type":"overloaded_error","message":"busy"}}));
    let premature = event("message_start", json!({"type":"message_start","message":{"id":"msg","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}));
    let cases = [
        (vec![], DecodeError::EmptyStream),
        (vec![Ok(Bytes::from(native_error))], DecodeError::ProviderError { error_type: "overloaded_error".to_string(), message: "busy".to_string() }),
        (vec![Ok(Bytes::from(premature))], DecodeError::PrematureEof { expected: "recognized non-null stop reason".to_string() }),
    ];
    for (chunks, expected) in cases {
        let mut stream = chunk_stream(chunks).await;
        assert!(matches!(stream.next().await, Some(Err(NativeStreamError::Decode(actual))) if actual == expected));
    }
    let mut stream = chunk_stream(vec![Err("closed".to_string())]).await;
    assert!(matches!(stream.next().await, Some(Err(NativeStreamError::Transport(message))) if message.contains("closed")));
}

#[tokio::test]
#[rustfmt::skip]
async fn cancellation_and_idle_are_distinct_typed_failures() {
    let cancel = CancellationToken::new();
    let mut stream = start_stream(CaptureTransport::pending(), Duration::from_secs(1), cancel.clone()).await; cancel.cancel();
    assert!(matches!(stream.next().await, Some(Err(NativeStreamError::Cancelled))));
    let mut stream = start_stream(CaptureTransport::pending(), Duration::from_millis(1), CancellationToken::new()).await; assert!(matches!(stream.next().await, Some(Err(NativeStreamError::IdleTimeout))));
}
