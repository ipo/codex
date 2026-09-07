use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use bytes::Bytes;
use chrono::TimeZone;
use codex_client::HttpTransport;
use codex_client::Request;
use codex_client::RequestBody;
use codex_client::Response;
use codex_client::StreamResponse;
use codex_client::TransportError;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::model_inference::KimiInferenceConfig;
use codex_protocol::model_inference::KimiThinkingPolicy;
use codex_protocol::model_inference::WireApi;
use futures::StreamExt;
use http::HeaderMap;
use http::HeaderValue;
use http::Method;
use http::StatusCode;
use pretty_assertions::assert_eq;
use serde_json::json;

use super::*;
use crate::RetryConfig;
use crate::kimi_chat::KimiContent;
use crate::kimi_chat::KimiInputEstimate;
use crate::kimi_chat::KimiMessage;
use crate::kimi_chat::KimiReasoning;
use crate::kimi_chat::KimiReasoningKey;
use crate::kimi_chat::KimiRequestSettings;
use crate::kimi_chat::KimiThinkingEffort;
use crate::kimi_chat::build_kimi_chat_request;

#[derive(Clone)]
struct CaptureTransport {
    request: Arc<Mutex<Option<Request>>>,
    chunks: Arc<Vec<Bytes>>,
}

impl CaptureTransport {
    fn new(body: &str) -> Self {
        Self {
            request: Arc::new(Mutex::new(None)),
            chunks: Arc::new(
                body.as_bytes()
                    .chunks(11)
                    .map(Bytes::copy_from_slice)
                    .collect(),
            ),
        }
    }
}

impl HttpTransport for CaptureTransport {
    async fn execute(&self, _request: Request) -> Result<Response, TransportError> {
        Err(TransportError::Build("execute should not run".to_string()))
    }

    async fn stream(&self, request: Request) -> Result<StreamResponse, TransportError> {
        *self.request.lock().expect("request mutex") = Some(request);
        let chunks = self.chunks.as_ref().clone();
        let bytes = futures::stream::iter(chunks.into_iter().map(Ok)).boxed();
        Ok(StreamResponse {
            status: StatusCode::OK,
            headers: HeaderMap::from_iter([(
                http::header::HeaderName::from_static("x-trace-id"),
                HeaderValue::from_static("trace-kimi-1"),
            )]),
            bytes,
        })
    }
}

struct TestAuth;

impl AuthProvider for TestAuth {
    fn add_auth_headers(&self, headers: &mut HeaderMap) {
        headers.insert(
            http::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer test-token"),
        );
    }
}

fn provider() -> Provider {
    Provider {
        name: "claudeflare-kimi".to_string(),
        base_url: "http://127.0.0.1:8080/v1/kimi".to_string(),
        query_params: None,
        headers: HeaderMap::new(),
        retry: RetryConfig {
            max_attempts: 0,
            base_delay: Duration::from_millis(1),
            retry_429: false,
            retry_5xx: false,
            retry_transport: false,
        },
        stream_idle_timeout: Duration::from_secs(1),
    }
}

fn request() -> KimiChatRequest {
    build_kimi_chat_request(
        &KimiInferenceConfig {
            wire_api: WireApi::ChatCompletions,
            dialect: InferenceDialect::Kimi,
            route: "kimi_code".to_string(),
            wire_model: "k3".to_string(),
            max_output_tokens: 131_072,
            thinking: KimiThinkingPolicy::RequiredWithEffort,
        },
        KimiRequestSettings {
            context_window: 1_048_576,
            input_estimate: KimiInputEstimate::Fixed(100),
            prompt_cache_key: "session-affinity".to_string(),
            thinking_effort: Some(KimiThinkingEffort::High),
            reasoning_key: None,
        },
        vec![KimiMessage::User {
            content: KimiContent::Text("inspect".to_string()),
        }],
        Vec::new(),
    )
    .expect("valid Kimi request")
}

fn http_error(status: StatusCode, headers: HeaderMap, body: &str) -> ApiError {
    ApiError::Transport(TransportError::Http {
        status,
        url: Some("https://example.test/chat/completions".to_string()),
        headers: Some(headers),
        body: Some(body.to_string()),
    })
}

#[test]
fn kimi_http_errors_preserve_retry_delays_and_recovery_classification() {
    let now = Utc.with_ymd_and_hms(2026, 8, 2, 0, 0, 0).unwrap();
    let mut millisecond_headers = HeaderMap::new();
    millisecond_headers.insert("retry-after-ms", HeaderValue::from_static("2750"));
    let mut date_headers = HeaderMap::new();
    date_headers.insert(
        http::header::RETRY_AFTER,
        HeaderValue::from_static("Sun, 2 Aug 2026 00:00:03 +0000"),
    );

    for (error, expected_delay) in [
        (
            http_error(
                StatusCode::TOO_MANY_REQUESTS,
                millisecond_headers,
                "rate limited",
            ),
            Some(Duration::from_millis(2_750)),
        ),
        (
            http_error(
                StatusCode::from_u16(529).expect("valid overload status"),
                date_headers,
                "overloaded",
            ),
            Some(Duration::from_secs(3)),
        ),
        (
            http_error(
                StatusCode::SERVICE_UNAVAILABLE,
                HeaderMap::new(),
                r#"{"error":{"code":"server_is_overloaded"}}"#,
            ),
            None,
        ),
    ] {
        let ApiError::Retryable { delay, .. } = classify_kimi_api_error_at(error, now) else {
            panic!("expected retryable Kimi error")
        };
        assert_eq!(delay, expected_delay);
    }

    assert!(matches!(
        classify_kimi_api_error_at(
            http_error(
                StatusCode::BAD_REQUEST,
                HeaderMap::new(),
                r#"{"error":{"message":"maximum context length exceeded"}}"#,
            ),
            now,
        ),
        ApiError::ContextWindowExceeded
    ));
    assert!(matches!(
        classify_kimi_api_error_at(
            http_error(
                StatusCode::TOO_MANY_REQUESTS,
                HeaderMap::new(),
                r#"{"error":{"type":"billing_error","message":"credit balance is too low"}}"#,
            ),
            now,
        ),
        ApiError::QuotaExceeded
    ));
}

#[tokio::test]
async fn posts_exact_request_and_maps_stream_metadata_and_items() {
    let body = concat!(
        "data: {\"id\":\"chat-kimi-1\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"think \"}}]}\n\n",
        "data: {\"id\":\"chat-kimi-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"}}]}\n\n",
        "data: {\"id\":\"chat-kimi-1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"type\":\"function\",\"function\":{\"name\":\"inspect\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":9,\"total_tokens\":29,\"prompt_tokens_details\":{\"cached_tokens\":7},\"completion_tokens_details\":{\"reasoning_tokens\":4}}}\n\n",
        "data: [DONE]\n\n"
    );
    let transport = CaptureTransport::new(body);
    let expected_request = request();
    let mut headers = HeaderMap::new();
    headers.insert("x-session-test", HeaderValue::from_static("present"));
    let mut stream =
        KimiChatClient::<CaptureTransport>::new(transport.clone(), provider(), Arc::new(TestAuth))
            .stream_request(expected_request.clone(), "/chat/completions", headers)
            .await
            .expect("stream starts");

    assert_eq!(stream.upstream_request_id.as_deref(), Some("trace-kimi-1"));
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.expect("valid canonical event"));
    }

    let captured = transport
        .request
        .lock()
        .expect("request mutex")
        .clone()
        .expect("captured request");
    assert_eq!(captured.method, Method::POST);
    assert_eq!(
        captured.url,
        "http://127.0.0.1:8080/v1/kimi/chat/completions"
    );
    assert_eq!(
        captured
            .headers
            .get(http::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok()),
        Some("Bearer test-token")
    );
    assert_eq!(
        captured
            .headers
            .get(http::header::ACCEPT)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
    assert_eq!(
        captured
            .headers
            .get("x-session-test")
            .and_then(|value| value.to_str().ok()),
        Some("present")
    );
    let Some(RequestBody::EncodedJson(encoded)) = captured.body else {
        panic!("encoded Kimi request body")
    };
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(encoded.as_bytes())
            .expect("captured request JSON"),
        serde_json::to_value(expected_request).expect("expected request JSON")
    );

    let reasoning_id = events
        .iter()
        .find_map(|event| match event {
            ResponseEvent::OutputItemAdded(ResponseItem::Reasoning { id, .. }) => id.clone(),
            _ => None,
        })
        .expect("reasoning item was presented");
    let message_id = events
        .iter()
        .find_map(|event| match event {
            ResponseEvent::OutputItemAdded(ResponseItem::Message { id, .. }) => id.clone(),
            _ => None,
        })
        .expect("message item was presented");
    let tool_id = events
        .iter()
        .find_map(|event| match event {
            ResponseEvent::OutputItemAdded(ResponseItem::FunctionCall { id, .. }) => id.clone(),
            _ => None,
        })
        .expect("tool item was presented");
    let marker = KimiReasoning::for_model(
        "k3",
        KimiReasoningKey::ReasoningContent,
        "think ".to_string(),
    )
    .expect("valid reasoning marker")
    .opaque_marker()
    .to_string();
    let completed_items = events
        .iter()
        .filter_map(|event| match event {
            ResponseEvent::OutputItemDone(item) => Some(item.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        completed_items,
        vec![
            ResponseItem::Reasoning {
                id: Some(reasoning_id),
                summary: Vec::new(),
                content: Some(vec![ReasoningItemContent::ReasoningText {
                    text: "think ".to_string(),
                }]),
                encrypted_content: Some(marker),
                internal_chat_message_metadata_passthrough: None,
            },
            ResponseItem::Message {
                id: Some(message_id),
                role: "assistant".to_string(),
                content: vec![ContentItem::OutputText {
                    text: "answer".to_string(),
                }],
                phase: None,
                internal_chat_message_metadata_passthrough: None,
            },
            ResponseItem::FunctionCall {
                id: Some(tool_id),
                name: "inspect".to_string(),
                namespace: None,
                arguments: "{}".to_string(),
                encrypted_function_args: None,
                call_id: "call-1".to_string(),
                internal_chat_message_metadata_passthrough: None,
            },
        ]
    );
    assert!(events.iter().any(|event| matches!(
        event,
        ResponseEvent::ReasoningContentDelta { delta, content_index: 0 }
            if delta == "think "
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ResponseEvent::OutputTextDelta(delta) if delta == "answer"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ResponseEvent::ToolCallInputDelta { call_id: Some(call_id), delta, .. }
            if call_id == "call-1" && delta == "{}"
    )));
    let completed = events.iter().find_map(|event| match event {
        ResponseEvent::Completed {
            response_id,
            token_usage,
            usage_metadata,
            end_turn,
        } => Some((response_id, token_usage, usage_metadata, end_turn)),
        _ => None,
    });
    let Some((response_id, token_usage, usage_metadata, end_turn)) = completed else {
        panic!("completed event")
    };
    assert_eq!(response_id.as_str(), "chat-kimi-1");
    assert_eq!(
        token_usage,
        &Some(TokenUsage {
            input_tokens: 20,
            cached_input_tokens: 7,
            cache_write_input_tokens: 0,
            output_tokens: 9,
            reasoning_output_tokens: 4,
            total_tokens: 29,
            codex_rollout_budget_units: None,
        })
    );
    assert_eq!(
        usage_metadata,
        &Some(ResponseUsageMetadata {
            amount: None,
            metadata: Some(json!({"x-trace-id":"trace-kimi-1"})),
        })
    );
    assert_eq!(end_turn, &Some(false));
}
