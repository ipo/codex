use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use bytes::Bytes;
use codex_api::AuthProvider;
use codex_api::HttpTransport;
use codex_api::Provider;
use codex_api::ResponseEvent;
use codex_api::RetryConfig;
use codex_api::TerminalOutcome;
use codex_client::Request;
use codex_client::Response;
use codex_client::StreamResponse;
use codex_client::TransportError;
use codex_protocol::model_inference::KimiThinkingPolicy;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;
use futures::SinkExt;
use futures::StreamExt;
use futures::channel::mpsc;
use http::HeaderMap;
use http::HeaderValue;
use http::StatusCode;
use serde_json::Value;
use serde_json::json;
use tokio_util::sync::CancellationToken;

use super::*;

type BodySender = mpsc::UnboundedSender<Result<Bytes, TransportError>>;
type BodyReceiver = mpsc::UnboundedReceiver<Result<Bytes, TransportError>>;

fn normalize_opaque_item_ids(mut debug: String) -> String {
    let mut replacements = Vec::new();
    for prefix in ["rs_", "msg_", "fc_"] {
        let mut search_from = 0;
        while let Some(offset) = debug[search_from..].find(prefix) {
            let start = search_from + offset;
            let end = debug[start..]
                .find(|character: char| {
                    !(character.is_ascii_alphanumeric() || character == '_' || character == '-')
                })
                .map_or(debug.len(), |end| start + end);
            let id = debug[start..end].to_string();
            if !replacements.iter().any(|(seen, _)| seen == &id) {
                let normalized = format!("{prefix}opaque_{}", replacements.len());
                replacements.push((id, normalized));
            }
            search_from = end;
        }
    }
    for (id, normalized) in replacements {
        debug = debug.replace(&id, &normalized);
    }
    debug
}

#[derive(Clone)]
struct CaptureTransport {
    request: Arc<Mutex<Option<Request>>>,
    receiver: Arc<Mutex<Option<BodyReceiver>>>,
}

impl CaptureTransport {
    fn new() -> (Self, BodySender) {
        let (sender, receiver) = mpsc::unbounded();
        (
            Self {
                request: Arc::new(Mutex::new(None)),
                receiver: Arc::new(Mutex::new(Some(receiver))),
            },
            sender,
        )
    }
}

impl HttpTransport for CaptureTransport {
    async fn execute(&self, _request: Request) -> Result<Response, TransportError> {
        Err(TransportError::Build("execute should not run".to_string()))
    }

    async fn stream(&self, request: Request) -> Result<StreamResponse, TransportError> {
        *self.request.lock().expect("request mutex") = Some(request);
        let bytes = self
            .receiver
            .lock()
            .expect("receiver mutex")
            .take()
            .expect("single stream request")
            .boxed();
        Ok(StreamResponse {
            status: StatusCode::OK,
            headers: HeaderMap::from_iter([(
                http::header::HeaderName::from_static("x-trace-id"),
                HeaderValue::from_static("trace-57"),
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
            HeaderValue::from_static("Bearer generic-auth"),
        );
    }
}

fn provider(idle: Duration) -> Provider {
    Provider {
        name: "claudeflare-kimi".to_string(),
        base_url: "http://127.0.0.1:8080/v1/kimi".to_string(),
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

fn dialect(model: &str) -> Arc<KimiDialect> {
    let (cap, policy, thinking) = if model.starts_with("kimi-for-coding") {
        (32_768, KimiThinkingPolicy::Required, KimiThinking::Enabled)
    } else {
        (
            131_072,
            KimiThinkingPolicy::RequiredWithEffort,
            KimiThinking::Effort(KimiThinkingEffort::High),
        )
    };
    Arc::new(super::tests::dialect(
        super::tests::profile(model, cap, policy),
        262_144,
        10,
        thinking,
    ))
}

fn request(dialect: &KimiDialect) -> codex_chat_completions::ChatCompletionsRequest {
    serde_json::from_value(super::tests::request(dialect, &[], &[])).expect("prepared request")
}

async fn start_control(
    transport: CaptureTransport,
    cancellation: CancellationToken,
    idle_ms: u64,
) -> KimiResponseStream {
    let dialect = dialect("k3");
    let request = request(&dialect);
    KimiHttpAdapter::new(
        transport,
        provider(Duration::from_millis(idle_ms)),
        Arc::new(TestAuth),
    )
    .stream_request(request, dialect, cancellation)
    .await
    .expect("stream starts")
}

async fn send_body(sender: &mut BodySender, body: impl Into<Bytes>) {
    sender.send(Ok(body.into())).await.unwrap();
}

#[tokio::test]
async fn posts_exact_prepared_k3_and_coding_requests_with_generic_metadata() {
    let mut captures = Vec::new();
    for model in ["k3", "kimi-for-coding"] {
        let dialect = dialect(model);
        let expected = request(&dialect);
        let (transport, _sender) = CaptureTransport::new();
        let stream = KimiHttpAdapter::new(
            transport.clone(),
            provider(Duration::from_secs(1)),
            Arc::new(TestAuth),
        )
        .stream_request(expected.clone(), dialect, CancellationToken::new())
        .await
        .expect("stream starts");
        let captured = transport
            .request
            .lock()
            .expect("request mutex")
            .clone()
            .expect("captured request");
        let Some(codex_client::RequestBody::EncodedJson(body)) = captured.body else {
            panic!("encoded request body")
        };
        captures.push(json!({
            "model":model, "url":captured.url, "method":captured.method.as_str(),
            "headers":{
                "authorization":captured.headers[http::header::AUTHORIZATION].to_str().unwrap(),
                "accept":captured.headers[http::header::ACCEPT].to_str().unwrap(),
                "content-type":captured.headers[http::header::CONTENT_TYPE].to_str().unwrap(),
            },
            "trace_id":stream.metadata.trace_id,
            "body":serde_json::from_slice::<Value>(body.as_bytes()).unwrap(),
        }));
    }
    insta::assert_snapshot!(serde_json::to_string_pretty(&captures).unwrap());
}

#[tokio::test]
async fn interleaved_events_are_immediate_and_done_waits_for_finish_and_done() {
    let (transport, mut sender) = CaptureTransport::new();
    let mut stream = start_control(transport, CancellationToken::new(), 1_000).await;
    send_body(
        &mut sender,
        "data: {\"id\":\"chat-57\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"think \"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chat-57\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer \"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chat-57\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":4,\"id\":\"call-a\",\"type\":\"function\",\"function\":{\"name\":\"alpha\",\"arguments\":\"{\\\"a\\\":\"}}]},\"finish_reason\":null}]}\n\n",
    )
    .await;
    let mut early = Vec::new();
    for _ in 0..6 {
        early.push(stream.next().await.unwrap().unwrap());
    }
    send_body(
        &mut sender,
        "data: {\"id\":\"chat-57\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":4,\"function\":{\"arguments\":\"1}\"}}]},\"finish_reason\":\"tool_calls\",\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":9,\"total_tokens\":29,\"prompt_tokens_details\":{\"cached_tokens\":7},\"completion_tokens_details\":{\"reasoning_tokens\":4}}}]}\n\n",
    )
    .await;
    early.push(stream.next().await.unwrap().unwrap());
    assert!(
        tokio::time::timeout(Duration::from_millis(10), stream.next())
            .await
            .is_err()
    );
    send_body(&mut sender, ": keepalive\n\ndata: [DONE]\n\n").await;
    let terminal = stream.collect::<Vec<_>>().await;
    insta::assert_snapshot!(normalize_opaque_item_ids(format!(
        "{:#?}",
        (early, terminal)
    )));
}

async fn stream_result(body: &str) -> Vec<Result<ResponseEvent, KimiStreamError>> {
    let (transport, mut sender) = CaptureTransport::new();
    let stream = start_control(transport, CancellationToken::new(), 1_000).await;
    if !body.is_empty() {
        send_body(&mut sender, Bytes::copy_from_slice(body.as_bytes())).await;
    }
    drop(sender);
    stream.collect().await
}

#[tokio::test]
async fn authoritative_final_usage_completes_observed_tool_stream() -> Result<(), KimiStreamError> {
    let preliminary_usage = json!({
        "prompt_tokens":16646,"completion_tokens":654,"total_tokens":17300
    });
    let final_usage = json!({
        "prompt_tokens":16648,"completion_tokens":654,"total_tokens":17302,
        "prompt_tokens_details":{"cached_tokens":12000},
        "completion_tokens_details":{"reasoning_tokens":378}
    });
    let body = [
        format!("data: {}\n\n", json!({"id":"chat-observed","choices":[{
            "index":0,
            "delta":{"reasoning_content":"inspect ","tool_calls":[{
                "index":0,"id":"call-observed","type":"function",
                "function":{"name":"exec_command","arguments":"{\"cmd\":\"pwd\""}
            }]},
            "finish_reason":null
        }]})),
        format!("data: {}\n\n", json!({"id":"chat-observed","choices":[{
            "index":0,
            "delta":{"tool_calls":[{
                "index":0,"function":{"arguments":",\"yield_time_ms\":1000,\"max_output_tokens\":1000}"}
            }]},
            "finish_reason":"tool_calls",
            "usage":preliminary_usage
        }]})),
        format!("data: {}\n\n", json!({
            "id":"chat-observed","choices":[],"usage":final_usage
        })),
        "data: [DONE]\n\n".to_string(),
    ]
    .concat();

    let results = stream_result(&body).await;
    assert!(results.iter().all(Result::is_ok));
    let events = results.into_iter().collect::<Result<Vec<_>, _>>()?;
    let completed_tools = events
        .iter()
        .filter_map(|event| match event {
            ResponseEvent::OutputItemDone(item @ ResponseItem::FunctionCall { .. }) => {
                Some(item.clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let tool_item_id = completed_tools[0].id().expect("tool item id").clone();
    assert!(tool_item_id.as_str().starts_with("fc_"));
    assert_eq!(
        completed_tools,
        [ResponseItem::FunctionCall {
            id: Some(tool_item_id),
            name: "exec_command".to_string(),
            namespace: None,
            arguments: r#"{"cmd":"pwd","yield_time_ms":1000,"max_output_tokens":1000}"#.to_string(),
            call_id: "call-observed".to_string(),
            internal_chat_message_metadata_passthrough: None,
        }]
    );
    let completed = events.iter().find_map(|event| match event {
        ResponseEvent::Completed {
            token_usage,
            terminal_outcome,
            ..
        } => Some((token_usage, terminal_outcome)),
        _ => None,
    });
    assert_eq!(
        completed,
        Some((
            &Some(TokenUsage {
                input_tokens: 16648,
                cached_input_tokens: 12000,
                cache_write_input_tokens: 0,
                output_tokens: 654,
                reasoning_output_tokens: 378,
                total_tokens: 17302,
            }),
            &TerminalOutcome::ToolsReady,
        ))
    );
    Ok(())
}

#[tokio::test]
async fn presentation_item_ids_are_stable_within_requests_and_distinct_between_requests() {
    let body = "data: {\"id\":\"chat-ids\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"think \"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chat-ids\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"more\",\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
    let requests = futures::future::join_all([stream_result(body), stream_result(body)]).await;
    let request_ids = requests
        .iter()
        .map(|events| {
            let events = events
                .iter()
                .map(Result::as_ref)
                .collect::<Result<Vec<_>, _>>()
                .expect("successful stream");
            let reasoning = events
                .iter()
                .filter_map(|event| match event {
                    ResponseEvent::OutputItemAdded(ResponseItem::Reasoning { id, .. })
                    | ResponseEvent::OutputItemDone(ResponseItem::Reasoning { id, .. }) => {
                        id.as_ref().map(ToString::to_string)
                    }
                    ResponseEvent::ReasoningContentDelta { item_id, .. } => item_id.clone(),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let message = events
                .iter()
                .filter_map(|event| match event {
                    ResponseEvent::OutputItemAdded(ResponseItem::Message { id, .. })
                    | ResponseEvent::OutputItemDone(ResponseItem::Message { id, .. }) => {
                        id.as_ref().map(ToString::to_string)
                    }
                    ResponseEvent::OutputTextDelta { item_id, .. } => item_id.clone(),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert!(reasoning.iter().all(|id| id == &reasoning[0]));
            assert!(message.iter().all(|id| id == &message[0]));
            assert!(reasoning[0].starts_with("rs_"));
            assert!(message[0].starts_with("msg_"));
            (reasoning[0].clone(), message[0].clone())
        })
        .collect::<Vec<_>>();
    assert_ne!(request_ids[0], request_ids[1]);
}

#[tokio::test]
async fn noncommittable_terminals_close_partial_presentation_without_committing_it() {
    let outcomes = futures::future::join_all([
        "data: {\"id\":\"chat-57\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":\"length\"}]}\n\ndata: [DONE]\n\n",
        "data: {\"id\":\"chat-57\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"unsafe\",\"tool_calls\":[{\"index\":2,\"id\":\"call-x\",\"type\":\"function\",\"function\":{\"name\":\"blocked\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"content_filter\"}]}\n\ndata: [DONE]\n\n",
        "data: {\"id\":\"chat-57\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chat-57\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"thought\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
        "data: {\"id\":\"chat-57\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chat-57\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
    ].map(stream_result)).await;
    insta::assert_snapshot!(normalize_opaque_item_ids(format!("{outcomes:#?}")));
}

#[tokio::test]
async fn terminal_and_transport_failures_remain_typed() {
    let invalid_tool = "data: {\"id\":\"chat-57\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call\",\"type\":\"function\",\"function\":{\"name\":\"bad\",\"arguments\":\"{\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n";
    let missing_done = "data: {\"id\":\"chat-57\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":\"stop\"}]}\n\n";
    let mut failures = Vec::new();
    for body in [
        "",
        ": keepalive\n\n",
        "data: {bad}\n\n",
        invalid_tool,
        "data: {\"id\":\"chat-57\",\"choices\":[]}\n\n",
        missing_done,
    ] {
        failures.push(
            stream_result(body)
                .await
                .into_iter()
                .find_map(Result::err)
                .expect("stream ended without error"),
        );
    }
    insta::assert_debug_snapshot!(failures);
    let (transport, mut sender) = CaptureTransport::new();
    let mut stream = start_control(transport, CancellationToken::new(), 1_000).await;
    sender
        .send(Err(TransportError::Network("closed".to_string())))
        .await
        .unwrap();
    assert!(
        matches!(stream.next().await, Some(Err(KimiStreamError::Transport(message))) if message.contains("closed"))
    );
}

#[tokio::test]
async fn optional_finish_reason_completes_content_and_tools() {
    let bodies = [
        (
            "data: {\"id\":\"chat-missing\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"}}]}\n\ndata: [DONE]\n\n",
            TerminalOutcome::Completed,
        ),
        (
            "data: {\"id\":\"chat-null\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n",
            TerminalOutcome::Completed,
        ),
        (
            "data: {\"id\":\"chat-tool\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call\",\"type\":\"function\",\"function\":{\"name\":\"run\",\"arguments\":\"{}\"}}]}}]}\n\ndata: [DONE]\n\n",
            TerminalOutcome::ToolsReady,
        ),
    ];
    for (body, expected) in bodies {
        let results = stream_result(body).await;
        assert!(results.iter().all(Result::is_ok));
        assert!(results.iter().any(|result| matches!(
            result,
            Ok(ResponseEvent::Completed { terminal_outcome, .. }) if terminal_outcome == &expected
        )));
    }
}

#[tokio::test]
async fn optional_finish_reason_keeps_incomplete_and_unusable_streams_retryable() {
    let cases = [
        "data: {\"id\":\"chat-eof\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n",
        "data: {\"id\":\"chat-empty\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n",
    ];
    for body in cases {
        assert!(matches!(
            stream_result(body).await.last(),
            Some(Err(KimiStreamError::RetryableStream(_)))
        ));
    }
    let thinking_only = "data: {\"id\":\"chat-think\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"thought\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n";
    assert!(matches!(
        stream_result(thinking_only).await.last(),
        Some(Err(KimiStreamError::ThinkingOnlyStop))
    ));
}

#[tokio::test]
async fn cancellation_and_idle_are_distinct_typed_failures() {
    let (transport, _sender) = CaptureTransport::new();
    let cancellation = CancellationToken::new();
    let mut stream = start_control(transport, cancellation.clone(), 1_000).await;
    cancellation.cancel();
    assert!(matches!(
        stream.next().await,
        Some(Err(KimiStreamError::Cancelled))
    ));

    let (transport, _sender) = CaptureTransport::new();
    let mut stream = start_control(transport, CancellationToken::new(), 1).await;
    assert!(matches!(
        stream.next().await,
        Some(Err(KimiStreamError::IdleTimeout))
    ));
}
