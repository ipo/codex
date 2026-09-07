use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::DateTime;
use chrono::Utc;
use codex_client::EncodedJsonBody;
use codex_client::HttpTransport;
use codex_client::RequestTelemetry;
use codex_client::TransportError;
use codex_protocol::ResponseItemId;
use codex_protocol::ResponseUsageMetadata;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;
use futures::StreamExt;
use http::HeaderMap;
use http::HeaderValue;
use http::Method;
use serde_json::json;
use tokio::sync::mpsc;
use tokio::time::timeout;

use super::KimiChatRequest;
use super::KimiDecodedResponse;
use super::KimiPendingResponse;
use super::KimiStreamDecoder;
use super::KimiStreamError;
use super::KimiStreamEvent;
use super::KimiTerminal;
use super::KimiUsage;
use crate::ApiError;
use crate::AuthProvider;
use crate::Provider;
use crate::ResponseEvent;
use crate::ResponseStream;
use crate::endpoint::session::EndpointSession;

const RESPONSE_STREAM_CHANNEL_CAPACITY: usize = 1600;

pub struct KimiChatClient<T: HttpTransport> {
    session: EndpointSession<T>,
    request_telemetry: Option<Arc<dyn RequestTelemetry>>,
}

impl<T: HttpTransport> KimiChatClient<T> {
    pub fn new(transport: T, provider: Provider, auth: Arc<dyn AuthProvider>) -> Self {
        Self {
            session: EndpointSession::new(transport, provider, auth),
            request_telemetry: None,
        }
    }

    pub fn with_telemetry(mut self, telemetry: Option<Arc<dyn RequestTelemetry>>) -> Self {
        self.request_telemetry = telemetry;
        self
    }

    #[tracing::instrument(
        name = "kimi_chat.stream_request",
        level = "info",
        skip_all,
        fields(transport = "kimi_chat_http", http.method = "POST", api.path = endpoint)
    )]
    pub async fn stream_request(
        self,
        request: KimiChatRequest,
        endpoint: &str,
        extra_headers: HeaderMap,
    ) -> Result<ResponseStream, ApiError> {
        let wire_model = request.model.clone();
        let body = EncodedJsonBody::encode(&request).map_err(|error| ApiError::InvalidRequest {
            message: error.to_string(),
        })?;
        let session = self.session.with_request_telemetry(self.request_telemetry);
        let response = session
            .stream_encoded_json_with(
                Method::POST,
                endpoint,
                extra_headers,
                Some(body),
                |request| {
                    request.headers.insert(
                        http::header::ACCEPT,
                        HeaderValue::from_static("text/event-stream"),
                    );
                },
            )
            .await
            .map_err(classify_kimi_api_error)?;
        let trace_id = response
            .headers
            .get("x-trace-id")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let upstream_request_id = trace_id.clone();
        let idle_timeout = session.provider().stream_idle_timeout;
        let (tx_event, rx_event) = mpsc::channel(RESPONSE_STREAM_CHANNEL_CAPACITY);
        tokio::spawn(async move {
            let mut decoder = KimiStreamDecoder::new(wire_model);
            if let Some(trace_id) = trace_id {
                decoder = decoder.with_trace_id(trace_id);
            }
            let mut presentation = PresentationState::default();
            let mut bytes = response.bytes;
            loop {
                let next = tokio::select! {
                    _ = tx_event.closed() => return,
                    next = timeout(idle_timeout, bytes.next()) => next,
                };
                match next {
                    Err(_) => {
                        send_error(
                            &tx_event,
                            ApiError::Stream(
                                "native Kimi stream was idle past its deadline".to_string(),
                            ),
                        )
                        .await;
                        return;
                    }
                    Ok(Some(Ok(chunk))) => match decoder.feed(&chunk) {
                        Ok(events) => {
                            for event in events {
                                for event in presentation.events(event) {
                                    if tx_event.send(Ok(event)).await.is_err() {
                                        return;
                                    }
                                }
                            }
                            if decoder.is_done() {
                                break;
                            }
                        }
                        Err(error) => {
                            send_error(&tx_event, map_stream_error(error)).await;
                            return;
                        }
                    },
                    Ok(Some(Err(error))) => {
                        send_error(
                            &tx_event,
                            classify_kimi_api_error(ApiError::Transport(error)),
                        )
                        .await;
                        return;
                    }
                    Ok(None) => break,
                }
            }
            match decoder.finish().and_then(validate_committable) {
                Ok(decoded) => match completed_events(decoded, &mut presentation) {
                    Ok(events) => {
                        for event in events {
                            if tx_event.send(Ok(event)).await.is_err() {
                                return;
                            }
                        }
                    }
                    Err(error) => send_error(&tx_event, error).await,
                },
                Err(error) => send_error(&tx_event, map_stream_error(error)).await,
            }
        });
        Ok(ResponseStream {
            rx_event,
            upstream_request_id,
        })
    }
}

async fn send_error(tx_event: &mpsc::Sender<Result<ResponseEvent, ApiError>>, error: ApiError) {
    let _ = tx_event.send(Err(error)).await;
}

fn validate_committable(
    decoded: KimiDecodedResponse,
) -> Result<KimiDecodedResponse, KimiStreamError> {
    match decoded.terminal {
        KimiTerminal::Completed | KimiTerminal::ToolsReady => Ok(decoded),
        KimiTerminal::OutputExhausted => Err(KimiStreamError::InvalidTransition(
            "native Kimi output exhausted before completion",
        )),
        KimiTerminal::Refusal => Err(KimiStreamError::InvalidTransition(
            "native Kimi response was filtered",
        )),
    }
}

fn map_stream_error(error: KimiStreamError) -> ApiError {
    match error {
        error @ (KimiStreamError::EmptyStream
        | KimiStreamError::InvalidUtf8
        | KimiStreamError::MalformedFraming(_)
        | KimiStreamError::MalformedChunk(_)
        | KimiStreamError::PrematureEof(_)) => ApiError::Stream(error.to_string()),
        KimiStreamError::InvalidTransition("successful terminal had no content") => {
            ApiError::Stream("native Kimi stop contained no usable output".to_string())
        }
        KimiStreamError::InvalidTransition(message)
            if message == "native Kimi output exhausted before completion" =>
        {
            ApiError::InvalidRequest {
                message: message.to_string(),
            }
        }
        error => ApiError::InvalidRequest {
            message: error.to_string(),
        },
    }
}

fn classify_kimi_api_error(error: ApiError) -> ApiError {
    classify_kimi_api_error_at(error, Utc::now())
}

fn classify_kimi_api_error_at(error: ApiError, now: DateTime<Utc>) -> ApiError {
    match error {
        ApiError::Transport(transport) => classify_kimi_transport_error(transport, now),
        ApiError::ServerOverloaded => retryable("server overloaded", None),
        ApiError::RateLimit(message) => retryable(message, None),
        error => error,
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
        error => ApiError::Transport(error),
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

#[derive(Default)]
struct PresentationState {
    reasoning_item_id: Option<ResponseItemId>,
    message_item_id: Option<ResponseItemId>,
    tool_item_ids: BTreeMap<usize, ResponseItemId>,
}

impl PresentationState {
    fn events(&mut self, event: KimiStreamEvent) -> Vec<ResponseEvent> {
        match event {
            KimiStreamEvent::Content { delta, .. } => {
                let mut events = Vec::new();
                if self.message_item_id.is_none() {
                    let id = ResponseItemId::new("msg");
                    events.push(ResponseEvent::OutputItemAdded(empty_message(id.clone())));
                    self.message_item_id = Some(id);
                }
                events.push(ResponseEvent::OutputTextDelta(delta));
                events
            }
            KimiStreamEvent::Reasoning { delta, .. } => {
                let mut events = Vec::new();
                if self.reasoning_item_id.is_none() {
                    let id = ResponseItemId::new("rs");
                    events.push(ResponseEvent::OutputItemAdded(empty_reasoning(id.clone())));
                    self.reasoning_item_id = Some(id);
                }
                events.push(ResponseEvent::ReasoningContentDelta {
                    delta,
                    content_index: 0,
                });
                events
            }
            KimiStreamEvent::ToolCall {
                index,
                id,
                name,
                arguments_delta,
                ..
            } => {
                let mut events = Vec::new();
                let item_id = match self.tool_item_ids.get(&index) {
                    Some(item_id) => item_id.clone(),
                    None => {
                        let item_id = ResponseItemId::new("fc");
                        events.push(ResponseEvent::OutputItemAdded(ResponseItem::FunctionCall {
                            id: Some(item_id.clone()),
                            name,
                            namespace: None,
                            arguments: String::new(),
                            encrypted_function_args: None,
                            call_id: id.clone(),
                            internal_chat_message_metadata_passthrough: None,
                        }));
                        self.tool_item_ids.insert(index, item_id.clone());
                        item_id
                    }
                };
                if !arguments_delta.is_empty() {
                    events.push(ResponseEvent::ToolCallInputDelta {
                        item_id: item_id.to_string(),
                        call_id: Some(id),
                        delta: arguments_delta,
                    });
                }
                events
            }
        }
    }
}

fn completed_events(
    decoded: KimiDecodedResponse,
    presentation: &mut PresentationState,
) -> Result<Vec<ResponseEvent>, ApiError> {
    let token_usage = decoded.usage.map(map_usage).transpose()?;
    let usage_metadata = decoded.trace_id.map(|trace_id| ResponseUsageMetadata {
        amount: None,
        metadata: Some(json!({"x-trace-id": trace_id})),
    });
    let end_turn = Some(decoded.terminal == KimiTerminal::Completed);
    let pending = decoded.pending.ok_or_else(|| ApiError::InvalidRequest {
        message: "native Kimi response had no committable output".to_string(),
    })?;
    let mut events = canonical_items(pending, presentation);
    events.push(ResponseEvent::Completed {
        response_id: decoded.response_id,
        token_usage,
        usage_metadata,
        end_turn,
    });
    Ok(events)
}

fn canonical_items(
    pending: KimiPendingResponse,
    presentation: &mut PresentationState,
) -> Vec<ResponseEvent> {
    let mut events = Vec::new();
    if !pending.reasoning.text.is_empty() {
        let was_started = presentation.reasoning_item_id.is_some();
        let id = presentation
            .reasoning_item_id
            .get_or_insert_with(|| ResponseItemId::new("rs"))
            .clone();
        let encrypted_content = Some(pending.reasoning.opaque_marker().to_string());
        let item = ResponseItem::Reasoning {
            id: Some(id.clone()),
            summary: Vec::new(),
            content: Some(vec![ReasoningItemContent::ReasoningText {
                text: pending.reasoning.text,
            }]),
            encrypted_content,
            internal_chat_message_metadata_passthrough: None,
        };
        if !was_started {
            events.push(ResponseEvent::OutputItemAdded(empty_reasoning(id.clone())));
        }
        events.push(ResponseEvent::OutputItemDone(item));
    }
    if !pending.content.is_empty() {
        let was_started = presentation.message_item_id.is_some();
        let id = presentation
            .message_item_id
            .get_or_insert_with(|| ResponseItemId::new("msg"))
            .clone();
        if !was_started {
            events.push(ResponseEvent::OutputItemAdded(empty_message(id.clone())));
        }
        events.push(ResponseEvent::OutputItemDone(ResponseItem::Message {
            id: Some(id),
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: pending.content,
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        }));
    }
    for (index, call) in pending.tool_calls.into_iter().enumerate() {
        let was_started = presentation.tool_item_ids.contains_key(&index);
        let item_id = presentation
            .tool_item_ids
            .entry(index)
            .or_insert_with(|| ResponseItemId::new("fc"))
            .clone();
        if !was_started {
            events.push(ResponseEvent::OutputItemAdded(ResponseItem::FunctionCall {
                id: Some(item_id.clone()),
                name: call.function.name.clone(),
                namespace: None,
                arguments: String::new(),
                encrypted_function_args: None,
                call_id: call.id.clone(),
                internal_chat_message_metadata_passthrough: None,
            }));
        }
        events.push(ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
            id: Some(item_id),
            name: call.function.name,
            namespace: None,
            arguments: call.function.arguments,
            encrypted_function_args: None,
            call_id: call.id,
            internal_chat_message_metadata_passthrough: None,
        }));
    }
    events
}

fn empty_message(id: ResponseItemId) -> ResponseItem {
    ResponseItem::Message {
        id: Some(id),
        role: "assistant".to_string(),
        content: Vec::new(),
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }
}

fn empty_reasoning(id: ResponseItemId) -> ResponseItem {
    ResponseItem::Reasoning {
        id: Some(id),
        summary: Vec::new(),
        content: Some(Vec::new()),
        encrypted_content: None,
        internal_chat_message_metadata_passthrough: None,
    }
}

fn map_usage(usage: KimiUsage) -> Result<TokenUsage, ApiError> {
    Ok(TokenUsage {
        input_tokens: usage_integer(usage.prompt_tokens)?,
        cached_input_tokens: usage_integer(usage.cached_prompt_tokens)?,
        cache_write_input_tokens: 0,
        output_tokens: usage_integer(usage.completion_tokens)?,
        reasoning_output_tokens: usage_integer(usage.reasoning_tokens)?,
        total_tokens: usage_integer(usage.total_tokens)?,
        codex_rollout_budget_units: None,
    })
}

fn usage_integer(value: u64) -> Result<i64, ApiError> {
    i64::try_from(value).map_err(|_| ApiError::InvalidRequest {
        message: "native Kimi usage exceeded canonical integer bounds".to_string(),
    })
}

#[cfg(test)]
#[path = "client_tests.rs"]
mod tests;
