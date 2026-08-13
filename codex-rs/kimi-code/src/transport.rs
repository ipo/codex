use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use codex_api::ApiError;
use codex_api::AuthProvider;
use codex_api::EndpointSession;
use codex_api::HttpTransport;
use codex_api::Provider;
use codex_api::RequestTelemetry;
use codex_api::ResponseEvent;
use codex_chat_completions::ChatCompletionsRequest;
use codex_chat_completions::DecodeError;
use codex_chat_completions::DecodeStream;
use codex_chat_completions::DecodedStream;
use codex_chat_completions::IncrementalDecoder;
use codex_chat_completions::PresentationDelta;
use codex_chat_completions::ResponseMetadata;
use codex_client::EncodedJsonBody;
use codex_protocol::ResponseItemId;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;
use futures::Stream;
use futures::StreamExt;
use http::HeaderMap;
use http::HeaderValue;
use http::Method;
use thiserror::Error;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

use crate::KimiDialect;
use crate::KimiResponseError;
use crate::response_items;

#[derive(Debug, Error)]
pub enum KimiStreamError {
    #[error("native Kimi request failed: {0}")]
    Request(#[from] ApiError),
    #[error("native Kimi stream was cancelled")]
    Cancelled,
    #[error("native Kimi stream was idle past its deadline")]
    IdleTimeout,
    #[error("native Kimi transport failed: {0}")]
    Transport(String),
    #[error(transparent)]
    Decode(#[from] DecodeError),
    #[error("retryable native Kimi stream failure: {0}")]
    RetryableStream(String),
    #[error(transparent)]
    Response(#[from] KimiResponseError),
    #[error("invalid native Kimi request: {0}")]
    InvalidRequest(String),
    #[error("native Kimi usage exceeded canonical integer bounds")]
    UsageOverflow,
    #[error("native Kimi stop contained thinking but no user-visible output")]
    ThinkingOnlyStop,
}

pub struct KimiResponseStream {
    rx: mpsc::UnboundedReceiver<Result<ResponseEvent, KimiStreamError>>,
    pub metadata: ResponseMetadata,
}

impl Stream for KimiResponseStream {
    type Item = Result<ResponseEvent, KimiStreamError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

pub struct KimiHttpAdapter<T: HttpTransport> {
    session: EndpointSession<T>,
}

impl<T: HttpTransport> KimiHttpAdapter<T> {
    pub fn new(transport: T, provider: Provider, auth: Arc<dyn AuthProvider>) -> Self {
        Self {
            session: EndpointSession::new(transport, provider, auth),
        }
    }

    pub fn with_telemetry(self, telemetry: Option<Arc<dyn RequestTelemetry>>) -> Self {
        Self {
            session: self.session.with_request_telemetry(telemetry),
        }
    }

    pub async fn stream_request(
        &self,
        request: ChatCompletionsRequest,
        dialect: Arc<KimiDialect>,
        cancellation: CancellationToken,
    ) -> Result<KimiResponseStream, KimiStreamError> {
        let body = EncodedJsonBody::encode(&request)
            .map_err(|error| KimiStreamError::InvalidRequest(error.to_string()))?;
        let response = tokio::select! {
            _ = cancellation.cancelled() => return Err(KimiStreamError::Cancelled),
            response = self.session.stream_encoded_json_with(
                Method::POST,
                "/chat/completions",
                HeaderMap::new(),
                Some(body),
                |request| {
                    request.headers.insert(
                        http::header::ACCEPT,
                        HeaderValue::from_static("text/event-stream"),
                    );
                },
            ) => response?,
        };
        let metadata = ResponseMetadata {
            trace_id: response
                .headers
                .get("x-trace-id")
                .and_then(|value| value.to_str().ok())
                .map(str::to_string),
        };
        let idle_timeout = self.session.provider().stream_idle_timeout;
        let (tx, rx) = mpsc::unbounded_channel();
        let delta_tx = tx.clone();
        let decode_metadata = metadata.clone();
        tokio::spawn(async move {
            let mut presentation = PresentationState::default();
            let decode = DecodeStream {
                context: dialect.context(),
                dialect: dialect.as_ref(),
                metadata: decode_metadata,
            };
            let mut decoder = IncrementalDecoder::new(decode, |delta| {
                for event in presentation.events(delta) {
                    let _ = delta_tx.send(Ok(event));
                }
            });
            let mut bytes = response.bytes;
            let mut reached_done = false;
            loop {
                let next = tokio::select! {
                    _ = cancellation.cancelled() => {
                        let _ = tx.send(Err(KimiStreamError::Cancelled));
                        return;
                    }
                    next = timeout(idle_timeout, bytes.next()) => next,
                };
                match next {
                    Err(_) => {
                        let _ = tx.send(Err(KimiStreamError::IdleTimeout));
                        return;
                    }
                    Ok(Some(Ok(chunk))) => {
                        if let Err(error) = decoder.feed(&chunk) {
                            let _ = tx.send(Err(classify_decode_error(
                                error, /*reached_done*/ false,
                            )));
                            return;
                        }
                        if decoder.is_complete() {
                            reached_done = true;
                            break;
                        }
                    }
                    Ok(Some(Err(error))) => {
                        let _ = tx.send(Err(KimiStreamError::Transport(error.to_string())));
                        return;
                    }
                    Ok(None) => break,
                }
            }
            let result = match decoder.finish() {
                Ok(decoded) if is_thinking_only_stop(&decoded) => {
                    Err(KimiStreamError::ThinkingOnlyStop)
                }
                Ok(decoded) => canonical_events(dialect.as_ref(), decoded, &mut presentation),
                Err(error) => Err(classify_decode_error(error, reached_done)),
            };
            match result {
                Ok(events) => {
                    for event in events {
                        if tx.send(Ok(event)).is_err() {
                            return;
                        }
                    }
                }
                Err(error) => {
                    let _ = tx.send(Err(error));
                }
            }
        });
        Ok(KimiResponseStream { rx, metadata })
    }
}

fn is_thinking_only_stop(decoded: &DecodedStream) -> bool {
    decoded.terminal_outcome == codex_api::TerminalOutcome::Completed
        && decoded.pending.as_ref().is_some_and(|pending| {
            pending.content.is_empty()
                && !pending.reasoning.is_empty()
                && pending.tool_calls.is_empty()
        })
}

fn classify_decode_error(error: DecodeError, reached_done: bool) -> KimiStreamError {
    match error {
        error @ (DecodeError::EmptyStream
        | DecodeError::InvalidUtf8
        | DecodeError::MalformedFraming(_)
        | DecodeError::MalformedChunk(_)
        | DecodeError::PrematureEof { .. }) => KimiStreamError::RetryableStream(error.to_string()),
        error @ (DecodeError::MissingFinishReason | DecodeError::NullFinishReason)
            if !reached_done =>
        {
            KimiStreamError::RetryableStream(error.to_string())
        }
        DecodeError::InvalidTransition(message)
            if message == "successful terminal had no content" =>
        {
            KimiStreamError::RetryableStream("native Kimi stop contained no output".to_string())
        }
        error => KimiStreamError::Decode(error),
    }
}

#[derive(Default)]
struct PresentationState {
    started: Vec<(String, ResponseItem)>,
    reasoning_item_id: Option<ResponseItemId>,
    message_item_id: Option<ResponseItemId>,
    tool_items: BTreeMap<String, String>,
}

impl PresentationState {
    fn events(&mut self, delta: PresentationDelta) -> Vec<ResponseEvent> {
        match delta {
            PresentationDelta::Content(delta) if delta.is_empty() => Vec::new(),
            PresentationDelta::Reasoning(delta) if delta.is_empty() => Vec::new(),
            PresentationDelta::Content(delta) => {
                let id = self
                    .message_item_id
                    .get_or_insert_with(|| ResponseItemId::new("msg"))
                    .clone();
                let item = ResponseItem::Message {
                    id: Some(id.clone()),
                    role: "assistant".to_string(),
                    content: Vec::new(),
                    phase: None,
                    internal_chat_message_metadata_passthrough: None,
                };
                let mut events = self.added(id.to_string(), item);
                self.append(id.as_ref(), delta.clone());
                events.push(ResponseEvent::OutputTextDelta {
                    item_id: Some(id.to_string()),
                    delta,
                });
                events
            }
            PresentationDelta::Reasoning(delta) => {
                let id = self
                    .reasoning_item_id
                    .get_or_insert_with(|| ResponseItemId::new("rs"))
                    .clone();
                let item = ResponseItem::Reasoning {
                    id: Some(id.clone()),
                    summary: Vec::new(),
                    content: Some(Vec::new()),
                    encrypted_content: None,
                    internal_chat_message_metadata_passthrough: None,
                };
                let mut events = self.added(id.to_string(), item);
                self.append(id.as_ref(), delta.clone());
                events.push(ResponseEvent::ReasoningContentDelta {
                    item_id: Some(id.to_string()),
                    delta,
                    content_index: 0,
                });
                events
            }
            PresentationDelta::Tool {
                index: _,
                id,
                name,
                delta,
            } => {
                let call_id = id.clone();
                let item_id = self
                    .tool_items
                    .get(&call_id)
                    .map(|id| ResponseItemId::from_server(id.clone()))
                    .unwrap_or_else(|| ResponseItemId::new("fc"));
                let item = ResponseItem::FunctionCall {
                    id: Some(item_id.clone()),
                    name,
                    namespace: None,
                    arguments: String::new(),
                    call_id: id,
                    internal_chat_message_metadata_passthrough: None,
                };
                let id = item_id.to_string();
                let mut events = self.added(id.clone(), item);
                if !events.is_empty() {
                    self.tool_items.insert(call_id.clone(), id.clone());
                }
                if !delta.is_empty() {
                    self.append(item_id.as_ref(), delta.clone());
                    events.push(ResponseEvent::ToolCallInputDelta {
                        item_id: id,
                        call_id: Some(call_id),
                        delta,
                    });
                }
                events
            }
        }
    }

    fn added(&mut self, id: String, item: ResponseItem) -> Vec<ResponseEvent> {
        if self.started.iter().any(|(started, _)| started == &id) {
            Vec::new()
        } else {
            self.started.push((id, item.clone()));
            vec![ResponseEvent::OutputItemAdded(item)]
        }
    }

    fn append(&mut self, id: &str, delta: String) {
        let Some(item) = self
            .started
            .iter_mut()
            .find_map(|(started, item)| (started == id).then_some(item))
        else {
            unreachable!("presentation item was not started");
        };
        match item {
            ResponseItem::Message { content, .. } => match content.as_mut_slice() {
                [ContentItem::OutputText { text }] => text.push_str(&delta),
                [] => content.push(ContentItem::OutputText { text: delta }),
                _ => unreachable!("Kimi message presentation had unexpected content"),
            },
            ResponseItem::Reasoning { content, .. } => match content.as_mut() {
                Some(content) => match content.as_mut_slice() {
                    [ReasoningItemContent::ReasoningText { text }] => text.push_str(&delta),
                    [] => content.push(ReasoningItemContent::ReasoningText { text: delta }),
                    _ => unreachable!("Kimi reasoning presentation had unexpected content"),
                },
                None => unreachable!("Kimi reasoning presentation had no content"),
            },
            ResponseItem::FunctionCall { arguments, .. } => arguments.push_str(&delta),
            _ => unreachable!("Kimi presentation emitted an unsupported item"),
        }
    }
}

fn canonical_events(
    dialect: &KimiDialect,
    decoded: DecodedStream,
    state: &mut PresentationState,
) -> Result<Vec<ResponseEvent>, KimiStreamError> {
    let token_usage = decoded
        .usage
        .as_ref()
        .map(|usage| map_usage(usage, decoded.usage_details))
        .transpose()?;
    let mut events = Vec::new();
    if let Some(pending) = decoded.pending {
        for mut item in response_items(dialect, pending)? {
            let id = match &item {
                ResponseItem::Reasoning { .. } => state
                    .reasoning_item_id
                    .get_or_insert_with(|| ResponseItemId::new("rs"))
                    .to_string(),
                ResponseItem::Message { .. } => state
                    .message_item_id
                    .get_or_insert_with(|| ResponseItemId::new("msg"))
                    .to_string(),
                ResponseItem::FunctionCall { call_id, .. } => {
                    state.tool_items.get(call_id).cloned().ok_or_else(|| {
                        KimiStreamError::InvalidRequest(
                            "completed tool had no presentation item".to_string(),
                        )
                    })?
                }
                _ => unreachable!("Kimi response conversion emitted an unsupported item"),
            };
            item.set_id(Some(ResponseItemId::from_server(id.clone())));
            if !state.started.iter().any(|(started, _)| started == &id) {
                events.push(ResponseEvent::OutputItemAdded(empty_item(&item)));
            }
            events.push(ResponseEvent::OutputItemDone(item));
        }
    } else {
        for (_, item) in &state.started {
            events.push(ResponseEvent::OutputItemDone(item.clone()));
        }
    }
    events.push(ResponseEvent::Completed {
        response_id: decoded.response_id,
        token_usage,
        terminal_outcome: decoded.terminal_outcome,
    });
    Ok(events)
}

fn empty_item(item: &ResponseItem) -> ResponseItem {
    let mut item = item.clone();
    match &mut item {
        ResponseItem::Message { content, .. } => content.clear(),
        ResponseItem::Reasoning {
            content,
            encrypted_content,
            ..
        } => {
            *content = Some(Vec::new());
            *encrypted_content = None;
        }
        ResponseItem::FunctionCall { arguments, .. } => arguments.clear(),
        _ => unreachable!("Kimi response conversion emitted an unsupported item"),
    }
    item
}

fn map_usage(
    usage: &codex_chat_completions::ChunkUsage,
    details: codex_chat_completions::UsageDetails,
) -> Result<TokenUsage, KimiStreamError> {
    Ok(TokenUsage {
        input_tokens: usage_integer(usage.prompt_tokens)?,
        cached_input_tokens: usage_integer(details.cached_prompt_tokens)?,
        cache_write_input_tokens: 0,
        output_tokens: usage_integer(usage.completion_tokens)?,
        reasoning_output_tokens: usage_integer(details.reasoning_tokens)?,
        total_tokens: usage_integer(usage.total_tokens)?,
    })
}

fn usage_integer(value: u64) -> Result<i64, KimiStreamError> {
    i64::try_from(value).map_err(|_| KimiStreamError::UsageOverflow)
}
