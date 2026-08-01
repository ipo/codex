use crate::AssembledRequest;
use crate::DecodeError;
use crate::DecodedBlock;
use crate::DecodedStream;
use crate::IncrementalDecoder;
use crate::PresentationDelta;
use crate::ThinkingReplayBlock;
use crate::encode_thinking_replay;
use codex_api::ApiError;
use codex_api::AuthProvider;
use codex_api::EndpointSession;
use codex_api::HttpTransport;
use codex_api::Provider;
use codex_api::RequestTelemetry;
use codex_api::ResponseEvent;
use codex_client::EncodedJsonBody;
use codex_protocol::ResponseItemId;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ResponseItem;
use futures::Stream;
use futures::StreamExt;
use http::HeaderMap;
use http::HeaderName;
use http::HeaderValue;
use http::Method;
use std::collections::BTreeSet;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use thiserror::Error;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Error)]
pub enum NativeStreamError {
    #[error("native Claude request failed: {0}")]
    Request(#[from] ApiError),
    #[error("native Claude stream was cancelled")]
    Cancelled,
    #[error("native Claude stream was idle past its deadline")]
    IdleTimeout,
    #[error("native Claude transport failed: {0}")]
    Transport(String),
    #[error(transparent)]
    Decode(#[from] DecodeError),
    #[error("invalid native Claude request metadata: {0}")]
    InvalidRequest(String),
}

pub struct ClaudeResponseStream {
    rx: mpsc::UnboundedReceiver<Result<ResponseEvent, NativeStreamError>>,
    pub upstream_request_id: Option<String>,
}

impl Stream for ClaudeResponseStream {
    type Item = Result<ResponseEvent, NativeStreamError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

pub struct ClaudeHttpAdapter<T: HttpTransport> {
    session: EndpointSession<T>,
}

impl<T: HttpTransport> ClaudeHttpAdapter<T> {
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
        request: AssembledRequest,
        cancellation: CancellationToken,
    ) -> Result<ClaudeResponseStream, NativeStreamError> {
        let headers = request_headers(&request)?;
        let path = request_path(&request);
        let body = EncodedJsonBody::encode(&request.body)
            .map_err(|error| NativeStreamError::InvalidRequest(error.to_string()))?;
        let stream_response = tokio::select! {
            _ = cancellation.cancelled() => return Err(NativeStreamError::Cancelled),
            response = self.session.stream_encoded_json_with(
                Method::POST,
                &path,
                headers,
                Some(body),
                |request| {
                    request.headers.insert(
                        http::header::ACCEPT,
                        HeaderValue::from_static("text/event-stream"),
                    );
                },
            ) => response?,
        };
        let upstream_request_id = stream_response
            .headers
            .get("x-request-id")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let idle_timeout = self.session.provider().stream_idle_timeout;
        let (tx, rx) = mpsc::unbounded_channel();
        let delta_tx = tx.clone();
        let mut started = BTreeSet::new();
        let mut decoder = IncrementalDecoder::new(move |delta| {
            if started.insert(delta_index(&delta)) {
                let _ = delta_tx.send(Ok(added_event(&delta)));
            }
            let _ = delta_tx.send(Ok(presentation_event(delta)));
        });
        tokio::spawn(async move {
            let mut bytes = stream_response.bytes;
            loop {
                let next = tokio::select! {
                    _ = cancellation.cancelled() => {
                        let _ = tx.send(Err(NativeStreamError::Cancelled));
                        return;
                    }
                    next = timeout(idle_timeout, bytes.next()) => next,
                };
                match next {
                    Err(_) => {
                        let _ = tx.send(Err(NativeStreamError::IdleTimeout));
                        return;
                    }
                    Ok(Some(Ok(chunk))) => {
                        if let Err(error) = decoder.feed(&chunk) {
                            let _ = tx.send(Err(error.into()));
                            return;
                        }
                        if decoder.is_complete() {
                            break;
                        }
                    }
                    Ok(Some(Err(error))) => {
                        let _ = tx.send(Err(NativeStreamError::Transport(error.to_string())));
                        return;
                    }
                    Ok(None) => break,
                }
            }
            match decoder.finish().and_then(canonical_events) {
                Ok(events) => {
                    for event in events {
                        if tx.send(Ok(event)).is_err() {
                            return;
                        }
                    }
                }
                Err(error) => {
                    let _ = tx.send(Err(error.into()));
                }
            }
        });
        Ok(ClaudeResponseStream {
            rx,
            upstream_request_id,
        })
    }
}

fn request_path(request: &AssembledRequest) -> String {
    let mut path = request.transport.path.to_string();
    if !request.transport.query.is_empty() {
        path.push('?');
        path.push_str(
            &request
                .transport
                .query
                .iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>()
                .join("&"),
        );
    }
    path
}

fn request_headers(request: &AssembledRequest) -> Result<HeaderMap, NativeStreamError> {
    let mut headers = HeaderMap::new();
    for (name, value) in &request.transport.headers {
        let name = HeaderName::try_from(name)
            .map_err(|error| NativeStreamError::InvalidRequest(error.to_string()))?;
        let value = HeaderValue::try_from(value)
            .map_err(|error| NativeStreamError::InvalidRequest(error.to_string()))?;
        headers.insert(name, value);
    }
    Ok(headers)
}

fn presentation_event(delta: PresentationDelta) -> ResponseEvent {
    match delta {
        PresentationDelta::Text { index, delta } => ResponseEvent::OutputTextDelta {
            item_id: Some(item_id("msg", index).to_string()),
            delta,
        },
        PresentationDelta::Thinking { index, delta } => ResponseEvent::ReasoningContentDelta {
            item_id: Some(item_id("rs", index).to_string()),
            delta,
            content_index: index as i64,
        },
        PresentationDelta::ToolInput {
            index, id, delta, ..
        } => ResponseEvent::ToolCallInputDelta {
            item_id: format!("fc_claude_{index}"),
            call_id: Some(id),
            delta,
        },
    }
}

fn delta_index(delta: &PresentationDelta) -> usize {
    match delta {
        PresentationDelta::Text { index, .. }
        | PresentationDelta::Thinking { index, .. }
        | PresentationDelta::ToolInput { index, .. } => *index,
    }
}

fn added_event(delta: &PresentationDelta) -> ResponseEvent {
    let index = delta_index(delta);
    let item = match delta {
        PresentationDelta::Text { .. } => ResponseItem::Message {
            id: Some(item_id("msg", index)),
            role: "assistant".to_string(),
            content: Vec::new(),
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        PresentationDelta::Thinking { .. } => ResponseItem::Reasoning {
            id: Some(item_id("rs", index)),
            summary: Vec::new(),
            content: Some(Vec::new()),
            encrypted_content: None,
            internal_chat_message_metadata_passthrough: None,
        },
        PresentationDelta::ToolInput { id, name, .. } => ResponseItem::FunctionCall {
            id: Some(item_id("fc", index)),
            name: name.clone(),
            namespace: None,
            arguments: String::new(),
            call_id: id.clone(),
            internal_chat_message_metadata_passthrough: None,
        },
    };
    ResponseEvent::OutputItemAdded(item)
}

fn canonical_events(decoded: DecodedStream) -> Result<Vec<ResponseEvent>, DecodeError> {
    let mut events = decoded
        .blocks
        .into_iter()
        .enumerate()
        .map(|(index, block)| canonical_item(index, &decoded.model, block))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(ResponseEvent::OutputItemDone)
        .collect::<Vec<_>>();
    events.push(ResponseEvent::Completed {
        response_id: decoded.message_id,
        token_usage: Some(decoded.token_usage),
        terminal_outcome: decoded.terminal_outcome,
    });
    Ok(events)
}

fn canonical_item(
    index: usize,
    model: &str,
    block: DecodedBlock,
) -> Result<ResponseItem, DecodeError> {
    let item = match block {
        DecodedBlock::Text { text } => ResponseItem::Message {
            id: Some(item_id("msg", index)),
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText { text }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        DecodedBlock::Thinking {
            thinking,
            signature,
        } => ResponseItem::Reasoning {
            id: Some(item_id("rs", index)),
            summary: Vec::new(),
            content: Some(vec![ReasoningItemContent::ReasoningText {
                text: thinking.clone(),
            }]),
            encrypted_content: Some(replay(
                model,
                ThinkingReplayBlock::Signed {
                    thinking,
                    signature,
                },
            )?),
            internal_chat_message_metadata_passthrough: None,
        },
        DecodedBlock::RedactedThinking { data } => ResponseItem::Reasoning {
            id: Some(item_id("rs", index)),
            summary: Vec::new(),
            content: None,
            encrypted_content: Some(replay(model, ThinkingReplayBlock::Redacted { data })?),
            internal_chat_message_metadata_passthrough: None,
        },
        DecodedBlock::ToolUse { id, name, input } => ResponseItem::FunctionCall {
            id: Some(item_id("fc", index)),
            name,
            namespace: None,
            arguments: input.to_string(),
            call_id: id,
            internal_chat_message_metadata_passthrough: None,
        },
    };
    Ok(item)
}

fn item_id(prefix: &str, index: usize) -> ResponseItemId {
    ResponseItemId::with_suffix(prefix, format!("claude_{index}"))
}

fn replay(model: &str, block: ThinkingReplayBlock) -> Result<String, DecodeError> {
    encode_thinking_replay(InferenceDialect::ClaudeCode, model, vec![block])
        .map_err(|error| DecodeError::Replay(error.to_string()))
}
