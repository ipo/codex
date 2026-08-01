#[path = "stream_error.rs"]
mod error;
#[path = "stream_helpers.rs"]
mod helpers;

use crate::Usage;
use codex_api::TerminalOutcome;
use codex_protocol::protocol::TokenUsage;
use helpers::map_stop_reason;
use helpers::map_usage;
use helpers::parse_sse;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

pub use error::DecodeError;

#[derive(Debug, Clone, PartialEq)]
pub enum DecodedBlock {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[rustfmt::skip]
pub enum PresentationDelta {
    Text { index: usize, delta: String },
    Thinking { index: usize, delta: String },
    ToolInput { index: usize, id: String, name: String, delta: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedStream {
    pub message_id: String,
    pub model: String,
    pub blocks: Vec<DecodedBlock>,
    pub terminal_outcome: TerminalOutcome,
    pub usage: Usage,
    pub token_usage: TokenUsage,
}

#[derive(Debug)]
enum PendingBlock {
    Text(String),
    Thinking {
        thinking: String,
        signature: String,
    },
    RedactedThinking(String),
    ToolUse {
        id: String,
        name: String,
        input: String,
    },
}

#[derive(Debug)]
struct BlockState {
    block: PendingBlock,
    stopped: bool,
}

#[derive(Debug, Deserialize)]
struct StartMessage {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    role: String,
    model: String,
    content: Vec<Value>,
    stop_reason: Value,
    stop_sequence: Value,
    usage: Usage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StartBlock {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
    },
    RedactedThinking {
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Delta {
    #[serde(rename = "text_delta")]
    Text { text: String },
    #[serde(rename = "thinking_delta")]
    Thinking { thinking: String },
    #[serde(rename = "signature_delta")]
    Signature { signature: String },
    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: String },
}

#[derive(Debug, Deserialize)]
struct MessageDeltaFields {
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DeltaUsage {
    output_tokens: u64,
    #[serde(default)]
    thinking_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ProviderError {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireEvent {
    Ping,
    MessageStart {
        message: StartMessage,
    },
    ContentBlockStart {
        index: usize,
        content_block: StartBlock,
    },
    ContentBlockDelta {
        index: usize,
        delta: Delta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: MessageDeltaFields,
        usage: DeltaUsage,
    },
    MessageStop,
    Error {
        error: ProviderError,
    },
}

impl WireEvent {
    fn name(&self) -> &'static str {
        match self {
            Self::Ping => "ping",
            Self::MessageStart { .. } => "message_start",
            Self::ContentBlockStart { .. } => "content_block_start",
            Self::ContentBlockDelta { .. } => "content_block_delta",
            Self::ContentBlockStop { .. } => "content_block_stop",
            Self::MessageDelta { .. } => "message_delta",
            Self::MessageStop => "message_stop",
            Self::Error { .. } => "error",
        }
    }
}

pub(super) struct Decoder<F> {
    sink: F,
    message: Option<(String, String)>,
    blocks: BTreeMap<usize, BlockState>,
    usage: Option<Usage>,
    outcome: Option<TerminalOutcome>,
    stopped: bool,
}

pub fn decode_stream<F>(body: &[u8], mut sink: F) -> Result<DecodedStream, DecodeError>
where
    F: FnMut(PresentationDelta),
{
    let body = std::str::from_utf8(body).map_err(|_| DecodeError::InvalidUtf8)?;
    if body.trim().is_empty() {
        return Err(DecodeError::EmptyStream);
    }
    let mut decoder = Decoder::new(&mut sink);
    let events = parse_sse(body)?;
    if events.is_empty() {
        return Err(DecodeError::EmptyStream);
    }
    for (event, data) in events {
        decoder.event(&event, &data)?;
    }
    decoder.finish()
}

impl<F> Decoder<F>
where
    F: FnMut(PresentationDelta),
{
    pub(super) fn new(sink: F) -> Self {
        Decoder {
            sink,
            message: None,
            blocks: BTreeMap::new(),
            usage: None,
            outcome: None,
            stopped: false,
        }
    }

    pub(super) fn event(&mut self, event: &str, data: &str) -> Result<(), DecodeError> {
        if !matches!(
            event,
            "ping"
                | "message_start"
                | "content_block_start"
                | "content_block_delta"
                | "content_block_stop"
                | "message_delta"
                | "message_stop"
                | "error"
        ) {
            return Err(DecodeError::UnknownEvent(event.to_string()));
        }
        let wire: WireEvent =
            serde_json::from_str(data).map_err(|error| DecodeError::MalformedEvent {
                event: event.to_string(),
                message: error.to_string(),
            })?;
        if wire.name() != event {
            return Err(DecodeError::MalformedEvent {
                event: event.to_string(),
                message: format!("data type {} did not match event name", wire.name()),
            });
        }
        if self.stopped {
            return Err(self.transition(event, "event followed message_stop"));
        }
        match wire {
            WireEvent::Ping => Ok(()),
            WireEvent::MessageStart { message } => self.message_start(event, message),
            WireEvent::ContentBlockStart {
                index,
                content_block,
            } => self.block_start(event, index, content_block),
            WireEvent::ContentBlockDelta { index, delta } => self.block_delta(event, index, delta),
            WireEvent::ContentBlockStop { index } => self.block_stop(event, index),
            WireEvent::MessageDelta { delta, usage } => self.message_delta(event, delta, usage),
            WireEvent::MessageStop => self.message_stop(event),
            WireEvent::Error { error } => Err(DecodeError::ProviderError {
                error_type: error.error_type,
                message: error.message,
            }),
        }
    }

    pub(super) fn is_stopped(&self) -> bool {
        self.stopped
    }

    fn message_start(&mut self, event: &str, message: StartMessage) -> Result<(), DecodeError> {
        if self.message.is_some() {
            return Err(self.transition(event, "duplicate message_start"));
        }
        if message.kind != "message"
            || message.role != "assistant"
            || !message.content.is_empty()
            || !message.stop_reason.is_null()
            || !message.stop_sequence.is_null()
        {
            return Err(self.transition(event, "message envelope was not an empty assistant start"));
        }
        self.usage = Some(message.usage);
        self.message = Some((message.id, message.model));
        Ok(())
    }

    fn block_start(
        &mut self,
        event: &str,
        index: usize,
        content_block: StartBlock,
    ) -> Result<(), DecodeError> {
        self.require_message(event)?;
        if self.outcome.is_some() {
            return Err(self.transition(event, "content block started after message_delta"));
        }
        if index != self.blocks.len() {
            return Err(
                self.transition(event, "content block index was duplicate or non-contiguous")
            );
        }
        let block = match content_block {
            StartBlock::Text { text } if text.is_empty() => PendingBlock::Text(text),
            StartBlock::Thinking { thinking } if thinking.is_empty() => PendingBlock::Thinking {
                thinking,
                signature: String::new(),
            },
            StartBlock::RedactedThinking { data } => PendingBlock::RedactedThinking(data),
            StartBlock::ToolUse { id, name, input } if input == serde_json::json!({}) => {
                PendingBlock::ToolUse {
                    id,
                    name,
                    input: String::new(),
                }
            }
            _ => {
                return Err(
                    self.transition(event, "streamed block contained non-empty initial content")
                );
            }
        };
        self.blocks.insert(
            index,
            BlockState {
                block,
                stopped: false,
            },
        );
        Ok(())
    }

    fn block_delta(&mut self, event: &str, index: usize, delta: Delta) -> Result<(), DecodeError> {
        let Some(state) = self.blocks.get_mut(&index) else {
            return Err(self.transition(event, "delta referenced an unknown block"));
        };
        if state.stopped {
            return Err(self.transition(event, "delta followed content_block_stop"));
        }
        let presentation = match (&mut state.block, delta) {
            (PendingBlock::Text(text), Delta::Text { text: delta }) => {
                text.push_str(&delta);
                PresentationDelta::Text { index, delta }
            }
            (PendingBlock::Thinking { thinking, .. }, Delta::Thinking { thinking: delta }) => {
                thinking.push_str(&delta);
                PresentationDelta::Thinking { index, delta }
            }
            (PendingBlock::Thinking { signature, .. }, Delta::Signature { signature: delta }) => {
                signature.push_str(&delta);
                return Ok(());
            }
            #[rustfmt::skip]
            (PendingBlock::ToolUse { id, name, input, .. }, Delta::InputJson { partial_json }) => {
                input.push_str(&partial_json);
                PresentationDelta::ToolInput { index, id: id.clone(), name: name.clone(), delta: partial_json }
            }
            _ => return Err(self.transition(event, "delta type did not match content block type")),
        };
        (self.sink)(presentation);
        Ok(())
    }

    fn block_stop(&mut self, event: &str, index: usize) -> Result<(), DecodeError> {
        let Some(state) = self.blocks.get_mut(&index) else {
            return Err(self.transition(event, "stop referenced an unknown block"));
        };
        if state.stopped {
            return Err(self.transition(event, "duplicate content_block_stop"));
        }
        if matches!(&state.block, PendingBlock::Thinking { signature, .. } if signature.is_empty())
        {
            return Err(self.transition(event, "thinking block had no signature"));
        }
        state.stopped = true;
        Ok(())
    }

    fn message_delta(
        &mut self,
        event: &str,
        delta: MessageDeltaFields,
        delta_usage: DeltaUsage,
    ) -> Result<(), DecodeError> {
        self.require_message(event)?;
        if self.outcome.is_some() || self.blocks.values().any(|block| !block.stopped) {
            return Err(self.transition(
                event,
                "message_delta was duplicate or preceded block completion",
            ));
        }
        let reason = delta.stop_reason.ok_or(DecodeError::MissingStopReason)?;
        self.outcome = Some(map_stop_reason(&reason)?);
        let Some(usage) = self.usage.as_mut() else {
            return Err(self.transition(event, "message_delta had no start usage"));
        };
        usage.output_tokens = delta_usage.output_tokens;
        usage.thinking_tokens = delta_usage.thinking_tokens;
        Ok(())
    }

    fn message_stop(&mut self, event: &str) -> Result<(), DecodeError> {
        self.require_message(event)?;
        if self.outcome.is_none() {
            return Err(self.transition(event, "message_stop preceded a valid message_delta"));
        }
        self.stopped = true;
        Ok(())
    }

    pub(super) fn finish(self) -> Result<DecodedStream, DecodeError> {
        if self.message.is_none() {
            return Err(DecodeError::PrematureEof {
                expected: "message_start".to_string(),
            });
        }
        if self.outcome.is_none() {
            return Err(DecodeError::PrematureEof {
                expected: "recognized non-null stop reason".to_string(),
            });
        }
        if !self.stopped {
            return Err(DecodeError::PrematureEof {
                expected: "message_stop".to_string(),
            });
        }
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for (index, state) in self.blocks {
            let block = match state.block {
                PendingBlock::Text(text) => DecodedBlock::Text { text },
                PendingBlock::Thinking {
                    thinking,
                    signature,
                } => DecodedBlock::Thinking {
                    thinking,
                    signature,
                },
                PendingBlock::RedactedThinking(data) => DecodedBlock::RedactedThinking { data },
                PendingBlock::ToolUse { id, name, input } => DecodedBlock::ToolUse {
                    id,
                    name,
                    input: serde_json::from_str(&input)
                        .map_err(|_| DecodeError::InvalidToolJson { index, input })?,
                },
            };
            blocks.push(block);
        }
        let Some((message_id, model)) = self.message else {
            return Err(DecodeError::PrematureEof {
                expected: "message_start".to_string(),
            });
        };
        let Some(usage) = self.usage else {
            return Err(DecodeError::PrematureEof {
                expected: "message_start usage".to_string(),
            });
        };
        let Some(terminal_outcome) = self.outcome else {
            return Err(DecodeError::PrematureEof {
                expected: "recognized non-null stop reason".to_string(),
            });
        };
        let token_usage = map_usage(&usage);
        Ok(DecodedStream {
            message_id,
            model,
            blocks,
            terminal_outcome,
            usage,
            token_usage,
        })
    }

    fn require_message(&self, event: &str) -> Result<(), DecodeError> {
        if self.message.is_none() {
            Err(self.transition(event, "event preceded message_start"))
        } else {
            Ok(())
        }
    }

    fn transition(&self, event: &str, reason: &str) -> DecodeError {
        DecodeError::InvalidTransition {
            event: event.to_string(),
            reason: reason.to_string(),
        }
    }
}
