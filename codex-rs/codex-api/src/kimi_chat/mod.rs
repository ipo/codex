//! Native Kimi Chat Completions wire types and codecs.
//!
//! This module deliberately starts after canonical history and tool projection.
//! Routing and projection can therefore be connected without teaching this wire
//! codec about model slugs, tool runtimes, or provider-specific history.

mod client;
mod request;
mod schema;
mod stream;
mod stream_types;

pub use client::KimiChatClient;
pub use request::KimiAssistantMessage;
pub use request::KimiAssistantMessageInput;
pub use request::KimiChatRequest;
pub use request::KimiContent;
pub use request::KimiFunction;
pub use request::KimiFunctionDefinition;
pub use request::KimiFunctionTool;
pub use request::KimiInputEstimate;
pub use request::KimiMessage;
pub use request::KimiReasoning;
pub use request::KimiReasoningKey;
pub use request::KimiRequestError;
pub use request::KimiRequestSettings;
pub use request::KimiThinkingEffort;
pub use request::KimiToolCall;
pub use request::build_kimi_chat_request;
pub use request::estimate_kimi_input_tokens;
pub use schema::KimiSchemaError;
pub use schema::normalize_kimi_schema;
pub use stream::KimiStreamDecoder;
pub use stream_types::KimiDecodedResponse;
pub use stream_types::KimiPendingResponse;
pub use stream_types::KimiStreamError;
pub use stream_types::KimiStreamEvent;
pub use stream_types::KimiTerminal;
pub use stream_types::KimiUsage;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
