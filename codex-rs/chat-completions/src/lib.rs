//! Provider-neutral Chat Completions request grammar and canonical-history encoder.

mod dialect;
mod encoder;
mod stream;
mod stream_error;
mod stream_types;
mod types;

pub use dialect::AssistantReasoningReplay;
pub use dialect::DialectContext;
pub use dialect::DialectError;
pub use dialect::DialectHooks;
pub use dialect::OpaqueReasoning;
pub use dialect::ReasoningDelta;
pub use dialect::UsageDetails;
pub use encoder::EncodeError;
pub use encoder::EncodeRequest;
pub use encoder::encode_request;
pub use stream::DecodeStream;
pub use stream::decode_stream;
pub use stream_error::DecodeError;
pub use stream_types::ChatCompletionChunk;
pub use stream_types::ChunkChoice;
pub use stream_types::ChunkDelta;
pub use stream_types::ChunkUsage;
pub use stream_types::DecodedStream;
pub use stream_types::FinishReason;
pub use stream_types::PendingResult;
pub use stream_types::PresentationDelta;
pub use stream_types::ResponseMetadata;
pub use stream_types::ToolCallFragment;
pub use stream_types::ToolFunctionFragment;
pub use types::AssistantMessage;
pub use types::ChatCompletionsRequest;
pub use types::ChatMessage;
pub use types::ContentPart;
pub use types::FunctionDefinition;
pub use types::FunctionTool;
pub use types::FunctionToolKind;
pub use types::ImageUrl;
pub use types::MessageContent;
pub use types::ToolCall;
pub use types::ToolCallFunction;
pub use types::ToolCallKind;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[cfg(test)]
#[path = "stream_tests.rs"]
mod stream_tests;
