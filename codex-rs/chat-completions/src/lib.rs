//! Provider-neutral Chat Completions request grammar and canonical-history encoder.

mod dialect;
mod encoder;
mod types;

pub use dialect::AssistantReasoningReplay;
pub use dialect::DialectContext;
pub use dialect::DialectError;
pub use dialect::DialectHooks;
pub use encoder::EncodeError;
pub use encoder::EncodeRequest;
pub use encoder::encode_request;
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
