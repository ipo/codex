//! Native Anthropic Messages wire types and opaque thinking replay support.

mod policy;
mod replay;
mod types;

pub use policy::AssembleError;
pub use policy::AssembleRequest;
pub use policy::AssembledRequest;
pub use policy::RequestTransport;
pub use policy::assemble_request;
pub use replay::ReplayDecision;
pub use replay::ReplayError;
pub use replay::ThinkingReplayBlock;
pub use replay::decode_thinking_replay;
pub use replay::encode_thinking_replay;
pub use types::CacheControl;
pub use types::CacheCreationUsage;
pub use types::CacheTtl;
pub use types::ContentBlock;
pub use types::ContextEdit;
pub use types::ContextKeep;
pub use types::ContextManagement;
pub use types::ImageSource;
pub use types::Message;
pub use types::MessagesRequest;
pub use types::OutputConfig;
pub use types::OutputEffort;
pub use types::RequestMetadata;
pub use types::Role;
pub use types::SystemBlock;
pub use types::Thinking;
pub use types::ThinkingDisplay;
pub use types::Tool;
pub use types::ToolResultBlock;
pub use types::ToolResultContent;
pub use types::Usage;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[cfg(test)]
#[path = "policy_tests.rs"]
mod policy_tests;
