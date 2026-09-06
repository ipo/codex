//! Native Claude Code request and response adapters.

mod claude_code_identity;
mod contracts;
mod opus_compatibility;
mod policy;
mod replay;
mod sonnet_compatibility;
mod stream;
mod types;

#[cfg(test)]
mod mock;

pub use claude_code_identity::ClaudeCodeIdentity;
pub use claude_code_identity::ClaudeCodeRequestKind;
pub use contracts::AnthropicThinkingPolicy;
pub use contracts::ClaudeFunctionTool;
pub use contracts::ClaudeRequestProfile;
pub use contracts::ClaudeToolSpec;
pub use contracts::InferenceDialect;
pub use contracts::ReasoningEffort;
pub use contracts::TerminalOutcome;
pub use contracts::TokenUsage;
pub use contracts::WireApi;
pub use opus_compatibility::ClaudeCodeEnvironment;
pub use opus_compatibility::OpusCompatibilityContext;
pub use opus_compatibility::OpusEnvironment;
pub use opus_compatibility::OpusRequestKind;
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
pub use sonnet_compatibility::SonnetCompatibilityContext;
pub use stream::DecodeError;
pub use stream::DecodedBlock;
pub use stream::DecodedStream;
pub use stream::IncrementalDecoder;
pub use stream::PresentationDelta;
pub use stream::decode_stream;
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
#[path = "replay_tests.rs"]
mod replay_tests;

#[cfg(test)]
#[path = "policy_tests.rs"]
mod policy_tests;

#[cfg(test)]
#[path = "stream_tests.rs"]
mod stream_tests;
