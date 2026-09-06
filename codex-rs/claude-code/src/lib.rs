//! Native Claude Code request and response adapters.

mod claude_code_identity;
mod encoder;
mod history;
mod history_content;
mod opus_compatibility;
mod policy;
mod replay;
mod sonnet_compatibility;
mod stream;
mod transport;
mod types;

#[cfg(test)]
mod mock;

pub use claude_code_identity::ClaudeCodeIdentity;
pub use claude_code_identity::ClaudeCodeRequestKind;
pub use codex_protocol::model_inference::AnthropicThinkingPolicy;
pub use codex_protocol::model_inference::InferenceDialect;
pub use codex_protocol::model_inference::ModelInferenceConfig;
pub use codex_protocol::model_inference::WireApi;
pub use codex_protocol::openai_models::ReasoningEffort;
pub use codex_protocol::protocol::TokenUsage;
pub use encoder::CanonicalOutputSchema;
pub use encoder::EncodeError;
pub use encoder::EncodeRequest;
pub use encoder::encode_request;
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
pub use transport::ClaudeHttpAdapter;
pub use transport::ClaudeResponseStream;
pub use transport::NativeStreamError;
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
pub use types::TerminalOutcome;
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

#[cfg(test)]
#[path = "transport_tests.rs"]
mod transport_tests;
