use crate::DialectError;
use thiserror::Error;

#[derive(Debug, PartialEq, Eq, Error)]
pub enum DecodeError {
    #[error("Chat Completions stream was empty")]
    EmptyStream,
    #[error("Chat Completions stream was not UTF-8")]
    InvalidUtf8,
    #[error("malformed data stream framing: {0}")]
    MalformedFraming(String),
    #[error("malformed Chat Completions chunk: {0}")]
    MalformedChunk(String),
    #[error("invalid Chat Completions transition: {0}")]
    InvalidTransition(String),
    #[error("Chat Completions finish reason was missing")]
    MissingFinishReason,
    #[error("Chat Completions finish reason was null")]
    NullFinishReason,
    #[error("unknown Chat Completions finish reason {0}")]
    UnknownFinishReason(String),
    #[error("Chat Completions stream contained a duplicate terminal")]
    DuplicateTerminal,
    #[error("Chat Completions stream contained conflicting terminal state")]
    ConflictingTerminal,
    #[error("tool call {index} was incomplete: {field}")]
    IncompleteTool { index: usize, field: String },
    #[error("tool call {index} contained invalid JSON object: {arguments}")]
    InvalidToolJson { index: usize, arguments: String },
    #[error("Chat Completions stream contained conflicting usage")]
    ConflictingUsage,
    #[error("Chat Completions stream ended before {expected}")]
    PrematureEof { expected: String },
    #[error(transparent)]
    Dialect(#[from] DialectError),
}
