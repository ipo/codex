use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DecodeError {
    #[error("native Claude stream was empty")]
    EmptyStream,
    #[error("native Claude stream was not UTF-8")]
    InvalidUtf8,
    #[error("malformed SSE framing: {0}")]
    MalformedSse(String),
    #[error("malformed {event} event: {message}")]
    MalformedEvent { event: String, message: String },
    #[error("unknown native Claude event {0}")]
    UnknownEvent(String),
    #[error("native Claude error {error_type}: {message}")]
    ProviderError { error_type: String, message: String },
    #[error("invalid native Claude transition for {event}: {reason}")]
    InvalidTransition { event: String, reason: String },
    #[error("native Claude stop reason was missing or null")]
    MissingStopReason,
    #[error("unknown native Claude stop reason {0}")]
    UnknownStopReason(String),
    #[error("tool block {index} contained invalid JSON: {input}")]
    InvalidToolJson { index: usize, input: String },
    #[error("native Claude stream ended before {expected}")]
    PrematureEof { expected: String },
}
