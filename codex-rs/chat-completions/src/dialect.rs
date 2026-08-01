use std::collections::BTreeMap;

use codex_protocol::model_inference::InferenceDialect;
use serde_json::Value;
use thiserror::Error;

/// Exact resolved dialect and native model supplied to every dialect hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DialectContext<'a> {
    pub dialect: InferenceDialect,
    pub model: &'a str,
}

/// Canonical reasoning material offered for replay in one assistant message.
#[derive(Debug, Clone, Copy)]
pub struct AssistantReasoningReplay<'a> {
    pub visible: &'a [String],
    pub opaque: Option<&'a str>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DialectError {
    #[error("malformed assistant reasoning replay: {0}")]
    MalformedReplay(String),
    #[error("invalid request extension: {0}")]
    InvalidRequestExtension(String),
}

/// Adds dialect-owned request fields and replays dialect-owned assistant reasoning.
///
/// Implementations must decide behavior exclusively from the supplied resolved
/// context and replay data. Model-name inference is deliberately outside this contract.
pub trait DialectHooks {
    fn request_extensions(
        &self,
        context: DialectContext<'_>,
    ) -> Result<BTreeMap<String, Value>, DialectError>;

    fn assistant_reasoning(
        &self,
        context: DialectContext<'_>,
        replay: AssistantReasoningReplay<'_>,
    ) -> Result<BTreeMap<String, Value>, DialectError>;
}
