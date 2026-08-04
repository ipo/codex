use std::collections::BTreeMap;

use crate::FinishReason;
use crate::stream_types::ChunkUsage;
use codex_api::TerminalOutcome;
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
    pub opaque: OpaqueReasoning<'a>,
}

/// Provider provenance for opaque canonical reasoning material.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpaqueReasoning<'a> {
    None,
    Other(&'a str),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningDelta {
    pub text: String,
    pub provenance: String,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UsageDetails {
    pub cached_prompt_tokens: u64,
    pub reasoning_tokens: u64,
}

/// Controls how streamed usage reports are combined by the generic decoder.
///
/// The policy applies both to choice-level versus top-level usage within one
/// chunk and to usage reported by successive chunks. Providers should select
/// `PreferLatest` only when their protocol defines later reports as authoritative.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum UsageMergePolicy {
    /// Repeated decoded usage must be identical.
    #[default]
    RequireIdentical,
    /// Replace prior usage with the most recently decoded report.
    PreferLatest,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DialectError {
    #[error("malformed assistant reasoning replay: {0}")]
    MalformedReplay(String),
    #[error("invalid request extension: {0}")]
    InvalidRequestExtension(String),
    #[error("invalid streamed response extension: {0}")]
    InvalidResponseExtension(String),
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

    fn reasoning_delta(
        &self,
        context: DialectContext<'_>,
        extensions: &BTreeMap<String, Value>,
    ) -> Result<Option<ReasoningDelta>, DialectError> {
        response_hook_unavailable(context, extensions)
    }

    fn finish_reason(
        &self,
        context: DialectContext<'_>,
        reason: FinishReason,
    ) -> Result<TerminalOutcome, DialectError> {
        response_hook_unavailable(context, reason)
    }

    fn usage_details(
        &self,
        context: DialectContext<'_>,
        usage: &ChunkUsage,
    ) -> Result<UsageDetails, DialectError> {
        response_hook_unavailable(context, usage)
    }

    /// Selects usage precedence for the exact resolved dialect and model.
    ///
    /// The default preserves strict conflict rejection for all dialects.
    fn usage_merge_policy(
        &self,
        _context: DialectContext<'_>,
    ) -> Result<UsageMergePolicy, DialectError> {
        Ok(UsageMergePolicy::RequireIdentical)
    }
}

fn response_hook_unavailable<T, U, V>(_: T, _: U) -> Result<V, DialectError> {
    Err(DialectError::InvalidResponseExtension(
        "response hook is not implemented".to_string(),
    ))
}
