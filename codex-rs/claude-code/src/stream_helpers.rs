use super::DecodeError;
use crate::Usage;
use codex_api::TerminalOutcome;
use codex_protocol::protocol::TokenUsage;

pub(super) fn map_stop_reason(reason: &str) -> Result<TerminalOutcome, DecodeError> {
    match reason {
        "end_turn" | "stop_sequence" => Ok(TerminalOutcome::Completed),
        "tool_use" => Ok(TerminalOutcome::ToolsReady),
        "pause_turn" => Ok(TerminalOutcome::Continue),
        "max_tokens" => Ok(TerminalOutcome::OutputExhausted),
        "refusal" | "safety" => Ok(TerminalOutcome::Refusal),
        other => Err(DecodeError::UnknownStopReason(other.to_string())),
    }
}

pub(super) fn map_usage(usage: &Usage) -> TokenUsage {
    let cached_input_tokens = usage.cache_read_input_tokens.unwrap_or(0) as i64;
    let cache_write_input_tokens = usage.cache_creation_input_tokens.unwrap_or(0) as i64;
    let input_tokens = usage.input_tokens as i64 + cached_input_tokens + cache_write_input_tokens;
    let output_tokens = usage.output_tokens as i64;
    TokenUsage {
        input_tokens,
        cached_input_tokens,
        cache_write_input_tokens,
        output_tokens,
        reasoning_output_tokens: usage.thinking_tokens.unwrap_or(0) as i64,
        total_tokens: input_tokens + output_tokens,
    }
}

pub(super) fn parse_sse(body: &str) -> Result<Vec<(String, String)>, DecodeError> {
    let normalized = body.replace("\r\n", "\n").replace('\r', "\n");
    let mut events = Vec::new();
    let mut event = None;
    let mut data = Vec::new();
    for line in normalized.split('\n').chain(std::iter::once("")) {
        if line.is_empty() {
            if event.is_none() && data.is_empty() {
                continue;
            }
            let event = event.take().ok_or_else(|| {
                DecodeError::MalformedSse("frame had data but no event".to_string())
            })?;
            if data.is_empty() {
                return Err(DecodeError::MalformedSse(format!(
                    "{event} frame had no data"
                )));
            }
            events.push((event, data.join("\n")));
            data.clear();
            continue;
        }
        let (field, value) = line
            .split_once(':')
            .ok_or_else(|| DecodeError::MalformedSse(format!("invalid line {line:?}")))?;
        let value = value.strip_prefix(' ').unwrap_or(value);
        match field {
            "event" if event.replace(value.to_string()).is_none() => {}
            "data" => data.push(value.to_string()),
            "event" => {
                return Err(DecodeError::MalformedSse(
                    "frame had duplicate event fields".to_string(),
                ));
            }
            _ => {
                return Err(DecodeError::MalformedSse(format!(
                    "unsupported SSE field {field}"
                )));
            }
        }
    }
    Ok(events)
}
