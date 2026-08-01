use super::*;
use codex_api::TerminalOutcome;
use codex_protocol::protocol::TokenUsage;
use pretty_assertions::assert_eq;
use serde_json::json;

fn event(name: &str, data: serde_json::Value) -> String {
    format!("event: {name}\ndata: {data}\n\n")
}

#[rustfmt::skip]
fn start(usage: serde_json::Value) -> String {
    event("message_start", json!({"type":"message_start","message":{"id":"msg-1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":usage}}))
}

fn block_start(index: usize, content_block: serde_json::Value) -> String {
    event(
        "content_block_start",
        json!({"type":"content_block_start","index":index,"content_block":content_block}),
    )
}

fn delta(index: usize, delta: serde_json::Value) -> String {
    event(
        "content_block_delta",
        json!({"type":"content_block_delta","index":index,"delta":delta}),
    )
}

fn block_stop(index: usize) -> String {
    event(
        "content_block_stop",
        json!({"type":"content_block_stop","index":index}),
    )
}

fn finish(reason: serde_json::Value, usage: serde_json::Value) -> String {
    event(
        "message_delta",
        json!({"type":"message_delta","delta":{"stop_reason":reason},"usage":usage}),
    ) + &event("message_stop", json!({"type":"message_stop"}))
}

fn minimal(reason: serde_json::Value, usage: serde_json::Value) -> String {
    start(json!({"input_tokens":10,"output_tokens":0})) + &finish(reason, usage)
}

#[rustfmt::skip]
fn expected_usage(write: Option<u64>, read: Option<u64>, thinking: Option<u64>) -> (Usage, TokenUsage) {
    let input = 10 + write.unwrap_or(0) as i64 + read.unwrap_or(0) as i64;
    (
        Usage { input_tokens:10, output_tokens:4, cache_creation_input_tokens:write, cache_read_input_tokens:read, thinking_tokens:thinking, cache_creation:None },
        TokenUsage { input_tokens:input, cached_input_tokens:read.unwrap_or(0) as i64, cache_write_input_tokens:write.unwrap_or(0) as i64, output_tokens:4, reasoning_output_tokens:thinking.unwrap_or(0) as i64, total_tokens:input + 4 },
    )
}

#[test]
#[rustfmt::skip]
fn reconstructs_ordered_interleaved_blocks_and_usage() {
    let fixture = [
        event("ping", json!({"type":"ping"})),
        start(json!({"input_tokens":100,"output_tokens":0,"cache_creation_input_tokens":20,"cache_read_input_tokens":30,"thinking_tokens":null,"cache_creation":{"ephemeral_5m_input_tokens":7,"ephemeral_1h_input_tokens":13}})),
        block_start(0, json!({"type":"thinking","thinking":""})),
        block_start(1, json!({"type":"redacted_thinking","data":"opaque"})),
        block_start(2, json!({"type":"text","text":""})),
        block_start(3, json!({"type":"tool_use","id":"call-a","name":"alpha","input":{}})),
        block_start(4, json!({"type":"tool_use","id":"call-b","name":"beta","input":{}})),
        delta(3, json!({"type":"input_json_delta","partial_json":"{\"x\":"})),
        delta(0, json!({"type":"thinking_delta","thinking":"consider "})),
        delta(4, json!({"type":"input_json_delta","partial_json":"{\"y\":"})),
        delta(2, json!({"type":"text_delta","text":"hello "})),
        delta(3, json!({"type":"input_json_delta","partial_json":"1}"})),
        delta(0, json!({"type":"thinking_delta","thinking":"carefully"})),
        delta(0, json!({"type":"signature_delta","signature":"sig"})),
        delta(4, json!({"type":"input_json_delta","partial_json":"2}"})),
        delta(2, json!({"type":"text_delta","text":"world"})),
        block_stop(0), block_stop(1), block_stop(2), block_stop(3), block_stop(4),
        finish("tool_use".into(), json!({"output_tokens":40,"thinking_tokens":11})),
    ].concat();
    let mut presentation = Vec::new();
    let decoded = decode_stream(fixture.as_bytes(), |item| presentation.push(item)).unwrap();

    assert_eq!(decoded, DecodedStream {
        message_id: "msg-1".into(), model: "claude-test".into(),
        blocks: vec![
            DecodedBlock::Thinking { thinking: "consider carefully".into(), signature: "sig".into() },
            DecodedBlock::RedactedThinking { data: "opaque".into() },
            DecodedBlock::Text { text: "hello world".into() },
            DecodedBlock::ToolUse { id: "call-a".into(), name: "alpha".into(), input: json!({"x":1}) },
            DecodedBlock::ToolUse { id: "call-b".into(), name: "beta".into(), input: json!({"y":2}) },
        ],
        terminal_outcome: TerminalOutcome::ToolsReady,
        usage: Usage { input_tokens:100, output_tokens:40, cache_creation_input_tokens:Some(20), cache_read_input_tokens:Some(30), thinking_tokens:Some(11), cache_creation:Some(CacheCreationUsage { ephemeral_5m_input_tokens:7, ephemeral_1h_input_tokens:13 }) },
        token_usage: TokenUsage { input_tokens:150, cached_input_tokens:30, cache_write_input_tokens:20, output_tokens:40, reasoning_output_tokens:11, total_tokens:190 },
    });
    assert_eq!(presentation, vec![
        PresentationDelta::ToolInput { index:3, delta:"{\"x\":".into() },
        PresentationDelta::Thinking { index:0, delta:"consider ".into() },
        PresentationDelta::ToolInput { index:4, delta:"{\"y\":".into() },
        PresentationDelta::Text { index:2, delta:"hello ".into() },
        PresentationDelta::ToolInput { index:3, delta:"1}".into() },
        PresentationDelta::Thinking { index:0, delta:"carefully".into() },
        PresentationDelta::ToolInput { index:4, delta:"2}".into() },
        PresentationDelta::Text { index:2, delta:"world".into() },
    ]);
}

#[test]
#[rustfmt::skip]
fn maps_every_supported_stop_reason_exhaustively() {
    let cases = [
        ("end_turn", TerminalOutcome::Completed), ("stop_sequence", TerminalOutcome::Completed),
        ("tool_use", TerminalOutcome::ToolsReady), ("pause_turn", TerminalOutcome::Continue),
        ("max_tokens", TerminalOutcome::OutputExhausted),
        ("refusal", TerminalOutcome::Refusal), ("safety", TerminalOutcome::Refusal),
    ];
    for (reason, expected) in cases {
        let decoded = decode_stream(minimal(reason.into(), json!({"output_tokens":2})).as_bytes(), |_| {}).unwrap();
        assert_eq!(decoded.terminal_outcome, expected, "stop reason {reason}");
    }
    assert_eq!(decode_stream(minimal("future_reason".into(), json!({"output_tokens":2})).as_bytes(), |_| {}), Err(DecodeError::UnknownStopReason("future_reason".into())));
}

#[test]
#[rustfmt::skip]
fn usage_preserves_optional_native_counters_without_double_counting() {
    let cases = [
        (json!({"input_tokens":10,"output_tokens":0}), json!({"output_tokens":4}), (None,None,None)),
        (json!({"input_tokens":10,"output_tokens":0,"cache_creation_input_tokens":null,"cache_read_input_tokens":null}), json!({"output_tokens":4,"thinking_tokens":null}), (None,None,None)),
        (json!({"input_tokens":10,"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}), json!({"output_tokens":4,"thinking_tokens":0}), (Some(0),Some(0),Some(0))),
        (json!({"input_tokens":10,"output_tokens":0,"cache_creation_input_tokens":3,"cache_read_input_tokens":2}), json!({"output_tokens":4,"thinking_tokens":1}), (Some(3),Some(2),Some(1))),
    ];
    for (start_usage, delta_usage, expected) in cases {
        let fixture = minimal("end_turn".into(), delta_usage).replace(&start(json!({"input_tokens":10,"output_tokens":0})), &start(start_usage));
        let decoded = decode_stream(fixture.as_bytes(), |_| {}).unwrap();
        assert_eq!((decoded.usage, decoded.token_usage), expected_usage(expected.0, expected.1, expected.2));
    }
}

#[test]
#[rustfmt::skip]
fn every_incomplete_or_malformed_terminal_path_is_typed_failure() {
    let invalid_transition = start(json!({"input_tokens":1,"output_tokens":0})) + &block_stop(0);
    let invalid_json = [
        start(json!({"input_tokens":1,"output_tokens":0})),
        block_start(0, json!({"type":"tool_use","id":"c","name":"f","input":{}})),
        delta(0, json!({"type":"input_json_delta","partial_json":"{"})), block_stop(0),
        finish("tool_use".into(), json!({"output_tokens":1})),
    ].concat();
    let cases = [
        (Vec::new(), DecodeError::EmptyStream),
        (event("message_start", json!({"type":"message_start"})).into_bytes(), DecodeError::MalformedEvent { event:"message_start".into(), message:"missing field `message`".into() }),
        (event("error", json!({"type":"error","error":{"type":"overloaded_error","message":"busy"}})).into_bytes(), DecodeError::ProviderError { error_type:"overloaded_error".into(), message:"busy".into() }),
        (invalid_transition.into_bytes(), DecodeError::InvalidTransition { event:"content_block_stop".into(), reason:"stop referenced an unknown block".into() }),
        (invalid_json.into_bytes(), DecodeError::InvalidToolJson { index:0, input:"{".into() }),
        (minimal(json!(null), json!({"output_tokens":1})).into_bytes(), DecodeError::MissingStopReason),
        (start(json!({"input_tokens":1,"output_tokens":0})).into_bytes(), DecodeError::PrematureEof { expected:"recognized non-null stop reason".into() }),
    ];
    for (fixture, expected) in cases { assert_eq!(decode_stream(&fixture, |_| {}), Err(expected)); }
}

#[test]
#[rustfmt::skip]
fn presentation_deltas_survive_failure_without_a_pending_result() {
    let fixture = [
        start(json!({"input_tokens":1,"output_tokens":0})),
        block_start(0, json!({"type":"thinking","thinking":""})),
        delta(0, json!({"type":"thinking_delta","thinking":"partial"})),
        event("error", json!({"type":"error","error":{"type":"api_error","message":"failed"}})),
    ].concat();
    let mut presentation = Vec::new();
    let result = decode_stream(fixture.as_bytes(), |item| presentation.push(item));
    assert_eq!(result, Err(DecodeError::ProviderError { error_type:"api_error".into(), message:"failed".into() }));
    assert_eq!(presentation, vec![PresentationDelta::Thinking { index:0, delta:"partial".into() }]);
}
