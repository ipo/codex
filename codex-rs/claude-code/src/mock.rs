use serde_json::Value;
use serde_json::json;

pub(crate) fn event(name: &str, data: Value) -> String {
    format!("event: {name}\ndata: {data}\n\n")
}

#[rustfmt::skip]
pub(crate) fn message_start(usage: Value) -> String {
    event("message_start", json!({"type":"message_start","message":{"id":"msg-1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":usage}}))
}

pub(crate) fn block_start(index: usize, content_block: Value) -> String {
    event(
        "content_block_start",
        json!({"type":"content_block_start","index":index,"content_block":content_block}),
    )
}

pub(crate) fn delta(index: usize, delta: Value) -> String {
    event(
        "content_block_delta",
        json!({"type":"content_block_delta","index":index,"delta":delta}),
    )
}

pub(crate) fn block_stop(index: usize) -> String {
    event(
        "content_block_stop",
        json!({"type":"content_block_stop","index":index}),
    )
}

pub(crate) fn finish(reason: Value, usage: Value) -> String {
    event(
        "message_delta",
        json!({"type":"message_delta","delta":{"stop_reason":reason},"usage":usage}),
    ) + &event("message_stop", json!({"type":"message_stop"}))
}

pub(crate) fn minimal_stream(reason: Value, usage: Value) -> String {
    message_start(json!({"input_tokens":10,"output_tokens":0})) + &finish(reason, usage)
}
