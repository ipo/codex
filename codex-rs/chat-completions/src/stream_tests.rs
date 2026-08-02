#[rustfmt::skip]
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::Mutex;

use codex_api::TerminalOutcome;
use codex_protocol::model_inference::InferenceDialect;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;

use super::*;

#[derive(Default)]
struct TestDialect(Mutex<Vec<(InferenceDialect, String, &'static str)>>);

#[rustfmt::skip]
impl TestDialect {
    fn record(&self, context: DialectContext<'_>, hook: &'static str) { self.0.lock().expect("test dialect mutex").push((context.dialect, context.model.to_string(), hook)); }
}

#[rustfmt::skip]
impl DialectHooks for TestDialect {
    fn request_extensions(&self, _context: DialectContext<'_>) -> Result<BTreeMap<String, Value>, DialectError> { unreachable!() }
    fn assistant_reasoning(&self, _context: DialectContext<'_>, _replay: AssistantReasoningReplay<'_>) -> Result<BTreeMap<String, Value>, DialectError> { unreachable!() }
    fn reasoning_delta(&self, context: DialectContext<'_>, extensions: &BTreeMap<String, Value>) -> Result<Option<ReasoningDelta>, DialectError> {
        self.record(context, "reasoning");
        Ok(extensions.get("reasoning_content").and_then(Value::as_str).map(|text| ReasoningDelta { text:text.to_string(), provenance:"reasoning_content".into() }))
    }
    fn finish_reason(&self, context: DialectContext<'_>, reason: FinishReason) -> Result<TerminalOutcome, DialectError> {
        self.record(context, "finish");
        Ok(match reason {
            FinishReason::Stop => TerminalOutcome::Completed,
            FinishReason::ToolCalls | FinishReason::FunctionCall => TerminalOutcome::ToolsReady,
            FinishReason::Length | FinishReason::MaxTokens => TerminalOutcome::OutputExhausted,
            FinishReason::ContentFilter => TerminalOutcome::Refusal,
        })
    }
    fn usage_details(&self, context: DialectContext<'_>, usage: &ChunkUsage) -> Result<UsageDetails, DialectError> {
        self.record(context, "usage");
        let reasoning_tokens = usage.details.get("completion_tokens_details").and_then(|value| value.get("reasoning_tokens")).and_then(Value::as_u64).unwrap_or(0);
        Ok(UsageDetails { cached_prompt_tokens:0, reasoning_tokens })
    }
}

fn frame(value: Value) -> String {
    format!("data: {value}\r\ndata:\r\n\r\n")
}

#[rustfmt::skip]
fn choice(delta: Value, finish_reason: Value, usage: Option<Value>) -> Value {
    let mut value = json!({"index":0,"delta":delta,"finish_reason":finish_reason});
    if let Some(usage) = usage { value["usage"] = usage; }
    value
}

#[rustfmt::skip]
fn chunk(choices: Vec<Value>, usage: Option<Value>) -> String {
    let mut value = json!({"id":"chat-1","choices":choices});
    if let Some(usage) = usage { value["usage"] = usage; }
    frame(value)
}

#[rustfmt::skip]
fn decode(body: &str, dialect: &TestDialect) -> (Result<DecodedStream, DecodeError>, Vec<PresentationDelta>) {
    let mut presentation = Vec::new();
    let result = decode_stream(params(dialect), body.as_bytes().chunks(1), |delta| presentation.push(delta));
    (result, presentation)
}

#[rustfmt::skip]
fn params(dialect: &TestDialect) -> DecodeStream<'_> { DecodeStream { context:DialectContext { dialect:InferenceDialect::Kimi, model:"deliberately-misleading-openai-slug" }, dialect, metadata:ResponseMetadata { trace_id:Some("trace-7".into()) } } }

#[test]
#[rustfmt::skip]
fn reconstructs_fragmented_interleaved_reasoning_content_parallel_tools_and_metadata() {
    let usage = json!({"prompt_tokens":10,"completion_tokens":8,"total_tokens":18,"completion_tokens_details":{"reasoning_tokens":3}});
    let body = [
        ": keepalive\r\n\r\n".into(),
        chunk(vec![choice(json!({"role":"assistant","reasoning_content":"th🧠ink "}), Value::Null, None)], None),
        chunk(vec![choice(json!({"tool_calls":[{"index":4,"id":"call-b","type":"function","function":{"name":"beta","arguments":"{\"b\":"}}]}), Value::Null, None)], None),
        chunk(vec![choice(json!({"content":"hello ","tool_calls":[{"index":2,"id":"call-a","type":"function","function":{"name":"alpha","arguments":"{\"a\":"}}]}), Value::Null, None)], None),
        chunk(vec![choice(json!({"reasoning_content":"carefully","tool_calls":[{"index":4,"function":{"arguments":"2}"}},{"index":2,"function":{"arguments":"1}"}}]}), "tool_calls".into(), Some(usage.clone()))], Some(usage.clone())),
        "data: [DONE]\n\n".into(),
    ].concat();
    let dialect = TestDialect::default();
    let (decoded, presentation) = decode(&body, &dialect);
    assert_eq!(decoded, Ok(DecodedStream {
        response_id:"chat-1".into(), terminal_outcome:TerminalOutcome::ToolsReady,
        pending:Some(PendingResult { content:"hello ".into(), reasoning:"th🧠ink carefully".into(), reasoning_provenance:Some("reasoning_content".into()), tool_calls:vec![
            ToolCall { id:"call-b".into(), kind:ToolCallKind::Function, function:ToolCallFunction { name:"beta".into(), arguments:"{\"b\":2}".into() } },
            ToolCall { id:"call-a".into(), kind:ToolCallKind::Function, function:ToolCallFunction { name:"alpha".into(), arguments:"{\"a\":1}".into() } },
        ]}), usage:Some(serde_json::from_value(usage).unwrap()), usage_details:UsageDetails { cached_prompt_tokens:0, reasoning_tokens:3 }, metadata:ResponseMetadata { trace_id:Some("trace-7".into()) },
    }));
    assert_eq!(presentation, vec![
        PresentationDelta::Reasoning("th🧠ink ".into()),
        PresentationDelta::Tool { index:4, id:"call-b".into(), name:"beta".into(), delta:"{\"b\":".into() },
        PresentationDelta::Content("hello ".into()),
        PresentationDelta::Tool { index:2, id:"call-a".into(), name:"alpha".into(), delta:"{\"a\":".into() },
        PresentationDelta::Reasoning("carefully".into()), PresentationDelta::Tool { index:4, id:"call-b".into(), name:"beta".into(), delta:"2}".into() },
        PresentationDelta::Tool { index:2, id:"call-a".into(), name:"alpha".into(), delta:"1}".into() },
    ]);
    assert!(dialect.0.lock().unwrap().iter().all(|(dialect, model, _)| *dialect == InferenceDialect::Kimi && model == "deliberately-misleading-openai-slug"));
    assert_eq!(dialect.0.lock().unwrap().iter().map(|entry| entry.2).collect::<Vec<_>>(), vec!["reasoning","reasoning","reasoning","reasoning","finish","usage"]);
}

#[test]
#[rustfmt::skip]
fn presents_a_complete_frame_before_requesting_later_fragments() {
    let seen = Rc::new(RefCell::new(Vec::new()));
    let iterator_seen = Rc::clone(&seen);
    let step = Cell::new(0);
    let fragments = std::iter::from_fn(|| match step.replace(step.get() + 1) {
        0 => Some(chunk(vec![choice(json!({"content":"early"}), Value::Null, None)], None)),
        1 => { assert_eq!(*iterator_seen.borrow(), vec![PresentationDelta::Content("early".into())]); Some(chunk(vec![choice(json!({}), "stop".into(), None)], None) + "data: [DONE]\n\n") }
        _ => None,
    });
    let dialect = TestDialect::default();
    assert!(decode_stream(params(&dialect), fragments, |delta| seen.borrow_mut().push(delta)).is_ok());
}

#[test]
#[rustfmt::skip]
fn maps_finish_reasons_and_discards_noncommittable_output() {
    let cases = [
        ("stop", TerminalOutcome::Completed, true), ("tool_calls", TerminalOutcome::ToolsReady, true),
        ("function_call", TerminalOutcome::ToolsReady, true), ("length", TerminalOutcome::OutputExhausted, false),
        ("max_tokens", TerminalOutcome::OutputExhausted, false), ("content_filter", TerminalOutcome::Refusal, false),
    ];
    for (reason, outcome, pending) in cases {
        let delta = if outcome == TerminalOutcome::ToolsReady { json!({"tool_calls":[{"index":0,"id":"c","type":"function","function":{"name":"f","arguments":"{}"}}]}) } else { json!({"content":"partial"}) };
        let body = chunk(vec![choice(delta, reason.into(), None)], None) + "data: [DONE]\n\n";
        let (decoded, _) = decode(&body, &TestDialect::default());
        let decoded = decoded.unwrap();
        assert_eq!((decoded.terminal_outcome, decoded.pending.is_some()), (outcome, pending), "{reason}");
    }
}

#[test]
#[rustfmt::skip]
fn rejects_every_strict_failure_without_returning_pending_tools() {
    let valid_tool = json!({"tool_calls":[{"index":0,"id":"c","type":"function","function":{"name":"f","arguments":"{}"}}]});
    let incomplete_tool = json!({"tool_calls":[{"index":0,"id":"c","type":"function","function":{"arguments":"{}"}}]});
    let cases = [
        ("", DecodeError::EmptyStream),
        ("data: {\n\n", DecodeError::MalformedChunk("EOF while parsing an object at line 1 column 1".into())),
        (&chunk(vec![json!({"index":1,"delta":{},"finish_reason":"stop"})], None), DecodeError::InvalidTransition("unsupported choice index 1".into())),
        (&(chunk(vec![choice(json!({"tool_calls":[{"index":0,"id":"c","function":{"name":"f","arguments":"[1]"}}]}), "tool_calls".into(), None)], None) + "data: [DONE]\n\n"), DecodeError::InvalidToolJson { index:0, arguments:"[1]".into() }),
        (&(chunk(vec![choice(incomplete_tool, "tool_calls".into(), None)], None) + "data: [DONE]\n\n"), DecodeError::IncompleteTool { index:0, field:"name".into() }),
        (&(chunk(vec![choice(json!({"content":"x"}), "future".into(), None)], None) + "data: [DONE]\n\n"), DecodeError::UnknownFinishReason("future".into())),
        (&(chunk(vec![choice(json!({"content":"x"}), Value::Null, None)], None) + "data: [DONE]\n\n"), DecodeError::NullFinishReason),
        (&(frame(json!({"id":"chat-1","choices":[]})) + "data: [DONE]\n\n"), DecodeError::MissingFinishReason),
        (&chunk(vec![choice(json!({"content":"x"}), "stop".into(), None)], None), DecodeError::PrematureEof { expected:"[DONE]".into() }),
        (&(chunk(vec![choice(valid_tool.clone(), "tool_calls".into(), None)], None) + &chunk(vec![choice(valid_tool, "tool_calls".into(), None)], None) + "data: [DONE]\n\n"), DecodeError::DuplicateTerminal),
        (&(chunk(vec![choice(json!({"content":"x"}), "stop".into(), None)], None) + &chunk(vec![choice(json!({}), "length".into(), None)], None) + "data: [DONE]\n\n"), DecodeError::ConflictingTerminal),
    ];
    for (body, expected) in cases { assert_eq!(decode(body, &TestDialect::default()).0, Err(expected)); }
}

#[test]
#[rustfmt::skip]
fn rejects_conflicting_usage_instead_of_double_counting() {
    let top = json!({"prompt_tokens":1,"completion_tokens":2,"total_tokens":3});
    let choice_usage = json!({"prompt_tokens":1,"completion_tokens":3,"total_tokens":4});
    let body = chunk(vec![choice(json!({"content":"x"}), "stop".into(), Some(choice_usage))], Some(top)) + "data: [DONE]\n\n";
    assert_eq!(decode(&body, &TestDialect::default()).0, Err(DecodeError::ConflictingUsage));
}
