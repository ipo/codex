use super::*;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::models::ResponseItem;
use codex_tools::JsonSchema;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolSpec;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use std::cell::RefCell;
use std::collections::BTreeMap;

type Extensions = BTreeMap<String, Value>;

struct RecordingDialect(RefCell<Vec<(InferenceDialect, String)>>);

impl RecordingDialect {
    fn record(&self, context: DialectContext<'_>) {
        self.0
            .borrow_mut()
            .push((context.dialect, context.model.to_string()));
    }
}

impl DialectHooks for RecordingDialect {
    fn request_extensions(&self, context: DialectContext<'_>) -> Result<Extensions, DialectError> {
        self.record(context);
        Ok(BTreeMap::from([(
            "dialect_request".into(),
            json!({"dialect": context.dialect.to_string(), "model": context.model}),
        )]))
    }

    fn assistant_reasoning(
        &self,
        context: DialectContext<'_>,
        replay: AssistantReasoningReplay<'_>,
    ) -> Result<Extensions, DialectError> {
        self.record(context);
        if replay.opaque == OpaqueReasoning::Other("malformed") {
            return Err(DialectError::MalformedReplay("invalid fixture".into()));
        }
        let opaque = match replay.opaque {
            OpaqueReasoning::None => None,
            OpaqueReasoning::Other(opaque) => Some(opaque),
        };
        Ok(BTreeMap::from([(
            "reasoning_content".into(),
            json!({"visible": replay.visible, "opaque": opaque}),
        )]))
    }

    fn reasoning_delta(
        &self,
        context: DialectContext<'_>,
        _extensions: &Extensions,
    ) -> Result<Option<ReasoningDelta>, DialectError> {
        self.record(context);
        Ok(None)
    }

    #[rustfmt::skip]
    fn finish_reason(&self, context: DialectContext<'_>, reason: FinishReason) -> Result<codex_api::TerminalOutcome, DialectError> {
        self.record(context);
        Ok(match reason {
            FinishReason::Stop => codex_api::TerminalOutcome::Completed,
            FinishReason::ToolCalls | FinishReason::FunctionCall => codex_api::TerminalOutcome::ToolsReady,
            FinishReason::Length | FinishReason::MaxTokens => codex_api::TerminalOutcome::OutputExhausted,
            FinishReason::ContentFilter => codex_api::TerminalOutcome::Refusal,
        })
    }

    fn usage_details(
        &self,
        context: DialectContext<'_>,
        _usage: &ChunkUsage,
    ) -> Result<UsageDetails, DialectError> {
        self.record(context);
        Ok(UsageDetails::default())
    }
}

fn item(value: Value) -> ResponseItem {
    serde_json::from_value(value).expect("canonical item fixture")
}

fn encode<'a>(
    context: DialectContext<'a>,
    history: &'a [ResponseItem],
    tools: &'a [ToolSpec],
    dialect: &'a dyn DialectHooks,
) -> Result<ChatCompletionsRequest, EncodeError> {
    encode_request(EncodeRequest {
        context,
        system: Some("You are Codex."),
        history,
        tools,
        dialect,
    })
}

#[test]
fn encodes_complete_ordered_request_parallel_calls_results_and_schema() {
    let history = [
        item(json!({"type":"message", "role":"user", "content":[
            {"type":"input_text", "text":"inspect "},
            {"type":"input_image", "image_url":"data:image/png;base64,aGVsbG8="}
        ]})),
        item(json!({"type":"reasoning", "summary":[],
            "content":[{"type":"reasoning_text", "text":"think exactly"}],
            "encrypted_content":"replay-v1"})),
        item(json!({"type":"message", "role":"assistant",
            "content":[{"type":"output_text", "text":"Checking both."}]})),
        item(
            json!({"type":"function_call", "name":"read", "arguments":"{ \"path\": \"a\" }", "call_id":"call-a"}),
        ),
        item(
            json!({"type":"function_call", "name":"read", "arguments":"{\"path\":\"b\"}", "call_id":"call-b"}),
        ),
        item(json!({"type":"function_call_output", "call_id":"call-b", "output":"second result"})),
        item(json!({"type":"function_call_output", "call_id":"call-a", "output":"first result"})),
    ];
    let schema = json!({
        "type":"object", "properties": {
            "target":{"$ref":"#/$defs/target"},
            "options":{"type":"array", "items":{"oneOf":[{"type":"string"}, {"type":"integer"}]}}
        },
        "required":["target"], "additionalProperties":false,
        "$defs":{"target":{"type":"string", "enum":["a", "b"]}}
    });
    let tools = [ToolSpec::Function(ResponsesApiTool {
        name: "read".into(),
        description: "Read one target".into(),
        strict: true,
        defer_loading: None,
        parameters: serde_json::from_value::<JsonSchema>(schema.clone()).expect("nested schema"),
        output_schema: None,
    })];
    let dialect = RecordingDialect(RefCell::default());
    let context = DialectContext {
        dialect: InferenceDialect::Kimi,
        model: "gpt-looking-but-resolved-kimi",
    };
    let request = encode(context, &history, &tools, &dialect).expect("complete request");
    assert_eq!(
        serde_json::to_value(request).expect("serialize request"),
        json!({
            "model":"gpt-looking-but-resolved-kimi",
            "messages":[
                {"role":"system", "content":"You are Codex."},
                {"role":"user", "content":[
                    {"type":"text", "text":"inspect "},
                    {"type":"image_url", "image_url":{"url":"data:image/png;base64,aGVsbG8="}}
                ]},
                {"role":"assistant", "content":"Checking both.",
                 "reasoning_content":{"visible":["think exactly"], "opaque":"replay-v1"},
                 "tool_calls":[
                    {"id":"call-a", "type":"function", "function":{"name":"read", "arguments":"{ \"path\": \"a\" }"}},
                    {"id":"call-b", "type":"function", "function":{"name":"read", "arguments":"{\"path\":\"b\"}"}}
                 ]},
                {"role":"tool", "tool_call_id":"call-b", "content":"second result"},
                {"role":"tool", "tool_call_id":"call-a", "content":"first result"}
            ],
            "tools":[{"type":"function", "function":{
                "name":"read", "description":"Read one target", "parameters":schema, "strict":true
            }}],
            "dialect_request":{"dialect":"kimi", "model":"gpt-looking-but-resolved-kimi"}
        })
    );
    assert_eq!(
        dialect.0.into_inner(),
        vec![(context.dialect, context.model.into()); 2]
    );
}

#[test]
fn failures_are_typed_and_return_no_partial_request() {
    let context = DialectContext {
        dialect: InferenceDialect::OpenAi,
        model: "kimi-looking-but-resolved-openai",
    };
    let dialect = RecordingDialect(RefCell::default());
    let cases = [
        (
            item(json!({"type":"function_call", "name":"tool", "arguments":"[]", "call_id":"bad"})),
            EncodeError::InvalidToolArguments {
                call_id: "bad".into(),
                message: "expected a JSON object".into(),
            },
        ),
        (
            item(json!({"type":"function_call_output", "call_id":"missing", "output":"orphan"})),
            EncodeError::UnmatchedToolResult {
                call_id: "missing".into(),
            },
        ),
        (
            item(json!({"type":"reasoning", "summary":[], "encrypted_content":"malformed"})),
            EncodeError::Dialect(DialectError::MalformedReplay("invalid fixture".into())),
        ),
        (
            item(
                json!({"type":"custom_tool_call", "call_id":"free", "name":"free", "input":"raw"}),
            ),
            EncodeError::UnsupportedHistoryItem {
                index: 0,
                kind: "freeform tool call",
            },
        ),
    ];
    for (history, expected) in cases {
        assert_eq!(encode(context, &[history], &[], &dialect), Err(expected));
    }

    let unsupported = [ToolSpec::ToolSearch {
        execution: "server".into(),
        description: "search".into(),
        parameters: JsonSchema::default(),
    }];
    assert_eq!(
        encode(context, &[], &unsupported, &dialect),
        Err(EncodeError::UnsupportedTool {
            index: 0,
            kind: "tool search"
        })
    );
}
