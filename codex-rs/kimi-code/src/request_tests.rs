use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::model_inference::KimiInferenceConfig;
use codex_protocol::model_inference::KimiThinkingPolicy;
use codex_protocol::model_inference::WireApi;
use codex_protocol::models::ResponseItem;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolSpec;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;

use super::*;

pub(super) fn profile(model: &str, cap: u32, policy: KimiThinkingPolicy) -> KimiInferenceConfig {
    KimiInferenceConfig {
        wire_api: WireApi::ChatCompletions,
        dialect: InferenceDialect::Kimi,
        route: "typed-route-not-inferred".to_string(),
        wire_model: model.to_string(),
        max_output_tokens: cap,
        thinking: policy,
    }
}

pub(super) fn dialect(
    profile: KimiInferenceConfig,
    context: u64,
    input: u64,
    thinking: KimiThinking,
) -> KimiDialect {
    KimiDialect::new(
        profile,
        KimiRequestSettings {
            context_window: context,
            estimated_input_tokens: input,
            prompt_cache_key: "stable-session-affinity".to_string(),
            thinking,
        },
    )
    .expect("valid Kimi request dialect")
}

pub(super) fn request(dialect: &KimiDialect, items: &[ResponseItem], specs: &[ToolSpec]) -> Value {
    let params = KimiEncodeRequest {
        system: Some("You are Codex."),
        history: items,
        tools: specs,
    };
    serde_json::to_value(encode_request(dialect, params).expect("valid request"))
        .expect("request JSON")
}

#[test]
fn snapshots_all_thinking_profiles_and_completion_budget_edges() {
    use KimiThinkingEffort as Effort;

    let k3 = |model| profile(model, 131_072, KimiThinkingPolicy::RequiredWithEffort);
    let coding = |model| profile(model, 32_768, KimiThinkingPolicy::Required);
    let effort = KimiThinking::Effort;
    let highspeed = coding("kimi-for-coding-highspeed");
    let cases = [
        dialect(k3("k3"), 1_048_576, 0, effort(Effort::Low)),
        dialect(k3("k3"), 1_048_576, 917_505, effort(Effort::High)),
        dialect(k3("k3"), 1_048_576, 1_048_575, effort(Effort::Max)),
        dialect(k3("k3-256k"), 262_144, 0, effort(Effort::Low)),
        dialect(k3("k3-256k"), 262_144, 131_073, effort(Effort::High)),
        dialect(k3("k3-256k"), 262_144, 262_144, effort(Effort::Max)),
        dialect(coding("kimi-for-coding"), 262_144, 0, KimiThinking::Enabled),
        dialect(highspeed, 262_144, 229_377, KimiThinking::Enabled),
    ];
    let requests: Vec<_> = cases.iter().map(|case| request(case, &[], &[])).collect();
    insta::assert_snapshot!(serde_json::to_string_pretty(&requests).expect("snapshot JSON"));
}

#[test]
fn rejects_disabled_or_profile_incompatible_thinking_before_encoding() {
    let k3 = profile("k3", 131_072, KimiThinkingPolicy::RequiredWithEffort);
    let coding = profile("kimi-for-coding", 32_768, KimiThinkingPolicy::Required);
    let settings = |thinking| KimiRequestSettings {
        context_window: 262_144,
        estimated_input_tokens: 0,
        prompt_cache_key: "affinity".to_string(),
        thinking,
    };
    assert_eq!(
        [
            KimiDialect::new(k3.clone(), settings(KimiThinking::Off)).err(),
            KimiDialect::new(k3, settings(KimiThinking::Enabled)).err(),
            KimiDialect::new(
                coding,
                settings(KimiThinking::Effort(KimiThinkingEffort::High)),
            )
            .err(),
        ],
        [
            Some(KimiError::DisabledThinking),
            Some(KimiError::MissingThinkingEffort),
            Some(KimiError::UnsupportedThinkingEffort),
        ]
    );
}

#[test]
fn preserves_reasoning_parallel_calls_and_ordered_results_in_one_message() {
    let long_id = "call-that-is-deliberately-longer-than-the-kimi-sixty-four-byte-limit-aaaa";
    let history = serde_json::from_value::<Vec<ResponseItem>>(json!([
        {"type":"message","role":"user","content":[{"type":"input_text","text":"inspect"}]},
        {"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"parallel thought"}]},
        {"type":"message","role":"assistant","content":[{"type":"output_text","text":""}]},
        {"type":"function_call","name":"first","arguments":"{}","call_id":long_id},
        {"type":"function_call","name":"second","arguments":"{\"n\":2}","call_id":"call-2"},
        {"type":"function_call_output","call_id":"call-2","output":"two"},
        {"type":"function_call_output","call_id":long_id,"output":"one"}
    ]))
    .expect("canonical history");
    let profile = profile(
        "unrelated-slug-shaped-wire-model",
        131_072,
        KimiThinkingPolicy::RequiredWithEffort,
    );
    let dialect = dialect(
        profile,
        262_144,
        0,
        KimiThinking::Effort(KimiThinkingEffort::Max),
    );
    assert_eq!(
        request(&dialect, &history, &[]),
        json!({
            "model":"unrelated-slug-shaped-wire-model","messages":[
                {"role":"system","content":"You are Codex."},{"role":"user","content":"inspect"},
                {"role":"assistant","reasoning_content":"parallel thought","tool_calls":[
                    {"id":"call-that-is-deliberately-longer-than-the-kimi--e0ee9bbd4c486e9b","type":"function","function":{"name":"first","arguments":"{}"}},
                    {"id":"call-2","type":"function","function":{"name":"second","arguments":"{\"n\":2}"}}
                ]},{"role":"tool","tool_call_id":"call-2","content":"two"},
                {"role":"tool","tool_call_id":"call-that-is-deliberately-longer-than-the-kimi--e0ee9bbd4c486e9b","content":"one"}
            ],"max_completion_tokens":131072,"prompt_cache_key":"stable-session-affinity","stream":true,
            "stream_options":{"include_usage":true},"thinking":{"type":"enabled","keep":"all","effort":"max"}
        })
    );
}

#[test]
fn normalizes_supported_tool_schema_without_losing_constraints() {
    let schema = json!({
        "properties": {
            "target":{"$ref":"#/$defs/Target","description":"kept"},
            "options":{"properties":{"mode":{"enum":["fast","safe"]}},"required":["mode"]},
            "path":{"pattern":"^src/","minLength":1},
            "limit":{"minimum":1,"maximum":100},
            "tags":{"items":{"const":"code"},"minItems":1,"uniqueItems":true}
        },
        "required":["target"],"additionalProperties":false,
        "$defs":{"Target":{"type":"string","enum":["a","b"]}}
    });
    let tools = [ToolSpec::Function(ResponsesApiTool {
        name: "inspect".to_string(),
        description: "Inspect a target".to_string(),
        strict: true,
        defer_loading: None,
        parameters: codex_tools::parse_tool_input_schema_without_compaction(&schema)
            .expect("schema fixture"),
        local_result_schema: None,
    })];
    let coding = profile("kimi-for-coding", 32_768, KimiThinkingPolicy::Required);
    let dialect = dialect(coding, 262_144, 0, KimiThinking::Enabled);
    assert_eq!(
        request(&dialect, &[], &tools)["tools"],
        json!([{"type":"function","function":{
            "name":"inspect","description":"Inspect a target","strict":true,"parameters":{"type":"object","properties":{
                    "target":{"type":"string","enum":["a","b"],"description":"kept"},
                    "options":{"type":"object","properties":{"mode":{"type":"string","enum":["fast","safe"]}},"required":["mode"]},
                    "path":{"type":"string","pattern":"^src/","minLength":1},
                    "limit":{"type":"number","minimum":1,"maximum":100},
                    "tags":{"type":"array","items":{"type":"string","enum":["code"]},"minItems":1,"uniqueItems":true}
                },"required":["target"],"additionalProperties":false
            }
        }}])
    );
}

#[test]
fn rejects_unsupported_schema_forms_with_exact_errors() {
    assert_eq!(
        [
            normalize_schema(&json!({"properties":{"x":{"$ref":"https://example/schema"}}})),
            normalize_schema(&json!({"properties":{},"required":["missing"]})),
        ],
        [
            Err(SchemaError::UnsupportedReference(
                "https://example/schema".to_string()
            )),
            Err(SchemaError::MissingRequiredProperty("missing".to_string())),
        ]
    );
}
