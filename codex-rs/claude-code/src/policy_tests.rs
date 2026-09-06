use std::collections::BTreeMap;

use codex_tools::FreeformTool;
use codex_tools::FreeformToolFormat;
use codex_tools::JsonSchema;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolSpec;
use pretty_assertions::assert_eq;
use serde_json::json;

use super::*;

const SESSION: &str = "019fbf00-0000-7000-8000-000000000043";

fn profile(
    model: &str,
    max_output_tokens: u32,
    thinking: AnthropicThinkingPolicy,
    supports_disabled_thinking: bool,
) -> ModelInferenceConfig {
    ModelInferenceConfig::Anthropic {
        wire_api: WireApi::AnthropicMessages,
        dialect: InferenceDialect::ClaudeCode,
        route: "claude_code".to_string(),
        wire_model: model.to_string(),
        max_output_tokens,
        thinking,
        supports_disabled_thinking,
    }
}

fn adaptive(model: &str, supports_disabled_thinking: bool) -> ModelInferenceConfig {
    profile(
        model,
        /*max_output_tokens*/ 64_000,
        AnthropicThinkingPolicy::Adaptive,
        supports_disabled_thinking,
    )
}

fn system(text: &str, cache_control: Option<CacheControl>) -> SystemBlock {
    SystemBlock::Text {
        text: text.to_string(),
        cache_control,
    }
}

fn user_text(text: &str, cache_control: Option<CacheControl>) -> Message {
    Message {
        role: Role::User,
        content: vec![ContentBlock::Text {
            text: text.to_string(),
            cache_control,
        }],
    }
}

fn function_tool(name: &str) -> ToolSpec {
    ToolSpec::Function(ResponsesApiTool {
        name: name.to_string(),
        description: format!("Run {name}"),
        strict: true,
        defer_loading: None,
        parameters: JsonSchema::object(
            BTreeMap::from([("path".to_string(), JsonSchema::string(None))]),
            Some(vec!["path".to_string()]),
            Some(false.into()),
        ),
        output_schema: None,
    })
}

fn environment() -> ClaudeCodeEnvironment {
    ClaudeCodeEnvironment {
        cwd: "/workspace/project".to_string(),
        is_git_repository: true,
        platform: "linux".to_string(),
        architecture: "x86_64".to_string(),
        shell: "bash".to_string(),
        os_version: "Linux test".to_string(),
    }
}

fn assemble(
    profile: &ModelInferenceConfig,
    effort: &ReasoningEffort,
    messages: &[Message],
    system: &[SystemBlock],
    tools: &[ToolSpec],
) -> Result<AssembledRequest, AssembleError> {
    assemble_request(AssembleRequest {
        profile,
        effort,
        messages,
        system,
        tools,
        resumable_session_id: SESSION,
        codex_version: "0.153.4",
        opus_compatibility: None,
        sonnet_compatibility: None,
    })
}

fn cache_control_count(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::Array(values) => values.iter().map(cache_control_count).sum(),
        serde_json::Value::Object(values) => {
            usize::from(values.contains_key("cache_control"))
                + values.values().map(cache_control_count).sum::<usize>()
        }
        _ => 0,
    }
}

#[test]
fn assembles_haiku_policy_headers_and_bounded_cache_breakpoints() {
    let five_minutes = Some(CacheControl::Ephemeral {
        ttl: Some(CacheTtl::FiveMinutes),
    });
    let systems = (0..6)
        .map(|index| system(&format!("system {index}"), five_minutes))
        .collect::<Vec<_>>();
    let messages = vec![
        user_text("old", five_minutes),
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text {
                text: "answer".to_string(),
                cache_control: five_minutes,
            }],
        },
        user_text("latest", five_minutes),
    ];
    let tools = vec![function_tool("read"), function_tool("write")];
    let request = assemble(
        &profile(
            "claude-haiku-4-5-20251001",
            /*max_output_tokens*/ 32_000,
            AnthropicThinkingPolicy::Budgeted {
                budget_tokens: 31_999,
            },
            /*supports_disabled_thinking*/ true,
        ),
        &ReasoningEffort::High,
        &messages,
        &systems,
        &tools,
    )
    .expect("assemble Haiku request");

    assert_eq!(request.transport.method, "POST");
    assert_eq!(request.transport.path, "/v1/messages");
    assert_eq!(
        request.transport.query.get("beta").map(String::as_str),
        Some("true")
    );
    assert_eq!(
        request
            .transport
            .headers
            .get("User-Agent")
            .map(String::as_str),
        Some("codex-cli/0.153.4")
    );
    assert_eq!(
        request.transport.headers.get("x-app").map(String::as_str),
        Some("codex")
    );
    assert_eq!(request.body.max_tokens, 32_000);
    assert_eq!(
        request.body.thinking,
        Thinking::Enabled {
            budget_tokens: 31_999,
            display: ThinkingDisplay::Omitted,
        }
    );
    assert_eq!(request.body.output_config, None);
    assert_eq!(
        request.body.tools[0].input_schema,
        json!({
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": false
        })
    );
    assert_eq!(
        request
            .body
            .tools
            .iter()
            .map(|tool| tool.cache_control)
            .collect::<Vec<_>>(),
        vec![None, None]
    );
    assert_eq!(
        cache_control_count(&serde_json::to_value(&request.body).unwrap()),
        2
    );
    assert_eq!(
        serde_json::to_value(&request.body.system).unwrap(),
        json!([
            {"type":"text","text":"system 0"},
            {"type":"text","text":"system 1"},
            {"type":"text","text":"system 2"},
            {"type":"text","text":"system 3"},
            {"type":"text","text":"system 4"},
            {"type":"text","text":"system 5","cache_control":{"type":"ephemeral","ttl":"1h"}}
        ])
    );
}

#[test]
fn thinking_policy_maps_all_supported_efforts_and_omits_disabled_context_management() {
    let model = adaptive("claude-opus-4-8", /*supports_disabled_thinking*/ true);
    for (effort, output_effort) in [
        (ReasoningEffort::Minimal, OutputEffort::Low),
        (ReasoningEffort::Low, OutputEffort::Low),
        (ReasoningEffort::Medium, OutputEffort::Medium),
        (ReasoningEffort::High, OutputEffort::High),
        (ReasoningEffort::XHigh, OutputEffort::Xhigh),
        (ReasoningEffort::Max, OutputEffort::Max),
    ] {
        let request = assemble(&model, &effort, &[], &[], &[]).unwrap();
        assert_eq!(
            (request.body.thinking, request.body.output_config),
            (
                Thinking::Adaptive {
                    display: Some(ThinkingDisplay::Omitted),
                },
                Some(OutputConfig {
                    effort: output_effort,
                }),
            )
        );
        assert!(request.body.context_management.is_some());
    }

    let disabled = assemble(&model, &ReasoningEffort::None, &[], &[], &[]).unwrap();
    assert_eq!(disabled.body.thinking, Thinking::Disabled);
    assert_eq!(disabled.body.output_config, None);
    assert_eq!(disabled.body.context_management, None);

    assert_eq!(
        assemble(
            &adaptive("claude-fable-5", /*supports_disabled_thinking*/ false,),
            &ReasoningEffort::None,
            &[],
            &[],
            &[],
        ),
        Err(AssembleError::DisabledThinkingUnsupported {
            model: "claude-fable-5".to_string(),
        })
    );
}

#[test]
fn compatibility_profiles_preserve_effort_and_exact_identity_shapes() {
    let tools = [function_tool("read")];
    let messages = [user_text("inspect", /*cache_control*/ None)];
    let opus = OpusCompatibilityContext {
        kind: OpusRequestKind::Subagent,
        session_id: SESSION.to_string(),
        thread_id: "019fbf00-0000-7000-8000-000000000044".to_string(),
        installation_id: "019fbf00-0000-7000-8000-000000000045".to_string(),
        environment: environment(),
    };
    let opus_request = assemble_request(AssembleRequest {
        profile: &adaptive("claude-opus-5", /*supports_disabled_thinking*/ true),
        effort: &ReasoningEffort::Max,
        messages: &messages,
        system: &[system("replaced", /*cache_control*/ None)],
        tools: &tools,
        resumable_session_id: SESSION,
        codex_version: "ignored",
        opus_compatibility: Some(&opus),
        sonnet_compatibility: None,
    })
    .unwrap();
    assert_eq!(opus_request.body.max_tokens, 64_000);
    assert_eq!(
        opus_request.body.thinking,
        Thinking::Adaptive { display: None }
    );
    assert_eq!(
        opus_request.body.output_config,
        Some(OutputConfig {
            effort: OutputEffort::Max
        })
    );
    assert_eq!(
        opus_request
            .transport
            .headers
            .get("User-Agent")
            .map(String::as_str),
        Some("claude-cli/2.1.224 (external, cli)")
    );
    assert_eq!(
        opus_request
            .transport
            .headers
            .get("x-app")
            .map(String::as_str),
        Some("cli")
    );
    assert_eq!(
        opus_request
            .transport
            .headers
            .get("x-claude-code-session-id")
            .map(String::as_str),
        Some(SESSION)
    );
    assert!(
        opus_request
            .transport
            .headers
            .get("x-claude-code-agent-id")
            .is_some_and(|value| value.len() == 17 && value.starts_with('a'))
    );
    assert_eq!(
        cache_control_count(&serde_json::to_value(&opus_request.body).unwrap()),
        3
    );
    assert!(
        matches!(&opus_request.body.system[0], SystemBlock::Text { text, cache_control: None } if text.contains("cc_is_subagent=true"))
    );

    let sonnet = SonnetCompatibilityContext {
        identity: ClaudeCodeIdentity {
            kind: ClaudeCodeRequestKind::Root,
            session_id: SESSION.to_string(),
            thread_id: "019fbf00-0000-7000-8000-000000000046".to_string(),
        },
        installation_id: "019fbf00-0000-7000-8000-000000000045".to_string(),
        environment: environment(),
    };
    let sonnet_request = assemble_request(AssembleRequest {
        profile: &adaptive("claude-sonnet-5", /*supports_disabled_thinking*/ true),
        effort: &ReasoningEffort::Low,
        messages: &messages,
        system: &[],
        tools: &tools,
        resumable_session_id: SESSION,
        codex_version: "ignored",
        opus_compatibility: None,
        sonnet_compatibility: Some(&sonnet),
    })
    .unwrap();
    assert_eq!(
        sonnet_request.body.output_config,
        Some(OutputConfig {
            effort: OutputEffort::Low
        })
    );
    assert_eq!(
        sonnet_request
            .transport
            .headers
            .get("User-Agent")
            .map(String::as_str),
        Some("claude-cli/2.1.223 (external, cli)")
    );
    assert!(
        !sonnet_request
            .transport
            .headers
            .contains_key("x-claude-code-agent-id")
    );
    assert!(
        matches!(&sonnet_request.body.system[1], SystemBlock::Text { text, .. } if text == "You are Claude Code, Anthropic's official CLI for Claude.")
    );
}

#[test]
fn rejects_incompatible_routes_unsupported_tools_and_invalid_native_blocks() {
    let non_anthropic = ModelInferenceConfig::OpenAi {
        wire_api: WireApi::Responses,
        dialect: InferenceDialect::OpenAi,
        route: "openai".to_string(),
        wire_model: "gpt-test".to_string(),
    };
    assert_eq!(
        assemble(&non_anthropic, &ReasoningEffort::Medium, &[], &[], &[]),
        Err(AssembleError::NotAnthropicModel)
    );
    let incompatible = ModelInferenceConfig::Anthropic {
        wire_api: WireApi::Responses,
        dialect: InferenceDialect::ClaudeCode,
        route: "claude_code".to_string(),
        wire_model: "claude-opus-4-8".to_string(),
        max_output_tokens: 64_000,
        thinking: AnthropicThinkingPolicy::Adaptive,
        supports_disabled_thinking: true,
    };
    assert_eq!(
        assemble(&incompatible, &ReasoningEffort::Medium, &[], &[], &[]),
        Err(AssembleError::UnsupportedRouteContract {
            wire_api: WireApi::Responses,
            dialect: InferenceDialect::ClaudeCode,
        })
    );
    assert_eq!(
        assemble(
            &adaptive("claude-opus-4-8", /*supports_disabled_thinking*/ true,),
            &ReasoningEffort::Medium,
            &[],
            &[],
            &[ToolSpec::Freeform(FreeformTool {
                name: "patch".to_string(),
                description: "Apply a patch".to_string(),
                defer_loading: None,
                format: FreeformToolFormat {
                    r#type: "grammar".to_string(),
                    syntax: "lark".to_string(),
                    definition: "start: /.+/".to_string(),
                },
            })],
        ),
        Err(AssembleError::UnsupportedTool {
            index: 0,
            kind: "freeform",
            name: "patch".to_string(),
        })
    );
    let invalid = [Message {
        role: Role::User,
        content: vec![ContentBlock::ToolUse {
            id: "call".to_string(),
            name: "read".to_string(),
            input: json!({}),
        }],
    }];
    assert_eq!(
        assemble(
            &adaptive("claude-opus-4-8", /*supports_disabled_thinking*/ true,),
            &ReasoningEffort::Medium,
            &invalid,
            &[],
            &[],
        ),
        Err(AssembleError::UnsupportedNativeBlock {
            index: 0,
            role: "user",
            kind: "tool use",
        })
    );
}
