use std::collections::BTreeMap;

use codex_protocol::model_inference::AnthropicThinkingPolicy;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::model_inference::ModelInferenceConfig;
use codex_protocol::model_inference::WireApi;
use codex_protocol::openai_models::ReasoningEffort;
use codex_tools::FreeformTool;
use codex_tools::FreeformToolFormat;
use codex_tools::JsonSchema;
use codex_tools::ResponsesApiNamespace;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolSpec;
use pretty_assertions::assert_eq;
use serde_json::json;

use super::*;

const SESSION_A: &str = "019fbf00-0000-7000-8000-000000000043";
const SESSION_B: &str = "019fbf00-0000-7000-8000-000000000044";

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

fn haiku() -> ModelInferenceConfig {
    profile(
        "claude-haiku-4-5-20251001",
        32_000,
        AnthropicThinkingPolicy::Budgeted {
            budget_tokens: 31_999,
        },
        true,
    )
}

fn adaptive(model: &str, supports_disabled: bool) -> ModelInferenceConfig {
    profile(
        model,
        64_000,
        AnthropicThinkingPolicy::Adaptive,
        supports_disabled,
    )
}

fn system(text: &str) -> SystemBlock {
    SystemBlock::Text {
        text: text.to_string(),
        cache_control: None,
    }
}

fn cached_system(text: &str, ttl: CacheTtl) -> SystemBlock {
    SystemBlock::Text {
        text: text.to_string(),
        cache_control: Some(CacheControl::Ephemeral { ttl: Some(ttl) }),
    }
}

fn user_text(text: &str) -> Message {
    Message {
        role: Role::User,
        content: vec![ContentBlock::Text {
            text: text.to_string(),
            cache_control: None,
        }],
    }
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

fn one_hour_cache_control() -> CacheControl {
    CacheControl::Ephemeral {
        ttl: Some(CacheTtl::OneHour),
    }
}

fn function_tool(name: &str) -> ToolSpec {
    ToolSpec::Function(ResponsesApiTool {
        name: name.to_string(),
        description: format!("Run {name}"),
        strict: true,
        defer_loading: None,
        parameters: JsonSchema::object(
            BTreeMap::from([(
                "path".to_string(),
                JsonSchema::string(Some("File path".to_string())),
            )]),
            Some(vec!["path".to_string()]),
            Some(false.into()),
        ),
        local_result_schema: None,
    })
}

fn opus_context(kind: OpusRequestKind) -> OpusCompatibilityContext {
    OpusCompatibilityContext {
        kind,
        session_id: SESSION_A.to_string(),
        thread_id: "019fbf00-0000-7000-8000-000000000045".to_string(),
        installation_id: "019fbf00-0000-7000-8000-000000000046".to_string(),
        environment: OpusEnvironment {
            cwd: "/workspace/project".to_string(),
            is_git_repository: true,
            platform: "linux".to_string(),
            architecture: "x86_64".to_string(),
            shell: "bash".to_string(),
            os_version: "Linux 6.17.0-test".to_string(),
        },
    }
}

fn assemble(
    profile: &ModelInferenceConfig,
    effort: &ReasoningEffort,
    messages: &[Message],
    system: &[SystemBlock],
    tools: &[ToolSpec],
) -> Result<AssembledRequest, AssembleError> {
    let opus_compatibility = match profile {
        ModelInferenceConfig::Anthropic { wire_model, .. } if wire_model == "claude-opus-5" => {
            Some(opus_context(OpusRequestKind::Root))
        }
        ModelInferenceConfig::Anthropic { .. }
        | ModelInferenceConfig::OpenAi { .. }
        | ModelInferenceConfig::Kimi(_) => None,
    };
    assemble_request(AssembleRequest {
        profile,
        effort,
        messages,
        system,
        tools,
        resumable_session_id: SESSION_A,
        codex_version: "0.146.0",
        opus_compatibility: opus_compatibility.as_ref(),
        sonnet_compatibility: None,
    })
}

#[test]
fn snapshots_complete_opus_root_and_subagent_requests() {
    let profile = adaptive("claude-opus-5", true);
    let messages = [
        user_text("inspect the workspace"),
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call-1".to_string(),
                name: "read".to_string(),
                input: json!({"path": "src/lib.rs"}),
            }],
        },
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "call-1".to_string(),
                content: ToolResultContent::Text("file contents".to_string()),
                is_error: false,
                cache_control: None,
            }],
        },
    ];
    let tools = [function_tool("read")];
    let requests = [OpusRequestKind::Root, OpusRequestKind::Subagent]
        .into_iter()
        .map(|kind| {
            let context = opus_context(kind);
            assemble_request(AssembleRequest {
                profile: &profile,
                effort: &ReasoningEffort::Max,
                messages: &messages,
                system: &[system("Codex instructions must be replaced")],
                tools: &tools,
                resumable_session_id: SESSION_B,
                codex_version: "0.146.0",
                opus_compatibility: Some(&context),
                sonnet_compatibility: None,
            })
            .expect("assemble Opus compatibility request")
        })
        .collect::<Vec<_>>();

    insta::assert_json_snapshot!("opus_5_root_and_subagent_requests", requests);
}

#[test]
fn opus_prompts_render_non_git_turn_environment() {
    let mut context = opus_context(OpusRequestKind::Root);
    context.environment.cwd = "/workspace/no-repo".to_string();
    context.environment.is_git_repository = false;
    let root_system = context.system();
    let SystemBlock::Text {
        text: root_prompt, ..
    } = &root_system[2];
    assert!(root_prompt.contains("Primary working directory: /workspace/no-repo"));
    assert!(root_prompt.contains("Is a git repository: false"));

    context.kind = OpusRequestKind::Subagent;
    let subagent_system = context.system();
    let SystemBlock::Text {
        text: subagent_prompt,
        ..
    } = &subagent_system[2];
    assert!(subagent_prompt.contains("Working directory: /workspace/no-repo"));
    assert!(subagent_prompt.contains("Is directory a git repo: No"));
}

#[test]
fn snapshots_complete_non_opus_request_regression() {
    let messages = [user_text("preserve the native request")];
    let system = [system("You are Codex."), system("Follow the user.")];
    let tools = [function_tool("read")];
    let cases = [
        (
            "sonnet",
            adaptive("claude-sonnet-5", true),
            ReasoningEffort::High,
        ),
        (
            "fable",
            adaptive("claude-fable-5", false),
            ReasoningEffort::High,
        ),
        ("haiku", haiku(), ReasoningEffort::High),
        (
            "opus_4_8",
            adaptive("claude-opus-4-8", true),
            ReasoningEffort::High,
        ),
    ];
    let requests = cases
        .iter()
        .map(|(name, profile, effort)| {
            (
                *name,
                assemble(profile, effort, &messages, &system, &tools)
                    .expect("assemble non-Opus regression request"),
            )
        })
        .collect::<Vec<_>>();

    insta::assert_json_snapshot!("non_opus_request_regression", requests);
}

#[test]
fn snapshots_all_thinking_policies_and_transport_metadata() {
    let messages = [user_text("hello")];
    let cached_system_blocks = [
        system("You are Codex, an AI coding agent."),
        system("Follow the user."),
    ];
    let cases = [
        ("haiku_enabled", haiku(), ReasoningEffort::Max),
        ("haiku_disabled", haiku(), ReasoningEffort::None),
        (
            "adaptive_minimal",
            adaptive("claude-sonnet-5", true),
            ReasoningEffort::Minimal,
        ),
        (
            "adaptive_low",
            adaptive("claude-opus-4-8", true),
            ReasoningEffort::Low,
        ),
        (
            "adaptive_medium",
            adaptive("claude-sonnet-5", true),
            ReasoningEffort::Medium,
        ),
        (
            "adaptive_high",
            adaptive("claude-fable-5", false),
            ReasoningEffort::High,
        ),
        (
            "adaptive_xhigh",
            adaptive("claude-sonnet-5", true),
            ReasoningEffort::XHigh,
        ),
        (
            "adaptive_max",
            adaptive("claude-sonnet-5", true),
            ReasoningEffort::Max,
        ),
        (
            "adaptive_disabled",
            adaptive("claude-sonnet-5", true),
            ReasoningEffort::None,
        ),
    ];
    let requests = cases
        .iter()
        .map(|(name, profile, effort)| {
            (
                *name,
                assemble(profile, effort, &messages, &cached_system_blocks, &[])
                    .expect("supported policy"),
            )
        })
        .collect::<Vec<_>>();

    insta::assert_json_snapshot!("thinking_policies_and_metadata", requests);
}

#[test]
fn snapshots_cache_placement_and_complete_function_schemas() {
    let messages = [
        user_text("initial user message"),
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call-1".to_string(),
                name: "read".to_string(),
                input: json!({"path": "a"}),
            }],
        },
        Message {
            role: Role::User,
            content: vec![
                ContentBlock::ToolResult {
                    tool_use_id: "call-1".to_string(),
                    content: ToolResultContent::Text("contents".to_string()),
                    is_error: false,
                    cache_control: None,
                },
                ContentBlock::Text {
                    text: "continue".to_string(),
                    cache_control: None,
                },
            ],
        },
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text {
                text: "trailing assistant content".to_string(),
                cache_control: None,
            }],
        },
    ];
    let request = assemble(
        &adaptive("claude-sonnet-5", true),
        &ReasoningEffort::High,
        &messages,
        &[system("first"), system("second")],
        &[function_tool("read"), function_tool("write")],
    )
    .expect("assemble native request");

    insta::assert_json_snapshot!("cache_placement_and_tools", request);
}

#[test]
fn function_local_result_schema_is_not_sent_to_claude() {
    let ToolSpec::Function(mut function) = function_tool("read") else {
        unreachable!();
    };
    function.local_result_schema = Some(json!({
        "type": "object",
        "properties": {"contents": {"type": "string"}},
        "required": ["contents"]
    }));

    let request = assemble(
        &adaptive("claude-sonnet-5", true),
        &ReasoningEffort::High,
        &[],
        &[system("system")],
        &[ToolSpec::Function(function)],
    )
    .expect("local result metadata is not a Claude capability");

    assert_eq!(
        request.body.tools,
        vec![Tool {
            name: "read".to_string(),
            description: "Run read".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path"
                    }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
            cache_control: None,
        }]
    );
}

#[test]
fn cache_placement_handles_no_tools_and_trailing_ineligible_user_content() {
    let profile = adaptive("claude-sonnet-5", true);
    let no_tools = assemble(
        &profile,
        &ReasoningEffort::High,
        &[user_text("last user")],
        &[system("system")],
        &[],
    )
    .expect("no tools are supported");
    assert!(no_tools.body.tools.is_empty());
    assert_eq!(
        no_tools.body.messages[0].content,
        vec![ContentBlock::Text {
            text: "last user".to_string(),
            cache_control: Some(CacheControl::Ephemeral {
                ttl: Some(CacheTtl::OneHour),
            }),
        }]
    );

    let invalid = [Message {
        role: Role::User,
        content: vec![ContentBlock::ToolUse {
            id: "call-1".to_string(),
            name: "read".to_string(),
            input: json!({}),
        }],
    }];
    assert!(matches!(
        assemble(
            &profile,
            &ReasoningEffort::High,
            &invalid,
            &[system("system")],
            &[]
        ),
        Err(AssembleError::UnsupportedNativeBlock { .. })
    ));
}

#[test]
fn cache_policy_is_bounded_across_system_tool_and_user_shapes() {
    let profile = adaptive("claude-sonnet-5", true);
    for system_count in [0, 1, 2, 4, 9] {
        for tool_count in [0, 3] {
            for has_eligible_user_block in [true, false] {
                let system = (0..system_count)
                    .map(|index| system(&format!("system {index}")))
                    .collect::<Vec<_>>();
                let tools = (0..tool_count)
                    .map(|index| function_tool(&format!("tool_{index}")))
                    .collect::<Vec<_>>();
                let messages = if has_eligible_user_block {
                    vec![user_text("latest user")]
                } else {
                    vec![Message {
                        role: Role::User,
                        content: vec![],
                    }]
                };

                let request =
                    assemble(&profile, &ReasoningEffort::High, &messages, &system, &tools)
                        .expect("assemble native request");
                let value = serde_json::to_value(&request).expect("serialize request");

                assert!(cache_control_count(&value) <= 4);
                assert_eq!(
                    cache_control_count(&value),
                    usize::from(system_count > 0) + usize::from(has_eligible_user_block)
                );
                assert!(
                    request
                        .body
                        .tools
                        .iter()
                        .all(|tool| tool.cache_control.is_none())
                );
                assert_eq!(
                    request.body.system.last().and_then(|block| match block {
                        SystemBlock::Text { cache_control, .. } => *cache_control,
                    }),
                    (system_count > 0).then(one_hour_cache_control)
                );
            }
        }
    }
}

#[test]
fn cache_policy_normalizes_caller_markers_without_changing_request_content() {
    let messages = [
        Message {
            role: Role::User,
            content: vec![
                ContentBlock::Text {
                    text: "older user".to_string(),
                    cache_control: Some(CacheControl::Ephemeral {
                        ttl: Some(CacheTtl::FiveMinutes),
                    }),
                },
                ContentBlock::Image {
                    source: ImageSource::Base64 {
                        media_type: "image/png".to_string(),
                        data: "aGVsbG8=".to_string(),
                    },
                    cache_control: Some(CacheControl::Ephemeral {
                        ttl: Some(CacheTtl::OneHour),
                    }),
                },
            ],
        },
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Thinking {
                thinking: "replayed thought".to_string(),
                signature: "sig".to_string(),
            }],
        },
        Message {
            role: Role::User,
            content: vec![
                ContentBlock::ToolResult {
                    tool_use_id: "call-1".to_string(),
                    content: ToolResultContent::Text("result".to_string()),
                    is_error: false,
                    cache_control: Some(CacheControl::Ephemeral {
                        ttl: Some(CacheTtl::FiveMinutes),
                    }),
                },
                ContentBlock::Text {
                    text: "latest user".to_string(),
                    cache_control: Some(CacheControl::Ephemeral {
                        ttl: Some(CacheTtl::OneHour),
                    }),
                },
            ],
        },
    ];
    let cached_system_blocks = [
        cached_system("first", CacheTtl::FiveMinutes),
        cached_system("second", CacheTtl::OneHour),
        cached_system("third", CacheTtl::FiveMinutes),
        cached_system("final", CacheTtl::OneHour),
    ];
    let expected_messages = [
        Message {
            role: Role::User,
            content: vec![
                ContentBlock::Text {
                    text: "older user".to_string(),
                    cache_control: None,
                },
                ContentBlock::Image {
                    source: ImageSource::Base64 {
                        media_type: "image/png".to_string(),
                        data: "aGVsbG8=".to_string(),
                    },
                    cache_control: None,
                },
            ],
        },
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Thinking {
                thinking: "replayed thought".to_string(),
                signature: "sig".to_string(),
            }],
        },
        Message {
            role: Role::User,
            content: vec![
                ContentBlock::ToolResult {
                    tool_use_id: "call-1".to_string(),
                    content: ToolResultContent::Text("result".to_string()),
                    is_error: false,
                    cache_control: None,
                },
                ContentBlock::Text {
                    text: "latest user".to_string(),
                    cache_control: None,
                },
            ],
        },
    ];
    let expected_system = [
        system("first"),
        system("second"),
        system("third"),
        system("final"),
    ];
    let profile = adaptive("claude-sonnet-5", true);
    let tools = [function_tool("read"), function_tool("write")];

    let request = assemble(
        &profile,
        &ReasoningEffort::High,
        &messages,
        &cached_system_blocks,
        &tools,
    )
    .expect("assemble cached input");
    let expected = assemble(
        &profile,
        &ReasoningEffort::High,
        &expected_messages,
        &expected_system,
        &tools,
    )
    .expect("assemble uncached input");

    assert_eq!(request, expected);
    assert_eq!(
        cache_control_count(&serde_json::to_value(&request).expect("serialize request")),
        2
    );
    assert_eq!(
        request.body.system,
        vec![
            system("first"),
            system("second"),
            system("third"),
            SystemBlock::Text {
                text: "final".to_string(),
                cache_control: Some(one_hour_cache_control()),
            },
        ]
    );
    assert_eq!(
        request.body.messages[2].content,
        vec![
            ContentBlock::ToolResult {
                tool_use_id: "call-1".to_string(),
                content: ToolResultContent::Text("result".to_string()),
                is_error: false,
                cache_control: None,
            },
            ContentBlock::Text {
                text: "latest user".to_string(),
                cache_control: Some(one_hour_cache_control()),
            },
        ]
    );
}

#[test]
fn session_metadata_is_stable_and_identity_free() {
    let profile = adaptive("claude-sonnet-5", true);
    let effort = ReasoningEffort::High;
    let messages = [user_text("hello")];
    let system = [system("system")];
    let params = |session_id| AssembleRequest {
        profile: &profile,
        effort: &effort,
        messages: &messages,
        system: &system,
        tools: &[],
        resumable_session_id: session_id,
        codex_version: "0.146.0",
        opus_compatibility: None,
        sonnet_compatibility: None,
    };
    let first = assemble_request(params(SESSION_A)).expect("first session");
    let resumed = assemble_request(params(SESSION_A)).expect("resumed session");
    let distinct = assemble_request(params(SESSION_B)).expect("distinct session");
    let session = &first.transport.headers["x-claude-code-session-id"];

    assert_eq!(
        session,
        &resumed.transport.headers["x-claude-code-session-id"]
    );
    assert_ne!(
        session,
        &distinct.transport.headers["x-claude-code-session-id"]
    );
    assert_ne!(session, SESSION_A);
    assert!(
        !first
            .transport
            .headers
            .contains_key("x-anthropic-billing-header")
    );
    assert!(
        !first
            .transport
            .headers
            .values()
            .any(|value| value == "claude-cli")
    );
    assert!(first.body.metadata.is_none());
}

#[test]
fn unsupported_tools_and_efforts_fail_before_assembling_a_request() {
    let fable = adaptive("claude-fable-5", false);
    assert!(matches!(
        assemble(
            &fable,
            &ReasoningEffort::None,
            &[],
            &[system("system")],
            &[]
        ),
        Err(AssembleError::DisabledThinkingUnsupported { .. })
    ));
    let sonnet = adaptive("claude-sonnet-5", true);
    assert!(matches!(
        assemble(
            &sonnet,
            &ReasoningEffort::Ultra,
            &[],
            &[system("system")],
            &[]
        ),
        Err(AssembleError::UnsupportedEffort { .. })
    ));

    let unsupported_tools = [
        (
            ToolSpec::Freeform(FreeformTool {
                name: "free".to_string(),
                description: "freeform".to_string(),
                format: FreeformToolFormat {
                    r#type: "grammar".to_string(),
                    syntax: "lark".to_string(),
                    definition: "start: /.+/".to_string(),
                },
            }),
            "freeform",
            "free",
        ),
        (
            ToolSpec::Namespace(ResponsesApiNamespace {
                name: "apps".to_string(),
                description: "apps".to_string(),
                tools: vec![],
            }),
            "namespace",
            "apps",
        ),
        (
            ToolSpec::ToolSearch {
                execution: "server".to_string(),
                description: "search".to_string(),
                parameters: JsonSchema::default(),
            },
            "hosted tool search",
            "tool_search",
        ),
        (
            ToolSpec::WebSearch {
                external_web_access: None,
                indexed_web_access: None,
                filters: None,
                user_location: None,
                search_context_size: None,
                search_content_types: None,
            },
            "hosted web search",
            "web_search",
        ),
    ];
    for (tool, kind, name) in unsupported_tools {
        assert_eq!(
            assemble(
                &sonnet,
                &ReasoningEffort::High,
                &[],
                &[system("system")],
                &[tool]
            ),
            Err(AssembleError::UnsupportedTool {
                index: 0,
                kind,
                name: name.to_string(),
            })
        );
    }
}
