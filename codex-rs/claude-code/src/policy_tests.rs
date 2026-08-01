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

fn user_text(text: &str) -> Message {
    Message {
        role: Role::User,
        content: vec![ContentBlock::Text {
            text: text.to_string(),
            cache_control: None,
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
            BTreeMap::from([(
                "path".to_string(),
                JsonSchema::string(Some("File path".to_string())),
            )]),
            Some(vec!["path".to_string()]),
            Some(false.into()),
        ),
        output_schema: None,
    })
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
        resumable_session_id: SESSION_A,
        codex_version: "0.146.0",
    })
}

#[test]
fn snapshots_all_thinking_policies_and_transport_metadata() {
    let messages = [user_text("hello")];
    let system = [
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
            adaptive("claude-opus-5", true),
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
            adaptive("claude-opus-5", true),
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
                assemble(profile, effort, &messages, &system, &[]).expect("supported policy"),
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
                ttl: CacheTtl::OneHour,
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
        ToolSpec::Freeform(FreeformTool {
            name: "free".to_string(),
            description: "freeform".to_string(),
            format: FreeformToolFormat {
                r#type: "grammar".to_string(),
                syntax: "lark".to_string(),
                definition: "start: /.+/".to_string(),
            },
        }),
        ToolSpec::Namespace(ResponsesApiNamespace {
            name: "apps".to_string(),
            description: "apps".to_string(),
            tools: vec![],
        }),
        ToolSpec::ToolSearch {
            execution: "server".to_string(),
            description: "search".to_string(),
            parameters: JsonSchema::default(),
        },
        ToolSpec::WebSearch {
            external_web_access: None,
            indexed_web_access: None,
            filters: None,
            user_location: None,
            search_context_size: None,
            search_content_types: None,
        },
        ToolSpec::Function(ResponsesApiTool {
            output_schema: Some(json!({"type": "object"})),
            ..match function_tool("structured") {
                ToolSpec::Function(tool) => tool,
                _ => unreachable!(),
            }
        }),
    ];
    for tool in unsupported_tools {
        assert!(matches!(
            assemble(
                &sonnet,
                &ReasoningEffort::High,
                &[],
                &[system("system")],
                &[tool]
            ),
            Err(AssembleError::UnsupportedTool { .. })
        ));
    }
}
