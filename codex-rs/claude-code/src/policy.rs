use std::collections::BTreeMap;

use codex_protocol::model_inference::AnthropicThinkingPolicy;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::model_inference::ModelInferenceConfig;
use codex_protocol::model_inference::WireApi;
use codex_protocol::openai_models::ReasoningEffort;
use codex_tools::ToolSpec;
use serde::Serialize;
use thiserror::Error;
use uuid::Uuid;

use crate::CacheControl;
use crate::ContentBlock;
use crate::ContextEdit;
use crate::ContextKeep;
use crate::ContextManagement;
use crate::Message;
use crate::MessagesRequest;
use crate::OutputConfig;
use crate::OutputEffort;
use crate::Role;
use crate::SystemBlock;
use crate::Thinking;
use crate::ThinkingDisplay;
use crate::Tool;

const ANTHROPIC_BETAS: &str = "oauth-2025-04-20,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,claude-code-20250219,extended-cache-ttl-2025-04-11";
const CLAUDE_SESSION_NAMESPACE: Uuid = Uuid::from_u128(0x2d2e_1ea2_4fd0_5c0b_918d_17aa_1299_302d);

/// The complete native Messages request and the transport facts required to send it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AssembledRequest {
    pub transport: RequestTransport,
    pub body: MessagesRequest,
}

/// HTTP method, endpoint, query parameters, and headers for one Claude request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RequestTransport {
    pub method: &'static str,
    pub path: &'static str,
    pub query: BTreeMap<String, String>,
    pub headers: BTreeMap<String, String>,
}

/// Native inputs assembled into a Claude Messages request without history conversion.
#[derive(Debug)]
pub struct AssembleRequest<'a> {
    pub profile: &'a ModelInferenceConfig,
    pub effort: &'a ReasoningEffort,
    pub messages: &'a [Message],
    pub system: &'a [SystemBlock],
    pub tools: &'a [ToolSpec],
    pub resumable_session_id: &'a str,
    pub codex_version: &'a str,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum AssembleError {
    #[error("Claude Messages assembly requires an Anthropic model profile")]
    NotAnthropicModel,
    #[error("unsupported Anthropic route contract: {wire_api}/{dialect}")]
    UnsupportedRouteContract {
        wire_api: WireApi,
        dialect: InferenceDialect,
    },
    #[error("reasoning effort `{effort}` is unsupported by adaptive Claude thinking")]
    UnsupportedEffort { effort: String },
    #[error("reasoning effort `none` is unsupported by native model `{model}`")]
    DisabledThinkingUnsupported { model: String },
    #[error("resumable Codex session ID must be a UUID: {session_id}")]
    InvalidSessionId { session_id: String },
    #[error(
        "unsupported tool at index {index}: {kind} `{name}`; only JSON-schema function tools are supported"
    )]
    UnsupportedTool {
        index: usize,
        kind: &'static str,
        name: String,
    },
    #[error("tool `{name}` at index {index} contains an invalid JSON schema: {message}")]
    InvalidToolSchema {
        index: usize,
        name: String,
        message: String,
    },
    #[error("native message at index {index} has unsupported {role} block `{kind}")]
    UnsupportedNativeBlock {
        index: usize,
        role: &'static str,
        kind: &'static str,
    },
}

/// Assembles an exact Claude Code Messages request from already-native content.
pub fn assemble_request(params: AssembleRequest<'_>) -> Result<AssembledRequest, AssembleError> {
    let ModelInferenceConfig::Anthropic {
        wire_api,
        dialect,
        wire_model,
        max_output_tokens,
        thinking,
        supports_disabled_thinking,
        ..
    } = params.profile
    else {
        return Err(AssembleError::NotAnthropicModel);
    };
    if (*wire_api, *dialect) != (WireApi::AnthropicMessages, InferenceDialect::ClaudeCode) {
        return Err(AssembleError::UnsupportedRouteContract {
            wire_api: *wire_api,
            dialect: *dialect,
        });
    }

    let session_id = claude_session_id(params.resumable_session_id)?;
    let (thinking, output_config) = thinking_config(
        *thinking,
        *supports_disabled_thinking,
        params.effort,
        wire_model,
    )?;
    let mut system = params.system.to_vec();
    let mut messages = params.messages.to_vec();
    validate_messages(&messages)?;
    let mut tools = encode_tools(params.tools)?;
    apply_cache_policy(&mut system, &mut messages, &mut tools);
    let context_management = match &thinking {
        Thinking::Enabled { .. } | Thinking::Adaptive { .. } => Some(ContextManagement {
            edits: vec![ContextEdit::ClearThinking20251015 {
                keep: ContextKeep::All,
            }],
        }),
        Thinking::Disabled => None,
    };

    Ok(AssembledRequest {
        transport: request_transport(params.codex_version, session_id),
        body: MessagesRequest {
            model: wire_model.clone(),
            max_tokens: *max_output_tokens,
            stream: true,
            system,
            messages,
            tools,
            thinking,
            output_config,
            context_management,
            metadata: None,
        },
    })
}

fn claude_session_id(resumable_session_id: &str) -> Result<String, AssembleError> {
    let codex_session =
        Uuid::parse_str(resumable_session_id).map_err(|_| AssembleError::InvalidSessionId {
            session_id: resumable_session_id.to_string(),
        })?;
    Ok(Uuid::new_v5(&CLAUDE_SESSION_NAMESPACE, codex_session.as_bytes()).to_string())
}

fn request_transport(codex_version: &str, session_id: String) -> RequestTransport {
    RequestTransport {
        method: "POST",
        path: "/v1/messages",
        query: BTreeMap::from([("beta".to_string(), "true".to_string())]),
        headers: BTreeMap::from([
            (
                "User-Agent".to_string(),
                format!("codex-cli/{codex_version}"),
            ),
            ("anthropic-beta".to_string(), ANTHROPIC_BETAS.to_string()),
            ("anthropic-version".to_string(), "2023-06-01".to_string()),
            ("x-app".to_string(), "codex".to_string()),
            ("x-claude-code-session-id".to_string(), session_id),
        ]),
    }
}

fn thinking_config(
    policy: AnthropicThinkingPolicy,
    supports_disabled: bool,
    effort: &ReasoningEffort,
    model: &str,
) -> Result<(Thinking, Option<OutputConfig>), AssembleError> {
    if matches!(effort, ReasoningEffort::None) {
        return if supports_disabled {
            Ok((Thinking::Disabled, None))
        } else {
            Err(AssembleError::DisabledThinkingUnsupported {
                model: model.to_string(),
            })
        };
    }
    match policy {
        AnthropicThinkingPolicy::Budgeted { budget_tokens } => Ok((
            Thinking::Enabled {
                budget_tokens,
                display: ThinkingDisplay::Omitted,
            },
            None,
        )),
        AnthropicThinkingPolicy::Adaptive => {
            let effort = match effort {
                ReasoningEffort::Minimal | ReasoningEffort::Low => OutputEffort::Low,
                ReasoningEffort::Medium => OutputEffort::Medium,
                ReasoningEffort::High => OutputEffort::High,
                ReasoningEffort::XHigh => OutputEffort::Xhigh,
                ReasoningEffort::Max => OutputEffort::Max,
                ReasoningEffort::None => unreachable!(),
                ReasoningEffort::Ultra | ReasoningEffort::Custom(_) => {
                    return Err(AssembleError::UnsupportedEffort {
                        effort: effort.to_string(),
                    });
                }
            };
            Ok((
                Thinking::Adaptive {
                    display: ThinkingDisplay::Omitted,
                },
                Some(OutputConfig { effort }),
            ))
        }
    }
}

fn encode_tools(specs: &[ToolSpec]) -> Result<Vec<Tool>, AssembleError> {
    specs
        .iter()
        .enumerate()
        .map(|(index, spec)| match spec {
            ToolSpec::Function(tool) => serde_json::to_value(&tool.parameters)
                .map(|input_schema| Tool {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    input_schema,
                    cache_control: None,
                })
                .map_err(|error| AssembleError::InvalidToolSchema {
                    index,
                    name: tool.name.clone(),
                    message: error.to_string(),
                }),
            ToolSpec::Namespace(tool) => Err(AssembleError::UnsupportedTool {
                index,
                kind: "namespace",
                name: tool.name.clone(),
            }),
            ToolSpec::ToolSearch { .. } => Err(AssembleError::UnsupportedTool {
                index,
                kind: "hosted tool search",
                name: spec.name().to_string(),
            }),
            ToolSpec::WebSearch { .. } => Err(AssembleError::UnsupportedTool {
                index,
                kind: "hosted web search",
                name: spec.name().to_string(),
            }),
            ToolSpec::Freeform(tool) => Err(AssembleError::UnsupportedTool {
                index,
                kind: "freeform",
                name: tool.name.clone(),
            }),
        })
        .collect()
}

fn validate_messages(messages: &[Message]) -> Result<(), AssembleError> {
    for (index, message) in messages.iter().enumerate() {
        for block in &message.content {
            let valid = match message.role {
                Role::User => matches!(
                    block,
                    ContentBlock::Text { .. }
                        | ContentBlock::Image { .. }
                        | ContentBlock::ToolResult { .. }
                ),
                Role::Assistant => matches!(
                    block,
                    ContentBlock::Text { .. }
                        | ContentBlock::Thinking { .. }
                        | ContentBlock::RedactedThinking { .. }
                        | ContentBlock::ToolUse { .. }
                ),
            };
            if !valid {
                return Err(AssembleError::UnsupportedNativeBlock {
                    index,
                    role: match message.role {
                        Role::User => "user",
                        Role::Assistant => "assistant",
                    },
                    kind: content_block_kind(block),
                });
            }
        }
    }
    Ok(())
}

fn content_block_kind(block: &ContentBlock) -> &'static str {
    match block {
        ContentBlock::Text { .. } => "text",
        ContentBlock::Image { .. } => "image",
        ContentBlock::Thinking { .. } => "thinking",
        ContentBlock::RedactedThinking { .. } => "redacted thinking",
        ContentBlock::ToolUse { .. } => "tool use",
        ContentBlock::ToolResult { .. } => "tool result",
    }
}

/// Removes caller cache hints and selects the bounded Claude Code-compatible breakpoints.
fn apply_cache_policy(system: &mut [SystemBlock], messages: &mut [Message], tools: &mut [Tool]) {
    for block in system.iter_mut() {
        match block {
            SystemBlock::Text { cache_control, .. } => *cache_control = None,
        }
    }
    for message in messages.iter_mut() {
        for block in &mut message.content {
            match block {
                ContentBlock::Text { cache_control, .. }
                | ContentBlock::Image { cache_control, .. }
                | ContentBlock::ToolResult { cache_control, .. } => *cache_control = None,
                ContentBlock::Thinking { .. }
                | ContentBlock::RedactedThinking { .. }
                | ContentBlock::ToolUse { .. } => {}
            }
        }
    }
    for tool in tools {
        tool.cache_control = None;
    }

    if let Some(SystemBlock::Text { cache_control, .. }) = system.last_mut() {
        *cache_control = Some(one_hour_cache_control());
    }

    let Some(message) = messages
        .iter_mut()
        .rev()
        .find(|message| message.role == Role::User)
    else {
        return;
    };
    let Some(block) = message.content.iter_mut().rev().find(|block| {
        matches!(
            block,
            ContentBlock::Text { .. }
                | ContentBlock::Image { .. }
                | ContentBlock::ToolResult { .. }
        )
    }) else {
        return;
    };
    match block {
        ContentBlock::Text { cache_control, .. }
        | ContentBlock::Image { cache_control, .. }
        | ContentBlock::ToolResult { cache_control, .. } => {
            *cache_control = Some(one_hour_cache_control());
        }
        ContentBlock::Thinking { .. }
        | ContentBlock::RedactedThinking { .. }
        | ContentBlock::ToolUse { .. } => {}
    }
}

fn one_hour_cache_control() -> CacheControl {
    CacheControl::Ephemeral {
        ttl: crate::CacheTtl::OneHour,
    }
}
