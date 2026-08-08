use std::collections::BTreeMap;

use codex_protocol::model_inference::AnthropicThinkingPolicy;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::model_inference::ModelInferenceConfig;
use codex_protocol::model_inference::WireApi;
use codex_protocol::openai_models::ReasoningEffort;
use codex_tools::JsonSchema;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolSpec;
use pretty_assertions::assert_eq;
use serde_json::json;

use super::*;

const SESSION: &str = "019fbf00-0000-7000-8000-000000000043";
const THREAD: &str = "019fbf00-0000-7000-8000-000000000044";

fn profile() -> ModelInferenceConfig {
    ModelInferenceConfig::Anthropic {
        wire_api: WireApi::AnthropicMessages,
        dialect: InferenceDialect::ClaudeCode,
        route: "claude_code".to_string(),
        wire_model: "claude-sonnet-5".to_string(),
        max_output_tokens: 64_000,
        thinking: AnthropicThinkingPolicy::Adaptive,
        supports_disabled_thinking: true,
    }
}

fn context(kind: ClaudeCodeRequestKind) -> SonnetCompatibilityContext {
    SonnetCompatibilityContext {
        identity: ClaudeCodeIdentity {
            kind,
            session_id: SESSION.to_string(),
            thread_id: THREAD.to_string(),
        },
    }
}

fn request(context: Option<&SonnetCompatibilityContext>) -> AssembledRequest {
    let profile = profile();
    let system = [SystemBlock::Text {
        text: "Preserve this native system prompt.".to_string(),
        cache_control: None,
    }];
    let messages = [
        Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "Inspect the workspace.".to_string(),
                cache_control: None,
            }],
        },
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
                content: ToolResultContent::Text("contents".to_string()),
                is_error: false,
                cache_control: None,
            }],
        },
    ];
    let tools = [ToolSpec::Function(ResponsesApiTool {
        name: "read".to_string(),
        description: "Read a file".to_string(),
        strict: true,
        defer_loading: None,
        parameters: JsonSchema::object(
            BTreeMap::from([("path".to_string(), JsonSchema::string(None))]),
            Some(vec!["path".to_string()]),
            Some(false.into()),
        ),
        local_result_schema: None,
    })];
    assemble_request(AssembleRequest {
        profile: &profile,
        effort: &ReasoningEffort::High,
        messages: &messages,
        system: &system,
        tools: &tools,
        resumable_session_id: SESSION,
        codex_version: "0.146.0",
        opus_compatibility: None,
        sonnet_compatibility: context,
    })
    .expect("assemble Sonnet request")
}

#[test]
fn snapshots_complete_sonnet_root_and_subagent_requests() {
    let root = context(ClaudeCodeRequestKind::Root);
    let subagent = context(ClaudeCodeRequestKind::Subagent);
    insta::assert_json_snapshot!(
        "sonnet_5_root_and_subagent_requests",
        [request(Some(&root)), request(Some(&subagent))]
    );
}

#[test]
fn sonnet_compatibility_changes_only_transport_headers() {
    let root = context(ClaudeCodeRequestKind::Root);
    let subagent = context(ClaudeCodeRequestKind::Subagent);
    let native = request(None);
    let root = request(Some(&root));
    let subagent = request(Some(&subagent));

    assert_eq!(
        serde_json::to_vec(&root.body).expect("serialize root body"),
        serde_json::to_vec(&native.body).expect("serialize native body")
    );
    assert_eq!(
        serde_json::to_vec(&subagent.body).expect("serialize subagent body"),
        serde_json::to_vec(&native.body).expect("serialize native body")
    );
    assert!(
        !root
            .transport
            .headers
            .contains_key("x-claude-code-agent-id")
    );
    let agent_id = subagent
        .transport
        .headers
        .get("x-claude-code-agent-id")
        .expect("subagent agent ID");
    assert_eq!(agent_id.len(), 17);
    assert!(agent_id.starts_with('a'));
    assert!(agent_id[1..].bytes().all(|byte| byte.is_ascii_hexdigit()));
    assert!(agent_id[1..].bytes().all(|byte| !byte.is_ascii_uppercase()));
}
