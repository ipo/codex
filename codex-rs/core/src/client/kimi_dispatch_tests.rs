use std::collections::BTreeMap;
use std::time::Duration;

use codex_protocol::models::AgentMessageInputContent;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ImageDetail;
use codex_tools::JsonSchema;
use codex_tools::ResponsesApiTool;
use pretty_assertions::assert_eq;
use serde_json::json;

use super::*;

fn message(role: &str, content: Vec<ContentItem>) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: role.to_string(),
        content,
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }
}

fn text_message(role: &str, text: &str) -> ResponseItem {
    let content = if role == "assistant" {
        ContentItem::OutputText {
            text: text.to_string(),
        }
    } else {
        ContentItem::InputText {
            text: text.to_string(),
        }
    };
    message(role, vec![content])
}

fn prompt(input: Vec<ResponseItem>) -> Prompt {
    Prompt {
        input,
        base_instructions: BaseInstructions {
            text: "base instructions".to_string(),
            provenance: None,
        },
        ..Default::default()
    }
}

fn reasoning(marker: &str, text: &str) -> ResponseItem {
    ResponseItem::Reasoning {
        id: None,
        summary: Vec::new(),
        content: Some(vec![ReasoningItemContent::ReasoningText {
            text: text.to_string(),
        }]),
        encrypted_content: Some(marker.to_string()),
        internal_chat_message_metadata_passthrough: None,
    }
}

fn kimi_route() -> ResolvedWireRoute {
    ResolvedWireRoute {
        name: Some("kimi_code".to_string()),
        wire_api: WireApi::ChatCompletions,
        dialect: InferenceDialect::Kimi,
        base_url: Some("http://127.0.0.1:8080/v1/kimi".to_string()),
        request_path: "chat/completions".to_string(),
        query_params: None,
        request_max_retries: 4,
        stream_max_retries: 10,
        stream_idle_timeout: Duration::from_secs(300),
    }
}

#[test]
fn projects_ordinary_history_in_exact_kimi_order() {
    let marker = KimiReasoning::for_model(
        "k3",
        KimiReasoningKey::ReasoningDetails,
        "private reasoning".to_string(),
    )
    .expect("valid Kimi marker")
    .opaque_marker()
    .to_string();
    let projected = project_messages(
        &prompt(vec![
            text_message("developer", "developer guidance"),
            message(
                "user",
                vec![
                    ContentItem::InputText {
                        text: "inspect this".to_string(),
                    },
                    ContentItem::InputImage {
                        image_url: "https://example.test/image.png".to_string(),
                        detail: Some(ImageDetail::High),
                    },
                ],
            ),
            reasoning(&marker, "private reasoning"),
            text_message("assistant", "working"),
            ResponseItem::FunctionCall {
                id: None,
                name: "inspect".to_string(),
                namespace: None,
                arguments: r#"{"path":"src/lib.rs"}"#.to_string(),
                encrypted_function_args: None,
                call_id: "call-1".to_string(),
                internal_chat_message_metadata_passthrough: None,
            },
            ResponseItem::FunctionCallOutput {
                id: None,
                call_id: Some("call-1".to_string()),
                name: Some("inspect".to_string()),
                namespace: None,
                output: FunctionCallOutputPayload::from_text("done".to_string()),
                internal_chat_message_metadata_passthrough: None,
            },
            text_message("assistant", "finished"),
        ]),
        "k3",
    )
    .expect("ordinary Kimi history projects");

    assert_eq!(
        serde_json::to_value(projected).expect("projected JSON"),
        json!([
            {"role":"system","content":"base instructions\n\ndeveloper guidance"},
            {"role":"user","content":[
                {"type":"text","text":"inspect this"},
                {"type":"image_url","image_url":{
                    "url":"https://example.test/image.png","detail":"high"
                }}
            ]},
            {"role":"assistant","content":"working","tool_calls":[{
                "id":"call-1","type":"function","function":{
                    "name":"inspect","arguments":"{\"path\":\"src/lib.rs\"}"
                }
            }],"reasoning_details":"private reasoning"},
            {"role":"tool","tool_call_id":"call-1","content":"done"},
            {"role":"assistant","content":"finished","reasoning_content":""}
        ])
    );
}

#[test]
fn responses_encrypted_reasoning_is_not_replayed_to_kimi() {
    let projected = project_messages(
        &prompt(vec![
            reasoning("responses-encrypted", "private Responses reasoning"),
            text_message("assistant", "answer"),
        ]),
        "k3",
    )
    .expect("incompatible reasoning is safely dropped");

    assert_eq!(
        serde_json::to_value(projected).expect("projected JSON"),
        json!([
            {"role":"system","content":"base instructions"},
            {"role":"assistant","content":"answer","reasoning_content":""}
        ])
    );
}

#[test]
fn kimi_reasoning_replays_only_to_its_exact_wire_model() {
    let marker = KimiReasoning::for_model(
        "k3",
        KimiReasoningKey::Reasoning,
        "private Kimi reasoning".to_string(),
    )
    .expect("valid Kimi marker")
    .opaque_marker()
    .to_string();
    let input = vec![
        reasoning(&marker, "private Kimi reasoning"),
        text_message("assistant", "answer"),
    ];

    assert_eq!(
        serde_json::to_value(
            project_messages(&prompt(input.clone()), "k3").expect("exact-model replay")
        )
        .expect("projected JSON"),
        json!([
            {"role":"system","content":"base instructions"},
            {"role":"assistant","content":"answer","reasoning":"private Kimi reasoning"}
        ])
    );
    assert_eq!(
        serde_json::to_value(
            project_messages(&prompt(input), "k3-256k").expect("model-switch projection")
        )
        .expect("projected JSON"),
        json!([
            {"role":"system","content":"base instructions"},
            {"role":"assistant","content":"answer"}
        ])
    );
}

#[test]
fn plaintext_agent_messages_project_as_addressed_user_messages() {
    let projected = project_messages(
        &prompt(vec![ResponseItem::AgentMessage {
            id: None,
            author: "/root".to_string(),
            recipient: "/root/reviewer".to_string(),
            content: vec![AgentMessageInputContent::InputText {
                text: "review this".to_string(),
            }],
            internal_chat_message_metadata_passthrough: None,
        }]),
        "k3",
    )
    .expect("plaintext collaboration history projects");

    assert_eq!(
        serde_json::to_value(projected).expect("projected JSON"),
        json!([
            {"role":"system","content":"base instructions"},
            {"role":"user","content":"Agent message from /root to /root/reviewer:\nreview this"}
        ])
    );
}

#[test]
fn adapted_native_function_tools_project_without_provider_special_cases() {
    let function = |name: &str| {
        ToolSpec::Function(ResponsesApiTool {
            name: name.to_string(),
            description: format!("Call {name}"),
            strict: false,
            defer_loading: None,
            parameters: JsonSchema::object(BTreeMap::new(), Some(Vec::new()), Some(false.into())),
            output_schema: None,
        })
    };
    let projected = project_tools(&[
        function("mcp__repo__inspect"),
        function("apply_patch"),
        function("tool_search"),
        function("spawn_agent"),
    ])
    .expect("landed native function plan projects");

    assert_eq!(
        projected
            .iter()
            .map(|tool| tool.function.name.as_str())
            .collect::<Vec<_>>(),
        [
            "mcp__repo__inspect",
            "apply_patch",
            "tool_search",
            "spawn_agent",
        ]
    );
}

#[test]
fn collaboration_mode_transitions_remain_chronological() {
    let default = "<collaboration_mode>default</collaboration_mode>";
    let plan = "<collaboration_mode>plan</collaboration_mode>";
    let projected = project_messages(
        &prompt(vec![
            text_message("developer", "developer guidance"),
            text_message("user", "first user input"),
            text_message("developer", default),
            text_message("assistant", "first assistant output"),
            text_message("user", "second user input"),
            text_message("developer", plan),
            text_message("assistant", "second assistant output"),
            text_message("user", "current user input"),
            text_message("developer", default),
        ]),
        "k3",
    )
    .expect("collaboration history projects");

    assert_eq!(
        serde_json::to_value(projected).expect("projected JSON"),
        json!([
            {"role":"system","content":format!(
                "base instructions\n\ndeveloper guidance\n\n{KIMI_COLLABORATION_REMINDER_INSTRUCTIONS}"
            )},
            {"role":"user","content":"first user input"},
            {"role":"user","content":format!("<system-reminder>{default}</system-reminder>")},
            {"role":"assistant","content":"first assistant output","reasoning_content":""},
            {"role":"user","content":"second user input"},
            {"role":"user","content":format!("<system-reminder>{plan}</system-reminder>")},
            {"role":"assistant","content":"second assistant output","reasoning_content":""},
            {"role":"user","content":"current user input"},
            {"role":"user","content":format!("<system-reminder>{default}</system-reminder>")}
        ])
    );
}

#[test]
fn nested_collaboration_blocks_collapse_without_splitting_tool_results() {
    let nested = "<collaboration_mode>outer <collaboration_mode>example</collaboration_mode> default</collaboration_mode>";
    let projected = project_messages(
        &prompt(vec![
            text_message("user", "user input"),
            text_message("developer", "<collaboration_mode>plan</collaboration_mode>"),
            text_message("developer", nested),
            ResponseItem::FunctionCall {
                id: None,
                name: "inspect".to_string(),
                namespace: None,
                arguments: "{}".to_string(),
                encrypted_function_args: None,
                call_id: "call-1".to_string(),
                internal_chat_message_metadata_passthrough: None,
            },
            ResponseItem::FunctionCallOutput {
                id: None,
                call_id: Some("call-1".to_string()),
                name: Some("inspect".to_string()),
                namespace: None,
                output: FunctionCallOutputPayload::from_text("done".to_string()),
                internal_chat_message_metadata_passthrough: None,
            },
        ]),
        "k3",
    )
    .expect("nested collaboration history projects");

    let value = serde_json::to_value(projected).expect("projected JSON");
    let messages = value.as_array().expect("message array");
    assert_eq!(
        messages[2],
        json!({"role":"user","content":format!("<system-reminder>{nested}</system-reminder>")})
    );
    assert_eq!(messages[3]["role"], "assistant");
    assert_eq!(messages[4]["role"], "tool");
    assert_eq!(messages[4]["tool_call_id"], "call-1");
}

#[test]
fn kimi_compaction_threshold_uses_both_required_boundaries() {
    let mut model = codex_models_manager::model_info::model_info_from_slug("kimi-test");
    model.context_window = Some(1_000_000);

    assert!(!should_compact(&model, 849_999));
    assert!(should_compact(&model, 850_000));

    model.context_window = Some(262_144);
    assert!(!should_compact(&model, 212_144));
    assert!(should_compact(&model, 212_145));
}

#[test]
fn validates_native_kimi_route_contract() {
    assert!(validate_route(&kimi_route()).is_ok());

    let mut wrong_dialect = kimi_route();
    wrong_dialect.dialect = InferenceDialect::OpenAi;
    let error = validate_route(&wrong_dialect).expect_err("wrong dialect must fail");
    assert!(error.to_string().contains("chat_completions/kimi"));

    let mut query_route = kimi_route();
    query_route.query_params = Some(Default::default());
    assert!(validate_route(&query_route).is_err());
}
