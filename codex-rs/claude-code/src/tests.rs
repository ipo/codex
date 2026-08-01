use codex_protocol::model_inference::InferenceDialect;
use pretty_assertions::assert_eq;
use serde_json::Value;

use super::*;

#[test]
fn messages_wire_shapes_round_trip() {
    let wire: Value = serde_json::from_str(
        r#"{
          "model":"claude-sonnet-5","max_tokens":64000,"stream":true,
          "system":[{"type":"text","text":"You are Codex.","cache_control":{"type":"ephemeral","ttl":"1h"}}],
          "messages":[
            {"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAEC"}}]},
            {"role":"assistant","content":[
              {"type":"thinking","thinking":"inspect","signature":"sig+/="},
              {"type":"redacted_thinking","data":"opaque+/="},
              {"type":"tool_use","id":"call-1","name":"read","input":{"path":"a"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"call-1","content":[
              {"type":"text","text":"failed"},
              {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"AwQ="}}
            ],"is_error":true,"cache_control":{"type":"ephemeral","ttl":"5m"}}]}],
          "tools":[{"name":"read","description":"Read a file","input_schema":{"type":"object","properties":{"path":{"type":"string"}}}}],
          "thinking":{"type":"adaptive","display":"omitted"},"output_config":{"effort":"xhigh"},
          "context_management":{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]},
          "metadata":{"user_id":"user"}
        }"#,
    )
    .expect("valid request JSON");
    let request: MessagesRequest = serde_json::from_value(wire.clone()).expect("valid request");

    assert_eq!(
        serde_json::to_value(request).expect("serialize request"),
        wire
    );
}

#[test]
fn usage_preserves_populated_and_zero_counters_without_inventing_absent_values() {
    let wire = [
        serde_json::json!({"input_tokens": 10, "output_tokens": 20, "cache_creation_input_tokens": 3, "cache_read_input_tokens": 4, "thinking_tokens": 5}),
        serde_json::json!({"input_tokens": 10, "output_tokens": 20, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "thinking_tokens": 0}),
        serde_json::json!({"input_tokens": 10, "output_tokens": 20, "cache_creation_input_tokens": null, "cache_read_input_tokens": null, "thinking_tokens": null}),
        serde_json::json!({"input_tokens": 10, "output_tokens": 20}),
    ];
    let expected = [
        (Some(3), Some(4), Some(5)),
        (Some(0), Some(0), Some(0)),
        (None, None, None),
        (None, None, None),
    ]
    .map(
        |(cache_creation_input_tokens, cache_read_input_tokens, thinking_tokens)| Usage {
            input_tokens: 10,
            output_tokens: 20,
            cache_creation_input_tokens,
            cache_read_input_tokens,
            thinking_tokens,
            cache_creation: None,
        },
    );
    let parsed = wire
        .iter()
        .cloned()
        .map(serde_json::from_value)
        .collect::<Result<Vec<Usage>, _>>()
        .expect("valid usage");

    assert_eq!(parsed, expected);
    assert_eq!(
        parsed
            .into_iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()
            .expect("serialize usage"),
        vec![
            wire[0].clone(),
            wire[1].clone(),
            wire[3].clone(),
            wire[3].clone()
        ]
    );
}

#[test]
fn replay_is_byte_exact_only_for_the_bound_model_and_dialect() {
    let blocks = vec![
        ThinkingReplayBlock::Signed {
            thinking: "think\0\n🦀+/=".to_string(),
            signature: "sig\0+/=".to_string(),
        },
        ThinkingReplayBlock::Redacted {
            data: "secret\0+/=".to_string(),
        },
    ];
    let opaque = encode_thinking_replay(
        InferenceDialect::ClaudeCode,
        "claude-sonnet-5",
        blocks.clone(),
    )
    .expect("encode replay");

    assert_eq!(
        decode_thinking_replay(
            &opaque,
            vec!["visible".to_string()],
            InferenceDialect::ClaudeCode,
            "claude-sonnet-5",
        ),
        Ok(ReplayDecision::Native(blocks))
    );
    for (dialect, model) in [
        (InferenceDialect::OpenAi, "claude-sonnet-5"),
        (InferenceDialect::ClaudeCode, "claude-opus-5"),
    ] {
        assert_eq!(
            decode_thinking_replay(&opaque, vec!["visible".to_string()], dialect, model),
            Ok(ReplayDecision::ModelOrDialectChanged {
                visible: vec!["visible".to_string()]
            })
        );
    }
}

#[test]
fn malformed_unknown_and_unrelated_opaque_content_are_explicit() {
    assert!(matches!(
        decode_thinking_replay(
            "codex:anthropic-thinking:{",
            vec![],
            InferenceDialect::ClaudeCode,
            "m"
        ),
        Err(ReplayError::Malformed(_))
    ));
    assert_eq!(
        decode_thinking_replay(
            r#"codex:anthropic-thinking:{"version":2,"future_shape":true}"#,
            vec![],
            InferenceDialect::ClaudeCode,
            "m",
        ),
        Err(ReplayError::UnknownVersion(2))
    );
    assert_eq!(
        decode_thinking_replay(
            "responses-encrypted-content",
            vec!["visible".to_string()],
            InferenceDialect::ClaudeCode,
            "m",
        ),
        Ok(ReplayDecision::UnrelatedOpaqueContent {
            visible: vec!["visible".to_string()]
        })
    );
}
