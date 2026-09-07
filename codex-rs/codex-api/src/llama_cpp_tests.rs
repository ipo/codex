use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use codex_http_client::HttpClientBuilder;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ReasoningEffort;
use futures::StreamExt;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use tokio::sync::Semaphore;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::Request;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;

use super::*;

const WIRE_MODEL: &str = r"F:\ai\llama-server\models\JonathanColetti\Qwen3.8-27B-Uncensored-GGUF\Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf";

fn direct_http_client() -> HttpClient {
    HttpClientBuilder::new()
        .build_direct()
        .expect("direct test HTTP client")
}

fn runtime(server: &MockServer) -> LlamaCppRuntime {
    LlamaCppRuntime::for_endpoint(
        direct_http_client(),
        &server.uri(),
        Arc::new(Semaphore::new(1)),
    )
}

fn discovery(data: Value) -> ModelsResponse {
    serde_json::from_value(json!({"data": data})).expect("valid discovery response")
}

fn message_history(text: &str) -> Vec<ResponseItem> {
    serde_json::from_value(json!([{
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": text}]
    }]))
    .expect("valid response history")
}

fn request_input(
    history: Vec<ResponseItem>,
    effort: Option<ReasoningEffort>,
) -> LlamaCppRequestInput {
    LlamaCppRequestInput {
        instructions: "You are a coding agent.".to_string(),
        history,
        tools: None,
        parallel_tool_calls: true,
        reasoning_effort: effort,
    }
}

fn sse(events: &[Value]) -> String {
    events
        .iter()
        .map(|event| format!("event: {}\ndata: {event}\n\n", event["type"]))
        .collect()
}

async fn mount_ready_and_models(server: &MockServer, data: Value) {
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"status": "ok"})))
        .mount(server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": data})))
        .mount(server)
        .await;
}

async fn mount_token_count(server: &MockServer, input_tokens: u64) {
    Mock::given(method("POST"))
        .and(path("/v1/responses/input_tokens"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({"input_tokens": input_tokens})),
        )
        .mount(server)
        .await;
}

fn one_model_data() -> Value {
    json!([{
        "id": WIRE_MODEL,
        "meta": {"n_ctx": 131072}
    }])
}

#[test]
fn derives_zero_one_and_multiple_model_catalogs() {
    let empty = LlamaCppCatalog::from_discovery(discovery(json!([])));
    assert_eq!(empty, LlamaCppCatalog::default());
    assert_eq!(
        empty.resolve("llama-cpp-local").unwrap_err(),
        LlamaCppModelSelectionError {
            requested: "llama-cpp-local".to_string(),
            currently_discovered: Vec::new(),
        }
    );

    let one = LlamaCppCatalog::from_discovery(discovery(one_model_data()));
    assert_eq!(
        one,
        LlamaCppCatalog {
            models: vec![LlamaCppCatalogEntry {
                canonical_id: format!("local/{WIRE_MODEL}"),
                wire_model: WIRE_MODEL.to_string(),
                display_name: "Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf".to_string(),
                aliases: vec![
                    "Qwen3.8-27B-Uncensored-noMTP-IQ2_M".to_string(),
                    "llama-cpp-local".to_string(),
                ],
                context_window: 131_072,
                max_input_tokens: 121_856,
                max_output_tokens: 8_192,
                safety_margin_tokens: 1_024,
            }],
        }
    );

    let multiple = LlamaCppCatalog::from_discovery(discovery(json!([
        {"id": "C:\\models\\shared.gguf", "meta": {"n_ctx": 9000}},
        {"id": "D:\\other\\shared.gguf", "meta": {"n_ctx": 0}},
        {"id": "/models/unique.gguf"}
    ])));
    assert_eq!(
        multiple
            .models
            .iter()
            .map(|model| (
                model.canonical_id.as_str(),
                model.wire_model.as_str(),
                model.display_name.as_str(),
                model.aliases.clone(),
                model.context_window,
                model.max_input_tokens,
            ))
            .collect::<Vec<_>>(),
        vec![
            (
                "local/C:\\models\\shared.gguf",
                "C:\\models\\shared.gguf",
                "shared.gguf",
                Vec::<String>::new(),
                9_000,
                0,
            ),
            (
                "local/D:\\other\\shared.gguf",
                "D:\\other\\shared.gguf",
                "shared.gguf",
                Vec::<String>::new(),
                32_768,
                23_552,
            ),
            (
                "local//models/unique.gguf",
                "/models/unique.gguf",
                "unique.gguf",
                vec!["unique".to_string()],
                32_768,
                23_552,
            ),
        ]
    );
}

#[test]
fn missing_model_error_lists_current_canonical_choices() {
    let catalog = LlamaCppCatalog::from_discovery(discovery(json!([
        {"id": "A.gguf", "meta": {"n_ctx": 32768}},
        {"id": "B.gguf", "meta": {"n_ctx": 65536}}
    ])));

    assert_eq!(
        catalog.resolve("local/gone.gguf").unwrap_err().to_string(),
        "llama.cpp model `local/gone.gguf` is unavailable; currently discovered choices: local/A.gguf, local/B.gguf"
    );
}

#[test]
fn aliases_are_unique_case_insensitively_after_removing_the_gguf_suffix() {
    let catalog = LlamaCppCatalog::from_discovery(discovery(json!([
        {"id": "C:\\models\\Qwen.gguf"},
        {"id": "D:\\models\\qwen.gguf"}
    ])));

    assert_eq!(
        catalog
            .models
            .iter()
            .map(|model| model.aliases.clone())
            .collect::<Vec<_>>(),
        vec![Vec::<String>::new(), Vec::<String>::new()]
    );
}

#[test]
fn deterministic_stream_failures_are_not_retryable() {
    let error = classify_error(ApiError::Retryable {
        message: "template error: unsupported item".to_string(),
        delay: None,
    });

    assert!(matches!(
        error,
        ApiError::InvalidRequest { message } if message == "template error: unsupported item"
    ));
}

#[tokio::test]
async fn exact_preflight_and_stream_replay_full_typed_history() {
    let server = MockServer::start().await;
    mount_ready_and_models(&server, one_model_data()).await;
    mount_token_count(&server, 42).await;
    let inference_attempt = Arc::new(AtomicUsize::new(0));
    let responder_attempt = Arc::clone(&inference_attempt);
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(move |_request: &Request| {
            let response = if responder_attempt.fetch_add(1, Ordering::SeqCst) == 0 {
                sse(&[
                    json!({"type": "response.created", "response": {"id": "resp-1"}}),
                    json!({
                        "type": "response.reasoning_text.delta",
                        "item_id": "reasoning-1",
                        "content_index": 0,
                        "delta": "Need the tool."
                    }),
                    json!({
                        "type": "response.output_item.done",
                        "item": {
                            "id": "reasoning-1",
                            "type": "reasoning",
                            "summary": [],
                            "content": [{"type": "reasoning_text", "text": "Need the tool."}],
                            "encrypted_content": ""
                        }
                    }),
                    json!({
                        "type": "response.function_call_arguments.delta",
                        "item_id": "function-1",
                        "call_id": "call-1",
                        "delta": "{\"path\":\"Cargo.toml\"}"
                    }),
                    json!({
                        "type": "response.output_item.done",
                        "item": {
                            "id": "function-1",
                            "type": "function_call",
                            "name": "read_file",
                            "arguments": "{\"path\":\"Cargo.toml\"}",
                            "call_id": "call-1"
                        }
                    }),
                    json!({"type": "response.completed", "response": {"id": "resp-1"}}),
                ])
            } else {
                sse(&[
                    json!({
                        "type": "response.output_text.delta",
                        "item_id": "message-1",
                        "delta": "done"
                    }),
                    json!({
                        "type": "response.output_item.done",
                        "item": {
                            "id": "message-1",
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "done"}]
                        }
                    }),
                    json!({"type": "response.completed", "response": {"id": "resp-2"}}),
                ])
            };
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(response)
        })
        .mount(&server)
        .await;

    let runtime = runtime(&server);
    let initial_history = message_history("Read Cargo.toml.");
    let prepared = runtime
        .prepare(
            "llama-cpp-local",
            request_input(initial_history.clone(), None),
        )
        .await
        .expect("first request preflight");
    assert_eq!(prepared.model.wire_model, WIRE_MODEL);
    let first_events = runtime
        .stream(prepared)
        .await
        .expect("first inference")
        .collect::<Vec<_>>()
        .await;
    assert!(first_events.iter().any(|event| matches!(
        event,
        Ok(ResponseEvent::ReasoningContentDelta { delta, .. }) if delta == "Need the tool."
    )));
    let completed_calls = first_events
        .iter()
        .filter(|event| {
            matches!(
                event,
                Ok(ResponseEvent::OutputItemDone(
                    ResponseItem::FunctionCall { .. }
                ))
            )
        })
        .count();
    assert_eq!(completed_calls, 1);
    assert!(matches!(
        first_events.last(),
        Some(Ok(ResponseEvent::Completed { response_id, .. })) if response_id == "resp-1"
    ));

    let continuation_history: Vec<ResponseItem> = serde_json::from_value(json!([
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Read Cargo.toml."}]
        },
        {
            "id": "reasoning-1",
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "Need the tool."}],
            "encrypted_content": ""
        },
        {
            "id": "function-1",
            "type": "function_call",
            "name": "read_file",
            "arguments": "{\"path\":\"Cargo.toml\"}",
            "call_id": "call-1"
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "workspace contents"
        }
    ]))
    .expect("valid continuation history");
    let prepared = runtime
        .prepare(
            &format!("local/{WIRE_MODEL}"),
            request_input(continuation_history.clone(), Some(ReasoningEffort::High)),
        )
        .await
        .expect("continuation preflight");
    let second_events = runtime
        .stream(prepared)
        .await
        .expect("continuation inference")
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(
        second_events.last(),
        Some(Ok(ResponseEvent::Completed { response_id, .. })) if response_id == "resp-2"
    ));

    let requests = server.received_requests().await.expect("recorded requests");
    assert_eq!(
        requests
            .iter()
            .map(|request| request.url.path())
            .collect::<Vec<_>>(),
        vec![
            "/health",
            "/v1/models",
            "/v1/responses/input_tokens",
            "/v1/responses",
            "/v1/responses/input_tokens",
            "/v1/responses",
        ]
    );
    for request in &requests {
        assert!(!request.headers.contains_key("authorization"));
    }
    let first_preflight: Value =
        serde_json::from_slice(&requests[2].body).expect("first preflight body");
    let first_inference: Value =
        serde_json::from_slice(&requests[3].body).expect("first inference body");
    assert_eq!(first_preflight, first_inference);
    assert_eq!(first_preflight["model"], WIRE_MODEL);
    assert_eq!(first_preflight["input"], json!(initial_history));
    assert_eq!(first_preflight["reasoning"], json!({"effort": "low"}));
    assert_eq!(
        first_preflight["chat_template_kwargs"],
        json!({"enable_thinking": true, "preserve_thinking": true})
    );
    assert_eq!(first_preflight["temperature"], 1.0);
    assert_eq!(first_preflight["top_p"], 0.95);
    assert_eq!(first_preflight["top_k"], 20);
    assert_eq!(first_preflight["min_p"], 0.0);
    assert_eq!(first_preflight["presence_penalty"], 0.0);
    assert_eq!(first_preflight["repeat_penalty"], 1.0);
    assert_eq!(first_preflight["max_output_tokens"], 8_192);
    assert_eq!(first_preflight["stream"], true);
    assert_eq!(first_preflight["cache_prompt"], true);
    assert!(first_preflight.get("previous_response_id").is_none());

    let second_preflight: Value =
        serde_json::from_slice(&requests[4].body).expect("second preflight body");
    let second_inference: Value =
        serde_json::from_slice(&requests[5].body).expect("second inference body");
    assert_eq!(second_preflight, second_inference);
    assert_eq!(second_preflight["input"], json!(continuation_history));
    assert_eq!(second_preflight["reasoning"], json!({"effort": "xhigh"}));
}

#[tokio::test]
async fn non_thinking_sampling_and_exact_context_preflight_are_enforced() {
    let server = MockServer::start().await;
    mount_ready_and_models(&server, one_model_data()).await;
    mount_token_count(&server, 121_857).await;
    let runtime = runtime(&server);

    let error = runtime
        .prepare(
            "llama-cpp-local",
            request_input(message_history("too large"), Some(ReasoningEffort::None)),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, ApiError::ContextWindowExceeded));

    let requests = server.received_requests().await.expect("recorded requests");
    let body: Value = serde_json::from_slice(&requests[2].body).expect("preflight body");
    assert_eq!(body["reasoning"], json!({"effort": "none"}));
    assert_eq!(
        body["chat_template_kwargs"],
        json!({"enable_thinking": false, "preserve_thinking": false})
    );
    assert_eq!(body["temperature"], 0.7);
    assert_eq!(body["top_p"], 0.8);
    assert_eq!(body["presence_penalty"], 1.5);
}

#[tokio::test]
async fn readiness_is_bounded_and_catalog_is_cached_until_invalidation() {
    let server = MockServer::start().await;
    let health_attempt = Arc::new(AtomicUsize::new(0));
    let responder_attempt = Arc::clone(&health_attempt);
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(move |_request: &Request| {
            if responder_attempt.fetch_add(1, Ordering::SeqCst) < 2 {
                ResponseTemplate::new(503)
            } else {
                ResponseTemplate::new(200).set_body_json(json!({"status": "ok"}))
            }
        })
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": one_model_data()})))
        .mount(&server)
        .await;
    let mut runtime = runtime(&server);
    runtime.readiness = ReadinessPolicy {
        attempts: 3,
        initial_delay: Duration::from_millis(1),
    };

    runtime
        .resolve_model("llama-cpp-local")
        .await
        .expect("eventual discovery");
    runtime
        .resolve_model("llama-cpp-local")
        .await
        .expect("cached discovery");

    assert_eq!(health_attempt.load(Ordering::SeqCst), 3);
    let requests = server.received_requests().await.expect("recorded requests");
    assert_eq!(
        requests
            .iter()
            .filter(|request| request.url.path() == "/v1/models")
            .count(),
        1
    );
}

#[tokio::test]
async fn transport_invalidation_rediscovers_and_reports_disappeared_selection() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"status": "ok"})))
        .mount(&server)
        .await;
    let discovery_attempt = Arc::new(AtomicUsize::new(0));
    let responder_attempt = Arc::clone(&discovery_attempt);
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(move |_request: &Request| {
            let id = if responder_attempt.fetch_add(1, Ordering::SeqCst) == 0 {
                "old.gguf"
            } else {
                "replacement.gguf"
            };
            ResponseTemplate::new(200)
                .set_body_json(json!({"data": [{"id": id, "meta": {"n_ctx": 32768}}]}))
        })
        .mount(&server)
        .await;
    mount_token_count(&server, 10).await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(503).set_body_json(json!({
            "error": {"message": "slot unavailable"}
        })))
        .mount(&server)
        .await;
    let runtime = runtime(&server);
    let prepared = runtime
        .prepare(
            "local/old.gguf",
            request_input(message_history("hello"), None),
        )
        .await
        .expect("old model preflight");
    let error = match runtime.stream(prepared).await {
        Ok(_) => panic!("inference should fail"),
        Err(error) => error,
    };
    assert!(matches!(error, ApiError::Retryable { .. }));

    let error = runtime.resolve_model("local/old.gguf").await.unwrap_err();
    assert_eq!(
        error.to_string(),
        "invalid request: llama.cpp model `local/old.gguf` is unavailable; currently discovered choices: local/replacement.gguf"
    );
    assert_eq!(discovery_attempt.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn inference_requests_share_one_endpoint_wide_lease() {
    let server = MockServer::start().await;
    mount_ready_and_models(&server, one_model_data()).await;
    mount_token_count(&server, 10).await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse(&[json!({
                    "type": "response.completed",
                    "response": {"id": "resp"}
                })]))
                .set_delay(Duration::from_millis(100)),
        )
        .mount(&server)
        .await;
    let lease = Arc::new(Semaphore::new(1));
    let first_runtime =
        LlamaCppRuntime::for_endpoint(direct_http_client(), &server.uri(), Arc::clone(&lease));
    let second_runtime =
        LlamaCppRuntime::for_endpoint(direct_http_client(), &server.uri(), Arc::clone(&lease));
    let first = first_runtime
        .prepare(
            "llama-cpp-local",
            request_input(message_history("first"), None),
        )
        .await
        .expect("first preflight");
    let second = second_runtime
        .prepare(
            "llama-cpp-local",
            request_input(message_history("second"), None),
        )
        .await
        .expect("second preflight");

    let first_task = tokio::spawn(async move {
        first_runtime
            .stream(first)
            .await
            .expect("first stream")
            .collect::<Vec<_>>()
            .await
    });
    tokio::time::sleep(Duration::from_millis(10)).await;
    let second_task = tokio::spawn(async move {
        second_runtime
            .stream(second)
            .await
            .expect("second stream")
            .collect::<Vec<_>>()
            .await
    });
    tokio::time::sleep(Duration::from_millis(30)).await;
    let active_inference_requests = server
        .received_requests()
        .await
        .expect("recorded requests")
        .into_iter()
        .filter(|request| request.url.path() == "/v1/responses")
        .count();
    assert_eq!(active_inference_requests, 1);

    assert!(
        first_task
            .await
            .expect("first task join")
            .iter()
            .all(Result::is_ok)
    );
    assert!(
        second_task
            .await
            .expect("second task join")
            .iter()
            .all(Result::is_ok)
    );
    assert_eq!(lease.available_permits(), 1);
}
