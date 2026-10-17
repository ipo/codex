use std::io::Cursor;

use anyhow::Result;
use codex_core::TurnInputRequest;
use codex_core::config::Config;
use codex_features::Feature;
use codex_model_provider_info::CLAUDEFLARE_PROVIDER_ID;
use codex_model_provider_info::ModelProviderWireRoute;
use codex_model_provider_info::WireApi;
use codex_model_provider_info::built_in_model_providers;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::user_input::UserInput;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::skip_if_wine_exec;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use image::DynamicImage;
use image::Rgb;
use image::RgbImage;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use wiremock::Mock;
use wiremock::Request;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn kimi_image_tool_continuation_and_resume_keep_output_budget() -> Result<()> {
    skip_if_no_network!(Ok(()));
    skip_if_wine_exec!(Ok(()), "code-mode exec is unavailable under Wine-exec");

    let server = start_mock_server().await;
    let base_url = format!("{}/v1", server.uri());
    let configure = move |config: &mut Config| {
        config.model = Some("kimi/k3-256k".to_string());
        config.model_provider = built_in_model_providers(/*openai_base_url*/ None)
            .remove(CLAUDEFLARE_PROVIDER_ID)
            .expect("Claudeflare provider");
        config.model_provider.base_url = Some(base_url.clone());
        config.model_provider.supports_websockets = false;
        config
            .features
            .enable(Feature::CodeMode)
            .expect("code mode");
        config.model_provider.wire_routes.insert(
            "kimi_code".to_string(),
            ModelProviderWireRoute {
                wire_api: WireApi::ChatCompletions,
                dialect: InferenceDialect::Kimi,
                base_url,
                request_path: "chat/completions".to_string(),
                query_params: None,
                request_max_retries: Some(0),
                stream_max_retries: Some(0),
                stream_idle_timeout_ms: None,
            },
        );
        config.base_instructions = Some("text".repeat(100_000));
    };
    let mut builder = test_codex().with_config(configure.clone());
    let test = builder.build_with_auto_env(&server).await?;

    // Incompressible synthetic RGB pixels yield four inline images near 1.9 MB.
    let mut noise = 42_u32;
    let pixels = RgbImage::from_fn(/*width*/ 350, /*height*/ 350, |_, _| {
        let mut rgb = [0; 3];
        for channel in &mut rgb {
            noise ^= noise << 13;
            noise ^= noise >> 17;
            noise ^= noise << 5;
            *channel = noise as u8;
        }
        Rgb(rgb)
    });
    let mut png = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(pixels).write_to(&mut png, image::ImageFormat::Png)?;
    test.fs()
        .write_file(
            &test.workspace_path_uri("noise.png")?,
            png.into_inner(),
            Default::default(),
            /*sandbox*/ None,
        )
        .await?;
    let code = "const result = await tools.view_image({path:'noise.png'}); for (let i=0; i<4; i++) image(result);";
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |request: &Request| {
            let body: Value = serde_json::from_slice(&request.body).expect("native request JSON");
            let has_output = body["messages"]
                .as_array()
                .expect("messages")
                .iter()
                .any(|message| message["role"] == "tool");
            let (delta, finish) = if has_output {
                (
                    json!({"role":"assistant","content":"images inspected"}),
                    "stop",
                )
            } else {
                (
                    json!({"role":"assistant","tool_calls":[{
                        "index":0,"id":"images-1","type":"function",
                        "function":{"name":"exec","arguments":json!({"code":code}).to_string()}
                    }]}),
                    "tool_calls",
                )
            };
            let event = json!({"id":"kimi-images","choices":[{
                "index":0,"delta":delta,"finish_reason":finish
            }],"usage":{"prompt_tokens":100000,"completion_tokens":10,"total_tokens":100010}});
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!("data: {event}\n\ndata: [DONE]\n\n"))
        })
        .mount(&server)
        .await;

    test.codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "inspect the four images".to_string(),
            text_elements: Vec::new(),
        }]))
        .await?;
    wait_for_event(&test.codex, |event| {
        assert!(
            !matches!(event, EventMsg::Error(_)),
            "native turn failed: {event:?}"
        );
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    let resumed = builder
        .with_config(configure)
        .restart(&server, &test)
        .await?;
    resumed
        .codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "continue from the recorded images".to_string(),
            text_elements: Vec::new(),
        }]))
        .await?;
    wait_for_event(&resumed.codex, |event| {
        assert!(
            !matches!(event, EventMsg::Error(_)),
            "resumed turn failed: {event:?}"
        );
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;

    let requests = server.received_requests().await.expect("captured requests");
    let bodies = requests
        .iter()
        .filter(|request| request.url.path() == "/v1/chat/completions")
        .map(|request| serde_json::from_slice::<Value>(&request.body).expect("JSON"))
        .collect::<Vec<_>>();
    assert_eq!(
        bodies.len(),
        3,
        "one tool call, continuation, then resume; no compaction"
    );
    let mut image_history = Vec::new();
    for body in &bodies[1..] {
        assert_eq!(body["max_completion_tokens"], 131_072);
        assert_eq!(body["model"], "k3-256k");
        let images = body["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .filter(|message| message["role"] == "tool")
            .filter_map(|message| message["content"].as_array())
            .flatten()
            .filter(|part| part["type"] == "image_url")
            .map(|part| {
                part["image_url"]["url"]
                    .as_str()
                    .expect("image URL")
                    .to_string()
            })
            .collect::<Vec<_>>();
        assert_eq!(images.len(), 4);
        assert!((1_800_000..2_100_000).contains(&images.iter().map(String::len).sum::<usize>()));
        image_history.push(images);
    }
    assert_eq!(
        image_history[0], image_history[1],
        "resume preserves image payloads"
    );

    // An explicitly requested compaction also uses the image-aware request budget.
    resumed.codex.submit(Op::Compact).await?;
    wait_for_event(&resumed.codex, |event| {
        assert!(
            !matches!(event, EventMsg::Error(_)),
            "compaction failed: {event:?}"
        );
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    let requests = server
        .received_requests()
        .await
        .expect("captured compaction");
    let native = requests
        .iter()
        .filter(|request| request.url.path() == "/v1/chat/completions")
        .collect::<Vec<_>>();
    assert_eq!(
        native.len(),
        4,
        "manual compaction adds exactly one request"
    );
    let compact: Value = serde_json::from_slice(&native[3].body)?;
    assert_eq!(compact["max_completion_tokens"], 131_072);
    assert!(
        compact["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .any(|message| message["role"] == "tool" && message["content"].is_array())
    );
    Ok(())
}
