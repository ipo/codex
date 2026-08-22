use pretty_assertions::assert_eq;

use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bounded_health_readiness_retries_503_before_discovery_and_inference() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(Sequence {
            next: AtomicUsize::new(0),
            responses: vec![
                ResponseTemplate::new(503),
                ResponseTemplate::new(503),
                ResponseTemplate::new(200).set_body_json(json!({"status": "ok"})),
            ],
        })
        .up_to_n_times(3)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{"id": WINDOWS_MODEL}]
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    mount_token_counts(&server, [20]).await;
    mount_local_inference(
        &server,
        vec![text_response(
            "ready",
            "ready-message",
            "ready after loading",
        )],
    )
    .await;
    let test = local_builder(&server).build_with_auto_env(&server).await?;

    submit_turn(&test, "Wait for bounded local readiness").await?;

    let requests = server.received_requests().await.expect("captured requests");
    assert_eq!(
        requests
            .iter()
            .map(|request| request.url.path())
            .collect::<Vec<_>>(),
        [
            "/health",
            "/health",
            "/health",
            "/v1/models",
            "/v1/responses/input_tokens",
            "/v1/responses",
        ]
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn truncated_transport_stream_retries_and_rediscovers_once() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = MockServer::start().await;
    mount_ready(&server).await;
    mount_token_counts(&server, [20, 20]).await;
    mount_local_inference(
        &server,
        vec![
            local_sse(vec![responses::ev_response_created("truncated")]),
            text_response("retried", "retried-message", "transport recovered"),
        ],
    )
    .await;
    let test = local_builder(&server).build_with_auto_env(&server).await?;

    submit_turn(&test, "Retry one truncated local transport").await?;

    let requests = server.received_requests().await.expect("captured requests");
    for (request_path, expected) in [
        ("/health", 2),
        ("/v1/models", 2),
        ("/v1/responses/input_tokens", 2),
        ("/v1/responses", 2),
    ] {
        assert_eq!(
            requests
                .iter()
                .filter(|request| request.url.path() == request_path)
                .count(),
            expected,
            "unexpected count for {request_path}"
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deterministic_template_400_fails_without_retry() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = MockServer::start().await;
    mount_ready(&server).await;
    mount_token_counts(&server, [20]).await;
    mount_local_inference(
        &server,
        vec![ResponseTemplate::new(400).set_body_json(json!({
            "error": {"message": "System message must be at the beginning"}
        }))],
    )
    .await;
    let test = local_builder(&server).build_with_auto_env(&server).await?;

    let error = submit_turn(&test, "Do not retry a template error")
        .await
        .expect_err("template error should fail the turn");

    assert!(
        error
            .to_string()
            .contains("System message must be at the beginning")
    );
    let requests = server.received_requests().await.expect("captured requests");
    for (request_path, expected) in [
        ("/health", 1),
        ("/v1/models", 1),
        ("/v1/responses/input_tokens", 1),
        ("/v1/responses", 1),
    ] {
        assert_eq!(
            requests
                .iter()
                .filter(|request| request.url.path() == request_path)
                .count(),
            expected,
            "unexpected count for {request_path}"
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_aborts_one_inference_without_retry() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = MockServer::start().await;
    mount_ready(&server).await;
    mount_token_counts(&server, [20]).await;
    mount_local_inference(
        &server,
        vec![
            text_response("cancelled", "never", "never returned")
                .set_delay(Duration::from_secs(30)),
        ],
    )
    .await;
    let test = local_builder(&server).build_with_auto_env(&server).await?;

    test.codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: "Cancel the active local request".to_string(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await?;
    wait_for_event_match(&test.codex, |event| {
        matches!(event, EventMsg::TurnStarted(_)).then_some(())
    })
    .await;
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if received(&server, "/v1/responses").await.len() == 1 {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await?;

    test.codex.submit(Op::Interrupt).await?;
    wait_for_event_match(&test.codex, |event| {
        matches!(event, EventMsg::TurnAborted(_)).then_some(())
    })
    .await;
    tokio::time::sleep(Duration::from_millis(750)).await;

    let requests = server.received_requests().await.expect("captured requests");
    for (request_path, expected) in [
        ("/health", 1),
        ("/v1/models", 1),
        ("/v1/responses/input_tokens", 1),
        ("/v1/responses", 1),
    ] {
        assert_eq!(
            requests
                .iter()
                .filter(|request| request.url.path() == request_path)
                .count(),
            expected,
            "unexpected count for {request_path}"
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retryable_inference_failure_invalidates_and_rediscovers_local_model() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = MockServer::start().await;
    mount_ready(&server).await;
    mount_token_counts(&server, [100, 100]).await;
    mount_local_inference(
        &server,
        vec![
            ResponseTemplate::new(503)
                .set_body_json(json!({"error": {"message": "slot unavailable"}})),
            text_response("recovered", "recovered-message", "recovered locally"),
        ],
    )
    .await;
    let test = local_builder(&server).build_with_auto_env(&server).await?;

    submit_turn(&test, "Recover after a local service failure").await?;

    let requests = server.received_requests().await.expect("captured requests");
    for (request_path, expected) in [
        ("/health", 2),
        ("/v1/models", 2),
        ("/v1/responses/input_tokens", 2),
        ("/v1/responses", 2),
    ] {
        assert_eq!(
            requests
                .iter()
                .filter(|request| request.url.path() == request_path)
                .count(),
            expected,
            "unexpected count for {request_path}"
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_local_preflight_overflow_runs_one_normal_compaction_cycle() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = MockServer::start().await;
    mount_ready(&server).await;
    mount_token_counts(&server, [230_913, 120, 140]).await;
    mount_local_inference(
        &server,
        vec![
            text_response(
                "compact-summary",
                "compact-message",
                "COMPACTED LOCAL HISTORY",
            ),
            text_response("after-compact", "final-message", "compaction complete"),
        ],
    )
    .await;
    let test = local_builder(&server).build_with_auto_env(&server).await?;

    submit_turn(&test, "Trigger exact preflight compaction").await?;

    let counts = received(&server, "/v1/responses/input_tokens").await;
    let inference = received(&server, "/v1/responses").await;
    assert_eq!((counts.len(), inference.len()), (3, 2));
    let compact_body: Value = inference[0].body_json()?;
    assert!(
        compact_body
            .to_string()
            .to_ascii_lowercase()
            .contains("summar"),
        "first inference after overflow should be the normal compaction prompt"
    );
    let retried_body: Value = inference[1].body_json()?;
    assert!(retried_body.to_string().contains("COMPACTED LOCAL HISTORY"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn process_wide_local_lease_serializes_inference_across_sessions() -> Result<()> {
    skip_if_no_network!(Ok(()));
    let server = MockServer::start().await;
    mount_ready(&server).await;
    Mock::given(method("POST"))
        .and(path("/v1/responses/input_tokens"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"input_tokens": 20})))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            text_response("leased", "leased-message", "lease complete")
                .set_delay(Duration::from_millis(400)),
        )
        .up_to_n_times(2)
        .mount(&server)
        .await;
    let first = local_builder(&server).build_with_auto_env(&server).await?;
    let second = local_builder(&server).build_with_auto_env(&server).await?;

    let observe_serialization = async {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        loop {
            let count = received(&server, "/v1/responses").await.len();
            if count == 1 {
                tokio::time::sleep(Duration::from_millis(100)).await;
                assert_eq!(received(&server, "/v1/responses").await.len(), 1);
                return Ok::<_, anyhow::Error>(());
            }
            if tokio::time::Instant::now() >= deadline {
                anyhow::bail!("first local inference request did not arrive in time");
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    tokio::try_join!(
        submit_turn(&first, "first lease holder"),
        submit_turn(&second, "second lease waiter"),
        observe_serialization,
    )?;

    assert_eq!(received(&server, "/v1/responses").await.len(), 2);
    Ok(())
}
