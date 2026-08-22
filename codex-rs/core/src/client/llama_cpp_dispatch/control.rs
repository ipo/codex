use super::*;

pub(super) async fn ensure_discovered_model(
    transport: &ReqwestTransport,
    provider: &ApiProvider,
    endpoint_key: &str,
    expected_basename: &str,
) -> Result<String> {
    let runtime = LlamaCppRuntime::global();
    if let Some(model) = runtime.discovered_model(endpoint_key)? {
        return Ok(model);
    }
    wait_until_ready(transport, provider).await?;
    let response = execute_json::<ModelsResponse>(
        transport,
        provider,
        Method::GET,
        "models",
        /*body*/ None,
    )
    .await
    .map_err(|error| codex_api::map_api_error(classify_api_error(error)))?;
    let model = response.data.into_iter().next().ok_or_else(|| {
        CodexErr::InvalidRequest("llama.cpp model discovery returned no models".to_string())
    })?;
    let basename = model
        .id
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(model.id.as_str());
    if basename != expected_basename {
        return Err(CodexErr::InvalidRequest(format!(
            "llama.cpp advertised unexpected model `{}`; expected basename `{expected_basename}`",
            model.id
        )));
    }
    runtime.cache_model(endpoint_key.to_string(), model.id.clone())?;
    Ok(model.id)
}

async fn wait_until_ready(transport: &ReqwestTransport, provider: &ApiProvider) -> Result<()> {
    let mut health_provider = provider.clone();
    health_provider.base_url = provider
        .base_url
        .trim_end_matches('/')
        .strip_suffix("/v1")
        .ok_or_else(|| CodexErr::InvalidRequest("llama.cpp base URL must end in `/v1`".into()))?
        .to_string();
    let mut delay = HEALTH_INITIAL_DELAY;
    for attempt in 0..HEALTH_ATTEMPTS {
        match execute_json::<Value>(
            transport,
            &health_provider,
            Method::GET,
            "health",
            /*body*/ None,
        )
        .await
        {
            Ok(response) if response["status"] == "ok" => return Ok(()),
            Ok(_) => {
                return Err(CodexErr::Stream(
                    "llama.cpp health response did not report ready".to_string(),
                ));
            }
            Err(ApiError::Transport(codex_api::TransportError::Http {
                status: StatusCode::SERVICE_UNAVAILABLE,
                ..
            })) if attempt + 1 < HEALTH_ATTEMPTS => {
                tokio::time::sleep(delay).await;
                delay = delay.saturating_mul(2);
            }
            Err(error) => return Err(codex_api::map_api_error(classify_api_error(error))),
        }
    }
    Err(CodexErr::Stream(
        "llama.cpp did not become ready within the bounded health retry window".to_string(),
    ))
}

pub(super) async fn execute_json<T: for<'de> Deserialize<'de>>(
    transport: &ReqwestTransport,
    provider: &ApiProvider,
    method: Method,
    path: &str,
    body: Option<Value>,
) -> std::result::Result<T, ApiError> {
    let mut request = provider.build_request(method, path);
    request.body = body.map(RequestBody::Json);
    request.timeout = Some(CONTROL_REQUEST_TIMEOUT);
    let response = transport
        .execute(request)
        .await
        .map_err(ApiError::Transport)?;
    serde_json::from_slice(&response.body).map_err(|error| ApiError::InvalidRequest {
        message: format!("invalid llama.cpp JSON response from `{path}`: {error}"),
    })
}

pub(super) fn classify_api_error(error: ApiError) -> ApiError {
    match error {
        ApiError::Transport(error) => classify_transport_error(error),
        ApiError::ServerOverloaded => ApiError::Retryable {
            message: "llama.cpp service unavailable".to_string(),
            delay: None,
        },
        ApiError::Stream(message) | ApiError::Retryable { message, .. }
            if deterministic_failure(&message) =>
        {
            ApiError::InvalidRequest { message }
        }
        ApiError::Stream(message) => ApiError::Retryable {
            message,
            delay: None,
        },
        error => error,
    }
}

pub(super) fn classify_transport_error(error: codex_api::TransportError) -> ApiError {
    match error {
        codex_api::TransportError::Http {
            status,
            url: _,
            headers: _,
            body,
        } => {
            let message = llama_error_message(body.as_deref().unwrap_or_default());
            if status == StatusCode::BAD_REQUEST
                || (status == StatusCode::INTERNAL_SERVER_ERROR && deterministic_failure(&message))
                || status.is_client_error()
            {
                ApiError::InvalidRequest { message }
            } else {
                ApiError::Retryable {
                    message: format!("llama.cpp request failed with HTTP {status}: {message}"),
                    delay: None,
                }
            }
        }
        codex_api::TransportError::Timeout | codex_api::TransportError::Network(_) => {
            ApiError::Retryable {
                message: error.to_string(),
                delay: None,
            }
        }
        codex_api::TransportError::RetryLimit => ApiError::Retryable {
            message: "llama.cpp transport retry limit reached".to_string(),
            delay: None,
        },
        codex_api::TransportError::Build(message) => ApiError::InvalidRequest { message },
    }
}

fn llama_error_message(body: &str) -> String {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .filter(|message| !message.trim().is_empty())
        .unwrap_or_else(|| body.to_string())
}

fn deterministic_failure(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    [
        "system message must be at the beginning",
        "system message must be the first",
        "previous_response_id",
        "cannot determine type of 'item'",
        "invalid request",
        "template error",
    ]
    .iter()
    .any(|needle| message.contains(needle))
}

pub(super) fn invalidates_discovery(error: &ApiError) -> bool {
    matches!(
        error,
        ApiError::Retryable { .. }
            | ApiError::Stream(_)
            | ApiError::ServerOverloaded
            | ApiError::Transport(codex_api::TransportError::Timeout)
            | ApiError::Transport(codex_api::TransportError::Network(_))
    )
}
