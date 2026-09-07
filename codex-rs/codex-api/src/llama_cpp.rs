use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;

use codex_client::HttpTransport;
use codex_client::RequestBody;
use codex_client::ReqwestTransport;
use codex_http_client::HttpClient;
use futures::StreamExt;
use http::HeaderMap;
use http::Method;
use http::StatusCode;
use serde::Deserialize;
use serde_json::Value;
use tokio::sync::Mutex;
use tokio::sync::Semaphore;

use crate::ApiError;
use crate::AuthProvider;
use crate::Compression;
use crate::Provider;
use crate::ResponseEvent;
use crate::ResponseStream;
use crate::ResponsesClient;
use crate::RetryConfig;

mod catalog;
mod request;

pub use catalog::LlamaCppCatalog;
pub use catalog::LlamaCppCatalogEntry;
pub use catalog::LlamaCppModelSelectionError;
pub use request::LlamaCppPreparedRequest;
pub use request::LlamaCppRequestInput;

use catalog::ModelsResponse;
use request::llama_cpp_request_body;

pub const LLAMA_CPP_LOCAL_ENDPOINT: &str = "http://desktop-uae.netbird.selfhosted:8080";

const HEALTH_ATTEMPTS: u32 = 5;
const HEALTH_INITIAL_DELAY: Duration = Duration::from_millis(250);
const CONTROL_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(300);

/// Direct llama.cpp runtime used by catalog discovery and selected local requests.
#[derive(Clone)]
pub struct LlamaCppRuntime {
    transport: ReqwestTransport,
    api_provider: Provider,
    health_provider: Provider,
    catalog: Arc<Mutex<Option<LlamaCppCatalog>>>,
    inference_lease: Arc<Semaphore>,
    readiness: ReadinessPolicy,
}

impl LlamaCppRuntime {
    pub fn new(http_client: HttpClient) -> Self {
        Self::from_parts(
            http_client,
            LLAMA_CPP_LOCAL_ENDPOINT,
            process_catalog(),
            process_inference_lease(),
        )
    }

    fn for_endpoint(
        http_client: HttpClient,
        endpoint: &str,
        inference_lease: Arc<Semaphore>,
    ) -> Self {
        Self::from_parts(
            http_client,
            endpoint,
            Arc::new(Mutex::new(None)),
            inference_lease,
        )
    }

    fn from_parts(
        http_client: HttpClient,
        endpoint: &str,
        catalog: Arc<Mutex<Option<LlamaCppCatalog>>>,
        inference_lease: Arc<Semaphore>,
    ) -> Self {
        let endpoint = endpoint.trim_end_matches('/');
        let transport = ReqwestTransport::from_http_client(http_client);
        Self {
            transport,
            api_provider: provider(format!("{endpoint}/v1")),
            health_provider: provider(endpoint.to_string()),
            catalog,
            inference_lease,
            readiness: ReadinessPolicy::default(),
        }
    }

    pub async fn catalog(&self) -> Result<LlamaCppCatalog, ApiError> {
        let mut cached = self.catalog.lock().await;
        if let Some(catalog) = cached.as_ref() {
            return Ok(catalog.clone());
        }
        self.wait_until_ready().await?;
        let response = self
            .execute_json::<ModelsResponse>(&self.api_provider, Method::GET, "models", None)
            .await?;
        let catalog = LlamaCppCatalog::from_discovery(response);
        *cached = Some(catalog.clone());
        Ok(catalog)
    }

    pub async fn invalidate_discovery(&self) {
        *self.catalog.lock().await = None;
    }

    pub async fn resolve_model(&self, requested: &str) -> Result<LlamaCppCatalogEntry, ApiError> {
        self.catalog()
            .await?
            .resolve(requested)
            .map_err(|error| ApiError::InvalidRequest {
                message: error.to_string(),
            })
    }

    pub async fn prepare(
        &self,
        requested_model: &str,
        input: LlamaCppRequestInput,
    ) -> Result<LlamaCppPreparedRequest, ApiError> {
        let model = self.resolve_model(requested_model).await?;
        let body = llama_cpp_request_body(&model, input)?;
        let count = match self
            .execute_json::<InputTokensResponse>(
                &self.api_provider,
                Method::POST,
                "responses/input_tokens",
                Some(body.clone()),
            )
            .await
        {
            Ok(count) => count,
            Err(error) => {
                let error = classify_error(error);
                if invalidates_discovery(&error) {
                    self.invalidate_discovery().await;
                }
                return Err(error);
            }
        };
        if count.input_tokens > model.max_input_tokens {
            return Err(ApiError::ContextWindowExceeded);
        }
        Ok(LlamaCppPreparedRequest { model, body })
    }

    pub async fn stream(
        &self,
        prepared: LlamaCppPreparedRequest,
    ) -> Result<ResponseStream, ApiError> {
        let permit = self
            .inference_lease
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| ApiError::Stream("llama.cpp inference queue closed".to_string()))?;
        let client = ResponsesClient::new(
            self.transport.clone(),
            self.api_provider.clone(),
            Arc::new(NoAuth),
        );
        let api_stream = match client
            .stream(
                prepared.body,
                HeaderMap::new(),
                Compression::None,
                /*turn_state*/ None,
            )
            .await
        {
            Ok(stream) => stream,
            Err(error) => {
                let error = classify_error(error);
                if invalidates_discovery(&error) {
                    self.invalidate_discovery().await;
                }
                return Err(error);
            }
        };
        let upstream_request_id = api_stream.upstream_request_id.clone();
        let catalog = Arc::clone(&self.catalog);
        let (tx_event, rx_event) = tokio::sync::mpsc::channel(1600);
        tokio::spawn(async move {
            let _permit = permit;
            let mut api_stream = api_stream;
            loop {
                let event = tokio::select! {
                    _ = tx_event.closed() => return,
                    event = api_stream.next() => event,
                };
                let Some(event) = event else {
                    break;
                };
                let event = event.map_err(classify_error);
                let completed = matches!(&event, Ok(ResponseEvent::Completed { .. }));
                let invalidated = event.as_ref().is_err_and(invalidates_discovery);
                if invalidated {
                    *catalog.lock().await = None;
                }
                if tx_event.send(event).await.is_err() {
                    return;
                }
                if completed || invalidated {
                    return;
                }
            }
            *catalog.lock().await = None;
        });
        Ok(ResponseStream {
            rx_event,
            upstream_request_id,
        })
    }

    async fn wait_until_ready(&self) -> Result<(), ApiError> {
        let mut delay = self.readiness.initial_delay;
        for attempt in 0..self.readiness.attempts {
            match self
                .execute_json::<HealthResponse>(&self.health_provider, Method::GET, "health", None)
                .await
            {
                Ok(HealthResponse { status }) if status == "ok" => return Ok(()),
                Ok(_) => {
                    return Err(ApiError::InvalidRequest {
                        message: "llama.cpp health response did not report ready".to_string(),
                    });
                }
                Err(ApiError::Transport(codex_client::TransportError::Http {
                    status: StatusCode::SERVICE_UNAVAILABLE,
                    ..
                })) if attempt + 1 < self.readiness.attempts => {
                    tokio::time::sleep(delay).await;
                    delay = delay.saturating_mul(2);
                }
                Err(error) => return Err(classify_error(error)),
            }
        }
        Err(ApiError::Retryable {
            message: "llama.cpp did not become ready within the bounded health retry window"
                .to_string(),
            delay: None,
        })
    }

    async fn execute_json<T: for<'de> Deserialize<'de>>(
        &self,
        provider: &Provider,
        method: Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<T, ApiError> {
        let mut request = provider.build_request(method, path);
        request.body = body.map(RequestBody::Json);
        request.timeout = Some(CONTROL_REQUEST_TIMEOUT);
        let response = self
            .transport
            .execute(request)
            .await
            .map_err(ApiError::Transport)?;
        serde_json::from_slice(&response.body).map_err(|error| ApiError::InvalidRequest {
            message: format!("invalid llama.cpp JSON response from `{path}`: {error}"),
        })
    }
}

fn process_catalog() -> Arc<Mutex<Option<LlamaCppCatalog>>> {
    static CATALOG: OnceLock<Arc<Mutex<Option<LlamaCppCatalog>>>> = OnceLock::new();
    Arc::clone(CATALOG.get_or_init(|| Arc::new(Mutex::new(None))))
}

fn process_inference_lease() -> Arc<Semaphore> {
    static LEASE: OnceLock<Arc<Semaphore>> = OnceLock::new();
    Arc::clone(LEASE.get_or_init(|| Arc::new(Semaphore::new(1))))
}

fn provider(base_url: String) -> Provider {
    Provider {
        name: "llama.cpp local".to_string(),
        base_url,
        query_params: None,
        headers: HeaderMap::new(),
        retry: RetryConfig {
            max_attempts: 0,
            base_delay: Duration::from_millis(250),
            retry_429: false,
            retry_5xx: false,
            retry_transport: false,
        },
        stream_idle_timeout: STREAM_IDLE_TIMEOUT,
    }
}

fn classify_error(error: ApiError) -> ApiError {
    match error {
        ApiError::Transport(codex_client::TransportError::Http {
            status,
            url: _,
            headers: _,
            body,
        }) => {
            let message = llama_error_message(body.as_deref().unwrap_or_default());
            if status.is_client_error()
                || (status == StatusCode::INTERNAL_SERVER_ERROR && deterministic_failure(&message))
            {
                ApiError::InvalidRequest { message }
            } else {
                ApiError::Retryable {
                    message: format!("llama.cpp request failed with HTTP {status}: {message}"),
                    delay: None,
                }
            }
        }
        ApiError::Transport(
            error @ (codex_client::TransportError::Timeout
            | codex_client::TransportError::Connection(_)
            | codex_client::TransportError::Network(_)),
        ) => ApiError::Retryable {
            message: error.to_string(),
            delay: None,
        },
        ApiError::Transport(codex_client::TransportError::RetryLimit) => ApiError::Retryable {
            message: "llama.cpp transport retry limit reached".to_string(),
            delay: None,
        },
        ApiError::Transport(codex_client::TransportError::Build(message)) => {
            ApiError::InvalidRequest { message }
        }
        ApiError::Stream(message) | ApiError::Retryable { message, .. }
            if deterministic_failure(&message) =>
        {
            ApiError::InvalidRequest { message }
        }
        ApiError::Stream(message) => ApiError::Retryable {
            message,
            delay: None,
        },
        ApiError::ServerOverloaded => ApiError::Retryable {
            message: "llama.cpp service unavailable".to_string(),
            delay: None,
        },
        error => error,
    }
}

fn invalidates_discovery(error: &ApiError) -> bool {
    matches!(
        error,
        ApiError::Retryable { .. }
            | ApiError::Stream(_)
            | ApiError::ServerOverloaded
            | ApiError::Transport(codex_client::TransportError::Timeout)
            | ApiError::Transport(codex_client::TransportError::Connection(_))
            | ApiError::Transport(codex_client::TransportError::Network(_))
    )
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

#[derive(Debug, Clone, Copy)]
struct ReadinessPolicy {
    attempts: u32,
    initial_delay: Duration,
}

impl Default for ReadinessPolicy {
    fn default() -> Self {
        Self {
            attempts: HEALTH_ATTEMPTS,
            initial_delay: HEALTH_INITIAL_DELAY,
        }
    }
}

#[derive(Debug, Deserialize)]
struct HealthResponse {
    status: String,
}

#[derive(Debug, Deserialize)]
struct InputTokensResponse {
    input_tokens: u64,
}

#[derive(Debug)]
struct NoAuth;

impl AuthProvider for NoAuth {
    fn add_auth_headers(&self, _headers: &mut HeaderMap) {}
}

#[cfg(test)]
#[path = "llama_cpp_tests.rs"]
mod tests;
