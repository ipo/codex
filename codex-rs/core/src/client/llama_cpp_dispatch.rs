use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::task::Context;
use std::task::Poll;
use std::time::Duration;

use codex_api::ApiError;
use codex_http_client::HttpTransport;
use codex_http_client::RequestBody;
use codex_model_provider::unauthenticated_auth_provider;
use codex_model_provider_info::ResolvedWireRoute;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::model_inference::LlamaCppInferenceConfig;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::models::render_plaintext_agent_message;
use codex_protocol::openai_models::ReasoningEffort;
use codex_tools::ToolSpec;
use futures::Stream;
use http::Method;
use http::StatusCode;
use serde::Deserialize;
use serde_json::Value;
use serde_json::json;
use tokio::sync::OwnedSemaphorePermit;
use tokio::sync::Semaphore;

use crate::context::AdditionalContextUserFragment;
use crate::context::ContextualUserFragment;

use super::*;

mod control;
mod request;

use self::control::*;
use self::request::*;

const RESPONSES_ENDPOINT: &str = "/responses";
const HEALTH_ATTEMPTS: u32 = 5;
const HEALTH_INITIAL_DELAY: Duration = Duration::from_millis(250);
const CONTROL_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

pub(super) struct LlamaCppPlan {
    pub config: LlamaCppInferenceConfig,
    pub route: ResolvedWireRoute,
}

struct LlamaCppRuntime {
    inference_slot: Arc<Semaphore>,
    discovered_models: Mutex<HashMap<String, String>>,
}

impl LlamaCppRuntime {
    fn global() -> &'static Self {
        static RUNTIME: OnceLock<LlamaCppRuntime> = OnceLock::new();
        RUNTIME.get_or_init(|| Self {
            inference_slot: Arc::new(Semaphore::new(1)),
            discovered_models: Mutex::new(HashMap::new()),
        })
    }

    fn discovered_model(&self, endpoint: &str) -> Result<Option<String>> {
        self.discovered_models
            .lock()
            .map_err(|_| CodexErr::InvalidRequest("llama.cpp discovery cache is poisoned".into()))
            .map(|models| models.get(endpoint).cloned())
    }

    fn cache_model(&self, endpoint: String, model: String) -> Result<()> {
        self.discovered_models
            .lock()
            .map_err(|_| CodexErr::InvalidRequest("llama.cpp discovery cache is poisoned".into()))?
            .insert(endpoint, model);
        Ok(())
    }

    fn invalidate(&self, endpoint: &str) {
        if let Ok(mut models) = self.discovered_models.lock() {
            models.remove(endpoint);
        }
    }
}

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<DiscoveredModel>,
}

#[derive(Deserialize)]
struct DiscoveredModel {
    id: String,
}

#[derive(Deserialize)]
struct InputTokensResponse {
    input_tokens: u32,
}

impl ModelClientSession {
    #[allow(clippy::too_many_arguments)]
    #[instrument(
        name = "model_client.stream_llama_cpp_responses",
        level = "info",
        skip_all,
        fields(
            model = %model_info.slug,
            wire_api = "responses",
            transport = "responses_http",
            http.method = "POST",
            api.path = "responses"
        )
    )]
    pub(super) async fn stream_llama_cpp(
        &self,
        prompt: &Prompt,
        model_info: &ModelInfo,
        session_telemetry: &SessionTelemetry,
        effort: Option<ReasoningEffortConfig>,
        _responses_metadata: &CodexResponsesMetadata,
        inference_trace: &InferenceTraceContext,
        plan: LlamaCppPlan,
    ) -> Result<ResponseStream> {
        validate_route(&plan.route)?;
        validate_config(&plan.config)?;
        validate_tools(prompt)?;
        if prompt.output_schema.is_some() {
            return Err(CodexErr::InvalidRequest(
                "structured output is unsupported by direct llama.cpp Responses".to_string(),
            ));
        }

        let base_provider = self.client.state.provider.api_provider().await?;
        let api_provider = provider_for_route(base_provider, &plan.route)?;
        let endpoint_key = api_provider.base_url.clone();
        let transport = self
            .client
            .build_api_transport(&api_provider, RESPONSES_ENDPOINT)?;
        let input = normalize_input(prompt)?;
        let wire_model = ensure_discovered_model(
            &transport,
            &api_provider,
            &endpoint_key,
            &plan.config.expected_model_basename,
        )
        .await?;
        let effort = local_effort(effort.or_else(|| model_info.default_reasoning_level.clone()))?;
        let mut request = ResponsesApiRequest {
            model: wire_model,
            instructions: prompt.base_instructions.text.clone(),
            input,
            tools: Some(create_tools_raw_json_for_responses_api(&prompt.tools)?.into()),
            tool_choice: "auto".to_string(),
            parallel_tool_calls: prompt.parallel_tool_calls,
            reasoning: Some(Reasoning {
                effort: Some(effort.clone()),
                summary: None,
                context: None,
            }),
            store: false,
            stream: true,
            stream_options: None,
            include: Vec::new(),
            service_tier: None,
            prompt_cache_key: None,
            text: None,
            client_metadata: None,
        };
        self.client
            .prepare_response_items_for_request(&mut request.input);
        request
            .input
            .iter_mut()
            .for_each(ResponseItem::clear_internal_chat_message_metadata_passthrough);
        let request_session_telemetry = session_telemetry_for_request(session_telemetry, &request);
        let request_body = llama_cpp_request_body(&request, &plan.config, &effort)?;

        let input_tokens = execute_json::<InputTokensResponse>(
            &transport,
            &api_provider,
            Method::POST,
            "responses/input_tokens",
            Some(request_body.clone()),
        )
        .await
        .map_err(|error| {
            let error = classify_api_error(error);
            if invalidates_discovery(&error) {
                LlamaCppRuntime::global().invalidate(&endpoint_key);
            }
            codex_api::map_api_error(error)
        })?
        .input_tokens;
        if input_tokens > plan.config.max_input_tokens {
            return Err(CodexErr::ContextWindowExceeded);
        }

        let permit = LlamaCppRuntime::global()
            .inference_slot
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| CodexErr::Stream("llama.cpp inference queue closed".to_string()))?;
        let api_auth = unauthenticated_auth_provider();
        let request_auth_context = AuthRequestTelemetryContext::new(
            None,
            api_auth.as_ref(),
            None,
            PendingUnauthorizedRetry::default(),
        );
        let (request_telemetry, sse_telemetry) = Self::build_streaming_telemetry(
            session_telemetry,
            request_auth_context,
            RequestRouteTelemetry::for_endpoint(RESPONSES_ENDPOINT),
            self.client.state.auth_env_telemetry.clone(),
        );
        let inference_trace_attempt = inference_trace.start_attempt();
        inference_trace_attempt.record_started(&request_body);
        let client = ApiResponsesClient::new(transport, api_provider, api_auth)
            .with_telemetry(Some(request_telemetry), Some(sse_telemetry));
        let api_stream = match client
            .stream(
                request_body,
                ApiHeaderMap::new(),
                Compression::None,
                /*turn_state*/ None,
            )
            .await
        {
            Ok(stream) => stream,
            Err(error) => {
                drop(permit);
                let error = classify_api_error(error);
                if invalidates_discovery(&error) {
                    LlamaCppRuntime::global().invalidate(&endpoint_key);
                }
                let error = self.client.state.provider.map_api_error(error);
                inference_trace_attempt.record_failed(
                    &error,
                    /*upstream_request_id*/ None,
                    /*output_items*/ &[],
                );
                return Err(error);
            }
        };
        let codex_api::ResponseStream {
            rx_event,
            upstream_request_id,
        } = api_stream;
        let guarded_stream = LeaseGuardedStream {
            stream: codex_api::ResponseStream {
                rx_event,
                upstream_request_id: None,
            },
            permit: Some(permit),
            endpoint_key,
        };
        Ok(map_response_events(
            upstream_request_id,
            guarded_stream,
            request_session_telemetry,
            inference_trace_attempt,
            Arc::clone(&self.client.state.provider),
        )
        .0)
    }
}

struct LeaseGuardedStream {
    stream: codex_api::ResponseStream,
    permit: Option<OwnedSemaphorePermit>,
    endpoint_key: String,
}

impl Stream for LeaseGuardedStream {
    type Item = std::result::Result<ResponseEvent, ApiError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.stream).poll_next(cx) {
            Poll::Ready(Some(Ok(event))) => {
                if matches!(event, ResponseEvent::Completed { .. }) {
                    self.permit.take();
                }
                Poll::Ready(Some(Ok(event)))
            }
            Poll::Ready(Some(Err(error))) => {
                let error = classify_api_error(error);
                if invalidates_discovery(&error) {
                    LlamaCppRuntime::global().invalidate(&self.endpoint_key);
                }
                self.permit.take();
                Poll::Ready(Some(Err(error)))
            }
            Poll::Ready(None) => {
                if self.permit.take().is_some() {
                    LlamaCppRuntime::global().invalidate(&self.endpoint_key);
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
#[path = "llama_cpp_dispatch_tests.rs"]
mod tests;
