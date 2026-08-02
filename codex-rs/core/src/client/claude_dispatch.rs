use super::*;
use codex_claude_code::CanonicalOutputSchema;
use codex_claude_code::ClaudeHttpAdapter;
use codex_claude_code::EncodeRequest;
use codex_claude_code::NativeStreamError;
use codex_claude_code::SystemBlock;
use codex_claude_code::encode_request;
use codex_model_provider_info::ResolvedWireRoute;
use codex_protocol::model_inference::AnthropicThinkingPolicy;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::model_inference::ModelInferenceConfig;
use futures::TryStreamExt;

const CLAUDE_MESSAGES_ENDPOINT: &str = "/v1/messages";

pub(super) struct ClaudePlan {
    pub wire_model: String,
    pub max_output_tokens: u32,
    pub thinking: AnthropicThinkingPolicy,
    pub supports_disabled_thinking: bool,
    pub route: ResolvedWireRoute,
}

impl ModelClientSession {
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn stream_claude(
        &self,
        prompt: &Prompt,
        model_info: &ModelInfo,
        session_telemetry: &SessionTelemetry,
        effort: Option<ReasoningEffortConfig>,
        responses_metadata: &CodexResponsesMetadata,
        inference_trace: &InferenceTraceContext,
        plan: ClaudePlan,
    ) -> Result<ResponseStream> {
        validate_route(&plan.route)?;
        let profile = ModelInferenceConfig::Anthropic {
            wire_api: plan.route.wire_api,
            dialect: plan.route.dialect,
            route: plan.route.name.clone().unwrap_or_default(),
            wire_model: plan.wire_model,
            max_output_tokens: plan.max_output_tokens,
            thinking: plan.thinking,
            supports_disabled_thinking: plan.supports_disabled_thinking,
        };
        let effort = effort
            .or_else(|| model_info.default_reasoning_level.clone())
            .ok_or_else(|| {
                CodexErr::InvalidRequest(format!(
                    "native Claude model `{}` requires an explicit reasoning effort",
                    model_info.slug
                ))
            })?;
        let (system, history) = native_system_and_history(prompt)?;
        let output_schema = prompt
            .output_schema
            .as_ref()
            .map_or(CanonicalOutputSchema::Disabled, |schema| {
                CanonicalOutputSchema::JsonSchema { schema }
            });
        let request = encode_request(EncodeRequest {
            profile: &profile,
            effort: &effort,
            system: &system,
            history: &history,
            tools: &prompt.tools,
            output_schema,
            resumable_session_id: &responses_metadata.session_id.to_string(),
            codex_version: env!("CARGO_PKG_VERSION"),
        })
        .map_err(|error| CodexErr::InvalidRequest(error.to_string()))?;

        let client_setup = self.client.current_client_setup().await?;
        let api_provider = provider_for_route(client_setup.api_provider, &plan.route)?;
        let transport = self
            .client
            .build_api_transport(&api_provider, CLAUDE_MESSAGES_ENDPOINT)?;
        let request_auth_context = AuthRequestTelemetryContext::new(
            client_setup.auth.as_ref().map(CodexAuth::auth_mode),
            client_setup.api_auth.as_ref(),
            client_setup.agent_identity_telemetry,
            PendingUnauthorizedRetry::default(),
        );
        let request_telemetry = ModelClient::build_request_telemetry(
            session_telemetry,
            request_auth_context,
            RequestRouteTelemetry::for_endpoint(CLAUDE_MESSAGES_ENDPOINT),
            self.client.state.auth_env_telemetry.clone(),
        );
        let inference_trace_attempt = inference_trace.start_attempt();
        inference_trace_attempt.record_started(&request);
        let cancellation = CancellationToken::new();
        let stream = ClaudeHttpAdapter::new(transport, api_provider, client_setup.api_auth)
            .with_telemetry(Some(request_telemetry))
            .stream_request(request, cancellation.clone())
            .await
            .map_err(map_native_error)
            .map_err(|error| self.client.state.provider.map_api_error(error))?;
        let upstream_request_id = stream.upstream_request_id.clone();
        let stream = stream.map_err(map_native_error);
        Ok(map_response_events_with_cancellation(
            upstream_request_id,
            stream,
            session_telemetry.clone(),
            inference_trace_attempt,
            Arc::clone(&self.client.state.provider),
            cancellation,
        )
        .0)
    }
}

fn native_system_and_history(prompt: &Prompt) -> Result<(Vec<SystemBlock>, Vec<ResponseItem>)> {
    let mut system = (!prompt.base_instructions.text.is_empty())
        .then(|| SystemBlock::Text {
            text: prompt.base_instructions.text.clone(),
            cache_control: None,
        })
        .into_iter()
        .collect::<Vec<_>>();
    let mut history = Vec::with_capacity(prompt.input.len());
    for (index, item) in prompt.input.iter().enumerate() {
        let ResponseItem::Message { role, content, .. } = item else {
            history.push(item.clone());
            continue;
        };
        if !matches!(role.as_str(), "developer" | "system") {
            history.push(item.clone());
            continue;
        }
        let mut text = String::new();
        for block in content {
            match block {
                ContentItem::InputText { text: block }
                | ContentItem::OutputText { text: block } => text.push_str(block),
                ContentItem::InputImage { .. } | ContentItem::InputAudio { .. } => {
                    return Err(CodexErr::InvalidRequest(format!(
                        "native Claude system message at history index {index} must contain only text"
                    )));
                }
            }
        }
        system.push(SystemBlock::Text {
            text,
            cache_control: None,
        });
    }
    Ok((system, history))
}

fn validate_route(route: &ResolvedWireRoute) -> Result<()> {
    let expected_query = HashMap::from([("beta".to_string(), "true".to_string())]);
    if route.wire_api != WireApi::AnthropicMessages
        || route.dialect != InferenceDialect::ClaudeCode
        || route.request_path.trim_matches('/') != "v1/messages"
        || route.query_params.as_ref() != Some(&expected_query)
    {
        return Err(CodexErr::InvalidRequest(format!(
            "Claude route `{}` must resolve to anthropic_messages/claude_code at `v1/messages?beta=true`",
            route.name.as_deref().unwrap_or("<legacy>")
        )));
    }
    Ok(())
}

fn provider_for_route(mut provider: ApiProvider, route: &ResolvedWireRoute) -> Result<ApiProvider> {
    provider.base_url = route.base_url.clone().ok_or_else(|| {
        CodexErr::InvalidRequest("native Claude route requires a base URL".to_string())
    })?;
    provider.query_params = None;
    provider.retry.max_attempts = route.request_max_retries;
    provider.stream_idle_timeout = route.stream_idle_timeout;
    Ok(provider)
}

fn map_native_error(error: NativeStreamError) -> ApiError {
    match error {
        NativeStreamError::Request(error) => error,
        NativeStreamError::InvalidRequest(message) => ApiError::InvalidRequest { message },
        NativeStreamError::Cancelled => ApiError::Stream("native Claude stream cancelled".into()),
        NativeStreamError::IdleTimeout => {
            ApiError::Stream("native Claude stream idle timeout".into())
        }
        NativeStreamError::Transport(message) => ApiError::Stream(message),
        NativeStreamError::Decode(error) => ApiError::Stream(error.to_string()),
    }
}
