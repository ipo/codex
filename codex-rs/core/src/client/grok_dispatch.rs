use super::*;
use codex_model_provider_info::ResolvedWireRoute;
use codex_protocol::model_inference::GrokInferenceConfig;
use codex_protocol::model_inference::InferenceDialect;

const GROK_RESPONSES_ENDPOINT: &str = "/responses";

pub(super) struct GrokPlan {
    pub config: GrokInferenceConfig,
    pub route: ResolvedWireRoute,
}

impl ModelClientSession {
    #[allow(clippy::too_many_arguments)]
    #[instrument(
        name = "model_client.stream_grok_responses",
        level = "info",
        skip_all,
        fields(
            model = %plan.config.wire_model,
            wire_api = "responses",
            transport = "responses_http",
            http.method = "POST",
            api.path = "responses"
        )
    )]
    pub(super) async fn stream_grok(
        &self,
        prompt: &Prompt,
        model_info: &ModelInfo,
        session_telemetry: &SessionTelemetry,
        effort: Option<ReasoningEffortConfig>,
        summary: ReasoningSummaryConfig,
        responses_metadata: &CodexResponsesMetadata,
        inference_trace: &InferenceTraceContext,
        plan: GrokPlan,
    ) -> Result<ResponseStream> {
        validate_route(&plan.route)?;
        let client_setup = self.client.current_client_setup().await?;
        let api_provider = provider_for_route(client_setup.api_provider, &plan.route)?;
        let transport = self
            .client
            .build_api_transport(&api_provider, GROK_RESPONSES_ENDPOINT)?;
        let request_auth_context = AuthRequestTelemetryContext::new(
            client_setup.auth.as_ref().map(CodexAuth::auth_mode),
            client_setup.api_auth.as_ref(),
            client_setup.agent_identity_telemetry,
            PendingUnauthorizedRetry::default(),
        );
        let (request_telemetry, sse_telemetry) = Self::build_streaming_telemetry(
            session_telemetry,
            request_auth_context,
            RequestRouteTelemetry::for_endpoint(GROK_RESPONSES_ENDPOINT),
            self.client.state.auth_env_telemetry.clone(),
        );

        let mut request = ResponsesApiRequest {
            model: plan.config.wire_model.clone(),
            instructions: prompt.base_instructions.text.clone(),
            input: prompt.get_formatted_input_for_request(/*use_responses_lite*/ false),
            tools: Some(create_tools_raw_json_for_responses_api(&prompt.tools)?.into()),
            tool_choice: "auto".to_string(),
            parallel_tool_calls: prompt.parallel_tool_calls,
            reasoning: Some(Reasoning {
                effort: effort
                    .or_else(|| model_info.default_reasoning_level.clone())
                    .map(reasoning_effort_for_request),
                summary: (model_info.supports_reasoning_summary_parameter
                    && summary != ReasoningSummaryConfig::None)
                    .then_some(summary),
                context: None,
            }),
            store: false,
            stream: true,
            stream_options: None,
            include: vec!["reasoning.encrypted_content".to_string()],
            service_tier: None,
            prompt_cache_key: Some(self.client.prompt_cache_key(responses_metadata)),
            text: create_text_param_for_request(
                /*verbosity*/ None,
                &prompt.output_schema,
                prompt.output_schema_strict,
            ),
            client_metadata: None,
        };
        self.client
            .prepare_response_items_for_request(&mut request.input);
        request
            .input
            .iter_mut()
            .for_each(ResponseItem::clear_internal_chat_message_metadata_passthrough);

        let mut extra_headers = grok_headers(responses_metadata, &plan.config.wire_model)?;
        let inference_trace_attempt = inference_trace.start_attempt();
        inference_trace_attempt.add_request_headers(&mut extra_headers);
        inference_trace_attempt.record_started(&request);
        let request_session_telemetry = session_telemetry_for_request(session_telemetry, &request);
        let client = ApiResponsesClient::new(transport, api_provider, client_setup.api_auth)
            .with_telemetry(Some(request_telemetry), Some(sse_telemetry));
        match client
            .stream_request(
                request,
                ApiResponsesOptions {
                    extra_headers,
                    turn_state: Some(Arc::clone(&self.turn_state)),
                    ..ApiResponsesOptions::default()
                },
            )
            .await
        {
            Ok(stream) => Ok(map_response_stream(
                stream,
                request_session_telemetry,
                inference_trace_attempt,
                Arc::clone(&self.client.state.provider),
            )
            .0),
            Err(error) => {
                let response_debug = extract_response_debug_context_from_api_error(&error);
                let error = self.client.state.provider.map_api_error(error);
                inference_trace_attempt.record_failed(
                    &error,
                    response_debug.request_id.as_deref(),
                    /*output_items*/ &[],
                );
                Err(error)
            }
        }
    }
}

fn grok_headers(metadata: &CodexResponsesMetadata, wire_model: &str) -> Result<ApiHeaderMap> {
    let values = [
        ("x-grok-conv-id", metadata.thread_id.as_str()),
        (
            "x-grok-req-id",
            metadata.turn_id.as_deref().unwrap_or(&metadata.thread_id),
        ),
        ("x-grok-session-id", metadata.session_id.as_str()),
        ("x-grok-agent-id", metadata.installation_id.as_str()),
        ("x-grok-model-override", wire_model),
    ];
    let mut headers = ApiHeaderMap::new();
    for (name, value) in values {
        let value = HeaderValue::from_str(value).map_err(|error| {
            CodexErr::InvalidRequest(format!("invalid Grok lineage header `{name}`: {error}"))
        })?;
        headers.insert(http::header::HeaderName::from_static(name), value);
    }
    Ok(headers)
}

fn validate_route(route: &ResolvedWireRoute) -> Result<()> {
    if route.wire_api != WireApi::Responses
        || route.dialect != InferenceDialect::Grok
        || route.request_path.trim_matches('/') != "responses"
        || route.query_params.is_some()
    {
        return Err(CodexErr::InvalidRequest(format!(
            "Grok route `{}` must resolve to responses/grok at `responses` without query parameters",
            route.name.as_deref().unwrap_or("<legacy>")
        )));
    }
    Ok(())
}

fn provider_for_route(mut provider: ApiProvider, route: &ResolvedWireRoute) -> Result<ApiProvider> {
    provider.base_url = route
        .base_url
        .clone()
        .ok_or_else(|| CodexErr::InvalidRequest("Grok route requires a base URL".to_string()))?;
    provider.query_params = None;
    provider.retry.max_attempts = 0;
    provider.stream_idle_timeout = route.stream_idle_timeout;
    Ok(provider)
}

#[cfg(test)]
#[path = "grok_dispatch_tests.rs"]
mod tests;
