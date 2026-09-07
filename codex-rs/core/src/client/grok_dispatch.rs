use super::*;
use codex_model_provider_info::ResolvedWireRoute;
use codex_protocol::model_inference::GrokInferenceConfig;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::models::ContentItem;
use codex_protocol::models::plaintext_agent_message_content;

const GROK_RESPONSES_ENDPOINT: &str = "/responses";

const GROK_CONVERSATION_ID_HEADER: &str = "x-grok-conv-id";
const GROK_REQUEST_ID_HEADER: &str = "x-grok-req-id";
const GROK_SESSION_ID_HEADER: &str = "x-grok-session-id";
const GROK_AGENT_ID_HEADER: &str = "x-grok-agent-id";
const GROK_MODEL_OVERRIDE_HEADER: &str = "x-grok-model-override";

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

        let request = self.client.build_responses_request(
            prompt,
            model_info,
            effort,
            summary,
            /*service_tier*/ None,
            responses_metadata,
        )?;
        let mut request = adapt_request(request, &plan.config.wire_model);
        project_history(&mut request.input)?;
        self.client
            .prepare_response_items_for_request(&mut request.input);

        let mut extra_headers = lineage_headers(responses_metadata, &plan.config.wire_model)?;
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

fn project_history(input: &mut [ResponseItem]) -> Result<()> {
    for (index, item) in input.iter_mut().enumerate() {
        let ResponseItem::AgentMessage {
            author,
            recipient,
            content,
            ..
        } = item
        else {
            continue;
        };
        let text = plaintext_agent_message_content(content).ok_or_else(|| {
            CodexErr::InvalidRequest(format!(
                "Grok history item at index {index} contains a non-plaintext structured agent message"
            ))
        })?;
        let text = format!("Agent message from {author} to {recipient}:\n{text}");
        *item = ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText { text }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        };
    }
    Ok(())
}

/// Applies the Grok HTTP Responses profile to a provider-neutral request.
///
/// Tool representation and cross-family history projection are intentionally supplied by their
/// owning layers. This adapter only owns Grok's request fields and encrypted-reasoning replay.
pub(super) fn adapt_request(
    mut request: ResponsesApiRequest,
    wire_model: &str,
) -> ResponsesApiRequest {
    request.model = wire_model.to_string();
    request.store = false;
    request.stream = true;
    request.stream_options = None;
    request.include = vec!["reasoning.encrypted_content".to_string()];
    request.service_tier = None;
    request.client_metadata = None;
    request.access_programs = None;

    if let Some(reasoning) = &mut request.reasoning {
        reasoning.context = None;
    }
    if let Some(text) = &mut request.text {
        text.verbosity = None;
        if text.format.is_none() {
            request.text = None;
        }
    }
    for item in &mut request.input {
        if let ResponseItem::Reasoning { content, .. } = item
            && content.is_none()
        {
            // Grok omits absent reasoning content while binding encrypted content to the exact
            // surrounding item. An empty vector uses the protocol's existing omission rule.
            *content = Some(Vec::new());
        }
        item.clear_internal_chat_message_metadata_passthrough();
    }

    request
}

pub(super) fn lineage_headers(
    metadata: &CodexResponsesMetadata,
    wire_model: &str,
) -> Result<ApiHeaderMap> {
    let values = [
        (GROK_CONVERSATION_ID_HEADER, metadata.thread_id.as_str()),
        (
            GROK_REQUEST_ID_HEADER,
            metadata.turn_id.as_deref().unwrap_or(&metadata.thread_id),
        ),
        (GROK_SESSION_ID_HEADER, metadata.session_id.as_str()),
        (GROK_AGENT_ID_HEADER, metadata.installation_id.as_str()),
        (GROK_MODEL_OVERRIDE_HEADER, wire_model),
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
