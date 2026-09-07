use codex_api::LlamaCppRequestInput;
use codex_api::LlamaCppRuntime;
use codex_model_provider_info::LLAMA_CPP_ROUTE_NAME;
use codex_model_provider_info::ResolvedWireRoute;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::model_inference::LlamaCppInferenceConfig;

use super::*;

mod request;

use self::request::normalize_input;
use self::request::validate_tools;

impl ModelClientSession {
    #[tracing::instrument(
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
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn stream_llama_cpp(
        &self,
        prompt: &Prompt,
        model_info: &ModelInfo,
        session_telemetry: &SessionTelemetry,
        effort: Option<ReasoningEffortConfig>,
        _summary: ReasoningSummaryConfig,
        _service_tier: Option<String>,
        _responses_metadata: &CodexResponsesMetadata,
        inference_trace: &InferenceTraceContext,
        config: LlamaCppInferenceConfig,
        route: ResolvedWireRoute,
    ) -> Result<ResponseStream> {
        validate_route(&route)?;
        if prompt.output_schema.is_some() {
            return Err(CodexErr::InvalidRequest(
                "structured output is unsupported by direct llama.cpp Responses".to_string(),
            ));
        }
        validate_tools(prompt)?;

        let request_url = route.base_url.as_deref().ok_or_else(|| {
            CodexErr::InvalidRequest("llama.cpp route requires a base URL".to_string())
        })?;
        let http_client = create_client_for_route(
            &self.client.http_client_factory,
            request_url,
            ClientRouteClass::Api,
        )
        .map_err(std::io::Error::from)?;
        let mut history = normalize_input(prompt)?;
        self.client.prepare_response_items_for_request(&mut history);
        let input = LlamaCppRequestInput {
            instructions: prompt.base_instructions.text.clone(),
            history,
            tools: Some(create_tools_raw_json_for_responses_api(&prompt.tools)?.into()),
            parallel_tool_calls: prompt.parallel_tool_calls,
            reasoning_effort: effort.or_else(|| model_info.default_reasoning_level.clone()),
        };
        let runtime = LlamaCppRuntime::new(http_client);
        let prepared = runtime
            .prepare(&model_info.slug, input)
            .await
            .map_err(|error| self.client.state.provider.map_api_error(error))?;
        if prepared.model.display_name != config.expected_model_basename {
            return Err(CodexErr::InvalidRequest(format!(
                "llama.cpp model metadata expected `{}` but discovery selected `{}`",
                config.expected_model_basename, prepared.model.display_name
            )));
        }

        let inference_trace_attempt = inference_trace.start_attempt();
        inference_trace_attempt.record_started(prepared.body());
        let api_stream = runtime.stream(prepared).await.map_err(|error| {
            let mapped = self.client.state.provider.map_api_error(error);
            inference_trace_attempt.record_failed(
                &mapped,
                /*upstream_request_id*/ None,
                /*output_items*/ &[],
            );
            mapped
        })?;
        Ok(map_response_stream(
            api_stream,
            session_telemetry.clone(),
            inference_trace_attempt,
            Arc::clone(&self.client.state.provider),
        )
        .0)
    }
}

#[cfg(test)]
#[path = "llama_cpp_dispatch_tests.rs"]
mod tests;

fn validate_route(route: &ResolvedWireRoute) -> Result<()> {
    let expected_base_url = format!("{}/v1", codex_api::LLAMA_CPP_LOCAL_ENDPOINT);
    if route.name.as_deref() != Some(LLAMA_CPP_ROUTE_NAME)
        || route.wire_api != WireApi::Responses
        || route.dialect != InferenceDialect::LlamaCpp
        || route.base_url.as_deref() != Some(expected_base_url.as_str())
        || route.request_path.trim_matches('/') != "responses"
        || route.query_params.is_some()
        || route.request_max_retries != 0
    {
        return Err(CodexErr::InvalidRequest(
            "llama.cpp must use the managed direct Responses route without query parameters or transport retries"
                .to_string(),
        ));
    }
    Ok(())
}
