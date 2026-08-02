use super::*;
use codex_kimi_code::KimiDialect;
use codex_kimi_code::KimiEncodeRequest;
use codex_kimi_code::KimiHttpAdapter;
use codex_kimi_code::KimiInputEstimate;
use codex_kimi_code::KimiRequestSettings;
use codex_kimi_code::KimiThinking;
use codex_kimi_code::KimiThinkingEffort;
use codex_kimi_code::encode_request;
use codex_model_provider_info::ResolvedWireRoute;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::model_inference::KimiInferenceConfig;
use futures::TryStreamExt;

use crate::sampling_retry::classify_kimi_error;

const KIMI_CHAT_ENDPOINT: &str = "/chat/completions";

pub(super) struct KimiPlan {
    pub config: KimiInferenceConfig,
    pub route: ResolvedWireRoute,
}

impl ModelClientSession {
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn stream_kimi(
        &self,
        prompt: &Prompt,
        model_info: &ModelInfo,
        session_telemetry: &SessionTelemetry,
        effort: Option<ReasoningEffortConfig>,
        responses_metadata: &CodexResponsesMetadata,
        inference_trace: &InferenceTraceContext,
        plan: KimiPlan,
    ) -> Result<ResponseStream> {
        validate_route(&plan.route)?;
        if prompt.output_schema.is_some() {
            return Err(CodexErr::InvalidRequest(
                "structured output is unsupported by native Kimi Chat Completions".to_string(),
            ));
        }
        let (system, history) = native_system_and_history(prompt)?;
        let context_window = context_window(model_info)?;
        let dialect = Arc::new(
            KimiDialect::new(
                plan.config,
                KimiRequestSettings {
                    context_window,
                    input_estimate: KimiInputEstimate::FinalSerialized,
                    prompt_cache_key: self.client.prompt_cache_key(responses_metadata),
                    thinking: thinking(
                        effort.or_else(|| model_info.default_reasoning_level.clone()),
                    )?,
                },
            )
            .map_err(|error| CodexErr::InvalidRequest(error.to_string()))?,
        );
        let request = encode_request(
            dialect.as_ref(),
            KimiEncodeRequest {
                system: system.as_deref(),
                history: &history,
                tools: &prompt.tools,
            },
        )
        .map_err(|error| CodexErr::InvalidRequest(error.to_string()))?;

        let client_setup = self.client.current_client_setup().await?;
        let api_provider = provider_for_route(client_setup.api_provider, &plan.route)?;
        let transport = self
            .client
            .build_api_transport(&api_provider, KIMI_CHAT_ENDPOINT)?;
        let request_auth_context = AuthRequestTelemetryContext::new(
            client_setup.auth.as_ref().map(CodexAuth::auth_mode),
            client_setup.api_auth.as_ref(),
            client_setup.agent_identity_telemetry,
            PendingUnauthorizedRetry::default(),
        );
        let request_telemetry = ModelClient::build_request_telemetry(
            session_telemetry,
            request_auth_context,
            RequestRouteTelemetry::for_endpoint(KIMI_CHAT_ENDPOINT),
            self.client.state.auth_env_telemetry.clone(),
        );
        let inference_trace_attempt = inference_trace.start_attempt();
        inference_trace_attempt.record_started(&request);
        let cancellation = CancellationToken::new();
        let stream = KimiHttpAdapter::new(transport, api_provider, client_setup.api_auth)
            .with_telemetry(Some(request_telemetry))
            .stream_request(request, dialect, cancellation.clone())
            .await
            .map_err(classify_kimi_error)
            .map_err(|error| self.client.state.provider.map_api_error(error))?;
        let trace_id = stream.metadata.trace_id.clone();
        let stream = stream.map_err(classify_kimi_error);
        Ok(map_response_events_with_cancellation(
            trace_id,
            stream,
            session_telemetry.clone(),
            inference_trace_attempt,
            Arc::clone(&self.client.state.provider),
            cancellation,
        )
        .0)
    }
}

fn thinking(effort: Option<ReasoningEffortConfig>) -> Result<KimiThinking> {
    let Some(effort) = effort else {
        return Ok(KimiThinking::Enabled);
    };
    let effort = match effort {
        ReasoningEffortConfig::Low => KimiThinkingEffort::Low,
        ReasoningEffortConfig::High => KimiThinkingEffort::High,
        ReasoningEffortConfig::Max => KimiThinkingEffort::Max,
        effort => {
            return Err(CodexErr::InvalidRequest(format!(
                "native Kimi thinking effort `{effort}` is unsupported"
            )));
        }
    };
    Ok(KimiThinking::Effort(effort))
}

fn native_system_and_history(prompt: &Prompt) -> Result<(Option<String>, Vec<ResponseItem>)> {
    let mut system = (!prompt.base_instructions.text.is_empty())
        .then(|| prompt.base_instructions.text.clone())
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
                        "native Kimi system message at history index {index} must contain only text"
                    )));
                }
            }
        }
        system.push(text);
    }
    Ok(((!system.is_empty()).then(|| system.join("\n\n")), history))
}

pub(crate) fn estimated_input_tokens(prompt: &Prompt, model_info: &ModelInfo) -> Result<u64> {
    let Some(codex_protocol::model_inference::ModelInferenceConfig::Kimi(config)) =
        model_info.inference.clone()
    else {
        return Err(CodexErr::InvalidRequest(
            "missing Kimi inference metadata".to_string(),
        ));
    };
    let context_window = context_window(model_info)?;
    let (system, history) = native_system_and_history(prompt)?;
    let dialect = KimiDialect::new(
        config,
        KimiRequestSettings {
            context_window,
            input_estimate: KimiInputEstimate::FinalSerialized,
            prompt_cache_key: "context-estimate".to_string(),
            thinking: thinking(model_info.default_reasoning_level.clone())?,
        },
    )
    .map_err(|error| CodexErr::InvalidRequest(error.to_string()))?;
    let request = encode_request(
        &dialect,
        KimiEncodeRequest {
            system: system.as_deref(),
            history: &history,
            tools: &prompt.tools,
        },
    )
    .map_err(|error| CodexErr::InvalidRequest(error.to_string()))?;
    Ok(codex_kimi_code::estimated_input_tokens(&request))
}

pub(crate) fn should_compact(model_info: &ModelInfo, estimated_input_tokens: u64) -> bool {
    let Ok(context_window) = context_window(model_info) else {
        return false;
    };
    at_least_85_percent(context_window, estimated_input_tokens)
        || under_50k_remaining(context_window, estimated_input_tokens)
}

fn context_window(model_info: &ModelInfo) -> Result<u64> {
    model_info
        .resolved_context_window()
        .and_then(|tokens| u64::try_from(tokens).ok())
        .filter(|tokens| *tokens > 0)
        .ok_or_else(|| CodexErr::InvalidRequest("invalid Kimi context window".to_string()))
}

pub(super) fn at_least_85_percent(context_window: u64, estimated_input_tokens: u64) -> bool {
    estimated_input_tokens.saturating_mul(100) >= context_window.saturating_mul(85)
}

pub(super) fn under_50k_remaining(context_window: u64, estimated_input_tokens: u64) -> bool {
    context_window.saturating_sub(estimated_input_tokens) < 50_000
}

fn validate_route(route: &ResolvedWireRoute) -> Result<()> {
    if route.wire_api != WireApi::ChatCompletions
        || route.dialect != InferenceDialect::Kimi
        || route.request_path.trim_matches('/') != "chat/completions"
        || route.query_params.is_some()
    {
        return Err(CodexErr::InvalidRequest(format!(
            "Kimi route `{}` must resolve to chat_completions/kimi at `chat/completions` without query parameters",
            route.name.as_deref().unwrap_or("<legacy>")
        )));
    }
    Ok(())
}

fn provider_for_route(mut provider: ApiProvider, route: &ResolvedWireRoute) -> Result<ApiProvider> {
    provider.base_url = route.base_url.clone().ok_or_else(|| {
        CodexErr::InvalidRequest("native Kimi route requires a base URL".to_string())
    })?;
    provider.query_params = None;
    provider.retry.max_attempts = 0;
    provider.stream_idle_timeout = route.stream_idle_timeout;
    Ok(provider)
}
