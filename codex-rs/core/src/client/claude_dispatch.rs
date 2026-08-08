use super::*;
use codex_claude_code::CanonicalOutputSchema;
use codex_claude_code::ClaudeHttpAdapter;
use codex_claude_code::EncodeRequest;
use codex_claude_code::OpusCompatibilityContext;
use codex_claude_code::OpusEnvironment;
use codex_claude_code::OpusRequestKind;
use codex_claude_code::SystemBlock;
use codex_claude_code::encode_request;
use codex_model_provider_info::ResolvedWireRoute;
use codex_protocol::model_inference::AnthropicThinkingPolicy;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::model_inference::ModelInferenceConfig;
use codex_utils_path_uri::LegacyAppPathString;
use futures::TryStreamExt;

use crate::sampling_retry::classify_native_error;

const CLAUDE_MESSAGES_ENDPOINT: &str = "/v1/messages";

#[cfg(test)]
#[path = "claude_dispatch_tests.rs"]
mod tests;

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
        let opus_compatibility_enabled = plan.wire_model == "claude-opus-5";
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
        let opus_compatibility = if opus_compatibility_enabled {
            let kind = if matches!(self.client.state.session_source, SessionSource::SubAgent(_)) {
                OpusRequestKind::Subagent
            } else {
                OpusRequestKind::Root
            };
            Some(OpusCompatibilityContext {
                kind,
                session_id: responses_metadata.session_id.clone(),
                thread_id: responses_metadata.thread_id.clone(),
                installation_id: responses_metadata.installation_id.clone(),
                environment: opus_environment(responses_metadata)?,
            })
        } else {
            None
        };
        let request = encode_request(EncodeRequest {
            profile: &profile,
            effort: &effort,
            system: &system,
            history: &history,
            tools: &prompt.tools,
            output_schema,
            resumable_session_id: &responses_metadata.session_id.to_string(),
            codex_version: env!("CARGO_PKG_VERSION"),
            opus_compatibility: opus_compatibility.as_ref(),
        })
        .map_err(|error| CodexErr::InvalidRequest(error.to_string()))?;

        let client_setup = self.client.current_client_setup().await?;
        let api_provider = provider_for_route(client_setup.api_provider, &plan.route)?;
        let transport = if opus_compatibility.is_some() {
            self.client
                .build_raw_api_transport(&api_provider, CLAUDE_MESSAGES_ENDPOINT)?
        } else {
            self.client
                .build_api_transport(&api_provider, CLAUDE_MESSAGES_ENDPOINT)?
        };
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
            .map_err(classify_native_error)
            .map_err(|error| self.client.state.provider.map_api_error(error))?;
        let upstream_request_id = stream.upstream_request_id.clone();
        let stream = stream.map_err(classify_native_error);
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

fn opus_environment(responses_metadata: &CodexResponsesMetadata) -> Result<OpusEnvironment> {
    let environment = responses_metadata
        .turn_environment
        .as_ref()
        .ok_or_else(|| {
            CodexErr::InvalidRequest(
                "Opus 5 compatibility requires authoritative execution environment facts"
                    .to_string(),
            )
        })?;
    let cwd = LegacyAppPathString::from_path_uri(
        &environment.cwd,
        environment.system.operating_system.path_convention(),
    )
    .map_err(|err| {
        CodexErr::InvalidRequest(format!(
            "Opus 5 compatibility could not render the execution environment cwd: {err}"
        ))
    })?
    .into_string();
    Ok(OpusEnvironment {
        cwd,
        is_git_repository: environment.is_git_repository,
        platform: environment.system.operating_system.platform().to_string(),
        architecture: environment.system.architecture.clone(),
        shell: environment.shell.clone(),
        os_version: environment.system.os_version.clone(),
    })
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
    // Native request and stream failures share one route-scoped retry budget in the turn loop.
    provider.retry.max_attempts = 0;
    provider.stream_idle_timeout = route.stream_idle_timeout;
    Ok(provider)
}
