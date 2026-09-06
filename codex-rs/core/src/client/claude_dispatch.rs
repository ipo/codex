use codex_claude_code::CanonicalOutputSchema;
use codex_claude_code::ClaudeCodeEnvironment;
use codex_claude_code::ClaudeCodeIdentity;
use codex_claude_code::ClaudeCodeRequestKind;
use codex_claude_code::ClaudeHttpAdapter;
use codex_claude_code::DecodeError;
use codex_claude_code::EncodeRequest;
use codex_claude_code::NativeStreamError;
use codex_claude_code::OpusCompatibilityContext;
use codex_claude_code::OpusRequestKind;
use codex_claude_code::SonnetCompatibilityContext;
use codex_claude_code::SystemBlock;
use codex_claude_code::TerminalOutcome;
use codex_claude_code::encode_request;
use codex_model_provider_info::ResolvedWireRoute;
use codex_protocol::model_inference::AnthropicThinkingPolicy;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::model_inference::ModelInferenceConfig;
use codex_utils_path_uri::LegacyAppPathString;
use futures::TryStreamExt;

use super::*;

const CLAUDE_MESSAGES_ENDPOINT: &str = "/v1/messages";
const CHATGPT_ACCOUNT_ID_HEADER: &str = "chatgpt-account-id";

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
        let sonnet_compatibility_enabled = plan.wire_model == "claude-sonnet-5";
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
        let environment = (opus_compatibility_enabled || sonnet_compatibility_enabled)
            .then(|| claude_code_environment(responses_metadata))
            .transpose()?;
        let opus_compatibility = if opus_compatibility_enabled {
            Some(OpusCompatibilityContext {
                kind: if matches!(self.client.state.session_source, SessionSource::SubAgent(_)) {
                    OpusRequestKind::Subagent
                } else {
                    OpusRequestKind::Root
                },
                session_id: responses_metadata.session_id.clone(),
                thread_id: responses_metadata.thread_id.clone(),
                installation_id: responses_metadata.installation_id.clone(),
                environment: environment
                    .clone()
                    .expect("compatibility environment was resolved"),
            })
        } else {
            None
        };
        let sonnet_compatibility = if sonnet_compatibility_enabled {
            Some(SonnetCompatibilityContext {
                identity: ClaudeCodeIdentity {
                    kind: if matches!(self.client.state.session_source, SessionSource::SubAgent(_))
                    {
                        ClaudeCodeRequestKind::Subagent
                    } else {
                        ClaudeCodeRequestKind::Root
                    },
                    session_id: responses_metadata.session_id.clone(),
                    thread_id: responses_metadata.thread_id.clone(),
                },
                installation_id: responses_metadata.installation_id.clone(),
                environment: environment.expect("compatibility environment was resolved"),
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
            resumable_session_id: &responses_metadata.session_id,
            codex_version: env!("CARGO_PKG_VERSION"),
            opus_compatibility: opus_compatibility.as_ref(),
            sonnet_compatibility: sonnet_compatibility.as_ref(),
        })
        .map_err(|error| CodexErr::InvalidRequest(error.to_string()))?;

        let client_setup = self.client.current_client_setup().await?;
        let api_provider = provider_for_route(client_setup.api_provider, &plan.route)?;
        let transport = if opus_compatibility.is_some() || sonnet_compatibility.is_some() {
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
        let api_auth = if opus_compatibility.is_some() || sonnet_compatibility.is_some() {
            compatibility_auth(client_setup.api_auth)
        } else {
            client_setup.api_auth
        };
        let stream = ClaudeHttpAdapter::new(transport, api_provider, api_auth)
            .with_telemetry(Some(request_telemetry))
            .stream_request(request, CancellationToken::new())
            .await
            .map_err(classify_native_error)
            .map_err(|error| self.client.state.provider.map_api_error(error))?;
        let upstream_request_id = stream.upstream_request_id.clone();
        let stream = stream.map_err(classify_native_error);
        Ok(map_response_events(
            upstream_request_id,
            stream,
            session_telemetry.clone(),
            inference_trace_attempt,
            Arc::clone(&self.client.state.provider),
        )
        .0)
    }
}

fn claude_code_environment(
    responses_metadata: &CodexResponsesMetadata,
) -> Result<ClaudeCodeEnvironment> {
    let environment = responses_metadata
        .turn_environment
        .as_ref()
        .ok_or_else(|| {
            CodexErr::InvalidRequest(
                "Claude Code compatibility requires authoritative execution environment facts"
                    .to_string(),
            )
        })?;
    let cwd = LegacyAppPathString::from_path_uri(
        &environment.cwd,
        environment.system.operating_system.path_convention(),
    )
    .map_err(|err| {
        CodexErr::InvalidRequest(format!(
            "Claude Code compatibility could not render the execution environment cwd: {err}"
        ))
    })?
    .into_string();
    Ok(ClaudeCodeEnvironment {
        cwd,
        is_git_repository: environment.is_git_repository,
        platform: environment.system.operating_system.platform().to_string(),
        architecture: environment.system.architecture.clone(),
        shell: environment.shell.clone(),
        os_version: environment.system.os_version.clone(),
    })
}

struct CompatibilityAuthProvider {
    inner: SharedAuthProvider,
}

impl AuthProvider for CompatibilityAuthProvider {
    fn add_auth_headers(&self, headers: &mut ApiHeaderMap) {
        self.inner.add_auth_headers(headers);
        headers.remove(CHATGPT_ACCOUNT_ID_HEADER);
    }

    fn apply_auth(&self, request: codex_http_client::Request) -> codex_api::AuthProviderFuture<'_> {
        Box::pin(async move {
            let mut request = self.inner.apply_auth(request).await?;
            request.headers.remove(CHATGPT_ACCOUNT_ID_HEADER);
            Ok(request)
        })
    }
}

fn compatibility_auth(auth: SharedAuthProvider) -> SharedAuthProvider {
    Arc::new(CompatibilityAuthProvider { inner: auth })
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
    let valid_query = route.query_params.as_ref().is_some_and(|query| {
        query.len() == 1
            && query
                .get("beta")
                .is_some_and(|value| value.as_ref() == "true")
    });
    if route.wire_api != WireApi::AnthropicMessages
        || route.dialect != InferenceDialect::ClaudeCode
        || route.request_path.trim_matches('/') != "v1/messages"
        || !valid_query
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
    provider.retry.max_attempts = 0;
    provider.stream_idle_timeout = route.stream_idle_timeout;
    Ok(provider)
}

fn classify_native_error(error: NativeStreamError) -> ApiError {
    match error {
        NativeStreamError::Request(ApiError::Transport(transport)) => {
            classify_transport_error(transport)
        }
        NativeStreamError::Request(ApiError::ServerOverloaded) => retryable("server overloaded"),
        NativeStreamError::Request(ApiError::RateLimit(message)) => retryable(message),
        NativeStreamError::Request(error) => error,
        NativeStreamError::IdleTimeout => retryable("native Claude stream idle timeout"),
        NativeStreamError::Transport(message) => retryable(message),
        NativeStreamError::Decode(DecodeError::EmptyStream) => {
            retryable("native Claude stream was empty")
        }
        NativeStreamError::Decode(DecodeError::PrematureEof { expected }) => {
            retryable(format!("native Claude stream ended before {expected}"))
        }
        NativeStreamError::Decode(DecodeError::ProviderError {
            error_type,
            message,
        }) => classify_provider_error(&error_type, message),
        NativeStreamError::UnsuccessfulTerminal(TerminalOutcome::OutputExhausted) => {
            retryable("native Claude exhausted its output before completing")
        }
        NativeStreamError::UnsuccessfulTerminal(TerminalOutcome::Refusal) => {
            ApiError::InvalidRequest {
                message: "native Claude refused the request".to_string(),
            }
        }
        NativeStreamError::UnsuccessfulTerminal(
            TerminalOutcome::Completed | TerminalOutcome::ToolsReady | TerminalOutcome::Continue,
        ) => unreachable!("successful terminal outcomes are emitted as completed events"),
        NativeStreamError::Decode(error) => ApiError::InvalidRequest {
            message: error.to_string(),
        },
        NativeStreamError::Cancelled => ApiError::InvalidRequest {
            message: "native Claude stream was cancelled".to_string(),
        },
        NativeStreamError::InvalidRequest(message) => ApiError::InvalidRequest { message },
    }
}

fn classify_transport_error(error: TransportError) -> ApiError {
    match error {
        TransportError::Http {
            status,
            url,
            headers,
            body,
        } => {
            let body_text = body.as_deref().unwrap_or_default();
            if contains_any(
                body_text,
                "context window|context limit|maximum context length|prompt is too long|too many input tokens",
            ) {
                ApiError::ContextWindowExceeded
            } else if contains_any(
                body_text,
                "quota_exceeded|insufficient_quota|billing_error|credit balance is too low|exceeded your current quota|usage_limit",
            ) {
                ApiError::QuotaExceeded
            } else if matches!(status.as_u16(), 408 | 409 | 429 | 529) || status.is_server_error() {
                retryable(format!("native Claude request failed with HTTP {status}"))
            } else {
                ApiError::Transport(TransportError::Http {
                    status,
                    url,
                    headers,
                    body,
                })
            }
        }
        TransportError::Timeout | TransportError::Network(_) => retryable(error.to_string()),
        TransportError::RetryLimit | TransportError::Build(_) => ApiError::Transport(error),
    }
}

fn classify_provider_error(error_type: &str, message: String) -> ApiError {
    if contains_any(
        &message,
        "context window|context limit|maximum context length|prompt is too long|too many input tokens",
    ) {
        ApiError::ContextWindowExceeded
    } else if matches!(error_type, "billing_error" | "quota_exceeded")
        || contains_any(
            &message,
            "quota_exceeded|insufficient_quota|billing_error|credit balance is too low|exceeded your current quota|usage_limit",
        )
    {
        ApiError::QuotaExceeded
    } else if matches!(
        error_type,
        "overloaded_error" | "rate_limit_error" | "api_error" | "timeout_error"
    ) {
        retryable(format!("native Claude {error_type}: {message}"))
    } else {
        ApiError::InvalidRequest {
            message: format!("native Claude {error_type}: {message}"),
        }
    }
}

fn retryable(message: impl Into<String>) -> ApiError {
    ApiError::Retryable {
        message: message.into(),
        delay: None,
    }
}

fn contains_any(message: &str, needles: &str) -> bool {
    let message = message.to_ascii_lowercase();
    needles.split('|').any(|needle| message.contains(needle))
}
