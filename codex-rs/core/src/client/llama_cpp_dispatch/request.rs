use super::*;

pub(super) fn validate_route(route: &ResolvedWireRoute) -> Result<()> {
    if route.wire_api != WireApi::Responses
        || route.dialect != InferenceDialect::LlamaCpp
        || route.request_path.trim_matches('/') != "responses"
        || route.query_params.is_some()
    {
        return Err(CodexErr::InvalidRequest(format!(
            "llama.cpp route `{}` must resolve to responses/llama_cpp at `responses` without query parameters",
            route.name.as_deref().unwrap_or("<legacy>")
        )));
    }
    Ok(())
}

pub(super) fn validate_config(config: &LlamaCppInferenceConfig) -> Result<()> {
    let reserved = config
        .max_input_tokens
        .saturating_add(config.max_output_tokens)
        .saturating_add(config.safety_margin_tokens);
    if reserved != config.context_window {
        return Err(CodexErr::InvalidRequest(format!(
            "llama.cpp token contract must reserve exactly its context window: {reserved} != {}",
            config.context_window
        )));
    }
    Ok(())
}

pub(super) fn provider_for_route(
    mut provider: ApiProvider,
    route: &ResolvedWireRoute,
) -> Result<ApiProvider> {
    provider.base_url = route
        .base_url
        .clone()
        .ok_or_else(|| CodexErr::InvalidRequest("llama.cpp route requires a base URL".into()))?;
    if !provider.base_url.trim_end_matches('/').ends_with("/v1") {
        return Err(CodexErr::InvalidRequest(
            "llama.cpp route base URL must end in `/v1`".to_string(),
        ));
    }
    provider.query_params = None;
    provider.headers.remove(http::header::AUTHORIZATION);
    provider.retry.max_attempts = 0;
    provider.stream_idle_timeout = route.stream_idle_timeout;
    Ok(provider)
}

pub(super) fn validate_tools(prompt: &Prompt) -> Result<()> {
    if prompt
        .tools
        .iter()
        .any(|tool| !matches!(tool, ToolSpec::Function(_)))
    {
        return Err(CodexErr::InvalidRequest(
            "direct llama.cpp Responses supports only plain function tools".to_string(),
        ));
    }
    Ok(())
}

pub(super) fn normalize_input(prompt: &Prompt) -> Result<Vec<ResponseItem>> {
    let mut input = Vec::with_capacity(prompt.input.len());
    let mut conversation_started = false;
    for (index, mut item) in prompt
        .get_formatted_input_for_request(/*use_responses_lite*/ false)
        .into_iter()
        .enumerate()
    {
        match &mut item {
            ResponseItem::Message { role, content, .. } => {
                let is_instruction_role = matches!(role.as_str(), "system" | "developer");
                if content.iter().any(|content| {
                    matches!(
                        content,
                        ContentItem::InputImage { .. } | ContentItem::InputAudio { .. }
                    )
                }) {
                    return Err(CodexErr::InvalidRequest(format!(
                        "llama.cpp history item at index {index} contains unsupported image or audio input"
                    )));
                }
                if is_instruction_role && conversation_started {
                    let text = content
                        .iter()
                        .map(|content| match content {
                            ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                                Ok(text.as_str())
                            }
                            ContentItem::InputImage { .. } | ContentItem::InputAudio { .. } => {
                                Err(CodexErr::InvalidRequest(
                                    "late llama.cpp context update must contain only text"
                                        .to_string(),
                                ))
                            }
                        })
                        .collect::<Result<Vec<_>>>()?
                        .join("\n");
                    item = ContextualUserFragment::into(AdditionalContextUserFragment::new(
                        format!("local_{role}_update"),
                        text,
                    ));
                } else if !matches!(role.as_str(), "system" | "developer" | "user" | "assistant") {
                    return Err(CodexErr::InvalidRequest(format!(
                        "llama.cpp history item at index {index} has unsupported message role `{role}`"
                    )));
                }
                if !is_instruction_role {
                    conversation_started = true;
                }
            }
            ResponseItem::AgentMessage {
                author,
                recipient,
                content,
                ..
            } => {
                let text = render_plaintext_agent_message(author, recipient, content).ok_or_else(
                    || {
                        CodexErr::InvalidRequest(format!(
                            "llama.cpp history item at index {index} contains an encrypted cross-agent message"
                        ))
                    },
                )?;
                item = ResponseItem::Message {
                    id: None,
                    role: "user".to_string(),
                    content: vec![ContentItem::InputText { text }],
                    phase: None,
                    internal_chat_message_metadata_passthrough: None,
                };
                conversation_started = true;
            }
            ResponseItem::Reasoning {
                encrypted_content, ..
            } => {
                if encrypted_content
                    .as_deref()
                    .is_some_and(|content| !content.is_empty())
                {
                    return Err(CodexErr::InvalidRequest(format!(
                        "llama.cpp history item at index {index} contains unsupported encrypted reasoning"
                    )));
                }
                conversation_started = true;
            }
            ResponseItem::FunctionCall { namespace, .. } => {
                if namespace.is_some() {
                    return Err(CodexErr::InvalidRequest(format!(
                        "llama.cpp history item at index {index} contains an unsupported namespace tool call"
                    )));
                }
                conversation_started = true;
            }
            ResponseItem::FunctionCallOutput { output, .. } => {
                if let Some(items) = output.content_items() {
                    let mut text = Vec::with_capacity(items.len());
                    for item in items {
                        match item {
                            FunctionCallOutputContentItem::InputText { text: value } => {
                                text.push(value.as_str());
                            }
                            FunctionCallOutputContentItem::InputImage { .. }
                            | FunctionCallOutputContentItem::InputAudio { .. }
                            | FunctionCallOutputContentItem::EncryptedContent { .. } => {
                                return Err(CodexErr::InvalidRequest(format!(
                                    "llama.cpp tool result at history index {index} contains unsupported media or encrypted content"
                                )));
                            }
                        }
                    }
                    *output = codex_protocol::models::FunctionCallOutputPayload::from_text(
                        text.join("\n"),
                    );
                }
                conversation_started = true;
            }
            ResponseItem::AdditionalTools { .. }
            | ResponseItem::LocalShellCall { .. }
            | ResponseItem::ToolSearchCall { .. }
            | ResponseItem::CustomToolCall { .. }
            | ResponseItem::CustomToolCallOutput { .. }
            | ResponseItem::ToolSearchOutput { .. }
            | ResponseItem::WebSearchCall { .. }
            | ResponseItem::ImageGenerationCall { .. }
            | ResponseItem::Compaction { .. }
            | ResponseItem::CompactionTrigger { .. }
            | ResponseItem::ContextCompaction { .. }
            | ResponseItem::Other => {
                return Err(CodexErr::InvalidRequest(format!(
                    "llama.cpp history item at index {index} uses an unsupported Responses item type"
                )));
            }
        }
        input.push(item);
    }
    Ok(input)
}

pub(super) fn local_effort(effort: Option<ReasoningEffortConfig>) -> Result<ReasoningEffortConfig> {
    match effort.unwrap_or(ReasoningEffort::Low) {
        effort @ (ReasoningEffort::None
        | ReasoningEffort::Low
        | ReasoningEffort::Medium
        | ReasoningEffort::XHigh) => Ok(effort),
        ReasoningEffort::High => Ok(ReasoningEffort::XHigh),
        effort @ (ReasoningEffort::Minimal
        | ReasoningEffort::Max
        | ReasoningEffort::Ultra
        | ReasoningEffort::Custom(_)) => Err(CodexErr::InvalidRequest(format!(
            "llama.cpp reasoning effort `{effort}` is unsupported"
        ))),
    }
}

pub(super) fn llama_cpp_request_body(
    request: &ResponsesApiRequest,
    config: &LlamaCppInferenceConfig,
    effort: &ReasoningEffortConfig,
) -> Result<Value> {
    let mut body = serde_json::to_value(request)?;
    let input = request
        .input
        .iter()
        .map(serialize_local_item)
        .collect::<Result<Vec<_>>>()?;
    let object = body.as_object_mut().ok_or_else(|| {
        CodexErr::InvalidRequest("llama.cpp request did not serialize as an object".to_string())
    })?;
    object.insert("input".to_string(), Value::Array(input));
    object.insert("cache_prompt".to_string(), Value::Bool(true));
    object.insert(
        "max_output_tokens".to_string(),
        json!(config.max_output_tokens),
    );
    let thinking = *effort != ReasoningEffort::None;
    object.insert(
        "chat_template_kwargs".to_string(),
        json!({"enable_thinking": thinking, "preserve_thinking": thinking}),
    );
    let (temperature, top_p, presence_penalty) = if thinking {
        (1.0, 0.95, 0.0)
    } else {
        (0.7, 0.8, 1.5)
    };
    object.insert("temperature".to_string(), json!(temperature));
    object.insert("top_p".to_string(), json!(top_p));
    object.insert("top_k".to_string(), json!(20));
    object.insert("min_p".to_string(), json!(0.0));
    object.insert("presence_penalty".to_string(), json!(presence_penalty));
    object.insert("repeat_penalty".to_string(), json!(1.0));
    Ok(body)
}

fn serialize_local_item(item: &ResponseItem) -> Result<Value> {
    let mut value = serde_json::to_value(item)?;
    if let Value::Object(object) = &mut value {
        object.remove("phase");
        object.remove("internal_chat_message_metadata_passthrough");
        if let ResponseItem::Reasoning {
            content: Some(content),
            ..
        } = item
        {
            object.insert("content".to_string(), serde_json::to_value(content)?);
        }
    }
    Ok(value)
}
