use super::multi_agents_common::model_supports_multi_agent_backend;
use codex_protocol::openai_models::ModelPreset;
use codex_protocol::protocol::MultiAgentVersion;

pub(super) const MAX_REQUESTED_MODEL_BYTES_IN_SPAWN_AGENT_ERROR: usize = 96;
/// Hard byte ceiling for the injected model catalog. Codex's shared context estimator uses four
/// bytes per token, so 39,996 bytes is at most 9,999 estimated tokens and remains below the
/// repository's 10K-token per-fragment limit.
pub(super) const MAX_SPAWN_AGENT_MODELS_DESCRIPTION_BYTES: usize = 39_996;
pub(super) const TRUNCATION_SUFFIX: &str = "…";

const SPAWN_AGENT_MODELS_DESCRIPTION_HEADER: &str =
    "Available model overrides (optional; inherited parent model is preferred):";
const SPAWN_AGENT_MODEL_GROUPS_LEGEND: &str = "Effort values are valid `reasoning_effort` inputs; * = default when omitted. Models in each group share that signature.";
pub(super) const SPAWN_AGENT_MODEL_CATALOG_TOO_LARGE: &str =
    "Preferred model selector catalog exceeds the model-visible context safety bound.";

struct SpawnAgentModelGroup<'a> {
    supported_efforts: Vec<&'a str>,
    default_effort: &'a str,
    selectors: Vec<&'a str>,
}

pub(super) fn spawn_agent_models_description(
    models: &[ModelPreset],
    multi_agent_version: MultiAgentVersion,
) -> String {
    let visible_models: Vec<&ModelPreset> = models
        .iter()
        .filter(|model| model.show_in_picker)
        .filter(|model| model_supports_multi_agent_backend(model, multi_agent_version))
        .collect();
    if visible_models.is_empty() {
        return "No picker-visible model overrides are currently loaded.".to_string();
    }

    let exact_selectors = visible_models
        .iter()
        .map(|model| preferred_spawn_agent_model_selector(model))
        .collect::<Vec<_>>()
        .join("\n");
    if exact_selectors.len() > MAX_SPAWN_AGENT_MODELS_DESCRIPTION_BYTES {
        return SPAWN_AGENT_MODEL_CATALOG_TOO_LARGE.to_string();
    }

    let mut groups: Vec<SpawnAgentModelGroup<'_>> = Vec::new();
    for model in visible_models {
        let supported_efforts = model
            .supported_reasoning_efforts
            .iter()
            .map(|preset| preset.effort.as_str())
            .collect::<Vec<_>>();
        let default_effort = model.default_reasoning_effort.as_str();
        let selector = preferred_spawn_agent_model_selector(model);
        if let Some(group) = groups.iter_mut().find(|group| {
            group.supported_efforts == supported_efforts && group.default_effort == default_effort
        }) {
            group.selectors.push(selector);
        } else {
            groups.push(SpawnAgentModelGroup {
                supported_efforts,
                default_effort,
                selectors: vec![selector],
            });
        }
    }

    let grouped_models = groups
        .iter()
        .map(|group| {
            let efforts = if group.supported_efforts.is_empty() {
                "(none)".to_string()
            } else {
                group
                    .supported_efforts
                    .iter()
                    .map(|effort| {
                        if *effort == group.default_effort {
                            format!("{effort}*")
                        } else {
                            effort.to_string()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            format!("Efforts: {efforts}\nModels: {}", group.selectors.join(", "))
        })
        .collect::<Vec<_>>()
        .join("\n");
    let description = format!(
        "{SPAWN_AGENT_MODELS_DESCRIPTION_HEADER}\n{SPAWN_AGENT_MODEL_GROUPS_LEGEND}\n{grouped_models}"
    );
    if description.len() <= MAX_SPAWN_AGENT_MODELS_DESCRIPTION_BYTES {
        return description;
    }

    let description = format!("{SPAWN_AGENT_MODELS_DESCRIPTION_HEADER}\n{exact_selectors}");
    if description.len() <= MAX_SPAWN_AGENT_MODELS_DESCRIPTION_BYTES {
        description
    } else {
        exact_selectors
    }
}

pub(super) fn bounded_spawn_agent_model_selectors(
    models: &[ModelPreset],
    multi_agent_version: MultiAgentVersion,
    max_bytes: usize,
) -> String {
    let visible_models = models
        .iter()
        .filter(|model| model.show_in_picker)
        .filter(|model| model_supports_multi_agent_backend(model, multi_agent_version))
        .collect::<Vec<_>>();
    if visible_models.is_empty() {
        return truncate_utf8_bytes(
            "No picker-visible model overrides are currently loaded.",
            max_bytes,
        );
    }

    let selectors = visible_models
        .iter()
        .map(|model| preferred_spawn_agent_model_selector(model))
        .collect::<Vec<_>>();
    let exact_selectors = selectors.join("\n");
    if exact_selectors.len() <= max_bytes {
        exact_selectors
    } else if max_bytes < TRUNCATION_SUFFIX.len() {
        String::new()
    } else {
        truncate_utf8_bytes(SPAWN_AGENT_MODEL_CATALOG_TOO_LARGE, max_bytes)
    }
}

pub(super) fn preferred_spawn_agent_model_selector(model: &ModelPreset) -> &str {
    model.aliases.first().unwrap_or(&model.model)
}

pub(super) fn truncate_utf8_bytes(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_string();
    }

    let prefix_budget = max_bytes.saturating_sub(TRUNCATION_SUFFIX.len());
    let prefix_end = (0..=prefix_budget)
        .rev()
        .find(|index| value.is_char_boundary(*index))
        .unwrap_or_default();
    let mut truncated = value[..prefix_end].to_string();
    truncated.push_str(TRUNCATION_SUFFIX);
    truncated
}
