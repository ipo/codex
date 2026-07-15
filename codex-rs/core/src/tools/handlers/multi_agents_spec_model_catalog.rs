use codex_protocol::openai_models::ModelPreset;

pub(super) const MAX_MODEL_OVERRIDES_IN_SPAWN_AGENT_DESCRIPTION: usize = 5;
const MAX_MODEL_SLUG_BYTES_IN_SPAWN_AGENT_DESCRIPTION: usize = 96;
const MAX_MODEL_DESCRIPTION_BYTES_IN_SPAWN_AGENT_DESCRIPTION: usize = 160;
const MAX_REASONING_EFFORTS_IN_SPAWN_AGENT_DESCRIPTION: usize = 8;
pub(super) const MAX_REASONING_EFFORT_BYTES_IN_SPAWN_AGENT_DESCRIPTION: usize = 32;
const MAX_SERVICE_TIERS_IN_SPAWN_AGENT_DESCRIPTION: usize = 4;
const MAX_SERVICE_TIER_BYTES_IN_SPAWN_AGENT_DESCRIPTION: usize = 32;
pub(super) const MAX_SPAWN_AGENT_MODELS_DESCRIPTION_BYTES: usize = 4_096;
pub(super) const TRUNCATION_SUFFIX: &str = "…";

pub(super) fn spawn_agent_models_description(
    models: &[ModelPreset],
    include_service_tiers: bool,
) -> String {
    let visible_models: Vec<&ModelPreset> = models
        .iter()
        .filter(|model| model.show_in_picker)
        .take(MAX_MODEL_OVERRIDES_IN_SPAWN_AGENT_DESCRIPTION)
        .collect();
    if visible_models.is_empty() {
        return "No picker-visible model overrides are currently loaded.".to_string();
    }

    let model_descriptions = visible_models
        .into_iter()
        .map(|model| {
            let default_reasoning_effort = &model.default_reasoning_effort;
            let efforts = model
                .supported_reasoning_efforts
                .iter()
                .take(MAX_REASONING_EFFORTS_IN_SPAWN_AGENT_DESCRIPTION)
                .map(|preset| {
                    let effort = truncate_utf8_bytes(
                        preset.effort.as_str(),
                        MAX_REASONING_EFFORT_BYTES_IN_SPAWN_AGENT_DESCRIPTION,
                    );
                    if &preset.effort == default_reasoning_effort {
                        format!("{effort} (default)")
                    } else {
                        effort
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            let reasoning_efforts_suffix = if efforts.is_empty() {
                String::new()
            } else {
                format!(" Reasoning efforts: {efforts}.")
            };
            let service_tiers_suffix = if include_service_tiers && !model.service_tiers.is_empty() {
                let service_tiers = model
                    .service_tiers
                    .iter()
                    .take(MAX_SERVICE_TIERS_IN_SPAWN_AGENT_DESCRIPTION)
                    .map(|tier| {
                        truncate_utf8_bytes(
                            tier.id.as_str(),
                            MAX_SERVICE_TIER_BYTES_IN_SPAWN_AGENT_DESCRIPTION,
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(" Service tiers: {service_tiers}.")
            } else {
                String::new()
            };
            let model_slug = truncate_utf8_bytes(
                &model.model,
                MAX_MODEL_SLUG_BYTES_IN_SPAWN_AGENT_DESCRIPTION,
            );
            let description = truncate_utf8_bytes(
                &model.description,
                MAX_MODEL_DESCRIPTION_BYTES_IN_SPAWN_AGENT_DESCRIPTION,
            );
            format!(
                "- `{model_slug}`: {description}{reasoning_efforts_suffix}{service_tiers_suffix}"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let description = format!(
        "Available model overrides (optional; inherited parent model is preferred):\n{model_descriptions}"
    );
    truncate_utf8_bytes(&description, MAX_SPAWN_AGENT_MODELS_DESCRIPTION_BYTES)
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
