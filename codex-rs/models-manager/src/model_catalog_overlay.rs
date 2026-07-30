use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ModelsResponse;
use serde_json::Map;
use serde_json::Value;
use std::collections::HashSet;
use std::fmt;

const MODEL_INFO_FIELDS: &[&str] = &[
    "slug",
    "aliases",
    "display_name",
    "description",
    "default_reasoning_level",
    "supported_reasoning_levels",
    "shell_type",
    "visibility",
    "supported_in_api",
    "priority",
    "additional_speed_tiers",
    "service_tiers",
    "default_service_tier",
    "availability_nux",
    "upgrade",
    "base_instructions",
    "model_messages",
    "include_skills_usage_instructions",
    "supports_reasoning_summary_parameter",
    "default_reasoning_summary",
    "support_verbosity",
    "default_verbosity",
    "apply_patch_tool_type",
    "web_search_tool_type",
    "truncation_policy",
    "supports_parallel_tool_calls",
    "supports_image_detail_original",
    "context_window",
    "max_context_window",
    "auto_compact_token_limit",
    "comp_hash",
    "history_compatibility_group",
    "requires_nonempty_assistant_messages",
    "effective_context_window_percent",
    "experimental_supported_tools",
    "input_modalities",
    "supports_search_tool",
    "use_responses_lite",
    "auto_review_model_override",
    "tool_mode",
    "multi_agent_version",
];

#[derive(Debug, Clone, PartialEq)]
/// A validated sequence of shallow model catalog patches and inherited additions.
pub struct ModelCatalogOverlay {
    entries: Vec<ModelCatalogOverlayEntry>,
}

/// A startup-validated overlay whose entries can be composed onto later catalog snapshots.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedModelCatalogOverlay {
    entries: Vec<ResolvedModelCatalogOverlayEntry>,
}

#[derive(Debug, Clone, PartialEq)]
struct ModelCatalogOverlayEntry {
    index: usize,
    slug: String,
    inherits: Option<String>,
    fields: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq)]
struct ResolvedModelCatalogOverlayEntry {
    entry: ModelCatalogOverlayEntry,
    fallback: ModelInfo,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Error returned while parsing or applying a model catalog overlay.
pub struct ModelCatalogOverlayError(String);

impl fmt::Display for ModelCatalogOverlayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ModelCatalogOverlayError {}

impl ModelCatalogOverlay {
    /// Parse and validate an overlay document with the shape `{ "models": [...] }`.
    pub fn from_json(json: &str) -> Result<Self, ModelCatalogOverlayError> {
        let value: Value = serde_json::from_str(json)
            .map_err(|err| ModelCatalogOverlayError(format!("invalid JSON: {err}")))?;
        let models = value
            .as_object()
            .and_then(|root| root.get("models"))
            .ok_or_else(|| ModelCatalogOverlayError("missing field `models`".to_string()))?
            .as_array()
            .ok_or_else(|| {
                ModelCatalogOverlayError("field `models` must be an array".to_string())
            })?;

        let mut entries = Vec::with_capacity(models.len());
        let mut slugs = HashSet::with_capacity(models.len());
        for (index, value) in models.iter().enumerate() {
            let fields = value.as_object().cloned().ok_or_else(|| {
                ModelCatalogOverlayError(format!("entry {index}: must be an object"))
            })?;
            let slug = fields
                .get("slug")
                .ok_or_else(|| {
                    ModelCatalogOverlayError(format!("entry {index}: missing field `slug`"))
                })?
                .as_str()
                .filter(|slug| !slug.is_empty())
                .ok_or_else(|| {
                    ModelCatalogOverlayError(format!(
                        "entry {index}: field `slug` must be a non-empty string"
                    ))
                })?
                .to_string();
            if !slugs.insert(slug.clone()) {
                return Err(ModelCatalogOverlayError(format!(
                    "entry {index} slug `{slug}`: duplicate slug"
                )));
            }
            let inherits = match fields.get("inherits") {
                Some(Value::String(inherits)) if !inherits.is_empty() => Some(inherits.clone()),
                Some(_) => {
                    return Err(ModelCatalogOverlayError(format!(
                        "entry {index} slug `{slug}`: field `inherits` must be a non-empty string"
                    )));
                }
                None => None,
            };
            if let Some(field) = fields.keys().find(|field| {
                field.as_str() != "inherits" && !MODEL_INFO_FIELDS.contains(&field.as_str())
            }) {
                return Err(ModelCatalogOverlayError(format!(
                    "entry {index} slug `{slug}`: invalid field `{field}`"
                )));
            }
            entries.push(ModelCatalogOverlayEntry {
                index,
                slug,
                inherits,
                fields,
            });
        }
        Ok(Self { entries })
    }

    /// Apply the overlay in order, replacing matching slugs in place and appending new slugs.
    pub fn apply(
        &self,
        mut catalog: ModelsResponse,
    ) -> Result<ModelsResponse, ModelCatalogOverlayError> {
        for entry in &self.entries {
            let model = apply_entry(entry, &catalog.models)?;
            replace_or_append(&mut catalog.models, model);
        }
        Ok(catalog)
    }

    /// Resolve every entry against the startup catalog for later refresh composition.
    pub fn resolve(
        self,
        mut catalog: ModelsResponse,
    ) -> Result<ResolvedModelCatalogOverlay, ModelCatalogOverlayError> {
        let mut entries = Vec::with_capacity(self.entries.len());
        for entry in self.entries {
            let fallback = apply_entry(&entry, &catalog.models)?;
            replace_or_append(&mut catalog.models, fallback.clone());
            entries.push(ResolvedModelCatalogOverlayEntry { entry, fallback });
        }
        Ok(ResolvedModelCatalogOverlay { entries })
    }
}

impl ResolvedModelCatalogOverlay {
    /// Compose the resolved overlay onto a later effective catalog snapshot.
    pub fn apply(&self, mut catalog: ModelsResponse) -> ModelsResponse {
        for resolved in &self.entries {
            let model = apply_entry(&resolved.entry, &catalog.models)
                .unwrap_or_else(|_| resolved.fallback.clone());
            replace_or_append(&mut catalog.models, model);
        }
        catalog
    }
}

fn apply_entry(
    entry: &ModelCatalogOverlayEntry,
    models: &[ModelInfo],
) -> Result<ModelInfo, ModelCatalogOverlayError> {
    let parent_slug = entry.inherits.as_deref().unwrap_or(&entry.slug);
    let Some(parent) = models.iter().find(|model| model.slug == parent_slug) else {
        let detail = if entry.inherits.is_some() {
            format!("missing parent model `{parent_slug}`")
        } else {
            "new model requires field `inherits`".to_string()
        };
        return Err(ModelCatalogOverlayError(format!(
            "entry {} slug `{}`: {detail}",
            entry.index, entry.slug
        )));
    };

    let parent_value = serde_json::to_value(parent).map_err(|err| {
        ModelCatalogOverlayError(format!(
            "entry {} slug `{}`: failed to serialize parent model: {err}",
            entry.index, entry.slug
        ))
    })?;
    let mut merged = parent_value.as_object().cloned().ok_or_else(|| {
        ModelCatalogOverlayError(format!(
            "entry {} slug `{}`: parent model did not serialize as an object",
            entry.index, entry.slug
        ))
    })?;
    if entry.inherits.is_some() {
        merged.insert("aliases".to_string(), Value::Array(Vec::new()));
        merged.remove("history_compatibility_group");
    }
    for (field, value) in &entry.fields {
        if field != "inherits" {
            merged.insert(field.clone(), value.clone());
        }
    }

    serde_json::from_value::<ModelInfo>(Value::Object(merged)).map_err(|err| {
        let invalid_field = entry.fields.iter().find_map(|(field, value)| {
            if field == "inherits" || field == "slug" {
                return None;
            }
            let mut candidate = parent_value.as_object()?.clone();
            candidate.insert("slug".to_string(), Value::String(entry.slug.clone()));
            candidate.insert(field.clone(), value.clone());
            serde_json::from_value::<ModelInfo>(Value::Object(candidate))
                .is_err()
                .then_some(field)
        });
        let detail = invalid_field.map_or_else(
            || format!("invalid ModelInfo: {err}"),
            |field| format!("invalid field `{field}`: {err}"),
        );
        ModelCatalogOverlayError(format!(
            "entry {} slug `{}`: {detail}",
            entry.index, entry.slug
        ))
    })
}

fn replace_or_append(models: &mut Vec<ModelInfo>, model: ModelInfo) {
    if let Some(target_index) = models
        .iter()
        .position(|candidate| candidate.slug == model.slug)
    {
        models[target_index] = model;
    } else {
        models.push(model);
    }
}

#[cfg(test)]
#[path = "model_catalog_overlay_tests.rs"]
mod tests;
