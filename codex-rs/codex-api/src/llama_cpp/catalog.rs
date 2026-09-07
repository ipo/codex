use std::collections::HashMap;

use serde::Deserialize;

const FALLBACK_CONTEXT_WINDOW: u64 = 32_768;
const MAX_OUTPUT_TOKENS: u64 = 8_192;
const SAFETY_MARGIN_TOKENS: u64 = 1_024;

/// One model exposed by the managed llama.cpp endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaCppCatalogEntry {
    pub canonical_id: String,
    pub wire_model: String,
    pub display_name: String,
    pub aliases: Vec<String>,
    pub context_window: u64,
    pub max_input_tokens: u64,
    pub max_output_tokens: u64,
    pub safety_margin_tokens: u64,
}

/// Runtime model catalog derived directly from llama.cpp's `/v1/models` response.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LlamaCppCatalog {
    pub models: Vec<LlamaCppCatalogEntry>,
}

impl LlamaCppCatalog {
    pub(super) fn from_discovery(response: ModelsResponse) -> Self {
        let alias_counts = response
            .data
            .iter()
            .fold(HashMap::new(), |mut counts, model| {
                let alias = without_gguf_suffix(model_basename(&model.id)).to_ascii_lowercase();
                *counts.entry(alias).or_insert(0) += 1;
                counts
            });
        let expose_convenience_alias = response.data.len() == 1;
        let models = response
            .data
            .into_iter()
            .map(|model| {
                let display_name = model_basename(&model.id).to_string();
                let mut aliases = Vec::new();
                let basename_alias = without_gguf_suffix(&display_name);
                if alias_counts.get(&basename_alias.to_ascii_lowercase()) == Some(&1) {
                    aliases.push(basename_alias.to_string());
                }
                if expose_convenience_alias
                    && !aliases.iter().any(|alias| alias == "llama-cpp-local")
                {
                    aliases.push("llama-cpp-local".to_string());
                }
                let context_window = model
                    .meta
                    .and_then(|meta| u64::try_from(meta.n_ctx?).ok())
                    .filter(|context_window| *context_window > 0)
                    .unwrap_or(FALLBACK_CONTEXT_WINDOW);
                LlamaCppCatalogEntry {
                    canonical_id: format!("local/{}", model.id),
                    wire_model: model.id,
                    display_name,
                    aliases,
                    context_window,
                    max_input_tokens: context_window
                        .saturating_sub(MAX_OUTPUT_TOKENS + SAFETY_MARGIN_TOKENS),
                    max_output_tokens: MAX_OUTPUT_TOKENS,
                    safety_margin_tokens: SAFETY_MARGIN_TOKENS,
                }
            })
            .collect();
        Self { models }
    }

    pub fn resolve(
        &self,
        requested: &str,
    ) -> Result<LlamaCppCatalogEntry, LlamaCppModelSelectionError> {
        self.models
            .iter()
            .find(|model| {
                model.canonical_id == requested
                    || model.aliases.iter().any(|alias| alias == requested)
            })
            .cloned()
            .ok_or_else(|| LlamaCppModelSelectionError {
                requested: requested.to_string(),
                currently_discovered: self
                    .models
                    .iter()
                    .map(|model| model.canonical_id.clone())
                    .collect(),
            })
    }
}

/// A selected local model is not present in the latest endpoint catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaCppModelSelectionError {
    pub requested: String,
    pub currently_discovered: Vec<String>,
}

impl std::fmt::Display for LlamaCppModelSelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.currently_discovered.is_empty() {
            write!(
                f,
                "llama.cpp model `{}` is unavailable; the endpoint currently advertises no models",
                self.requested
            )
        } else {
            write!(
                f,
                "llama.cpp model `{}` is unavailable; currently discovered choices: {}",
                self.requested,
                self.currently_discovered.join(", ")
            )
        }
    }
}

impl std::error::Error for LlamaCppModelSelectionError {}

fn model_basename(model_id: &str) -> &str {
    model_id.rsplit(['/', '\\']).next().unwrap_or(model_id)
}

fn without_gguf_suffix(basename: &str) -> &str {
    basename.strip_suffix(".gguf").unwrap_or(basename)
}

#[derive(Debug, Deserialize)]
pub(super) struct ModelsResponse {
    data: Vec<DiscoveredModel>,
}

#[derive(Debug, Deserialize)]
struct DiscoveredModel {
    id: String,
    meta: Option<DiscoveredModelMeta>,
}

#[derive(Debug, Deserialize)]
struct DiscoveredModelMeta {
    n_ctx: Option<i64>,
}
