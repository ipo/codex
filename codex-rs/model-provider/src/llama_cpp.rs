use std::sync::Arc;
use std::time::Duration;

use codex_api::LLAMA_CPP_LOCAL_ENDPOINT;
use codex_api::LlamaCppCatalog;
use codex_api::LlamaCppCatalogEntry;
use codex_api::LlamaCppRuntime;
use codex_http_client::ClientRouteClass;
use codex_http_client::HttpClientFactory;
use codex_login::AuthManager;
use codex_login::default_client::create_client_for_route_async;
use codex_model_provider_info::LLAMA_CPP_ROUTE_NAME;
use codex_models_manager::manager::ModelsManager;
use codex_models_manager::manager::ModelsManagerFuture;
use codex_models_manager::manager::RefreshStrategy;
use codex_models_manager::manager::SharedModelsManager;
use codex_protocol::config_types::CollaborationModeMask;
use codex_protocol::config_types::ReasoningSummary;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::model_inference::LlamaCppInferenceConfig;
use codex_protocol::model_inference::ModelInferenceConfig;
use codex_protocol::model_inference::WireApi;
use codex_protocol::openai_models::InputModality;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ModelToolCapability;
use codex_protocol::openai_models::ReasoningEffort;
use codex_protocol::openai_models::ReasoningEffortPreset;
use codex_protocol::openai_models::ToolMode;
use codex_protocol::openai_models::WebSearchToolType;
use codex_protocol::protocol::MultiAgentVersion;
use tokio::sync::RwLock;
use tokio::sync::TryLockError;
use tracing::warn;

const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(5);
const BASELINE_MODEL: &str = "gpt-5.6-sol";

#[derive(Debug)]
pub(crate) struct LlamaCppModelsManager {
    inner: SharedModelsManager,
    local_models: RwLock<Vec<ModelInfo>>,
}

impl LlamaCppModelsManager {
    fn new(inner: SharedModelsManager) -> Self {
        Self {
            inner,
            local_models: RwLock::new(Vec::new()),
        }
    }

    async fn refresh_local_models(
        &self,
        base_models: &[ModelInfo],
        http_client_factory: HttpClientFactory,
    ) {
        let discovery = tokio::time::timeout(DISCOVERY_TIMEOUT, async move {
            let http_client = create_client_for_route_async(
                http_client_factory,
                LLAMA_CPP_LOCAL_ENDPOINT.to_string(),
                ClientRouteClass::Api,
            )
            .await
            .map_err(|error| error.to_string())?;
            LlamaCppRuntime::new(http_client)
                .catalog()
                .await
                .map_err(|error| error.to_string())
        })
        .await;
        let catalog = match discovery {
            Ok(Ok(catalog)) => catalog,
            Ok(Err(error)) => {
                warn!(error, "llama.cpp model discovery failed");
                return;
            }
            Err(_) => {
                warn!("llama.cpp model discovery timed out");
                return;
            }
        };
        let Some(models) = local_model_infos(base_models, &catalog) else {
            warn!("llama.cpp model discovery could not find baseline model metadata");
            return;
        };
        *self.local_models.write().await = models;
    }

    async fn append_local_models(&self, models: &mut Vec<ModelInfo>) {
        for local in self.local_models.read().await.iter().cloned() {
            replace_or_append(models, local);
        }
    }
}

impl ModelsManager for LlamaCppModelsManager {
    fn raw_model_catalog(
        &self,
        refresh_strategy: RefreshStrategy,
        http_client_factory: HttpClientFactory,
    ) -> ModelsManagerFuture<'_, codex_protocol::openai_models::ModelsResponse> {
        Box::pin(async move {
            let mut catalog = self
                .inner
                .raw_model_catalog(refresh_strategy, http_client_factory.clone())
                .await;
            if refresh_strategy != RefreshStrategy::Offline {
                self.refresh_local_models(&catalog.models, http_client_factory)
                    .await;
            }
            self.append_local_models(&mut catalog.models).await;
            catalog
        })
    }

    fn get_remote_models(&self) -> ModelsManagerFuture<'_, Vec<ModelInfo>> {
        Box::pin(async move {
            let mut models = self.inner.get_remote_models().await;
            self.append_local_models(&mut models).await;
            models
        })
    }

    fn try_get_remote_models(&self) -> Result<Vec<ModelInfo>, TryLockError> {
        let mut models = self.inner.try_get_remote_models()?;
        for local in self.local_models.try_read()?.iter().cloned() {
            replace_or_append(&mut models, local);
        }
        Ok(models)
    }

    fn auth_manager(&self) -> Option<&AuthManager> {
        self.inner.auth_manager()
    }

    fn list_collaboration_modes(&self) -> Vec<CollaborationModeMask> {
        self.inner.list_collaboration_modes()
    }

    fn refresh_if_new_etag(
        &self,
        etag: String,
        http_client_factory: HttpClientFactory,
    ) -> ModelsManagerFuture<'_, ()> {
        Box::pin(async move {
            self.inner
                .refresh_if_new_etag(etag, http_client_factory.clone())
                .await;
            let base_models = self.inner.get_remote_models().await;
            self.refresh_local_models(&base_models, http_client_factory)
                .await;
        })
    }
}

pub(crate) fn with_llama_cpp_models(inner: SharedModelsManager) -> SharedModelsManager {
    Arc::new(LlamaCppModelsManager::new(inner))
}

fn local_model_infos(
    base_models: &[ModelInfo],
    catalog: &LlamaCppCatalog,
) -> Option<Vec<ModelInfo>> {
    let baseline = base_models
        .iter()
        .find(|model| model.slug == BASELINE_MODEL)
        .cloned()
        .or_else(|| {
            codex_models_manager::bundled_models_response()
                .ok()?
                .models
                .into_iter()
                .find(|model| model.slug == BASELINE_MODEL)
        })?;
    Some(
        catalog
            .models
            .iter()
            .map(|entry| local_model_info(&baseline, entry))
            .collect(),
    )
}

fn local_model_info(baseline: &ModelInfo, entry: &LlamaCppCatalogEntry) -> ModelInfo {
    let context_window = u32::try_from(entry.context_window).unwrap_or(u32::MAX);
    let max_input_tokens = u32::try_from(entry.max_input_tokens).unwrap_or(u32::MAX);
    let max_output_tokens = u32::try_from(entry.max_output_tokens).unwrap_or(u32::MAX);
    let safety_margin_tokens = u32::try_from(entry.safety_margin_tokens).unwrap_or(u32::MAX);
    let mut model = baseline.clone();
    model.slug.clone_from(&entry.canonical_id);
    model.inference = Some(ModelInferenceConfig::LlamaCpp(LlamaCppInferenceConfig {
        wire_api: WireApi::Responses,
        dialect: InferenceDialect::LlamaCpp,
        route: LLAMA_CPP_ROUTE_NAME.to_string(),
        expected_model_basename: entry.display_name.clone(),
        context_window,
        max_input_tokens,
        max_output_tokens,
        safety_margin_tokens,
    }));
    model.aliases.clone_from(&entry.aliases);
    model.display_name.clone_from(&entry.display_name);
    model.description = Some(format!(
        "Locally hosted Qwen3.8-compatible model through llama.cpp ({})",
        entry.display_name
    ));
    model.default_reasoning_level = Some(ReasoningEffort::Low);
    model.supported_reasoning_levels = local_reasoning_levels();
    model.visibility = codex_protocol::openai_models::ModelVisibility::List;
    model.supported_in_api = true;
    model.priority = 36;
    model.additional_speed_tiers.clear();
    model.service_tiers.clear();
    model.default_service_tier = None;
    model.availability_nux = None;
    model.upgrade = None;
    model.include_apps_usage_instructions = false;
    model.supports_reasoning_summary_parameter = false;
    model.default_reasoning_summary = ReasoningSummary::None;
    model.support_verbosity = false;
    model.default_verbosity = None;
    model.apply_patch_tool_type = None;
    model.web_search_tool_type = WebSearchToolType::Text;
    model.supports_image_detail_original = false;
    model.context_window = Some(i64::from(context_window));
    model.max_context_window = Some(i64::from(context_window));
    model.auto_compact_token_limit = Some(i64::from(max_input_tokens));
    model.comp_hash = None;
    model.history_compatibility_group = Some("local_llama_cpp_qwen3_8".to_string());
    model.requires_nonempty_assistant_messages = false;
    model.effective_context_window_percent = 100;
    model.experimental_supported_tools.clear();
    model.disabled_tools = vec![
        ModelToolCapability::ToolSearch,
        ModelToolCapability::WebSearch,
        ModelToolCapability::ImageGeneration,
        ModelToolCapability::CodexApps,
    ];
    model.input_modalities = vec![InputModality::Text];
    model.used_fallback_model_metadata = false;
    model.supports_search_tool = false;
    model.use_responses_lite = false;
    model.node_repl_auto_review_required = false;
    model.node_repl_disabled = false;
    model.auto_review_model_override = None;
    model.model_specialty = None;
    model.tool_mode = Some(ToolMode::Direct);
    model.multi_agent_version = Some(MultiAgentVersion::V2);
    model.multi_agent_reasoning_effort = None;
    model
}

fn local_reasoning_levels() -> Vec<ReasoningEffortPreset> {
    [
        (ReasoningEffort::None, "Disable thinking"),
        (ReasoningEffort::Low, "Brief focused thinking"),
        (ReasoningEffort::Medium, "Balanced thinking"),
        (ReasoningEffort::XHigh, "Most extensive thinking"),
    ]
    .into_iter()
    .map(|(effort, description)| ReasoningEffortPreset {
        effort,
        description: description.to_string(),
    })
    .collect()
}

fn replace_or_append(models: &mut Vec<ModelInfo>, model: ModelInfo) {
    if let Some(index) = models
        .iter()
        .position(|candidate| candidate.slug == model.slug)
    {
        models[index] = model;
    } else {
        models.push(model);
    }
}

#[cfg(test)]
#[path = "llama_cpp_tests.rs"]
mod tests;
