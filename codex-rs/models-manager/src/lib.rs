pub(crate) mod cache;
pub mod collaboration_mode_presets;
pub(crate) mod config;
pub mod manager;
pub mod model_catalog_overlay;
pub mod model_info;
pub mod model_presets;
pub mod test_support;

pub use codex_protocol::auth::AuthMode;
pub use config::ModelsManagerConfig;
pub use model_catalog_overlay::ModelCatalogOverlay;
pub use model_catalog_overlay::ModelCatalogOverlayError;
pub use model_catalog_overlay::ResolvedModelCatalogOverlay;

/// Load the bundled model catalog shipped with `codex-models-manager`.
pub fn bundled_models_response()
-> std::result::Result<codex_protocol::openai_models::ModelsResponse, serde_json::Error> {
    let catalog = serde_json::from_str(include_str!("../models.json"))?;
    let catalog = ModelCatalogOverlay::from_json(include_str!("../claude_models.json"))
        .map_err(bundled_catalog_error)?
        .apply(catalog)
        .map_err(bundled_catalog_error)?;
    let catalog = ModelCatalogOverlay::from_json(include_str!("../kimi_models.json"))
        .map_err(bundled_catalog_error)?
        .apply(catalog)
        .map_err(bundled_catalog_error)?;
    let catalog = ModelCatalogOverlay::from_json(include_str!("../grok_models.json"))
        .map_err(bundled_catalog_error)?
        .apply(catalog)
        .map_err(bundled_catalog_error)?;
    ModelCatalogOverlay::from_json(include_str!("../local_models.json"))
        .map_err(bundled_catalog_error)?
        .apply(catalog)
        .map_err(bundled_catalog_error)
}

fn bundled_catalog_error(error: ModelCatalogOverlayError) -> serde_json::Error {
    serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::InvalidData, error))
}

/// Convert the client version string to a whole version string (e.g. "1.2.3-alpha.4" -> "1.2.3").
pub fn client_version_to_whole() -> String {
    format!(
        "{}.{}.{}",
        env!("CARGO_PKG_VERSION_MAJOR"),
        env!("CARGO_PKG_VERSION_MINOR"),
        env!("CARGO_PKG_VERSION_PATCH")
    )
}
