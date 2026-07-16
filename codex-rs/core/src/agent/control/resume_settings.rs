use crate::config::Config;
use codex_protocol::protocol::RolloutItem;
use codex_protocol::protocol::SubagentBackendRoute;
use codex_thread_store::StoredThread;

pub(super) fn apply_stored_thread_runtime_settings(
    config: &mut Config,
    stored_thread: &StoredThread,
) {
    if config.model_provider_id != stored_thread.model_provider
        && let Some(model_provider) = config
            .model_providers
            .get(&stored_thread.model_provider)
            .cloned()
    {
        config.model_provider_id = stored_thread.model_provider.clone();
        config.model_provider = model_provider;
    }
    if let Some(model) = stored_thread.model.clone() {
        config.model = Some(model);
        config.model_reasoning_effort = stored_thread.reasoning_effort.clone();
    }
}

pub(super) fn stored_thread_uses_main_session_route(stored_thread: &StoredThread) -> bool {
    stored_thread.history.as_ref().and_then(|history| {
        history.items.iter().find_map(|item| match item {
            RolloutItem::SessionMeta(meta_line) => Some(meta_line.meta.subagent_backend_route),
            _ => None,
        })
    }) == Some(SubagentBackendRoute::MainSession)
}
