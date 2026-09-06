use std::sync::Arc;
use std::sync::Mutex;

use codex_tools::ToolSpec;

/// Per-router model-visible tools loaded by a function-form deferred-tool search.
#[derive(Default)]
pub(crate) struct DeferredToolLoadState {
    loaded_tools: Mutex<Arc<[ToolSpec]>>,
}

impl DeferredToolLoadState {
    pub(crate) fn loaded_tools(&self) -> Arc<[ToolSpec]> {
        self.loaded_tools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub(crate) fn replace(&self, tools: Vec<ToolSpec>) {
        *self
            .loaded_tools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = tools.into();
    }
}
