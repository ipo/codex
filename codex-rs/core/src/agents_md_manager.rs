use crate::agents_md::LoadedAgentsMd;
use crate::agents_md::load_project_instructions;
use crate::config::Config;
use crate::environment_selection::TurnEnvironmentSnapshot;
use crate::session::turn_context::TurnEnvironment;
use codex_exec_server::FileSystemSandboxContext;
use codex_extension_api::UserInstructions;
use codex_protocol::protocol::TurnEnvironmentSelection;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Owns the inputs and cached result of AGENTS.md discovery for a session.
pub(crate) struct AgentsMdManager {
    user_instructions: Option<UserInstructions>,
    cache: Mutex<AgentsMdCache>,
}

#[derive(Default)]
struct AgentsMdCache {
    selections: Option<Vec<TurnEnvironmentSelection>>,
    sandbox_contexts: Option<Vec<FileSystemSandboxContext>>,
    loaded: Option<Arc<LoadedAgentsMd>>,
}

impl AgentsMdManager {
    pub(crate) fn new(user_instructions: Option<UserInstructions>) -> Self {
        Self {
            user_instructions: user_instructions
                .filter(|instructions| !instructions.text.trim().is_empty()),
            cache: Mutex::new(AgentsMdCache::default()),
        }
    }

    #[tracing::instrument(name = "agents_md.refresh", skip_all)]
    pub(crate) async fn refresh(
        &self,
        config: &Config,
        environments: &TurnEnvironmentSnapshot,
        sandbox_context_for: impl Fn(&TurnEnvironment) -> FileSystemSandboxContext,
    ) {
        let selections = environments.to_selections();
        let sandbox_contexts = environments
            .turn_environments()
            .map(&sandbox_context_for)
            .collect::<Vec<_>>();
        {
            let cache = self.cache.lock().await;
            if cache.selections.as_ref() == Some(&selections)
                && cache.sandbox_contexts.as_ref() == Some(&sandbox_contexts)
            {
                return;
            }
        }

        let loaded = load_project_instructions(
            config,
            self.user_instructions.clone(),
            environments,
            sandbox_context_for,
        )
        .await
        .map(Arc::new);
        let mut cache = self.cache.lock().await;
        cache.selections = Some(selections);
        cache.sandbox_contexts = Some(sandbox_contexts);
        cache.loaded = loaded;
    }

    pub(crate) async fn get_loaded(&self) -> Option<Arc<LoadedAgentsMd>> {
        self.cache.lock().await.loaded.clone()
    }

    pub(crate) fn user_instructions(&self) -> Option<UserInstructions> {
        self.user_instructions.clone()
    }
}
