use std::path::Path;
use std::path::PathBuf;

use chrono::DateTime;
use chrono::Utc;
use codex_protocol::protocol::SessionSource;

use super::ProviderFilter;
use super::Row;
use super::SessionTarget;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum InitialPageLoadMode {
    IndexedThenReconcile,
    AuthoritativeOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ThreadListLookupMode {
    StateDbOnly,
    ScanAndRepair,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum InitialPageLoadState {
    Authoritative,
    WaitingForIndexed { request_token: usize },
    Provisional { request_token: usize },
}

impl InitialPageLoadState {
    pub(super) fn begin(mode: InitialPageLoadMode, request_token: usize) -> Self {
        match mode {
            InitialPageLoadMode::IndexedThenReconcile => Self::WaitingForIndexed { request_token },
            InitialPageLoadMode::AuthoritativeOnly => Self::Authoritative,
        }
    }

    pub(super) fn accept_indexed(&mut self, request_token: usize) -> bool {
        if *self == (Self::WaitingForIndexed { request_token }) {
            *self = Self::Provisional { request_token };
            true
        } else {
            false
        }
    }

    pub(super) fn is_provisional_for(self, request_token: usize) -> bool {
        self == Self::Provisional { request_token }
    }

    pub(super) fn is_provisional(self) -> bool {
        matches!(self, Self::Provisional { .. })
    }
}

#[derive(Clone)]
pub(super) struct ProvisionalSelectionPolicy {
    pub(super) codex_home: PathBuf,
    pub(super) include_non_interactive: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ProvisionalSelectionError {
    Stale,
}

pub(super) struct ProvisionalSelectionRequest<'a> {
    pub(super) row: &'a Row,
    pub(super) policy: &'a ProvisionalSelectionPolicy,
    pub(super) provider_filter: &'a ProviderFilter,
    pub(super) cwd_filter: Option<&'a Path>,
    pub(super) query: &'a str,
}

pub(super) async fn validate_provisional_selection(
    request: ProvisionalSelectionRequest<'_>,
) -> Result<SessionTarget, ProvisionalSelectionError> {
    let stale = || ProvisionalSelectionError::Stale;
    let thread_id = request.row.thread_id.ok_or_else(stale)?;
    let indexed_path = request.row.path.as_deref().ok_or_else(stale)?;
    let active_root = tokio::fs::canonicalize(
        request
            .policy
            .codex_home
            .join(codex_rollout::SESSIONS_SUBDIR),
    )
    .await
    .map_err(|_| stale())?;
    let physical_path = codex_rollout::existing_rollout_path(indexed_path)
        .await
        .ok_or_else(stale)?;
    let rollout_path = tokio::fs::canonicalize(physical_path)
        .await
        .map_err(|_| stale())?;
    if !rollout_path.starts_with(&active_root) {
        return Err(stale());
    }

    let item = codex_rollout::read_thread_item_from_rollout(rollout_path.clone())
        .await
        .ok_or_else(stale)?;
    if item.thread_id != Some(thread_id)
        || !source_is_eligible(item.source.as_ref(), request.policy.include_non_interactive)
        || !provider_is_eligible(item.model_provider.as_deref(), request.provider_filter)
        || !cwd_is_eligible(item.cwd.as_deref(), request.cwd_filter)
    {
        return Err(stale());
    }

    if !request.query.is_empty() {
        let thread_name =
            codex_rollout::find_thread_name_by_id(&request.policy.codex_home, &thread_id)
                .await
                .ok()
                .flatten();
        let refreshed = Row {
            path: Some(rollout_path.clone()),
            preview: item
                .preview
                .unwrap_or_else(|| "(no message yet)".to_string()),
            thread_id: Some(thread_id),
            thread_name,
            created_at: None,
            updated_at: None,
            cwd: item.cwd,
            git_branch: item.git_branch,
        };
        if !refreshed.matches_query(&request.query.to_lowercase()) {
            return Err(stale());
        }
    }

    Ok(SessionTarget {
        path: Some(rollout_path),
        thread_id,
    })
}

fn source_is_eligible(source: Option<&SessionSource>, include_non_interactive: bool) -> bool {
    matches!(source, Some(SessionSource::Cli | SessionSource::VSCode))
        || include_non_interactive
            && matches!(source, Some(SessionSource::Exec | SessionSource::Mcp))
}

fn provider_is_eligible(provider: Option<&str>, filter: &ProviderFilter) -> bool {
    match filter {
        ProviderFilter::Any => true,
        ProviderFilter::MatchDefault(default_provider) => {
            provider.unwrap_or(default_provider) == default_provider
        }
    }
}

fn cwd_is_eligible(cwd: Option<&Path>, filter: Option<&Path>) -> bool {
    match filter {
        Some(filter) => cwd.is_some_and(|cwd| super::paths_match(cwd, filter)),
        None => true,
    }
}

pub(super) fn format_estimated_runtime(
    created_at: Option<DateTime<Utc>>,
    updated_at: Option<DateTime<Utc>>,
) -> String {
    let Some(seconds) = created_at
        .zip(updated_at)
        .map(|(created_at, updated_at)| (updated_at - created_at).num_seconds())
        .filter(|seconds| *seconds >= 0)
    else {
        return "—".to_string();
    };
    if seconds < 60 {
        return "<1m".to_string();
    }
    let minutes = seconds / 60;
    if minutes < 60 {
        return format!("{minutes}m");
    }
    let hours = minutes / 60;
    if hours < 24 {
        return format!("{hours}h {}m", minutes % 60);
    }
    format!("{}d {}h", hours / 24, hours % 24)
}

#[cfg(test)]
#[path = "indexed_first_page_tests.rs"]
mod tests;
