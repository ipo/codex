use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::Duration;
use codex_protocol::ThreadId;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::RolloutItem;
use codex_protocol::protocol::RolloutLine;
use codex_protocol::protocol::SessionMeta;
use codex_protocol::protocol::SessionMetaLine;
use codex_protocol::protocol::SessionSource;
use codex_protocol::protocol::UserMessageEvent;
use pretty_assertions::assert_eq;
use tempfile::TempDir;

use super::super::*;
use super::*;
use crate::tui::FrameRequester;

#[test]
fn modes_and_runtime_boundaries() {
    assert_eq!(
        [
            initial_page_load_mode(SessionPickerAction::Resume, false),
            initial_page_load_mode(SessionPickerAction::Resume, true),
            initial_page_load_mode(SessionPickerAction::Fork, false),
        ],
        [
            InitialPageLoadMode::IndexedThenReconcile,
            InitialPageLoadMode::AuthoritativeOnly,
            InitialPageLoadMode::AuthoritativeOnly,
        ]
    );

    let created = chrono::DateTime::parse_from_rfc3339("2026-01-01T00:00:00Z")
        .expect("timestamp")
        .with_timezone(&chrono::Utc);
    assert_eq!(
        [
            Some(created + Duration::seconds(59)),
            Some(created + Duration::minutes(17)),
            Some(created + Duration::hours(3) + Duration::minutes(4)),
            Some(created + Duration::days(2) + Duration::hours(5)),
            None,
            Some(created - Duration::seconds(1)),
        ]
        .map(|updated_at| format_estimated_runtime(Some(created), updated_at)),
        ["<1m", "17m", "3h 4m", "2d 5h", "—", "—"]
    );
    assert_eq!(format_estimated_runtime(None, Some(created)), "—");
}

#[tokio::test]
async fn indexed_page_reconciles_without_losing_query_or_selection() {
    let selected_id = ThreadId::new();
    let authoritative_selected = row(selected_id, "beta authoritative");
    let mut state = indexed_state();
    state.start_initial_load();
    state
        .handle_background_event(BackgroundEvent::IndexedPage {
            request_token: 0,
            page: page(
                vec![
                    row(ThreadId::new(), "alpha indexed"),
                    row(selected_id, "beta indexed"),
                ],
                Some("indexed-next"),
            ),
        })
        .await
        .unwrap();
    assert_eq!(
        (
            state.filtered_rows.len(),
            state.pagination.loading.is_pending(),
            state.pagination.next_cursor.is_none(),
            state.initial_page_state,
        ),
        (
            2,
            true,
            true,
            InitialPageLoadState::Provisional { request_token: 0 },
        )
    );

    state.selected = 1;
    state.set_query("beta".to_string());
    state
        .handle_background_event(BackgroundEvent::Page {
            request_token: 0,
            search_token: None,
            page: Ok(page(vec![authoritative_selected.clone()], None)),
        })
        .await
        .unwrap();
    assert_eq!(
        (
            state.query.clone(),
            state.filtered_rows.clone(),
            state.selected,
            state.pagination.loading.is_pending(),
            state.initial_page_state,
        ),
        (
            "beta".to_string(),
            vec![authoritative_selected],
            0,
            false,
            InitialPageLoadState::Authoritative,
        )
    );
}

#[tokio::test]
async fn indexed_and_authoritative_failures_have_safe_fallbacks() {
    let authoritative = row(ThreadId::new(), "authoritative fallback");
    let mut fallback_state = indexed_state();
    fallback_state.start_initial_load();
    fallback_state
        .handle_background_event(BackgroundEvent::Page {
            request_token: 0,
            search_token: None,
            page: Ok(page(vec![authoritative.clone()], None)),
        })
        .await
        .unwrap();
    assert_eq!(
        (
            fallback_state.all_rows,
            fallback_state.pagination.loading.is_pending(),
            fallback_state.search_state.is_active(),
            fallback_state.initial_page_state,
            fallback_state.inline_error,
        ),
        (
            vec![authoritative],
            false,
            false,
            InitialPageLoadState::Authoritative,
            None,
        )
    );

    let indexed = row(ThreadId::new(), "indexed result");
    let mut state = indexed_state();
    state.start_initial_load();
    state
        .handle_background_event(BackgroundEvent::IndexedPage {
            request_token: 0,
            page: page(vec![indexed.clone()], Some("ignored")),
        })
        .await
        .unwrap();
    state
        .handle_background_event(BackgroundEvent::Page {
            request_token: 0,
            search_token: None,
            page: Err(std::io::Error::other("refresh failed")),
        })
        .await
        .unwrap();
    assert_eq!(
        (
            state.all_rows.clone(),
            state.pagination.next_cursor.is_none(),
            state.inline_error.clone(),
        ),
        (
            vec![indexed],
            true,
            Some("Could not refresh sessions; showing indexed results".to_string()),
        )
    );
}

#[tokio::test]
async fn provisional_selection_validates_one_rollout_and_rejects_stale_rows() {
    let home = TempDir::new().unwrap();
    let cwd = home.path().join("workspace");
    let valid = rollout_row(
        home.path(),
        codex_rollout::SESSIONS_SUBDIR,
        &cwd,
        SessionSource::Cli,
        "openai",
        "select me",
    );
    let expected = SessionTarget {
        path: Some(std::fs::canonicalize(valid.path.as_ref().unwrap()).unwrap()),
        thread_id: valid.thread_id.unwrap(),
    };
    let mut state = indexed_state();
    state.filter_mode = SessionFilterMode::Cwd;
    state.filter_cwd = Some(cwd.clone());
    state.all_rows = vec![valid.clone()];
    state.filtered_rows = vec![valid.clone()];
    state.initial_page_state = InitialPageLoadState::Provisional { request_token: 4 };
    state.provisional_selection_policy = Some(ProvisionalSelectionPolicy {
        codex_home: home.path().to_path_buf(),
        include_non_interactive: false,
    });
    state.pagination.loading = LoadingState::Pending(PendingLoad {
        request_token: 4,
        search_token: None,
    });
    let selection = state
        .handle_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Enter,
            crossterm::event::KeyModifiers::NONE,
        ))
        .await
        .unwrap();
    assert_eq!(
        (selection, state.pagination.loading.is_pending()),
        (Some(SessionSelection::Resume(expected)), true)
    );

    let mut wrong_thread = valid.clone();
    wrong_thread.thread_id = Some(ThreadId::new());
    let make = |subdir, row_cwd: &Path, source, provider| {
        rollout_row(home.path(), subdir, row_cwd, source, provider, "stale")
    };
    let stale = [
        (
            indexed_row(
                home.path().join("sessions/missing.jsonl"),
                ThreadId::new(),
                &cwd,
                "missing",
            ),
            "",
        ),
        (wrong_thread, ""),
        (
            make(
                codex_rollout::SESSIONS_SUBDIR,
                &cwd,
                SessionSource::Exec,
                "openai",
            ),
            "",
        ),
        (
            make(
                codex_rollout::SESSIONS_SUBDIR,
                &cwd,
                SessionSource::Cli,
                "other",
            ),
            "",
        ),
        (
            make(
                codex_rollout::SESSIONS_SUBDIR,
                &home.path().join("other"),
                SessionSource::Cli,
                "openai",
            ),
            "",
        ),
        (valid.clone(), "not present"),
        (
            make(
                codex_rollout::ARCHIVED_SESSIONS_SUBDIR,
                &cwd,
                SessionSource::Cli,
                "openai",
            ),
            "",
        ),
    ];
    let mut results = Vec::new();
    for (row, query) in &stale {
        results.push(validate(row, home.path(), &cwd, query).await);
    }
    assert_eq!(
        results,
        vec![Err(ProvisionalSelectionError::Stale); stale.len()]
    );
}

fn indexed_state() -> PickerState {
    let mut state = PickerState::new(
        FrameRequester::test_dummy(),
        Arc::new(|_| {}),
        ProviderFilter::MatchDefault("openai".to_string()),
        /*show_all*/ true,
        /*filter_cwd*/ None,
        SessionPickerAction::Resume,
    );
    state.initial_page_mode = InitialPageLoadMode::IndexedThenReconcile;
    state
}

fn row(thread_id: ThreadId, preview: &str) -> Row {
    indexed_row(
        PathBuf::from(format!("/tmp/{thread_id}.jsonl")),
        thread_id,
        Path::new("/tmp/workspace"),
        preview,
    )
}

fn page(rows: Vec<Row>, next_cursor: Option<&str>) -> PickerPage {
    PickerPage {
        num_scanned_files: rows.len(),
        rows,
        next_cursor: next_cursor.map(|cursor| PageCursor::AppServer(cursor.to_string())),
        reached_scan_cap: false,
    }
}

async fn validate(
    row: &Row,
    codex_home: &Path,
    cwd_filter: &Path,
    query: &str,
) -> Result<SessionTarget, ProvisionalSelectionError> {
    let policy = ProvisionalSelectionPolicy {
        codex_home: codex_home.to_path_buf(),
        include_non_interactive: false,
    };
    let provider_filter = ProviderFilter::MatchDefault("openai".to_string());
    validate_provisional_selection(ProvisionalSelectionRequest {
        row,
        policy: &policy,
        provider_filter: &provider_filter,
        cwd_filter: Some(cwd_filter),
        query,
    })
    .await
}

fn rollout_row(
    home: &Path,
    subdir: &str,
    cwd: &Path,
    source: SessionSource,
    provider: &str,
    preview: &str,
) -> Row {
    let thread_id = ThreadId::new();
    let dir = home.join(subdir).join("2026/01/01");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("rollout-2026-01-01T00-00-00-{thread_id}.jsonl"));
    let meta = RolloutLine {
        timestamp: "2026-01-01T00:00:00Z".to_string(),
        ordinal: None,
        item: RolloutItem::SessionMeta(SessionMetaLine {
            meta: SessionMeta {
                session_id: thread_id.into(),
                id: thread_id,
                timestamp: "2026-01-01T00:00:00Z".to_string(),
                cwd: cwd.to_path_buf(),
                originator: "test".to_string(),
                cli_version: "test".to_string(),
                source,
                model_provider: Some(provider.to_string()),
                ..SessionMeta::default()
            },
            git: None,
        }),
    };
    let message = RolloutLine {
        timestamp: "2026-01-01T00:00:01Z".to_string(),
        ordinal: None,
        item: RolloutItem::EventMsg(EventMsg::UserMessage(UserMessageEvent {
            message: preview.to_string(),
            ..UserMessageEvent::default()
        })),
    };
    std::fs::write(
        &path,
        format!(
            "{}\n{}\n",
            serde_json::to_string(&meta).unwrap(),
            serde_json::to_string(&message).unwrap()
        ),
    )
    .unwrap();
    indexed_row(path, thread_id, cwd, preview)
}

fn indexed_row(path: PathBuf, thread_id: ThreadId, cwd: &Path, preview: &str) -> Row {
    Row {
        path: Some(path),
        preview: preview.to_string(),
        thread_id: Some(thread_id),
        thread_name: None,
        created_at: None,
        updated_at: None,
        cwd: Some(cwd.to_path_buf()),
        git_branch: None,
    }
}
