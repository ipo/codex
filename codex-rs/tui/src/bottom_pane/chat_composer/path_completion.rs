use std::fs;
use std::ops::Range;
use std::path::MAIN_SEPARATOR;
use std::path::Path;
use std::path::PathBuf;

use codex_utils_absolute_path::AbsolutePathBuf;
use crossterm::event::KeyCode;
use crossterm::event::KeyEvent;
use crossterm::event::KeyModifiers;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::widgets::WidgetRef;

use super::ChatComposer;
use super::InputResult;
use super::popup_state::ActivePopup;
use super::slash_input;
use crate::app_event::AppEvent;
use crate::bottom_pane::popup_consts::MAX_POPUP_ROWS;
use crate::bottom_pane::scroll_state::ScrollState;
use crate::bottom_pane::selection_popup_common::GenericDisplayRow;
use crate::bottom_pane::selection_popup_common::render_rows;
use crate::bottom_pane::textarea::TextArea;
use crate::key_hint::KeyBindingListExt;
use crate::render::Insets;
use crate::render::RectExt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PathCompletionTarget {
    pub(crate) range: Range<usize>,
    pub(crate) path_prefix: String,
    quoted: bool,
}

impl PathCompletionTarget {
    pub(crate) fn at_cursor(textarea: &TextArea) -> Option<Self> {
        let text = textarea.text();
        let cursor = textarea.cursor().min(text.len());
        if !text.is_char_boundary(cursor) {
            return None;
        }

        let mut boundary = 0;
        for element in textarea.text_element_ranges() {
            if element.start < cursor && cursor < element.end {
                return None;
            }
            if element.end <= cursor {
                boundary = element.end;
            } else if element.start == cursor {
                boundary = cursor;
                break;
            } else {
                break;
            }
        }

        let mut token_start = boundary;
        let mut in_double_quotes = false;
        for (offset, ch) in text[boundary..cursor].char_indices() {
            let absolute = boundary + offset;
            if ch == '"' {
                in_double_quotes = !in_double_quotes;
            } else if ch.is_whitespace() && !in_double_quotes {
                token_start = absolute + ch.len_utf8();
            }
        }

        let raw = text.get(token_start..cursor)?;
        let quoted = raw.starts_with('"');
        let unquoted = raw.strip_prefix('"').unwrap_or(raw);
        let path_prefix = unquoted.strip_suffix('"').unwrap_or(unquoted);
        if path_prefix.is_empty() {
            return None;
        }

        Some(Self {
            range: token_start..cursor,
            path_prefix: path_prefix.to_string(),
            quoted,
        })
    }

    pub(crate) fn replacement(&self, candidate: &str, draft: &str) -> String {
        let needs_quotes = self.quoted || candidate.chars().any(char::is_whitespace);
        if !needs_quotes || candidate.contains('"') {
            return candidate.to_string();
        }
        if self.quoted && draft[self.range.end..].starts_with('"') {
            format!("\"{candidate}")
        } else {
            format!("\"{candidate}\"")
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PathCompletionRequest {
    pub(crate) request_id: u64,
    pub(crate) cwd: AbsolutePathBuf,
    pub(crate) draft: String,
    pub(crate) cursor: usize,
    pub(crate) target: PathCompletionTarget,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PathCompletionCandidate {
    pub(crate) insertion: String,
    pub(crate) is_dir: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct PathCompletionResult {
    pub(crate) request_id: u64,
    pub(crate) candidates: Vec<PathCompletionCandidate>,
}

pub(crate) fn complete(request: PathCompletionRequest) -> PathCompletionResult {
    let candidates = completion_candidates(&request.cwd, &request.target.path_prefix);
    PathCompletionResult {
        request_id: request.request_id,
        candidates,
    }
}

fn completion_candidates(cwd: &AbsolutePathBuf, path_prefix: &str) -> Vec<PathCompletionCandidate> {
    completion_candidates_with_home(cwd, path_prefix, dirs::home_dir().as_deref())
}

fn completion_candidates_with_home(
    cwd: &AbsolutePathBuf,
    path_prefix: &str,
    home: Option<&Path>,
) -> Vec<PathCompletionCandidate> {
    let (directory_notation, name_prefix) = split_path_prefix(path_prefix);
    let Some(directory) = resolve_directory(cwd, directory_notation, home) else {
        return Vec::new();
    };
    let Ok(entries) = fs::read_dir(directory) else {
        return Vec::new();
    };
    let lowercase_prefix = name_prefix.to_lowercase();
    let separator = preferred_separator(path_prefix);
    let mut candidates = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.to_lowercase().starts_with(&lowercase_prefix) {
                return None;
            }
            let file_type = entry.file_type().ok()?;
            let is_dir = file_type.is_dir()
                || file_type.is_symlink()
                    && fs::metadata(entry.path()).is_ok_and(|metadata| metadata.is_dir());
            let mut insertion = format!("{directory_notation}{name}");
            if is_dir {
                insertion.push(separator);
            }
            Some(PathCompletionCandidate { insertion, is_dir })
        })
        .collect::<Vec<_>>();
    candidates.sort_by_key(|candidate| {
        (
            !candidate.is_dir,
            candidate.insertion.to_lowercase(),
            candidate.insertion.clone(),
        )
    });
    candidates.truncate(100);
    candidates
}

fn split_path_prefix(path_prefix: &str) -> (&str, &str) {
    path_prefix
        .char_indices()
        .rfind(|(_, ch)| is_separator(*ch))
        .map_or(("", path_prefix), |(index, ch)| {
            let end = index + ch.len_utf8();
            (&path_prefix[..end], &path_prefix[end..])
        })
}

fn resolve_directory(
    cwd: &AbsolutePathBuf,
    directory_notation: &str,
    home: Option<&Path>,
) -> Option<PathBuf> {
    if directory_notation.starts_with("~/")
        || cfg!(windows) && directory_notation.starts_with("~\\")
    {
        let home = home?;
        return Some(home.join(&directory_notation[2..]));
    }

    let directory = Path::new(directory_notation);
    if directory.is_absolute() {
        Some(directory.to_path_buf())
    } else {
        Some(cwd.join(directory).to_path_buf())
    }
}

fn preferred_separator(path_prefix: &str) -> char {
    path_prefix
        .chars()
        .rev()
        .find(|ch| is_separator(*ch))
        .unwrap_or(MAIN_SEPARATOR)
}

fn is_separator(ch: char) -> bool {
    ch == MAIN_SEPARATOR || cfg!(windows) && matches!(ch, '/' | '\\')
}

pub(crate) struct PathCompletionPopup {
    candidates: Vec<PathCompletionCandidate>,
    state: ScrollState,
}

impl PathCompletionPopup {
    pub(crate) fn new(candidates: Vec<PathCompletionCandidate>) -> Self {
        let mut state = ScrollState::new();
        state.clamp_selection(candidates.len());
        Self { candidates, state }
    }

    fn move_selection(&mut self, code: KeyCode) {
        match code {
            KeyCode::Up => self.state.move_up_wrap(self.candidates.len()),
            KeyCode::Down => self.state.move_down_wrap(self.candidates.len()),
            _ => return,
        }
        self.state
            .ensure_visible(self.candidates.len(), MAX_POPUP_ROWS);
    }

    pub(crate) fn selected(&self) -> Option<&PathCompletionCandidate> {
        self.state
            .selected_idx
            .and_then(|index| self.candidates.get(index))
    }

    pub(crate) fn calculate_required_height(&self) -> u16 {
        self.candidates.len().min(MAX_POPUP_ROWS) as u16
    }
}

impl WidgetRef for &PathCompletionPopup {
    fn render_ref(&self, area: Rect, buf: &mut Buffer) {
        let rows = self
            .candidates
            .iter()
            .map(|candidate| GenericDisplayRow {
                name: candidate.insertion.clone(),
                ..Default::default()
            })
            .collect::<Vec<_>>();
        render_rows(
            area.inset(Insets::tlbr(
                /*top*/ 0, /*left*/ 2, /*bottom*/ 0, /*right*/ 0,
            )),
            buf,
            &rows,
            &self.state,
            MAX_POPUP_ROWS,
            "no matches",
        );
    }
}

impl ChatComposer {
    pub(crate) fn on_path_completion_result(&mut self, result: PathCompletionResult) -> bool {
        let Some(pending) = self.pending_path_completion.take() else {
            return false;
        };
        if pending.request_id != result.request_id
            || self.draft.textarea.text() != pending.draft
            || self.draft.textarea.cursor() != pending.cursor
            || !matches!(self.popups.active, ActivePopup::None)
            || PathCompletionTarget::at_cursor(&self.draft.textarea).as_ref()
                != Some(&pending.target)
        {
            return false;
        }

        match result.candidates.as_slice() {
            [] => false,
            [candidate] => {
                self.apply_path_completion(&pending.target, &candidate.insertion);
                true
            }
            _ => {
                self.popups.active = ActivePopup::Path(PathCompletionPopup::new(result.candidates));
                true
            }
        }
    }

    pub(super) fn request_path_completion(&mut self) -> (InputResult, bool) {
        if self.draft.textarea.is_vim_normal_mode() {
            return (InputResult::None, true);
        }
        let Some(cwd) = self.cwd.clone() else {
            return (InputResult::None, true);
        };
        let Some(target) = PathCompletionTarget::at_cursor(&self.draft.textarea) else {
            return (InputResult::None, true);
        };
        let first_line = self.draft.textarea.text().lines().next().unwrap_or("");
        self.popups.dismissed_command_token =
            slash_input::command_popup_filter_text(first_line, /*cursor*/ 0);
        self.next_path_completion_request_id = self.next_path_completion_request_id.wrapping_add(1);
        let request = PathCompletionRequest {
            request_id: self.next_path_completion_request_id,
            cwd,
            draft: self.draft.textarea.text().to_string(),
            cursor: self.draft.textarea.cursor(),
            target,
        };
        self.pending_path_completion = Some(request.clone());
        self.app_event_tx
            .send(AppEvent::StartPathCompletion(request));
        (InputResult::None, true)
    }

    fn apply_path_completion(&mut self, target: &PathCompletionTarget, candidate: &str) {
        let replacement = target.replacement(candidate, self.draft.textarea.text());
        let start = target.range.start;
        self.draft
            .textarea
            .replace_range(target.range.clone(), &replacement);
        self.draft.textarea.set_cursor(start + replacement.len());
    }

    pub(super) fn handle_key_event_with_path_popup(
        &mut self,
        key_event: KeyEvent,
    ) -> (InputResult, bool) {
        let completion_pressed = self.complete_keys.is_pressed(key_event);
        let ActivePopup::Path(popup) = &mut self.popups.active else {
            unreachable!();
        };
        match key_event.code {
            KeyCode::Up | KeyCode::Down => {
                popup.move_selection(key_event.code);
                (InputResult::None, true)
            }
            KeyCode::Esc => {
                self.popups.active = ActivePopup::None;
                (InputResult::None, true)
            }
            KeyCode::Enter if key_event.modifiers == KeyModifiers::NONE && !completion_pressed => {
                self.accept_path_completion_popup()
            }
            _ if completion_pressed => self.accept_path_completion_popup(),
            _ => {
                self.popups.active = ActivePopup::None;
                self.handle_input_basic(key_event)
            }
        }
    }

    fn accept_path_completion_popup(&mut self) -> (InputResult, bool) {
        let ActivePopup::Path(popup) = &self.popups.active else {
            unreachable!();
        };
        let selection = popup
            .selected()
            .map(|candidate| candidate.insertion.clone());
        let target = PathCompletionTarget::at_cursor(&self.draft.textarea);
        self.popups.active = ActivePopup::None;
        if let (Some(selection), Some(target)) = (selection, target) {
            self.apply_path_completion(&target, &selection);
        }
        (InputResult::None, true)
    }
}

#[cfg(test)]
#[path = "path_completion_tests.rs"]
mod tests;
