use std::fs;
use std::path::MAIN_SEPARATOR;

use codex_utils_absolute_path::AbsolutePathBuf;
use pretty_assertions::assert_eq;
use tempfile::tempdir;
use tokio::sync::mpsc::UnboundedReceiver;

use super::*;
use crate::keymap::RuntimeKeymap;
use crate::render::renderable::Renderable;

fn absolute(path: &Path) -> AbsolutePathBuf {
    AbsolutePathBuf::try_from(path).unwrap()
}

fn insertions(cwd: &AbsolutePathBuf, prefix: &str, home: Option<&Path>) -> Vec<String> {
    completion_candidates_with_home(cwd, prefix, home)
        .into_iter()
        .map(|candidate| candidate.insertion)
        .collect()
}

fn composer(root: &Path, text: &str) -> (ChatComposer, UnboundedReceiver<AppEvent>) {
    let (mut composer, rx) = super::super::tests::new_test_composer();
    composer.set_cwd(absolute(root));
    composer.set_text_content(text.to_string(), Vec::new(), Vec::new());
    composer.move_cursor_to_end();
    (composer, rx)
}

fn request(
    composer: &mut ChatComposer,
    rx: &mut UnboundedReceiver<AppEvent>,
    key: KeyCode,
) -> PathCompletionRequest {
    composer.handle_key_event(KeyEvent::new(key, KeyModifiers::NONE));
    let AppEvent::StartPathCompletion(request) = rx.try_recv().unwrap() else {
        panic!("expected path completion request");
    };
    request
}

#[test]
fn target_parsing_and_replacement_respect_elements_quotes_and_cursor_suffix() {
    let mut textarea = TextArea::new();
    textarea.insert_str("before ");
    textarea.insert_element("[Image #1]");
    textarea.insert_str(" ./srTAIL");
    textarea.set_cursor(textarea.text().len() - 4);
    let target = PathCompletionTarget::at_cursor(&textarea).unwrap();
    assert_eq!(target.path_prefix, "./sr");
    assert_eq!(
        target.replacement("my file", textarea.text()),
        "\"my file\""
    );
}

#[test]
fn candidates_cover_notation_order_hidden_entries_symlinks_and_limit() {
    let root = tempdir().unwrap();
    let nested = root.path().join("nested");
    fs::create_dir(&nested).unwrap();
    fs::create_dir(nested.join("Source Dir")).unwrap();
    fs::write(nested.join("sample.txt"), "").unwrap();
    fs::write(nested.join(".secret"), "").unwrap();
    let cwd = absolute(root.path());
    assert_eq!(
        insertions(&cwd, "nested/S", Some(root.path())),
        ["nested/Source Dir/", "nested/sample.txt"]
    );
    let absolute_prefix = format!("{}/nested/.s", root.path().display());
    assert_eq!(
        insertions(&cwd, &absolute_prefix, None),
        [format!("{}/nested/.secret", root.path().display())]
    );
    assert_eq!(
        insertions(&cwd, "~/nested/sa", Some(root.path())),
        ["~/nested/sample.txt"]
    );

    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&nested, root.path().join("linked-dir")).unwrap();
        assert_eq!(
            insertions(&cwd, "linked", None),
            [format!("linked-dir{MAIN_SEPARATOR}")]
        );
    }
    for index in 0..105 {
        fs::write(root.path().join(format!("match-{index:03}")), "").unwrap();
    }
    let limited = completion_candidates_with_home(&cwd, "match-", None);
    assert_eq!(limited.len(), 100);
}

#[test]
fn composer_applies_single_match_ignores_none_and_uses_latest_cwd() {
    let first = tempdir().unwrap();
    let second = tempdir().unwrap();
    fs::create_dir(first.path().join("src")).unwrap();
    fs::write(second.path().join("match-second"), "").unwrap();
    let (mut composer, mut rx) = composer(first.path(), "./sr");
    let completion = request(&mut composer, &mut rx, KeyCode::Tab);
    assert!(composer.on_path_completion_result(complete(completion)));
    assert_eq!(composer.current_text(), "./src/");

    composer.set_text_content("missing".to_string(), Vec::new(), Vec::new());
    composer.move_cursor_to_end();
    let completion = request(&mut composer, &mut rx, KeyCode::Tab);
    assert!(!composer.on_path_completion_result(complete(completion)));
    composer.set_text_content("match-".to_string(), Vec::new(), Vec::new());
    composer.move_cursor_to_end();
    let stale = request(&mut composer, &mut rx, KeyCode::Tab);
    composer.set_cwd(absolute(second.path()));
    assert!(!composer.on_path_completion_result(complete(stale)));
    let current = request(&mut composer, &mut rx, KeyCode::Tab);
    assert!(composer.on_path_completion_result(complete(current)));
    assert_eq!(composer.current_text(), "match-second");
}

#[test]
fn stale_results_are_rejected_after_cursor_submission_or_popup_transition() {
    for invalidate in [KeyCode::Left, KeyCode::Enter] {
        let root = tempdir().unwrap();
        let (mut composer, mut rx) = composer(root.path(), "./sr");
        let stale = request(&mut composer, &mut rx, KeyCode::Tab);
        composer.handle_key_event(KeyEvent::new(invalidate, KeyModifiers::NONE));
        assert!(!composer.on_path_completion_result(complete(stale)));
    }

    let root = tempdir().unwrap();
    let (mut composer, mut rx) = composer(root.path(), "./sr");
    let stale = request(&mut composer, &mut rx, KeyCode::Tab);
    composer.popups.active = ActivePopup::Path(PathCompletionPopup::new(Vec::new()));
    assert!(!composer.on_path_completion_result(complete(stale)));
}

#[test]
fn configured_key_controls_popup_selection() {
    let root = tempdir().unwrap();
    fs::create_dir(root.path().join("src")).unwrap();
    fs::write(root.path().join("script.rs"), "").unwrap();
    let (mut composer, mut rx) = composer(root.path(), "./s");
    let mut keymap = RuntimeKeymap::defaults();
    keymap.composer.complete = vec![crate::key_hint::plain(KeyCode::F(2))];
    composer.set_keymap_bindings(&keymap);
    composer.handle_key_event(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
    assert!(rx.try_recv().is_err());
    let completion = request(&mut composer, &mut rx, KeyCode::F(2));
    assert!(composer.on_path_completion_result(complete(completion)));
    let width = 60;
    let height = composer.desired_height(width);
    let mut terminal =
        ratatui::Terminal::new(ratatui::backend::TestBackend::new(width, height)).unwrap();
    terminal
        .draw(|frame| composer.render(frame.area(), frame.buffer_mut()))
        .unwrap();
    insta::with_settings!({snapshot_path => "../snapshots"}, {
        insta::assert_snapshot!("filesystem_path_completion_popup", terminal.backend());
    });
    composer.handle_key_event(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
    composer.handle_key_event(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    assert_eq!(composer.current_text(), "./script.rs");
}
