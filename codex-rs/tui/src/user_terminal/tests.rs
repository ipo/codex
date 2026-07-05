use super::*;
use crate::key_hint;
use crossterm::event::KeyEvent;
use pretty_assertions::assert_eq;
use ratatui::style::Modifier;

fn surface() -> UserTerminalSurface {
    UserTerminalSurface::new(
        TerminalMetadata::new("default", Some("/repo".to_string()), "running"),
        vec![key_hint::ctrl(KeyCode::Char('x'))],
    )
}

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

fn ctrl(ch: char) -> KeyEvent {
    KeyEvent::new(KeyCode::Char(ch), KeyModifiers::CONTROL)
}

fn alt(ch: char) -> KeyEvent {
    KeyEvent::new(KeyCode::Char(ch), KeyModifiers::ALT)
}

fn render_to_string(surface: &mut UserTerminalSurface, width: u16, height: u16) -> String {
    let area = Rect::new(0, 0, width, height);
    let mut buf = Buffer::empty(area);
    surface.render(area, &mut buf);
    buffer_to_string(&buf)
}

fn buffer_to_string(buf: &Buffer) -> String {
    let area = buf.area();
    (0..area.height)
        .map(|row| {
            let mut line = String::new();
            for col in 0..area.width {
                line.push_str(buf[(col, row)].symbol());
            }
            line.trim_end().to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn ansi_output_renders_cursor_movement_and_color() {
    let mut terminal = surface();
    terminal.process_output(b"hello\nworld\r\x1b[A\x1b[31mred");
    let area = Rect::new(0, 0, 40, 10);
    let mut buf = Buffer::empty(area);

    terminal.render(area, &mut buf);

    insta::assert_snapshot!("terminal_ansi_render", buffer_to_string(&buf));
    let styled_cell = buf[(0, HEADER_ROWS)].style();
    assert_eq!(styled_cell.fg, Some(Color::Red));
}

#[test]
fn focused_input_mode_places_host_cursor_at_emulated_cursor() {
    let mut terminal = surface();
    terminal.process_output(b"abc\nxy");
    let area = Rect::new(5, 3, 40, 12);
    let mut buf = Buffer::empty(area);

    let outcome = terminal.render(area, &mut buf);

    assert_eq!(
        outcome.cursor_position,
        Some(Position {
            x: 5 + 5,
            y: 3 + HEADER_ROWS + 1
        })
    );
}

#[test]
fn frame_control_mode_hides_host_cursor() {
    let mut terminal = surface();
    terminal.handle_key_event(ctrl('x'));
    let area = Rect::new(0, 0, 40, 10);
    let mut buf = Buffer::empty(area);

    let outcome = terminal.render(area, &mut buf);

    assert_eq!(outcome.cursor_position, None);
}

#[test]
fn common_shell_keys_encode_to_pty_bytes() {
    let cases = [
        (key(KeyCode::Enter), b"\r".as_slice()),
        (key(KeyCode::Backspace), b"\x7f".as_slice()),
        (key(KeyCode::Up), b"\x1b[A".as_slice()),
        (key(KeyCode::Down), b"\x1b[B".as_slice()),
        (key(KeyCode::Right), b"\x1b[C".as_slice()),
        (key(KeyCode::Left), b"\x1b[D".as_slice()),
        (ctrl('c'), b"\x03".as_slice()),
        (ctrl('d'), b"\x04".as_slice()),
        (alt('b'), b"\x1bb".as_slice()),
        (key(KeyCode::Home), b"\x1b[H".as_slice()),
        (key(KeyCode::End), b"\x1b[F".as_slice()),
        (key(KeyCode::PageUp), b"\x1b[5~".as_slice()),
        (key(KeyCode::PageDown), b"\x1b[6~".as_slice()),
        (key(KeyCode::Delete), b"\x1b[3~".as_slice()),
        (key(KeyCode::Esc), b"\x1b".as_slice()),
        (key(KeyCode::F(1)), b"\x1bOP".as_slice()),
        (key(KeyCode::F(12)), b"\x1b[24~".as_slice()),
    ];

    for (event, expected) in cases {
        assert_eq!(encode_key_event(event), expected);
    }
}

#[test]
fn paste_text_is_sent_as_raw_pty_input() {
    let terminal = surface();

    let outcome = terminal.handle_paste("echo pasted\n");

    assert_eq!(
        outcome,
        TerminalInputOutcome::pty_input(b"echo pasted\n".to_vec())
    );
}

#[test]
fn paste_text_is_consumed_in_frame_control_mode() {
    let mut terminal = surface();
    terminal.handle_key_event(ctrl('x'));

    let outcome = terminal.handle_paste("x");

    assert_eq!(outcome, TerminalInputOutcome::empty());
}

#[test]
fn open_controls_enters_frame_control_mode_without_pty_input() {
    let mut terminal = surface();

    let outcome = terminal.handle_key_event(ctrl('x'));

    assert_eq!(outcome, TerminalInputOutcome::empty());
    assert_eq!(terminal.focus_mode(), TerminalFocusMode::FrameControls);
}

#[test]
fn frame_control_keys_are_consumed_without_pty_input() {
    for event in [
        key(KeyCode::Char('x')),
        key(KeyCode::Char('q')),
        key(KeyCode::Char('f')),
        key(KeyCode::Char('h')),
        key(KeyCode::Char('+')),
        key(KeyCode::Char('=')),
        key(KeyCode::Char('-')),
        key(KeyCode::Esc),
    ] {
        let mut terminal = surface();
        terminal.handle_key_event(ctrl('x'));

        let outcome = terminal.handle_key_event(event);

        assert_eq!(outcome.pty_input, Vec::<u8>::new());
    }
}

#[test]
fn frame_control_actions_update_state() {
    let mut terminal = surface();
    terminal.handle_key_event(ctrl('x'));

    let fullscreen = terminal.handle_key_event(key(KeyCode::Char('f')));
    assert_eq!(terminal.layout_mode(), TerminalLayoutMode::Fullscreen);
    assert!(fullscreen.needs_resize);

    terminal.handle_key_event(key(KeyCode::Char('f')));
    assert_eq!(terminal.layout_mode(), TerminalLayoutMode::Half);

    terminal.handle_key_event(key(KeyCode::Char('+')));
    assert_eq!(terminal.layout_mode(), TerminalLayoutMode::CustomHeight(14));

    terminal.handle_key_event(key(KeyCode::Char('-')));
    assert_eq!(terminal.layout_mode(), TerminalLayoutMode::CustomHeight(12));

    terminal.handle_key_event(key(KeyCode::Char('h')));
    assert_eq!(terminal.layout_mode(), TerminalLayoutMode::Half);

    terminal.handle_key_event(key(KeyCode::Esc));
    assert_eq!(terminal.focus_mode(), TerminalFocusMode::Input);
}

#[test]
fn dismiss_keys_return_frame_action() {
    for key_code in [KeyCode::Char('x'), KeyCode::Char('q')] {
        let mut terminal = surface();
        terminal.handle_key_event(ctrl('x'));

        let outcome = terminal.handle_key_event(key(key_code));

        assert_eq!(
            outcome,
            TerminalInputOutcome::frame_action(TerminalFrameAction::Dismiss)
        );
        assert_eq!(terminal.focus_mode(), TerminalFocusMode::Input);
    }
}

#[test]
fn non_control_typing_goes_to_pty_input() {
    let mut terminal = surface();

    let outcome = terminal.handle_key_event(key(KeyCode::Char('a')));

    assert_eq!(outcome, TerminalInputOutcome::pty_input(b"a".to_vec()));
    assert_eq!(terminal.focus_mode(), TerminalFocusMode::Input);
}

#[test]
fn layouts_expose_updated_content_size() {
    let mut terminal = surface();
    let area = Rect::new(0, 0, 80, 20);

    assert_eq!(
        terminal.content_size_for_area(area),
        TerminalContentSize { rows: 8, cols: 80 }
    );

    terminal.set_layout_mode(TerminalLayoutMode::Fullscreen);
    assert_eq!(
        terminal.content_size_for_area(area),
        TerminalContentSize { rows: 18, cols: 80 }
    );

    terminal.set_layout_mode(TerminalLayoutMode::CustomHeight(7));
    assert_eq!(
        terminal.content_size_for_area(area),
        TerminalContentSize { rows: 5, cols: 80 }
    );
}

#[test]
fn layout_snapshots_cover_half_fullscreen_custom_and_controls() {
    let mut terminal = surface();
    terminal.process_output(b"prompt$ echo hi\nhi");
    insta::assert_snapshot!(
        "terminal_layout_half",
        render_to_string(&mut terminal, 60, 16)
    );

    terminal.set_layout_mode(TerminalLayoutMode::Fullscreen);
    insta::assert_snapshot!(
        "terminal_layout_fullscreen",
        render_to_string(&mut terminal, 60, 16)
    );

    terminal.set_layout_mode(TerminalLayoutMode::CustomHeight(7));
    insta::assert_snapshot!(
        "terminal_layout_custom",
        render_to_string(&mut terminal, 60, 16)
    );

    terminal.handle_key_event(ctrl('x'));
    insta::assert_snapshot!(
        "terminal_layout_frame_controls",
        render_to_string(&mut terminal, 60, 16)
    );
}

#[test]
fn style_renderer_sets_text_modifiers() {
    let mut terminal = surface();
    terminal.process_output(b"\x1b[1;4mbold underline");
    let area = Rect::new(0, 0, 40, 8);
    let mut buf = Buffer::empty(area);

    terminal.render(area, &mut buf);

    let style = buf[(0, HEADER_ROWS)].style();
    assert!(style.add_modifier.contains(Modifier::BOLD));
    assert!(style.add_modifier.contains(Modifier::UNDERLINED));
}
