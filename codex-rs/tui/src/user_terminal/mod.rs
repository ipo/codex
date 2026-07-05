//! Isolated terminal surface for user-owned shared shells.
//!
//! This module deliberately has no app-server or slash-command dependencies.
//! Wiring code supplies raw output bytes, receives raw PTY input bytes, and
//! propagates [`TerminalRenderOutcome::content_size`] through the resize RPC.

#![allow(dead_code)]

use crate::key_hint::KeyBinding;
use crate::key_hint::KeyBindingListExt;
use crate::terminal_palette::indexed_color;
use crate::terminal_palette::rgb_color;
use crossterm::event::KeyCode;
use crossterm::event::KeyEvent;
use crossterm::event::KeyEventKind;
use crossterm::event::KeyModifiers;
use ratatui::buffer::Buffer;
use ratatui::layout::Position;
use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::style::Modifier;
use ratatui::style::Style;
use ratatui::style::Stylize;
use ratatui::text::Line;
use ratatui::widgets::Clear;
use ratatui::widgets::Paragraph;
use ratatui::widgets::Widget;

const HEADER_ROWS: u16 = 1;
const HINT_ROWS: u16 = 1;
const CHROME_ROWS: u16 = HEADER_ROWS + HINT_ROWS;
const MIN_FRAME_HEIGHT: u16 = CHROME_ROWS + 1;
const DEFAULT_ROWS: u16 = 24;
const DEFAULT_COLS: u16 = 80;
const RESIZE_STEP_ROWS: u16 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TerminalLayoutMode {
    Half,
    Fullscreen,
    CustomHeight(u16),
}

impl TerminalLayoutMode {
    fn label(self) -> String {
        match self {
            Self::Half => "half".to_string(),
            Self::Fullscreen => "fullscreen".to_string(),
            Self::CustomHeight(height) => format!("{height} rows"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TerminalFocusMode {
    Input,
    FrameControls,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TerminalContentSize {
    pub(crate) rows: u16,
    pub(crate) cols: u16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TerminalMetadata {
    pub(crate) label: String,
    pub(crate) cwd: Option<String>,
    pub(crate) status: String,
}

impl TerminalMetadata {
    pub(crate) fn new(
        label: impl Into<String>,
        cwd: Option<String>,
        status: impl Into<String>,
    ) -> Self {
        Self {
            label: label.into(),
            cwd,
            status: status.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TerminalFrameAction {
    Dismiss,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TerminalInputOutcome {
    pub(crate) pty_input: Vec<u8>,
    pub(crate) frame_action: Option<TerminalFrameAction>,
    pub(crate) needs_resize: bool,
}

impl TerminalInputOutcome {
    fn empty() -> Self {
        Self {
            pty_input: Vec::new(),
            frame_action: None,
            needs_resize: false,
        }
    }

    fn pty_input(pty_input: Vec<u8>) -> Self {
        Self {
            pty_input,
            frame_action: None,
            needs_resize: false,
        }
    }

    fn resize() -> Self {
        Self {
            pty_input: Vec::new(),
            frame_action: None,
            needs_resize: true,
        }
    }

    fn frame_action(frame_action: TerminalFrameAction) -> Self {
        Self {
            pty_input: Vec::new(),
            frame_action: Some(frame_action),
            needs_resize: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TerminalRenderOutcome {
    pub(crate) frame_area: Rect,
    pub(crate) content_area: Rect,
    pub(crate) content_size: TerminalContentSize,
    pub(crate) cursor_position: Option<Position>,
}

pub(crate) struct UserTerminalSurface {
    metadata: TerminalMetadata,
    parser: vt100::Parser,
    layout_mode: TerminalLayoutMode,
    previous_non_fullscreen_layout: TerminalLayoutMode,
    focus_mode: TerminalFocusMode,
    open_controls: Vec<KeyBinding>,
    last_frame_height: u16,
}

impl UserTerminalSurface {
    pub(crate) fn new(metadata: TerminalMetadata, open_controls: Vec<KeyBinding>) -> Self {
        Self {
            metadata,
            parser: vt100::Parser::new(DEFAULT_ROWS, DEFAULT_COLS, /*scrollback_len*/ 0),
            layout_mode: TerminalLayoutMode::Half,
            previous_non_fullscreen_layout: TerminalLayoutMode::Half,
            focus_mode: TerminalFocusMode::Input,
            open_controls,
            last_frame_height: DEFAULT_ROWS / 2,
        }
    }

    pub(crate) fn metadata(&self) -> &TerminalMetadata {
        &self.metadata
    }

    pub(crate) fn focus_mode(&self) -> TerminalFocusMode {
        self.focus_mode
    }

    pub(crate) fn layout_mode(&self) -> TerminalLayoutMode {
        self.layout_mode
    }

    pub(crate) fn set_layout_mode(&mut self, layout_mode: TerminalLayoutMode) {
        if !matches!(layout_mode, TerminalLayoutMode::Fullscreen) {
            self.previous_non_fullscreen_layout = layout_mode;
        }
        self.layout_mode = layout_mode;
    }

    pub(crate) fn set_metadata(&mut self, metadata: TerminalMetadata) {
        self.metadata = metadata;
    }

    pub(crate) fn set_open_controls(&mut self, open_controls: Vec<KeyBinding>) {
        self.open_controls = open_controls;
    }

    pub(crate) fn process_output(&mut self, bytes: &[u8]) {
        self.parser.process(bytes);
    }

    pub(crate) fn handle_paste(&self, text: &str) -> TerminalInputOutcome {
        if self.focus_mode != TerminalFocusMode::Input {
            return TerminalInputOutcome::empty();
        }
        TerminalInputOutcome::pty_input(text.as_bytes().to_vec())
    }

    pub(crate) fn handle_key_event(&mut self, event: KeyEvent) -> TerminalInputOutcome {
        if !matches!(event.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
            return TerminalInputOutcome::empty();
        }

        match self.focus_mode {
            TerminalFocusMode::Input => {
                if self.open_controls.is_pressed(event) {
                    self.focus_mode = TerminalFocusMode::FrameControls;
                    return TerminalInputOutcome::empty();
                }
                TerminalInputOutcome::pty_input(encode_key_event(event))
            }
            TerminalFocusMode::FrameControls => self.handle_frame_control(event),
        }
    }

    fn handle_frame_control(&mut self, event: KeyEvent) -> TerminalInputOutcome {
        match event.code {
            KeyCode::Char('x' | 'X' | 'q' | 'Q') => {
                self.focus_mode = TerminalFocusMode::Input;
                TerminalInputOutcome::frame_action(TerminalFrameAction::Dismiss)
            }
            KeyCode::Char('f' | 'F') => {
                self.toggle_fullscreen();
                TerminalInputOutcome::resize()
            }
            KeyCode::Char('h' | 'H') => {
                self.set_layout_mode(TerminalLayoutMode::Half);
                TerminalInputOutcome::resize()
            }
            KeyCode::Char('+' | '=') => {
                self.grow_custom_height();
                TerminalInputOutcome::resize()
            }
            KeyCode::Char('-') => {
                self.shrink_custom_height();
                TerminalInputOutcome::resize()
            }
            KeyCode::Char(_) => TerminalInputOutcome::empty(),
            KeyCode::Esc => {
                self.focus_mode = TerminalFocusMode::Input;
                TerminalInputOutcome::empty()
            }
            KeyCode::Backspace
            | KeyCode::Enter
            | KeyCode::Left
            | KeyCode::Right
            | KeyCode::Up
            | KeyCode::Down
            | KeyCode::Home
            | KeyCode::End
            | KeyCode::PageUp
            | KeyCode::PageDown
            | KeyCode::Tab
            | KeyCode::BackTab
            | KeyCode::Delete
            | KeyCode::Insert
            | KeyCode::F(_)
            | KeyCode::Null
            | KeyCode::CapsLock
            | KeyCode::ScrollLock
            | KeyCode::NumLock
            | KeyCode::PrintScreen
            | KeyCode::Pause
            | KeyCode::Menu
            | KeyCode::KeypadBegin
            | KeyCode::Media(_)
            | KeyCode::Modifier(_) => TerminalInputOutcome::empty(),
        }
    }

    fn toggle_fullscreen(&mut self) {
        match self.layout_mode {
            TerminalLayoutMode::Fullscreen => {
                self.layout_mode = self.previous_non_fullscreen_layout;
            }
            TerminalLayoutMode::Half | TerminalLayoutMode::CustomHeight(_) => {
                self.previous_non_fullscreen_layout = self.layout_mode;
                self.layout_mode = TerminalLayoutMode::Fullscreen;
            }
        }
    }

    fn grow_custom_height(&mut self) {
        let base_height = self.custom_height_base();
        self.set_layout_mode(TerminalLayoutMode::CustomHeight(
            base_height.saturating_add(RESIZE_STEP_ROWS),
        ));
    }

    fn shrink_custom_height(&mut self) {
        let base_height = self.custom_height_base();
        self.set_layout_mode(TerminalLayoutMode::CustomHeight(
            base_height
                .saturating_sub(RESIZE_STEP_ROWS)
                .max(MIN_FRAME_HEIGHT),
        ));
    }

    fn custom_height_base(&self) -> u16 {
        match self.layout_mode {
            TerminalLayoutMode::Half | TerminalLayoutMode::Fullscreen => self.last_frame_height,
            TerminalLayoutMode::CustomHeight(height) => height,
        }
        .max(MIN_FRAME_HEIGHT)
    }

    pub(crate) fn render(&mut self, area: Rect, buf: &mut Buffer) -> TerminalRenderOutcome {
        let frame_area = self.frame_area(area);
        self.last_frame_height = frame_area.height;
        Clear.render(frame_area, buf);

        let content_area = content_area(frame_area);
        let content_size = TerminalContentSize {
            rows: content_area.height.max(1),
            cols: content_area.width.max(1),
        };
        self.parser
            .screen_mut()
            .set_size(content_size.rows, content_size.cols);

        self.render_header(frame_area, buf);
        self.render_terminal_content(content_area, buf);
        self.render_hints(frame_area, buf);

        let cursor_position = self.cursor_position(content_area);
        TerminalRenderOutcome {
            frame_area,
            content_area,
            content_size,
            cursor_position,
        }
    }

    pub(crate) fn content_size_for_area(&self, area: Rect) -> TerminalContentSize {
        let content_area = content_area(self.frame_area(area));
        TerminalContentSize {
            rows: content_area.height.max(1),
            cols: content_area.width.max(1),
        }
    }

    fn frame_area(&self, viewport: Rect) -> Rect {
        let height = match self.layout_mode {
            TerminalLayoutMode::Half => {
                if viewport.height <= MIN_FRAME_HEIGHT {
                    viewport.height
                } else {
                    viewport.height.div_ceil(2).max(MIN_FRAME_HEIGHT)
                }
            }
            TerminalLayoutMode::Fullscreen => viewport.height,
            TerminalLayoutMode::CustomHeight(height) => height
                .max(MIN_FRAME_HEIGHT.min(viewport.height))
                .min(viewport.height),
        };
        Rect::new(viewport.x, viewport.y, viewport.width, height)
    }

    fn render_header(&self, frame_area: Rect, buf: &mut Buffer) {
        if frame_area.height == 0 {
            return;
        }
        let row = Rect::new(frame_area.x, frame_area.y, frame_area.width, HEADER_ROWS);
        let cwd = self.metadata.cwd.as_deref().unwrap_or("cwd unknown");
        let line = Line::from(vec![
            "/sh ".cyan().bold(),
            self.metadata.label.clone().bold(),
            "  ".into(),
            self.layout_mode.label().dim(),
            "  ".into(),
            "shared full-access".cyan(),
            "  ".into(),
            self.metadata.status.clone().dim(),
            "  ".into(),
            cwd.to_string().dim(),
        ]);
        Paragraph::new(line).render(row, buf);
    }

    fn render_terminal_content(&self, content_area: Rect, buf: &mut Buffer) {
        if content_area.width == 0 || content_area.height == 0 {
            return;
        }

        let screen = self.parser.screen();
        for row in 0..content_area.height {
            for col in 0..content_area.width {
                let x = content_area.x + col;
                let y = content_area.y + row;
                let Some(cell) = screen.cell(row, col) else {
                    buf[(x, y)].set_symbol(" ");
                    continue;
                };
                if cell.is_wide_continuation() {
                    buf[(x, y)].set_symbol(" ");
                    continue;
                }
                let symbol = if cell.has_contents() {
                    cell.contents()
                } else {
                    " "
                };
                buf[(x, y)]
                    .set_symbol(symbol)
                    .set_style(style_for_vt100_cell(cell));
            }
        }
    }

    fn render_hints(&self, frame_area: Rect, buf: &mut Buffer) {
        if frame_area.height == 0 {
            return;
        }
        let row = Rect::new(
            frame_area.x,
            frame_area.bottom().saturating_sub(HINT_ROWS),
            frame_area.width,
            HINT_ROWS,
        );
        let line = match self.focus_mode {
            TerminalFocusMode::Input => {
                let controls = self
                    .open_controls
                    .first()
                    .map_or_else(|| "unbound".to_string(), KeyBinding::display_label);
                Line::from(vec![
                    " ".into(),
                    controls.dim(),
                    " controls".dim(),
                    "   ".into(),
                    "shell input".dim(),
                ])
            }
            TerminalFocusMode::FrameControls => Line::from(vec![
                " frame controls: ".cyan(),
                "x/q".dim(),
                " dismiss  ".dim(),
                "f".dim(),
                " fullscreen  ".dim(),
                "h".dim(),
                " half  ".dim(),
                "+/-".dim(),
                " resize  ".dim(),
                "esc".dim(),
                " input".dim(),
            ]),
        };
        Paragraph::new(line).render(row, buf);
    }

    fn cursor_position(&self, content_area: Rect) -> Option<Position> {
        if self.focus_mode != TerminalFocusMode::Input
            || content_area.width == 0
            || content_area.height == 0
        {
            return None;
        }
        let (row, col) = self.parser.screen().cursor_position();
        let x = content_area
            .x
            .saturating_add(col.min(content_area.width.saturating_sub(1)));
        let y = content_area
            .y
            .saturating_add(row.min(content_area.height.saturating_sub(1)));
        Some(Position { x, y })
    }
}

fn content_area(frame_area: Rect) -> Rect {
    let content_y = frame_area.y.saturating_add(HEADER_ROWS);
    let content_height = frame_area.height.saturating_sub(CHROME_ROWS);
    Rect::new(frame_area.x, content_y, frame_area.width, content_height)
}

fn encode_key_event(event: KeyEvent) -> Vec<u8> {
    let mut bytes = match event.code {
        KeyCode::Backspace => b"\x7f".to_vec(),
        KeyCode::Enter => b"\r".to_vec(),
        KeyCode::Left => b"\x1b[D".to_vec(),
        KeyCode::Right => b"\x1b[C".to_vec(),
        KeyCode::Up => b"\x1b[A".to_vec(),
        KeyCode::Down => b"\x1b[B".to_vec(),
        KeyCode::Home => b"\x1b[H".to_vec(),
        KeyCode::End => b"\x1b[F".to_vec(),
        KeyCode::PageUp => b"\x1b[5~".to_vec(),
        KeyCode::PageDown => b"\x1b[6~".to_vec(),
        KeyCode::Tab | KeyCode::BackTab => b"\t".to_vec(),
        KeyCode::Delete => b"\x1b[3~".to_vec(),
        KeyCode::Insert => b"\x1b[2~".to_vec(),
        KeyCode::Esc => b"\x1b".to_vec(),
        KeyCode::F(number) => encode_function_key(number),
        KeyCode::Char(ch) => encode_char(ch, event.modifiers),
        KeyCode::Null
        | KeyCode::CapsLock
        | KeyCode::ScrollLock
        | KeyCode::NumLock
        | KeyCode::PrintScreen
        | KeyCode::Pause
        | KeyCode::Menu
        | KeyCode::KeypadBegin
        | KeyCode::Media(_)
        | KeyCode::Modifier(_) => Vec::new(),
    };

    if event.modifiers.contains(KeyModifiers::ALT) && !matches!(event.code, KeyCode::Char(_)) {
        bytes.insert(0, b'\x1b');
    }
    bytes
}

fn encode_char(ch: char, modifiers: KeyModifiers) -> Vec<u8> {
    let mut bytes = if modifiers.contains(KeyModifiers::CONTROL) {
        encode_ctrl_char(ch).into_iter().collect()
    } else {
        let mut buf = [0; 4];
        ch.encode_utf8(&mut buf).as_bytes().to_vec()
    };

    if modifiers.contains(KeyModifiers::ALT) {
        bytes.insert(0, b'\x1b');
    }
    bytes
}

fn encode_ctrl_char(ch: char) -> Option<u8> {
    match ch {
        ' ' | '2' => Some(0x00),
        'a'..='z' => Some(ch as u8 - b'a' + 0x01),
        'A'..='Z' => Some(ch as u8 - b'A' + 0x01),
        '[' | '3' => Some(0x1b),
        '\\' | '4' => Some(0x1c),
        ']' | '5' => Some(0x1d),
        '^' | '6' => Some(0x1e),
        '_' | '7' | '/' => Some(0x1f),
        '8' | '?' => Some(0x7f),
        _ => None,
    }
}

fn encode_function_key(number: u8) -> Vec<u8> {
    match number {
        1 => b"\x1bOP".to_vec(),
        2 => b"\x1bOQ".to_vec(),
        3 => b"\x1bOR".to_vec(),
        4 => b"\x1bOS".to_vec(),
        5 => b"\x1b[15~".to_vec(),
        6 => b"\x1b[17~".to_vec(),
        7 => b"\x1b[18~".to_vec(),
        8 => b"\x1b[19~".to_vec(),
        9 => b"\x1b[20~".to_vec(),
        10 => b"\x1b[21~".to_vec(),
        11 => b"\x1b[23~".to_vec(),
        12 => b"\x1b[24~".to_vec(),
        13..=24 => format!("\x1b[{}~", number + 12).into_bytes(),
        0 | 25..=u8::MAX => Vec::new(),
    }
}

fn style_for_vt100_cell(cell: &vt100::Cell) -> Style {
    let mut style = Style::default();
    if let Some(color) = color_from_vt100(cell.fgcolor()) {
        style = style.fg(color);
    }
    if let Some(color) = color_from_vt100(cell.bgcolor()) {
        style = style.bg(color);
    }
    if cell.bold() {
        style = style.add_modifier(Modifier::BOLD);
    }
    if cell.dim() {
        style = style.add_modifier(Modifier::DIM);
    }
    if cell.italic() {
        style = style.add_modifier(Modifier::ITALIC);
    }
    if cell.underline() {
        style = style.add_modifier(Modifier::UNDERLINED);
    }
    if cell.inverse() {
        style = style.add_modifier(Modifier::REVERSED);
    }
    style
}

fn color_from_vt100(color: vt100::Color) -> Option<Color> {
    match color {
        vt100::Color::Default => None,
        vt100::Color::Idx(0) => Some(Color::Black),
        vt100::Color::Idx(1) => Some(Color::Red),
        vt100::Color::Idx(2) => Some(Color::Green),
        vt100::Color::Idx(3) => Some(Color::Yellow),
        vt100::Color::Idx(4) => Some(Color::Blue),
        vt100::Color::Idx(5) => Some(Color::Magenta),
        vt100::Color::Idx(6) => Some(Color::Cyan),
        vt100::Color::Idx(7) => Some(Color::Gray),
        vt100::Color::Idx(8) => Some(Color::DarkGray),
        vt100::Color::Idx(9) => Some(Color::LightRed),
        vt100::Color::Idx(10) => Some(Color::LightGreen),
        vt100::Color::Idx(11) => Some(Color::LightYellow),
        vt100::Color::Idx(12) => Some(Color::LightBlue),
        vt100::Color::Idx(13) => Some(Color::LightMagenta),
        vt100::Color::Idx(14) => Some(Color::LightCyan),
        vt100::Color::Idx(15) => Some(Color::White),
        vt100::Color::Idx(index) => Some(indexed_color(index)),
        vt100::Color::Rgb(red, green, blue) => Some(rgb_color((red, green, blue))),
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
