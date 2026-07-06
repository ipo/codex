use super::PreviousSectionState;
use super::WorldStateSection;
use crate::codex_thread::BackgroundTerminalInfo;
use crate::codex_thread::BackgroundTerminalSource;
use crate::codex_thread::BackgroundTerminalStatus;
use crate::context::ContextualUserFragment;
use crate::context::environment_context::push_xml_escaped_text;
use serde::Deserialize;
use serde::Serialize;
use std::collections::BTreeMap;

const MAX_ACTIVE_TERMINALS: usize = 16;
const MAX_CONTEXT_FIELD_CHARS: usize = 512;
const MAX_CONTEXT_OUTPUT_CHARS: usize = 2_000;
const SHARED_TERMINAL_SOURCE: &str = "sharedTerminal";
const EXITED_STATUS: &str = "exited";
const RUNNING_STATUS: &str = "running";
const UNAVAILABLE_STATUS: &str = "unavailable";

/// Bounded metadata for shared terminals currently available to the model.
#[derive(Clone, Debug, Default)]
pub(crate) struct ActiveTerminalsState {
    terminals: BTreeMap<String, ActiveTerminalSnapshot>,
}

impl ActiveTerminalsState {
    pub(crate) fn from_background_terminals(terminals: Vec<BackgroundTerminalInfo>) -> Self {
        let mut terminals = terminals
            .into_iter()
            .filter(|terminal| {
                terminal.source == BackgroundTerminalSource::SharedTerminal
                    && matches!(
                        terminal.status,
                        BackgroundTerminalStatus::Running | BackgroundTerminalStatus::Exited { .. }
                    )
            })
            .collect::<Vec<_>>();
        terminals.sort_by_key(|terminal| terminal.process_id.parse::<i32>().unwrap_or(i32::MAX));

        Self {
            terminals: terminals
                .into_iter()
                .take(MAX_ACTIVE_TERMINALS)
                .map(|terminal| {
                    let process_id = truncate_for_context(&terminal.process_id);
                    let (status, exit_code, final_output) = match terminal.status {
                        BackgroundTerminalStatus::Running => {
                            (RUNNING_STATUS.to_string(), None, None)
                        }
                        BackgroundTerminalStatus::Exited { exit_code } => (
                            EXITED_STATUS.to_string(),
                            exit_code,
                            terminal
                                .final_output
                                .as_deref()
                                .map(truncate_output_for_context),
                        ),
                    };
                    (
                        process_id.clone(),
                        ActiveTerminalSnapshot {
                            label: truncate_for_context(
                                terminal.label.as_deref().unwrap_or("default"),
                            ),
                            process_id,
                            cwd: truncate_for_context(&terminal.cwd.inferred_native_path_string()),
                            command: truncate_for_context(&terminal.command),
                            source: SHARED_TERMINAL_SOURCE.to_string(),
                            status,
                            exit_code,
                            final_output,
                        },
                    )
                })
                .collect(),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct ActiveTerminalsSnapshot {
    terminals: BTreeMap<String, ActiveTerminalSnapshot>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct ActiveTerminalSnapshot {
    label: String,
    process_id: String,
    cwd: String,
    command: String,
    source: String,
    status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    exit_code: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    final_output: Option<String>,
}

impl WorldStateSection for ActiveTerminalsState {
    const ID: &'static str = "active_terminals";
    type Snapshot = ActiveTerminalsSnapshot;

    fn snapshot(&self) -> Self::Snapshot {
        ActiveTerminalsSnapshot {
            terminals: self.terminals.clone(),
        }
    }

    fn render_diff(
        &self,
        previous: PreviousSectionState<'_, Self::Snapshot>,
    ) -> Option<Box<dyn ContextualUserFragment>> {
        let current = self.snapshot();
        let empty = ActiveTerminalsSnapshot::default();
        let previous = match previous {
            PreviousSectionState::Known(previous) => previous,
            PreviousSectionState::Absent | PreviousSectionState::Unknown => &empty,
        };
        if previous == &current {
            return None;
        }

        let mut updates = BTreeMap::new();
        for (process_id, terminal) in &current.terminals {
            if previous.terminals.get(process_id) != Some(terminal) {
                updates.insert(
                    process_id.clone(),
                    ActiveTerminalUpdate::Current(terminal.clone()),
                );
            }
        }
        for (process_id, terminal) in &previous.terminals {
            if !current.terminals.contains_key(process_id) {
                updates.insert(
                    process_id.clone(),
                    ActiveTerminalUpdate::Unavailable(UnavailableTerminalSnapshot {
                        label: terminal.label.clone(),
                        process_id: process_id.clone(),
                        source: terminal.source.clone(),
                    }),
                );
            }
        }

        (!updates.is_empty()).then(|| {
            Box::new(RenderedActiveTerminals { updates }) as Box<dyn ContextualUserFragment>
        })
    }
}

struct RenderedActiveTerminals {
    updates: BTreeMap<String, ActiveTerminalUpdate>,
}

enum ActiveTerminalUpdate {
    Current(ActiveTerminalSnapshot),
    Unavailable(UnavailableTerminalSnapshot),
}

struct UnavailableTerminalSnapshot {
    label: String,
    process_id: String,
    source: String,
}

impl ContextualUserFragment for RenderedActiveTerminals {
    fn role(&self) -> &'static str {
        "user"
    }

    fn markers(&self) -> (&'static str, &'static str) {
        Self::type_markers()
    }

    fn type_markers() -> (&'static str, &'static str) {
        active_terminals_markers()
    }

    fn body(&self) -> String {
        let mut rendered = "\n".to_string();
        for update in self.updates.values() {
            match update {
                ActiveTerminalUpdate::Current(terminal) => {
                    rendered.push_str("  <terminal label=\"");
                    push_xml_escaped_text(&mut rendered, &terminal.label);
                    rendered.push_str("\" process_id=\"");
                    push_xml_escaped_text(&mut rendered, &terminal.process_id);
                    rendered.push_str("\" source=\"");
                    push_xml_escaped_text(&mut rendered, &terminal.source);
                    rendered.push_str("\" status=\"");
                    push_xml_escaped_text(&mut rendered, &terminal.status);
                    if let Some(exit_code) = terminal.exit_code {
                        rendered.push_str("\" exit_code=\"");
                        push_xml_escaped_text(&mut rendered, &exit_code.to_string());
                    }
                    rendered.push_str("\">\n");
                    push_text_element(&mut rendered, "cwd", &terminal.cwd);
                    push_text_element(&mut rendered, "command", &terminal.command);
                    if let Some(final_output) = &terminal.final_output {
                        push_text_element(&mut rendered, "final_output", final_output);
                    }
                    rendered.push_str("  </terminal>\n");
                }
                ActiveTerminalUpdate::Unavailable(terminal) => {
                    rendered.push_str("  <terminal label=\"");
                    push_xml_escaped_text(&mut rendered, &terminal.label);
                    rendered.push_str("\" process_id=\"");
                    push_xml_escaped_text(&mut rendered, &terminal.process_id);
                    rendered.push_str("\" source=\"");
                    push_xml_escaped_text(&mut rendered, &terminal.source);
                    rendered.push_str("\" status=\"");
                    rendered.push_str(UNAVAILABLE_STATUS);
                    rendered.push_str("\" />\n");
                }
            }
        }
        rendered
    }
}

fn active_terminals_markers() -> (&'static str, &'static str) {
    ("<active_terminals>", "</active_terminals>")
}

fn push_text_element(rendered: &mut String, tag: &str, value: &str) {
    rendered.push_str("    <");
    rendered.push_str(tag);
    rendered.push('>');
    push_xml_escaped_text(rendered, value);
    rendered.push_str("</");
    rendered.push_str(tag);
    rendered.push_str(">\n");
}

fn truncate_for_context(value: &str) -> String {
    truncate_chars(value, MAX_CONTEXT_FIELD_CHARS)
}

fn truncate_output_for_context(value: &str) -> String {
    truncate_chars(value, MAX_CONTEXT_OUTPUT_CHARS)
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    let char_count = value.chars().count();
    if char_count <= max_chars {
        return value.to_string();
    }

    let take = max_chars.saturating_sub(3);
    let mut truncated = value.chars().take(take).collect::<String>();
    truncated.push_str("...");
    truncated
}

#[cfg(test)]
#[path = "active_terminals_tests.rs"]
mod tests;
