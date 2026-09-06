use pretty_assertions::assert_eq;

use super::*;

#[test]
fn canonical_flat_name_normalizes_namespace_delimiters() {
    assert_eq!(
        ToolName::namespaced("mcp__calendar__", "__create_event")
            .canonical_flat_name()
            .as_ref(),
        "mcp__calendar__create_event"
    );
    assert_eq!(
        ToolName::namespaced("collaboration", "send_message")
            .canonical_flat_name()
            .as_ref(),
        "collaboration__send_message"
    );
}

#[test]
fn canonical_flat_name_preserves_default_namespace_names() {
    assert_eq!(
        ToolName::plain("exec_command").canonical_flat_name(),
        "exec_command"
    );
    assert_eq!(
        ToolName::namespaced(DEFAULT_FUNCTION_NAMESPACE, "exec_command").canonical_flat_name(),
        "exec_command"
    );
}
