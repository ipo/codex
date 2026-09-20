use super::*;
use pretty_assertions::assert_eq;

#[test]
fn buffered_notifications_are_delivered_once_to_next_result() {
    let broker = CodeModeDispatchBroker::new(None);
    let cell_id = CellId::new("cell-1".to_string());
    broker
        .notifications
        .lock()
        .unwrap()
        .insert(cell_id.clone(), vec!["notification".to_string()]);
    let mut exec = FunctionToolOutput::from_text("exec yielded".to_string(), Some(true));
    broker.append_notifications(&cell_id, &mut exec, None);
    assert!(exec.into_text().contains("notification"));

    let mut wait = FunctionToolOutput::from_text("wait completed".to_string(), Some(true));
    broker.append_notifications(&cell_id, &mut wait, None);
    assert_eq!(wait.into_text(), "wait completed");

    broker
        .notifications
        .lock()
        .unwrap()
        .insert(cell_id.clone(), vec!["while waiting".to_string()]);
    let mut next_wait = FunctionToolOutput::from_text("wait completed".to_string(), Some(true));
    broker.append_notifications(&cell_id, &mut next_wait, None);
    assert!(next_wait.into_text().contains("while waiting"));
}
