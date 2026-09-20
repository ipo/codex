use super::parse_function_code;
use crate::function_tool::FunctionCallError;
use pretty_assertions::assert_eq;

#[test]
fn function_exec_extracts_complete_source() {
    let source =
        "// @exec: {\"yield_time_ms\": 1000}\ntext(await tools.exec_command({cmd: 'pwd'}));";
    assert_eq!(
        parse_function_code(&serde_json::json!({ "code": source }).to_string()).unwrap(),
        source
    );
}

#[test]
fn function_exec_rejects_invalid_arguments() {
    for (arguments, expected) in [
        ("{", "invalid exec JSON arguments"),
        ("{}", "exec requires a code field"),
        ("{\"code\":17}", "exec code must be a string"),
    ] {
        match parse_function_code(arguments) {
            Err(FunctionCallError::RespondToModel(message)) => {
                assert!(message.contains(expected), "{message}")
            }
            other => panic!("expected model-visible error, got {other:?}"),
        }
    }
}
