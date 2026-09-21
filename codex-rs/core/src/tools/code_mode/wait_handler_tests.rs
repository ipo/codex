use super::*;

use pretty_assertions::assert_eq;

#[test]
fn parses_integer_wait_arguments() {
    let arguments = parse_arguments::<ExecWaitArgs>(
        r#"{"cell_id":"cell","yield_time_ms":3300000,"max_tokens":5000,"terminate":true}"#,
    )
    .expect("integer wait arguments should parse");

    assert_eq!(
        arguments,
        ExecWaitArgs {
            cell_id: "cell".to_string(),
            yield_time_ms: IntegralValue(3_300_000),
            max_tokens: Some(IntegralValue(5_000)),
            terminate: true,
        }
    );
}

#[test]
fn parses_integral_float_wait_arguments() {
    let arguments = parse_arguments::<ExecWaitArgs>(
        r#"{"cell_id":"cell","yield_time_ms":3300000.0,"max_tokens":5000.0}"#,
    )
    .expect("integral floating-point wait arguments should parse");

    assert_eq!(
        arguments,
        ExecWaitArgs {
            cell_id: "cell".to_string(),
            yield_time_ms: IntegralValue(3_300_000),
            max_tokens: Some(IntegralValue(5_000)),
            terminate: false,
        }
    );
}

#[test]
fn preserves_omitted_wait_argument_defaults() {
    let arguments = parse_arguments::<ExecWaitArgs>(r#"{"cell_id":"cell"}"#)
        .expect("wait arguments with defaults should parse");

    assert_eq!(
        arguments,
        ExecWaitArgs {
            cell_id: "cell".to_string(),
            yield_time_ms: IntegralValue(DEFAULT_WAIT_YIELD_TIME_MS),
            max_tokens: None,
            terminate: false,
        }
    );
}

#[test]
fn rejects_invalid_wait_argument_numbers() {
    let max_tokens_overflow = (usize::MAX as u128) + 1;
    let invalid_arguments = [
        r#"{"cell_id":"cell","yield_time_ms":1.5}"#.to_string(),
        r#"{"cell_id":"cell","max_tokens":1.5}"#.to_string(),
        r#"{"cell_id":"cell","yield_time_ms":-1}"#.to_string(),
        r#"{"cell_id":"cell","max_tokens":-1.0}"#.to_string(),
        r#"{"cell_id":"cell","yield_time_ms":18446744073709551616.0}"#.to_string(),
        format!(r#"{{"cell_id":"cell","max_tokens":{max_tokens_overflow}}}"#),
        r#"{"cell_id":"cell","yield_time_ms":"3300000"}"#.to_string(),
        r#"{"cell_id":"cell","max_tokens":"5000"}"#.to_string(),
        r#"{"cell_id":"cell","max_tokens":null}"#.to_string(),
        r#"{"cell_id":1}"#.to_string(),
        r#"[]"#.to_string(),
        r#"{"cell_id":"cell"#.to_string(),
    ];

    for arguments in invalid_arguments {
        assert!(
            parse_arguments::<ExecWaitArgs>(&arguments).is_err(),
            "arguments should be rejected: {arguments}"
        );
    }
}
