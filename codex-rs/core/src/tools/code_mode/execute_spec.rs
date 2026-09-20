use codex_code_mode::ImageDetailVisibility;
use codex_code_mode::ToolDefinition as CodeModeToolDefinition;
use codex_tools::FreeformTool;
use codex_tools::FreeformToolFormat;
use codex_tools::JsonSchema;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolSpec;
use std::collections::BTreeMap;

pub(crate) fn create_code_mode_tool(
    enabled_tools: &[CodeModeToolDefinition],
    deferred_tools: &[CodeModeToolDefinition],
    namespace_descriptions: &BTreeMap<String, codex_code_mode::ToolNamespaceDescription>,
    default_exec_yield_time_ms: u64,
    code_mode_only: bool,
    image_detail_visibility: ImageDetailVisibility,
    representation: ExecRepresentation,
) -> ToolSpec {
    const CODE_MODE_FREEFORM_GRAMMAR: &str = r#"
start: pragma_source | plain_source
pragma_source: PRAGMA_LINE NEWLINE SOURCE
plain_source: SOURCE

PRAGMA_LINE: /[ \t]*\/\/ @exec:[^\r\n]*/
NEWLINE: /\r?\n/
SOURCE: /[\s\S]+/
"#;

    let description = codex_code_mode::build_exec_tool_description(
        enabled_tools,
        deferred_tools,
        namespace_descriptions,
        default_exec_yield_time_ms,
        code_mode_only,
        image_detail_visibility,
    );
    if representation == ExecRepresentation::Function {
        return ToolSpec::Function(ResponsesApiTool {
            name: codex_code_mode::PUBLIC_TOOL_NAME.to_string(),
            description: description.replace(
                "- Accepts raw JavaScript source text, not JSON, quoted strings, or markdown code fences.",
                "- Pass complete JavaScript source as the `code` string argument; do not wrap the source in markdown code fences.",
            ).replace(
                "- You may optionally start the tool input with a first-line pragma",
                "- You may optionally start the `code` string with a first-line pragma",
            ),
            strict: false,
            defer_loading: None,
            parameters: JsonSchema::object(
                BTreeMap::from([(
                    "code".to_string(),
                    JsonSchema::string(Some(
                        "Complete JavaScript source, including an optional // @exec: pragma."
                            .to_string(),
                    )),
                )]),
                Some(vec!["code".to_string()]),
                Some(false.into()),
            ),
            output_schema: None,
        });
    }
    ToolSpec::Freeform(FreeformTool {
        name: codex_code_mode::PUBLIC_TOOL_NAME.to_string(),
        description,
        defer_loading: None,
        format: FreeformToolFormat {
            r#type: "grammar".to_string(),
            syntax: "lark".to_string(),
            definition: CODE_MODE_FREEFORM_GRAMMAR.to_string(),
        },
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExecRepresentation {
    Freeform,
    Function,
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_tools::ToolName;
    use pretty_assertions::assert_eq;

    #[test]
    fn create_code_mode_tool_matches_expected_spec() {
        let enabled_tools = vec![codex_code_mode::ToolDefinition {
            name: "update_plan".to_string(),
            tool_name: ToolName::plain("update_plan"),
            description: "Update the plan".to_string(),
            kind: codex_code_mode::CodeModeToolKind::Function,
            input_schema: None,
            output_schema: None,
        }];

        assert_eq!(
            create_code_mode_tool(
                &enabled_tools,
                &[],
                &BTreeMap::new(),
                codex_code_mode::DEFAULT_EXEC_YIELD_TIME_MS,
                /*code_mode_only*/ true,
                ImageDetailVisibility::Visible,
                ExecRepresentation::Freeform,
            ),
            ToolSpec::Freeform(FreeformTool {
                name: codex_code_mode::PUBLIC_TOOL_NAME.to_string(),
                description: codex_code_mode::build_exec_tool_description(
                    &enabled_tools,
                    &[],
                    &BTreeMap::new(),
                    codex_code_mode::DEFAULT_EXEC_YIELD_TIME_MS,
                    /*code_mode_only*/ true,
                    ImageDetailVisibility::Visible,
                ),
                defer_loading: None,
                format: FreeformToolFormat {
                    r#type: "grammar".to_string(),
                    syntax: "lark".to_string(),
                    definition: r#"
start: pragma_source | plain_source
pragma_source: PRAGMA_LINE NEWLINE SOURCE
plain_source: SOURCE

PRAGMA_LINE: /[ \t]*\/\/ @exec:[^\r\n]*/
NEWLINE: /\r?\n/
SOURCE: /[\s\S]+/
"#
                    .to_string(),
                },
            })
        );
    }
}
