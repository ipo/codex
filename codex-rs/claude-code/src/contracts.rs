use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct ClaudeFunctionTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ClaudeToolSpec {
    Function(ClaudeFunctionTool),
    Unsupported { kind: &'static str, name: String },
}

impl ClaudeToolSpec {
    pub fn name(&self) -> &str {
        match self {
            Self::Function(tool) => &tool.name,
            Self::Unsupported { name, .. } => name,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalOutcome {
    Completed,
    ToolsReady,
    Continue,
    OutputExhausted,
    Refusal,
}
