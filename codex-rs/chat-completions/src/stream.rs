use codex_api::TerminalOutcome;
use serde_json::Value;

#[rustfmt::skip]
use crate::{stream_types::{DecodedStream, ToolCallFragment}, ChatCompletionChunk, ChunkUsage, DecodeError, DialectContext, DialectHooks, FinishReason, PendingResult, PresentationDelta, ResponseMetadata, ToolCall, ToolCallFunction, ToolCallKind};

pub struct DecodeStream<'a> {
    pub context: DialectContext<'a>,
    pub dialect: &'a dyn DialectHooks,
    pub metadata: ResponseMetadata,
}

#[derive(Default)]
struct Framing {
    line: Vec<u8>,
    data: Vec<String>,
    skip_lf: bool,
}

#[derive(Debug, Default)]
struct PendingTool {
    index: usize,
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

struct Decoder<'a, F> {
    params: DecodeStream<'a>,
    sink: F,
    response_id: Option<String>,
    content: String,
    reasoning: String,
    tools: Vec<PendingTool>,
    usage: Option<ChunkUsage>,
    terminal: Option<(FinishReason, TerminalOutcome)>,
    recognized: bool,
    saw_finish_field: bool,
    saw_null_finish: bool,
    done: bool,
}

#[rustfmt::skip]
pub fn decode_stream<I, B, F>(params: DecodeStream<'_>, fragments: I, sink: F) -> Result<DecodedStream, DecodeError>
where
    I: IntoIterator<Item = B>,
    B: AsRef<[u8]>,
    F: FnMut(PresentationDelta),
{
    let mut decoder = Decoder {
        params, sink, response_id: None, content: String::new(), reasoning: String::new(), tools: Vec::new(), usage: None,
        terminal: None, recognized: false, saw_finish_field: false, saw_null_finish: false, done: false,
    };
    let mut framing = Framing::default();
    let mut empty = true;
    let mut deferred_error = None;
    for fragment in fragments {
        let fragment = fragment.as_ref();
        let whitespace = fragment.iter().all(u8::is_ascii_whitespace);
        if !whitespace && let Some(error) = deferred_error { return Err(error); }
        empty &= whitespace;
        if let Err(error) = framing.push(fragment, &mut decoder) {
            if !empty { return Err(error); }
            deferred_error = Some(error);
        }
    }
    if empty { return Err(DecodeError::EmptyStream); }
    if let Some(error) = deferred_error { return Err(error); }
    framing.finish(&mut decoder)?;
    decoder.finish()
}

#[rustfmt::skip]
impl Framing {
    fn push<F: FnMut(PresentationDelta)>(&mut self, fragment: &[u8], decoder: &mut Decoder<'_, F>) -> Result<(), DecodeError> {
        for &byte in fragment {
            if self.skip_lf {
                self.skip_lf = false;
                if byte == b'\n' { continue; }
            }
            match byte {
                b'\r' => { self.line(decoder)?; self.skip_lf = true; }
                b'\n' => self.line(decoder)?,
                _ => self.line.push(byte),
            }
        }
        Ok(())
    }

    fn finish<F: FnMut(PresentationDelta)>(mut self, decoder: &mut Decoder<'_, F>) -> Result<(), DecodeError> {
        if !self.line.is_empty() { self.line(decoder)?; }
        self.emit(decoder)
    }

    fn line<F: FnMut(PresentationDelta)>(&mut self, decoder: &mut Decoder<'_, F>) -> Result<(), DecodeError> {
        if self.line.is_empty() { return self.emit(decoder); }
        let line = std::mem::take(&mut self.line);
        let line = std::str::from_utf8(&line).map_err(|_| DecodeError::InvalidUtf8)?;
        let value = line.strip_prefix("data:").ok_or_else(|| DecodeError::MalformedFraming(format!("unsupported stream line {line:?}")))?;
        self.data.push(value.strip_prefix(' ').unwrap_or(value).to_string());
        Ok(())
    }

    fn emit<F: FnMut(PresentationDelta)>(&mut self, decoder: &mut Decoder<'_, F>) -> Result<(), DecodeError> {
        if self.data.is_empty() { return Ok(()); }
        decoder.frame(&std::mem::take(&mut self.data).join("\n"))
    }
}

impl<F: FnMut(PresentationDelta)> Decoder<'_, F> {
    fn frame(&mut self, data: &str) -> Result<(), DecodeError> {
        if self.done {
            return Err(DecodeError::InvalidTransition(
                "data followed [DONE]".to_string(),
            ));
        }
        if data == "[DONE]" {
            self.done = true;
            return Ok(());
        }
        let raw: Value = serde_json::from_str(data)
            .map_err(|error| DecodeError::MalformedChunk(error.to_string()))?;
        let chunk: ChatCompletionChunk = serde_json::from_value(raw.clone())
            .map_err(|error| DecodeError::MalformedChunk(error.to_string()))?;
        self.recognized = true;
        self.set_response_id(chunk.id)?;
        let raw_choices = raw
            .get("choices")
            .and_then(Value::as_array)
            .ok_or_else(|| DecodeError::MalformedChunk("choices was not an array".to_string()))?;
        if raw_choices.len() != chunk.choices.len() {
            return Err(DecodeError::MalformedChunk(
                "choice representation mismatch".to_string(),
            ));
        }
        let mut choice_usage = None;
        for (choice, raw_choice) in chunk.choices.into_iter().zip(raw_choices) {
            if choice.index != 0 {
                return Err(DecodeError::InvalidTransition(format!(
                    "unsupported choice index {}",
                    choice.index
                )));
            }
            if self.terminal.is_some() {
                return self.terminal_conflict(raw_choice);
            }
            self.delta(choice.delta)?;
            self.finish_reason(raw_choice)?;
            merge_usage(&mut choice_usage, choice.usage)?;
        }
        merge_usage(&mut choice_usage, chunk.usage)?;
        merge_usage(&mut self.usage, choice_usage)
    }

    fn set_response_id(&mut self, id: String) -> Result<(), DecodeError> {
        match &self.response_id {
            None => self.response_id = Some(id),
            Some(existing) if existing == &id => {}
            Some(_) => {
                return Err(DecodeError::InvalidTransition(
                    "response id changed between chunks".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn delta(&mut self, delta: crate::ChunkDelta) -> Result<(), DecodeError> {
        if let Some(role) = delta.role
            && role != "assistant"
        {
            return Err(DecodeError::InvalidTransition(format!(
                "unexpected delta role {role}"
            )));
        }
        if let Some(content) = delta.content {
            self.content.push_str(&content);
            (self.sink)(PresentationDelta::Content(content));
        }
        if let Some(reasoning) = self
            .params
            .dialect
            .reasoning_delta(self.params.context, &delta.extensions)?
        {
            self.reasoning.push_str(&reasoning);
            (self.sink)(PresentationDelta::Reasoning(reasoning));
        }
        for fragment in delta.tool_calls {
            self.tool_fragment(fragment)?;
        }
        Ok(())
    }

    fn tool_fragment(&mut self, fragment: ToolCallFragment) -> Result<(), DecodeError> {
        let position = self
            .tools
            .iter()
            .position(|tool| tool.index == fragment.index)
            .unwrap_or_else(|| {
                self.tools.push(PendingTool {
                    index: fragment.index,
                    ..Default::default()
                });
                self.tools.len() - 1
            });
        let tool = &mut self.tools[position];
        set_once(&mut tool.id, fragment.id, "tool id")?;
        if let Some(kind) = fragment.kind
            && kind != "function"
        {
            return Err(DecodeError::InvalidTransition(format!(
                "unsupported tool type {kind}"
            )));
        }
        if let Some(function) = fragment.function {
            set_once(&mut tool.name, function.name, "tool name")?;
            if let Some(arguments) = function.arguments {
                tool.arguments.push_str(&arguments);
                (self.sink)(PresentationDelta::ToolArguments {
                    index: fragment.index,
                    delta: arguments,
                });
            }
        }
        Ok(())
    }

    fn finish_reason(&mut self, raw_choice: &Value) -> Result<(), DecodeError> {
        let Some(value) = raw_choice.get("finish_reason") else {
            return Ok(());
        };
        self.saw_finish_field = true;
        if value.is_null() {
            self.saw_null_finish = true;
            return Ok(());
        }
        let reason = value
            .as_str()
            .ok_or_else(|| DecodeError::MalformedChunk("finish_reason was not a string".into()))?;
        let reason = FinishReason::parse(reason)
            .ok_or_else(|| DecodeError::UnknownFinishReason(reason.to_string()))?;
        let outcome = self
            .params
            .dialect
            .finish_reason(self.params.context, reason)?;
        self.terminal = Some((reason, outcome));
        Ok(())
    }

    fn terminal_conflict(&self, raw_choice: &Value) -> Result<(), DecodeError> {
        match raw_choice.get("finish_reason").and_then(Value::as_str) {
            Some(reason) if FinishReason::parse(reason) == self.terminal.map(|value| value.0) => {
                Err(DecodeError::DuplicateTerminal)
            }
            _ => Err(DecodeError::ConflictingTerminal),
        }
    }

    fn finish(self) -> Result<DecodedStream, DecodeError> {
        if !self.recognized {
            return Err(DecodeError::PrematureEof {
                expected: "a recognized chunk".to_string(),
            });
        }
        let Some((_, terminal_outcome)) = self.terminal else {
            return Err(if self.saw_null_finish {
                DecodeError::NullFinishReason
            } else if !self.saw_finish_field {
                DecodeError::MissingFinishReason
            } else {
                DecodeError::PrematureEof {
                    expected: "a supported non-null finish reason".to_string(),
                }
            });
        };
        if !self.done {
            return Err(DecodeError::PrematureEof {
                expected: "[DONE]".to_string(),
            });
        }
        let tools = complete_tools(self.tools)?;
        validate_terminal(terminal_outcome, &self.content, &self.reasoning, &tools)?;
        let usage_details = self
            .usage
            .as_ref()
            .map(|usage| {
                self.params
                    .dialect
                    .usage_details(self.params.context, usage)
            })
            .transpose()?
            .unwrap_or_default();
        let pending = match terminal_outcome {
            TerminalOutcome::Completed
            | TerminalOutcome::ToolsReady
            | TerminalOutcome::Continue => Some(PendingResult {
                content: self.content,
                reasoning: self.reasoning,
                tool_calls: tools,
            }),
            TerminalOutcome::OutputExhausted | TerminalOutcome::Refusal => None,
        };
        Ok(DecodedStream {
            response_id: self.response_id.unwrap_or_default(),
            terminal_outcome,
            pending,
            usage: self.usage,
            usage_details,
            metadata: self.params.metadata,
        })
    }
}

fn set_once(
    target: &mut Option<String>,
    value: Option<String>,
    field: &str,
) -> Result<(), DecodeError> {
    if let Some(value) = value {
        match target {
            None => *target = Some(value),
            Some(existing) if existing == &value => {}
            Some(_) => {
                return Err(DecodeError::InvalidTransition(format!(
                    "conflicting {field} fragments"
                )));
            }
        }
    }
    Ok(())
}

fn merge_usage(
    target: &mut Option<ChunkUsage>,
    usage: Option<ChunkUsage>,
) -> Result<(), DecodeError> {
    if let Some(usage) = usage {
        match target {
            None => *target = Some(usage),
            Some(existing) if existing == &usage => {}
            Some(_) => return Err(DecodeError::ConflictingUsage),
        }
    }
    Ok(())
}

fn complete_tools(tools: Vec<PendingTool>) -> Result<Vec<ToolCall>, DecodeError> {
    tools
        .into_iter()
        .map(|tool| {
            let missing = |field: &str| DecodeError::IncompleteTool {
                index: tool.index,
                field: field.to_string(),
            };
            let id = tool.id.ok_or_else(|| missing("id"))?;
            let name = tool.name.ok_or_else(|| missing("name"))?;
            let value: Value = serde_json::from_str(&tool.arguments).map_err(|_| {
                DecodeError::InvalidToolJson {
                    index: tool.index,
                    arguments: tool.arguments.clone(),
                }
            })?;
            if !value.is_object() {
                return Err(DecodeError::InvalidToolJson {
                    index: tool.index,
                    arguments: tool.arguments,
                });
            }
            Ok(ToolCall {
                id,
                kind: ToolCallKind::Function,
                function: ToolCallFunction {
                    name,
                    arguments: tool.arguments,
                },
            })
        })
        .collect()
}

fn validate_terminal(
    outcome: TerminalOutcome,
    content: &str,
    reasoning: &str,
    tools: &[ToolCall],
) -> Result<(), DecodeError> {
    match outcome {
        TerminalOutcome::Completed | TerminalOutcome::Continue
            if tools.is_empty() && (!content.is_empty() || !reasoning.is_empty()) =>
        {
            Ok(())
        }
        TerminalOutcome::ToolsReady if !tools.is_empty() => Ok(()),
        TerminalOutcome::OutputExhausted | TerminalOutcome::Refusal => Ok(()),
        TerminalOutcome::Completed | TerminalOutcome::Continue => Err(
            DecodeError::InvalidTransition("successful terminal had no content".to_string()),
        ),
        TerminalOutcome::ToolsReady => Err(DecodeError::InvalidTransition(
            "tool terminal had no complete tools".to_string(),
        )),
    }
}
