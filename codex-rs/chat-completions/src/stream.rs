use codex_api::TerminalOutcome;
use serde_json::Value;

#[rustfmt::skip]
use crate::{stream_types::{DecodedStream, ToolCallFragment}, ChatCompletionChunk, ChunkUsage, DecodeError, DialectContext, DialectHooks, FinishReason, PendingResult, PresentationDelta, ResponseMetadata, ToolCall, ToolCallFunction, ToolCallKind, UsageMergePolicy};

pub struct DecodeStream<'a> {
    pub context: DialectContext<'a>,
    pub dialect: &'a (dyn DialectHooks + Sync),
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
    presented: bool,
    unpresented_arguments: String,
}

struct Decoder<'a, F> {
    params: DecodeStream<'a>,
    sink: F,
    response_id: Option<String>,
    content: String,
    reasoning: String,
    reasoning_provenance: Option<String>,
    tools: Vec<PendingTool>,
    usage: Option<ChunkUsage>,
    terminal: Option<(FinishReason, TerminalOutcome)>,
    recognized: bool,
    done: bool,
}

pub struct IncrementalDecoder<'a, F> {
    decoder: Decoder<'a, F>,
    framing: Framing,
    empty: bool,
    deferred_error: Option<DecodeError>,
}

impl<'a, F: FnMut(PresentationDelta)> IncrementalDecoder<'a, F> {
    pub fn new(params: DecodeStream<'a>, sink: F) -> Self {
        Self {
            decoder: Decoder {
                params,
                sink,
                response_id: None,
                content: String::new(),
                reasoning: String::new(),
                reasoning_provenance: None,
                tools: Vec::new(),
                usage: None,
                terminal: None,
                recognized: false,
                done: false,
            },
            framing: Framing::default(),
            empty: true,
            deferred_error: None,
        }
    }

    pub fn feed(&mut self, fragment: impl AsRef<[u8]>) -> Result<(), DecodeError> {
        let fragment = fragment.as_ref();
        let whitespace = fragment.iter().all(u8::is_ascii_whitespace);
        if !whitespace && let Some(error) = self.deferred_error.take() {
            return Err(error);
        }
        self.empty &= whitespace;
        if let Err(error) = self.framing.push(fragment, &mut self.decoder) {
            if !self.empty {
                return Err(error);
            }
            self.deferred_error = Some(error);
        }
        Ok(())
    }

    pub fn is_complete(&self) -> bool {
        self.decoder.done
    }

    pub fn finish(mut self) -> Result<DecodedStream, DecodeError> {
        if self.empty {
            return Err(DecodeError::EmptyStream);
        }
        if let Some(error) = self.deferred_error {
            return Err(error);
        }
        self.framing.finish(&mut self.decoder)?;
        self.decoder.finish()
    }
}

#[rustfmt::skip]
pub fn decode_stream<I, B, F>(params: DecodeStream<'_>, fragments: I, sink: F) -> Result<DecodedStream, DecodeError>
where
    I: IntoIterator<Item = B>,
    B: AsRef<[u8]>,
    F: FnMut(PresentationDelta),
{
    let mut decoder = IncrementalDecoder::new(params, sink);
    for fragment in fragments {
        decoder.feed(fragment)?;
    }
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
        if line.starts_with(':') { return Ok(()); }
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
        let usage_merge_policy = self
            .params
            .dialect
            .usage_merge_policy(self.params.context)?;
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
            merge_usage(&mut choice_usage, choice.usage, usage_merge_policy)?;
        }
        merge_usage(&mut choice_usage, chunk.usage, usage_merge_policy)?;
        merge_usage(&mut self.usage, choice_usage, usage_merge_policy)
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
            self.reasoning.push_str(&reasoning.text);
            self.reasoning_provenance = Some(reasoning.provenance);
            (self.sink)(PresentationDelta::Reasoning(reasoning.text));
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
        let arguments = if let Some(function) = fragment.function {
            set_once(&mut tool.name, function.name, "tool name")?;
            function.arguments
        } else {
            None
        };
        if let Some(arguments) = arguments {
            tool.arguments.push_str(&arguments);
            if tool.presented {
                (self.sink)(PresentationDelta::Tool {
                    index: fragment.index,
                    id: tool.id.clone().unwrap_or_default(),
                    name: tool.name.clone().unwrap_or_default(),
                    delta: arguments,
                });
            } else {
                tool.unpresented_arguments.push_str(&arguments);
            }
        }
        if !tool.presented
            && let (Some(id), Some(name)) = (&tool.id, &tool.name)
        {
            (self.sink)(PresentationDelta::Tool {
                index: fragment.index,
                id: id.clone(),
                name: name.clone(),
                delta: std::mem::take(&mut tool.unpresented_arguments),
            });
            tool.presented = true;
        }
        Ok(())
    }

    fn finish_reason(&mut self, raw_choice: &Value) -> Result<(), DecodeError> {
        let Some(value) = raw_choice.get("finish_reason") else {
            return Ok(());
        };
        if value.is_null() {
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
        if !self.done {
            return Err(DecodeError::PrematureEof {
                expected: "[DONE]".to_string(),
            });
        }
        let tools = complete_tools(self.tools)?;
        let terminal_outcome = self.terminal.map_or_else(
            || {
                if tools.is_empty() {
                    TerminalOutcome::Completed
                } else {
                    TerminalOutcome::ToolsReady
                }
            },
            |(_, outcome)| outcome,
        );
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
                reasoning_provenance: self.reasoning_provenance,
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
    policy: UsageMergePolicy,
) -> Result<(), DecodeError> {
    if let Some(usage) = usage {
        match target {
            None => *target = Some(usage),
            Some(existing) if existing == &usage => {}
            Some(_) => match policy {
                UsageMergePolicy::RequireIdentical => return Err(DecodeError::ConflictingUsage),
                UsageMergePolicy::PreferLatest => *target = Some(usage),
            },
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
