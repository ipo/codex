use std::collections::BTreeMap;

use serde_json::Value;

use super::request::KimiReasoning;
use super::request::KimiReasoningKey;
use super::request::KimiToolCall;
use super::stream_types::Chunk;
use super::stream_types::Delta;
use super::stream_types::KimiDecodedResponse;
use super::stream_types::KimiPendingResponse;
use super::stream_types::KimiStreamError;
use super::stream_types::KimiStreamEvent;
use super::stream_types::KimiTerminal;
use super::stream_types::KimiUsage;
use super::stream_types::ToolFragment;

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

/// A single-attempt decoder that presents deltas immediately but only returns
/// committable output after terminal validation and `[DONE]`.
pub struct KimiStreamDecoder {
    wire_model: String,
    trace_id: Option<String>,
    framing: Framing,
    response_id: Option<String>,
    content: String,
    reasoning: String,
    reasoning_key: Option<KimiReasoningKey>,
    tools: Vec<PendingTool>,
    usage: Option<KimiUsage>,
    terminal: Option<KimiTerminal>,
    recognized: bool,
    done: bool,
    empty: bool,
}

impl KimiStreamDecoder {
    pub fn new(wire_model: impl Into<String>) -> Self {
        Self {
            wire_model: wire_model.into(),
            trace_id: None,
            framing: Framing::default(),
            response_id: None,
            content: String::new(),
            reasoning: String::new(),
            reasoning_key: None,
            tools: Vec::new(),
            usage: None,
            terminal: None,
            recognized: false,
            done: false,
            empty: true,
        }
    }

    pub fn with_trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    pub fn feed(
        &mut self,
        fragment: impl AsRef<[u8]>,
    ) -> Result<Vec<KimiStreamEvent>, KimiStreamError> {
        let fragment = fragment.as_ref();
        self.empty &= fragment.iter().all(u8::is_ascii_whitespace);
        let mut events = Vec::new();
        for &byte in fragment {
            if self.framing.skip_lf {
                self.framing.skip_lf = false;
                if byte == b'\n' {
                    continue;
                }
            }
            match byte {
                b'\r' => {
                    self.process_line(&mut events)?;
                    self.framing.skip_lf = true;
                }
                b'\n' => self.process_line(&mut events)?,
                _ => self.framing.line.push(byte),
            }
        }
        Ok(events)
    }

    pub fn finish(mut self) -> Result<KimiDecodedResponse, KimiStreamError> {
        let mut ignored_events = Vec::new();
        if !self.framing.line.is_empty() {
            self.process_line(&mut ignored_events)?;
        }
        self.process_frame(&mut ignored_events)?;
        if self.empty {
            return Err(KimiStreamError::EmptyStream);
        }
        if !self.recognized {
            return Err(KimiStreamError::PrematureEof("a recognized chunk"));
        }
        if !self.done {
            return Err(KimiStreamError::PrematureEof("[DONE]"));
        }
        let tools = complete_tools(self.tools)?;
        let terminal = self.terminal.unwrap_or(if tools.is_empty() {
            KimiTerminal::Completed
        } else {
            KimiTerminal::ToolsReady
        });
        validate_terminal(terminal, &self.content, &tools)?;
        let pending = match terminal {
            KimiTerminal::Completed | KimiTerminal::ToolsReady => {
                let key = self
                    .reasoning_key
                    .unwrap_or(KimiReasoningKey::ReasoningContent);
                Some(KimiPendingResponse {
                    content: self.content,
                    reasoning: KimiReasoning::from_response(&self.wire_model, key, self.reasoning)?,
                    tool_calls: tools,
                })
            }
            KimiTerminal::OutputExhausted | KimiTerminal::Refusal => None,
        };
        Ok(KimiDecodedResponse {
            response_id: self.response_id.unwrap_or_default(),
            terminal,
            pending,
            usage: self.usage,
            trace_id: self.trace_id,
        })
    }

    fn process_line(&mut self, events: &mut Vec<KimiStreamEvent>) -> Result<(), KimiStreamError> {
        if self.framing.line.is_empty() {
            return self.process_frame(events);
        }
        let line = std::mem::take(&mut self.framing.line);
        let line = std::str::from_utf8(&line).map_err(|_| KimiStreamError::InvalidUtf8)?;
        if line.starts_with(':') {
            return Ok(());
        }
        let value = line
            .strip_prefix("data:")
            .ok_or_else(|| KimiStreamError::MalformedFraming(line.to_string()))?;
        self.framing
            .data
            .push(value.strip_prefix(' ').unwrap_or(value).to_string());
        Ok(())
    }

    fn process_frame(&mut self, events: &mut Vec<KimiStreamEvent>) -> Result<(), KimiStreamError> {
        if self.framing.data.is_empty() {
            return Ok(());
        }
        let data = std::mem::take(&mut self.framing.data).join("\n");
        if self.done {
            return Err(KimiStreamError::InvalidTransition("data followed [DONE]"));
        }
        if data == "[DONE]" {
            self.done = true;
            return Ok(());
        }
        let chunk: Chunk = serde_json::from_str(&data)
            .map_err(|error| KimiStreamError::MalformedChunk(error.to_string()))?;
        self.recognized = true;
        self.set_response_id(chunk.id)?;
        let mut choice_usage = None;
        for choice in chunk.choices {
            if choice.index != 0 {
                return Err(KimiStreamError::InvalidTransition(
                    "only choice index zero is supported",
                ));
            }
            if self.terminal.is_some() {
                return Err(KimiStreamError::DuplicateTerminal);
            }
            self.apply_delta(choice.delta, events)?;
            if let Some(reason) = choice.finish_reason {
                self.terminal = Some(parse_finish_reason(&reason)?);
            }
            if let Some(usage) = choice.usage {
                choice_usage = Some(usage.into_usage());
            }
        }
        if let Some(usage) = chunk.usage {
            choice_usage = Some(usage.into_usage());
        }
        if choice_usage.is_some() {
            self.usage = choice_usage;
        }
        Ok(())
    }

    fn set_response_id(&mut self, response_id: String) -> Result<(), KimiStreamError> {
        match &self.response_id {
            None => self.response_id = Some(response_id),
            Some(existing) if existing == &response_id => {}
            Some(_) => {
                return Err(KimiStreamError::InvalidTransition(
                    "response id changed between chunks",
                ));
            }
        }
        Ok(())
    }

    fn apply_delta(
        &mut self,
        delta: Delta,
        events: &mut Vec<KimiStreamEvent>,
    ) -> Result<(), KimiStreamError> {
        if let Some(role) = delta.role
            && role != "assistant"
        {
            return Err(KimiStreamError::InvalidTransition(
                "delta role was not assistant",
            ));
        }
        let response_id = self.response_id.clone().unwrap_or_default();
        if let Some(content) = delta.content
            && !content.is_empty()
        {
            self.content.push_str(&content);
            events.push(KimiStreamEvent::Content {
                response_id: response_id.clone(),
                delta: content,
            });
        }
        if let Some((key, reasoning)) = reasoning_delta(&delta.extensions)? {
            if self.reasoning_key.is_some_and(|existing| existing != key) {
                return Err(KimiStreamError::ConflictingReasoningKeys);
            }
            self.reasoning_key = Some(key);
            if !reasoning.is_empty() {
                self.reasoning.push_str(&reasoning);
                events.push(KimiStreamEvent::Reasoning {
                    response_id: response_id.clone(),
                    delta: reasoning,
                });
            }
        }
        for fragment in delta.tool_calls {
            self.apply_tool_fragment(fragment, &response_id, events)?;
        }
        Ok(())
    }

    fn apply_tool_fragment(
        &mut self,
        fragment: ToolFragment,
        response_id: &str,
        events: &mut Vec<KimiStreamEvent>,
    ) -> Result<(), KimiStreamError> {
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
        if fragment
            .kind
            .as_deref()
            .is_some_and(|kind| kind != "function")
        {
            return Err(KimiStreamError::InvalidTransition(
                "tool type was not function",
            ));
        }
        if let Some(function) = fragment.function {
            set_once(&mut tool.name, function.name, "tool name")?;
            if let Some(arguments) = function.arguments {
                tool.arguments.push_str(&arguments);
                if tool.presented {
                    events.push(tool_event(response_id, tool, arguments));
                } else {
                    tool.unpresented_arguments.push_str(&arguments);
                }
            }
        }
        if !tool.presented && tool.id.is_some() && tool.name.is_some() {
            let arguments_delta = std::mem::take(&mut tool.unpresented_arguments);
            events.push(tool_event(response_id, tool, arguments_delta));
            tool.presented = true;
        }
        Ok(())
    }
}

fn tool_event(response_id: &str, tool: &PendingTool, arguments_delta: String) -> KimiStreamEvent {
    KimiStreamEvent::ToolCall {
        response_id: response_id.to_string(),
        index: tool.index,
        id: tool.id.clone().unwrap_or_default(),
        name: tool.name.clone().unwrap_or_default(),
        arguments_delta,
    }
}

fn reasoning_delta(
    extensions: &BTreeMap<String, Value>,
) -> Result<Option<(KimiReasoningKey, String)>, KimiStreamError> {
    for (name, key) in [
        ("reasoning_content", KimiReasoningKey::ReasoningContent),
        ("reasoning", KimiReasoningKey::Reasoning),
        ("reasoning_details", KimiReasoningKey::ReasoningDetails),
    ] {
        if let Some(text) = extensions.get(name).and_then(Value::as_str) {
            return Ok(Some((key, text.to_string())));
        }
    }
    Ok(None)
}

fn parse_finish_reason(reason: &str) -> Result<KimiTerminal, KimiStreamError> {
    match reason {
        "stop" => Ok(KimiTerminal::Completed),
        "tool_calls" | "function_call" => Ok(KimiTerminal::ToolsReady),
        "length" | "max_tokens" => Ok(KimiTerminal::OutputExhausted),
        "content_filter" => Ok(KimiTerminal::Refusal),
        _ => Err(KimiStreamError::UnknownFinishReason(reason.to_string())),
    }
}

fn set_once(
    target: &mut Option<String>,
    value: Option<String>,
    field: &'static str,
) -> Result<(), KimiStreamError> {
    if let Some(value) = value {
        match target {
            None => *target = Some(value),
            Some(existing) if existing == &value => {}
            Some(_) => return Err(KimiStreamError::ConflictingToolField(field)),
        }
    }
    Ok(())
}

fn complete_tools(tools: Vec<PendingTool>) -> Result<Vec<KimiToolCall>, KimiStreamError> {
    let mut tools = tools;
    tools.sort_by_key(|tool| tool.index);
    if tools
        .iter()
        .enumerate()
        .any(|(expected, tool)| tool.index != expected)
    {
        return Err(KimiStreamError::InvalidTransition(
            "tool indexes were not contiguous from zero",
        ));
    }
    tools
        .into_iter()
        .map(|tool| {
            let id = tool.id.ok_or(KimiStreamError::IncompleteTool {
                index: tool.index,
                field: "id",
            })?;
            let name = tool.name.ok_or(KimiStreamError::IncompleteTool {
                index: tool.index,
                field: "name",
            })?;
            let arguments: Value = serde_json::from_str(&tool.arguments).map_err(|_| {
                KimiStreamError::InvalidToolJson {
                    index: tool.index,
                    arguments: tool.arguments.clone(),
                }
            })?;
            if !arguments.is_object() {
                return Err(KimiStreamError::InvalidToolJson {
                    index: tool.index,
                    arguments: tool.arguments,
                });
            }
            Ok(KimiToolCall::function(id, name, tool.arguments))
        })
        .collect()
}

fn validate_terminal(
    terminal: KimiTerminal,
    content: &str,
    tools: &[KimiToolCall],
) -> Result<(), KimiStreamError> {
    match terminal {
        KimiTerminal::Completed if tools.is_empty() && !content.is_empty() => Ok(()),
        KimiTerminal::ToolsReady if !tools.is_empty() => Ok(()),
        KimiTerminal::OutputExhausted | KimiTerminal::Refusal => Ok(()),
        KimiTerminal::Completed => Err(KimiStreamError::InvalidTransition(
            "successful terminal had no content",
        )),
        KimiTerminal::ToolsReady => Err(KimiStreamError::InvalidTransition(
            "tool terminal had no complete tools",
        )),
    }
}
