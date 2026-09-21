use std::fmt;
use std::marker::PhantomData;

use serde::Deserialize;
use serde::Deserializer;
use serde::de::Visitor;

use crate::function_tool::FunctionCallError;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolOutput;
use crate::tools::context::ToolPayload;
use crate::tools::context::boxed_tool_output;
use crate::tools::registry::CoreToolRuntime;
use crate::tools::registry::PostToolUsePayload;
use crate::tools::registry::PreToolUsePayload;
use crate::tools::registry::ToolExecutor;
use codex_tools::ToolName;
use codex_tools::ToolSpec;

use super::DEFAULT_WAIT_YIELD_TIME_MS;
use super::ExecContext;
use super::WAIT_TOOL_NAME;
use super::handle_runtime_response;
use super::telemetry::CodeModeToolCallGuard;
use super::wait_spec::create_wait_tool;

pub struct CodeModeWaitHandler;

#[derive(Debug, Deserialize, Eq, PartialEq)]
struct ExecWaitArgs {
    cell_id: String,
    #[serde(default = "default_wait_yield_time_ms")]
    yield_time_ms: IntegralValue<u64>,
    #[serde(default, deserialize_with = "deserialize_optional_integral_value")]
    max_tokens: Option<IntegralValue<usize>>,
    #[serde(default)]
    terminate: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct IntegralValue<T>(T);

impl<'de, T> Deserialize<'de> for IntegralValue<T>
where
    T: TryFrom<u64>,
    T::Error: fmt::Display,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(IntegralValueVisitor(PhantomData))
    }
}

struct IntegralValueVisitor<T>(PhantomData<T>);

impl<T> Visitor<'_> for IntegralValueVisitor<T>
where
    T: TryFrom<u64>,
    T::Error: fmt::Display,
{
    type Value = IntegralValue<T>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a non-negative whole number within range")
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        u64::try_from(value)
            .map_err(E::custom)
            .and_then(|value| self.visit_u64(value))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        T::try_from(value).map(IntegralValue).map_err(E::custom)
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        if value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value < u64::MAX as f64 {
            return self.visit_u64(value as u64);
        }

        Err(E::custom(
            "expected a non-negative whole number within range",
        ))
    }
}

fn default_wait_yield_time_ms() -> IntegralValue<u64> {
    IntegralValue(DEFAULT_WAIT_YIELD_TIME_MS)
}

fn deserialize_optional_integral_value<'de, D>(
    deserializer: D,
) -> Result<Option<IntegralValue<usize>>, D::Error>
where
    D: Deserializer<'de>,
{
    IntegralValue::deserialize(deserializer).map(Some)
}

fn parse_arguments<T>(arguments: &str) -> Result<T, FunctionCallError>
where
    T: for<'de> Deserialize<'de>,
{
    serde_json::from_str(arguments).map_err(|err| {
        FunctionCallError::RespondToModel(format!("failed to parse function arguments: {err}"))
    })
}

impl ToolExecutor<ToolInvocation> for CodeModeWaitHandler {
    fn tool_name(&self) -> ToolName {
        ToolName::plain(WAIT_TOOL_NAME)
    }

    fn spec(&self) -> ToolSpec {
        create_wait_tool()
    }

    fn handle<'a>(&'a self, invocation: ToolInvocation) -> codex_tools::ToolExecutorFuture<'a>
    where
        ToolInvocation: 'a,
    {
        Box::pin(self.handle_call(invocation))
    }
}

impl CodeModeWaitHandler {
    async fn handle_call(
        &self,
        invocation: ToolInvocation,
    ) -> Result<Box<dyn crate::tools::context::ToolOutput>, FunctionCallError> {
        let ToolInvocation {
            session,
            turn,
            call_id,
            tool_name,
            payload,
            ..
        } = invocation;

        let mut telemetry = CodeModeToolCallGuard::new(
            session.services.analytics_events_client.clone(),
            session.thread_id.to_string(),
            turn.sub_id.clone(),
            turn.turn_metadata_state.clone(),
            call_id.clone(),
            WAIT_TOOL_NAME,
        );
        let result = match payload {
            ToolPayload::Function { arguments }
                if tool_name.is_default_namespace()
                    && tool_name.name.as_str() == WAIT_TOOL_NAME =>
            {
                let args: ExecWaitArgs = parse_arguments(&arguments).inspect_err(|_error| {
                    telemetry.finish(/*success*/ false);
                })?;
                let exec = ExecContext { session, turn };
                let started_at = std::time::Instant::now();
                telemetry.cell_id = Some(args.cell_id.clone());
                let cell_id = codex_code_mode::CellId::new(args.cell_id.clone());
                let wait_response = if args.terminate {
                    exec.session
                        .services
                        .code_mode_service
                        .terminate(cell_id)
                        .await
                } else {
                    exec.session
                        .services
                        .code_mode_service
                        .wait(codex_code_mode::WaitRequest {
                            cell_id,
                            yield_time_ms: args.yield_time_ms.0,
                        })
                        .await
                }
                .map_err(|error| {
                    telemetry.finish(/*success*/ false);
                    FunctionCallError::RespondToModel(error)
                })?;
                if let codex_code_mode::WaitOutcome::LiveCell(response) = &wait_response {
                    let runtime_cell_id = match response {
                        codex_code_mode::RuntimeResponse::Yielded { cell_id, .. }
                        | codex_code_mode::RuntimeResponse::Terminated { cell_id, .. }
                        | codex_code_mode::RuntimeResponse::Result { cell_id, .. } => cell_id,
                    };
                    telemetry.cell_id = Some(runtime_cell_id.to_string());
                    if let Some(executed_tool_calls) =
                        exec.session.services.executed_tool_calls.as_ref()
                    {
                        executed_tool_calls.register_cell(runtime_cell_id, &call_id);
                    }
                    if !matches!(response, codex_code_mode::RuntimeResponse::Yielded { .. }) {
                        exec.session
                            .services
                            .rollout_thread_trace
                            .code_cell_trace_context(
                                exec.turn.sub_id.as_str(),
                                runtime_cell_id.as_str(),
                            )
                            .record_ended(response);
                        exec.session
                            .services
                            .code_mode_service
                            .finish_cell_dispatch(runtime_cell_id);
                        exec.session
                            .services
                            .analytics_events_client
                            .track_code_mode_tool_call(
                                codex_analytics::CodeModeToolCallFact::CellClosed {
                                    thread_id: exec.session.thread_id.to_string(),
                                    turn_id: exec.turn.sub_id.clone(),
                                    cell_id: runtime_cell_id.to_string(),
                                },
                            );
                    }
                }
                if let Some(code_mode_host_duration) = wait_response.code_mode_host_duration() {
                    telemetry.record_code_mode_host_duration(code_mode_host_duration);
                }
                exec.session.services.elicitations.wait_until_clear().await;
                let wall_time = wait_response
                    .code_mode_host_duration()
                    .unwrap_or_else(|| started_at.elapsed());
                let cell_id = codex_code_mode::CellId::new(args.cell_id);
                let max_tokens = args.max_tokens.map(|max_tokens| max_tokens.0);
                handle_runtime_response(&exec, wait_response.into(), max_tokens, wall_time)
                    .await
                    .map_err(FunctionCallError::RespondToModel)
                    .map(|mut output| {
                        exec.session
                            .services
                            .code_mode_service
                            .append_notifications(&cell_id, &mut output, max_tokens);
                        output
                    })
                    .map(boxed_tool_output)
            }
            _ => Err(FunctionCallError::RespondToModel(format!(
                "{WAIT_TOOL_NAME} expects JSON arguments"
            ))),
        };
        telemetry.finish(
            result
                .as_ref()
                .is_ok_and(codex_tools::ToolOutput::success_for_logging),
        );
        result
    }
}

impl CoreToolRuntime for CodeModeWaitHandler {
    fn pre_tool_use_payload(&self, _invocation: &ToolInvocation) -> Option<PreToolUsePayload> {
        // Code-mode `wait` is runtime control for an existing code cell, not a
        // standalone user action. Tool calls made from code mode still flow
        // through normal dispatch, but hooks should not block or rewrite the
        // wait loop itself.
        None
    }

    fn post_tool_use_payload(
        &self,
        _invocation: &ToolInvocation,
        _result: &dyn ToolOutput,
    ) -> Option<PostToolUsePayload> {
        // The wait result feeds code-mode control flow, so do not let
        // PostToolUse replace it with model-facing hook feedback.
        None
    }
}

#[cfg(test)]
#[path = "wait_handler_tests.rs"]
mod tests;
