//! Adapts MultiAgentV2 collaboration tools between encrypted Responses calls and native wires.

use std::sync::Arc;

use codex_tools::ToolExecutor;
use codex_tools::ToolExposure;
use codex_tools::ToolName;
use codex_tools::ToolSearchInfo;
use codex_tools::ToolSpec;
use futures::future::BoxFuture;

use crate::function_tool::FunctionCallError;
use crate::session::session::Session;
use crate::tools::context::ToolCallSource;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::registry::CoreToolRuntime;

pub(crate) fn adapt_multi_agent_v2_handler(
    handler: Arc<dyn CoreToolRuntime>,
) -> Arc<dyn CoreToolRuntime> {
    if is_multi_agent_v2_message_tool(&handler.tool_name().name) {
        Arc::new(NativePlaintextMessageAdapter { handler })
    } else {
        handler
    }
}

struct NativePlaintextMessageAdapter {
    handler: Arc<dyn CoreToolRuntime>,
}

impl ToolExecutor<ToolInvocation> for NativePlaintextMessageAdapter {
    fn tool_name(&self) -> ToolName {
        self.handler.tool_name()
    }

    fn spec(&self) -> ToolSpec {
        self.handler.spec()
    }

    fn exposure(&self) -> ToolExposure {
        self.handler.exposure()
    }

    fn supports_parallel_tool_calls(&self) -> bool {
        self.handler.supports_parallel_tool_calls()
    }

    fn search_info(&self) -> Option<ToolSearchInfo> {
        self.handler.search_info()
    }

    fn handle<'a>(&'a self, invocation: ToolInvocation) -> codex_tools::ToolExecutorFuture<'a>
    where
        ToolInvocation: 'a,
    {
        if !crate::tools::wire_adaptation::wire_supports_encrypted_tool_content(
            &invocation.turn,
            &invocation.step_context.settings.model_info,
        ) {
            return match adapt_native_plaintext_collaboration_invocation(invocation) {
                Ok(invocation) => self.handler.handle(invocation),
                Err(err) => Box::pin(async move { Err(err) }),
            };
        }
        self.handler.handle(invocation)
    }
}

impl CoreToolRuntime for NativePlaintextMessageAdapter {
    fn wait_until_ready<'a>(&'a self, session: &'a Arc<Session>) -> Option<BoxFuture<'a, ()>> {
        self.handler.wait_until_ready(session)
    }

    fn matches_kind(&self, payload: &ToolPayload) -> bool {
        self.handler.matches_kind(payload)
    }

    fn create_diff_consumer(
        &self,
    ) -> Option<Box<dyn crate::tools::registry::ToolArgumentDiffConsumer>> {
        self.handler.create_diff_consumer()
    }
}

pub(crate) fn is_multi_agent_v2_message_tool(tool_name: &str) -> bool {
    matches!(tool_name, "spawn_agent" | "send_message" | "followup_task")
}

fn adapt_native_plaintext_collaboration_invocation(
    invocation: ToolInvocation,
) -> Result<ToolInvocation, FunctionCallError> {
    let arguments = match &invocation.payload {
        ToolPayload::Function { arguments } => arguments,
        _ => {
            return Err(FunctionCallError::RespondToModel(
                "native collaboration tool invoked with incompatible payload".to_string(),
            ));
        }
    };
    let arguments = adapt_native_plaintext_collaboration_arguments(arguments)?;

    Ok(ToolInvocation {
        source: ToolCallSource::DirectPlaintextMessage,
        payload: ToolPayload::Function { arguments },
        ..invocation
    })
}

pub(crate) fn adapt_native_plaintext_collaboration_arguments(
    arguments: &str,
) -> Result<String, FunctionCallError> {
    let mut arguments = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(
        arguments,
    )
    .map_err(|err| {
        FunctionCallError::RespondToModel(format!(
            "failed to parse native collaboration arguments: {err}"
        ))
    })?;
    if arguments.contains_key("message") {
        return Err(FunctionCallError::RespondToModel(
            "Encrypted collaboration arguments are unavailable on this inference wire; retry with plaintext_message."
                .to_string(),
        ));
    }
    let plaintext_message = arguments.remove("plaintext_message").ok_or_else(|| {
        FunctionCallError::RespondToModel(
            "Native collaboration calls require plaintext_message.".to_string(),
        )
    })?;
    if !plaintext_message.is_string() {
        return Err(FunctionCallError::RespondToModel(
            "plaintext_message must be a string.".to_string(),
        ));
    }
    arguments.insert("message".to_string(), plaintext_message);
    serde_json::to_string(&arguments).map_err(|err| {
        FunctionCallError::RespondToModel(format!(
            "failed to serialize native collaboration arguments: {err}"
        ))
    })
}
