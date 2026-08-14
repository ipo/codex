//! Implements the MultiAgentV2 collaboration tool surface.

use crate::agent::AgentStatus;
use crate::agent::agent_resolver::resolve_agent_target;
use crate::function_tool::FunctionCallError;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolOutput;
use crate::tools::context::ToolPayload;
use crate::tools::context::boxed_tool_output;
use crate::tools::handlers::multi_agents_common::*;
use crate::tools::handlers::parse_arguments;
use crate::tools::registry::CoreToolRuntime;
use crate::tools::registry::ToolExecutor;
use codex_protocol::AgentPath;
use codex_protocol::items::CollabAgentTool;
use codex_protocol::items::CollabAgentToolCallItem;
use codex_protocol::items::CollabAgentToolCallStatus;
use codex_protocol::items::SubAgentActivityItem;
use codex_protocol::items::TurnItem;
use codex_protocol::models::ResponseInputItem;
use codex_protocol::openai_models::ReasoningEffort;
use codex_protocol::protocol::InterAgentCommunication;
use codex_protocol::protocol::SubAgentActivityKind;
use codex_tools::ToolName;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value as JsonValue;

pub(crate) use followup_task::Handler as FollowupTaskHandler;
pub(crate) use interrupt_agent::Handler as InterruptAgentHandler;
pub(crate) use list_agents::Handler as ListAgentsHandler;
pub(crate) use send_message::Handler as SendMessageHandler;
pub(crate) use spawn::Handler as SpawnAgentHandler;
pub(crate) use wait::Handler as WaitAgentHandler;

mod followup_task;
mod interrupt_agent;
mod list_agents;
mod message_tool;
mod send_message;
mod spawn;
pub(crate) mod wait;

pub(crate) async fn emit_sub_agent_activity(
    session: &crate::session::session::Session,
    turn: &crate::session::turn_context::TurnContext,
    item: SubAgentActivityItem,
) {
    let item = TurnItem::SubAgentActivity(item);
    session.emit_turn_item_started(turn, &item).await;
    session.emit_turn_item_completed(turn, item).await;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ToolMessage {
    Encrypted(String),
    Plaintext(String),
}

pub(super) fn communication_from_tool_message(
    author: AgentPath,
    recipient: AgentPath,
    message: ToolMessage,
) -> InterAgentCommunication {
    match message {
        ToolMessage::Encrypted(message) => InterAgentCommunication::new_encrypted(
            author,
            recipient,
            Vec::new(),
            message,
            /*trigger_turn*/ true,
        ),
        ToolMessage::Plaintext(message) => InterAgentCommunication::new(
            author,
            recipient,
            Vec::new(),
            message,
            /*trigger_turn*/ true,
        ),
    }
}

fn validate_tool_message_family(
    sender: &codex_protocol::openai_models::ModelInfo,
    recipient: &codex_protocol::openai_models::ModelInfo,
    message: &ToolMessage,
) -> Result<(), FunctionCallError> {
    if matches!(message, ToolMessage::Encrypted(_)) && !sender.is_history_compatible_with(recipient)
    {
        return Err(FunctionCallError::RespondToModel(format!(
            "Encrypted collaboration messages cannot cross model families (`{}` to `{}`); retry with plaintext_message.",
            sender.slug, recipient.slug
        )));
    }
    Ok(())
}

fn reject_unsupported_encrypted_message(
    turn: &crate::session::turn_context::TurnContext,
    message: Option<&str>,
) -> Result<(), FunctionCallError> {
    if message.is_some()
        && !crate::tools::wire_adaptation::wire_supports_encrypted_tool_content(turn)
    {
        return Err(FunctionCallError::RespondToModel(
            "Encrypted collaboration arguments are unavailable on this inference wire; retry with plaintext_message."
                .to_string(),
        ));
    }
    Ok(())
}
