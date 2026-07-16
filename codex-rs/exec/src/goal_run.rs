use std::time::Duration;

use codex_app_server_client::InProcessAppServerClient;
use codex_app_server_client::InProcessServerEvent;
use codex_app_server_protocol::ClientRequest;
use codex_app_server_protocol::ServerNotification;
use codex_app_server_protocol::ThreadGoal;
use codex_app_server_protocol::ThreadGoalGetParams;
use codex_app_server_protocol::ThreadGoalGetResponse;
use codex_app_server_protocol::ThreadGoalSetParams;
use codex_app_server_protocol::ThreadGoalSetResponse;
use codex_app_server_protocol::ThreadGoalStatus;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tracing::Span;
use tracing::warn;

use crate::RequestIdSequencer;
use crate::event_processor::EventProcessor;
use crate::handle_server_request;
use crate::interrupt_turn;
use crate::lagged_event_warning_message;
use crate::maybe_backfill_turn_completed_items;
use crate::request_shutdown;
use crate::send_request_with_response;
use crate::should_process_notification;

#[cfg(test)]
use crate::goal_prompt::GoalInvocation;
#[cfg(test)]
use crate::goal_prompt::GoalPreflight;
#[cfg(test)]
use crate::goal_prompt::GoalPreflightError;
#[cfg(test)]
use crate::goal_prompt::classify_goal_prompt;

pub(crate) const NEXT_GOAL_TURN_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GoalDisposition {
    Continue,
    Complete,
    Incomplete,
}

pub(crate) fn disposition_for_status(status: ThreadGoalStatus) -> GoalDisposition {
    match status {
        ThreadGoalStatus::Active => GoalDisposition::Continue,
        ThreadGoalStatus::Complete => GoalDisposition::Complete,
        ThreadGoalStatus::Paused
        | ThreadGoalStatus::Blocked
        | ThreadGoalStatus::UsageLimited
        | ThreadGoalStatus::BudgetLimited => GoalDisposition::Incomplete,
    }
}

#[derive(Debug)]
pub(crate) struct GoalRun {
    thread_id: String,
    active_turn_id: Option<String>,
    next_turn_deadline: Option<Instant>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum GoalCancellationAction {
    PauseGoal,
    InterruptTurn { turn_id: String },
}

pub(crate) fn cancellation_actions(run: &GoalRun) -> Vec<GoalCancellationAction> {
    let mut actions = vec![GoalCancellationAction::PauseGoal];
    if let Some(turn_id) = run.active_turn_id() {
        actions.push(GoalCancellationAction::InterruptTurn {
            turn_id: turn_id.to_string(),
        });
    }
    actions
}

impl GoalRun {
    pub(crate) fn new(thread_id: String) -> Self {
        Self {
            thread_id,
            active_turn_id: None,
            next_turn_deadline: Some(Instant::now() + NEXT_GOAL_TURN_TIMEOUT),
        }
    }

    pub(crate) fn thread_id(&self) -> &str {
        &self.thread_id
    }

    pub(crate) fn active_turn_id(&self) -> Option<&str> {
        self.active_turn_id.as_deref()
    }

    pub(crate) fn next_turn_deadline(&self) -> Option<Instant> {
        self.next_turn_deadline
    }

    pub(crate) fn observe_turn_started(&mut self, thread_id: &str, turn_id: &str) -> bool {
        if thread_id != self.thread_id {
            return false;
        }
        if let Some(active_turn_id) = self.active_turn_id.as_deref() {
            return active_turn_id == turn_id;
        }
        self.active_turn_id = Some(turn_id.to_string());
        self.next_turn_deadline = None;
        true
    }

    pub(crate) fn observe_turn_completed(
        &mut self,
        thread_id: &str,
        turn_id: &str,
        status: ThreadGoalStatus,
    ) -> Option<GoalDisposition> {
        if thread_id != self.thread_id || self.active_turn_id.as_deref() != Some(turn_id) {
            return None;
        }
        self.active_turn_id = None;
        let disposition = disposition_for_status(status);
        self.next_turn_deadline = (disposition == GoalDisposition::Continue)
            .then(|| Instant::now() + NEXT_GOAL_TURN_TIMEOUT);
        Some(disposition)
    }
}

enum GoalLoopEvent {
    Interrupt(Option<()>),
    TurnTimeout,
    Server(Option<Box<InProcessServerEvent>>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GoalRunOutcome {
    Complete,
    Failed,
}

pub(crate) async fn follow_goal(
    client: &mut InProcessAppServerClient,
    request_ids: &mut RequestIdSequencer,
    event_processor: &mut dyn EventProcessor,
    interrupt_rx: &mut mpsc::UnboundedReceiver<()>,
    mut run: GoalRun,
    exec_span: &Span,
) -> GoalRunOutcome {
    let mut error_seen = false;
    let mut goal_completed = false;
    let mut interrupt_channel_open = true;
    loop {
        let goal_deadline = run.next_turn_deadline();
        let loop_event = tokio::select! {
            maybe_interrupt = interrupt_rx.recv(), if interrupt_channel_open => {
                GoalLoopEvent::Interrupt(maybe_interrupt)
            }
            _ = wait_until_goal_deadline(goal_deadline), if goal_deadline.is_some() => {
                GoalLoopEvent::TurnTimeout
            }
            maybe_event = client.next_event() => {
                GoalLoopEvent::Server(maybe_event.map(Box::new))
            },
        };

        let server_event = match loop_event {
            GoalLoopEvent::Interrupt(maybe_interrupt) => {
                if maybe_interrupt.is_none() {
                    interrupt_channel_open = false;
                    continue;
                }
                error_seen = true;
                let thread_id = run.thread_id().to_string();
                for action in cancellation_actions(&run) {
                    match action {
                        GoalCancellationAction::PauseGoal => {
                            match set_goal_status(
                                client,
                                request_ids,
                                &thread_id,
                                ThreadGoalStatus::Paused,
                            )
                            .await
                            {
                                Ok(goal) => {
                                    let _ = event_processor.process_goal_update(&goal);
                                }
                                Err(err) => {
                                    let message =
                                        format!("Failed to pause goal after Ctrl-C: {err}");
                                    let _ = event_processor.process_error(message);
                                }
                            }
                        }
                        GoalCancellationAction::InterruptTurn { turn_id } => {
                            if let Err(err) =
                                interrupt_turn(client, request_ids, &thread_id, &turn_id).await
                            {
                                warn!("turn/interrupt failed: {err}");
                            }
                        }
                    }
                }
                break;
            }
            GoalLoopEvent::TurnTimeout => {
                match get_thread_goal(client, request_ids, run.thread_id()).await {
                    Ok(Some(goal)) => {
                        let _ = event_processor.process_goal_update(&goal);
                        match disposition_for_status(goal.status) {
                            GoalDisposition::Complete => {
                                goal_completed = true;
                            }
                            GoalDisposition::Incomplete => {
                                error_seen = true;
                            }
                            GoalDisposition::Continue => {
                                error_seen = true;
                                match set_goal_status(
                                    client,
                                    request_ids,
                                    run.thread_id(),
                                    ThreadGoalStatus::Paused,
                                )
                                .await
                                {
                                    Ok(paused_goal) => {
                                        let _ = event_processor.process_goal_update(&paused_goal);
                                    }
                                    Err(err) => {
                                        let _ = event_processor.process_error(format!(
                                            "Failed to pause goal after continuation timeout: {err}"
                                        ));
                                    }
                                }
                                let _ = event_processor.process_error(
                                    "Goal continuation did not start within 30 seconds; the goal was paused."
                                        .to_string(),
                                );
                            }
                        }
                    }
                    Ok(None) => {
                        error_seen = true;
                        let _ = event_processor.process_error(
                            "Goal disappeared while exec was waiting for its next turn."
                                .to_string(),
                        );
                    }
                    Err(err) => {
                        error_seen = true;
                        let _ = event_processor.process_error(format!(
                            "Failed to read goal while waiting for its next turn: {err}"
                        ));
                    }
                }
                break;
            }
            GoalLoopEvent::Server(maybe_event) => maybe_event.map(|event| *event),
        };

        let Some(server_event) = server_event else {
            error_seen = true;
            let _ = event_processor.process_error(
                "App-server event stream ended before the goal reached a terminal status."
                    .to_string(),
            );
            break;
        };

        match server_event {
            InProcessServerEvent::ServerRequest(request) => {
                handle_server_request(client, request, &mut error_seen).await;
            }
            InProcessServerEvent::ServerNotification(mut notification) => {
                if let ServerNotification::TurnStarted(payload) = &notification
                    && run.observe_turn_started(&payload.thread_id, &payload.turn.id)
                {
                    exec_span.record("turn.id", payload.turn.id.as_str());
                    let _ = event_processor.process_server_notification(notification);
                    continue;
                }

                let completed_turn = match &notification {
                    ServerNotification::TurnCompleted(payload)
                        if payload.thread_id == run.thread_id()
                            && run.active_turn_id() == Some(payload.turn.id.as_str()) =>
                    {
                        Some((
                            payload.thread_id.clone(),
                            payload.turn.id.clone(),
                            payload.turn.status.clone(),
                        ))
                    }
                    _ => None,
                };
                if let ServerNotification::Error(payload) = &notification
                    && payload.thread_id == run.thread_id()
                    && run.active_turn_id() == Some(payload.turn_id.as_str())
                    && !payload.will_retry
                {
                    error_seen = true;
                }

                if let Some((thread_id, turn_id, turn_status)) = completed_turn {
                    if matches!(
                        turn_status,
                        codex_app_server_protocol::TurnStatus::Failed
                            | codex_app_server_protocol::TurnStatus::Interrupted
                    ) {
                        error_seen = true;
                    }
                    maybe_backfill_turn_completed_items(
                        /*thread_ephemeral*/ false,
                        client,
                        request_ids,
                        &mut notification,
                    )
                    .await;
                    let _ = event_processor.process_server_notification(notification);

                    let goal = match get_thread_goal(client, request_ids, run.thread_id()).await {
                        Ok(Some(goal)) => goal,
                        Ok(None) => {
                            error_seen = true;
                            let _ = event_processor.process_error(format!(
                                "Goal disappeared after turn {turn_id} completed."
                            ));
                            break;
                        }
                        Err(err) => {
                            error_seen = true;
                            let _ = event_processor.process_error(format!(
                                "Failed to read goal after turn {turn_id} completed: {err}"
                            ));
                            break;
                        }
                    };
                    let _ = event_processor.process_goal_update(&goal);
                    let Some(disposition) =
                        run.observe_turn_completed(&thread_id, &turn_id, goal.status)
                    else {
                        error_seen = true;
                        let _ = event_processor.process_error(format!(
                            "Goal controller lost track of completed turn {turn_id}."
                        ));
                        break;
                    };
                    match disposition {
                        GoalDisposition::Continue => continue,
                        GoalDisposition::Complete => {
                            goal_completed = true;
                            break;
                        }
                        GoalDisposition::Incomplete => {
                            error_seen = true;
                            break;
                        }
                    }
                }

                if should_process_goal_notification(&notification, &run) {
                    let _ = event_processor.process_server_notification(notification);
                }
            }
            InProcessServerEvent::Lagged { skipped } => {
                let message = lagged_event_warning_message(skipped);
                warn!("{message}");
                event_processor.process_warning(message);
            }
        }
    }

    if let Err(err) = request_shutdown(client, request_ids, run.thread_id()).await {
        warn!("thread/unsubscribe failed during goal shutdown: {err}");
    }
    if error_seen || !goal_completed {
        GoalRunOutcome::Failed
    } else {
        GoalRunOutcome::Complete
    }
}

async fn get_thread_goal(
    client: &InProcessAppServerClient,
    request_ids: &mut RequestIdSequencer,
    thread_id: &str,
) -> Result<Option<ThreadGoal>, String> {
    send_request_with_response::<ThreadGoalGetResponse>(
        client,
        ClientRequest::ThreadGoalGet {
            request_id: request_ids.next(),
            params: ThreadGoalGetParams {
                thread_id: thread_id.to_string(),
            },
        },
        "thread/goal/get",
    )
    .await
    .map(|response| response.goal)
}

async fn set_goal_status(
    client: &InProcessAppServerClient,
    request_ids: &mut RequestIdSequencer,
    thread_id: &str,
    status: ThreadGoalStatus,
) -> Result<ThreadGoal, String> {
    send_request_with_response::<ThreadGoalSetResponse>(
        client,
        ClientRequest::ThreadGoalSet {
            request_id: request_ids.next(),
            params: ThreadGoalSetParams {
                thread_id: thread_id.to_string(),
                objective: None,
                status: Some(status),
                token_budget: None,
            },
        },
        "thread/goal/set",
    )
    .await
    .map(|response| response.goal)
}

async fn wait_until_goal_deadline(deadline: Option<Instant>) {
    if let Some(deadline) = deadline {
        tokio::time::sleep_until(deadline).await;
    } else {
        std::future::pending::<()>().await;
    }
}

fn should_process_goal_notification(notification: &ServerNotification, run: &GoalRun) -> bool {
    if let Some(turn_id) = run.active_turn_id() {
        return should_process_notification(notification, run.thread_id(), turn_id);
    }
    match notification {
        ServerNotification::ConfigWarning(_) | ServerNotification::DeprecationNotice(_) => true,
        ServerNotification::Warning(notification) => notification
            .thread_id
            .as_deref()
            .is_none_or(|candidate| candidate == run.thread_id()),
        ServerNotification::HookCompleted(notification) => {
            notification.thread_id == run.thread_id() && notification.turn_id.is_none()
        }
        ServerNotification::HookStarted(notification) => {
            notification.thread_id == run.thread_id() && notification.turn_id.is_none()
        }
        _ => false,
    }
}

#[cfg(test)]
#[path = "goal_run_tests.rs"]
mod tests;
