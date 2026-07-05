use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use codex_network_proxy::PROXY_ACTIVE_ENV_KEY;
use codex_protocol::models::PermissionProfile;
use codex_sandboxing::SandboxType;
use codex_shell_command::parse_command::shlex_join;
use codex_utils_path_uri::PathUri;
use codex_utils_pty::TerminalSize;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::exec::ExecCapturePolicy;
use crate::exec::ExecExpiration;
use crate::exec_env::create_env;
use crate::sandboxing::ExecRequest;
use crate::session::session::Session;
use crate::session::turn_context::TurnContext;
use crate::shell::Shell;
use crate::shell::ShellType;
use crate::tools::context::ExecCommandToolOutput;
use crate::tools::runtimes::strip_managed_proxy_env;
use crate::unified_exec::NoopSpawnLifecycle;
use crate::unified_exec::ProcessEntry;
use crate::unified_exec::UnifiedExecError;
use crate::unified_exec::UnifiedExecProcessManager;
use crate::unified_exec::WriteStdinRequest;
use crate::unified_exec::process_manager::unregister_network_approval_for_entry;
use codex_utils_output_truncation::TruncationPolicy;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SharedTerminalMetadata {
    pub(crate) label: String,
}

/// Request to open or reattach a user-owned shared terminal for a thread.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SharedTerminalOpenRequest {
    pub label: String,
    pub terminal_size: Option<TerminalSize>,
}

/// Current lifecycle state for a shared terminal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SharedTerminalStatus {
    Running,
    Exited { exit_code: Option<i32> },
}

/// Metadata for a user-owned shared terminal backed by unified exec.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SharedTerminalInfo {
    pub label: String,
    pub item_id: String,
    pub process_id: i32,
    pub command: String,
    pub cwd: PathUri,
    pub tty: bool,
    pub terminal_size: Option<TerminalSize>,
    pub status: SharedTerminalStatus,
}

/// Output collected after writing to a shared terminal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SharedTerminalWriteOutput {
    pub output: Vec<u8>,
    pub process_id: Option<i32>,
    pub exit_code: Option<i32>,
}

impl UnifiedExecProcessManager {
    pub(crate) async fn open_shared_terminal(
        &self,
        session: Arc<Session>,
        turn: Arc<TurnContext>,
        request: SharedTerminalOpenRequest,
    ) -> Result<SharedTerminalInfo, UnifiedExecError> {
        let label = validate_shared_terminal_label(&request.label)?;

        if let Some(entry) = self
            .remove_exited_or_return_live_shared_terminal(&label)
            .await?
        {
            return Ok(entry);
        }

        let Some((turn_environment, environment_shell)) = turn
            .environments
            .local()
            .and_then(|environment| environment.shell.as_ref().map(|shell| (environment, shell)))
        else {
            return Err(UnifiedExecError::create_process(
                "shell is unavailable in this session".to_string(),
            ));
        };

        let native_cwd =
            turn_environment
                .cwd()
                .to_abs_path()
                .map_err(|_| UnifiedExecError::ForeignPath {
                    path: turn_environment.cwd().clone(),
                })?;
        let cwd = turn_environment.cwd().clone();
        let command = interactive_shell_command(environment_shell);
        let hook_command = shlex_join(&command);
        let terminal_size = request.terminal_size.unwrap_or_default();

        let mut env = create_env(
            &turn.config.permissions.shell_environment_policy,
            Some(session.thread_id),
        );
        if env.contains_key(PROXY_ACTIVE_ENV_KEY) {
            strip_managed_proxy_env(&mut env);
        }

        let permission_profile = PermissionProfile::Disabled;
        let exec_request = ExecRequest::new(
            command.clone(),
            native_cwd,
            env,
            None,
            None,
            ExecExpiration::Cancellation(CancellationToken::new()),
            ExecCapturePolicy::ShellTool,
            SandboxType::None,
            turn.config.effective_workspace_roots(),
            turn.windows_sandbox_level,
            turn.config.permissions.windows_sandbox_private_desktop,
            permission_profile,
            None,
        );

        let process_id = self.allocate_process_id().await;
        let process = match self
            .open_session_with_prepared_exec_env(
                process_id,
                &exec_request,
                /*tty*/ true,
                Box::new(NoopSpawnLifecycle),
                turn_environment.environment.as_ref(),
                terminal_size,
            )
            .await
        {
            Ok(process) => Arc::new(process),
            Err(err) => {
                self.release_process_id(process_id).await;
                return Err(err);
            }
        };

        if process.has_exited() {
            self.release_process_id(process_id).await;
            return Err(UnifiedExecError::process_failed(format!(
                "shared terminal `{label}` exited immediately"
            )));
        }

        let started_at = Instant::now();
        let call_id = format!("shared-terminal:{label}:{process_id}");
        let info = SharedTerminalInfo {
            label: label.clone(),
            item_id: call_id.clone(),
            process_id,
            command: hook_command.clone(),
            cwd: cwd.clone(),
            tty: true,
            terminal_size: Some(terminal_size),
            status: SharedTerminalStatus::Running,
        };
        let entry = ProcessEntry {
            process: Arc::clone(&process),
            call_id,
            process_id,
            cwd,
            initial_exec_command_active: Arc::new(AtomicBool::new(false)),
            hook_command,
            tty: true,
            network_approval: None,
            session: Arc::downgrade(&session),
            last_used: started_at,
            shared_terminal: Some(SharedTerminalMetadata { label }),
            terminal_size: Some(terminal_size),
        };

        let (existing_info, pruned_entry) = {
            let mut store = self.process_store.lock().await;
            if let Some(existing_info) = live_shared_terminal_info(&store, &entry) {
                (Some(existing_info), None)
            } else {
                let pruned_entry = Self::prune_processes_if_needed(&mut store);
                store.processes.insert(process_id, entry);
                (None, pruned_entry)
            }
        };

        if let Some(existing_info) = existing_info {
            self.release_process_id(process_id).await;
            process.terminate();
            return Ok(existing_info);
        }

        if let Some(pruned_entry) = pruned_entry {
            unregister_network_approval_for_entry(&pruned_entry).await;
            pruned_entry.process.terminate();
        }

        Ok(info)
    }

    pub(crate) async fn resize_shared_terminal(
        &self,
        process_id: i32,
        terminal_size: TerminalSize,
    ) -> Result<SharedTerminalInfo, UnifiedExecError> {
        let process = {
            let store = self.process_store.lock().await;
            let Some(entry) = store.processes.get(&process_id) else {
                return Err(UnifiedExecError::UnknownProcessId { process_id });
            };
            if entry.shared_terminal.is_none() || entry.process.has_exited() {
                return Err(UnifiedExecError::UnknownProcessId { process_id });
            }
            Arc::clone(&entry.process)
        };

        process.resize(terminal_size).await?;

        let mut store = self.process_store.lock().await;
        let Some(entry) = store.processes.get_mut(&process_id) else {
            return Err(UnifiedExecError::UnknownProcessId { process_id });
        };
        if !Arc::ptr_eq(&entry.process, &process) || entry.shared_terminal.is_none() {
            return Err(UnifiedExecError::UnknownProcessId { process_id });
        }
        entry.terminal_size = Some(terminal_size);
        shared_terminal_info_for_entry(entry)
            .ok_or(UnifiedExecError::UnknownProcessId { process_id })
    }

    pub(crate) async fn write_shared_terminal(
        &self,
        process_id: i32,
        input: &str,
        yield_time_ms: u64,
    ) -> Result<SharedTerminalWriteOutput, UnifiedExecError> {
        {
            let store = self.process_store.lock().await;
            let Some(entry) = store.processes.get(&process_id) else {
                return Err(UnifiedExecError::UnknownProcessId { process_id });
            };
            if entry.shared_terminal.is_none() {
                return Err(UnifiedExecError::UnknownProcessId { process_id });
            }
        }

        let ExecCommandToolOutput {
            raw_output,
            process_id,
            exit_code,
            ..
        } = self
            .write_stdin(WriteStdinRequest {
                process_id,
                input,
                yield_time_ms,
                max_output_tokens: None,
                truncation_policy: TruncationPolicy::Tokens(10_000),
            })
            .await?;

        Ok(SharedTerminalWriteOutput {
            output: raw_output,
            process_id,
            exit_code,
        })
    }

    async fn remove_exited_or_return_live_shared_terminal(
        &self,
        label: &str,
    ) -> Result<Option<SharedTerminalInfo>, UnifiedExecError> {
        let exited_entry = {
            let mut store = self.process_store.lock().await;
            let process_id = store.processes.iter().find_map(|(process_id, entry)| {
                entry
                    .shared_terminal
                    .as_ref()
                    .filter(|metadata| metadata.label == label)
                    .map(|_| *process_id)
            });

            let Some(process_id) = process_id else {
                return Ok(None);
            };
            let Some(entry) = store.processes.get(&process_id) else {
                return Ok(None);
            };
            if !entry.process.has_exited() {
                return Ok(shared_terminal_info_for_entry(entry));
            }
            store.remove(process_id)
        };

        if let Some(exited_entry) = exited_entry {
            unregister_network_approval_for_entry(&exited_entry).await;
            exited_entry.process.terminate();
        }

        Ok(None)
    }
}

fn validate_shared_terminal_label(label: &str) -> Result<String, UnifiedExecError> {
    let label = label.trim();
    let valid = !label.is_empty()
        && label.len() <= 64
        && label
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'));

    if valid {
        Ok(label.to_string())
    } else {
        Err(UnifiedExecError::InvalidSharedTerminalLabel {
            label: label.to_string(),
        })
    }
}

fn interactive_shell_command(shell: &Shell) -> Vec<String> {
    let shell_path = shell.shell_path.to_string_lossy().to_string();
    match shell.shell_type {
        ShellType::Bash | ShellType::Zsh => vec![shell_path, "-l".to_string()],
        ShellType::PowerShell => vec![shell_path, "-NoExit".to_string()],
        ShellType::Sh | ShellType::Cmd => vec![shell_path],
    }
}

fn live_shared_terminal_info(
    store: &crate::unified_exec::ProcessStore,
    candidate: &ProcessEntry,
) -> Option<SharedTerminalInfo> {
    let label = &candidate.shared_terminal.as_ref()?.label;
    store.processes.values().find_map(|entry| {
        entry
            .shared_terminal
            .as_ref()
            .filter(|metadata| metadata.label == label.as_str())
            .and_then(|_| {
                if entry.process.has_exited() {
                    None
                } else {
                    shared_terminal_info_for_entry(entry)
                }
            })
    })
}

fn shared_terminal_info_for_entry(entry: &ProcessEntry) -> Option<SharedTerminalInfo> {
    let metadata = entry.shared_terminal.as_ref()?;
    let status = if entry.process.has_exited() {
        SharedTerminalStatus::Exited {
            exit_code: entry.process.exit_code(),
        }
    } else {
        SharedTerminalStatus::Running
    };
    Some(SharedTerminalInfo {
        label: metadata.label.clone(),
        item_id: entry.call_id.clone(),
        process_id: entry.process_id,
        command: entry.hook_command.clone(),
        cwd: entry.cwd.clone(),
        tty: entry.tty,
        terminal_size: entry.terminal_size,
        status,
    })
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::validate_shared_terminal_label;

    #[test]
    fn shared_terminal_labels_trim_and_accept_safe_chars() {
        assert_eq!(
            validate_shared_terminal_label("  default.shell-1  ").expect("label should validate"),
            "default.shell-1".to_string()
        );
    }

    #[test]
    fn shared_terminal_labels_reject_empty_or_unsafe_chars() {
        assert!(validate_shared_terminal_label("   ").is_err());
        assert!(validate_shared_terminal_label("bad label").is_err());
        assert!(validate_shared_terminal_label("bad/label").is_err());
        assert!(validate_shared_terminal_label(&"a".repeat(65)).is_err());
    }
}
