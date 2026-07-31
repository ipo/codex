use super::session::SessionConfiguration;
use codex_file_system::FileSystemSandboxContext;
use codex_utils_path_uri::PathUri;

impl SessionConfiguration {
    pub(super) fn file_system_sandbox_context(&self, cwd: &PathUri) -> FileSystemSandboxContext {
        let mut workspace_roots: Vec<PathUri> = self
            .primary_workspace_roots()
            .iter()
            .map(PathUri::from_abs_path)
            .collect();
        for root in self.profile_workspace_roots() {
            let root = PathUri::from_abs_path(root);
            if !workspace_roots.contains(&root) {
                workspace_roots.push(root);
            }
        }
        FileSystemSandboxContext {
            permissions: self.permission_profile().into(),
            cwd: Some(cwd.clone()),
            workspace_roots,
            windows_sandbox_level: self.windows_sandbox_level,
            windows_sandbox_private_desktop: self
                .original_config_do_not_use
                .permissions
                .windows_sandbox_private_desktop,
            windows_sandbox_proxy_settings_mode: None,
            use_legacy_landlock: self
                .original_config_do_not_use
                .features
                .use_legacy_landlock(),
        }
    }
}
