use std::sync::Arc;

use codex_api::AuthProvider;
use codex_exec_server::EnvironmentOperatingSystem;
use codex_exec_server::EnvironmentSystemInfo;
use codex_http_client::Request;
use codex_utils_path_uri::PathUri;
use http::HeaderMap;
use http::HeaderValue;
use http::Method;

use super::claude_code_environment;
use super::compatibility_auth;
use crate::responses_metadata::CodexResponsesMetadata;
use crate::responses_metadata::TurnExecutionEnvironment;

struct AccountAuth;

impl AuthProvider for AccountAuth {
    fn add_auth_headers(&self, headers: &mut HeaderMap) {
        headers.insert(
            http::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer external"),
        );
        headers.insert("chatgpt-account-id", HeaderValue::from_static("account_id"));
    }
}

#[tokio::test]
async fn compatibility_auth_preserves_bearer_and_removes_only_account_id() {
    let filtered = compatibility_auth(Arc::new(AccountAuth));
    let headers = filtered.to_auth_headers();
    assert_eq!(headers[http::header::AUTHORIZATION], "Bearer external");
    assert!(!headers.contains_key("chatgpt-account-id"));

    let request = filtered
        .apply_auth(Request::new(
            Method::POST,
            "https://example.test".to_string(),
        ))
        .await
        .expect("delegated auth succeeds");
    assert_eq!(
        request.headers[http::header::AUTHORIZATION],
        "Bearer external"
    );
    assert!(!request.headers.contains_key("chatgpt-account-id"));
}

#[test]
fn compatibility_environment_uses_authoritative_target_facts() {
    let mut metadata = CodexResponsesMetadata::new(
        "installation".to_string(),
        "session".to_string(),
        "thread".to_string(),
        "window".to_string(),
    );
    metadata.turn_environment = Some(TurnExecutionEnvironment {
        cwd: PathUri::parse("file:///workspace/project").expect("path URI"),
        is_git_repository: true,
        shell: "/bin/bash".to_string(),
        system: EnvironmentSystemInfo {
            operating_system: EnvironmentOperatingSystem::Linux,
            architecture: "x86_64".to_string(),
            os_version: "Linux test".to_string(),
        },
    });

    let environment = claude_code_environment(&metadata).expect("environment");
    assert_eq!(
        environment,
        codex_claude_code::ClaudeCodeEnvironment {
            cwd: "/workspace/project".to_string(),
            is_git_repository: true,
            platform: "linux".to_string(),
            architecture: "x86_64".to_string(),
            shell: "/bin/bash".to_string(),
            os_version: "Linux test".to_string(),
        }
    );
}
