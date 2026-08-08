use std::sync::Arc;

use codex_api::AuthProvider;
use codex_http_client::Request;
use http::HeaderMap;
use http::HeaderValue;
use http::Method;

use super::claude_code_environment;
use super::compatibility_auth;
use crate::responses_metadata::CodexResponsesMetadata;

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
    let plain: Arc<dyn AuthProvider> = Arc::new(AccountAuth);
    let plain_headers = plain.to_auth_headers();
    assert_eq!(plain_headers["chatgpt-account-id"], "account_id");

    let filtered = compatibility_auth(plain);
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
fn claude_code_environment_fails_closed_without_authoritative_target_facts() {
    let metadata = CodexResponsesMetadata::new(
        "installation".to_string(),
        "session".to_string(),
        "thread".to_string(),
        "window".to_string(),
    );

    let error =
        claude_code_environment(&metadata).expect_err("legacy target facts must be rejected");

    assert_eq!(
        error.to_string(),
        "Claude Code compatibility requires authoritative execution environment facts"
    );
}
