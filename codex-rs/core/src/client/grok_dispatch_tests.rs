use super::*;
use codex_protocol::model_inference::WireApi;
use std::collections::HashMap;
use std::time::Duration;

fn route() -> ResolvedWireRoute {
    ResolvedWireRoute {
        name: Some("grok".to_string()),
        wire_api: WireApi::Responses,
        dialect: InferenceDialect::Grok,
        base_url: Some("http://127.0.0.1:8080/v1/grok".to_string()),
        request_path: "responses".to_string(),
        query_params: None,
        request_max_retries: 0,
        stream_max_retries: 10,
        stream_idle_timeout: Duration::from_secs(300),
    }
}

#[test]
fn grok_dispatch_rejects_noncanonical_routes() {
    validate_route(&route()).expect("canonical Grok route");

    let mut wrong_path = route();
    wrong_path.request_path = "chat/completions".to_string();
    let mut query = route();
    query.query_params = Some(HashMap::from([("beta".to_string(), "true".to_string())]));
    let mut wrong_dialect = route();
    wrong_dialect.dialect = InferenceDialect::OpenAi;

    for invalid in [wrong_path, query, wrong_dialect] {
        assert!(
            validate_route(&invalid)
                .expect_err("noncanonical Grok route")
                .to_string()
                .contains("must resolve to responses/grok")
        );
    }
}
