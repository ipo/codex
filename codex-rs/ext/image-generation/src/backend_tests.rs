use codex_api::ImageBackground;
use codex_api::ImageEditRequest;
use codex_api::ImageGenerationRequest;
use codex_api::ImageQuality;
use codex_api::ImageUrl;
use codex_login::AuthManager;
use codex_login::CodexAuth;
use codex_model_provider::create_model_provider;
use codex_model_provider_info::ModelProviderInfo;
use pretty_assertions::assert_eq;
use serde_json::json;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::ResponseTemplate;
use wiremock::matchers::header;
use wiremock::matchers::method;
use wiremock::matchers::path;

use super::CodexImagesBackend;

#[tokio::test]
async fn image_generation_and_edit_use_distinct_authenticated_openai_endpoint() {
    let inference = MockServer::start().await;
    let auxiliary = MockServer::start().await;
    assert_ne!(inference.uri(), auxiliary.uri());
    let response = ResponseTemplate::new(200).set_body_json(json!({
        "created": 1,
        "data": [{"b64_json": "cG5n"}],
    }));
    for endpoint in ["/images/generations", "/images/edits"] {
        Mock::given(method("POST"))
            .and(path(endpoint))
            .and(header("authorization", "Bearer sk-auxiliary"))
            .respond_with(response.clone())
            .expect(1)
            .mount(&auxiliary)
            .await;
    }
    let auth_manager = AuthManager::from_auth_for_testing(CodexAuth::from_api_key("sk-auxiliary"));
    let backend = CodexImagesBackend::new(
        create_model_provider(
            ModelProviderInfo::create_openai_provider(Some(auxiliary.uri())),
            Some(auth_manager),
        ),
        None,
    );
    let generation = backend
        .generate(ImageGenerationRequest {
            prompt: "a moonlit lake".to_string(),
            background: Some(ImageBackground::Auto),
            model: "gpt-image-2".to_string(),
            n: None,
            quality: Some(ImageQuality::Auto),
            size: Some("auto".to_string()),
        })
        .await
        .expect("image generation should succeed");
    let edit = backend
        .edit(ImageEditRequest {
            images: vec![ImageUrl {
                image_url: "data:image/png;base64,cG5n".to_string(),
            }],
            prompt: "add stars".to_string(),
            background: Some(ImageBackground::Auto),
            model: "gpt-image-2".to_string(),
            n: None,
            quality: Some(ImageQuality::Auto),
            size: Some("auto".to_string()),
        })
        .await
        .expect("image edit should succeed");

    assert_eq!(generation, edit);
    assert!(
        inference
            .received_requests()
            .await
            .expect("inference requests should be recorded")
            .is_empty()
    );
    auxiliary.verify().await;
}
