use super::*;
use codex_utils_absolute_path::AbsolutePathBuf;
use codex_utils_absolute_path::AbsolutePathBufGuard;
use pretty_assertions::assert_eq;
use std::num::NonZeroU64;
use tempfile::tempdir;

#[test]
fn test_deserialize_ollama_model_provider_toml() {
    let azure_provider_toml = r#"
name = "Ollama"
base_url = "http://localhost:11434/v1"
        "#;
    let expected_provider = ModelProviderInfo {
        name: "Ollama".into(),
        base_url: Some("http://localhost:11434/v1".into()),
        env_key: None,
        env_key_instructions: None,
        experimental_bearer_token: None,
        auth: None,
        aws: None,
        wire_api: WireApi::Responses,
        wire_routes: Default::default(),
        query_params: None,
        http_headers: None,
        env_http_headers: None,
        request_max_retries: None,
        stream_max_retries: None,
        stream_idle_timeout_ms: None,
        websocket_connect_timeout_ms: None,
        requires_openai_auth: false,
        supports_websockets: false,
        supports_standalone_web_search: false,
    };

    let provider: ModelProviderInfo = toml::from_str(azure_provider_toml).unwrap();
    assert_eq!(expected_provider, provider);
}

#[test]
fn test_deserialize_azure_model_provider_toml() {
    let azure_provider_toml = r#"
name = "Azure"
base_url = "https://xxxxx.openai.azure.com/openai"
env_key = "AZURE_OPENAI_API_KEY"
query_params = { api-version = "2025-04-01-preview" }
        "#;
    let expected_provider = ModelProviderInfo {
        name: "Azure".into(),
        base_url: Some("https://xxxxx.openai.azure.com/openai".into()),
        env_key: Some("AZURE_OPENAI_API_KEY".into()),
        env_key_instructions: None,
        experimental_bearer_token: None,
        auth: None,
        aws: None,
        wire_api: WireApi::Responses,
        wire_routes: Default::default(),
        query_params: Some(maplit::hashmap! {
            "api-version".to_string() => "2025-04-01-preview".to_string(),
        }),
        http_headers: None,
        env_http_headers: None,
        request_max_retries: None,
        stream_max_retries: None,
        stream_idle_timeout_ms: None,
        websocket_connect_timeout_ms: None,
        requires_openai_auth: false,
        supports_websockets: false,
        supports_standalone_web_search: false,
    };

    let provider: ModelProviderInfo = toml::from_str(azure_provider_toml).unwrap();
    assert_eq!(expected_provider, provider);
}

#[test]
fn test_deserialize_example_model_provider_toml() {
    let azure_provider_toml = r#"
name = "Example"
base_url = "https://example.com"
env_key = "API_KEY"
http_headers = { "X-Example-Header" = "example-value" }
env_http_headers = { "X-Example-Env-Header" = "EXAMPLE_ENV_VAR" }
supports_standalone_web_search = true
        "#;
    let expected_provider = ModelProviderInfo {
        name: "Example".into(),
        base_url: Some("https://example.com".into()),
        env_key: Some("API_KEY".into()),
        env_key_instructions: None,
        experimental_bearer_token: None,
        auth: None,
        aws: None,
        wire_api: WireApi::Responses,
        wire_routes: Default::default(),
        query_params: None,
        http_headers: Some(maplit::hashmap! {
            "X-Example-Header".to_string() => "example-value".to_string(),
        }),
        env_http_headers: Some(maplit::hashmap! {
            "X-Example-Env-Header".to_string() => "EXAMPLE_ENV_VAR".to_string(),
        }),
        request_max_retries: None,
        stream_max_retries: None,
        stream_idle_timeout_ms: None,
        websocket_connect_timeout_ms: None,
        requires_openai_auth: false,
        supports_websockets: false,
        supports_standalone_web_search: true,
    };

    let provider: ModelProviderInfo = toml::from_str(azure_provider_toml).unwrap();
    assert_eq!(expected_provider, provider);
}

#[test]
fn test_deserialize_chat_wire_api_shows_helpful_error() {
    let provider_toml = r#"
name = "OpenAI using Chat Completions"
base_url = "https://api.openai.com/v1"
env_key = "OPENAI_API_KEY"
wire_api = "chat"
        "#;

    let err = toml::from_str::<ModelProviderInfo>(provider_toml).unwrap_err();
    assert!(
        err.to_string()
            .contains("`wire_api = \"chat\"` is no longer supported")
    );
}

#[test]
fn named_route_toml_and_anthropic_plan_resolve_complete_transport_contract() {
    let provider: ModelProviderInfo = toml::from_str(
        r#"
name = "Claudeflare"
base_url = "http://127.0.0.1:8080/v1/ccflare/openai"
wire_api = "responses"

[wire_routes.claude_code]
wire_api = "anthropic_messages"
dialect = "claude_code"
base_url = "http://127.0.0.1:8080/v1/claude-code"
request_path = "v1/messages"
query_params = { beta = "true" }
stream_max_retries = 10
"#,
    )
    .expect("provider route should deserialize");
    let inference = ModelInferenceConfig::Anthropic {
        wire_api: WireApi::AnthropicMessages,
        dialect: InferenceDialect::ClaudeCode,
        route: "claude_code".to_string(),
        wire_model: "claude-haiku-4-5-20251001".to_string(),
        max_output_tokens: 32_000,
        thinking: AnthropicThinkingPolicy::Budgeted {
            budget_tokens: 31_999,
        },
        supports_disabled_thinking: true,
    };

    let plan = provider
        .resolve_inference_contract("anthropic/claude-haiku-4-5-20251001", Some(&inference))
        .expect("compatible route should resolve");

    assert_eq!(
        plan,
        ResolvedInferencePlan::Anthropic {
            wire_model: "claude-haiku-4-5-20251001".to_string(),
            max_output_tokens: 32_000,
            thinking: AnthropicThinkingPolicy::Budgeted {
                budget_tokens: 31_999,
            },
            supports_disabled_thinking: true,
            route: ResolvedWireRoute {
                name: Some("claude_code".to_string()),
                wire_api: WireApi::AnthropicMessages,
                dialect: InferenceDialect::ClaudeCode,
                base_url: Some("http://127.0.0.1:8080/v1/claude-code".to_string()),
                request_path: "v1/messages".to_string(),
                query_params: Some(HashMap::from([("beta".to_string(), "true".to_string())])),
                request_max_retries: 4,
                stream_max_retries: 10,
                stream_idle_timeout: Duration::from_millis(300_000),
            },
        }
    );
}

#[test]
fn missing_and_incompatible_routes_are_actionable_before_sampling() {
    let inference = ModelInferenceConfig::Anthropic {
        wire_api: WireApi::AnthropicMessages,
        dialect: InferenceDialect::ClaudeCode,
        route: "claude_code".to_string(),
        wire_model: "claude-sonnet-5".to_string(),
        max_output_tokens: 64_000,
        thinking: AnthropicThinkingPolicy::Adaptive,
        supports_disabled_thinking: true,
    };
    let missing = ModelProviderInfo::default()
        .resolve_inference_contract("anthropic/claude-sonnet-5", Some(&inference))
        .expect_err("missing route should fail");
    assert_eq!(
        missing.to_string(),
        "model `anthropic/claude-sonnet-5` (anthropic) requires provider wire route `claude_code`; configure `model_providers.<provider>.wire_routes.claude_code` with `wire_api = \"anthropic_messages\"` and `dialect = \"claude_code\"`"
    );

    let provider = ModelProviderInfo {
        wire_routes: HashMap::from([(
            "claude_code".to_string(),
            ModelProviderWireRoute {
                wire_api: WireApi::Responses,
                dialect: InferenceDialect::OpenAi,
                base_url: "http://127.0.0.1:8080/v1/claude-code".to_string(),
                request_path: "v1/messages".to_string(),
                query_params: None,
                request_max_retries: None,
                stream_max_retries: None,
                stream_idle_timeout_ms: None,
            },
        )]),
        ..ModelProviderInfo::default()
    };
    let incompatible = provider
        .resolve_inference_contract("anthropic/claude-sonnet-5", Some(&inference))
        .expect_err("incompatible route should fail");
    assert_eq!(
        incompatible.to_string(),
        "provider wire route `claude_code` is incompatible with model `anthropic/claude-sonnet-5` (anthropic): expected wire_api `anthropic_messages` and dialect `claude_code`, found wire_api `responses` and dialect `open_ai`"
    );
}

#[test]
fn family_incompatible_profile_is_rejected_even_when_named_route_matches() {
    let inference = ModelInferenceConfig::Anthropic {
        wire_api: WireApi::Responses,
        dialect: InferenceDialect::OpenAi,
        route: "claude_code".to_string(),
        wire_model: "claude-sonnet-5".to_string(),
        max_output_tokens: 64_000,
        thinking: AnthropicThinkingPolicy::Adaptive,
        supports_disabled_thinking: true,
    };
    let provider = ModelProviderInfo {
        wire_routes: HashMap::from([(
            "claude_code".to_string(),
            ModelProviderWireRoute {
                wire_api: WireApi::Responses,
                dialect: InferenceDialect::OpenAi,
                base_url: "http://127.0.0.1:8080/v1/claude-code".to_string(),
                request_path: "responses".to_string(),
                query_params: None,
                request_max_retries: None,
                stream_max_retries: None,
                stream_idle_timeout_ms: None,
            },
        )]),
        ..ModelProviderInfo::default()
    };

    assert_eq!(
        provider
            .resolve_inference_contract("anthropic/claude-sonnet-5", Some(&inference))
            .expect_err("family-incompatible profile should fail")
            .to_string(),
        "model `anthropic/claude-sonnet-5` declares an incompatible inference contract for family `anthropic`: `wire_api = \"responses\"` with `dialect = \"open_ai\"`; supported: `wire_api = \"anthropic_messages\"` with `dialect = \"claude_code\"`"
    );
}

#[test]
fn metadata_free_model_resolves_legacy_provider_route() {
    let provider = ModelProviderInfo {
        base_url: Some("https://example.test/v1".to_string()),
        wire_api: WireApi::Responses,
        stream_max_retries: Some(8),
        ..ModelProviderInfo::default()
    };

    assert_eq!(
        provider
            .resolve_inference_contract("legacy-model", None)
            .expect("legacy route should resolve"),
        ResolvedInferencePlan::Legacy {
            wire_model: "legacy-model".to_string(),
            route: ResolvedWireRoute {
                name: None,
                wire_api: WireApi::Responses,
                dialect: InferenceDialect::OpenAi,
                base_url: Some("https://example.test/v1".to_string()),
                request_path: "responses".to_string(),
                query_params: None,
                request_max_retries: 4,
                stream_max_retries: 8,
                stream_idle_timeout: Duration::from_millis(300_000),
            },
        }
    );
}

#[test]
fn test_deserialize_websocket_connect_timeout() {
    let provider_toml = r#"
name = "OpenAI"
base_url = "https://api.openai.com/v1"
websocket_connect_timeout_ms = 15000
supports_websockets = true
        "#;

    let provider: ModelProviderInfo = toml::from_str(provider_toml).unwrap();
    assert_eq!(provider.websocket_connect_timeout_ms, Some(15_000));
}

#[test]
fn test_supports_remote_compaction_for_openai() {
    let provider = ModelProviderInfo::create_openai_provider(/*base_url*/ None);

    assert!(provider.supports_remote_compaction());
}

#[test]
fn test_personal_access_token_uses_chatgpt_codex_base_url() {
    let api_provider = ModelProviderInfo::create_openai_provider(/*base_url*/ None)
        .to_api_provider(Some(AuthMode::PersonalAccessToken))
        .expect("OpenAI provider should build API provider");

    assert_eq!(api_provider.base_url, CHATGPT_CODEX_BASE_URL);
}

#[test]
fn test_header_auth_uses_chatgpt_codex_base_url() {
    let api_provider = ModelProviderInfo::create_openai_provider(/*base_url*/ None)
        .to_api_provider(Some(AuthMode::Headers))
        .expect("OpenAI provider should build API provider");

    assert_eq!(api_provider.base_url, CHATGPT_CODEX_BASE_URL);
}

#[test]
fn test_supports_remote_compaction_for_azure_name() {
    let provider = ModelProviderInfo {
        name: "Azure".into(),
        base_url: Some("https://example.com/openai".into()),
        env_key: Some("AZURE_OPENAI_API_KEY".into()),
        env_key_instructions: None,
        experimental_bearer_token: None,
        auth: None,
        aws: None,
        wire_api: WireApi::Responses,
        wire_routes: Default::default(),
        query_params: None,
        http_headers: None,
        env_http_headers: None,
        request_max_retries: None,
        stream_max_retries: None,
        stream_idle_timeout_ms: None,
        websocket_connect_timeout_ms: None,
        requires_openai_auth: false,
        supports_websockets: false,
        supports_standalone_web_search: false,
    };

    assert!(provider.supports_remote_compaction());
}

#[test]
fn test_supports_remote_compaction_for_non_openai_non_azure_provider() {
    let provider = ModelProviderInfo {
        name: "Example".into(),
        base_url: Some("https://example.com/v1".into()),
        env_key: Some("API_KEY".into()),
        env_key_instructions: None,
        experimental_bearer_token: None,
        auth: None,
        aws: None,
        wire_api: WireApi::Responses,
        wire_routes: Default::default(),
        query_params: None,
        http_headers: None,
        env_http_headers: None,
        request_max_retries: None,
        stream_max_retries: None,
        stream_idle_timeout_ms: None,
        websocket_connect_timeout_ms: None,
        requires_openai_auth: false,
        supports_websockets: false,
        supports_standalone_web_search: false,
    };

    assert!(!provider.supports_remote_compaction());
}

#[test]
fn test_uses_openai_actor_authorization() {
    let mut provider = ModelProviderInfo {
        http_headers: Some(maplit::hashmap! {
            "X-OpenAI-Actor-Authorization".to_string() => "actor-token".to_string(),
        }),
        ..ModelProviderInfo::default()
    };
    assert!(provider.uses_openai_actor_authorization());

    provider.http_headers = None;
    assert!(!provider.uses_openai_actor_authorization());

    provider.http_headers = Some(maplit::hashmap! {
        OPENAI_ACTOR_AUTHORIZATION_HEADER.to_string() => "  ".to_string(),
    });
    assert!(!provider.uses_openai_actor_authorization());

    provider.http_headers = Some(maplit::hashmap! {
        OPENAI_ACTOR_AUTHORIZATION_HEADER.to_string() => "actor-token".to_string(),
    });
    provider.requires_openai_auth = true;
    assert!(!provider.uses_openai_actor_authorization());
}

#[test]
fn test_deserialize_provider_auth_config_defaults() {
    let base_dir = tempdir().unwrap();
    let provider_toml = r#"
name = "Corp"

[auth]
command = "./scripts/print-token"
args = ["--format=text"]
        "#;

    let provider: ModelProviderInfo = {
        let _guard = AbsolutePathBufGuard::new(base_dir.path());
        toml::from_str(provider_toml).unwrap()
    };

    assert_eq!(
        provider.auth,
        Some(ModelProviderAuthInfo {
            command: "./scripts/print-token".to_string(),
            args: vec!["--format=text".to_string()],
            timeout_ms: NonZeroU64::new(5_000).unwrap(),
            refresh_interval_ms: 300_000,
            cwd: AbsolutePathBuf::resolve_path_against_base(".", base_dir.path()),
        })
    );
}

#[test]
fn test_deserialize_provider_aws_config() {
    let provider_toml = r#"
name = "Amazon Bedrock"
base_url = "https://bedrock.example.com/v1"

[aws]
profile = "codex-bedrock"
region = "us-west-2"
        "#;

    let provider: ModelProviderInfo = toml::from_str(provider_toml).unwrap();

    assert_eq!(
        provider.aws,
        Some(ModelProviderAwsAuthInfo {
            profile: Some("codex-bedrock".to_string()),
            region: Some("us-west-2".to_string()),
        })
    );
}

#[test]
fn test_create_amazon_bedrock_provider() {
    assert_eq!(
        ModelProviderInfo::create_amazon_bedrock_provider(/*aws*/ None),
        ModelProviderInfo {
            name: "Amazon Bedrock".to_string(),
            base_url: None,
            env_key: None,
            env_key_instructions: None,
            experimental_bearer_token: None,
            auth: None,
            aws: Some(ModelProviderAwsAuthInfo {
                profile: None,
                region: None,
            }),
            wire_api: WireApi::Responses,
            wire_routes: Default::default(),
            query_params: None,
            http_headers: Some(maplit::hashmap! {
                AMAZON_BEDROCK_MANTLE_CLIENT_AGENT_HEADER.to_string() =>
                    AMAZON_BEDROCK_MANTLE_CLIENT_AGENT_VALUE.to_string(),
            }),
            env_http_headers: None,
            request_max_retries: None,
            stream_max_retries: None,
            stream_idle_timeout_ms: None,
            websocket_connect_timeout_ms: None,
            requires_openai_auth: false,
            supports_websockets: false,
            supports_standalone_web_search: false,
        }
    );
}

fn provider_auth_for_test() -> ModelProviderAuthInfo {
    ModelProviderAuthInfo {
        command: "token-fetcher".to_string(),
        args: vec!["fetch".to_string()],
        timeout_ms: NonZeroU64::new(5_000).expect("timeout should be non-zero"),
        refresh_interval_ms: 300_000,
        cwd: std::env::current_dir()
            .expect("current directory should be available")
            .try_into()
            .expect("current directory should be absolute"),
    }
}

#[test]
fn test_amazon_bedrock_provider_adds_mantle_client_agent_header() {
    let api_provider = ModelProviderInfo::create_amazon_bedrock_provider(/*aws*/ None)
        .to_api_provider(/*auth_mode*/ None)
        .expect("Amazon Bedrock provider should build API provider");

    assert_eq!(
        api_provider
            .headers
            .get(AMAZON_BEDROCK_MANTLE_CLIENT_AGENT_HEADER)
            .and_then(|value| value.to_str().ok()),
        Some(AMAZON_BEDROCK_MANTLE_CLIENT_AGENT_VALUE)
    );
}

#[test]
fn test_built_in_model_providers_include_amazon_bedrock() {
    let providers = built_in_model_providers(/*openai_base_url*/ None);

    assert_eq!(
        providers
            .get(AMAZON_BEDROCK_PROVIDER_ID)
            .map(ModelProviderInfo::is_amazon_bedrock),
        Some(true)
    );
}

#[test]
fn built_in_claudeflare_provider_has_managed_native_routes() {
    let provider = built_in_model_providers(/*openai_base_url*/ None)
        .remove(CLAUDEFLARE_PROVIDER_ID)
        .expect("Claudeflare provider should be built in");

    assert_eq!(
        provider,
        ModelProviderInfo {
            name: "Claudeflare".to_string(),
            base_url: Some(CLAUDEFLARE_RESPONSES_BASE_URL.to_string()),
            wire_api: WireApi::Responses,
            wire_routes: HashMap::from([
                (
                    "claude_code".to_string(),
                    ModelProviderWireRoute {
                        wire_api: WireApi::AnthropicMessages,
                        dialect: InferenceDialect::ClaudeCode,
                        base_url: CLAUDEFLARE_CLAUDE_BASE_URL.to_string(),
                        request_path: "v1/messages".to_string(),
                        query_params: Some(HashMap::from([(
                            "beta".to_string(),
                            "true".to_string(),
                        )])),
                        request_max_retries: None,
                        stream_max_retries: Some(10),
                        stream_idle_timeout_ms: None,
                    },
                ),
                (
                    "kimi_code".to_string(),
                    ModelProviderWireRoute {
                        wire_api: WireApi::ChatCompletions,
                        dialect: InferenceDialect::Kimi,
                        base_url: CLAUDEFLARE_KIMI_BASE_URL.to_string(),
                        request_path: "chat/completions".to_string(),
                        query_params: None,
                        request_max_retries: None,
                        stream_max_retries: Some(10),
                        stream_idle_timeout_ms: None,
                    },
                ),
                (
                    "grok".to_string(),
                    ModelProviderWireRoute {
                        wire_api: WireApi::Responses,
                        dialect: InferenceDialect::Grok,
                        base_url: CLAUDEFLARE_GROK_BASE_URL.to_string(),
                        request_path: "responses".to_string(),
                        query_params: None,
                        request_max_retries: Some(0),
                        stream_max_retries: Some(10),
                        stream_idle_timeout_ms: None,
                    },
                ),
            ]),
            stream_max_retries: Some(10),
            supports_websockets: false,
            ..ModelProviderInfo::default()
        }
    );
}

#[test]
fn built_in_claudeflare_resolves_grok_route() {
    let provider = built_in_model_providers(/*openai_base_url*/ None)
        .remove(CLAUDEFLARE_PROVIDER_ID)
        .expect("Claudeflare provider should be built in");
    let inference = ModelInferenceConfig::Grok(GrokInferenceConfig {
        wire_api: WireApi::Responses,
        dialect: InferenceDialect::Grok,
        route: "grok".to_string(),
        wire_model: "grok-4.6".to_string(),
    });

    assert_eq!(
        provider
            .resolve_inference_contract("xai/grok-4.6", Some(&inference))
            .expect("Grok route should resolve"),
        ResolvedInferencePlan::Grok {
            config: GrokInferenceConfig {
                wire_api: WireApi::Responses,
                dialect: InferenceDialect::Grok,
                route: "grok".to_string(),
                wire_model: "grok-4.6".to_string(),
            },
            route: ResolvedWireRoute {
                name: Some("grok".to_string()),
                wire_api: WireApi::Responses,
                dialect: InferenceDialect::Grok,
                base_url: Some(CLAUDEFLARE_GROK_BASE_URL.to_string()),
                request_path: "responses".to_string(),
                query_params: None,
                request_max_retries: 0,
                stream_max_retries: 10,
                stream_idle_timeout: Duration::from_millis(DEFAULT_STREAM_IDLE_TIMEOUT_MS),
            },
        }
    );
}

#[test]
fn test_merge_configured_model_providers_adds_custom_provider() {
    let custom_provider = ModelProviderInfo {
        name: "Custom".to_string(),
        base_url: Some("https://example.com/v1".to_string()),
        ..ModelProviderInfo::default()
    };
    let configured_model_providers =
        std::collections::HashMap::from([("custom".to_string(), custom_provider.clone())]);

    let mut expected = built_in_model_providers(/*openai_base_url*/ None);
    expected.insert("custom".to_string(), custom_provider);

    assert_eq!(
        merge_configured_model_providers(
            built_in_model_providers(/*openai_base_url*/ None),
            configured_model_providers,
        ),
        Ok(expected)
    );
}

#[test]
fn test_merge_configured_model_providers_applies_amazon_bedrock_profile_override() {
    let configured_model_providers = std::collections::HashMap::from([(
        AMAZON_BEDROCK_PROVIDER_ID.to_string(),
        ModelProviderInfo {
            aws: Some(ModelProviderAwsAuthInfo {
                profile: Some("codex-bedrock".to_string()),
                region: Some("us-west-2".to_string()),
            }),
            ..ModelProviderInfo::default()
        },
    )]);

    let mut expected = built_in_model_providers(/*openai_base_url*/ None);
    expected
        .get_mut(AMAZON_BEDROCK_PROVIDER_ID)
        .expect("Amazon Bedrock provider should be built in")
        .aws = Some(ModelProviderAwsAuthInfo {
        profile: Some("codex-bedrock".to_string()),
        region: Some("us-west-2".to_string()),
    });

    assert_eq!(
        merge_configured_model_providers(
            built_in_model_providers(/*openai_base_url*/ None),
            configured_model_providers,
        ),
        Ok(expected)
    );
}

#[test]
fn test_merge_configured_model_providers_applies_amazon_bedrock_transport_overrides() {
    let auth = provider_auth_for_test();
    let configured_model_providers = std::collections::HashMap::from([(
        AMAZON_BEDROCK_PROVIDER_ID.to_string(),
        ModelProviderInfo {
            base_url: Some("https://proxy.example.com/v1".to_string()),
            auth: Some(auth.clone()),
            aws: Some(ModelProviderAwsAuthInfo {
                profile: Some("codex-bedrock".to_string()),
                region: Some("us-west-2".to_string()),
            }),
            http_headers: Some(maplit::hashmap! {
                "x-example-header".to_string() => "value".to_string(),
            }),
            ..ModelProviderInfo::default()
        },
    )]);

    let mut expected = built_in_model_providers(/*openai_base_url*/ None);
    let expected_provider = expected
        .get_mut(AMAZON_BEDROCK_PROVIDER_ID)
        .expect("Amazon Bedrock provider should be built in");
    expected_provider.base_url = Some("https://proxy.example.com/v1".to_string());
    expected_provider.auth = Some(auth);
    expected_provider.aws = Some(ModelProviderAwsAuthInfo {
        profile: Some("codex-bedrock".to_string()),
        region: Some("us-west-2".to_string()),
    });
    expected_provider
        .http_headers
        .get_or_insert_default()
        .insert("x-example-header".to_string(), "value".to_string());

    assert_eq!(
        merge_configured_model_providers(
            built_in_model_providers(/*openai_base_url*/ None),
            configured_model_providers,
        ),
        Ok(expected)
    );
}

#[test]
fn test_merge_configured_model_providers_rejects_amazon_bedrock_non_default_fields() {
    let configured_model_providers = std::collections::HashMap::from([(
        AMAZON_BEDROCK_PROVIDER_ID.to_string(),
        ModelProviderInfo {
            name: "Custom Bedrock".to_string(),
            aws: Some(ModelProviderAwsAuthInfo {
                profile: Some("codex-bedrock".to_string()),
                region: None,
            }),
            ..ModelProviderInfo::default()
        },
    )]);

    assert_eq!(
        merge_configured_model_providers(
            built_in_model_providers(/*openai_base_url*/ None),
            configured_model_providers,
        ),
        Err(
            "model_providers.amazon-bedrock only supports changing `base_url`, `auth`, `http_headers`, `aws.profile`, and `aws.region`; other non-default provider fields are not supported"
                .to_string()
        )
    );
}

#[test]
fn test_merge_configured_model_providers_allows_amazon_bedrock_default_fields() {
    let configured_model_providers = std::collections::HashMap::from([(
        AMAZON_BEDROCK_PROVIDER_ID.to_string(),
        ModelProviderInfo {
            aws: Some(ModelProviderAwsAuthInfo {
                profile: None,
                region: None,
            }),
            wire_api: WireApi::Responses,
            ..ModelProviderInfo::default()
        },
    )]);

    assert_eq!(
        merge_configured_model_providers(
            built_in_model_providers(/*openai_base_url*/ None),
            configured_model_providers,
        ),
        Ok(built_in_model_providers(/*openai_base_url*/ None))
    );
}

#[test]
fn test_validate_provider_aws_rejects_conflicting_auth() {
    let provider = ModelProviderInfo {
        aws: Some(ModelProviderAwsAuthInfo {
            profile: None,
            region: None,
        }),
        env_key: Some("AWS_BEARER_TOKEN_BEDROCK".to_string()),
        supports_websockets: false,
        ..ModelProviderInfo::create_openai_provider(/*base_url*/ None)
    };

    assert_eq!(
        provider.validate(),
        Err("provider aws cannot be combined with env_key, requires_openai_auth".to_string())
    );
}

#[test]
fn test_validate_provider_aws_rejects_websockets() {
    let provider = ModelProviderInfo {
        aws: Some(ModelProviderAwsAuthInfo {
            profile: None,
            region: None,
        }),
        requires_openai_auth: false,
        supports_websockets: true,
        ..ModelProviderInfo::create_openai_provider(/*base_url*/ None)
    };

    assert_eq!(
        provider.validate(),
        Err("provider aws cannot be combined with supports_websockets".to_string())
    );
}

#[test]
fn test_deserialize_provider_auth_config_allows_zero_refresh_interval() {
    let base_dir = tempdir().unwrap();
    let provider_toml = r#"
name = "Corp"

[auth]
command = "./scripts/print-token"
refresh_interval_ms = 0
        "#;

    let provider: ModelProviderInfo = {
        let _guard = AbsolutePathBufGuard::new(base_dir.path());
        toml::from_str(provider_toml).unwrap()
    };

    let auth = provider.auth.expect("auth config should deserialize");
    assert_eq!(auth.refresh_interval_ms, 0);
    assert_eq!(auth.refresh_interval(), None);
}
