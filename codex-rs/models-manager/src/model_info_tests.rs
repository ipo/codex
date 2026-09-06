use super::*;
use crate::ModelsManagerConfig;
use codex_protocol::config_types::Personality;
use codex_protocol::model_inference::GrokInferenceConfig;
use codex_protocol::model_inference::InferenceDialect;
use codex_protocol::model_inference::KimiInferenceConfig;
use codex_protocol::model_inference::KimiThinkingPolicy;
use codex_protocol::model_inference::ModelInferenceConfig;
use codex_protocol::model_inference::WireApi;
use codex_protocol::openai_models::ApprovalMessages;
use codex_protocol::openai_models::AutoReviewMessages;
use codex_protocol::openai_models::CollaborationModeMessages;
use codex_protocol::openai_models::ConfirmationPolicies;
use codex_protocol::openai_models::GuardianV2ModelConfig;
use codex_protocol::openai_models::ModelTokenBudgetConfig;
use codex_protocol::openai_models::MultiAgentMessages;
use codex_protocol::openai_models::MultiAgentModeMessages;
use codex_protocol::openai_models::MultiAgentRoleMessages;
use codex_protocol::openai_models::PermissionMessages;
use codex_protocol::openai_models::ToolMessage;
use codex_protocol::openai_models::ToolMessages;
use pretty_assertions::assert_eq;

fn config_with_personality(personality: Option<Personality>) -> ModelsManagerConfig {
    ModelsManagerConfig {
        personality_enabled: true,
        personality,
        ..Default::default()
    }
}

#[test]
fn base_instruction_override_is_literal_and_preserves_catalog_messages() {
    let override_instructions = "override {{ personality }}";
    let persistent_instructions = "Follow up on the active task.";
    let async_message_description = "Catalog async message description.";
    let mut model = model_info_from_slug("unknown-model");
    let approvals = ApprovalMessages {
        on_request: Some("user approvals".to_string()),
        on_request_auto_review: Some("auto approvals".to_string()),
        never: Some("never approvals".to_string()),
        unless_trusted: Some("unless-trusted approvals".to_string()),
    };
    let collaboration_modes = CollaborationModeMessages {
        default: Some("default instructions".to_string()),
        plan: Some("plan instructions".to_string()),
    };
    let auto_review = AutoReviewMessages {
        policy: Some("review policy".to_string()),
        policy_template: Some("review policy template".to_string()),
        node_repl_policy: None,
        rejection_instructions: Some("rejection instructions".to_string()),
        timeout_instructions: Some(String::new()),
    };
    let permissions = PermissionMessages {
        danger_full_access: Some("danger".to_string()),
        workspace_write: Some(String::new()),
        read_only: None,
    };
    let multi_agent = MultiAgentMessages {
        role: Some(MultiAgentRoleMessages {
            root: Some("root base".to_string()),
            subagent: Some("subagent base".to_string()),
        }),
        mode: Some(MultiAgentModeMessages {
            explicit: Some("explicit mode".to_string()),
            proactive: Some("proactive mode".to_string()),
            hint_text: Some("mode hint".to_string()),
        }),
    };
    let token_budget = ModelTokenBudgetConfig {
        enabled: false,
        use_history_notes_extension: false,
        reminder_threshold_tokens: 128,
        reminder_message_template: "budget reminder".to_string(),
        guidance_message: "budget guidance".to_string(),
        auto_compact_fallback_prompt: "compact prompt".to_string(),
        auto_compact_fallback_buffer_tokens: 64,
    };
    let guardian_v2 = GuardianV2ModelConfig {
        classifier_instructions: Some("Guardian experiment".to_string()),
        ..Default::default()
    };
    let confirmation_policies = ConfirmationPolicies {
        browser_use: Some("# Browser policy\n\n{{literal_markdown}}\n".to_string()),
        computer_use: Some("  # Native policy\r\n\n${native_markdown}\n".to_string()),
    };
    model.model_messages = Some(ModelMessages {
        persistent_instructions: Some(persistent_instructions.to_string()),
        tools: Some(ToolMessages {
            send_user_message_async: Some(ToolMessage {
                description: Some(async_message_description.to_string()),
            }),
        }),
        instructions_template: Some("template".to_string()),
        instructions_variables: Some(ModelInstructionsVariables {
            personality_default: Some("default".to_string()),
            personality_friendly: Some("friendly".to_string()),
            personality_pragmatic: Some("pragmatic".to_string()),
        }),
        approvals: Some(approvals.clone()),
        collaboration_modes: Some(collaboration_modes.clone()),
        auto_review: Some(auto_review.clone()),
        permissions: Some(permissions.clone()),
        multi_agent: Some(multi_agent.clone()),
        token_budget: Some(token_budget.clone()),
        confirmation_policies: Some(confirmation_policies.clone()),
        guardian_v2: Some(guardian_v2.clone()),
    });
    let config = ModelsManagerConfig {
        base_instructions: Some(override_instructions.to_string()),
        ..Default::default()
    };

    let updated = with_config_overrides(model, &config);

    assert_eq!(
        updated.model_messages,
        Some(ModelMessages {
            persistent_instructions: Some(persistent_instructions.to_string()),
            tools: Some(ToolMessages {
                send_user_message_async: Some(ToolMessage {
                    description: Some(async_message_description.to_string()),
                }),
            }),
            instructions_template: Some(override_instructions.to_string()),
            instructions_variables: None,
            approvals: Some(approvals),
            collaboration_modes: Some(collaboration_modes),
            auto_review: Some(auto_review),
            permissions: Some(permissions),
            multi_agent: Some(multi_agent),
            token_budget: Some(token_budget),
            confirmation_policies: Some(confirmation_policies),
            guardian_v2: Some(guardian_v2),
        })
    );
    assert_eq!(
        updated.get_model_instructions(/*personality*/ None),
        override_instructions
    );
}

#[test]
fn disabled_personality_bakes_default_and_preserves_catalog_approval_messages() {
    let mut model = model_info_from_slug("unknown-model");
    let approvals = ApprovalMessages {
        on_request: Some("user approvals".to_string()),
        on_request_auto_review: None,
        never: None,
        unless_trusted: None,
    };
    model.model_messages = Some(ModelMessages {
        persistent_instructions: Some(String::new()),
        tools: Some(ToolMessages {
            send_user_message_async: Some(ToolMessage {
                description: Some(String::new()),
            }),
        }),
        instructions_template: Some("before {{ personality }} after".to_string()),
        instructions_variables: Some(ModelInstructionsVariables {
            personality_default: Some("default".to_string()),
            personality_friendly: Some("friendly".to_string()),
            personality_pragmatic: Some("pragmatic".to_string()),
        }),
        approvals: Some(approvals.clone()),
        collaboration_modes: None,
        auto_review: None,
        permissions: None,
        multi_agent: None,
        token_budget: None,
        confirmation_policies: None,
        guardian_v2: None,
    });
    let config = ModelsManagerConfig {
        personality_enabled: false,
        ..Default::default()
    };

    let updated = with_config_overrides(model, &config);

    assert_eq!(
        updated.model_messages,
        Some(ModelMessages {
            persistent_instructions: Some(String::new()),
            tools: Some(ToolMessages {
                send_user_message_async: Some(ToolMessage {
                    description: Some(String::new()),
                }),
            }),
            instructions_template: Some("before default after".to_string()),
            instructions_variables: None,
            approvals: Some(approvals),
            collaboration_modes: None,
            auto_review: None,
            permissions: None,
            multi_agent: None,
            token_budget: None,
            confirmation_policies: None,
            guardian_v2: None,
        })
    );
}

#[test]
fn disabled_personality_uses_plain_base_instructions_for_local_personality_models() {
    let config = ModelsManagerConfig {
        personality_enabled: false,
        personality: Some(Personality::Friendly),
        ..Default::default()
    };

    for slug in ["gpt-5.2-codex", "exp-codex-personality"] {
        let updated = with_config_overrides(model_info_from_slug(slug), &config);

        assert_eq!(
            updated.model_messages,
            Some(ModelMessages {
                persistent_instructions: None,
                tools: None,
                instructions_template: Some(BASE_INSTRUCTIONS.to_string()),
                instructions_variables: None,
                approvals: None,
                collaboration_modes: None,
                auto_review: None,
                permissions: None,
                multi_agent: None,
                token_budget: None,
                confirmation_policies: None,
                guardian_v2: None,
            }),
            "unexpected model messages for {slug}"
        );
    }
}

#[test]
fn personality_none_strips_catalog_instruction_sources_through_the_next_h1() {
    let cases = [
        (
            "Intro\n\n# Personality\n\nRemove me\n\n## Writing Style\n\nRemove me too\n\n# Safety\n\nKeep me",
            "Intro\n\n# Safety\n\nKeep me",
        ),
        ("Intro\n\n# Personality\n\nRemove me", "Intro\n\n"),
        (
            "Intro\n\n## Personality\n\nKeep me",
            "Intro\n\n## Personality\n\nKeep me",
        ),
        (
            "Intro\n\n# Personality \n\nKeep me",
            "Intro\n\n# Personality \n\nKeep me",
        ),
        (
            "Intro\r\n\r\n# Personality\r\n\r\nRemove me\r\n\r\n## Writing Style\r\n\r\nRemove me too\r\n\r\n# General\r\n\r\nKeep me",
            "Intro\r\n\r\n# General\r\n\r\nKeep me",
        ),
    ];
    let config = config_with_personality(Some(Personality::None));

    for (instructions, expected) in cases {
        let mut model = model_info_from_slug("unknown-model");
        model.model_messages = Some(ModelMessages {
            persistent_instructions: None,
            tools: None,
            instructions_template: Some(instructions.to_string()),
            instructions_variables: None,
            approvals: None,
            collaboration_modes: None,
            auto_review: None,
            permissions: None,
            multi_agent: None,
            token_budget: None,
            confirmation_policies: None,
            guardian_v2: None,
        });

        let updated = with_config_overrides(model, &config);
        let instructions_template = updated
            .model_messages
            .as_ref()
            .and_then(|messages| messages.instructions_template.as_deref());

        assert_eq!(
            (
                updated.get_model_instructions(Some(Personality::None)),
                instructions_template
            ),
            (expected.to_string(), Some(expected))
        );
    }
}

#[test]
fn baked_personality_section_is_preserved_without_enabled_explicit_none() {
    let instructions = "Intro\n# Personality\nKeep me\n# General\nKeep me too";
    let configs = [
        config_with_personality(/*personality*/ None),
        config_with_personality(Some(Personality::Friendly)),
        config_with_personality(Some(Personality::Pragmatic)),
        ModelsManagerConfig {
            personality: Some(Personality::None),
            ..Default::default()
        },
    ];

    for config in configs {
        let mut model = model_info_from_slug("unknown-model");
        model
            .model_messages
            .as_mut()
            .expect("fallback model messages")
            .instructions_template = Some(instructions.to_string());

        assert_eq!(
            with_config_overrides(model, &config).get_model_instructions(config.personality),
            instructions
        );
    }
}

#[test]
fn unknown_model_uses_builtin_instruction_template() {
    let model = model_info_from_slug("unknown-model");

    assert_eq!(
        model.get_model_instructions(/*personality*/ None),
        BASE_INSTRUCTIONS
    );
    assert!(model.used_fallback_model_metadata);
}

#[test]
fn external_inference_uses_neutral_identity_without_rewriting_openai_instructions() {
    let catalog = crate::bundled_models_response().expect("bundled catalog should parse");
    let openai = catalog
        .models
        .into_iter()
        .find(|model| model.slug == "gpt-6-astra")
        .expect("bundled Astra model");
    let config = ModelsManagerConfig::default();
    let openai_instructions = with_config_overrides(openai.clone(), &config)
        .get_model_instructions(/*personality*/ None);
    let mut external = openai;
    external.slug = "xai/grok-4.6".to_string();
    external.inference = Some(ModelInferenceConfig::Grok(GrokInferenceConfig {
        wire_api: WireApi::Responses,
        dialect: InferenceDialect::Grok,
        route: "grok".to_string(),
        wire_model: "grok-4.6".to_string(),
    }));
    let external_instructions =
        with_config_overrides(external, &config).get_model_instructions(/*personality*/ None);

    assert!(openai_instructions.starts_with("You are Codex, an agent based on GPT-6."));
    assert!(external_instructions.starts_with("You are Codex, an AI coding agent."));
    assert!(!external_instructions.contains("based on GPT-6"));
}

#[test]
fn bundled_kimi_profiles_resolve_inherited_runtime_metadata() {
    let catalog = crate::bundled_models_response().expect("bundled catalog should parse");
    let profiles = [
        "kimi/k3",
        "kimi/k3-256k",
        "kimi/kimi-for-coding",
        "kimi/kimi-for-coding-highspeed",
    ]
    .map(|slug| {
        let model = catalog
            .models
            .iter()
            .find(|model| model.slug == slug)
            .unwrap_or_else(|| panic!("missing bundled profile {slug}"));
        (
            model.slug.as_str(),
            model.context_window,
            model.max_context_window,
            model.auto_compact_token_limit,
            model.effective_context_window_percent,
            model.input_modalities.clone(),
            model.default_reasoning_level.clone(),
            model.inference.clone(),
        )
    });

    assert_eq!(
        profiles,
        [
            (
                "kimi/k3",
                Some(1_048_576),
                Some(1_048_576),
                Some(891_289),
                100,
                vec![
                    codex_protocol::openai_models::InputModality::Text,
                    codex_protocol::openai_models::InputModality::Image,
                ],
                Some(codex_protocol::openai_models::ReasoningEffort::High),
                Some(ModelInferenceConfig::Kimi(KimiInferenceConfig {
                    wire_api: WireApi::ChatCompletions,
                    dialect: InferenceDialect::Kimi,
                    route: "kimi_code".to_string(),
                    wire_model: "k3".to_string(),
                    max_output_tokens: 131_072,
                    thinking: KimiThinkingPolicy::RequiredWithEffort,
                })),
            ),
            (
                "kimi/k3-256k",
                Some(262_144),
                Some(262_144),
                Some(212_144),
                100,
                vec![
                    codex_protocol::openai_models::InputModality::Text,
                    codex_protocol::openai_models::InputModality::Image,
                ],
                Some(codex_protocol::openai_models::ReasoningEffort::High),
                Some(ModelInferenceConfig::Kimi(KimiInferenceConfig {
                    wire_api: WireApi::ChatCompletions,
                    dialect: InferenceDialect::Kimi,
                    route: "kimi_code".to_string(),
                    wire_model: "k3-256k".to_string(),
                    max_output_tokens: 131_072,
                    thinking: KimiThinkingPolicy::RequiredWithEffort,
                })),
            ),
            (
                "kimi/kimi-for-coding",
                Some(262_144),
                Some(262_144),
                Some(212_144),
                100,
                vec![
                    codex_protocol::openai_models::InputModality::Text,
                    codex_protocol::openai_models::InputModality::Image,
                ],
                None,
                Some(ModelInferenceConfig::Kimi(KimiInferenceConfig {
                    wire_api: WireApi::ChatCompletions,
                    dialect: InferenceDialect::Kimi,
                    route: "kimi_code".to_string(),
                    wire_model: "kimi-for-coding".to_string(),
                    max_output_tokens: 32_768,
                    thinking: KimiThinkingPolicy::Required,
                })),
            ),
            (
                "kimi/kimi-for-coding-highspeed",
                Some(262_144),
                Some(262_144),
                Some(212_144),
                100,
                vec![
                    codex_protocol::openai_models::InputModality::Text,
                    codex_protocol::openai_models::InputModality::Image,
                ],
                None,
                Some(ModelInferenceConfig::Kimi(KimiInferenceConfig {
                    wire_api: WireApi::ChatCompletions,
                    dialect: InferenceDialect::Kimi,
                    route: "kimi_code".to_string(),
                    wire_model: "kimi-for-coding-highspeed".to_string(),
                    max_output_tokens: 32_768,
                    thinking: KimiThinkingPolicy::Required,
                })),
            ),
        ]
    );
}

#[test]
fn model_context_window_override_clamps_to_max_context_window() {
    let mut model = model_info_from_slug("unknown-model");
    model.context_window = Some(273_000);
    model.max_context_window = Some(400_000);
    let config = ModelsManagerConfig {
        model_context_window: Some(500_000),
        ..Default::default()
    };

    let updated = with_config_overrides(model.clone(), &config);
    let mut expected = model;
    expected.context_window = Some(400_000);

    assert_eq!(updated, expected);
}

#[test]
fn model_context_window_uses_model_value_without_override() {
    let mut model = model_info_from_slug("unknown-model");
    model.context_window = Some(273_000);
    model.max_context_window = Some(400_000);
    let config = ModelsManagerConfig::default();

    let updated = with_config_overrides(model.clone(), &config);

    assert_eq!(updated, model);
}
