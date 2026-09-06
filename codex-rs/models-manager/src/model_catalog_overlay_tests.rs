use super::*;
use crate::bundled_models_response;
use codex_protocol::openai_models::ModelToolCapability;
use codex_protocol::openai_models::ModelVisibility;
use codex_protocol::openai_models::ReasoningEffort;
use pretty_assertions::assert_eq;
use serde_json::json;

fn apply(value: Value) -> Result<ModelsResponse, ModelCatalogOverlayError> {
    ModelCatalogOverlay::from_json(&value.to_string())?
        .apply(bundled_models_response().expect("bundled catalog should parse"))
}

#[test]
fn patches_existing_and_adds_inherited_model_without_changing_unrelated_models() {
    let original = bundled_models_response().expect("bundled catalog should parse");
    let parent = original.models.first().expect("bundled model");
    let unrelated = original
        .models
        .get(1)
        .expect("second bundled model")
        .clone();
    let added_slug = "external/overlay-model";

    let applied = apply(json!({"models": [
        {"slug": parent.slug, "display_name": "Patched", "visibility": "hide"},
        {
            "slug": added_slug,
            "inherits": parent.slug,
            "display_name": "External",
            "visibility": "list",
            "supported_reasoning_levels": [
                {"effort": "on", "description": "Provider reasoning"}
            ],
            "default_reasoning_level": "on",
            "use_responses_lite": true,
            "tool_mode": "code_mode"
        }
    ]}))
    .expect("overlay should apply");

    assert_eq!(applied.models.get(1), Some(&unrelated));
    let patched = applied.models.first().expect("patched model");
    assert_eq!(patched.display_name, "Patched");
    assert_eq!(patched.visibility, ModelVisibility::Hide);
    let added = applied.models.last().expect("added model");
    assert_eq!(added.slug, added_slug);
    assert_eq!(
        added.default_reasoning_level,
        Some(ReasoningEffort::Custom("on".to_string()))
    );
    assert!(added.use_responses_lite);
}

#[test]
fn shallow_replacement_and_explicit_null_are_preserved() {
    let original = bundled_models_response().expect("bundled catalog should parse");
    let parent = original.models.first().expect("bundled model");
    let applied = apply(json!({"models": [{
        "slug": parent.slug,
        "description": null,
        "model_messages": null,
        "supported_reasoning_levels": []
    }]}))
    .expect("overlay should apply");
    let patched = applied.models.first().expect("patched model");

    assert_eq!(patched.description, None);
    assert_eq!(patched.model_messages, None);
    assert_eq!(patched.supported_reasoning_levels, Vec::new());
}

#[test]
fn inherited_overlay_models_only_receive_explicit_aliases() {
    let original = bundled_models_response().expect("bundled catalog should parse");
    let parent = original.models.first().expect("bundled model").clone();
    let inherited_slug = "external/inherited-without-aliases";
    let explicit_slug = "external/inherited-with-aliases";
    let applied = apply(json!({"models": [
        {"slug": parent.slug, "aliases": ["parent-alias"]},
        {"slug": inherited_slug, "inherits": parent.slug},
        {
            "slug": explicit_slug,
            "inherits": parent.slug,
            "aliases": ["child-alias"],
            "history_compatibility_group": "external-family",
            "requires_nonempty_assistant_messages": true
        }
    ]}))
    .expect("overlay should apply");

    let mut expected_inherited = parent.clone();
    expected_inherited.slug = inherited_slug.to_string();
    expected_inherited.aliases = Vec::new();
    expected_inherited.history_compatibility_group = None;
    let mut expected_explicit = parent;
    expected_explicit.slug = explicit_slug.to_string();
    expected_explicit.aliases = vec!["child-alias".to_string()];
    expected_explicit.history_compatibility_group = Some("external-family".to_string());
    expected_explicit.requires_nonempty_assistant_messages = true;
    assert_eq!(
        applied.models[applied.models.len() - 2..],
        [expected_inherited, expected_explicit]
    );
}

#[test]
fn reports_entry_slug_and_invalid_or_missing_fields() {
    let parent = bundled_models_response()
        .expect("bundled catalog should parse")
        .models
        .first()
        .expect("bundled model")
        .slug
        .clone();
    let cases = [
        (
            json!({"models": [{"display_name": "missing"}]}),
            "entry 0: missing field `slug`",
        ),
        (
            json!({"models": [{"slug": parent, "priority": "high"}]}),
            "entry 0 slug",
        ),
        (
            json!({"models": [{"slug": "new"}]}),
            "new model requires field `inherits`",
        ),
        (
            json!({"models": [{"slug": "new", "inherits": "missing"}]}),
            "missing parent model `missing`",
        ),
    ];
    for (value, expected) in cases {
        let err = apply(value).expect_err("overlay should fail");
        assert!(
            err.to_string().contains(expected),
            "unexpected error: {err}"
        );
    }
}

#[test]
fn rejects_duplicate_slugs_invalid_inherits_and_unknown_fields() {
    let parent = bundled_models_response()
        .expect("bundled catalog should parse")
        .models
        .first()
        .expect("bundled model")
        .slug
        .clone();
    let duplicate = json!({"models": [{"slug": parent}, {"slug": parent}]});
    let err =
        ModelCatalogOverlay::from_json(&duplicate.to_string()).expect_err("duplicates should fail");
    assert!(err.to_string().contains("entry 1 slug"));
    assert!(err.to_string().contains("duplicate slug"));

    let invalid_inherits = json!({"models": [{"slug": parent, "inherits": false}]});
    let err = ModelCatalogOverlay::from_json(&invalid_inherits.to_string())
        .expect_err("invalid inherits should fail");
    assert!(
        err.to_string()
            .contains("field `inherits` must be a non-empty string")
    );

    let unknown = json!({"models": [{
        "slug": parent,
        "misspelled_model_field": true
    }]});
    let err = ModelCatalogOverlay::from_json(&unknown.to_string())
        .expect_err("unknown fields should fail");
    assert_eq!(
        err.to_string(),
        format!("entry 0 slug `{parent}`: invalid field `misspelled_model_field`")
    );
}

#[test]
fn allows_typed_inference_and_remaining_staged_deployed_overlay_fields() {
    let parent = bundled_models_response()
        .expect("bundled catalog should parse")
        .models
        .first()
        .expect("bundled model")
        .slug
        .clone();
    let overlay = json!({"models": [{
        "slug": parent,
        "aliases": ["short-name"],
        "history_compatibility_group": "provider",
        "inference": {
            "family": "grok",
            "wire_api": "responses",
            "dialect": "grok",
            "route": "grok",
            "wire_model": "grok-4.6"
        },
        "requires_nonempty_assistant_messages": false,
        "supports_parallel_tool_calls": true,
        "disabled_tools": ["web_search", "image_generation"]
    }]});

    let applied = apply(overlay).expect("typed and staged deployed fields should remain accepted");
    assert_eq!(
        applied.models[0].disabled_tools,
        [
            ModelToolCapability::WebSearch,
            ModelToolCapability::ImageGeneration,
        ]
    );
}
