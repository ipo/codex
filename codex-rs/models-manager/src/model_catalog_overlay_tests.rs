use super::*;
use crate::bundled_models_response;
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
        "model_messages": {"instructions_template": null, "instructions_variables": null, "approvals": null, "auto_review": null, "permissions": null},
        "supported_reasoning_levels": []
    }]}))
    .expect("overlay should apply");
    let patched = applied.models.first().expect("patched model");

    assert_eq!(patched.description, None);
    assert_eq!(patched.supported_reasoning_levels, Vec::new());
    assert_eq!(
        patched
            .model_messages
            .as_ref()
            .and_then(|messages| messages.instructions_template.clone()),
        None
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
fn rejects_duplicate_slugs_and_unknown_fields() {
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

    let unknown = json!({"models": [{"slug": parent, "not_a_model_field": true}]});
    let err = ModelCatalogOverlay::from_json(&unknown.to_string())
        .expect_err("unknown fields should fail");
    assert!(
        err.to_string()
            .contains("invalid field `not_a_model_field`")
    );
}
