//! Tests for provider-oriented JSON Schema cleaning.

use super::*;
use serde_json::json;

#[test]
fn removes_unsupported_keywords() {
    let schema = json!({
        "type": "string",
        "minLength": 1,
        "maxLength": 100,
        "pattern": "^[a-z]+$",
        "description": "A lowercase string"
    });

    let cleaned = SchemaCleanr::clean_for_gemini(schema);

    assert_eq!(cleaned["type"], "string");
    assert_eq!(cleaned["description"], "A lowercase string");
    assert!(cleaned.get("minLength").is_none());
    assert!(cleaned.get("maxLength").is_none());
    assert!(cleaned.get("pattern").is_none());
}

#[test]
fn resolves_local_refs() {
    let schema = json!({
        "type": "object",
        "properties": {
            "age": {
                "$ref": "#/$defs/Age"
            }
        },
        "$defs": {
            "Age": {
                "type": "integer",
                "minimum": 0
            }
        }
    });

    let cleaned = SchemaCleanr::clean_for_gemini(schema);

    assert_eq!(cleaned["properties"]["age"]["type"], "integer");
    assert!(cleaned["properties"]["age"].get("minimum").is_none());
    assert!(cleaned.get("$defs").is_none());
}

#[test]
fn flattens_literal_union() {
    let schema = json!({
        "anyOf": [
            { "const": "admin", "type": "string" },
            { "const": "user", "type": "string" },
            { "const": "guest", "type": "string" }
        ]
    });

    let cleaned = SchemaCleanr::clean_for_gemini(schema);

    assert_eq!(cleaned["type"], "string");
    let enum_values = cleaned["enum"].as_array().unwrap();
    assert_eq!(enum_values.len(), 3);
    assert!(enum_values.contains(&json!("admin")));
    assert!(enum_values.contains(&json!("user")));
    assert!(enum_values.contains(&json!("guest")));
}

#[test]
fn strips_null_from_union() {
    let schema = json!({
        "oneOf": [
            { "type": "string" },
            { "type": "null" }
        ]
    });

    let cleaned = SchemaCleanr::clean_for_gemini(schema);

    assert_eq!(cleaned["type"], "string");
    assert!(cleaned.get("oneOf").is_none());
}

#[test]
fn converts_const_to_enum() {
    let schema = json!({
        "const": "fixed_value",
        "description": "A constant"
    });

    let cleaned = SchemaCleanr::clean_for_gemini(schema);

    assert_eq!(cleaned["enum"], json!(["fixed_value"]));
    assert_eq!(cleaned["description"], "A constant");
    assert!(cleaned.get("const").is_none());
}

#[test]
fn preserves_metadata_across_ref_resolution() {
    let schema = json!({
        "$ref": "#/$defs/Name",
        "description": "User's name",
        "title": "Name Field",
        "default": "Anonymous",
        "$defs": {
            "Name": {
                "type": "string"
            }
        }
    });

    let cleaned = SchemaCleanr::clean_for_gemini(schema);

    assert_eq!(cleaned["type"], "string");
    assert_eq!(cleaned["description"], "User's name");
    assert_eq!(cleaned["title"], "Name Field");
    assert_eq!(cleaned["default"], "Anonymous");
}

#[test]
fn breaks_circular_refs_without_panicking() {
    let schema = json!({
        "type": "object",
        "properties": {
            "parent": {
                "$ref": "#/$defs/Node"
            }
        },
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "child": {
                        "$ref": "#/$defs/Node"
                    }
                }
            }
        }
    });

    let cleaned = SchemaCleanr::clean_for_gemini(schema);

    assert_eq!(cleaned["properties"]["parent"]["type"], "object");
}

#[test]
fn validates_schema_shape() {
    let valid = json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" }
        }
    });

    assert!(SchemaCleanr::validate(&valid).is_ok());

    let invalid = json!({
        "properties": {
            "name": { "type": "string" }
        }
    });

    assert!(SchemaCleanr::validate(&invalid).is_err());
}

#[test]
fn strategies_preserve_expected_keywords() {
    let schema = json!({
        "type": "string",
        "minLength": 1,
        "description": "A string field"
    });

    let gemini = SchemaCleanr::clean_for_gemini(schema.clone());
    assert!(gemini.get("minLength").is_none());
    assert_eq!(gemini["type"], "string");
    assert_eq!(gemini["description"], "A string field");

    let openai = SchemaCleanr::clean_for_openai(schema.clone());
    assert_eq!(openai["minLength"], 1);
    assert_eq!(openai["type"], "string");
}

#[test]
fn cleans_nested_properties() {
    let schema = json!({
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 1
                    }
                },
                "additionalProperties": false
            }
        }
    });

    let cleaned = SchemaCleanr::clean_for_gemini(schema);

    assert!(
        cleaned["properties"]["user"]["properties"]["name"]
            .get("minLength")
            .is_none()
    );
    assert!(
        cleaned["properties"]["user"]
            .get("additionalProperties")
            .is_none()
    );
}

#[test]
fn removes_null_from_type_array() {
    let schema = json!({
        "type": ["string", "null"]
    });

    let cleaned = SchemaCleanr::clean_for_gemini(schema);

    assert_eq!(cleaned["type"], "string");
}

#[test]
fn preserves_null_in_type_array_for_openai() {
    let schema = json!({
        "type": ["string", "null"]
    });

    let cleaned = SchemaCleanr::clean_for_openai(schema);

    assert_eq!(cleaned["type"], json!(["string", "null"]));
}

#[test]
fn preserves_only_null_type_array() {
    let schema = json!({
        "type": ["null"]
    });

    let cleaned = SchemaCleanr::clean_for_gemini(schema);

    assert_eq!(cleaned["type"], "null");
}

#[test]
fn decodes_json_pointer_escapes_in_refs() {
    let schema = json!({
        "$ref": "#/$defs/Foo~1Bar",
        "$defs": {
            "Foo/Bar": {
                "type": "string"
            }
        }
    });

    let cleaned = SchemaCleanr::clean_for_gemini(schema);

    assert_eq!(cleaned["type"], "string");
}

#[test]
fn skips_type_when_non_simplifiable_union_exists() {
    let schema = json!({
        "type": "object",
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "a": { "type": "string" }
                }
            },
            {
                "type": "object",
                "properties": {
                    "b": { "type": "number" }
                }
            }
        ]
    });

    let cleaned = SchemaCleanr::clean_for_gemini(schema);

    assert!(cleaned.get("type").is_none());
    assert!(cleaned.get("oneOf").is_some());
}

#[test]
fn cleans_nested_unknown_schema_keyword() {
    let schema = json!({
        "not": {
            "$ref": "#/$defs/Age"
        },
        "$defs": {
            "Age": {
                "type": "integer",
                "minimum": 0
            }
        }
    });

    let cleaned = SchemaCleanr::clean_for_gemini(schema);

    assert_eq!(cleaned["not"]["type"], "integer");
    assert!(cleaned["not"].get("minimum").is_none());
}

#[test]
fn validate_against_schema_accepts_conforming_values() {
    let schema = json!({
        "type": "object",
        "required": ["approved"],
        "properties": { "approved": { "type": "boolean" } }
    });
    assert!(validate_against_schema(&schema, &json!({ "approved": true })).is_ok());
    // A null/empty schema accepts anything.
    assert!(validate_against_schema(&json!(null), &json!("anything")).is_ok());
    assert!(validate_against_schema(&json!({}), &json!(42)).is_ok());
}

#[test]
fn validate_against_schema_rejects_wrong_type_and_missing_required() {
    let schema = json!({
        "type": "object",
        "required": ["approved"],
        "properties": { "approved": { "type": "boolean" } }
    });
    let wrong_type = validate_against_schema(&schema, &json!({ "approved": "yes" })).unwrap_err();
    assert!(
        matches!(&wrong_type, TinyAgentsError::Validation(msg) if msg.contains("value.approved")),
        "got {wrong_type:?}"
    );
    let missing = validate_against_schema(&schema, &json!({})).unwrap_err();
    assert!(
        matches!(&missing, TinyAgentsError::Validation(msg) if msg.contains("required")),
        "got {missing:?}"
    );
}
