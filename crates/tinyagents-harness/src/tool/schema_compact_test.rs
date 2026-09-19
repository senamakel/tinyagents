use serde_json::json;
use tinyinference_llm::tool::ToolSchema;

use super::*;
use crate::tool::{SchemaPreparation, prepare_tool_schemas};

fn nested(depth: usize) -> serde_json::Value {
    if depth == 0 {
        return json!({"type": "string", "description": "leaf"});
    }
    json!({
        "type": "object",
        "description": format!("level {depth}"),
        "properties": {"child": nested(depth - 1)},
    })
}

#[test]
fn prunes_only_unreachable_definitions() {
    let schema = json!({
        "type": "object",
        "$defs": {
            "Used": {"type": "string"},
            "Chained": {"$ref": "#/$defs/Used"},
            "Dead": {"type": "integer"},
        },
        "properties": {"id": {"$ref": "#/$defs/Chained"}},
    });
    let pruned = prune_unreachable_definitions(schema);
    let defs = pruned["$defs"].as_object().unwrap();
    assert!(defs.contains_key("Used"));
    assert!(defs.contains_key("Chained"));
    assert!(!defs.contains_key("Dead"));

    let none = prune_unreachable_definitions(json!({
        "type": "object",
        "$defs": {"Dead": {"type": "integer"}},
        "properties": {"id": {"type": "string"}},
    }));
    assert!(none.get("$defs").is_none());
}

#[test]
fn strip_descriptions_respects_keep_depth() {
    let schema = json!({
        "type": "object",
        "description": "root",
        "properties": {
            "top": {
                "type": "object",
                "description": "top-level property",
                "properties": {"inner": {"type": "string", "description": "nested"}},
            }
        }
    });
    let below_top = strip_descriptions(schema.clone(), 1);
    assert_eq!(below_top["description"], "root");
    assert_eq!(
        below_top["properties"]["top"]["description"],
        "top-level property"
    );
    assert!(
        below_top["properties"]["top"]["properties"]["inner"]
            .get("description")
            .is_none()
    );

    let all = strip_descriptions(schema, 0);
    assert_eq!(all["description"], "root");
    assert!(all["properties"]["top"].get("description").is_none());
}

#[test]
fn drop_definitions_opens_surviving_refs() {
    let schema = json!({
        "type": "object",
        "$defs": {"X": {"type": "string"}},
        "properties": {"x": {"$ref": "#/$defs/X"}},
    });
    let dropped = drop_definitions(schema);
    assert!(dropped.get("$defs").is_none());
    assert_eq!(dropped["properties"]["x"], json!({"type": "object"}));
}

#[test]
fn collapse_deep_objects_keeps_top_levels() {
    let collapsed = collapse_deep_objects(nested(5), 0);
    // depth 0..COLLAPSE_DEPTH-1 survive, the object at COLLAPSE_DEPTH collapses.
    let mut cursor = &collapsed;
    for _ in 0..COLLAPSE_DEPTH {
        assert!(cursor["properties"].is_object());
        cursor = &cursor["properties"]["child"];
    }
    assert_eq!(cursor["type"], "object");
    assert!(cursor.get("properties").is_none());
}

#[test]
fn drop_compositions_leaves_an_open_object() {
    let dropped = drop_compositions(json!({
        "type": "object",
        "properties": {"v": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
    }));
    assert_eq!(dropped["properties"]["v"], json!({"type": "object"}));
}

#[test]
fn compact_parameters_stops_at_the_first_fitting_rung() {
    let schema = json!({
        "type": "object",
        "$defs": {"Dead": {"type": "integer", "description": "x".repeat(200)}},
        "properties": {"a": {"type": "string", "description": "keep me"}},
    });
    let compacted = compact_parameters(schema, 200);
    // Rung 1 (pruning the dead definition) was enough; descriptions survive.
    assert!(compacted.get("$defs").is_none());
    assert_eq!(compacted["properties"]["a"]["description"], "keep me");

    let untouched = compact_parameters(json!({"type": "object", "properties": {}}), 10_000);
    assert_eq!(untouched, json!({"type": "object", "properties": {}}));
}

#[test]
fn compact_parameters_walks_the_whole_ladder_for_a_huge_schema() {
    let schema = json!({
        "type": "object",
        "properties": {"deep": nested(8), "u": {"oneOf": [nested(2), {"type": "string"}]}},
    });
    let before = serde_json::to_vec(&schema).unwrap().len();
    let compacted = compact_parameters(schema, 60);
    let after = serde_json::to_vec(&compacted).unwrap().len();
    assert!(after < before);
    // The top-level argument surface survives every rung.
    assert!(compacted["properties"]["deep"].is_object());
    assert!(compacted["properties"]["u"].is_object());
}

#[test]
fn compact_tool_schema_caps_description_on_a_char_boundary() {
    let schema = ToolSchema::new("t", "héllo wörld, this is long", json!({"type": "object"}));
    let compaction = SchemaCompaction {
        max_description_bytes: Some(8),
        max_schema_bytes: None,
    };
    let compacted = compact_tool_schema(&schema, &compaction);
    assert!(compacted.description.len() <= 8);
    assert!(compacted.description.ends_with('…'));
    assert_eq!(compacted.name, "t");

    let short = compact_tool_schema(&ToolSchema::new("t", "ok", json!({})), &compaction);
    assert_eq!(short.description, "ok");
}

#[test]
fn preparation_applies_compaction_after_cleaning() {
    let declared = vec![ToolSchema::new(
        "lookup",
        "d".repeat(50),
        json!({
            "type": "object",
            "$defs": {"Id": {"type": "string"}, "Dead": {"type": "string"}},
            "properties": {"id": {"$ref": "#/$defs/Id"}},
        }),
    )];
    let preparation = SchemaPreparation::openai().with_compaction(SchemaCompaction {
        max_schema_bytes: Some(60),
        max_description_bytes: Some(10),
    });
    let wire = prepare_tool_schemas(&declared, &preparation);
    assert_eq!(wire[0].parameters["properties"]["id"]["type"], "string");
    assert!(wire[0].parameters.get("$defs").is_none());
    assert!(wire[0].description.len() <= 10);
}
