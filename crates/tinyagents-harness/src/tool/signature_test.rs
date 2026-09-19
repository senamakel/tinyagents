use serde_json::json;

use super::*;

#[test]
fn renders_flat_objects_with_optional_markers() {
    let schema = json!({
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Where to read. Must exist."},
            "limit": {"type": "integer"},
            "mode": {"type": "string", "enum": ["fast", "full"]},
        },
        "required": ["path"]
    });
    assert_eq!(
        type_signature(&schema),
        "{limit?: integer, mode?: \"fast\" | \"full\", path: string}"
    );
    assert_eq!(argument_notes(&schema), vec!["path: Where to read"]);
}

#[test]
fn renders_arrays_unions_and_nesting() {
    let schema = json!({
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"type": "string"}},
            "rows": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}},
            "v": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "t": {"type": ["number", "null"]},
        },
        "required": ["tags", "rows", "v", "t"]
    });
    assert_eq!(
        type_signature(&schema),
        "{rows: Array<{id: integer}>, t: number | null, tags: string[], v: string | null}"
    );
}

#[test]
fn collapses_past_max_depth_and_clips_long_signatures() {
    fn nested(depth: usize) -> serde_json::Value {
        if depth == 0 {
            return json!({"type": "string"});
        }
        json!({"type": "object", "properties": {"c": nested(depth - 1)}, "required": ["c"]})
    }
    let rendered = type_signature(&nested(8));
    assert!(rendered.contains("object"));
    assert!(rendered.matches('{').count() <= MAX_DEPTH);

    let mut properties = serde_json::Map::new();
    for index in 0..60 {
        properties.insert(format!("property_number_{index}"), json!({"type": "string"}));
    }
    let wide = json!({"type": "object", "properties": properties});
    let rendered = type_signature(&wide);
    assert!(rendered.chars().count() <= MAX_SIGNATURE_CHARS + 1);
    assert!(rendered.ends_with('…'));
}

#[test]
fn handles_degenerate_schemas() {
    assert_eq!(type_signature(&json!({"type": "object"})), "object");
    assert_eq!(type_signature(&json!({"type": "object", "properties": {}})), "{}");
    assert_eq!(type_signature(&json!(null)), "unknown");
    assert_eq!(type_signature(&json!({})), "unknown");
    assert!(argument_notes(&json!({"type": "object"})).is_empty());
}
