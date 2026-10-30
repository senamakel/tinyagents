//! Persisted snapshot shape is a wire/storage contract: pin it as literal JSON.

use serde_json::json;

use super::types::{TurnLifecycle, TurnState};

#[test]
fn started_snapshot_serializes_to_the_documented_camel_case_shape() {
    let mut state = TurnState::started("t1", "r1", 25, "2026-05-04T10:00:00Z");
    state.lifecycle = TurnLifecycle::Streaming;
    assert_eq!(
        serde_json::to_value(&state).unwrap(),
        json!({
            "threadId": "t1",
            "requestId": "r1",
            "lifecycle": "streaming",
            "iteration": 0,
            "maxIterations": 25,
            "streamingText": "",
            "thinking": "",
            "toolTimeline": [],
            "startedAt": "2026-05-04T10:00:00Z",
            "updatedAt": "2026-05-04T10:00:00Z"
        })
    );
}

#[test]
fn snapshot_written_before_optional_fields_still_loads() {
    let raw = json!({
        "threadId": "t1", "requestId": "r1", "lifecycle": "interrupted",
        "iteration": 2, "maxIterations": 10, "streamingText": "x", "thinking": "",
        "toolTimeline": [], "startedAt": "a", "updatedAt": "b"
    });
    let state: TurnState = serde_json::from_value(raw).unwrap();
    assert_eq!(state.lifecycle, TurnLifecycle::Interrupted);
    assert!(state.transcript.is_empty());
}

#[test]
fn turn_state_json_schema_tracks_the_serialized_wire_names() {
    let schema = serde_json::to_value(
        schemars::generate::SchemaSettings::draft2020_12()
            .for_serialize()
            .into_generator()
            .into_root_schema_for::<TurnState>(),
    )
    .unwrap();
    let properties = schema["properties"].as_object().unwrap();
    for field in [
        "threadId",
        "requestId",
        "maxIterations",
        "toolTimeline",
        "transcript",
        "startedAt",
        "updatedAt",
    ] {
        assert!(properties.contains_key(field), "schema is missing {field}");
    }
    let required = schema["required"].as_array().unwrap();
    for field in ["threadId", "streamingText", "thinking", "toolTimeline"] {
        assert!(required.iter().any(|required| required == field));
    }
    assert!(!required.iter().any(|required| required == "transcript"));
    let subagent = &schema["$defs"]["SubagentActivity"];
    let subagent_required = subagent["required"].as_array().unwrap();
    assert!(subagent_required.iter().any(|field| field == "toolCalls"));
    assert!(!subagent_required.iter().any(|field| field == "transcript"));
}
