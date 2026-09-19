use super::*;
use tempfile::tempdir;

fn meta() -> TranscriptMeta {
    TranscriptMeta {
        agent_name: "agent".into(),
        agent_id: Some("agent-id".into()),
        agent_type: Some("root".into()),
        dispatcher: "native".into(),
        provider: Some("provider".into()),
        model: Some("model".into()),
        created: "2026-01-01T00:00:00Z".into(),
        updated: "2026-01-01T00:00:00Z".into(),
        turn_count: 1,
        input_tokens: 1,
        output_tokens: 2,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: None,
        task_id: None,
    }
}

#[test]
fn jsonl_round_trip_keeps_raw_tool_arguments_and_provider_extension() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "session").unwrap();
    let messages = vec![
        TranscriptMessage::new("user", "use a tool"),
        TranscriptMessage::assistant("working"),
    ];
    let usage = TurnUsage {
        provider: "provider".into(),
        model: "model".into(),
        usage: MessageUsage {
            input: 1,
            output: 2,
            cached_input: 0,
            context_window: 3,
            cost_usd: 0.0,
        },
        ts: "2026-01-01T00:00:00Z".into(),
        reasoning_content: Some("reasoning".into()),
        tool_calls: vec![TranscriptToolCall {
            id: "call-1".into(),
            name: "tool".into(),
            arguments: "{not-json}".into(),
            extra_content: Some(serde_json::json!({"google": {"thought_signature": "opaque"}})),
        }],
        iteration: 1,
    };

    write_transcript(&path, &messages, &meta(), Some(&usage)).unwrap();
    let loaded = read_transcript(&path).unwrap();
    assert_eq!(loaded.messages[0], messages[0]);
    let restored_usage = super::metadata::turn_usage_from_metadata(&loaded.messages[1]).unwrap();
    assert_eq!(restored_usage.tool_calls[0].arguments, "{not-json}");
    assert_eq!(
        restored_usage.tool_calls[0].extra_content,
        Some(serde_json::json!({"google": {"thought_signature": "opaque"}}))
    );
    let raw = std::fs::read_to_string(path).unwrap();
    assert!(raw.contains("{not-json}"));
    assert!(raw.contains("thought_signature"));
}

#[test]
fn unknown_jsonl_records_are_ignored() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "session").unwrap();
    let header = serde_json::json!({"_meta": {"agent": "agent", "dispatcher": "native", "created": "", "updated": "", "turn_count": 0, "input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "charged_amount_usd": 0.0}});
    std::fs::write(&path, format!("{header}\n{{\"kind\":\"future\",\"payload\":true}}\n{{\"role\":\"user\",\"content\":\"hello\"}}\n")).unwrap();
    let loaded = read_transcript(&path).unwrap();
    assert_eq!(
        loaded.messages,
        vec![TranscriptMessage::new("user", "hello")]
    );
}
