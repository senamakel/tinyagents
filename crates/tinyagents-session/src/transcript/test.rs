//! Module-local unit tests for [`crate::transcript`]: full-rewrite and
//! append-only writers, model-context and display readers, compaction replay,
//! interrupted partials, path resolution/resume, thread-usage summaries, and
//! the [`super::history`] locator/handle seam.
//!
//! Consolidated here per AGENTS.md: one `test.rs` per module directory.

use super::*;
use tempfile::tempdir;

fn meta() -> TranscriptMeta {
    TranscriptMeta { session_id: None, parent_session_id: None,
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
    assert_eq!(loaded.messages[0].role, messages[0].role);
    assert_eq!(loaded.messages[0].content, messages[0].content);
    assert!(loaded.messages[0].preserve_request_id);
    let restored_usage = loaded.messages[1].turn_usage.as_ref().unwrap();
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
        vec![TranscriptMessage {
            preserve_request_id: true,
            ..TranscriptMessage::new("user", "hello")
        }]
    );
}

#[test]
fn malformed_required_message_record_is_skipped_without_losing_valid_rows() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "session").unwrap();
    let header = serde_json::json!({"_meta": {"agent": "agent", "dispatcher": "native", "created": "", "updated": "", "turn_count": 0, "input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "charged_amount_usd": 0.0}});
    std::fs::write(&path, format!("{header}\n{{\"role\":\"user\"}}\n")).unwrap();
    let loaded = read_transcript(&path).unwrap();
    assert!(loaded.messages.is_empty());
}

#[test]
fn append_replays_delta_compaction_request_ids_and_interrupted_partials() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "session").unwrap();
    let first = vec![
        TranscriptMessage::new("system", "system"),
        TranscriptMessage::new("user", "first"),
        TranscriptMessage::assistant("answer"),
    ];
    append_transcript_turn(&path, &[], &first, &meta(), None, Some("request-a")).unwrap();
    let extended = [
        first.clone(),
        vec![TranscriptMessage::new("user", "second")],
    ]
    .concat();
    append_transcript_turn(&path, &first, &extended, &meta(), None, Some("request-b")).unwrap();
    let compacted = vec![
        TranscriptMessage::new("system", "system"),
        TranscriptMessage::assistant("summary"),
        TranscriptMessage::new("user", "second"),
    ];
    append_transcript_turn(
        &path,
        &extended,
        &compacted,
        &meta(),
        None,
        Some("request-c"),
    )
    .unwrap();
    append_interrupted_partial(
        &path,
        "partial answer",
        Some("request-c"),
        Some(1),
        Some("partial reasoning"),
    )
    .unwrap();

    let replayed = read_transcript(&path).unwrap();
    assert_eq!(
        replayed
            .messages
            .iter()
            .map(|message| (&message.role, &message.content))
            .collect::<Vec<_>>(),
        compacted
            .iter()
            .map(|message| (&message.role, &message.content))
            .collect::<Vec<_>>()
    );
    let display = read_transcript_display(&path).unwrap();
    assert!(
        display
            .records
            .iter()
            .any(|record| matches!(record, DisplayRecord::Compaction(_)))
    );
    assert!(
        display
            .records
            .iter()
            .any(|record| matches!(record, DisplayRecord::Message(message) if message.interrupted))
    );
    let raw = std::fs::read_to_string(path).unwrap();
    assert!(raw.contains("request-a"));
    assert!(raw.contains("request-b"));
    assert!(raw.contains("request-c"));
}

#[test]
fn file_history_never_converts_or_drops_durable_fields() {
    let dir = tempdir().unwrap();
    let history = FileTranscriptHistory::new(dir.path(), "session", meta()).unwrap();
    let assistant = TranscriptMessage {
        id: Some("assistant-id".into()),
        role: "assistant".into(),
        content: r#"{"tool_calls":[{"id":"call-1"}]}"#.into(),
        extra_metadata: Some(serde_json::json!({
            "trusted_verbatim": true,
            "artifacts": ["artifact-1"],
            "provider_extension": {"opaque": [1, 2]}
        })),
        cache_breakpoints: vec![4],
        turn_usage: None,
        request_id: None,
        preserve_request_id: false,
        interrupted: false,
        tool_failure: None,
    };
    TranscriptHistory::append(&history, assistant.clone()).unwrap();
    let replayed = TranscriptHistory::messages(&history).unwrap();
    assert_eq!(replayed.len(), 1);
    assert_eq!(replayed[0].id, assistant.id);
    assert_eq!(replayed[0].role, assistant.role);
    assert_eq!(replayed[0].content, assistant.content);
    assert_eq!(replayed[0].extra_metadata, assistant.extra_metadata);
    assert_eq!(replayed[0].cache_breakpoints, vec![4]);
    TranscriptHistory::replace(&history, std::slice::from_ref(&assistant)).unwrap();
    TranscriptHistory::clear(&history).unwrap();
    assert!(TranscriptHistory::messages(&history).unwrap().is_empty());
    let display = read_transcript_display(history.path()).unwrap();
    assert!(
        display
            .records
            .iter()
            .any(|record| matches!(record, DisplayRecord::Compaction(_)))
    );
    let raw = std::fs::read_to_string(history.path()).unwrap();
    assert!(raw.contains("trusted_verbatim"));
    assert!(raw.contains("artifact-1"));
    assert!(raw.contains("\"cache_breakpoints\":[4]"));
}

#[test]
fn discovery_and_legacy_read_replay_the_canonical_format() {
    let dir = tempdir().unwrap();
    let root = resolve_keyed_transcript_path(dir.path(), "1714000000_agent").unwrap();
    let mut root_meta = meta();
    root_meta.thread_id = Some("thread-1".into());
    write_transcript(
        &root,
        &[TranscriptMessage::new("user", "hello")],
        &root_meta,
        None,
    )
    .unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    let found = locator.root_for_thread("thread-1").unwrap();
    assert_eq!(
        found.read_session().unwrap().unwrap().messages[0].content,
        "hello"
    );

    let legacy = dir.path().join("legacy.md");
    std::fs::write(&legacy, "<!-- session_transcript\nagent: agent\ndispatcher: native\n-->\n<!--MSG role=\"user\"-->\nlegacy body\n<!--/MSG-->\n").unwrap();
    assert_eq!(
        read_transcript_legacy_md(&legacy).unwrap().messages[0].content,
        "legacy body"
    );
}
