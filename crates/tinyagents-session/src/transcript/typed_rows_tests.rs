//! Typed rows: tool calls, tool results and inline images are fields of the
//! in-memory row and of the stored line, and `content` is plain text. Legacy
//! rows (a native envelope or `[OH_IMAGE:]` marker inside `content`, literal
//! fixtures below) keep loading, normalized into the same typed fields, and a
//! file may hold both generations. [`TranscriptMessage::legacy_content`]
//! rebuilds the old string for compatibility adapters.

use super::*;
use std::fs;
use tempfile::tempdir;
use tinytools_agent::dialect::{NativeToolCall, encode_assistant_envelope, encode_tool_envelope};

fn meta() -> TranscriptMeta {
    TranscriptMeta {
        session_id: None,
        parent_session_id: None,
        agent_name: "agent".into(),
        agent_id: Some("agent-id".into()),
        agent_type: Some("root".into()),
        dispatcher: "native".into(),
        provider: Some("provider".into()),
        model: Some("model".into()),
        created: "2026-01-01T00:00:00Z".into(),
        updated: "2026-01-01T00:00:00Z".into(),
        turn_count: 1,
        prefix_message_count: None,
        prefix_reset: false,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: None,
        task_id: None,
    }
}

fn call(id: &str, extra: Option<serde_json::Value>) -> NativeToolCall {
    NativeToolCall {
        id: id.into(),
        name: "shell".into(),
        arguments: r#"{"command":"ls"}"#.into(),
        extra_content: extra,
    }
}

fn message_lines(path: &std::path::Path) -> Vec<serde_json::Value> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|value| value.get("role").is_some())
        .collect()
}

/// The same turn as typed rows.
fn typed_turn() -> Vec<TranscriptMessage> {
    let image = "data:image/png;base64,AAAA";
    vec![
        TranscriptMessage::user_with_parts(vec![
            TranscriptPart::Text {
                text: "look ".into(),
            },
            TranscriptPart::Image { url: image.into() },
            TranscriptPart::Text {
                text: " please".into(),
            },
        ]),
        TranscriptMessage::assistant_with_calls(
            "on it",
            vec![
                TranscriptToolCall::from(call(
                    "c1",
                    Some(serde_json::json!({"google": {"thought_signature": "sig"}})),
                )),
                TranscriptToolCall::from(call("c2", None)),
            ],
        ),
        TranscriptMessage::tool_result("c1", "out \"1\"").with_id("c1"),
        TranscriptMessage::tool_result("c2", "out 2").with_id("c2"),
        TranscriptMessage::assistant("done"),
    ]
}

/// The same turn as the legacy string rows an older host built.
fn legacy_turn() -> Vec<TranscriptMessage> {
    let image = "[OH_IMAGE:data:image/png;base64,AAAA]";
    vec![
        TranscriptMessage::user(format!("look {image} please")),
        TranscriptMessage::assistant(encode_assistant_envelope(
            Some("on it"),
            &[
                call(
                    "c1",
                    Some(serde_json::json!({"google": {"thought_signature": "sig"}})),
                ),
                call("c2", None),
            ],
            None,
        )),
        TranscriptMessage::tool(encode_tool_envelope("c1", "out \"1\"")).with_id("c1"),
        TranscriptMessage::tool(encode_tool_envelope("c2", "out 2")).with_id("c2"),
        TranscriptMessage::assistant("done"),
    ]
}

#[test]
fn typed_rows_store_fields_and_read_back_equal() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "typed").unwrap();
    let rows = typed_turn();
    write_transcript(&path, &rows, &meta(), None).unwrap();

    let lines = message_lines(&path);
    assert_eq!(lines[0]["v"], 2);
    assert_eq!(lines[0]["shape"], "user_parts");
    assert_eq!(lines[0]["content"], "look  please");
    assert_eq!(lines[0]["parts"][1]["type"], "image");
    assert_eq!(lines[1]["shape"], "assistant_calls");
    assert_eq!(lines[1]["content"], "on it");
    assert_eq!(lines[1]["tool_calls"][1]["id"], "c2");
    assert_eq!(lines[2]["shape"], "tool_result");
    assert_eq!(lines[2]["tool_call_id"], "c1");
    assert_eq!(lines[2]["content"], "out \"1\"");
    // A plain assistant answer is not a typed row.
    assert!(lines[4].get("v").is_none() && lines[4].get("shape").is_none());
    // No envelope string is stored anywhere in a typed row.
    for line in &lines[..4] {
        assert!(
            !line["content"].as_str().unwrap().starts_with('{'),
            "{line}"
        );
    }

    let read = read_transcript(&path).unwrap();
    assert_eq!(read.messages.len(), rows.len());
    for (read, written) in read.messages.iter().zip(&rows) {
        assert!(read.same_row_as(written), "{read:?} vs {written:?}");
    }
    let display = read_transcript_display(&path).unwrap();
    let displayed: Vec<_> = display
        .records
        .iter()
        .filter_map(|record| match record {
            DisplayRecord::Message(message) => Some(message.message.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(displayed.len(), rows.len());
    for (read, written) in displayed.iter().zip(&rows) {
        assert!(read.same_row_as(written));
    }
}

/// A host that still builds string rows gets the same typed lines, and the
/// rebuilt legacy string of a typed row is exactly the old string.
#[test]
fn string_rows_and_typed_rows_write_identical_lines() {
    let dir = tempdir().unwrap();
    let typed_path = resolve_keyed_transcript_path(dir.path(), "typed").unwrap();
    let legacy_path = resolve_keyed_transcript_path(dir.path(), "legacy").unwrap();
    write_transcript(&typed_path, &typed_turn(), &meta(), None).unwrap();
    write_transcript(&legacy_path, &legacy_turn(), &meta(), None).unwrap();
    assert_eq!(message_lines(&typed_path), message_lines(&legacy_path));
    for (typed, legacy) in typed_turn().iter().zip(legacy_turn()) {
        assert_eq!(typed.legacy_content(), legacy.content);
        assert!(
            TranscriptMessage::from_legacy(&legacy.role, legacy.content).same_row_as(
                &TranscriptMessage {
                    id: None,
                    ..typed.clone()
                }
            )
        );
    }
}

#[test]
fn non_canonical_strings_stay_legacy_lines_and_load_normalized() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "opaque").unwrap();
    let rows = vec![
        TranscriptMessage::user("hi"),
        // null content, embedded reasoning, extra key, plain JSON prose
        TranscriptMessage::assistant(encode_assistant_envelope(None, &[call("c1", None)], None)),
        TranscriptMessage::assistant(encode_assistant_envelope(
            Some("x"),
            &[call("c2", None)],
            Some("why"),
        )),
        TranscriptMessage::assistant(r#"{"answer":42}"#),
        TranscriptMessage::tool(r#"{"tool_call_id":"c1","content":"a","name":"n"}"#),
        TranscriptMessage::tool("bare result"),
        TranscriptMessage::user("no image marker, [OH_IMAGE:unterminated"),
    ];
    write_transcript(&path, &rows, &meta(), None).unwrap();
    let lines = message_lines(&path);
    for (line, row) in lines.iter().zip(&rows) {
        assert!(line.get("v").is_none(), "{line}");
        // Stored exactly as given.
        assert_eq!(line["content"], row.content.as_str());
    }
    let read = read_transcript(&path).unwrap();
    for (read, written) in read.messages.iter().zip(&rows) {
        assert!(
            read.same_row_as(&TranscriptMessage::from_legacy(
                &written.role,
                written.content.clone()
            )),
            "{read:?}"
        );
    }
    // The null-content envelope lifts into a calls row with empty text; the
    // one with reasoning keeps its calls (reasoning lives in metadata); plain
    // JSON prose and a bare result stay text.
    assert_eq!(read.messages[1].tool_calls.len(), 1);
    assert_eq!(read.messages[1].content, "");
    assert_eq!(read.messages[2].tool_calls[0].id, "c2");
    assert!(read.messages[3].tool_calls.is_empty());
    assert_eq!(read.messages[4].tool_call_id.as_deref(), Some("c1"));
    assert_eq!(read.messages[5].tool_call_id, None);
    assert!(read.messages[6].parts.is_none());
}

/// Rows exactly as the pre-typed writer produced them (envelope in `content`).
const LEGACY_FILE: &str = concat!(
    r#"{"_meta":{"agent":"agent","dispatcher":"native","created":"2026-01-01T00:00:00Z","updated":"2026-01-01T00:00:00Z","turn_count":1,"input_tokens":0,"output_tokens":0,"cached_input_tokens":0,"charged_amount_usd":0.0}}"#,
    "\n",
    r#"{"role":"user","content":"run it"}"#,
    "\n",
    r#"{"role":"assistant","content":"{\"content\":\"on it\",\"tool_calls\":[{\"arguments\":\"{}\",\"id\":\"c1\",\"name\":\"shell\"}]}","tool_calls":[{"id":"c1","name":"shell","arguments":"{}"}]}"#,
    "\n",
    r#"{"id":"c1","role":"tool","content":"{\"content\":\"ok\",\"tool_call_id\":\"c1\"}"}"#,
    "\n",
);

#[test]
fn legacy_envelope_rows_still_load_and_a_new_turn_appends_typed_rows() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "mixed").unwrap();
    fs::write(&path, LEGACY_FILE).unwrap();

    let before = read_transcript(&path).unwrap();
    assert_eq!(before.messages.len(), 3);
    assert_eq!(before.messages[1].content, "on it");
    assert_eq!(before.messages[1].tool_calls.len(), 1);
    assert_eq!(
        before.messages[1].legacy_content(),
        r#"{"content":"on it","tool_calls":[{"arguments":"{}","id":"c1","name":"shell"}]}"#
    );
    assert_eq!(before.messages[2].content, "ok");
    assert_eq!(before.messages[2].tool_call_id.as_deref(), Some("c1"));
    assert_eq!(
        before.messages[2].legacy_content(),
        r#"{"content":"ok","tool_call_id":"c1"}"#
    );

    // Continue the old session: the existing bytes are never rewritten.
    let prior = before.messages.clone();
    let mut next = prior.clone();
    next.push(TranscriptMessage::user("again"));
    next.push(TranscriptMessage::assistant_with_calls(
        "sure",
        vec![TranscriptToolCall::from(call("c9", None))],
    ));
    next.push(TranscriptMessage::tool_result("c9", "done").with_id("c9"));
    append_transcript_turn(&path, &prior, &next, &meta(), None, Some("req-2")).unwrap();

    let file = fs::read_to_string(&path).unwrap();
    assert!(
        file.starts_with(LEGACY_FILE),
        "existing bytes must be untouched"
    );
    let lines = message_lines(&path);
    assert_eq!(lines.len(), 6);
    assert!(lines[..3].iter().all(|l| l.get("v").is_none()));
    assert_eq!(lines[4]["shape"], "assistant_calls");
    assert_eq!(lines[5]["shape"], "tool_result");

    let after = read_transcript(&path).unwrap();
    assert_eq!(after.messages.len(), 6);
    assert_eq!(&after.messages[..3], &before.messages[..3]);
    assert!(after.messages[4].same_row_as(&next[4]));
    assert!(after.messages[5].same_row_as(&next[5]));
}

#[test]
fn an_unknown_shape_or_missing_fields_keep_the_stored_content() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "future").unwrap();
    let body = concat!(
        r#"{"_meta":{"agent":"a","dispatcher":"native","created":"c","updated":"u","turn_count":0,"input_tokens":0,"output_tokens":0,"cached_input_tokens":0,"charged_amount_usd":0.0}}"#,
        "\n",
        r#"{"v":3,"shape":"hologram","role":"assistant","content":"kept"}"#,
        "\n",
        r#"{"v":2,"shape":"assistant_calls","role":"assistant","content":"no calls"}"#,
        "\n",
    );
    fs::write(&path, body).unwrap();
    let read = read_transcript(&path).unwrap();
    assert_eq!(read.messages[0].content, "kept");
    assert_eq!(read.messages[1].content, "no calls");
}

#[test]
fn a_future_version_or_part_kind_keeps_the_stored_content() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "future-parts").unwrap();
    let body = concat!(
        r#"{"_meta":{"agent":"a","dispatcher":"native","created":"c","updated":"u","turn_count":0,"input_tokens":0,"output_tokens":0,"cached_input_tokens":0,"charged_amount_usd":0.0}}"#,
        "\n",
        // A later writer reusing a known shape name under a newer version.
        r#"{"v":3,"shape":"tool_result","tool_call_id":"c1","role":"tool","content":"stored-v3"}"#,
        "\n",
        // The current version carrying a part kind this reader does not know.
        r#"{"v":2,"shape":"user_parts","role":"user","content":"stored-parts","parts":[{"kind":"document","uri":"x"}]}"#,
        "\n",
    );
    fs::write(&path, body).unwrap();
    let read = read_transcript(&path).unwrap();
    assert_eq!(read.messages.len(), 2);
    assert_eq!(read.messages[0].content, "stored-v3");
    assert_eq!(read.messages[1].content, "stored-parts");
}

#[test]
fn typed_rows_project_to_the_same_display_items_as_their_legacy_strings() {
    use super::view::DisplayItem;
    use super::view::project_records;

    let dir = tempdir().unwrap();
    let project = |name: &str, rows: &[TranscriptMessage]| {
        let path = resolve_keyed_transcript_path(dir.path(), name).unwrap();
        write_transcript(&path, rows, &meta(), None).unwrap();
        project_records(&read_transcript_display(&path).unwrap().records)
    };
    let typed = project("typed", &typed_turn());
    let legacy = project("legacy", &legacy_turn());
    assert_eq!(typed, legacy);
    assert!(
        typed
            .iter()
            .any(|item| matches!(item, DisplayItem::ToolCall { call_id, .. } if call_id == "c2"))
    );
    // The user's image renders in place as the host-visible marker.
    assert!(typed.iter().any(|item| matches!(
        item,
        DisplayItem::UserMessage { content, .. }
            if content == "look [IMAGE:data:image/png;base64,AAAA] please"
    )));
}

#[test]
fn display_and_legacy_text_of_typed_rows() {
    let rows = typed_turn();
    assert_eq!(
        rows[0].display_content(),
        "look [IMAGE:data:image/png;base64,AAAA] please"
    );
    assert_eq!(
        rows[0].legacy_content(),
        "look [OH_IMAGE:data:image/png;base64,AAAA] please"
    );
    assert_eq!(rows[1].display_content(), "on it");
    assert!(rows[1].legacy_content().starts_with('{'));
    assert_eq!(rows[4].legacy_content(), "done");
    // Normalizing a row that is already typed changes nothing.
    for row in &rows {
        assert_eq!(&row.clone().normalized(), row);
    }
    // A parts list without an image is just text.
    let text_only = TranscriptMessage::user_with_parts(vec![TranscriptPart::Text {
        text: "hello".into(),
    }]);
    assert!(text_only.parts.is_none());
    assert_eq!(text_only.content, "hello");
}

#[test]
fn non_canonical_legacy_rows_survive_a_read_then_rewrite_byte_for_byte() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "rewrite").unwrap();
    // Envelope with embedded reasoning, and a tool result with an extra key:
    // neither is canonical, so a typed v2 rewrite would drop those fields.
    let assistant = encode_assistant_envelope(Some("x"), &[call("c1", None)], Some("why"));
    let tool = r#"{"tool_call_id":"c1","content":"a","name":"n"}"#.to_string();
    let rows = vec![
        TranscriptMessage::user("go"),
        TranscriptMessage::assistant(assistant.clone()),
        TranscriptMessage::tool(tool.clone()).with_id("c1"),
    ];
    write_transcript(&path, &rows, &meta(), None).unwrap();

    let read = read_transcript(&path).unwrap();
    assert!(read.messages[1].is_typed(), "lifted on read");
    let rewritten = resolve_keyed_transcript_path(dir.path(), "rewrite-2").unwrap();
    write_transcript(&rewritten, &read.messages, &meta(), None).unwrap();

    let lines = message_lines(&rewritten);
    assert_eq!(lines[1]["content"], assistant.as_str());
    assert_eq!(lines[2]["content"], tool.as_str());
    assert!(lines[1].get("v").is_none() && lines[2].get("v").is_none());

    // An edited lifted row is stored typed, from its current fields.
    let mut edited = read.messages.clone();
    edited[2].content = "changed".into();
    write_transcript(&rewritten, &edited, &meta(), None).unwrap();
    let lines = message_lines(&rewritten);
    assert_eq!(lines[2]["shape"], "tool_result");
    assert_eq!(lines[2]["content"], "changed");
}

#[test]
fn legacy_and_lifted_rows_are_the_same_row() {
    let legacy = TranscriptMessage::assistant(encode_assistant_envelope(
        Some("on it"),
        &[call("c1", None)],
        None,
    ));
    let lifted = legacy.clone().normalized();
    assert!(legacy.same_row_as(&lifted));
    assert!(lifted.same_row_as(&legacy));
    assert!(!legacy.same_row_as(&TranscriptMessage::assistant("on it")));
}

#[test]
fn the_in_memory_history_returns_lifted_rows_like_the_file_backend() {
    use crate::testkit::InMemoryTranscriptHistory;
    let envelope = encode_tool_envelope("c1", "ok");
    let history = InMemoryTranscriptHistory::new("lifted", meta());
    history
        .append(TranscriptMessage::tool(envelope.clone()))
        .unwrap();
    history
        .replace(&[TranscriptMessage::tool(envelope.clone())])
        .unwrap();
    let rows = history.messages().unwrap();
    assert_eq!(rows[0].content, "ok");
    assert_eq!(rows[0].tool_call_id.as_deref(), Some("c1"));
    assert_eq!(rows[0].legacy_content(), envelope);
}

#[test]
fn legacy_content_keeps_the_string_form_of_typed_rows() {
    let typed = TranscriptMessage::assistant_with_calls(
        "on it",
        vec![TranscriptToolCall::from(call("c1", None))],
    );
    let legacy = typed.legacy_content();
    // The legacy string reads back as the original typed row.
    let back = TranscriptMessage::from_legacy("assistant", legacy);
    assert!(back.same_row_as(&typed));
    let image = TranscriptMessage::user_with_parts(vec![
        TranscriptPart::Text {
            text: "see ".into(),
        },
        TranscriptPart::Image {
            url: "data:image/png;base64,AA".into(),
        },
    ]);
    let legacy = image.legacy_content();
    assert!(legacy.contains("[OH_IMAGE:"));
    assert!(TranscriptMessage::from_legacy("user", legacy).same_row_as(&image));
}

#[test]
fn a_lifted_rows_original_string_survives_a_serde_round_trip() {
    let envelope = encode_assistant_envelope(Some("x"), &[call("c1", None)], Some("why"));
    let lifted = TranscriptMessage::from_legacy("assistant", envelope.clone());
    assert!(lifted.is_typed());
    let json = serde_json::to_string(&lifted).unwrap();
    let back: TranscriptMessage = serde_json::from_str(&json).unwrap();
    assert_eq!(back.legacy_content(), envelope);
    // A row that was never lifted serializes without the memo.
    let plain = serde_json::to_value(TranscriptMessage::user("hi")).unwrap();
    assert!(plain.get("legacy").is_none());
}

#[test]
fn an_edited_lifted_row_does_not_serialize_its_superseded_text() {
    let envelope = encode_assistant_envelope(Some("secret"), &[call("c1", None)], Some("why"));
    let mut row = TranscriptMessage::from_legacy("assistant", envelope);
    row.content = "[redacted]".into();
    let json = serde_json::to_string(&row).unwrap();
    assert!(!json.contains("secret"), "{json}");
    assert!(!json.contains("why"), "{json}");
    let back: TranscriptMessage = serde_json::from_str(&json).unwrap();
    assert_eq!(back.content, "[redacted]");
}

#[test]
fn rows_from_a_newer_version_are_not_normalized() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "v3-opaque").unwrap();
    let body = concat!(
        r#"{"_meta":{"agent":"a","dispatcher":"native","created":"c","updated":"u","turn_count":0,"input_tokens":0,"output_tokens":0,"cached_input_tokens":0,"charged_amount_usd":0.0}}"#,
        "\n",
        r#"{"v":3,"role":"user","content":"look [OH_IMAGE:data:image/png;base64,AA] here"}"#,
        "\n",
    );
    fs::write(&path, body).unwrap();
    let row = &read_transcript(&path).unwrap().messages[0];
    assert!(row.parts.is_none());
    assert_eq!(row.content, "look [OH_IMAGE:data:image/png;base64,AA] here");
}

#[test]
fn rows_lifted_from_different_legacy_envelopes_are_different_rows() {
    let with_reasoning = TranscriptMessage::from_legacy(
        "assistant",
        encode_assistant_envelope(Some("x"), &[call("c1", None)], Some("private")),
    );
    let redacted = TranscriptMessage::from_legacy(
        "assistant",
        encode_assistant_envelope(Some("x"), &[call("c1", None)], None),
    );
    assert_eq!(with_reasoning.content, redacted.content);
    assert_eq!(with_reasoning.tool_calls, redacted.tool_calls);
    assert!(!with_reasoning.same_row_as(&redacted));
    assert!(with_reasoning.same_row_as(&with_reasoning.clone()));
    // A typed row built from scratch carries no memo and matches on structure.
    let built = TranscriptMessage::assistant_with_calls(
        "x",
        vec![TranscriptToolCall::from(call("c1", None))],
    );
    assert!(redacted.same_row_as(&built));
}

#[test]
fn durable_media_parts_round_trip_without_inline_bytes() {
    let parts = vec![
        TranscriptPart::Text {
            text: "inspect these".into(),
        },
        TranscriptPart::Audio {
            source: TranscriptMediaRef::Path {
                path: "uploads/a/song.mp3".into(),
            },
            mime_type: "audio/mpeg".into(),
        },
        TranscriptPart::Video {
            source: TranscriptMediaRef::Url {
                url: "https://example.com/movie.mp4".into(),
            },
            mime_type: "video/mp4".into(),
        },
        TranscriptPart::Document {
            source: TranscriptMediaRef::Path {
                path: "uploads/a/report.pdf".into(),
            },
            mime_type: "application/pdf".into(),
        },
    ];
    let row = TranscriptMessage::user_with_parts(parts.clone());
    assert_eq!(row.content, "inspect these");
    assert_eq!(row.parts, Some(parts));
    assert_eq!(
        TranscriptMessage::from_legacy("user", row.legacy_content()).parts,
        row.parts
    );
    let dir = tempdir().unwrap();
    let path = dir.path().join("media.jsonl");
    write_transcript(&path, std::slice::from_ref(&row), &meta(), None).unwrap();
    let read = read_transcript(&path).unwrap();
    assert_eq!(read.messages[0].parts, row.parts);
    let stored = fs::read_to_string(&path).unwrap();
    assert!(!stored.contains("base64"));
    assert!(!stored.contains("data:"));
    assert!(row.display_content().contains("[AUDIO:uploads/a/song.mp3]"));
}

#[test]
fn arbitrary_json_is_not_a_media_compatibility_envelope() {
    let audio = TranscriptMessage::user_with_parts(vec![TranscriptPart::Audio {
        source: TranscriptMediaRef::Path {
            path: "uploads/song.mp3".into(),
        },
        mime_type: "audio/mpeg".into(),
    }]);
    let canonical = audio.legacy_content();
    let mut extra: serde_json::Value = serde_json::from_str(&canonical).unwrap();
    extra["user_note"] = serde_json::json!("keep this ordinary JSON");
    let image_only = serde_json::json!({"_tinyagents_media_parts":[TranscriptPart::Image {url:"https://example.com/image.png".into()}]}).to_string();
    let pretty = serde_json::to_string_pretty(
        &serde_json::from_str::<serde_json::Value>(&canonical).unwrap(),
    )
    .unwrap();
    for text in [extra.to_string(), image_only, pretty] {
        let row = TranscriptMessage::from_legacy("user", &text);
        assert_eq!(row.content, text);
        assert!(row.parts.is_none());
        assert_eq!(row.legacy_content(), text);
    }
    assert_eq!(
        TranscriptMessage::from_legacy("user", canonical).parts,
        audio.parts
    );
}
