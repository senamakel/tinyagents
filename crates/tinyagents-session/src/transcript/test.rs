//! Module-local unit tests for [`crate::transcript`]: full-rewrite and
//! append-only writers, model-context and display readers, compaction replay,
//! interrupted partials, path resolution/resume, thread-usage summaries, and
//! the [`super::history`] locator/handle seam.
//!
//! Consolidated here per AGENTS.md: one `test.rs` per module directory.

use super::*;
use std::sync::{Arc, Barrier};
use tempfile::tempdir;

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

// ── Session identity ──────────────────────────────────────────────────

/// One conversation, two cold sessions: the second must find the first's file
/// rather than mint a second stem for the same thread. This is the regression
/// that cost a real user the opening turns of a thread.
#[test]
fn one_session_resolves_to_one_transcript_across_separate_bindings() {
    let dir = tempdir().unwrap();
    let session = SessionRef::scoped("thread-9fa08", "orchestrator");

    let first = FileTranscriptLocator::new(dir.path())
        .open_session(&session, meta())
        .unwrap();
    first
        .append(TranscriptMessage::new(
            "user",
            "i want to plan a trip to kashmir",
        ))
        .unwrap();

    // A brand-new locator and handle, as a restarted process would build.
    let second = FileTranscriptLocator::new(dir.path())
        .open_session(&session, meta())
        .unwrap();
    second
        .append(TranscriptMessage::new("user", "hello?"))
        .unwrap();

    assert_eq!(first.path(), second.path());
    let messages = second.messages().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].content, "i want to plan a trip to kashmir");
    assert_eq!(messages[1].content, "hello?");

    let roots: Vec<_> = std::fs::read_dir(dir.path().join("session_raw"))
        .unwrap()
        .flatten()
        .map(|entry| entry.file_name())
        .collect();
    assert_eq!(
        roots.len(),
        1,
        "one conversation must not sprawl: {roots:?}"
    );
}

#[test]
fn an_unwritten_session_reads_as_absent_rather_than_erroring() {
    let dir = tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    let session = SessionRef::scoped("thread-new", "orchestrator");

    assert!(locator.read_session_transcript(&session).is_none());
    assert!(!locator.session_exists(&session));
    assert_eq!(locator.head_generation(&session), session);
}

#[test]
fn session_identity_round_trips_through_the_jsonl_meta() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "identity").unwrap();
    let mut written = meta();
    written.session_id = Some("thread-1.orchestrator.g1".into());
    written.parent_session_id = Some("thread-1.orchestrator".into());
    write_transcript(
        &path,
        &[TranscriptMessage::new("user", "hi")],
        &written,
        None,
    )
    .unwrap();

    let read = read_transcript(&path).unwrap();
    assert_eq!(
        read.meta.session_id.as_deref(),
        Some("thread-1.orchestrator.g1")
    );
    assert_eq!(
        read.meta.parent_session_id.as_deref(),
        Some("thread-1.orchestrator")
    );
}

/// A compaction must never destroy what it replaces. It seals the current
/// generation and opens the next, so the replaced turns stay on disk.
#[test]
fn a_compaction_seals_a_generation_and_leaves_it_untouched() {
    let dir = tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    let session = SessionRef::scoped("thread-1", "orchestrator");

    let first = locator.open_session(&session, meta()).unwrap();
    for turn in ["one", "two", "three"] {
        first.append(TranscriptMessage::new("user", turn)).unwrap();
    }
    let sealed_path = first.path().to_path_buf();
    let sealed_bytes = std::fs::read(&sealed_path).unwrap();

    let (successor, handle) = locator.begin_generation(&session, meta()).unwrap();
    // The successor is bound but empty; the retained set is written through the
    // ordinary turn path so usage and request ids are recorded as usual.
    handle
        .replace(&[TranscriptMessage::new("user", "three")])
        .unwrap();

    assert_eq!(successor.generation, 1);
    assert_eq!(
        std::fs::read(&sealed_path).unwrap(),
        sealed_bytes,
        "the sealed generation must be byte-identical afterwards"
    );
    assert_ne!(handle.path(), sealed_path);

    let carried = handle.messages().unwrap();
    assert_eq!(carried.len(), 1);
    assert_eq!(carried[0].content, "three");

    let successor_meta = handle.read_session().unwrap().unwrap().meta;
    assert_eq!(
        successor_meta.session_id.as_deref(),
        Some(session_stem(&successor).as_str())
    );
    assert_eq!(
        successor_meta.parent_session_id.as_deref(),
        Some(session_stem(&session).as_str())
    );
}

/// After a compaction, a resume must land on the newest generation — the one
/// the model is actually continuing — not on the sealed original.
#[test]
fn head_generation_follows_the_compaction_chain() {
    let dir = tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    let session = SessionRef::scoped("thread-1", "orchestrator");

    locator
        .open_session(&session, meta())
        .unwrap()
        .append(TranscriptMessage::new("user", "one"))
        .unwrap();
    assert_eq!(locator.head_generation(&session), session);

    let (first_successor, first_handle) = locator.begin_generation(&session, meta()).unwrap();
    first_handle
        .append(TranscriptMessage::new("user", "one"))
        .unwrap();
    assert_eq!(locator.head_generation(&session), first_successor);

    let (second_successor, second_handle) =
        locator.begin_generation(&first_successor, meta()).unwrap();
    second_handle
        .append(TranscriptMessage::new("user", "one"))
        .unwrap();
    assert_eq!(locator.head_generation(&session).generation, 2);
    assert_eq!(locator.head_generation(&session), second_successor);
}

#[test]
fn opening_a_generation_that_already_exists_is_refused() {
    let dir = tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    let session = SessionRef::scoped("thread-1", "orchestrator");

    let (_, handle) = locator.begin_generation(&session, meta()).unwrap();
    handle
        .append(TranscriptMessage::new("user", "one"))
        .unwrap();
    let second = locator.begin_generation(&session, meta());

    assert!(
        second.is_err(),
        "sealing the same generation twice would overwrite durable history"
    );
}

/// Two handles on one session — the shape two cores over one workspace
/// produce — must both land in the same file, with neither losing the other's
/// turns.
#[test]
fn concurrent_handles_on_one_session_both_extend_it() {
    let dir = tempdir().unwrap();
    let session = SessionRef::scoped("thread-1", "orchestrator");
    let left = FileTranscriptLocator::new(dir.path())
        .open_session(&session, meta())
        .unwrap();
    let right = FileTranscriptLocator::new(dir.path())
        .open_session(&session, meta())
        .unwrap();

    // Seed the file first, so both appends below exercise the append-only
    // path — `write_logical_set` re-reading `persisted` fresh immediately
    // before every write, which is the actual claim under test — rather than
    // racing on first-write file creation, a distinct, pre-existing concern
    // this test is not about.
    left.append(TranscriptMessage::new("user", "seed")).unwrap();

    // Genuinely overlapping, not merely interleaved: both handles race to
    // append from separate OS threads, released together by a barrier so
    // neither can start before the other is ready. A handle that cached its
    // own view of `persisted` instead of re-reading it fresh before every
    // write could lose whichever append the barrier let land second.
    let barrier = Arc::new(Barrier::new(2));
    let left_barrier = Arc::clone(&barrier);
    let left_thread = std::thread::spawn(move || {
        left_barrier.wait();
        left.append(TranscriptMessage::new("user", "from left"))
            .unwrap();
    });
    let right_barrier = Arc::clone(&barrier);
    let right_thread = std::thread::spawn(move || {
        right_barrier.wait();
        right
            .append(TranscriptMessage::new("user", "from right"))
            .unwrap();
    });
    left_thread.join().unwrap();
    right_thread.join().unwrap();

    let reread = FileTranscriptLocator::new(dir.path())
        .open_session(&session, meta())
        .unwrap();
    let mut contents: Vec<String> = reread
        .messages()
        .unwrap()
        .into_iter()
        .map(|message| message.content)
        .collect();
    assert_eq!(contents.remove(0), "seed");
    contents.sort();
    assert_eq!(
        contents,
        ["from left", "from right"],
        "an overlapping append from either handle must not be lost"
    );
}

/// The model reads only the head generation, but a host rendering or
/// exporting the conversation needs every segment, in order.
#[test]
fn a_session_chain_lists_every_generation_oldest_first() {
    let dir = tempdir().unwrap();
    let locator = FileTranscriptLocator::new(dir.path());
    let session = SessionRef::scoped("thread-1", "orchestrator");

    assert!(locator.session_chain(&session).is_empty());

    locator
        .open_session(&session, meta())
        .unwrap()
        .append(TranscriptMessage::new("user", "one"))
        .unwrap();
    let (successor, handle) = locator.begin_generation(&session, meta()).unwrap();
    handle
        .append(TranscriptMessage::new("user", "two"))
        .unwrap();

    let chain = locator.session_chain(&session);
    assert_eq!(chain, vec![session.clone(), successor]);
    // Asking from any generation returns the same whole chain.
    assert_eq!(locator.session_chain(&chain[1]), chain);
}

/// [`write_transcript_if_absent`] is what keeps adoption from clobbering a
/// destination a concurrent normal turn created while adoption was still
/// scanning legacy roots (see `adoption::adopt_legacy_session_transcripts`).
/// The guarantee has to hold at the level of this primitive: it must publish
/// when nothing is there, and never overwrite when something already is —
/// regardless of *why* the destination already exists.
#[test]
fn write_transcript_if_absent_publishes_once_and_never_overwrites() {
    let dir = tempdir().unwrap();
    let path = resolve_keyed_transcript_path(dir.path(), "identity").unwrap();

    let published = write_transcript_if_absent(
        &path,
        &[TranscriptMessage::new("user", "first writer")],
        &meta(),
    )
    .unwrap();
    assert!(published, "nothing was there yet");
    assert_eq!(
        read_transcript(&path).unwrap().messages[0].content,
        "first writer"
    );

    let published_again = write_transcript_if_absent(
        &path,
        &[TranscriptMessage::new("user", "second writer, loses the race")],
        &meta(),
    )
    .unwrap();
    assert!(!published_again, "the destination already exists");
    // The loser's content must never have touched disk.
    assert_eq!(
        read_transcript(&path).unwrap().messages[0].content,
        "first writer"
    );
}
