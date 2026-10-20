//! End-to-end contract coverage for append-only session transcripts, through
//! the storage-driver store (SQLite) a host installs.

mod common;

use common::storage;
use serde_json::json;
use tinyagents_session::port::{DriverSessionStores, SessionStoreProvider};
use tinyagents_session::thread_spend_in;
use tinyagents_session::transcript::{
    DisplayRecord, MessageUsage, TranscriptMessage, TranscriptMeta, TranscriptPartial,
    TranscriptToolCall, TranscriptTurn, TurnUsage,
};

fn meta(turn_count: usize, input_tokens: u64, output_tokens: u64) -> TranscriptMeta {
    TranscriptMeta {
        session_id: None,
        parent_session_id: None,
        agent_name: "researcher".into(),
        agent_id: Some("researcher-v1".into()),
        agent_type: Some("root".into()),
        dispatcher: "offline-fixture".into(),
        provider: Some("fixture-provider".into()),
        model: Some("fixture-model".into()),
        created: "2026-09-19T10:00:00Z".into(),
        updated: format!("2026-09-19T10:00:0{turn_count}Z"),
        turn_count,
        prefix_message_count: None,
        input_tokens,
        output_tokens,
        cached_input_tokens: 3,
        charged_amount_usd: 0.125,
        thread_id: Some("thread-transcript-e2e".into()),
        task_id: Some("research-task".into()),
    }
}

fn usage(iteration: u32, input: u64, output: u64) -> TurnUsage {
    TurnUsage {
        provider: "fixture-provider".into(),
        model: "fixture-model".into(),
        usage: MessageUsage {
            input,
            output,
            cached_input: 3,
            context_window: 128,
            cost_usd: 0.125,
            ..Default::default()
        },
        ts: format!("2026-09-19T10:00:0{iteration}Z"),
        reasoning_content: Some("checked durable evidence".into()),
        tool_calls: vec![TranscriptToolCall {
            id: "call-search-1".into(),
            name: "search".into(),
            arguments: "{intentionally-not-json}".into(),
            extra_content: Some(json!({"provider": {"opaque": [1, 2]}})),
        }],
        iteration,
    }
}

#[test]
fn append_only_transcript_reopens_with_compacted_context_and_full_display_history() {
    let database = tempfile::tempdir().unwrap();
    let open = || {
        DriverSessionStores::new(storage::fixed_factory(storage::sqlite(
            database.path(),
            "sessions.sqlite",
        )))
        .unwrap()
        .for_agent("researcher")
        .transcripts
    };
    let locator = open();
    let stem = "1700000000_researcher";
    let history = locator.open_stem(stem, meta(0, 0, 0)).unwrap();

    let initial = vec![
        TranscriptMessage {
            id: Some("system-1".into()),
            role: "system".into(),
            content: "Preserve source provenance.".into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            parts: None,
            legacy: Default::default(),
            extra_metadata: Some(json!({"policy_version": 1})),
            cache_breakpoints: vec![8],
            turn_usage: None,
            request_id: None,
            preserve_request_id: false,
            interrupted: false,
            tool_failure: None,
        },
        TranscriptMessage::new("user", "Find the durable transcript contract."),
        TranscriptMessage::assistant("I will search the contract."),
    ];
    let turn = |prev: &[TranscriptMessage],
                next: &[TranscriptMessage],
                meta: &TranscriptMeta,
                usage: &TurnUsage,
                request_id: &str| {
        history
            .append_turn(TranscriptTurn {
                prev,
                next,
                meta,
                turn_usage: Some(usage),
                request_id: Some(request_id),
                tools: None,
            })
            .unwrap();
    };
    turn(
        &[],
        &initial,
        &meta(1, 8, 3),
        &usage(1, 8, 3),
        "request-initial",
    );

    let extended = [
        initial.clone(),
        vec![
            TranscriptMessage::new("tool", "search returned transcript docs"),
            TranscriptMessage::assistant("The transcript is append-only."),
        ],
    ]
    .concat();
    turn(
        &initial,
        &extended,
        &meta(2, 15, 7),
        &usage(2, 7, 4),
        "request-extension",
    );

    let compacted = vec![
        TranscriptMessage::new("system", "Preserve source provenance."),
        TranscriptMessage::assistant("Summary: transcript persistence is append-only."),
        TranscriptMessage::new("user", "Continue from the compacted context."),
    ];
    turn(
        &extended,
        &compacted,
        &meta(3, 21, 9),
        &usage(3, 6, 2),
        "request-compaction",
    );
    assert!(
        locator
            .append_interrupted_partial(
                "thread-transcript-e2e",
                None,
                &TranscriptPartial {
                    content: "This partial must not reach a resumed model.".into(),
                    reasoning_content: Some("stream was cancelled".into()),
                    iteration: Some(4),
                },
                Some("request-interrupted"),
            )
            .unwrap()
    );

    // A fresh handle over the same database, as after a restart.
    drop(history);
    drop(locator);
    let locator = open();
    let reopened = locator
        .root_for_thread("thread-transcript-e2e")
        .expect("the thread's root is found by lookup");
    let replayed = reopened.read_session().unwrap().unwrap();
    assert_eq!(replayed.meta.turn_count, 3);
    assert_eq!(replayed.meta.input_tokens, 21);
    assert_eq!(replayed.meta.output_tokens, 9);
    assert_eq!(
        replayed
            .messages
            .iter()
            .map(|message| (message.role.as_str(), message.content.as_str()))
            .collect::<Vec<_>>(),
        compacted
            .iter()
            .map(|message| (message.role.as_str(), message.content.as_str()))
            .collect::<Vec<_>>(),
    );
    assert!(
        replayed
            .messages
            .iter()
            .all(|message| !message.content.contains("partial must not"))
    );

    let display = reopened.read_display().unwrap().unwrap();
    assert_eq!(display.records.len(), 7);
    assert!(matches!(
        &display.records[0],
        DisplayRecord::Message(message)
            if message.message.id.as_deref() == Some("system-1")
                && message.message.extra_metadata == Some(json!({"policy_version": 1}))
                && message.message.cache_breakpoints == [8]
    ));
    assert!(matches!(
        &display.records[2],
        DisplayRecord::Message(message)
            if message.turn_usage.as_ref().is_some_and(|turn| {
                turn.tool_calls[0].arguments == "{intentionally-not-json}"
                    && turn.tool_calls[0].extra_content == Some(json!({"provider": {"opaque": [1, 2]}}))
            })
    ));
    assert!(matches!(
        &display.records[5],
        DisplayRecord::Compaction(marker)
            if marker.request_id.as_deref() == Some("request-compaction")
                && marker.replacement.len() == compacted.len()
    ));
    assert!(matches!(
        &display.records[6],
        DisplayRecord::Message(message)
            if message.interrupted
                && message.request_id.as_deref() == Some("request-interrupted")
                && message.message.content == "This partial must not reach a resumed model."
                && message.reasoning_content.as_deref() == Some("stream was cancelled")
    ));

    let spend = thread_spend_in(locator.as_ref(), "thread-transcript-e2e");
    assert!(spend.found_transcript);
    assert_eq!(spend.root.input_tokens, 8 + 7 + 6);
    assert_eq!(spend.root.output_tokens, 3 + 4 + 2);
    assert_eq!(spend.root.turns, 3);
    assert_eq!(spend.root.last_input_tokens, 6);
    assert_eq!(spend.root.last_output_tokens, 2);
    assert_eq!(spend.root.model.as_deref(), Some("fixture-model"));
    assert!(spend.subagents.is_empty());
}
