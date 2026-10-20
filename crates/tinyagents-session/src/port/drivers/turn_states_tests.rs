use super::*;
use crate::testkit::MemoryStorage;
use crate::turn_state::types::{SubagentTranscriptItem, TranscriptItem};
use crate::turn_state::{
    SubagentActivity, SubagentToolCall, ToolTimelineEntry, ToolTimelineStatus,
};

use serde_json::json;
use tinyagents_storage_kit::Scope;

fn store() -> (DriverTurnStates, Arc<dyn DocumentStore>) {
    let storage = MemoryStorage::new();
    let docs = Arc::clone(
        storage
            .for_scope(&Scope::new("local").unwrap())
            .unwrap()
            .documents(),
    );
    (
        DriverTurnStates::new(Arc::clone(&docs), Blocking::new().unwrap()),
        docs,
    )
}

fn started(thread: &str, request: &str, minute: u32) -> TurnState {
    TurnState::started(thread, request, 8, format!("2026-01-01T00:{minute:02}:00Z"))
}

fn completed(thread: &str, request: &str, minute: u32) -> TurnState {
    let mut turn = started(thread, request, minute);
    turn.lifecycle = TurnLifecycle::Completed;
    turn.updated_at = turn.started_at.clone();
    turn
}

#[test]
fn completed_turns_are_kept_up_to_the_retention_limit() {
    let (turns, _) = store();
    for minute in 0..(COMPLETED_RETENTION as u32 + 3) {
        turns
            .put(&completed("t", &format!("r{minute:02}"), minute))
            .unwrap();
    }
    turns.put(&started("t", "live", 59)).unwrap();
    let kept = turns.list_thread("t").unwrap();
    assert_eq!(kept.len(), COMPLETED_RETENTION + 1);
    assert!(
        turns.get_turn("t", "r00").unwrap().is_none(),
        "the oldest completed turns go first"
    );
    assert!(
        turns.get_turn("t", "live").unwrap().is_some(),
        "live turns stay"
    );
}

#[test]
fn settling_prunes_too() {
    let (turns, _) = store();
    for minute in 0..COMPLETED_RETENTION as u32 {
        turns
            .put(&completed("t", &format!("r{minute:02}"), minute))
            .unwrap();
    }
    turns.put(&started("t", "last", 58)).unwrap();
    assert!(
        turns
            .settle_turn(
                "t",
                "last",
                TurnLifecycle::Completed,
                "2026-01-01T01:00:00Z"
            )
            .unwrap()
    );
    assert_eq!(turns.list_thread("t").unwrap().len(), COMPLETED_RETENTION);
}

#[test]
fn put_unless_completed_inserts_updates_and_then_holds() {
    let (turns, _) = store();
    let mut turn = started("t", "r", 0);
    assert!(
        turns.put_unless_completed(&turn).unwrap(),
        "a new turn is written"
    );
    turn.lifecycle = TurnLifecycle::Completed;
    assert!(
        turns.put_unless_completed(&turn).unwrap(),
        "a live turn is updated"
    );
    assert!(
        !turns.put_unless_completed(&started("t", "r", 1)).unwrap(),
        "a completed turn is never overwritten conditionally"
    );
    assert_eq!(
        turns.get_turn("t", "r").unwrap().unwrap().lifecycle,
        TurnLifecycle::Completed
    );
}

#[test]
fn settling_needs_a_live_turn() {
    let (turns, _) = store();
    let now = "2026-01-01T01:00:00Z";
    assert!(
        !turns
            .settle_turn("t", "missing", TurnLifecycle::Interrupted, now)
            .unwrap()
    );
    turns.put(&completed("t", "done", 0)).unwrap();
    assert!(
        !turns
            .settle_turn("t", "done", TurnLifecycle::Interrupted, now)
            .unwrap()
    );
    let mut live = started("t", "live", 1);
    live.phase = Some(crate::turn_state::TurnPhase::Thinking);
    turns.put(&live).unwrap();
    assert!(
        turns
            .settle_turn("t", "live", TurnLifecycle::Interrupted, now)
            .unwrap()
    );
    let settled = turns.get_turn("t", "live").unwrap().unwrap();
    assert_eq!(settled.lifecycle, TurnLifecycle::Interrupted);
    assert_eq!(settled.phase, None);
    assert_eq!(settled.updated_at, now);
}

#[test]
fn deletes_report_whether_anything_went() {
    let (turns, _) = store();
    assert!(!turns.delete("t").unwrap());
    assert!(!turns.delete_turn("t", "r").unwrap());
    turns.put(&started("t", "r1", 0)).unwrap();
    turns.put(&started("t", "r2", 1)).unwrap();
    turns.put(&started("u", "r1", 2)).unwrap();
    assert!(turns.delete("t").unwrap());
    assert!(turns.list_thread("t").unwrap().is_empty());
    assert_eq!(turns.list().unwrap().len(), 1);
    assert_eq!(turns.clear_all().unwrap(), 1);
    assert_eq!(turns.clear_all().unwrap(), 0);
}

#[test]
fn list_answers_each_threads_newest_turn() {
    let (turns, _) = store();
    turns.put(&started("t", "old", 0)).unwrap();
    turns.put(&started("t", "new", 5)).unwrap();
    turns.put(&started("u", "only", 1)).unwrap();
    let mut latest: Vec<(String, String)> = turns
        .list()
        .unwrap()
        .into_iter()
        .map(|turn| (turn.thread_id, turn.request_id))
        .collect();
    latest.sort();
    assert_eq!(
        latest,
        [
            ("t".to_string(), "new".to_string()),
            ("u".to_string(), "only".to_string())
        ]
    );
}

#[test]
fn ids_with_separators_do_not_collide() {
    let (turns, _) = store();
    turns.put(&started("a/b", "c", 0)).unwrap();
    turns.put(&started("a", "b/c", 1)).unwrap();
    assert_eq!(turns.list().unwrap().len(), 2);
}

#[test]
fn an_unreadable_snapshot_is_an_error() {
    let (turns, docs) = store();
    turns.put(&started("t", "r", 0)).unwrap();
    let bridge = Blocking::new().unwrap();
    let raw = Arc::clone(&docs);
    bridge
        .run(async move {
            raw.put(
                COLLECTION,
                &id("t", "bodiless").unwrap(),
                json!({"thread_id": "t", "request_id": "bodiless"}),
                Precondition::None,
            )
            .await
        })
        .unwrap()
        .unwrap();
    let error = turns.list_thread("t").unwrap_err();
    assert!(error.contains("unreadable"), "{error}");
    let raw = Arc::clone(&docs);
    bridge
        .run(async move {
            raw.put(
                COLLECTION,
                &id("t", "bodiless").unwrap(),
                json!({"thread_id": "t", "request_id": "bodiless", "started_ms": "nope", "updated_ms": 0}),
                Precondition::None,
            )
            .await
        })
        .unwrap()
        .unwrap();
    let error = turns.get_turn("t", "bodiless").unwrap_err();
    assert!(error.contains("unreadable"), "{error}");
}

#[test]
fn only_a_lost_race_counts_as_contention() {
    assert!(is_race(&StorageError::conflict("raced")));
    assert!(!is_race(&StorageError::unavailable("down")));
}

#[test]
fn a_turn_is_stored_as_a_bare_snake_case_document_with_epoch_ms() {
    let (turns, docs) = store();
    let mut turn = started("thread/1", "req/1", 5);
    turn.updated_at = "2026-01-01T00:06:30.250Z".into();
    turn.max_iterations = 9;
    turn.streaming_text = "hi".into();
    turns.put(&turn).unwrap();

    let bridge = Blocking::new().unwrap();
    let stored = bridge
        .run(async move {
            docs.get("session.turn_states", "thread%2F1/req%2F1")
                .await
                .unwrap()
        })
        .unwrap()
        .expect("stored under Key::of([thread, request])");
    let started_ms = 1_767_225_600_000_i64 + 5 * 60_000; // 2026-01-01T00:05:00Z
    assert_eq!(
        stored.doc,
        json!({
            "thread_id": "thread/1",
            "request_id": "req/1",
            "lifecycle": "started",
            "iteration": 0,
            "max_iterations": 9,
            "streaming_text": "hi",
            "thinking": "",
            "tool_timeline": [],
            "started_ms": started_ms,
            "updated_ms": started_ms + 90_250,
        })
    );
    assert_eq!(stored.meta.schema, 2, "written at the collection's version");
    assert_eq!(turns.get("thread/1").unwrap(), Some(turn));
}

#[test]
fn nested_wire_fields_are_snake_case_in_storage_and_args_stay_verbatim() {
    let (turns, docs) = store();
    let mut turn = started("t", "r", 0);
    turn.transcript = vec![
        TranscriptItem::Thinking {
            round: 1,
            seq: 0,
            text: "hmm".into(),
            started_at: Some(1_000),
            ended_at: Some(2_500),
        },
        TranscriptItem::ToolCall {
            round: 1,
            seq: 1,
            call_id: "c1".into(),
        },
    ];
    turn.tool_timeline.push(ToolTimelineEntry {
        id: "c1".into(),
        name: "search".into(),
        round: 1,
        status: ToolTimelineStatus::Success,
        args_buffer: Some("{}".into()),
        display_name: None,
        detail: None,
        source_tool_name: None,
        subagent: Some(SubagentActivity {
            task_id: "s".into(),
            agent_id: "a".into(),
            status: None,
            mode: None,
            dedicated_thread: None,
            child_iteration: None,
            child_max_iterations: None,
            iterations: None,
            elapsed_ms: Some(7),
            output_chars: None,
            worker_thread_id: None,
            parent_call_id: None,
            output: None,
            tool_calls: vec![SubagentToolCall {
                call_id: "k".into(),
                tool_name: "fetch".into(),
                status: ToolTimelineStatus::Success,
                iteration: None,
                elapsed_ms: None,
                output_chars: None,
                display_name: None,
                detail: None,
                args: Some(json!({"someKey": {"nestedCamel": 1}})),
                failure: None,
                output: None,
            }],
            transcript: Vec::new(),
        }),
        failure: None,
        output: None,
        seq: Some(2),
    });
    turns.put(&turn).unwrap();

    let stored = Blocking::new()
        .unwrap()
        .run(async move { docs.get("session.turn_states", "t/r").await.unwrap() })
        .unwrap()
        .unwrap()
        .doc;
    assert_eq!(stored["transcript"][0]["started_ms"], 1_000);
    assert_eq!(stored["transcript"][0]["ended_ms"], 2_500);
    assert!(stored["transcript"][0].get("startedAt").is_none());
    assert_eq!(stored["transcript"][1]["call_id"], "c1");
    let entry = &stored["tool_timeline"][0];
    assert_eq!(entry["args_buffer"], "{}");
    assert_eq!(entry["subagent"]["task_id"], "s");
    assert_eq!(entry["subagent"]["elapsed_ms"], 7);
    assert_eq!(
        entry["subagent"]["tool_calls"][0]["args"],
        json!({"someKey": {"nestedCamel": 1}}),
        "the tool's own arguments are not ours to rename"
    );
    assert_eq!(turns.get("t").unwrap(), Some(turn));
}

#[test]
fn a_turn_whose_time_is_not_rfc3339_is_refused() {
    let (turns, _) = store();
    let mut turn = started("t", "r", 0);
    turn.started_at = "yesterday".into();
    let error = turns.put(&turn).unwrap_err();
    assert!(error.contains("RFC 3339"), "{error}");
}

#[test]
fn concurrent_settles_and_sweeps_never_lose_a_turn() {
    let (turns, _) = store();
    for i in 0..8 {
        turns.put(&started("t", &format!("r{i}"), i)).unwrap();
    }
    let handles: Vec<_> = (0..4)
        .map(|worker| {
            let turns = turns.clone();
            std::thread::spawn(move || {
                if worker == 0 {
                    turns.mark_all_interrupted("2026-01-01T02:00:00Z").unwrap();
                } else {
                    for i in 0..8 {
                        turns
                            .settle_turn(
                                "t",
                                &format!("r{i}"),
                                TurnLifecycle::Completed,
                                "2026-01-01T02:00:00Z",
                            )
                            .unwrap();
                    }
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
    let all = turns.list_thread("t").unwrap();
    assert_eq!(all.len(), 8);
    assert!(all.iter().all(is_terminal), "every turn ended up terminal");
}

#[test]
fn roundtrips_subagent_interleaved_transcript_with_full_fidelity() {
    // A settled turn whose subagent streamed reasoning, called a tool, then
    // narrated — the interleaved transcript (not just the flat tool rows) plus
    // the per-row `seq` ordering keys must survive a driver round-trip verbatim,
    // so a reopened transcript rehydrates without losing the subagent's
    // reasoning (the gap SubagentDrawer/chatRuntimeSlice documented).
    let (store, _docs) = store();
    let mut state = started("thread-sub", "req-1", 0);
    state.lifecycle = TurnLifecycle::Completed;
    state.updated_at = state.started_at.clone();

    let activity = SubagentActivity {
        task_id: "sub-1".into(),
        agent_id: "researcher".into(),
        status: Some("completed".into()),
        mode: Some("typed".into()),
        dedicated_thread: Some(false),
        child_iteration: Some(1),
        child_max_iterations: Some(8),
        iterations: Some(2),
        elapsed_ms: Some(1234),
        output_chars: Some(42),
        worker_thread_id: Some("worker-thread-9".into()),
        parent_call_id: Some("call-1".into()),
        output: Some("done".into()),
        tool_calls: vec![SubagentToolCall {
            call_id: "c1".into(),
            tool_name: "search".into(),
            status: ToolTimelineStatus::Success,
            iteration: Some(1),
            elapsed_ms: Some(12),
            output_chars: Some(6),
            display_name: Some("Searching".into()),
            detail: None,
            args: Some(serde_json::json!({ "query": "rust serde" })),
            failure: None,
            output: Some("3 hits".into()),
        }],
        transcript: vec![
            SubagentTranscriptItem::Thinking {
                iteration: Some(1),
                text: "let me search.".into(),
            },
            SubagentTranscriptItem::Tool {
                iteration: Some(1),
                call_id: "c1".into(),
                tool_name: "search".into(),
                status: ToolTimelineStatus::Success,
                elapsed_ms: Some(12),
                output_chars: Some(6),
                display_name: Some("Searching".into()),
                detail: None,
            },
            SubagentTranscriptItem::Text {
                iteration: Some(1),
                text: "Found it.".into(),
            },
        ],
    };

    state.tool_timeline.push(ToolTimelineEntry {
        id: "subagent:sub-1".into(),
        name: "subagent:researcher".into(),
        round: 1,
        status: ToolTimelineStatus::Success,
        args_buffer: None,
        display_name: Some("Researcher".into()),
        detail: None,
        source_tool_name: Some("spawn_subagent".into()),
        subagent: Some(activity),
        failure: None,
        output: None,
        seq: Some(3),
    });

    store.put(&state).expect("put");
    let loaded = store.get("thread-sub").expect("get").expect("present");
    // Structural equality proves nothing in the interleaved transcript,
    // subagent activity, or the `seq` ordering keys was dropped or reordered.
    assert_eq!(loaded, state);

    // Spot-check the interleaving explicitly so a future regression that keeps
    // the fields but loses the ordering still fails here.
    let activity = loaded.tool_timeline[0]
        .subagent
        .as_ref()
        .expect("subagent activity restored");
    assert_eq!(activity.transcript.len(), 3);
    assert!(matches!(
        activity.transcript[0],
        SubagentTranscriptItem::Thinking { .. }
    ));
    assert!(matches!(
        activity.transcript[1],
        SubagentTranscriptItem::Tool { .. }
    ));
    assert!(matches!(
        activity.transcript[2],
        SubagentTranscriptItem::Text { .. }
    ));
    // The child call's arguments are what the reloaded row's "Input" block
    // renders from. Spot-checked by value (like the interleaving above) so a
    // regression names the field instead of failing as one opaque struct
    // inequality (#5987).
    assert_eq!(
        activity.tool_calls[0].args,
        Some(serde_json::json!({ "query": "rust serde" })),
        "child tool arguments must survive the driver round-trip"
    );
    assert_eq!(loaded.tool_timeline[0].seq, Some(3));
}
