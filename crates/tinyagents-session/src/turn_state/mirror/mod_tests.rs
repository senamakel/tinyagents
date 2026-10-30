//! Tests for [`TurnStateMirror`]: snapshot priming, transcript coalescing,
//! ordering keys and interrupted-turn finalization.

use super::*;
use crate::port::InMemoryTranscriptLocator;
use crate::port::InMemoryTurnStates;
use crate::transcript::{
    DisplayRecord, TranscriptHistory, TranscriptLocator, TranscriptMessage, TranscriptMeta,
};
use crate::turn_state::types::{SubagentActivity, ToolTimelineEntry, ToolTimelineStatus};
use crate::turn_state::types::{TranscriptItem, TurnLifecycle};

fn memory_store() -> Arc<InMemoryTurnStates> {
    Arc::new(InMemoryTurnStates::default())
}

fn fresh(thread_id: &str) -> (Arc<InMemoryTurnStates>, TurnStateMirror) {
    let store = memory_store();
    let mirror = TurnStateMirror::new(store.clone(), thread_id, "req-1");
    (store, mirror)
}

/// A mirror whose interrupted partials land in `transcripts`.
fn mirror_over(
    transcripts: &Arc<InMemoryTranscriptLocator>,
    store: Arc<dyn TurnStates>,
    thread_id: &str,
    request_id: &str,
) -> TurnStateMirror {
    TurnStateMirror::new(store, thread_id, request_id).with_transcripts(transcripts.clone(), None)
}

fn transcripts() -> Arc<InMemoryTranscriptLocator> {
    Arc::new(InMemoryTranscriptLocator::new("mirror-tests"))
}

/// A store whose conditional write cannot read the stored outcome.
struct UnreadableOutcome {
    inner: InMemoryTurnStates,
    unreadable: std::sync::atomic::AtomicBool,
}

impl TurnStates for UnreadableOutcome {
    fn put(&self, state: &TurnState) -> Result<(), String> {
        self.inner.put(state)
    }
    fn put_unless_completed(&self, state: &TurnState) -> Result<bool, String> {
        if self.unreadable.load(std::sync::atomic::Ordering::SeqCst) {
            return Err("stored snapshot is unreadable".into());
        }
        self.inner.put_unless_completed(state)
    }
    fn get(&self, thread_id: &str) -> Result<Option<TurnState>, String> {
        self.inner.get(thread_id)
    }
    fn get_turn(&self, thread_id: &str, request_id: &str) -> Result<Option<TurnState>, String> {
        self.inner.get_turn(thread_id, request_id)
    }
    fn delete(&self, thread_id: &str) -> Result<bool, String> {
        self.inner.delete(thread_id)
    }
    fn delete_turn(&self, thread_id: &str, request_id: &str) -> Result<bool, String> {
        self.inner.delete_turn(thread_id, request_id)
    }
    fn list(&self) -> Result<Vec<TurnState>, String> {
        self.inner.list()
    }
    fn list_thread(&self, thread_id: &str) -> Result<Vec<TurnState>, String> {
        self.inner.list_thread(thread_id)
    }
    fn clear_all(&self) -> Result<usize, String> {
        self.inner.clear_all()
    }
    fn mark_all_interrupted(&self, now: &str) -> Result<usize, String> {
        self.inner.mark_all_interrupted(now)
    }
    fn settle_turn(
        &self,
        thread_id: &str,
        request_id: &str,
        lifecycle: TurnLifecycle,
        now: &str,
    ) -> Result<bool, String> {
        self.inner
            .settle_turn(thread_id, request_id, lifecycle, now)
    }
}

fn seed_root_transcript(
    transcripts: &InMemoryTranscriptLocator,
    thread_id: &str,
) -> Arc<dyn TranscriptHistory> {
    let meta = TranscriptMeta {
        session_id: None,
        parent_session_id: None,
        agent_name: "orchestrator".into(),
        agent_id: None,
        agent_type: Some("root".into()),
        dispatcher: "native".into(),
        provider: None,
        model: None,
        created: "2026-07-21T00:00:00Z".into(),
        updated: "2026-07-21T00:00:00Z".into(),
        turn_count: 1,
        prefix_message_count: None,
        prefix_reset: false,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: Some(thread_id.to_string()),
        task_id: None,
    };
    let history = transcripts
        .open_stem("100_orchestrator", meta)
        .expect("open transcript");
    history
        .append(TranscriptMessage::new("user", "hello there"))
        .expect("seed transcript");
    history
}

#[test]
fn new_flushes_a_started_snapshot() {
    let (store, mirror) = fresh("thr_new");
    assert_eq!(mirror.snapshot().lifecycle, TurnLifecycle::Started);
    let stored = store
        .get("thr_new")
        .expect("get")
        .expect("snapshot in the store right after new()");
    assert_eq!(stored.request_id, "req-1");
    assert_eq!(stored.lifecycle, TurnLifecycle::Started);
}

#[test]
fn ordering_keys_are_monotonic_and_independent() {
    let (_dir, mut m) = fresh("thr");
    assert_eq!((m.next_seq(), m.next_seq(), m.next_seq()), (0, 1, 2));
    assert_eq!((m.next_tool_seq(), m.next_tool_seq()), (0, 1));
    assert_eq!(m.next_seq(), 3, "tool keys do not consume transcript keys");
}

#[test]
fn narration_coalesces_within_a_round_and_splits_across_rounds() {
    let (_dir, mut m) = fresh("thr");
    m.push_transcript_narration(1, "Hel");
    m.push_transcript_narration(1, "lo");
    m.push_transcript_narration(2, "next");
    assert_eq!(m.state.transcript.len(), 2);
    match &m.state.transcript[0] {
        TranscriptItem::Narration { round, seq, text } => {
            assert_eq!((*round, *seq, text.as_str()), (1, 0, "Hello"));
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn thinking_coalesces_and_an_intervening_tool_starts_a_new_block() {
    let (_dir, mut m) = fresh("thr");
    m.push_transcript_thinking(1, "a");
    m.push_transcript_thinking(1, "b");
    m.push_transcript_tool(1, "call-1");
    m.push_transcript_thinking(1, "c");
    assert_eq!(m.state.transcript.len(), 3);
    match &m.state.transcript[0] {
        TranscriptItem::Thinking {
            text,
            started_at,
            ended_at,
            ..
        } => {
            assert_eq!(text, "ab");
            assert!(started_at.is_some() && ended_at.is_some());
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn tool_call_is_recorded_once_per_call_id() {
    let (_dir, mut m) = fresh("thr");
    m.push_transcript_tool(1, "call-1");
    m.push_transcript_tool(1, "call-1");
    m.push_transcript_tool(1, "call-2");
    assert_eq!(m.state.transcript.len(), 2);
}

#[test]
fn subagent_prose_coalesces_per_kind_and_iteration() {
    let (_dir, mut m) = fresh("thr");
    m.state.tool_timeline.push(ToolTimelineEntry {
        id: "subagent:t1".into(),
        name: "spawn".into(),
        round: 1,
        status: ToolTimelineStatus::Running,
        args_buffer: None,
        display_name: None,
        detail: None,
        source_tool_name: None,
        subagent: Some(
            serde_json::from_value::<SubagentActivity>(
                serde_json::json!({"taskId": "t1", "agentId": "researcher"}),
            )
            .unwrap(),
        ),
        failure: None,
        output: None,
        seq: Some(0),
    });
    m.push_subagent_prose("t1", 1, "x", false);
    m.push_subagent_prose("t1", 1, "y", false);
    m.push_subagent_prose("t1", 1, "think", true);
    m.push_subagent_prose("unknown", 1, "dropped", false);
    let activity = m
        .find_subagent_entry_mut("t1")
        .unwrap()
        .subagent
        .as_ref()
        .unwrap();
    assert_eq!(activity.transcript.len(), 2);
}

#[test]
fn finish_marks_an_unfinished_turn_interrupted() {
    let transcripts = transcripts();
    let store = memory_store();
    let mut m = mirror_over(&transcripts, store.clone(), "thr_new", "req-1");
    m.state.streaming_text = "orphan partial".into();
    // No transcript exists yet: must not panic, snapshot keeps the partial.
    m.finish();
    let stored = store
        .get("thr_new")
        .expect("get")
        .expect("snapshot present");
    assert_eq!(stored.lifecycle, TurnLifecycle::Interrupted);
    assert_eq!(stored.streaming_text, "orphan partial");
}

#[test]
fn finish_appends_interrupted_partial_to_existing_transcript() {
    let transcripts = transcripts();
    let path = seed_root_transcript(&transcripts, "thr_abc");
    let mut m = mirror_over(&transcripts, memory_store(), "thr_abc", "req-9");
    m.state.iteration = 2;
    m.state.thinking = "hmm".into();
    m.state.streaming_text = "half an answer".into();
    m.finish();

    let model = path
        .read_session()
        .expect("read model context")
        .expect("transcript present");
    assert!(
        !model
            .messages
            .iter()
            .any(|msg| msg.content.contains("half an answer")),
        "interrupted partial must be excluded from the model context"
    );
    let display = path
        .read_display()
        .expect("read display")
        .expect("transcript present");
    let partial = display
        .records
        .iter()
        .find_map(|r| match r {
            DisplayRecord::Message(msg) if msg.interrupted => Some(msg),
            _ => None,
        })
        .expect("display must include the interrupted partial");
    assert_eq!(partial.message.content, "half an answer");
    assert_eq!(partial.request_id.as_deref(), Some("req-9"));
    assert_eq!(partial.iteration, Some(2));
    assert_eq!(partial.reasoning_content.as_deref(), Some("hmm"));
}

#[test]
fn finish_after_completion_writes_no_partial() {
    let transcripts = transcripts();
    let path = seed_root_transcript(&transcripts, "thr_done");
    let mut m = mirror_over(&transcripts, memory_store(), "thr_done", "req-done");
    m.state.streaming_text = "final answer".into();
    m.turn_completed = true;
    m.finish();
    let display = path
        .read_display()
        .expect("read display")
        .expect("transcript present");
    assert!(
        !display
            .records
            .iter()
            .any(|r| matches!(r, DisplayRecord::Message(msg) if msg.interrupted)),
        "a completed turn must not append an interrupted partial"
    );
}

#[test]
fn finish_keeps_a_terminal_outcome_the_driver_already_recorded() {
    let transcripts = transcripts();
    let path = seed_root_transcript(&transcripts, "thr_settled");
    let store = memory_store();
    let mut m = mirror_over(&transcripts, store.clone(), "thr_settled", "req-s");
    m.state.streaming_text = "delivered reply".into();
    // The turn driver settles the snapshot as completed behind the mirror's back.
    let mut settled = m.state.clone();
    settled.lifecycle = TurnLifecycle::Completed;
    store.put(&settled).expect("settle");
    m.finish();
    let stored = store.get_turn("thr_settled", "req-s").unwrap().unwrap();
    assert_eq!(stored.lifecycle, TurnLifecycle::Completed);
    let display = path
        .read_display()
        .expect("read display")
        .expect("transcript present");
    assert!(
        !display
            .records
            .iter()
            .any(|r| matches!(r, DisplayRecord::Message(msg) if msg.interrupted)),
        "a settled turn's reply must not be duplicated as a partial"
    );
}

#[test]
fn finish_leaves_an_unreadable_snapshot_alone_and_appends_no_partial() {
    let transcripts = transcripts();
    let path = seed_root_transcript(&transcripts, "thr_corrupt");
    let store = Arc::new(UnreadableOutcome {
        inner: InMemoryTurnStates::default(),
        unreadable: std::sync::atomic::AtomicBool::new(false),
    });
    let mut m = mirror_over(&transcripts, store.clone(), "thr_corrupt", "req-c");
    m.state.streaming_text = "maybe delivered".into();
    store
        .unreadable
        .store(true, std::sync::atomic::Ordering::SeqCst);
    m.finish();
    let stored = store.get_turn("thr_corrupt", "req-c").unwrap().unwrap();
    assert_eq!(
        stored.lifecycle,
        TurnLifecycle::Started,
        "an outcome that cannot be read must not be overwritten"
    );
    let display = path
        .read_display()
        .expect("read display")
        .expect("transcript present");
    assert!(
        !display
            .records
            .iter()
            .any(|r| matches!(r, DisplayRecord::Message(msg) if msg.interrupted)),
        "no partial may be appended when completion cannot be ruled out"
    );
}

#[test]
fn late_boundary_flush_keeps_a_snapshot_the_driver_settled() {
    let store = memory_store();
    let mut m = TurnStateMirror::new(store.clone(), "thr_late", "req-l");
    let mut settled = m.state.clone();
    settled.lifecycle = TurnLifecycle::Completed;
    store.put(&settled).expect("settle");
    m.state.lifecycle = TurnLifecycle::Streaming;
    m.flush();
    let stored = store.get_turn("thr_late", "req-l").unwrap().unwrap();
    assert_eq!(stored.lifecycle, TurnLifecycle::Completed);
}

#[test]
fn finish_without_a_transcript_locator_keeps_the_partial_in_the_snapshot_only() {
    let store = memory_store();
    let mut m = TurnStateMirror::new(store.clone(), "thr_bare", "req-b");
    m.state.streaming_text = "orphan partial".into();
    m.finish();
    let stored = store.get_turn("thr_bare", "req-b").unwrap().unwrap();
    assert_eq!(stored.lifecycle, TurnLifecycle::Interrupted);
    assert_eq!(stored.streaming_text, "orphan partial");
}
