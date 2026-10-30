use super::*;
use crate::transcript::test_locator::driver_locator;
use crate::transcript::{
    DisplayRecord, ToolFailure, TranscriptHistory, TranscriptLocator, TranscriptMeta,
    TranscriptPart, TranscriptRead, TranscriptTurn, lock_session_turn,
};
use std::sync::Arc;

fn meta() -> TranscriptMeta {
    TranscriptMeta {
        session_id: None,
        parent_session_id: None,
        agent_name: "orchestrator".into(),
        agent_id: Some("orchestrator".into()),
        agent_type: None,
        dispatcher: "native".into(),
        provider: None,
        model: None,
        created: "2026-10-06T00:00:00Z".into(),
        updated: "2026-10-06T00:00:00Z".into(),
        turn_count: 1,
        prefix_message_count: None,
        prefix_reset: false,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: Some("thread-1".into()),
        task_id: None,
    }
}

fn session() -> SessionRef {
    SessionRef::scoped("thread-1", "orchestrator")
}

fn options(key: &str) -> BackgroundAppend {
    BackgroundAppend::new(
        key,
        serde_json::json!({"kind": "cron", "job_id": "job-1", "run_id": key}),
    )
}

fn locator() -> Arc<dyn TranscriptLocator> {
    driver_locator()
}

/// Writes one turn the way the live turn path does: through the session's own
/// history handle, diffing `next` against `prev`.
fn live_turn(
    locator: &dyn TranscriptLocator,
    session: &SessionRef,
    prev: &[TranscriptMessage],
    next: &[TranscriptMessage],
) {
    let handle = locator.open_session(session, meta()).unwrap();
    handle
        .append_turn(TranscriptTurn {
            prev,
            next,
            meta: &meta(),
            turn_usage: None,
            request_id: None,
            tools: None,
        })
        .unwrap();
}

fn first_turn() -> Vec<TranscriptMessage> {
    vec![
        TranscriptMessage::user("remind me to stretch at 5"),
        TranscriptMessage::assistant("Scheduled."),
    ]
}

/// The head generation's transcript of `session`.
fn head(locator: &dyn TranscriptLocator, session: &SessionRef) -> Arc<dyn TranscriptRead> {
    let head = locator.head_generation(session);
    locator
        .read_session_transcript(&head)
        .expect("the head generation holds a transcript")
}

/// The head generation's model-context messages.
fn head_messages(locator: &dyn TranscriptLocator, session: &SessionRef) -> Vec<TranscriptMessage> {
    head(locator, session)
        .read_session()
        .unwrap()
        .expect("transcript present")
        .messages
}

async fn deliver(
    locator: &dyn TranscriptLocator,
    key: &str,
) -> anyhow::Result<BackgroundAppendOutcome> {
    append_background_message_in(
        locator,
        &session(),
        TranscriptMessage::assistant("Time to stretch!"),
        options(key),
    )
    .await
}

#[tokio::test]
async fn an_appended_message_is_resumed_as_an_assistant_turn() {
    let locator = locator();
    live_turn(locator.as_ref(), &session(), &[], &first_turn());

    let outcome = deliver(locator.as_ref(), "run-1").await.unwrap();

    assert_eq!(outcome, BackgroundAppendOutcome::Appended { generation: 0 });
    let resumed = head_messages(locator.as_ref(), &session());
    let last = resumed.last().unwrap();
    assert_eq!(resumed.len(), 3);
    assert_eq!(last.role, "assistant");
    assert_eq!(last.content, "Time to stretch!");
}

#[tokio::test]
async fn a_session_without_a_transcript_is_reported_and_not_created() {
    let locator = locator();

    let outcome = deliver(locator.as_ref(), "run-1").await.unwrap();

    assert_eq!(outcome, BackgroundAppendOutcome::NoSession);
    assert!(!locator.session_exists(&locator.head_generation(&session())));
}

#[tokio::test]
async fn the_display_reader_shows_the_message_with_its_provenance() {
    let locator = locator();
    live_turn(locator.as_ref(), &session(), &[], &first_turn());

    deliver(locator.as_ref(), "run-1").await.unwrap();

    let display = head(locator.as_ref(), &session())
        .read_display()
        .unwrap()
        .unwrap();
    let Some(DisplayRecord::Message(last)) = display.records.last() else {
        panic!("expected a message record last");
    };
    assert_eq!(last.message.role, "assistant");
    assert_eq!(last.message.content, "Time to stretch!");
    assert!(!last.interrupted);
    assert_eq!(
        last.background,
        Some(BackgroundOrigin {
            idempotency_key: "run-1".into(),
            provenance: serde_json::json!({"kind": "cron", "job_id": "job-1", "run_id": "run-1"}),
        })
    );
    // Rows the live turn wrote carry no background origin.
    let Some(DisplayRecord::Message(first)) = display.records.first() else {
        panic!("expected a message record first");
    };
    assert_eq!(first.background, None);
}

#[tokio::test]
async fn a_repeated_idempotency_key_is_a_duplicate_and_writes_nothing() {
    let locator = locator();
    live_turn(locator.as_ref(), &session(), &[], &first_turn());
    let first = deliver(locator.as_ref(), "run-1").await.unwrap();
    let before = head_messages(locator.as_ref(), &session());

    let again = deliver(locator.as_ref(), "run-1").await.unwrap();

    assert_eq!(first, BackgroundAppendOutcome::Appended { generation: 0 });
    assert_eq!(again, BackgroundAppendOutcome::Duplicate { generation: 0 });
    assert_eq!(head_messages(locator.as_ref(), &session()), before);
    // A different key is a different delivery.
    let other = deliver(locator.as_ref(), "run-2").await.unwrap();
    assert_eq!(other, BackgroundAppendOutcome::Appended { generation: 0 });
}

/// Seals generation 0 and opens generation 1 with `retained`, the way a
/// compaction in the live turn path does.
fn compact(locator: &dyn TranscriptLocator, retained: &[TranscriptMessage]) {
    let (_, handle) = locator.begin_generation(&session(), meta()).unwrap();
    handle
        .append_turn(TranscriptTurn {
            prev: &[],
            next: retained,
            meta: &meta(),
            turn_usage: None,
            request_id: None,
            tools: None,
        })
        .unwrap();
}

#[tokio::test]
async fn a_stale_expected_generation_writes_nothing() {
    let locator = locator();
    live_turn(locator.as_ref(), &session(), &[], &first_turn());
    compact(locator.as_ref(), &[TranscriptMessage::system("summary")]);
    let sealed_before = locator
        .read_session_transcript(&session())
        .unwrap()
        .read_session()
        .unwrap()
        .unwrap()
        .messages;
    let head_before = head_messages(locator.as_ref(), &session());

    let outcome = append_background_message_in(
        locator.as_ref(),
        &session(),
        TranscriptMessage::assistant("Time to stretch!"),
        options("run-1").expecting_generation(0),
    )
    .await
    .unwrap();

    assert_eq!(
        outcome,
        BackgroundAppendOutcome::StaleGeneration {
            expected: 0,
            head: 1
        }
    );
    let sealed_after = locator
        .read_session_transcript(&session())
        .unwrap()
        .read_session()
        .unwrap()
        .unwrap()
        .messages;
    assert_eq!(sealed_after, sealed_before);
    assert_eq!(head_messages(locator.as_ref(), &session()), head_before);
}

#[tokio::test]
async fn the_append_lands_in_the_head_generation() {
    let locator = locator();
    live_turn(locator.as_ref(), &session(), &[], &first_turn());
    compact(locator.as_ref(), &[TranscriptMessage::system("summary")]);

    let outcome = append_background_message_in(
        locator.as_ref(),
        &session(),
        TranscriptMessage::assistant("Time to stretch!"),
        options("run-1").expecting_generation(1),
    )
    .await
    .unwrap();

    assert_eq!(outcome, BackgroundAppendOutcome::Appended { generation: 1 });
    assert_eq!(
        head_messages(locator.as_ref(), &session())
            .iter()
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>(),
        ["summary", "Time to stretch!"]
    );
    let sealed = locator
        .read_session_transcript(&session())
        .unwrap()
        .read_session()
        .unwrap()
        .unwrap();
    assert_eq!(sealed.messages.len(), 2);
}

/// The cron job finishes while the user is mid-turn in the same thread. The
/// live turn holds the turn lock from its resume read to its persist, so the
/// background append waits for it, and the next turn resumes over both.
///
/// Single-threaded on purpose: the spawned delivery runs only when this task
/// yields, so "it is still waiting" is decided by the lock, not by timing.
#[tokio::test]
async fn a_background_append_waits_for_the_live_turn_and_keeps_both_in_order() {
    let locator = locator();
    live_turn(locator.as_ref(), &session(), &[], &first_turn());

    // Turn two starts: lock, then read its baseline.
    let turn = lock_session_turn(locator.as_ref(), &session())
        .await
        .unwrap();
    let baseline = head_messages(locator.as_ref(), &session());

    let delivery = {
        let locator = Arc::clone(&locator);
        tokio::spawn(async move { deliver(locator.as_ref(), "run-1").await })
    };
    for _ in 0..50 {
        tokio::task::yield_now().await;
    }
    assert!(
        !delivery.is_finished(),
        "delivery must wait for the live turn"
    );
    assert_eq!(head_messages(locator.as_ref(), &session()), baseline);

    // Turn two persists against the baseline it read, then releases the lock.
    let mut second = baseline.clone();
    second.push(TranscriptMessage::user("what's the weather?"));
    second.push(TranscriptMessage::assistant("Sunny."));
    live_turn(locator.as_ref(), &session(), &baseline, &second);
    drop(turn);

    assert_eq!(
        delivery.await.unwrap().unwrap(),
        BackgroundAppendOutcome::Appended { generation: 0 }
    );

    // Turn three resumes over everything and appends on top of it.
    let _turn = lock_session_turn(locator.as_ref(), &session())
        .await
        .unwrap();
    let resumed = head_messages(locator.as_ref(), &session());
    let mut third = resumed.clone();
    third.push(TranscriptMessage::user("did you remind me?"));
    third.push(TranscriptMessage::assistant("Yes, at 5."));
    live_turn(locator.as_ref(), &session(), &resumed, &third);

    let contents: Vec<(String, String)> = head_messages(locator.as_ref(), &session())
        .into_iter()
        .map(|message| (message.role, message.content))
        .collect();
    let expected: Vec<(String, String)> = [
        ("user", "remind me to stretch at 5"),
        ("assistant", "Scheduled."),
        ("user", "what's the weather?"),
        ("assistant", "Sunny."),
        ("assistant", "Time to stretch!"),
        ("user", "did you remind me?"),
        ("assistant", "Yes, at 5."),
    ]
    .into_iter()
    .map(|(role, content)| (role.to_string(), content.to_string()))
    .collect();
    assert_eq!(contents, expected);
}

#[tokio::test]
async fn only_plain_assistant_messages_with_a_key_are_accepted() {
    let locator = locator();
    live_turn(locator.as_ref(), &session(), &[], &first_turn());
    let before = head_messages(locator.as_ref(), &session());
    let mut interrupted = TranscriptMessage::assistant("half an ans");
    interrupted.interrupted = true;
    let calls = TranscriptMessage::assistant_with_calls(
        "",
        vec![crate::transcript::TranscriptToolCall {
            id: "call-1".into(),
            name: "shell".into(),
            arguments: "{}".into(),
            extra_content: None,
        }],
    );
    let mut tool_result_id = TranscriptMessage::assistant("not a tool result");
    tool_result_id.tool_call_id = Some("call-1".into());
    let mut parts = TranscriptMessage::assistant("not a user row");
    parts.parts = Some(vec![TranscriptPart::Text {
        text: "text".into(),
    }]);
    let mut failed = TranscriptMessage::assistant("failed");
    failed.tool_failure = Some(ToolFailure {
        failed: true,
        detail: Some("failed".into()),
    });

    for (message, key) in [
        (TranscriptMessage::user("hi"), "run-1"),
        (TranscriptMessage::system("hi"), "run-1"),
        (calls, "run-1"),
        (tool_result_id, "run-1"),
        (parts, "run-1"),
        (failed, "run-1"),
        (interrupted, "run-1"),
        (TranscriptMessage::assistant("fine"), "  "),
    ] {
        let result =
            append_background_message_in(locator.as_ref(), &session(), message, options(key)).await;
        assert!(result.is_err(), "{result:?}");
    }
    assert_eq!(head_messages(locator.as_ref(), &session()), before);
}

#[tokio::test]
async fn an_attempt_against_a_sealed_head_writes_nothing_and_asks_to_re_resolve() {
    let locator = locator();
    live_turn(locator.as_ref(), &session(), &[], &first_turn());
    compact(locator.as_ref(), &[TranscriptMessage::system("summary")]);
    let sealed = locator.read_session_transcript(&session()).unwrap();
    let before = sealed.read_session().unwrap().unwrap().messages;

    let attempt = append_to_stored_head(
        locator.as_ref(),
        &session(),
        &TranscriptMessage::assistant("Time to stretch!"),
        &options("run-1"),
    )
    .unwrap();

    assert!(matches!(attempt, Attempt::HeadMoved));
    assert_eq!(sealed.read_session().unwrap().unwrap().messages, before);
}

/// A store-like locator that names no destination, over real transcripts.
struct Unnamed(Arc<dyn TranscriptLocator>);

impl TranscriptLocator for Unnamed {
    fn latest_for_agent(&self, agent_name: &str) -> Option<Arc<dyn TranscriptRead>> {
        self.0.latest_for_agent(agent_name)
    }
    fn root_for_thread(&self, thread_id: &str) -> Option<Arc<dyn TranscriptRead>> {
        self.0.root_for_thread(thread_id)
    }
    fn open_stem(
        &self,
        stem: &str,
        seed: TranscriptMeta,
    ) -> anyhow::Result<Arc<dyn TranscriptHistory>> {
        self.0.open_stem(stem, seed)
    }
}

#[tokio::test]
async fn appends_through_one_unnamed_locator_are_serialised() {
    let inner = locator();
    live_turn(inner.as_ref(), &session(), &[], &first_turn());
    let locator = Arc::new(Unnamed(inner));
    assert!(locator.destination_key().is_none());

    let held = super::super::turn_lock::lock_session_turn_or_identity(
        locator.as_ref() as &dyn TranscriptLocator,
        &session(),
    )
    .await;
    let delivery = {
        let locator = Arc::clone(&locator);
        tokio::spawn(async move { deliver(locator.as_ref(), "run-1").await })
    };
    for _ in 0..50 {
        tokio::task::yield_now().await;
    }
    assert!(
        !delivery.is_finished(),
        "an append through the same unnamed locator waits for its lock"
    );
    drop(held);
    assert_eq!(
        delivery.await.unwrap().unwrap(),
        BackgroundAppendOutcome::Appended { generation: 0 }
    );
}

/// A compaction that keeps the delivered message carries its provenance into
/// the successor, so a retry is a duplicate there too.
#[tokio::test]
async fn a_delivery_carried_through_a_compaction_is_still_a_duplicate() {
    let locator = locator();
    live_turn(locator.as_ref(), &session(), &[], &first_turn());
    assert_eq!(
        deliver(locator.as_ref(), "run-1").await.unwrap(),
        BackgroundAppendOutcome::Appended { generation: 0 }
    );

    // The runtime compacts from the context it read back.
    let retained = head_messages(locator.as_ref(), &session());
    compact(locator.as_ref(), &retained);

    assert_eq!(
        deliver(locator.as_ref(), "run-1").await.unwrap(),
        BackgroundAppendOutcome::Duplicate { generation: 1 }
    );
    let display = head(locator.as_ref(), &session())
        .read_display()
        .unwrap()
        .unwrap();
    let carried = display
        .records
        .iter()
        .filter_map(|record| match record {
            DisplayRecord::Message(message) => message.background.as_ref(),
            DisplayRecord::Compaction(_) => None,
        })
        .count();
    assert_eq!(carried, 1, "the successor's row keeps its provenance");
}

/// A compaction inside the *same* generation (the handle's `replace`) that
/// drops the delivered message removes it from the live context, so the key is
/// free again: retrying the delivery appends it instead of reporting a
/// duplicate of a row the model no longer sees.
#[tokio::test]
async fn a_delivery_dropped_by_a_same_generation_compaction_is_appended_again() {
    let locator = locator();
    live_turn(locator.as_ref(), &session(), &[], &first_turn());
    assert_eq!(
        deliver(locator.as_ref(), "run-1").await.unwrap(),
        BackgroundAppendOutcome::Appended { generation: 0 }
    );

    head_handle(locator.as_ref())
        .replace(&[TranscriptMessage::system("summary")])
        .unwrap();

    assert_eq!(
        deliver(locator.as_ref(), "run-1").await.unwrap(),
        BackgroundAppendOutcome::Appended { generation: 0 },
        "the compaction dropped the delivery, so the key is not held"
    );
    assert_eq!(
        deliver(locator.as_ref(), "run-1").await.unwrap(),
        BackgroundAppendOutcome::Duplicate { generation: 0 },
        "the fresh delivery is held again"
    );
}

/// The same-generation counterpart: a compaction that keeps the delivery keeps
/// its provenance, so the retry is still a duplicate.
#[tokio::test]
async fn a_delivery_kept_by_a_same_generation_compaction_is_still_a_duplicate() {
    let locator = locator();
    live_turn(locator.as_ref(), &session(), &[], &first_turn());
    deliver(locator.as_ref(), "run-1").await.unwrap();

    let retained = head_messages(locator.as_ref(), &session());
    head_handle(locator.as_ref()).replace(&retained).unwrap();

    assert_eq!(
        deliver(locator.as_ref(), "run-1").await.unwrap(),
        BackgroundAppendOutcome::Duplicate { generation: 0 }
    );
}

fn head_handle(locator: &dyn TranscriptLocator) -> Arc<dyn TranscriptHistory> {
    let head = locator.head_generation(&session());
    locator.open_session(&head, meta()).unwrap()
}
