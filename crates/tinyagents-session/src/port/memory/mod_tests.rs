use super::*;
use crate::testkit::conformance::{session_store_conformance, session_store_isolation_conformance};
use crate::transcript::{SessionRef, TranscriptMessage, TranscriptTurn};

fn test_meta(thread_id: &str) -> TranscriptMeta {
    TranscriptMeta {
        session_id: None,
        parent_session_id: None,
        agent_name: "planner".to_string(),
        agent_id: Some("planner".to_string()),
        agent_type: None,
        dispatcher: "native".to_string(),
        provider: None,
        model: None,
        created: "2026-01-01T00:00:00Z".to_string(),
        updated: "2026-01-01T00:00:00Z".to_string(),
        turn_count: 0,
        prefix_message_count: None,
        prefix_reset: false,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: Some(thread_id.to_string()),
        task_id: None,
    }
}

#[tokio::test]
async fn in_memory_stores_meet_the_session_store_contract() {
    session_store_conformance(&InMemorySessionStores::new()).await;
}

#[tokio::test]
async fn in_memory_stores_keep_agents_apart() {
    session_store_isolation_conformance(&InMemorySessionStores::new()).await;
}

#[test]
fn an_agent_gets_the_same_stores_every_time() {
    let provider = InMemorySessionStores::new();
    let first = provider.for_agent("a");
    let again = provider.for_agent("a");
    assert!(Arc::ptr_eq(&first.turn_states, &again.turn_states));
    assert_eq!(
        first.transcripts.destination_key(),
        again.transcripts.destination_key()
    );
    assert!(format!("{provider:?}").contains("agents: 1"));
    assert!(format!("{first:?}").contains("AgentStores"));
    assert!(
        provider
            .destination_key()
            .is_some_and(|key| key.starts_with("memory://"))
    );
}

#[test]
fn in_memory_stores_can_lease_their_stable_agent_handles() {
    let provider = InMemorySessionStores::new();
    let opened = provider.for_agent("a");
    let lease = provider
        .lease_for_agent("a")
        .expect("the process-lifetime memory provider pins its handles");

    assert!(Arc::ptr_eq(
        &opened.turn_states,
        &lease.stores().turn_states
    ));
    assert_eq!(
        opened.transcripts.destination_key(),
        lease.stores().transcripts.destination_key()
    );
}

#[test]
fn clearing_in_memory_history_clears_display_partials() {
    let history = InMemoryTranscriptHistory::new("clear-partial", test_meta("clear-partial"));
    history.record_partial(
        TranscriptPartial::new("half an answer"),
        Some("req-1".into()),
    );

    TranscriptHistory::clear(&history).unwrap();

    assert!(TranscriptHistory::messages(&history).unwrap().is_empty());
    assert!(history.partials().is_empty());
}

#[test]
fn recovery_interrupts_every_agents_turns_in_flight() {
    let provider = InMemorySessionStores::new();
    for agent in ["a", "b"] {
        provider
            .for_agent(agent)
            .turn_states
            .put(&TurnState::started("t", "r", 4, "2026-01-01T00:00:00Z"))
            .unwrap();
    }
    provider.recover().unwrap();
    for agent in ["a", "b"] {
        let turn = provider.for_agent(agent).turn_states.get("t").unwrap();
        assert_eq!(turn.map(|t| t.lifecycle), Some(TurnLifecycle::Interrupted));
    }
    // Shared behind an Arc, it is still the same provider.
    let shared: Arc<dyn SessionStoreProvider> = Arc::new(provider);
    assert!(shared.recover().is_ok());
    assert!(shared.destination_key().is_some());
    assert!(shared.workspace_dir().is_none(), "memory keeps no files");
    assert!(
        Arc::new(shared.clone())
            .for_agent("a")
            .turn_states
            .get("t")
            .unwrap()
            .is_some()
    );
}

#[test]
fn completed_turns_are_kept_up_to_the_retention_limit() {
    let turns = InMemoryTurnStates::default();
    for minute in 0..(COMPLETED_RETENTION + 3) {
        let mut turn = TurnState::started(
            "t",
            format!("r{minute}"),
            4,
            format!("2026-01-01T00:{minute:02}:00Z"),
        );
        turn.lifecycle = TurnLifecycle::Completed;
        turns.put(&turn).unwrap();
    }
    let kept = turns.list_thread("t").unwrap();
    assert_eq!(kept.len(), COMPLETED_RETENTION);
    assert_eq!(kept[0].request_id, format!("r{}", COMPLETED_RETENTION + 2));
    // A conditional completed write prunes too, and a missing turn does not
    // settle.
    let mut late = TurnState::started("t", "late", 4, "2026-01-01T01:00:00Z");
    late.lifecycle = TurnLifecycle::Completed;
    assert!(turns.put_unless_completed(&late).unwrap());
    assert_eq!(turns.list_thread("t").unwrap().len(), COMPLETED_RETENTION);
    assert!(
        !turns
            .settle_turn("t", "missing", TurnLifecycle::Completed, "now")
            .unwrap()
    );
    assert!(
        !turns
            .settle_turn("t", "late", TurnLifecycle::Interrupted, "now")
            .unwrap(),
        "a terminal turn does not settle again"
    );
    assert!(turns.delete("t").unwrap());
    assert!(!turns.delete("t").unwrap());
}

#[test]
fn the_newest_root_wins_and_sub_agents_are_never_roots() {
    let locator = InMemoryTranscriptLocator::new("test");
    let meta = |agent: &str| TranscriptMeta {
        session_id: None,
        parent_session_id: None,
        agent_name: agent.to_string(),
        agent_id: Some(agent.to_string()),
        agent_type: None,
        dispatcher: "native".to_string(),
        provider: None,
        model: None,
        created: "2026-01-01T00:00:00Z".to_string(),
        updated: "2026-01-01T00:00:00Z".to_string(),
        turn_count: 0,
        prefix_message_count: None,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: Some("t".to_string()),
        task_id: None,
    };
    let write = |stem: &str, meta: TranscriptMeta, text: &str| {
        let message = crate::transcript::TranscriptMessage::user(text);
        locator
            .open_stem(stem, meta.clone())
            .unwrap()
            .append_turn(TranscriptTurn {
                prev: &[],
                next: &[message],
                meta: &meta,
                turn_usage: None,
                request_id: None,
                tools: None,
            })
            .unwrap();
    };
    write("old", meta("planner"), "old");
    write("new", meta("planner"), "new");
    write("new__child", meta("planner"), "child");
    let content = |found: Option<Arc<dyn TranscriptRead>>| {
        found.unwrap().read_session().unwrap().unwrap().messages[0]
            .content
            .clone()
    };
    assert_eq!(content(locator.latest_for_agent("planner")), "new");
    assert_eq!(content(locator.root_for_thread("t")), "new");
    assert!(locator.root_for_thread("  ").is_none());
    assert!(locator.latest_for_agent("nobody").is_none());
    // An unwritten probe is never a root.
    let _probe = locator
        .open_stem(
            &crate::transcript::session_stem(&SessionRef::root("p")),
            meta("planner"),
        )
        .unwrap();
    assert_eq!(content(locator.root_for_thread("t")), "new");
    // A partial goes to the newest root and stays out of the replay.
    assert!(
        locator
            .append_interrupted_partial("t", Some("planner"), &TranscriptPartial::new("par"), None)
            .unwrap()
    );
    assert!(
        !locator
            .append_interrupted_partial("t", Some("other"), &TranscriptPartial::new("x"), None)
            .unwrap()
    );
}

#[test]
fn probing_then_opening_a_session_replaces_placeholder_metadata() {
    let locator = InMemoryTranscriptLocator::new("test");
    let session = SessionRef::scoped("thread-meta", "planner");
    assert!(locator.read_session_transcript(&session).is_none());

    let meta = test_meta("thread-meta");
    locator
        .open_session(&session, meta.clone())
        .unwrap()
        .append(TranscriptMessage::user("hello"))
        .unwrap();

    let stored = locator
        .read_session_transcript(&session)
        .unwrap()
        .read_session()
        .unwrap()
        .unwrap()
        .meta;
    assert_eq!(stored.agent_name, meta.agent_name);
    assert_eq!(stored.agent_id, meta.agent_id);
    assert_eq!(stored.thread_id, meta.thread_id);
}

#[test]
fn an_unwritten_successor_is_reserved_until_written_or_dropped() {
    let locator = InMemoryTranscriptLocator::new("test");
    let session = SessionRef::scoped("thread-reservation", "planner");
    let (_, first) = locator
        .begin_generation(&session, test_meta("thread-reservation"))
        .unwrap();
    assert!(
        locator
            .begin_generation(&session, test_meta("thread-reservation"))
            .is_err()
    );
    drop(first);
    assert!(
        locator
            .begin_generation(&session, test_meta("thread-reservation"))
            .is_err()
    );
}

#[test]
fn completed_retention_uses_completion_time() {
    let turns = InMemoryTurnStates::default();
    for (request_id, started_at, updated_at) in [
        ("old-start", "2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"),
        ("new-start", "2026-01-01T02:00:00Z", "2026-01-01T00:30:00Z"),
    ] {
        let mut turn = TurnState::started("t", request_id, 4, started_at);
        turn.lifecycle = TurnLifecycle::Completed;
        turn.updated_at = updated_at.to_string();
        turns.put(&turn).unwrap();
    }
    let listed = turns.list_thread("t").unwrap();
    assert_eq!(listed[0].request_id, "new-start");
    assert_eq!(listed[1].request_id, "old-start");
}

fn write_rows(history: &dyn TranscriptHistory, meta: &TranscriptMeta, rows: &[&str]) {
    let rows: Vec<TranscriptMessage> = rows
        .iter()
        .map(|row| TranscriptMessage::user(*row))
        .collect();
    history
        .append_turn(TranscriptTurn {
            prev: &[],
            next: &rows,
            meta,
            turn_usage: None,
            request_id: None,
            tools: None,
        })
        .unwrap();
}

#[test]
fn a_thread_lists_every_generation_and_sub_agent() {
    let locator = InMemoryTranscriptLocator::new("listing");
    let session = SessionRef::scoped("t", "planner");
    let mut seed = test_meta("t");
    seed.session_id = Some(session.session_id());
    // A pre-identity root, folded into the session and dropped from the view.
    write_rows(
        locator
            .open_stem("1700000000_planner", test_meta("t"))
            .unwrap()
            .as_ref(),
        &test_meta("t"),
        &["legacy"],
    );
    write_rows(
        locator
            .open_session(&session, seed.clone())
            .unwrap()
            .as_ref(),
        &seed,
        &["one"],
    );
    let (successor, head) = locator.begin_generation(&session, seed.clone()).unwrap();
    write_rows(head.as_ref(), &seed, &["two"]);
    let root = crate::transcript::session_stem(&session);
    write_rows(
        locator
            .open_stem(&format!("{root}__1_child"), test_meta("worker"))
            .unwrap()
            .as_ref(),
        &test_meta("worker"),
        &["by prefix"],
    );
    write_rows(
        locator
            .open_stem("elsewhere__2_child", test_meta("t"))
            .unwrap()
            .as_ref(),
        &test_meta("t"),
        &["by thread"],
    );

    let listed = locator.thread_transcripts("t", None).expect("listed");
    let stems = |items: &[ThreadTranscript]| -> Vec<String> {
        items.iter().map(|item| item.stem.clone()).collect()
    };
    assert_eq!(
        stems(&listed.roots),
        vec![root.clone(), crate::transcript::session_stem(&successor)]
    );
    assert_eq!(
        stems(&listed.subagents),
        vec!["elsewhere__2_child".to_string(), format!("{root}__1_child")]
    );
    let scoped = locator.thread_transcripts("t", Some("planner")).unwrap();
    assert_eq!(stems(&scoped.subagents), vec![format!("{root}__1_child")]);
    assert!(locator.thread_transcripts("missing", None).is_none());
}

#[test]
fn the_display_read_shows_recorded_partials_where_they_were_written() {
    let history = InMemoryTranscriptHistory::new("display", test_meta("display"));
    write_rows(&history, &test_meta("display"), &["question"]);
    assert!(history.record_partial(
        TranscriptPartial::new("half an answer"),
        Some("req-1".into())
    ));

    let display = history.read_display().unwrap().expect("written");
    let rows: Vec<(String, bool)> = display
        .records
        .iter()
        .map(|record| match record {
            crate::transcript::DisplayRecord::Message(message) => {
                (message.message.content.clone(), message.interrupted)
            }
            crate::transcript::DisplayRecord::Compaction(_) => panic!("no compaction here"),
        })
        .collect();
    assert_eq!(
        rows,
        vec![
            ("question".to_string(), false),
            ("half an answer".to_string(), true)
        ]
    );
}
