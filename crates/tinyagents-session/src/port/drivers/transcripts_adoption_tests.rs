use super::*;
use crate::testkit::MemoryStorage;
use std::sync::Arc;

use crate::port::drivers::transcripts::tests::meta;
use crate::transcript::TranscriptMessage;
use tinyagents_storage_kit::{Blocking, DocumentStore, Scope};

const THREAD: &str = "t";

fn locator() -> DriverTranscriptLocator {
    let storage = MemoryStorage::new();
    let docs: Arc<dyn DocumentStore> = Arc::clone(
        storage
            .for_scope(&Scope::new("local").unwrap())
            .unwrap()
            .documents(),
    );
    DriverTranscriptLocator::new(docs, Blocking::new().unwrap(), "memory://adoption")
}

/// A root of `THREAD` under `stem`, written with `meta` (as an import does).
fn write_root(locator: &DriverTranscriptLocator, stem: &str, meta: &TranscriptMeta, rows: &[&str]) {
    let rows: Vec<TranscriptMessage> = rows
        .iter()
        .map(|row| TranscriptMessage::user(*row))
        .collect();
    locator
        .open_stem(stem, meta.clone())
        .unwrap()
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

fn legacy(agent: &str, created: &str, turns: usize, input: u64) -> TranscriptMeta {
    let mut meta = meta(THREAD);
    meta.agent_name = agent.into();
    meta.agent_id = Some(agent.into());
    meta.created = created.into();
    meta.updated = created.into();
    meta.turn_count = turns;
    meta.input_tokens = input;
    meta
}

fn contents(locator: &dyn TranscriptLocator, session: &SessionRef) -> Vec<String> {
    locator
        .read_session_transcript(session)
        .expect("the session has a transcript")
        .read_session()
        .unwrap()
        .unwrap()
        .messages
        .into_iter()
        .map(|message| message.content)
        .collect()
}

#[test]
fn legacy_roots_are_folded_oldest_first_once() {
    let locator = locator();
    // Written newest first: the fold orders by `_meta.created`, not by write.
    write_root(
        &locator,
        "1700000200_planner",
        &legacy("planner", "2026-01-02", 1, 7),
        &["two"],
    );
    write_root(
        &locator,
        "1700000100_planner",
        &legacy("planner", "2026-01-01", 2, 5),
        &["one"],
    );
    let session = SessionRef::scoped(THREAD, "planner");

    let adoption = locator
        .adopt_legacy(&session, THREAD, &meta(THREAD))
        .unwrap()
        .expect("two legacy roots to adopt");
    assert_eq!(adoption.messages, 2);
    assert_eq!(adoption.adopted.len(), 2);
    assert!(adoption.adopted[0].ends_with("1700000100_planner"));
    assert!(adoption.path.ends_with(session_stem(&session)));
    assert_eq!(contents(&locator, &session), vec!["one", "two"]);

    let folded = locator
        .read_session_transcript(&session)
        .unwrap()
        .read_session()
        .unwrap()
        .unwrap()
        .meta;
    assert_eq!(folded.session_id, Some(session.session_id()));
    assert_eq!(folded.thread_id.as_deref(), Some(THREAD));
    assert_eq!((folded.turn_count, folded.input_tokens), (3, 12));
    assert_eq!(
        (folded.created.as_str(), folded.updated.as_str()),
        ("2026-01-01", "2026-01-02")
    );

    assert!(
        locator
            .adopt_legacy(&session, THREAD, &meta(THREAD))
            .unwrap()
            .is_none(),
        "a second call finds the session already written"
    );
    assert_eq!(contents(&locator, &session), vec!["one", "two"]);
    // The legacy roots are left as they were.
    let untouched = locator
        .open_stem("1700000100_planner", meta(THREAD))
        .unwrap()
        .read_session()
        .unwrap()
        .unwrap();
    assert_eq!(untouched.messages.len(), 1);
    assert!(untouched.meta.session_id.is_none());
}

#[test]
fn session_identified_roots_and_other_agents_are_not_folded() {
    let locator = locator();
    write_root(
        &locator,
        "1700000100_planner",
        &legacy("planner", "2026-01-01", 1, 0),
        &["mine"],
    );
    write_root(
        &locator,
        "1700000150_writer",
        &legacy("writer", "2026-01-01", 1, 0),
        &["writer's"],
    );
    let other = SessionRef::scoped("another-key", "planner");
    let mut identified = legacy("planner", "2026-01-01", 1, 0);
    identified.session_id = Some(other.session_id());
    write_root(
        &locator,
        &session_stem(&other),
        &identified,
        &["already a session"],
    );

    let session = SessionRef::scoped(THREAD, "planner");
    let adoption = locator
        .adopt_legacy(&session, THREAD, &meta(THREAD))
        .unwrap()
        .unwrap();
    assert_eq!(adoption.adopted.len(), 1);
    assert_eq!(contents(&locator, &session), vec!["mine"]);
}

#[test]
fn nothing_to_fold_writes_nothing() {
    let locator = locator();
    let session = SessionRef::scoped(THREAD, "planner");
    assert!(
        locator
            .adopt_legacy(&session, THREAD, &meta(THREAD))
            .unwrap()
            .is_none()
    );
    write_root(
        &locator,
        "1700000100_writer",
        &legacy("writer", "2026-01-01", 1, 0),
        &["x"],
    );
    assert!(
        locator
            .adopt_legacy(&session, THREAD, &meta(THREAD))
            .unwrap()
            .is_none()
    );
    assert!(!locator.session_exists(&session));
}

#[test]
fn an_existing_session_is_left_alone() {
    let locator = locator();
    write_root(
        &locator,
        "1700000100_planner",
        &legacy("planner", "2026-01-01", 1, 0),
        &["old"],
    );
    let session = SessionRef::scoped(THREAD, "planner");
    let mut own = meta(THREAD);
    own.session_id = Some(session.session_id());
    write_root(&locator, &session_stem(&session), &own, &["new"]);
    assert!(
        locator
            .adopt_legacy(&session, THREAD, &meta(THREAD))
            .unwrap()
            .is_none()
    );
    assert_eq!(contents(&locator, &session), vec!["new"]);
}

#[test]
fn a_reserved_stem_reports_nothing_adopted() {
    let locator = locator();
    write_root(
        &locator,
        "1700000100_planner",
        &legacy("planner", "2026-01-01", 1, 0),
        &["old"],
    );
    let session = SessionRef::scoped(THREAD, "planner");
    // A concurrent writer holds the session's stem.
    let stem = session_stem(&session);
    let docs = Arc::clone(&locator.docs);
    let tables = Arc::clone(&locator.tables);
    let declared = Arc::clone(&locator.declared);
    run_on(&locator.bridge, async move {
        declared.ensure(&docs).await?;
        Ok(reservation::reserve(&docs, &tables.leases, &stem).await)
    })
    .unwrap()
    .unwrap();
    assert!(
        locator
            .adopt_legacy(&session, THREAD, &meta(THREAD))
            .unwrap()
            .is_none()
    );
}

#[test]
fn pre_identity_roots_are_adopted_and_resume_the_folded_history() {
    let locator = locator();
    write_root(
        &locator,
        "1700000100_orchestrator",
        &legacy("orchestrator", "2026-01-01", 1, 0),
        &["first"],
    );
    write_root(
        &locator,
        "1700000200_orchestrator",
        &legacy("orchestrator", "2026-01-02", 1, 0),
        &["second"],
    );
    let session = SessionRef::scoped(THREAD, "orchestrator");
    let seed = legacy("orchestrator", "2026-02-01", 0, 0);

    let adoption = locator
        .adopt_legacy(&session, THREAD, &seed)
        .unwrap()
        .expect("the pre-identity roots are adopted");
    assert_eq!(adoption.messages, 2);
    assert!(
        locator
            .adopt_legacy(&session, THREAD, &seed)
            .unwrap()
            .is_none()
    );
    assert!(
        locator
            .adopt_legacy(&session, THREAD, &seed)
            .unwrap()
            .is_none()
    );
    assert_eq!(contents(&locator, &session), vec!["first", "second"]);
}

#[test]
fn a_root_missing_from_the_index_is_still_folded() {
    let locator = locator();
    write_root(
        &locator,
        "1700000100_planner",
        &legacy("planner", "2026-01-01", 1, 0),
        &["one"],
    );
    write_root(
        &locator,
        "1700000200_planner",
        &legacy("planner", "2026-01-02", 1, 0),
        &["two"],
    );
    // The second write was durable but every index refresh after it failed.
    let docs = Arc::clone(&locator.docs);
    Blocking::new()
        .unwrap()
        .run(async move {
            docs.delete(
                super::super::INDEX,
                &super::super::index_id("1700000200_planner").unwrap(),
                tinystoragedrivers_types::Precondition::None,
            )
            .await
        })
        .unwrap()
        .unwrap();
    let session = SessionRef::scoped(THREAD, "planner");

    let adoption = locator
        .adopt_legacy(&session, THREAD, &meta(THREAD))
        .unwrap()
        .expect("both legacy roots to adopt");

    assert_eq!(adoption.adopted.len(), 2);
    assert_eq!(contents(&locator, &session), vec!["one", "two"]);
}
