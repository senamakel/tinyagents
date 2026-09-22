use super::*;
use crate::transcript::{FileTranscriptLocator, TranscriptLocator, read_transcript};
use tempfile::tempdir;

fn legacy_meta(created: &str, updated: &str, thread_id: &str) -> TranscriptMeta {
    TranscriptMeta {
        session_id: None,
        parent_session_id: None,
        agent_name: "orchestrator".into(),
        agent_id: Some("orchestrator".into()),
        agent_type: Some("root".into()),
        dispatcher: "native".into(),
        provider: None,
        model: Some("model".into()),
        created: created.into(),
        updated: updated.into(),
        turn_count: 1,
        input_tokens: 10,
        output_tokens: 5,
        cached_input_tokens: 2,
        charged_amount_usd: 0.5,
        thread_id: Some(thread_id.into()),
        task_id: None,
    }
}

fn write_legacy(dir: &Path, stem: &str, created: &str, body: &str, thread_id: &str) {
    let path = resolve_keyed_transcript_path(dir, stem).unwrap();
    write_transcript(
        &path,
        &[TranscriptMessage::new("user", body)],
        &legacy_meta(created, created, thread_id),
        None,
    )
    .unwrap();
}

/// The shape that cost a real user their thread: one conversation spread
/// across three timestamped stems, of which resume only ever saw the newest.
#[test]
fn legacy_roots_fold_into_one_session_in_created_order() {
    let dir = tempdir().unwrap();
    let thread = "thread-9fa08c44";
    write_legacy(
        dir.path(),
        "1790062247_orchestrator",
        "2026-09-22T07:30:47Z",
        "i want to plan a trip to kashmir",
        thread,
    );
    write_legacy(
        dir.path(),
        "1790100297_orchestrator",
        "2026-09-22T18:05:22Z",
        "hello?",
        thread,
    );
    write_legacy(
        dir.path(),
        "1790100352_orchestrator",
        "2026-09-22T18:06:27Z",
        "what did i ask here?",
        thread,
    );

    let session = SessionRef::scoped(thread, "orchestrator");
    let adoption = adopt_legacy_session_transcripts(
        dir.path(),
        &session,
        thread,
        &legacy_meta("", "", thread),
    )
    .unwrap()
    .expect("three legacy roots should be adopted");

    assert_eq!(adoption.adopted.len(), 3);
    assert_eq!(adoption.messages, 3);

    let adopted = read_transcript(&adoption.path).unwrap();
    let contents: Vec<&str> = adopted
        .messages
        .iter()
        .map(|message| message.content.as_str())
        .collect();
    assert_eq!(
        contents,
        [
            "i want to plan a trip to kashmir",
            "hello?",
            "what did i ask here?"
        ]
    );
    assert_eq!(adopted.meta.session_id, Some(session.session_id()));
    assert_eq!(adopted.meta.thread_id.as_deref(), Some(thread));
}

#[test]
fn adoption_sums_usage_and_spans_the_whole_conversation() {
    let dir = tempdir().unwrap();
    let thread = "thread-1";
    write_legacy(dir.path(), "1000_a", "2026-01-01T00:00:00Z", "one", thread);
    write_legacy(dir.path(), "2000_a", "2026-02-02T00:00:00Z", "two", thread);

    let session = SessionRef::scoped(thread, "orchestrator");
    let adoption =
        adopt_legacy_session_transcripts(dir.path(), &session, thread, &legacy_meta("", "", thread))
            .unwrap()
            .unwrap();

    let meta = read_transcript(&adoption.path).unwrap().meta;
    assert_eq!(meta.turn_count, 2);
    assert_eq!(meta.input_tokens, 20);
    assert_eq!(meta.output_tokens, 10);
    assert_eq!(meta.cached_input_tokens, 4);
    assert_eq!(meta.charged_amount_usd, 1.0);
    assert_eq!(meta.created, "2026-01-01T00:00:00Z");
    assert_eq!(meta.updated, "2026-02-02T00:00:00Z");
}

#[test]
fn adoption_never_touches_the_files_it_reads() {
    let dir = tempdir().unwrap();
    let thread = "thread-1";
    write_legacy(dir.path(), "1000_a", "2026-01-01T00:00:00Z", "one", thread);
    let legacy_path = resolve_keyed_transcript_path(dir.path(), "1000_a").unwrap();
    let before = std::fs::read(&legacy_path).unwrap();

    let session = SessionRef::scoped(thread, "orchestrator");
    adopt_legacy_session_transcripts(dir.path(), &session, thread, &legacy_meta("", "", thread))
        .unwrap()
        .unwrap();

    assert_eq!(std::fs::read(&legacy_path).unwrap(), before);
}

#[test]
fn adoption_is_idempotent_and_never_re_folds() {
    let dir = tempdir().unwrap();
    let thread = "thread-1";
    write_legacy(dir.path(), "1000_a", "2026-01-01T00:00:00Z", "one", thread);
    let session = SessionRef::scoped(thread, "orchestrator");
    let seed = legacy_meta("", "", thread);

    let first = adopt_legacy_session_transcripts(dir.path(), &session, thread, &seed)
        .unwrap()
        .unwrap();
    // The adopted transcript is itself a root matching this thread, so a
    // second pass must recognise the session as already backed rather than
    // folding the conversation into itself.
    let second = adopt_legacy_session_transcripts(dir.path(), &session, thread, &seed).unwrap();

    assert!(second.is_none());
    assert_eq!(read_transcript(&first.path).unwrap().messages.len(), 1);
}

#[test]
fn a_thread_with_no_legacy_roots_adopts_nothing() {
    let dir = tempdir().unwrap();
    let session = SessionRef::scoped("thread-fresh", "orchestrator");
    let result = adopt_legacy_session_transcripts(
        dir.path(),
        &session,
        "thread-fresh",
        &legacy_meta("", "", "thread-fresh"),
    )
    .unwrap();

    assert!(result.is_none());
}

#[test]
fn an_adopted_session_is_what_the_locator_then_resolves() {
    let dir = tempdir().unwrap();
    let thread = "thread-1";
    write_legacy(dir.path(), "1000_a", "2026-01-01T00:00:00Z", "one", thread);
    let session = SessionRef::scoped(thread, "orchestrator");
    adopt_legacy_session_transcripts(dir.path(), &session, thread, &legacy_meta("", "", thread))
        .unwrap()
        .unwrap();

    let locator = FileTranscriptLocator::new(dir.path());
    let read = locator
        .read_session_transcript(&session)
        .expect("the adopted transcript backs the session");
    assert_eq!(
        read.read_session().unwrap().unwrap().messages[0].content,
        "one"
    );
}
