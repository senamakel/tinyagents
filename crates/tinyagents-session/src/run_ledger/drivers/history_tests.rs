use super::history::message_fts_id;
use super::storage::{HIST_FTS, HIST_MESSAGES, HIST_SESSIONS, HIST_TOOL_CALLS};
use super::*;
use crate::run_ledger::history::{
    MAX_TOOL_OUTPUT_BYTES, NewSessionMessage, NewSessionToolCall, SessionEnd, SessionHistoryStore,
    SessionStart,
};
use crate::testkit::MemoryStorage;
use crate::types::{SessionSearchParams, SessionStatus};
use chrono::Duration;
use tinyagents_storage_kit::Precondition;
use tinystoragedrivers_api::StorageBackend;

fn ledger_on(backend: &dyn StorageBackend, scope: &str) -> DriverRunLedger {
    let scope = Scope::new(scope).unwrap();
    DriverRunLedger::new(ScopedStorage::new(
        backend.for_scope(&scope).unwrap(),
        scope,
    ))
}

fn start(id: &str, agent: &str, name: &str) -> SessionStart {
    SessionStart {
        id: id.into(),
        agent_definition_id: agent.into(),
        agent_definition_name: name.into(),
        session_key: format!("key-{id}"),
        ..Default::default()
    }
}

fn end(status: SessionStatus) -> SessionEnd {
    SessionEnd {
        status,
        turn_count: 3,
        input_tokens: 10,
        output_tokens: 20,
        cached_input_tokens: 5,
        cost_usd: 0.25,
    }
}

async fn say(ledger: &DriverRunLedger, session: &str, content: &str) -> i64 {
    ledger
        .record_message(NewSessionMessage {
            session_id: session.into(),
            role: "user".into(),
            content: content.into(),
            ..Default::default()
        })
        .await
        .unwrap()
}

async fn call(ledger: &DriverRunLedger, session: &str, tool: &str) -> i64 {
    ledger
        .record_tool_call(NewSessionToolCall {
            session_id: session.into(),
            tool_name: tool.into(),
            status: "ok".into(),
            ..Default::default()
        })
        .await
        .unwrap()
}

async fn search(ledger: &DriverRunLedger, query: &str) -> Vec<String> {
    ledger
        .search_sessions(&SessionSearchParams {
            query: Some(query.into()),
            ..Default::default()
        })
        .await
        .unwrap()
        .sessions
        .into_iter()
        .map(|session| session.id)
        .collect()
}

async fn records_and_queries(backend: &dyn StorageBackend) {
    let ledger = ledger_on(backend, "history");
    let opened = ledger
        .record_session_start(start("s1", "researcher", "Deep Researcher"))
        .await
        .unwrap();
    assert_eq!(opened.status, SessionStatus::Running);
    assert!(
        ledger
            .record_session_start(start("s1", "x", "x"))
            .await
            .is_err(),
        "a duplicate session id is refused"
    );
    let mut child = start("s2", "coder", "Coder");
    child.parent_session_id = Some("s1".into());
    child.thread_id = Some("thread".into());
    child.source_channel = Some("web".into());
    ledger.record_session_start(child).await.unwrap();

    // Message and tool-call ids are dense, per sequence, starting at one.
    assert_eq!(say(&ledger, "s1", "build the parser").await, 1);
    assert_eq!(say(&ledger, "s2", "refactor C++ code").await, 2);
    assert_eq!(call(&ledger, "s1", "grep").await, 1);
    assert_eq!(call(&ledger, "s2", "edit_file").await, 2);
    let missing = ledger
        .record_message(NewSessionMessage {
            session_id: "ghost".into(),
            ..Default::default()
        })
        .await;
    assert!(missing.is_err(), "messages need an existing session");

    let big = "é".repeat(MAX_TOOL_OUTPUT_BYTES);
    ledger
        .record_tool_call(NewSessionToolCall {
            session_id: "s1".into(),
            tool_name: "cat".into(),
            tool_output: Some(big),
            status: "ok".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    let calls = ledger.list_tool_calls("s1", None).await.unwrap();
    assert_eq!(calls.iter().map(|c| c.id).collect::<Vec<_>>(), [1, 3]);
    let output = calls[1].tool_output.as_deref().unwrap();
    assert!(output.ends_with("\n...[truncated]"));
    assert!(output.len() <= MAX_TOOL_OUTPUT_BYTES + "\n...[truncated]".len());

    let closed = ledger
        .record_session_end("s1", end(SessionStatus::Completed))
        .await
        .unwrap();
    assert_eq!(closed.status, SessionStatus::Completed);
    assert_eq!(closed.turn_count, 3);
    assert!(closed.ended_at.is_some());
    assert!(
        ledger
            .record_session_end("ghost", end(SessionStatus::Failed))
            .await
            .is_err()
    );
    assert_eq!(ledger.get_session("s1").await.unwrap().unwrap(), closed);
    assert!(ledger.get_session("ghost").await.unwrap().is_none());

    let messages = ledger.list_messages("s1", None).await.unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].content, "build the parser");
    let kids = ledger.list_children("s1").await.unwrap();
    assert_eq!(
        kids.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(),
        ["s2"]
    );

    // Newest first, with the total independent of the page.
    let page = ledger
        .list_sessions(Some(1), None, None, None)
        .await
        .unwrap();
    assert_eq!(page.total, 2);
    assert_eq!(page.sessions[0].id, "s2");
    let running = ledger
        .list_sessions(None, None, Some("running"), None)
        .await
        .unwrap();
    assert_eq!(running.total, 1);

    // Full-text: agent name, message text, tool name; terms are ANDed within a row.
    assert_eq!(search(&ledger, "researcher").await, ["s1"]);
    assert_eq!(search(&ledger, "parser").await, ["s1"]);
    assert_eq!(search(&ledger, "refactor").await, ["s2"]);
    assert_eq!(search(&ledger, "edit_file").await, ["s2"]);
    assert_eq!(
        search(&ledger, "refactor parser").await,
        Vec::<String>::new()
    );
    assert_eq!(search(&ledger, "nothing-here").await, Vec::<String>::new());
    // Structured filters compose with text.
    let by_tool = ledger
        .search_sessions(&SessionSearchParams {
            tool_name: Some("grep".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(by_tool.total, 1);
    assert_eq!(by_tool.sessions[0].id, "s1");
    let combined = ledger
        .search_sessions(&SessionSearchParams {
            query: Some("refactor".into()),
            source_channel: Some("web".into()),
            thread_id: Some("thread".into()),
            parent_session_id: Some("s1".into()),
            status: Some("running".into()),
            agent_id: Some("coder".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(combined.total, 1);
    let none = ledger
        .search_sessions(&SessionSearchParams {
            query: Some("refactor".into()),
            agent_id: Some("researcher".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(none.total, 0);
    // Pagination keeps the total.
    let paged = ledger
        .search_sessions(&SessionSearchParams {
            tool_name: Some("grep".into()),
            limit: Some(0),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!((paged.total, paged.sessions.len()), (1, 0));

    // A peer scope sees none of it.
    let peer = ledger_on(backend, "peer");
    assert!(peer.get_session("s1").await.unwrap().is_none());
    assert!(search(&peer, "parser").await.is_empty());

    assert_eq!(ledger.mark_interrupted().await.unwrap(), 1);
    assert_eq!(
        ledger.get_session("s2").await.unwrap().unwrap().status,
        SessionStatus::Interrupted
    );
    assert_eq!(ledger.mark_interrupted().await.unwrap(), 0);
}

#[tokio::test]
async fn memory_history_records_and_answers_queries() {
    records_and_queries(&MemoryStorage::new()).await;
}

#[tokio::test]
async fn reconstructed_memory_history_records_and_answers_queries() {
    let backend = MemoryStorage::new();
    records_and_queries(&backend).await;
}

/// Rewrite a session's timestamps so retention has something to age out.
async fn age_session(ledger: &DriverRunLedger, id: &str, ended: Option<chrono::DateTime<Utc>>) {
    let found = ledger.read_session(id).await.unwrap().unwrap();
    let mut session = found.doc;
    session.ended_at = ended;
    ledger
        .scoped
        .documents()
        .put(
            HIST_SESSIONS,
            &found.id,
            storage::encode(&session).unwrap(),
            Precondition::Version(found.version),
        )
        .await
        .unwrap();
}

async fn count(ledger: &DriverRunLedger, collection: &str) -> u64 {
    ledger
        .scoped
        .documents()
        .count(collection, &Filter::default())
        .await
        .unwrap()
}

async fn retention_removes_only_what_is_stale(backend: &dyn StorageBackend) {
    let ledger = ledger_on(backend, "retention");
    let cutoff = now();
    for id in ["old", "recent", "live"] {
        ledger
            .record_session_start(start(id, "agent", "Agent"))
            .await
            .unwrap();
        say(&ledger, id, &format!("hello from {id}")).await;
        call(&ledger, id, "tool").await;
    }
    for id in ["old", "recent"] {
        ledger
            .record_session_end(id, end(SessionStatus::Completed))
            .await
            .unwrap();
    }
    age_session(&ledger, "old", Some(cutoff - Duration::days(2))).await;
    age_session(&ledger, "recent", Some(cutoff + Duration::days(2))).await;
    // A running session is never stale, whatever its age: no `ended_at` at all.
    assert_eq!(count(&ledger, HIST_SESSIONS).await, 3);

    assert_eq!(ledger.prune_sessions_before(cutoff).await.unwrap(), 1);
    assert!(ledger.get_session("old").await.unwrap().is_none());
    assert!(ledger.list_messages("old", None).await.unwrap().is_empty());
    assert!(
        ledger
            .list_tool_calls("old", None)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(search(&ledger, "old").await.is_empty());
    assert_eq!(search(&ledger, "recent").await, ["recent"]);
    assert_eq!(search(&ledger, "live").await, ["live"]);
    assert_eq!(count(&ledger, HIST_MESSAGES).await, 2);
    assert_eq!(count(&ledger, HIST_TOOL_CALLS).await, 2);
    // 2 sessions + 2 messages + 2 tool calls remain indexed.
    assert_eq!(count(&ledger, HIST_FTS).await, 6);
    assert_eq!(ledger.prune_sessions_before(cutoff).await.unwrap(), 0);
}

#[tokio::test]
async fn memory_prune_sessions_cascades_to_children_and_index() {
    retention_removes_only_what_is_stale(&MemoryStorage::new()).await;
}

#[tokio::test]
async fn reconstructed_memory_prune_sessions_cascades_to_children_and_index() {
    let backend = MemoryStorage::new();
    retention_removes_only_what_is_stale(&backend).await;
}

#[tokio::test]
async fn trim_keeps_the_newest_messages_and_their_index_rows() {
    let backend = MemoryStorage::new();
    let ledger = ledger_on(&backend, "trim");
    ledger
        .record_session_start(start("s", "agent", "Agent"))
        .await
        .unwrap();
    ledger
        .record_session_start(start("other", "agent", "Agent"))
        .await
        .unwrap();
    for n in 1..=5 {
        say(&ledger, "s", &format!("message{n}")).await;
    }
    say(&ledger, "other", "bystander").await;
    assert_eq!(ledger.trim_session_messages("s", 2).await.unwrap(), 3);
    let kept = ledger.list_messages("s", None).await.unwrap();
    assert_eq!(
        kept.iter().map(|m| m.content.as_str()).collect::<Vec<_>>(),
        ["message4", "message5"]
    );
    // Trimmed text is no longer findable; kept text and other sessions are.
    assert!(search(&ledger, "message1").await.is_empty());
    assert_eq!(search(&ledger, "message5").await, ["s"]);
    assert_eq!(search(&ledger, "bystander").await, ["other"]);
    assert_eq!(ledger.list_messages("other", None).await.unwrap().len(), 1);
    assert_eq!(ledger.trim_session_messages("s", 10).await.unwrap(), 0);
    assert_eq!(ledger.trim_session_messages("s", 0).await.unwrap(), 2);
}

#[tokio::test]
async fn tool_call_and_event_pruning_respect_the_cutoff() {
    let backend = MemoryStorage::new();
    let ledger = ledger_on(&backend, "prune");
    ledger
        .record_session_start(start("s", "agent", "Agent"))
        .await
        .unwrap();
    call(&ledger, "s", "alpha").await;
    call(&ledger, "s", "beta").await;
    let after = now() + Duration::seconds(5);
    assert_eq!(
        ledger
            .prune_tool_calls_before(now() - Duration::days(1))
            .await
            .unwrap(),
        0
    );
    assert_eq!(ledger.prune_tool_calls_before(after).await.unwrap(), 2);
    assert!(ledger.list_tool_calls("s", None).await.unwrap().is_empty());
    assert_eq!(
        count(&ledger, HIST_FTS).await,
        1,
        "only the session row stays indexed"
    );

    for n in 0..3 {
        ledger
            .append_run_event(RunEventAppend {
                run_id: "run".into(),
                event_type: "tick".into(),
                payload: serde_json::json!({"n": n}),
            })
            .await
            .unwrap();
    }
    assert_eq!(
        ledger
            .prune_run_events_before(now() - Duration::days(1))
            .await
            .unwrap(),
        0
    );
    assert_eq!(ledger.prune_run_events_before(after).await.unwrap(), 3);
    // Pruning the head of a log never reuses a sequence number.
    let next = ledger
        .append_run_event(RunEventAppend {
            run_id: "run".into(),
            event_type: "tick".into(),
            payload: serde_json::json!({}),
        })
        .await
        .unwrap();
    assert_eq!(next.sequence, 4);
}

#[tokio::test]
async fn apply_retention_reports_every_family_it_removed() {
    let backend = MemoryStorage::new();
    let ledger = ledger_on(&backend, "all");
    ledger
        .record_session_start(start("old", "agent", "Agent"))
        .await
        .unwrap();
    say(&ledger, "old", "gone").await;
    call(&ledger, "old", "tool").await;
    ledger
        .record_session_end("old", end(SessionStatus::Failed))
        .await
        .unwrap();
    ledger
        .record_session_start(start("live", "agent", "Agent"))
        .await
        .unwrap();
    call(&ledger, "live", "tool").await;
    ledger
        .append_run_event(RunEventAppend {
            run_id: "run".into(),
            event_type: "tick".into(),
            payload: serde_json::json!({}),
        })
        .await
        .unwrap();
    ledger
        .upsert_run_telemetry(RunTelemetryUpsert {
            run_id: "run".into(),
            input_tokens: Some(1),
            ..Default::default()
        })
        .await
        .unwrap();
    let cutoff = now() + Duration::seconds(5);
    age_session(&ledger, "old", Some(now() - Duration::days(1))).await;
    let report = ledger.apply_retention(cutoff).await.unwrap();
    // The cascade removes `old`'s call with it; `live`'s call is only aged out.
    assert_eq!(
        report,
        RetentionReport {
            sessions: 1,
            messages: 0,
            tool_calls: 1,
            run_events: 1,
            run_telemetry: 1,
        }
    );
    assert_eq!(report.total(), 4);
    assert!(ledger.get_session("live").await.unwrap().is_some());
    assert_eq!(ledger.apply_retention(cutoff).await.unwrap().total(), 0);
}

#[tokio::test]
async fn reindex_restores_missing_rows_and_drops_stale_ones() {
    let backend = MemoryStorage::new();
    let ledger = ledger_on(&backend, "reindex");
    ledger
        .record_session_start(start("s", "agent", "Indexed Agent"))
        .await
        .unwrap();
    let id = say(&ledger, "s", "findable words").await;
    call(&ledger, "s", "probe").await;
    assert_eq!(search(&ledger, "findable").await, ["s"]);

    // Lose a message's index row (the gap databases written before atomic
    // pairs carried) and plant a row for a record that no longer exists.
    ledger
        .scoped
        .documents()
        .delete(HIST_FTS, &message_fts_id(id).unwrap(), Precondition::None)
        .await
        .unwrap();
    ledger
        .scoped
        .documents()
        .put(
            HIST_FTS,
            "stale-row",
            serde_json::json!({
                "session_id": "ghost", "agent_name": "", "content": "phantom", "tool_name": ""
            }),
            Precondition::None,
        )
        .await
        .unwrap();
    assert!(search(&ledger, "findable").await.is_empty());

    assert_eq!(ledger.reindex_fts().await.unwrap(), 3);
    assert_eq!(search(&ledger, "findable").await, ["s"]);
    assert_eq!(search(&ledger, "probe").await, ["s"]);
    assert!(search(&ledger, "phantom").await.is_empty());
    assert_eq!(count(&ledger, HIST_FTS).await, 3);
    // Idempotent.
    assert_eq!(ledger.reindex_fts().await.unwrap(), 3);
}

#[tokio::test]
async fn indexing_truncates_long_non_ascii_content_on_a_character_boundary() {
    let backend = MemoryStorage::new();
    let ledger = ledger_on(&backend, "unicode");
    ledger
        .record_session_start(start("s", "agent", "Agent"))
        .await
        .unwrap();
    // Two-byte characters straddle the default 2000-byte cap exactly.
    let content = format!("{} tail", "é".repeat(1500));
    say(&ledger, "s", &content).await;
    assert!(
        search(&ledger, "tail").await.is_empty(),
        "past the snippet cap"
    );
    assert_eq!(
        ledger.list_messages("s", None).await.unwrap()[0].content,
        content
    );
}
