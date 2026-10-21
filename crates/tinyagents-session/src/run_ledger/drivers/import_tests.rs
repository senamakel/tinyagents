use super::*;
use crate::run_ledger::legacy_import::LegacySessionImport;
use crate::testkit::MemoryStorage;
use crate::types::{SessionMessage, SessionRecord, SessionStatus, SessionToolCall};
use chrono::{DateTime, Utc};
use serde_json::json;
use std::sync::Arc;
use tinyagents_storage_kit::primitives::Counters;
use tinyagents_storage_kit::{Scope, ScopedStorage};
use tinystoragedrivers_api::StorageBackend;

fn instant(ms: i64) -> DateTime<Utc> {
    DateTime::from_timestamp_millis(ms).unwrap()
}

fn ledger(backend: &dyn StorageBackend) -> DriverRunLedger {
    let scope = Scope::new("legacy-import").unwrap();
    DriverRunLedger::new(ScopedStorage::new(
        backend.for_scope(&scope).unwrap(),
        scope,
    ))
}

fn fixture_batch() -> LegacySessionImport {
    LegacySessionImport {
        sessions: vec![SessionRecord {
            id: "session-legacy".into(),
            agent_definition_id: "agent".into(),
            agent_definition_name: "Legacy agent".into(),
            session_key: "stable-session-key".into(),
            parent_session_id: None,
            thread_id: Some("thread".into()),
            source_channel: Some("desktop".into()),
            status: SessionStatus::Completed,
            model: Some("model".into()),
            turn_count: 3,
            input_tokens: 17,
            output_tokens: 9,
            cached_input_tokens: 4,
            cost_usd: 0.125,
            transcript_path: Some("sessions/session.jsonl".into()),
            started_at: instant(1_700_000_000_123),
            ended_at: Some(instant(1_700_000_001_456)),
        }],
        messages: vec![SessionMessage {
            id: 41,
            session_id: "session-legacy".into(),
            role: "assistant".into(),
            content: "preserved body".into(),
            reasoning_content: Some("private reasoning".into()),
            model: Some("model".into()),
            input_tokens: Some(11),
            output_tokens: Some(7),
            cost_usd: Some(0.02),
            created_at: instant(1_700_000_000_789),
        }],
        tool_calls: vec![SessionToolCall {
            id: 73,
            session_id: "session-legacy".into(),
            message_id: Some(41),
            tool_name: "search".into(),
            tool_input: Some("{}".into()),
            tool_output: Some("found".into()),
            status: "ok".into(),
            duration_ms: Some(19),
            created_at: instant(1_700_000_000_900),
        }],
        events: vec![RunEvent {
            run_id: "run-legacy".into(),
            sequence: 8,
            event_type: "phase_started".into(),
            payload: json!({"phase":"prepare"}),
            timestamp: instant(1_700_000_001_002),
        }],
        ..Default::default()
    }
}

#[tokio::test]
async fn imports_preserve_explicit_history_ids_events_and_advance_counters_atomically() {
    let backend = Arc::new(MemoryStorage::new());
    let ledger = ledger(backend.as_ref());
    let batch = fixture_batch();

    let first = ledger.import_legacy_batch(batch.clone()).await.unwrap();
    assert_eq!(first.messages.imported, 1);
    assert_eq!(first.tool_calls.imported, 1);
    assert_eq!(first.events.imported, 1);
    assert!(first.fts_rows >= 3);

    let second = ledger.import_legacy_batch(batch).await.unwrap();
    assert_eq!(second.messages.imported, 0);
    assert_eq!(second.messages.already_present, 1);
    assert_eq!(second.tool_calls.imported, 0);
    assert_eq!(second.events.imported, 0);
    assert_eq!(second.fts_rows, 0);

    let scope = Scope::new("legacy-import").unwrap();
    let docs = backend.for_scope(&scope).unwrap().documents().clone();
    let history_counters = Counters::new(&docs);
    assert_eq!(
        history_counters.peek("history", "messages").await.unwrap(),
        41
    );
    assert_eq!(
        history_counters
            .peek("history", "tool_calls")
            .await
            .unwrap(),
        73
    );
    assert_eq!(
        history_counters.next("history", "messages").await.unwrap(),
        42,
        "a future append must not reuse an imported message id",
    );
    assert_eq!(
        history_counters
            .next("history", "tool_calls")
            .await
            .unwrap(),
        74,
        "a future append must not reuse an imported tool-call id",
    );

    let events = ledger
        .events(&RunEventListRequest {
            run_id: "run-legacy".into(),
            after_sequence: None,
            limit: Some(10),
        })
        .await
        .unwrap();
    assert_eq!(events.events[0].sequence, 8);
    assert_eq!(events.events[0].payload, json!({"phase":"prepare"}));
}

#[tokio::test]
async fn conflicting_existing_row_refuses_the_whole_import_before_counters_advance() {
    let backend = Arc::new(MemoryStorage::new());
    let ledger = ledger(backend.as_ref());
    let mut batch = fixture_batch();
    let mut existing = batch.sessions[0].clone();
    existing.agent_definition_name = "different".into();
    let docs = ledger.scoped.documents();
    docs.ensure_collection(&tinyagents_storage_kit::CollectionSpec::new(
        super::storage::HIST_SESSIONS,
    ))
    .await
    .unwrap();
    docs.put(
        super::storage::HIST_SESSIONS,
        &super::storage::key(&[&existing.id]).unwrap(),
        super::storage::encode(&existing).unwrap(),
        tinyagents_storage_kit::Precondition::Absent,
    )
    .await
    .unwrap();

    batch.messages[0].id = 99;
    assert!(ledger.import_legacy_batch(batch).await.is_err());
    let counter = Counters::new(&docs);
    assert_eq!(counter.peek("history", "messages").await.unwrap(), 0);
    assert_eq!(counter.peek("history", "tool_calls").await.unwrap(), 0);
    assert!(
        docs.get(
            super::storage::HIST_MESSAGES,
            &super::history::message_id(99).unwrap(),
        )
        .await
        .unwrap()
        .is_none()
    );
}
