use super::*;
use crate::run_ledger::legacy_import::LegacySessionImport;
use crate::run_ledger::tool_effects::{ToolEffectRow, ToolEffectStatus};
use crate::run_ledger::types::{
    AgentRun, AgentRunKind, AgentRunStatus, AgentTeam, AgentTeamMember, AgentTeamMemberStatus,
    AgentTeamStatus, AgentTeamTask, AgentTeamTaskStatus, RunTelemetry, WorkflowRun,
    WorkflowRunStatus,
};
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
    let counter = Counters::new(docs);
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

#[tokio::test]
async fn concurrent_identical_imports_reconcile_after_one_checked_commit() {
    let backend = Arc::new(MemoryStorage::new());
    let ledger = Arc::new(ledger(backend.as_ref()));
    let batch = fixture_batch();
    let (left, right) = tokio::join!(
        ledger.import_legacy_batch(batch.clone()),
        ledger.import_legacy_batch(batch),
    );
    let left = left.unwrap();
    let right = right.unwrap();
    assert_eq!(left.messages.imported + right.messages.imported, 1);
    assert_eq!(
        left.messages.already_present + right.messages.already_present,
        1
    );
    assert_eq!(left.tool_calls.imported + right.tool_calls.imported, 1);
    assert_eq!(left.events.imported + right.events.imported, 1);

    let counters = Counters::new(ledger.scoped.documents());
    assert_eq!(counters.peek("history", "messages").await.unwrap(), 41);
    assert_eq!(counters.peek("history", "tool_calls").await.unwrap(), 73);
}

#[tokio::test]
async fn imports_all_run_ledger_families_in_the_same_typed_batch() {
    let backend = Arc::new(MemoryStorage::new());
    let ledger = ledger(backend.as_ref());
    let at = instant(1_700_000_000_123);
    let report = ledger
        .import_legacy_batch(LegacySessionImport {
            agent_runs: vec![AgentRun {
                id: "agent-run".into(),
                kind: AgentRunKind::BackgroundAgent,
                parent_run_id: None,
                parent_thread_id: Some("thread".into()),
                agent_id: Some("agent".into()),
                status: AgentRunStatus::Completed,
                prompt_ref: Some("prompt-ref".into()),
                worker_thread_id: None,
                checkpoint_path: None,
                checkpoint: Some(json!({"offset":3})),
                summary: Some("done".into()),
                error: None,
                metadata: json!({"source":"legacy"}),
                telemetry: None,
                started_at: at,
                updated_at: at,
                completed_at: Some(at),
            }],
            telemetry: vec![RunTelemetry {
                run_id: "agent-run".into(),
                input_tokens: 5,
                output_tokens: 4,
                cached_input_tokens: 2,
                cost_usd: 0.03,
                elapsed_ms: Some(17),
                tool_count: 1,
                model: Some("model".into()),
                provider: Some("provider".into()),
                error: None,
                updated_at: Some(at),
            }],
            workflow_runs: vec![WorkflowRun {
                id: "workflow".into(),
                definition_id: "definition".into(),
                parent_thread_id: Some("thread".into()),
                input: json!({"key":"value"}),
                phase_states: json!({"plan":"done"}),
                child_run_ids: vec!["agent-run".into()],
                status: WorkflowRunStatus::Running,
                summary: Some("finished".into()),
                started_at: at,
                updated_at: at,
                completed_at: None,
                revision: 4,
                lease_owner: Some("legacy-driver".into()),
                lease_expires_at: Some(Utc::now() + chrono::Duration::minutes(5)),
            }],
            teams: vec![AgentTeam {
                id: "team".into(),
                parent_thread_id: Some("thread".into()),
                lead_agent_id: "agent".into(),
                status: AgentTeamStatus::Active,
                summary: Some("coordinating".into()),
                created_at: at,
                updated_at: at,
                closed_at: None,
            }],
            members: vec![AgentTeamMember {
                id: "member".into(),
                team_id: "team".into(),
                name: "worker".into(),
                agent_id: Some("agent".into()),
                member_status: AgentTeamMemberStatus::Idle,
                current_task_id: None,
                worker_thread_id: Some("worker-thread".into()),
                run_id: Some("agent-run".into()),
                created_at: at,
                updated_at: at,
            }],
            tasks: vec![AgentTeamTask {
                id: "task".into(),
                team_id: "team".into(),
                title: "work".into(),
                objective: Some("finish".into()),
                status: AgentTeamTaskStatus::Done,
                owner_member_id: Some("member".into()),
                claimed_by_member_id: None,
                claim_token: None,
                depends_on: vec![],
                gate_status: "passed".into(),
                gate_reason: None,
                evidence: vec!["proof".into()],
                source_run_id: Some("agent-run".into()),
                order_index: 2,
                created_at: at,
                updated_at: at,
            }],
            tool_effects: vec![ToolEffectRow {
                run_id: "agent-run".into(),
                call_id: "call".into(),
                tool: "search".into(),
                status: ToolEffectStatus::Completed,
                idempotency_key: Some("effect-key".into()),
                effect_summary: Some("one result".into()),
                started_at: at,
                settled_at: Some(at),
            }],
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(report.agent_runs.imported, 1);
    assert_eq!(report.telemetry.imported, 1);
    assert_eq!(report.workflow_runs.imported, 1);
    assert_eq!(report.teams.imported, 1);
    assert_eq!(report.members.imported, 1);
    assert_eq!(report.tasks.imported, 1);
    assert_eq!(report.tool_effects.imported, 1);

    let docs = ledger.scoped.documents();
    let team_id = super::storage::key(&["team"]).unwrap();
    let stored_team = docs
        .get(super::storage::TEAMS, &team_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stored_team.doc["lead_agent_id"], "agent");
    assert_eq!(stored_team.doc["created_ms"], at.timestamp_millis());
    let member_id = super::storage::key(&["member"]).unwrap();
    let stored_member = docs
        .get(super::storage::MEMBERS, &member_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stored_member.doc["worker_thread_id"], "worker-thread");
    let effect_id = super::storage::key(&["agent-run", "call"]).unwrap();
    let stored_effect = docs
        .get(super::storage::EFFECTS, &effect_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stored_effect.doc["idempotency_key"], "effect-key");
    let lease_id = super::storage::key(&[
        super::storage::LEDGER_OWNER,
        &super::storage::workflow_lease_key("workflow").unwrap(),
    ])
    .unwrap();
    let lease = docs
        .get(tinyagents_storage_kit::primitives::LEASES, &lease_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(lease.doc["holder"], "legacy-driver");
}
