//! Importer tests over classic databases built from the verbatim classic schema.

use super::fixtures::classic_database;
use super::*;
use crate::run_ledger::{DriverRunLedger, RunEventAppend, RunLedgerStore, SessionHistoryStore};
use crate::types::SessionSearchParams;
use std::path::Path;
use tinystoragedrivers_core::{MemoryStorage, Scope, StorageBackend};
use tinystoragedrivers_sqlite::rusqlite::{Connection, params};

const T0: &str = "2026-01-01T00:00:00+00:00";
const T1: &str = "2026-01-02T00:00:00+00:00";

fn ledger_on(backend: &dyn StorageBackend, scope: &str) -> DriverRunLedger {
    DriverRunLedger::new(backend.for_scope(&Scope::new(scope).unwrap()).unwrap())
}

fn source_for(path: &Path, ledger: &DriverRunLedger) -> QuiescentSource {
    QuiescentSource::asserted_by_trusted_host(
        path,
        SourceOwner::asserted_by_trusted_host(ledger.scope().clone()),
    )
}

/// One of everything, with deliberate gaps in the dense ids.
fn populate(db: &Connection) {
    db.execute_batch(&format!(
        "INSERT INTO sessions (id, agent_definition_id, agent_definition_name, session_key,
                               parent_session_id, thread_id, source_channel, status, model,
                               turn_count, input_tokens, output_tokens, cached_input_tokens,
                               cost_usd, transcript_path, started_at, ended_at)
         VALUES ('s-root', 'researcher', 'Deep Researcher', 'key-root', NULL, 'thread-1', 'web',
                 'completed', 'gpt-5', 3, 100, 200, 50, 0.5, '/t/root.jsonl', '{T0}', '{T1}'),
                ('s-child', 'coder', 'Coder', 'key-child', 's-root', 'thread-1', NULL,
                 'running', NULL, 0, 0, 0, 0, 0.0, NULL, '{T1}', NULL);
         INSERT INTO session_messages (id, session_id, role, content, reasoning_content, model,
                                       input_tokens, output_tokens, cost_usd, created_at)
         VALUES (1, 's-root', 'user', 'summarize the parser design', NULL, NULL, NULL, NULL, NULL, '{T0}'),
                (2, 's-root', 'assistant', 'the parser is recursive descent', 'thinking hard', 'gpt-5', 10, 20, 0.01, '{T0}'),
                (5, 's-child', 'user', 'refactor the lexer', NULL, NULL, NULL, NULL, NULL, '{T1}');
         INSERT INTO session_tool_calls (id, session_id, message_id, tool_name, tool_input,
                                         tool_output, status, duration_ms, created_at)
         VALUES (3, 's-root', 2, 'grep', '{{\"q\":1}}', 'match', 'ok', 12, '{T0}'),
                (7, 's-child', NULL, 'edit_file', NULL, NULL, 'pending', NULL, '{T1}');
         INSERT INTO agent_runs (id, kind, parent_run_id, parent_thread_id, agent_id, status,
                                 prompt_ref, worker_thread_id, checkpoint_path, checkpoint_json,
                                 summary, error, metadata_json, started_at, updated_at, completed_at)
         VALUES ('run-1', 'subagent', NULL, 'thread-1', 'coder', 'completed', 'p', 'w', NULL,
                 '{{\"step\":2}}', 'did it', NULL, '{{\"k\":\"v\"}}', '{T0}', '{T1}', '{T1}');
         INSERT INTO run_telemetry (run_id, input_tokens, output_tokens, cached_input_tokens,
                                    cost_usd, elapsed_ms, tool_count, model, provider, error, updated_at)
         VALUES ('run-1', 7, 8, 1, 0.25, 900, 2, 'gpt-5', 'openai', NULL, '{T1}');
         INSERT INTO run_events (run_id, sequence, event_type, payload_json, timestamp)
         VALUES ('run-1', 3, 'tick', '{{\"n\":3}}', '{T0}'),
                ('run-1', 4, 'tick', '{{\"n\":4}}', '{T1}');
         INSERT INTO workflow_runs (id, definition_id, parent_thread_id, input_json,
                                    phase_states_json, child_run_ids_json, status, summary,
                                    started_at, updated_at, completed_at, revision, lease_owner,
                                    lease_expires_at)
         VALUES ('wf-1', 'def', 'thread-1', '{{\"in\":1}}', '{{}}', '[\"run-1\"]', 'running', NULL,
                 '{T0}', '{T1}', NULL, 4, 'driver-a', '2999-01-01T00:00:00+00:00');
         INSERT INTO agent_teams (id, parent_thread_id, lead_agent_id, status, summary,
                                  created_at, updated_at, closed_at)
         VALUES ('team-1', 'thread-1', 'lead', 'active', NULL, '{T0}', '{T1}', NULL);
         INSERT INTO agent_team_members (id, team_id, name, agent_id, member_status,
                                         current_task_id, worker_thread_id, run_id, created_at,
                                         updated_at)
         VALUES ('m-1', 'team-1', 'alice', NULL, 'active', 'task-1', 'w', 'run-1', '{T0}', '{T1}');
         INSERT INTO agent_team_tasks (id, team_id, title, objective, status, owner_member_id,
                                       claimed_by_member_id, claim_token, depends_on_json,
                                       gate_status, gate_reason, evidence_json, source_run_id,
                                       order_index, created_at, updated_at)
         VALUES ('task-1', 'team-1', 'build', NULL, 'in_progress', NULL, 'm-1', 'tok', '[]',
                 'pending', NULL, '[\"pr#1\"]', NULL, 2, '{T0}', '{T1}');
         INSERT INTO tool_effects (run_id, call_id, tool, status, idempotency_key, effect_summary,
                                   started_at, settled_at)
         VALUES ('run-1', 'call-1', 'send_email', 'started', 'idem-1', 'to bob', '{T0}', NULL);"
    ))
    .expect("populate classic database");
}

fn classic(dir: &Path) -> std::path::PathBuf {
    let path = dir.join("sessions.db");
    classic_database(&path, 7, populate);
    path
}

/// Adds retired entry-table names with sentinel rows to an existing classic database.
fn add_retired_tables(path: &Path) {
    let entries = ["entry", "_tree", "_entries"].concat();
    let labels = ["entry", "_tree", "_labels"].concat();
    let sql = format!(
        "CREATE TABLE {entries} (id TEXT); CREATE TABLE {labels} (id TEXT); \
         INSERT INTO {entries} VALUES ('kept-entry'); \
         INSERT INTO {labels} VALUES ('kept-label');"
    );
    Connection::open(path).unwrap().execute_batch(&sql).unwrap();
}

fn directory_bytes(dir: &Path) -> Vec<(std::path::PathBuf, Vec<u8>)> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| {
            let path = entry.unwrap().path();
            let bytes = std::fs::read(&path).unwrap();
            (path, bytes)
        })
        .collect();
    entries.sort();
    entries
}

#[tokio::test]
async fn a_full_database_imports_every_family_with_its_ids() {
    let dir = tempfile::tempdir().unwrap();
    let path = classic(dir.path());
    let before = directory_bytes(dir.path());
    let backend = MemoryStorage::new();
    let ledger = ledger_on(&backend, "agent");

    let report = ledger
        .import_legacy_database(&source_for(&path, &ledger))
        .await
        .unwrap();
    assert_eq!(
        report.sessions,
        ImportCounts {
            read: 2,
            imported: 2,
            already_present: 0
        }
    );
    assert_eq!(report.messages.imported, 3);
    assert_eq!(report.tool_calls.imported, 2);
    assert_eq!(report.agent_runs.imported, 1);
    assert_eq!(report.telemetry.imported, 1);
    assert_eq!(report.events.imported, 2);
    assert_eq!(report.workflow_runs.imported, 1);
    assert_eq!(report.teams.imported, 1);
    assert_eq!(report.members.imported, 1);
    assert_eq!(report.tasks.imported, 1);
    assert_eq!(report.tool_effects.imported, 1);
    // 2 sessions + 3 messages + 2 tool calls.
    assert_eq!(report.fts_rows, 7);

    // The source is untouched, down to the directory listing.
    assert_eq!(directory_bytes(dir.path()), before);

    // Session history keeps ids, lineage and every field.
    let root = ledger.get_session("s-root").await.unwrap().unwrap();
    assert_eq!(root.agent_definition_name, "Deep Researcher");
    assert_eq!(root.cached_input_tokens, 50);
    assert_eq!(root.transcript_path.as_deref(), Some("/t/root.jsonl"));
    assert!(root.ended_at.is_some());
    let kids = ledger.list_children("s-root").await.unwrap();
    assert_eq!(kids.len(), 1);
    assert_eq!(kids[0].id, "s-child");
    let messages = ledger.list_messages("s-root", None).await.unwrap();
    assert_eq!(messages.iter().map(|m| m.id).collect::<Vec<_>>(), [1, 2]);
    assert_eq!(
        messages[1].reasoning_content.as_deref(),
        Some("thinking hard")
    );
    assert_eq!(messages[1].model.as_deref(), Some("gpt-5"));
    let calls = ledger.list_tool_calls("s-root", None).await.unwrap();
    assert_eq!((calls[0].id, calls[0].message_id), (3, Some(2)));

    // The rebuilt full-text index answers like the classic one.
    let found = ledger
        .search_sessions(&SessionSearchParams {
            query: Some("lexer".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(
        found
            .sessions
            .iter()
            .map(|s| s.id.as_str())
            .collect::<Vec<_>>(),
        ["s-child"]
    );
    let by_tool = ledger
        .search_sessions(&SessionSearchParams {
            tool_name: Some("grep".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(by_tool.total, 1);

    // Dense ids continue past the highest imported id, not from one.
    let next_message = ledger
        .record_message(crate::run_ledger::NewSessionMessage {
            session_id: "s-root".into(),
            role: "user".into(),
            content: "new".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(next_message, 6);
    let next_call = ledger
        .record_tool_call(crate::run_ledger::NewSessionToolCall {
            session_id: "s-root".into(),
            tool_name: "t".into(),
            status: "ok".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(next_call, 8);

    // Run ledger families.
    let run = ledger.get_agent_run("run-1").await.unwrap().unwrap();
    assert_eq!(run.summary.as_deref(), Some("did it"));
    assert_eq!(run.checkpoint, Some(serde_json::json!({"step": 2})));
    assert_eq!(run.telemetry.as_ref().map(|t| t.output_tokens), Some(8));
    let events = ledger
        .list_recent_run_events(&crate::run_ledger::RunEventListRequest {
            run_id: "run-1".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(
        events.events.iter().map(|e| e.sequence).collect::<Vec<_>>(),
        [3, 4]
    );
    // The event counter continues past the imported sequences.
    let appended = ledger
        .append_run_event(RunEventAppend {
            run_id: "run-1".into(),
            event_type: "tick".into(),
            payload: serde_json::json!({}),
        })
        .await
        .unwrap();
    assert_eq!(appended.sequence, 5);
    let workflow = ledger.get_workflow_run("wf-1").await.unwrap().unwrap();
    assert_eq!(workflow.revision, 4);
    assert_eq!(workflow.lease_owner.as_deref(), Some("driver-a"));
    assert_eq!(workflow.child_run_ids, ["run-1"]);
    let tasks = ledger.list_agent_team_tasks("team-1").await.unwrap();
    assert_eq!(tasks[0].claimed_by_member_id.as_deref(), Some("m-1"));
    assert_eq!(tasks[0].evidence, ["pr#1"]);
    assert_eq!(
        ledger
            .list_agent_team_members("team-1")
            .await
            .unwrap()
            .len(),
        1
    );
    let unresolved = ledger.list_unresolved_tool_effects("run-1").await.unwrap();
    assert_eq!(unresolved.len(), 1);
    assert_eq!(unresolved[0].idempotency_key.as_deref(), Some("idem-1"));
}

/// Importing a classic database leaves rows in retired tables untouched.
#[tokio::test]
async fn importing_a_store_with_retired_tables_leaves_them_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let path = classic(dir.path());
    add_retired_tables(&path);
    let backend = MemoryStorage::new();
    let ledger = ledger_on(&backend, "agent");

    let report = ledger
        .import_legacy_database(&source_for(&path, &ledger))
        .await
        .unwrap();

    assert_eq!(report.sessions.imported, 2);
    let db = Connection::open(&path).unwrap();
    let entries = ["entry", "_tree", "_entries"].concat();
    let labels = ["entry", "_tree", "_labels"].concat();
    assert_eq!(
        db.query_row(&format!("SELECT id FROM {entries}"), [], |row| row
            .get::<_, String>(0))
            .unwrap(),
        "kept-entry"
    );
    assert_eq!(
        db.query_row(&format!("SELECT id FROM {labels}"), [], |row| row
            .get::<_, String>(0))
            .unwrap(),
        "kept-label"
    );
}

#[tokio::test]
async fn a_second_run_changes_nothing_and_never_overwrites_edits() {
    let dir = tempfile::tempdir().unwrap();
    let path = classic(dir.path());
    let backend = MemoryStorage::new();
    let ledger = ledger_on(&backend, "agent");
    let source = source_for(&path, &ledger);
    ledger.import_legacy_database(&source).await.unwrap();

    // The target moves on: a session ends differently and a new message lands.
    ledger
        .record_session_end(
            "s-child",
            crate::run_ledger::SessionEnd {
                status: crate::types::SessionStatus::Failed,
                turn_count: 9,
                input_tokens: 1,
                output_tokens: 1,
                cached_input_tokens: 0,
                cost_usd: 1.0,
            },
        )
        .await
        .unwrap();
    ledger
        .transition_agent_run_status(
            "run-1",
            crate::run_ledger::AgentRunStatus::Failed,
            Some("x"),
            None,
        )
        .await
        .unwrap();

    let again = ledger.import_legacy_database(&source).await.unwrap();
    for counts in [
        again.sessions,
        again.messages,
        again.tool_calls,
        again.agent_runs,
        again.telemetry,
        again.events,
        again.workflow_runs,
        again.teams,
        again.members,
        again.tasks,
        again.tool_effects,
    ] {
        assert_eq!(counts.imported, 0, "{counts:?}");
        assert_eq!(counts.already_present, counts.read);
    }
    let child = ledger.get_session("s-child").await.unwrap().unwrap();
    assert_eq!(
        child.status,
        crate::types::SessionStatus::Failed,
        "edit survives"
    );
    assert_eq!(child.turn_count, 9);
    let run = ledger.get_agent_run("run-1").await.unwrap().unwrap();
    assert_eq!(run.status, crate::run_ledger::AgentRunStatus::Failed);
}

#[tokio::test]
async fn an_interrupted_pass_is_finished_by_running_again() {
    let dir = tempfile::tempdir().unwrap();
    let path = classic(dir.path());
    let backend = MemoryStorage::new();
    let ledger = ledger_on(&backend, "agent");
    // Part of the data is already there, as after a pass that stopped early.
    ledger
        .record_session_start(crate::run_ledger::SessionStart {
            id: "s-root".into(),
            agent_definition_id: "researcher".into(),
            agent_definition_name: "Deep Researcher".into(),
            session_key: "key-root".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    let report = ledger
        .import_legacy_database(&source_for(&path, &ledger))
        .await
        .unwrap();
    assert_eq!(
        report.sessions,
        ImportCounts {
            read: 2,
            imported: 1,
            already_present: 1
        }
    );
    assert_eq!(report.messages.imported, 3);
}

#[tokio::test]
async fn an_old_schema_without_later_columns_imports_with_defaults() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("old.db");
    // Version 2: no reasoning_content, workflow leases or tool effects.
    classic_database(&path, 2, |db| {
        db.execute_batch(&format!(
            "INSERT INTO sessions (id, agent_definition_id, agent_definition_name, session_key,
                                   status, started_at)
             VALUES ('old', 'a', 'Old Agent', 'k', 'completed', '{T0}');
             INSERT INTO session_messages (id, session_id, role, content, created_at)
             VALUES (1, 'old', 'user', 'legacy words', '{T0}');
             INSERT INTO workflow_runs (id, definition_id, status, started_at, updated_at)
             VALUES ('wf', 'd', 'running', '{T0}', '{T0}');"
        ))
        .unwrap();
    });
    let backend = MemoryStorage::new();
    let ledger = ledger_on(&backend, "agent");
    let report = ledger
        .import_legacy_database(&source_for(&path, &ledger))
        .await
        .unwrap();
    assert_eq!(report.messages.imported, 1);
    assert_eq!(report.workflow_runs.imported, 1);
    let message = &ledger.list_messages("old", None).await.unwrap()[0];
    assert_eq!(message.reasoning_content, None);
    let workflow = ledger.get_workflow_run("wf").await.unwrap().unwrap();
    assert_eq!((workflow.revision, workflow.lease_owner), (0, None));
    assert_eq!(
        ledger
            .search_sessions(&SessionSearchParams {
                query: Some("legacy".into()),
                ..Default::default()
            })
            .await
            .unwrap()
            .total,
        1
    );
}

#[tokio::test]
async fn a_source_must_belong_to_the_target_scope_and_be_checkpointed() {
    let dir = tempfile::tempdir().unwrap();
    let path = classic(dir.path());
    let backend = MemoryStorage::new();
    let ledger = ledger_on(&backend, "agent");
    let foreign = QuiescentSource::asserted_by_trusted_host(
        &path,
        SourceOwner::asserted_by_trusted_host(Scope::new("someone-else").unwrap()),
    );
    let error = ledger.import_legacy_database(&foreign).await.unwrap_err();
    assert!(
        matches!(error, crate::run_ledger::LedgerError::Driver(ref e)
        if e.kind() == tinystoragedrivers_core::ErrorKind::InvalidInput)
    );

    // A database with committed data still in its WAL is refused, never guessed at.
    let wal = dir.path().join("wal.db");
    let conn = Connection::open(&wal).unwrap();
    conn.execute_batch(
        "PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0;
         CREATE TABLE sessions (id TEXT); INSERT INTO sessions VALUES ('in-wal');",
    )
    .unwrap();
    let error = ledger
        .import_legacy_database(&source_for(&wal, &ledger))
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        crate::run_ledger::LedgerError::SourceRequiresSnapshot { .. }
    ));
    drop(conn);

    // Something that is not a classic database is refused up front.
    let other = dir.path().join("other.db");
    Connection::open(&other)
        .unwrap()
        .execute_batch("CREATE TABLE unrelated (x INTEGER);")
        .unwrap();
    assert!(
        ledger
            .import_legacy_database(&source_for(&other, &ledger))
            .await
            .is_err()
    );
    // And nothing reached the target.
    assert!(ledger.get_session("s-root").await.unwrap().is_none());
}

#[tokio::test]
async fn a_row_that_does_not_decode_refuses_the_pass_before_any_write() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.db");
    classic_database(&path, 7, |db| {
        populate(db);
        // An unknown run kind: the classic parser would silently remap it.
        db.execute(
            "INSERT INTO agent_runs (id, kind, status, metadata_json, started_at, updated_at)
             VALUES ('weird', 'not-a-kind', 'running', '{}', ?1, ?1)",
            params![T0],
        )
        .unwrap();
    });
    let backend = MemoryStorage::new();
    let ledger = ledger_on(&backend, "agent");
    assert!(
        ledger
            .import_legacy_database(&source_for(&path, &ledger))
            .await
            .is_err()
    );
    assert!(
        ledger.get_session("s-root").await.unwrap().is_none(),
        "everything is read before the first write"
    );
}

#[tokio::test]
async fn the_import_lands_in_the_scope_of_the_target_only() {
    let dir = tempfile::tempdir().unwrap();
    let path = classic(dir.path());
    let backend = MemoryStorage::new();
    let ledger = ledger_on(&backend, "agent");
    let peer = ledger_on(&backend, "peer");
    ledger
        .import_legacy_database(&source_for(&path, &ledger))
        .await
        .unwrap();
    assert!(peer.get_session("s-root").await.unwrap().is_none());
    assert!(peer.get_agent_run("run-1").await.unwrap().is_none());
}

#[tokio::test]
async fn the_import_survives_a_reopened_sqlite_target() {
    let dir = tempfile::tempdir().unwrap();
    let path = classic(dir.path());
    let target = dir.path().join("target.sqlite");
    {
        let backend = tinystoragedrivers_sqlite::SqliteStorage::open(&target).unwrap();
        let ledger = ledger_on(&backend, "agent");
        ledger
            .import_legacy_database(&source_for(&path, &ledger))
            .await
            .unwrap();
    }
    let backend = tinystoragedrivers_sqlite::SqliteStorage::open(&target).unwrap();
    let ledger = ledger_on(&backend, "agent");
    assert_eq!(ledger.list_messages("s-root", None).await.unwrap().len(), 2);
    assert_eq!(
        ledger
            .search_sessions(&SessionSearchParams {
                query: Some("parser".into()),
                ..Default::default()
            })
            .await
            .unwrap()
            .total,
        1
    );
}
