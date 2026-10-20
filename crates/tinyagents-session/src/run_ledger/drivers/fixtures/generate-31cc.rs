//! Generates the checked-in session fixture using the 31cc public writers.

use std::path::PathBuf;

use tinyagents_session::run_ledger::{
    DriverRunLedger, NewSessionMessage, RunLedgerStore, SessionHistoryStore, SessionStart,
    WorkflowRunStatus, WorkflowRunUpsert,
};
use tinystoragedrivers_core::{MemoryStorage, Query, Scope, StorageBackend};

fn main() {
    let output = PathBuf::from(std::env::args().nth(1).expect("output path"));
    let backend = MemoryStorage::new();
    let scoped = backend
        .for_scope(&Scope::new("legacy-session-writer").unwrap())
        .unwrap();
    let ledger = DriverRunLedger::new(scoped.clone());
    let runtime = tokio::runtime::Runtime::new().unwrap();

    runtime.block_on(async {
        ledger
            .upsert_workflow_run(WorkflowRunUpsert {
                id: "fixture-workflow".into(),
                definition_id: "fixture-definition".into(),
                parent_thread_id: Some("fixture-thread".into()),
                input: serde_json::json!({"prompt":"archived writer fixture"}),
                phase_states: serde_json::json!({}),
                child_run_ids: vec![],
                status: WorkflowRunStatus::Running,
                summary: None,
                started_at: None,
                completed_at: None,
            })
            .await
            .unwrap();
        ledger
            .record_session_start(SessionStart {
                id: "fixture-session".into(),
                agent_definition_id: "fixture-agent".into(),
                agent_definition_name: "Archived fixture".into(),
                session_key: "fixture-session-key".into(),
                thread_id: Some("fixture-thread".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        ledger
            .record_message(NewSessionMessage {
                session_id: "fixture-session".into(),
                role: "user".into(),
                content: "session writer compatibility fixture".into(),
                ..Default::default()
            })
            .await
            .unwrap();
    });

    let mut collections = serde_json::Map::new();
    for collection in [
        "ledger.workflow_runs",
        "history.sessions",
        "history.messages",
        "tsd.counters",
    ] {
        let rows = runtime
            .block_on(scoped.documents().query(collection, &Query::all()))
            .unwrap()
            .items;
        collections.insert(collection.into(), serde_json::to_value(rows).unwrap());
    }
    std::fs::write(output, serde_json::to_vec_pretty(&collections).unwrap()).unwrap();
}
