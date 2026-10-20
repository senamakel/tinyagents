use super::*;
use crate::run_ledger::history::{NewSessionMessage, SessionHistoryStore, SessionStart};
use crate::run_ledger::types::*;
use crate::run_ledger::{DriverRunLedger, RunLedgerStore};
use chrono::{Duration, Utc};
use serde_json::json;
use tinyagents_storage_kit::Scope;
use tinystoragedrivers_api::DocumentStoreExt;
use tinystoragedrivers_api::StorageBackend;
use tinystoragedrivers_testkit::memory_backend;
use tinystoragedrivers_types::WriteOptions;

fn scoped() -> ScopedStorage {
    let backend = memory_backend();
    let scope = Scope::new("legacy-ledger-tests").unwrap();
    let storage = backend.for_scope(&scope).unwrap();
    ScopedStorage::new(storage, scope)
}

async fn put(storage: &ScopedStorage, collection: &str, id: &str, body: Value) {
    let docs = storage.documents();
    docs.ensure_collection(&CollectionSpec::new(collection))
        .await
        .unwrap();
    docs.put_with(
        collection,
        id,
        body,
        Precondition::None,
        WriteOptions::new(),
    )
    .await
    .unwrap();
}

async fn put_fixture_row(storage: &ScopedStorage, collection: &str, row: &Value) {
    let docs = storage.documents();
    let meta = &row["meta"];
    let mut options = WriteOptions::new()
        .schema(meta["schema"].as_u64().unwrap() as u32)
        .created_ms(meta["created_ms"].as_i64().unwrap());
    if let Some(expires_ms) = meta["expires_ms"].as_i64() {
        options = options.expires_ms(expires_ms);
    }
    let id = row["id"].as_str().unwrap();
    docs.ensure_collection(&CollectionSpec::new(collection))
        .await
        .unwrap();
    docs.put_with(
        collection,
        id,
        row["doc"].clone(),
        Precondition::None,
        options,
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn live_workflow_leases_and_history_counters_move_to_the_shared_primitives() {
    let storage = scoped();
    let live = (Utc::now() + Duration::hours(1)).to_rfc3339();
    let lapsed = (Utc::now() - Duration::hours(1)).to_rfc3339();
    for (run, expires) in [("live", &live), ("lapsed", &lapsed)] {
        put(
            &storage,
            "run_ledger_workflow_runs",
            &legacy_key(&[run]),
            json!({"schemaVersion": 1, "record": {"id": run, "leaseOwner": "driver", "leaseExpiresAt": expires}}),
        )
        .await;
    }
    put(
        &storage,
        "session_history_counters",
        &legacy_key(&["messages"]),
        json!({"schemaVersion": 1, "record": {"next": 11}}),
    )
    .await;
    put(
        &storage,
        "run_ledger_agent_imports",
        "receipt",
        json!({"record": {}}),
    )
    .await;

    let report = import_legacy_ledger(&storage).await.unwrap();
    assert_eq!(
        report,
        LegacyLedgerImport {
            leases: 1,
            counters: 1
        }
    );
    assert_eq!(
        import_legacy_ledger(&storage).await.unwrap(),
        LegacyLedgerImport::default()
    );

    let ledger = DriverRunLedger::new(storage.clone());
    // The migrated lease is held by the old driver: nobody else can claim it.
    ledger
        .upsert_workflow_run(WorkflowRunUpsert {
            id: "live".into(),
            definition_id: "def".into(),
            parent_thread_id: None,
            input: json!({}),
            phase_states: json!({}),
            child_run_ids: vec![],
            status: WorkflowRunStatus::Running,
            summary: None,
            started_at: None,
            completed_at: None,
        })
        .await
        .unwrap();
    assert!(matches!(
        ledger.try_claim_workflow_run("live", "other", Duration::hours(1)).await.unwrap(),
        WorkflowLeaseClaim::Busy(row) if row.lease_owner.as_deref() == Some("driver")
    ));
    // The next message id continues after the v1 counter.
    ledger
        .record_session_start(SessionStart {
            id: "s".into(),
            session_key: "s".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    let id = ledger
        .record_message(NewSessionMessage {
            session_id: "s".into(),
            role: "user".into(),
            content: "x".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(id, 11);
    // The v1 collections are emptied.
    for legacy in ["session_history_counters", "run_ledger_agent_imports"] {
        assert!(
            storage
                .documents()
                .query_all(legacy, &Default::default())
                .await
                .unwrap()
                .is_empty()
        );
    }
}

#[tokio::test]
async fn a_workflow_lease_is_a_tsd_lease_and_not_part_of_the_stored_run() {
    let storage = scoped();
    let ledger = DriverRunLedger::new(storage.clone());
    ledger
        .upsert_workflow_run(WorkflowRunUpsert {
            id: "wf".into(),
            definition_id: "def".into(),
            parent_thread_id: None,
            input: json!({}),
            phase_states: json!({}),
            child_run_ids: vec![],
            status: WorkflowRunStatus::Running,
            summary: None,
            started_at: None,
            completed_at: None,
        })
        .await
        .unwrap();
    let WorkflowLeaseClaim::Acquired(row) = ledger
        .try_claim_workflow_run("wf", "driver", Duration::hours(1))
        .await
        .unwrap()
    else {
        panic!("a free lease is acquired");
    };
    assert_eq!(row.lease_owner.as_deref(), Some("driver"));
    let stored = storage
        .documents()
        .get("ledger.workflow_runs", "wf")
        .await
        .unwrap()
        .unwrap()
        .doc;
    assert!(stored.get("lease_owner").is_none() && stored.get("lease_expires_at").is_none());
    let lease = storage
        .documents()
        .get("tsd.leases", "ledger/workflow%2Fwf")
        .await
        .unwrap()
        .unwrap()
        .doc;
    assert_eq!(lease["holder"], "driver");
    assert_eq!(lease["epoch"], 1);
}

#[tokio::test]
async fn archived_public_writer_rows_open_through_current_typed_owners() {
    let backend = std::sync::Arc::new(memory_backend())
        as std::sync::Arc<dyn tinystoragedrivers_api::StorageBackend>;
    let scope = Scope::new("fixture-agent-scope").unwrap();
    let storage = ScopedStorage::new(backend.for_scope(&scope).unwrap(), scope);
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/legacy-31cc-session.json")).unwrap();
    for collection in [
        "ledger.workflow_runs",
        "history.sessions",
        "history.messages",
        "tsd.counters",
    ] {
        for row in fixture[collection].as_array().unwrap() {
            put_fixture_row(&storage, collection, row).await;
        }
    }
    put(
        &storage,
        "session_kv",
        "legacy-setting",
        json!({"ns": "prefs", "key": "theme", "value": "dark"}),
    )
    .await;

    // The provider's real open path binds the host-supplied scope, runs the
    // session-owned legacy-state import, and migrates every session owner
    // before exposing the driver handles.
    let provider = crate::DriverSessionStores::new(crate::testkit::scoped_storage_factory(
        backend,
        std::collections::HashMap::from([(
            "fixture-agent".to_owned(),
            "fixture-agent-scope".to_owned(),
        )]),
    ))
    .unwrap();
    provider.try_for_agent("fixture-agent").unwrap();
    let imported_kv = storage
        .documents()
        .query_all("tsd.kv", &Default::default())
        .await
        .unwrap();
    assert!(imported_kv.iter().any(|row| {
        row.doc["owner"] == "session"
            && row.doc["ns"] == "prefs"
            && row.doc["key"] == "theme"
            && row.doc["value"] == "dark"
    }));
    assert!(
        storage
            .documents()
            .query_all("session_kv", &Default::default())
            .await
            .unwrap()
            .is_empty()
    );
    let ledger = provider.driver_run_ledger("fixture-agent").unwrap();
    let workflow = ledger
        .get_workflow_run("fixture-workflow")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(workflow.definition_id, "fixture-definition");
    assert_eq!(workflow.parent_thread_id.as_deref(), Some("fixture-thread"));
    let session = ledger
        .get_session("fixture-session")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(session.agent_definition_id, "fixture-agent");
    assert_eq!(session.agent_definition_name, "Archived fixture");
    let messages = ledger.list_messages("fixture-session", None).await.unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].id, 1);
    assert_eq!(messages[0].content, "session writer compatibility fixture");
    let next = ledger
        .record_message(NewSessionMessage {
            session_id: "fixture-session".into(),
            role: "assistant".into(),
            content: "new writer after opening archived data".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(next, 2);
    assert_eq!(
        ledger
            .list_messages("fixture-session", None)
            .await
            .unwrap()
            .iter()
            .map(|message| message.id)
            .collect::<Vec<_>>(),
        [1, 2]
    );
}
