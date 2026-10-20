use super::*;
use crate::testkit::MemoryStorage;
use serde_json::json;

pub(super) fn run(id: &str) -> AgentRunUpsert {
    AgentRunUpsert {
        id: id.into(),
        kind: AgentRunKind::WorkerThread,
        parent_run_id: Some("parent".into()),
        parent_thread_id: Some("thread".into()),
        agent_id: Some("agent".into()),
        status: AgentRunStatus::Running,
        prompt_ref: Some("prompt".into()),
        worker_thread_id: Some("worker".into()),
        checkpoint_path: None,
        checkpoint: Some(json!({"step":1})),
        summary: Some("original".into()),
        error: None,
        metadata: json!({"keep":true}),
        started_at: Some(now()),
        completed_at: None,
    }
}

#[tokio::test]
async fn independent_agent_handles_preserve_merge_and_isolate_profiles() {
    let backend = MemoryStorage::new();
    let scope = Scope::new("alice").unwrap();
    let a = DriverRunLedger::new(backend.for_scope(&scope).unwrap());
    let peer = DriverRunLedger::new(backend.for_scope(&scope).unwrap());
    let other = DriverRunLedger::new(backend.for_scope(&Scope::new("bob").unwrap()).unwrap());
    let first = a.upsert_agent_run(run("run")).await.unwrap();
    let mut edit = run("run");
    edit.kind = AgentRunKind::Subagent;
    edit.started_at = None;
    edit.summary = None;
    edit.metadata = json!({});
    let changed = peer.upsert_agent_run(edit).await.unwrap();
    assert_eq!(changed.kind, AgentRunKind::WorkerThread);
    assert_eq!(changed.started_at, first.started_at);
    assert_eq!(changed.summary, first.summary);
    assert_eq!(changed.metadata, first.metadata);
    assert_eq!(a.get_agent_run("run").await.unwrap().unwrap(), changed);
    assert!(other.get_agent_run("run").await.unwrap().is_none());
}

#[tokio::test]
async fn native_status_compare_and_swap_records_only_the_winning_event() {
    let backend = MemoryStorage::new();
    let scope = Scope::new("alice").unwrap();
    let a = DriverRunLedger::new(backend.for_scope(&scope).unwrap());
    a.upsert_agent_run(run("run")).await.unwrap();
    let event = || {
        Some(RunEventAppend {
            run_id: "run".into(),
            event_type: "completed".into(),
            payload: json!({}),
        })
    };
    let result = a
        .transition_agent_run_status_from(
            "run",
            AgentRunStatus::Running,
            AgentRunStatus::Completed,
            None,
            Some(now()),
            event(),
        )
        .await
        .unwrap();
    assert!(matches!(result, RunTransition::Applied(_)));
    let stale = a
        .transition_agent_run_status_from(
            "run",
            AgentRunStatus::Running,
            AgentRunStatus::Failed,
            Some("stale"),
            None,
            event(),
        )
        .await
        .unwrap();
    assert!(matches!(stale, RunTransition::StatusChanged(_)));
    let events = a
        .list_recent_run_events(&RunEventListRequest {
            run_id: "run".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(events.events.len(), 1);
    assert_eq!(events.events[0].sequence, 1);
}

#[tokio::test]
async fn agent_queries_join_telemetry_and_exclusive_sweep_preserves_resumable_runs() {
    let (a, b) = super::tests::ledgers();
    for (id, status) in [
        ("running", AgentRunStatus::Running),
        ("pending", AgentRunStatus::Pending),
        ("paused", AgentRunStatus::Paused),
        ("awaiting", AgentRunStatus::AwaitingUser),
    ] {
        let mut input = run(id);
        input.status = status;
        a.upsert_agent_run(input).await.unwrap();
    }
    a.upsert_run_telemetry(RunTelemetryUpsert {
        run_id: "running".into(),
        input_tokens: Some(42),
        ..Default::default()
    })
    .await
    .unwrap();
    let page = a
        .list_agent_runs(&AgentRunListRequest {
            limit: Some(1),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(page.count, 4);
    assert_eq!(page.runs.len(), 1);
    assert_eq!(
        a.get_agent_run("running")
            .await
            .unwrap()
            .unwrap()
            .telemetry
            .unwrap()
            .input_tokens,
        42
    );
    assert_eq!(a.interrupt_orphaned_agent_runs().await.unwrap(), 2);
    assert_eq!(
        a.get_agent_run("running").await.unwrap().unwrap().status,
        AgentRunStatus::Interrupted
    );
    assert_eq!(
        a.get_agent_run("paused").await.unwrap().unwrap().status,
        AgentRunStatus::Paused
    );
    assert_eq!(
        a.get_agent_run("awaiting").await.unwrap().unwrap().status,
        AgentRunStatus::AwaitingUser
    );
    assert_eq!(
        b.list_agent_runs(&Default::default()).await.unwrap().count,
        0
    );
}

#[tokio::test]
async fn independent_event_allocators_replay_without_duplicates_and_keep_cursor_order() {
    let backend = MemoryStorage::new();
    let scope = Scope::new("alice").unwrap();
    let a = DriverRunLedger::new(backend.for_scope(&scope).unwrap());
    let b = DriverRunLedger::new(backend.for_scope(&scope).unwrap());
    let input = |label: &str| RunEventAppend {
        run_id: "run".into(),
        event_type: label.into(),
        payload: json!({}),
    };
    let (one, two) = tokio::join!(
        a.append_run_event(input("one")),
        b.append_run_event(input("two"))
    );
    let mut sequences = [one.unwrap().sequence, two.unwrap().sequence];
    sequences.sort();
    assert_eq!(sequences, [1, 2]);
    assert_eq!(a.reconcile_run_events("run").await.unwrap(), 2);
    assert_eq!(b.reconcile_run_events("run").await.unwrap(), 2);
    let entries = backend
        .for_scope(&scope)
        .unwrap()
        .streams()
        .read_window(&storage::event_stream("run").unwrap(), 0, 10)
        .await
        .unwrap();
    assert_eq!(entries.len(), 2);
    let page = b
        .list_recent_run_events(&RunEventListRequest {
            run_id: "run".into(),
            after_sequence: Some(1),
            limit: Some(1),
        })
        .await
        .unwrap();
    assert_eq!(page.count, 1);
    assert_eq!(page.events[0].sequence, 2);
}

#[derive(Debug)]
struct RefusingProjection(Arc<dyn tinystoragedrivers_api::StreamStore>);
#[async_trait::async_trait]
impl tinystoragedrivers_api::StreamStore for RefusingProjection {
    fn capabilities(&self) -> tinystoragedrivers_types::Capabilities {
        self.0.capabilities()
    }

    async fn append(
        &self,
        stream: &str,
        value: serde_json::Value,
    ) -> std::result::Result<u64, StorageError> {
        self.0.append(stream, value).await
    }
    async fn append_batch(
        &self,
        stream: &str,
        values: Vec<serde_json::Value>,
    ) -> std::result::Result<u64, StorageError> {
        self.0.append_batch(stream, values).await
    }
    async fn append_batch_if_offset(
        &self,
        _: &str,
        _: u64,
        _: Vec<serde_json::Value>,
    ) -> std::result::Result<u64, StorageError> {
        Err(StorageError::unavailable("projection grant expired"))
    }
    async fn read_window(
        &self,
        stream: &str,
        from: u64,
        limit: usize,
    ) -> std::result::Result<Vec<tinystoragedrivers_types::StreamEntry>, StorageError> {
        self.0.read_window(stream, from, limit).await
    }
    async fn len(&self, stream: &str) -> std::result::Result<u64, StorageError> {
        self.0.len(stream).await
    }
    async fn truncate_before(
        &self,
        stream: &str,
        offset: u64,
    ) -> std::result::Result<u64, StorageError> {
        self.0.truncate_before(stream, offset).await
    }
    async fn delete_stream(&self, stream: &str) -> std::result::Result<bool, StorageError> {
        self.0.delete_stream(stream).await
    }
    async fn streams(&self, prefix: &str) -> std::result::Result<Vec<String>, StorageError> {
        self.0.streams(prefix).await
    }
}

#[tokio::test]
async fn committed_status_and_event_survive_projection_refusal_and_process_reconstruction() {
    let backend = MemoryStorage::new();
    let scope = Scope::new("alice").unwrap();
    let native = backend.for_scope(&scope).unwrap();
    let ports = tinystoragedrivers_api::StorageBackend::for_scope(&backend, &scope).unwrap();
    let refused = ScopedStorage::new(
        tinystoragedrivers_api::Storage::new(
            ports.documents().clone(),
            Arc::new(RefusingProjection(native.streams().clone())),
            ports.kv().clone(),
            ports.blobs().clone(),
            ports.queues().clone(),
            ports.secrets().clone(),
        ),
        scope.clone(),
    );
    let a = DriverRunLedger::new(refused);
    a.upsert_agent_run(run("run")).await.unwrap();
    a.transition_agent_run_status_from(
        "run",
        AgentRunStatus::Running,
        AgentRunStatus::Completed,
        None,
        Some(now()),
        Some(RunEventAppend {
            run_id: "run".into(),
            event_type: "done".into(),
            payload: json!({"ok":true}),
        }),
    )
    .await
    .unwrap();
    assert!(
        matches!(a.reconcile_run_events("run").await.unwrap_err(), super::super::LedgerError::Driver(ref e) if e.kind() == tinystoragedrivers_types::ErrorKind::Unavailable)
    );
    let reconstructed = DriverRunLedger::new(backend.for_scope(&scope).unwrap());
    assert_eq!(
        reconstructed
            .get_agent_run("run")
            .await
            .unwrap()
            .unwrap()
            .status,
        AgentRunStatus::Completed
    );
    assert_eq!(
        reconstructed
            .list_recent_run_events(&RunEventListRequest {
                run_id: "run".into(),
                ..Default::default()
            })
            .await
            .unwrap()
            .count,
        1
    );
    assert_eq!(reconstructed.reconcile_run_events("run").await.unwrap(), 1);
    assert_eq!(reconstructed.reconcile_run_events("run").await.unwrap(), 1);
}

#[tokio::test]
async fn competing_handles_commit_exactly_one_status_and_event() {
    let backend = MemoryStorage::new();
    let scope = Scope::new("alice").unwrap();
    let a = DriverRunLedger::new(backend.for_scope(&scope).unwrap());
    let b = DriverRunLedger::new(backend.for_scope(&scope).unwrap());
    a.upsert_agent_run(run("run")).await.unwrap();
    let event = |label: &str| {
        Some(RunEventAppend {
            run_id: "run".into(),
            event_type: label.into(),
            payload: json!({}),
        })
    };
    let (one, two) = tokio::join!(
        a.transition_agent_run_status_from(
            "run",
            AgentRunStatus::Running,
            AgentRunStatus::Completed,
            None,
            None,
            event("one")
        ),
        b.transition_agent_run_status_from(
            "run",
            AgentRunStatus::Running,
            AgentRunStatus::Failed,
            Some("failure"),
            None,
            event("two")
        )
    );
    let results = [one.unwrap(), two.unwrap()];
    assert_eq!(
        results
            .iter()
            .filter(|r| matches!(r, RunTransition::Applied(_)))
            .count(),
        1
    );
    assert_eq!(
        results
            .iter()
            .filter(|r| matches!(r, RunTransition::StatusChanged(_)))
            .count(),
        1
    );
    assert_eq!(
        a.list_recent_run_events(&RunEventListRequest {
            run_id: "run".into(),
            ..Default::default()
        })
        .await
        .unwrap()
        .count,
        1
    );
}

#[tokio::test]
async fn native_command_center_controls_use_atomic_status_and_event_receipts() {
    let (a, _) = super::tests::ledgers();
    a.upsert_agent_run(run("run")).await.unwrap();
    let no_names = |_: &str| None;
    let view = a.list_agent_work(None, &no_names).await.unwrap();
    assert_eq!(view.groups.iter().map(|b| b.rows.len()).sum::<usize>(), 1);
    a.apply_control("run", ControlVerb::Stop, None, Some("stop"), &no_names)
        .await
        .unwrap();
    assert_eq!(
        a.get_agent_run("run").await.unwrap().unwrap().status,
        AgentRunStatus::Cancelled
    );
    a.apply_control("run", ControlVerb::Retry, None, None, &no_names)
        .await
        .unwrap();
    let row = a.get_agent_run("run").await.unwrap().unwrap();
    assert_eq!(row.status, AgentRunStatus::Pending);
    assert!(row.error.is_none());
    assert!(row.completed_at.is_none());
    assert_eq!(
        a.list_recent_run_events(&RunEventListRequest {
            run_id: "run".into(),
            ..Default::default()
        })
        .await
        .unwrap()
        .count,
        2
    );
}

#[tokio::test]
async fn agent_and_event_receipts_survive_driver_reconstruction() {
    let scope = Scope::new("alice").unwrap();
    let backend = MemoryStorage::new();
    let ledger = DriverRunLedger::new(backend.for_scope(&scope).unwrap());
    ledger.upsert_agent_run(run("run")).await.unwrap();
    ledger
        .transition_agent_run_status_from(
            "run",
            AgentRunStatus::Running,
            AgentRunStatus::Completed,
            None,
            None,
            Some(RunEventAppend {
                run_id: "run".into(),
                event_type: "done".into(),
                payload: json!({}),
            }),
        )
        .await
        .unwrap();
    drop(ledger);
    let ledger = DriverRunLedger::new(backend.for_scope(&scope).unwrap());
    assert_eq!(
        ledger.get_agent_run("run").await.unwrap().unwrap().status,
        AgentRunStatus::Completed
    );
    assert_eq!(ledger.reconcile_run_events("run").await.unwrap(), 1);
    assert_eq!(ledger.reconcile_run_events("run").await.unwrap(), 1);
}

#[tokio::test]
async fn projection_refuses_an_existing_stream_that_disagrees_with_committed_receipts() {
    let backend = MemoryStorage::new();
    let scope = Scope::new("alice").unwrap();
    let scoped = backend.for_scope(&scope).unwrap();
    let a = DriverRunLedger::new(scoped.clone());
    a.append_run_event(RunEventAppend {
        run_id: "run".into(),
        event_type: "real".into(),
        payload: json!({}),
    })
    .await
    .unwrap();
    scoped
        .streams()
        .append(
            &storage::event_stream("run").unwrap(),
            json!({"foreign":true}),
        )
        .await
        .unwrap();
    assert!(
        matches!(a.reconcile_run_events("run").await.unwrap_err(), super::super::LedgerError::Driver(ref error)
        if error.kind() == tinystoragedrivers_types::ErrorKind::Serialization)
    );
}

#[path = "agent_fault_tests.rs"]
mod fault_tests;

#[tokio::test]
async fn projection_refuses_a_valid_receipt_that_skips_an_earlier_committed_event() {
    let backend = MemoryStorage::new();
    let native = backend.for_scope(&Scope::new("alice").unwrap()).unwrap();
    let ledger = DriverRunLedger::new(native.clone());
    ledger
        .append_run_event(RunEventAppend {
            run_id: "run".into(),
            event_type: "first".into(),
            payload: json!({}),
        })
        .await
        .unwrap();
    let second = ledger
        .append_run_event(RunEventAppend {
            run_id: "run".into(),
            event_type: "second".into(),
            payload: json!({}),
        })
        .await
        .unwrap();
    let stream = storage::event_stream("run").unwrap();
    native
        .streams()
        .append(&stream, serde_json::to_value(second).unwrap())
        .await
        .unwrap();
    let error = ledger.reconcile_run_events("run").await.unwrap_err();
    assert!(
        matches!(error, crate::run_ledger::LedgerError::Driver(error) if error.kind() == tinystoragedrivers_types::ErrorKind::Serialization)
    );
    assert_eq!(native.streams().len(&stream).await.unwrap(), 1);
    assert_eq!(
        ledger
            .list_recent_run_events(&RunEventListRequest {
                run_id: "run".into(),
                ..Default::default()
            })
            .await
            .unwrap()
            .count,
        2
    );
}
