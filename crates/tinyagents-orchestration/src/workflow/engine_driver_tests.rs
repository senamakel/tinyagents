use super::*;
use tinyagents_session::run_ledger::{DriverRunLedger, LedgerError};
use tinyagents_storage_kit::{ErrorKind, Scope, StorageBackend};
use tinystoragedrivers_testkit::MemoryBackend;

fn native_kind(error: &WorkflowPersistenceError) -> ErrorKind {
    match innermost(error) {
        WorkflowPersistenceError::Ledger(LedgerError::Driver(error)) => error.kind(),
        other => panic!("native source flattened: {other:?}"),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn independent_scoped_drivers_keep_complete_workflow_runs_separate() {
    let backend = MemoryBackend::new();
    let a = Arc::new(DriverRunLedger::new(
        backend
            .for_scope(&Scope::new("profile-a").unwrap())
            .unwrap(),
    ));
    let b = Arc::new(DriverRunLedger::new(
        backend
            .for_scope(&Scope::new("profile-b").unwrap())
            .unwrap(),
    ));
    let engine = WorkflowEngine::new_async(
        Arc::new(LedgerWorkflowStore::new(a.clone())),
        Arc::new(RegisteredExecutor),
    );
    let definition = tests::definition();
    engine
        .initialise_async("run".into(), &definition, json!({}), None)
        .await
        .unwrap();
    engine
        .drive_async("run", &definition, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(
        a.get_workflow_run("run").await.unwrap().unwrap().status,
        WorkflowRunStatus::Completed
    );
    assert!(b.get_workflow_run("run").await.unwrap().is_none());
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn registration_and_child_failures_retain_native_unavailable_kind() {
    let ledger_backend = MemoryBackend::new();
    // Invoke both owning regressions under the driver's native error variant.
    let error = refusal();
    assert_eq!(native_kind(&error), ErrorKind::Unavailable);
    let _workspace = tempfile::tempdir().unwrap();
    let ledger = Arc::new(crate::test_support::ledger_in(&ledger_backend));
    let store = Arc::new(RegistrationRefusal {
        inner: LedgerWorkflowStore::new(ledger),
        cas: AtomicUsize::new(0),
    });
    let engine = WorkflowEngine::new_async(store, Arc::new(IgnoreFailedRegistration::default()));
    let definition = tests::definition();
    engine
        .initialise_async("run".into(), &definition, json!({}), None)
        .await
        .unwrap();
    let error = engine
        .drive_async("run", &definition, CancellationToken::new())
        .await
        .unwrap_err();
    assert_eq!(native_kind(&error), ErrorKind::Unavailable);
}

#[derive(Default)]
struct HeldChild {
    started: tokio::sync::Notify,
    release: tokio::sync::Notify,
    executions: AtomicUsize,
}
#[async_trait]
impl AsyncWorkflowExecutor for HeldChild {
    async fn execute_async(
        &self,
        _: WorkflowChildRequest,
        _: CancellationToken,
        registration: Arc<dyn AsyncWorkflowChildRegistration>,
    ) -> WorkflowPersistenceResult<WorkflowChildResult> {
        self.executions.fetch_add(1, Ordering::SeqCst);
        registration.register_async("held-child".into()).await?;
        self.started.notify_one();
        self.release.notified().await;
        Ok(WorkflowChildResult {
            child_id: "held-child".into(),
            output: json!("done"),
        })
    }
    async fn cancel_children(&self, _: &[String]) {}
}

#[tokio::test(flavor = "current_thread")]
async fn independent_authoritative_engines_acquire_only_one_live_owner() {
    let backend = MemoryBackend::new();
    let scope = Scope::new("profile-a").unwrap();
    let first = Arc::new(DriverRunLedger::new(backend.for_scope(&scope).unwrap()));
    let second = Arc::new(DriverRunLedger::new(backend.for_scope(&scope).unwrap()));
    let executor = Arc::new(HeldChild::default());
    let a = Arc::new(WorkflowEngine::new_async(
        Arc::new(LedgerWorkflowStore::new(first.clone())),
        executor.clone(),
    ));
    let b = WorkflowEngine::new_async(Arc::new(LedgerWorkflowStore::new(second)), executor.clone());
    let mut definition = tests::definition();
    definition.phases.truncate(1);
    a.initialise_async("run".into(), &definition, json!({}), None)
        .await
        .unwrap();
    let task = tokio::spawn({
        let a = a.clone();
        let definition = definition.clone();
        async move {
            a.drive_async("run", &definition, CancellationToken::new())
                .await
        }
    });
    executor.started.notified().await;
    b.drive_async("run", &definition, CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(executor.executions.load(Ordering::SeqCst), 1);
    executor.release.notify_one();
    task.await.unwrap().unwrap();
    assert_eq!(
        first.get_workflow_run("run").await.unwrap().unwrap().status,
        WorkflowRunStatus::Completed
    );
}

struct RefusedAuthority(ErrorKind);
impl RefusedAuthority {
    fn error<T>(&self) -> WorkflowPersistenceResult<T> {
        Err(WorkflowPersistenceError::Ledger(
            tinyagents_storage_kit::StorageError::new(self.0, "configured authority refused")
                .into(),
        ))
    }
}
#[async_trait]
impl AsyncWorkflowStore for RefusedAuthority {
    async fn load(&self, _: &str) -> WorkflowPersistenceResult<Option<WorkflowRun>> {
        self.error()
    }
    async fn upsert(&self, _: WorkflowRunUpsert) -> WorkflowPersistenceResult<WorkflowRun> {
        self.error()
    }
    async fn claim(
        &self,
        _: &str,
        _: &str,
        _: Duration,
    ) -> WorkflowPersistenceResult<WorkflowLeaseClaim> {
        self.error()
    }
    async fn renew(&self, _: &str, _: &str, _: Duration) -> WorkflowPersistenceResult<bool> {
        self.error()
    }
    async fn compare_and_swap(
        &self,
        _: WorkflowRunUpsert,
        _: u64,
        _: &str,
        _: Duration,
    ) -> WorkflowPersistenceResult<Option<WorkflowRun>> {
        self.error()
    }
}

#[tokio::test(flavor = "current_thread")]
async fn configured_refusal_never_uses_legacy_initialise_or_drive() {
    let ambient_db = std::env::current_dir()
        .unwrap()
        .join("session_db/sessions.db");
    let existed_before = ambient_db.exists();
    for kind in [
        ErrorKind::Unavailable,
        ErrorKind::Conflict,
        ErrorKind::Unsupported(tinyagents_storage_kit::Capability::Transactions),
    ] {
        let engine = WorkflowEngine::new_async(
            Arc::new(RefusedAuthority(kind)),
            Arc::new(RegisteredExecutor),
        );
        let definition = tests::definition();
        assert_eq!(
            native_kind(
                &engine
                    .initialise_async("run".into(), &definition, json!({}), None)
                    .await
                    .unwrap_err()
            ),
            kind
        );
        assert_eq!(
            native_kind(
                &engine
                    .drive_async("run", &definition, CancellationToken::new())
                    .await
                    .unwrap_err()
            ),
            kind
        );
        assert!(
            engine
                .initialise("legacy".into(), &definition, json!({}), None)
                .is_err()
        );
        assert!(
            engine
                .drive("legacy", &definition, CancellationToken::new())
                .await
                .is_err()
        );
    }
    assert_eq!(
        ambient_db.exists(),
        existed_before,
        "configured refusal created an ambient SQLite fallback"
    );
}

struct WorkflowTakeoverChild {
    ledger: Arc<DriverRunLedger>,
    replacement: parking_lot::Mutex<Option<WorkflowRun>>,
    cancelled: parking_lot::Mutex<Vec<String>>,
}
#[async_trait]
impl AsyncWorkflowExecutor for WorkflowTakeoverChild {
    async fn execute_async(
        &self,
        _: WorkflowChildRequest,
        _: CancellationToken,
        registration: Arc<dyn AsyncWorkflowChildRegistration>,
    ) -> WorkflowPersistenceResult<WorkflowChildResult> {
        registration.register_async("takeover-child".into()).await?;
        let current = self.ledger.get_workflow_run("run").await.unwrap().unwrap();
        let revision = current.revision;
        let next = WorkflowRunUpsert {
            id: current.id,
            definition_id: current.definition_id,
            parent_thread_id: current.parent_thread_id,
            input: current.input,
            phase_states: current.phase_states,
            child_run_ids: current.child_run_ids,
            status: WorkflowRunStatus::Running,
            summary: None,
            started_at: Some(current.started_at),
            completed_at: None,
        };
        self.ledger
            .compare_and_swap_workflow_run_lifecycle(next, revision)
            .await
            .unwrap()
            .unwrap();
        let WorkflowLeaseClaim::Acquired(replacement) = self
            .ledger
            .try_claim_workflow_run("run", "replacement-owner", chrono::Duration::minutes(1))
            .await
            .unwrap()
        else {
            panic!("replacement claim refused")
        };
        *self.replacement.lock() = Some(replacement);
        std::future::pending().await
    }
    async fn cancel_children(&self, ids: &[String]) {
        self.cancelled.lock().extend_from_slice(ids);
    }
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn workflow_owner_takeover_returns_typed_loss_without_mutating_replacement() {
    let backend = MemoryBackend::new();
    let scope = Scope::new("profile-a").unwrap();
    let ledger = Arc::new(DriverRunLedger::new(backend.for_scope(&scope).unwrap()));
    let executor = Arc::new(WorkflowTakeoverChild {
        ledger: ledger.clone(),
        replacement: parking_lot::Mutex::new(None),
        cancelled: parking_lot::Mutex::new(vec![]),
    });
    let engine = WorkflowEngine::new_async(
        Arc::new(LedgerWorkflowStore::new(ledger.clone())),
        executor.clone(),
    )
    .with_lease_duration(Duration::from_secs(60));
    let mut definition = tests::definition();
    definition.phases.truncate(1);
    engine
        .initialise_async("run".into(), &definition, json!({}), None)
        .await
        .unwrap();
    let error = tokio::time::timeout(
        Duration::from_secs(30),
        engine.drive_async("run", &definition, CancellationToken::new()),
    )
    .await
    .expect("stale workflow owner cannot keep renewing")
    .unwrap_err();
    assert!(
        !matches!(error, WorkflowPersistenceError::Legacy(_)),
        "authoritative ownership loss was flattened into the legacy string error"
    );
    assert!(error.to_string().contains("lease renewal failed"));
    assert_eq!(
        ledger.get_workflow_run("run").await.unwrap(),
        executor.replacement.lock().clone()
    );
    assert_eq!(*executor.cancelled.lock(), vec!["takeover-child"]);
}
