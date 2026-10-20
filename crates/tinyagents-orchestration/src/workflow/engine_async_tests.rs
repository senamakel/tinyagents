use super::*;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;
use tinyagents_harness::CancellationToken;
use tinyagents_session::run_ledger::{RunLedgerStore, WorkflowRunStatus};
use tinystoragedrivers_testkit::MemoryBackend;

struct RegisteredExecutor;
#[async_trait]
impl AsyncWorkflowExecutor for RegisteredExecutor {
    async fn execute_async(
        &self,
        request: WorkflowChildRequest,
        _: CancellationToken,
        registration: Arc<dyn AsyncWorkflowChildRegistration>,
    ) -> WorkflowPersistenceResult<WorkflowChildResult> {
        let id = format!("{}-{}", request.phase, request.index_in_phase);
        registration.register_async(id.clone()).await?;
        Ok(WorkflowChildResult {
            child_id: id,
            output: json!({"ok":true}),
        })
    }
    async fn cancel_children(&self, _: &[String]) {}
}

#[tokio::test(flavor = "current_thread")]
async fn authoritative_engine_awaits_registration_and_completes_on_current_thread() {
    let ledger_backend = MemoryBackend::new();
    let _workspace = tempfile::tempdir().unwrap();
    let ledger = Arc::new(crate::test_support::ledger_in(&ledger_backend));
    let engine = WorkflowEngine::new_async(
        Arc::new(LedgerWorkflowStore::new(ledger.clone())),
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
    let run = ledger.get_workflow_run("run").await.unwrap().unwrap();
    assert_eq!(run.status, WorkflowRunStatus::Completed);
    assert_eq!(run.child_run_ids.len(), 4);
}

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tinyagents_session::run_ledger::{WorkflowLeaseClaim, WorkflowRun, WorkflowRunUpsert};

fn refusal() -> WorkflowPersistenceError {
    WorkflowPersistenceError::Ledger(
        tinyagents_storage_kit::StorageError::new(
            tinyagents_storage_kit::ErrorKind::Unavailable,
            "registration backend unavailable",
        )
        .into(),
    )
}

struct RegistrationRefusal {
    inner: LedgerWorkflowStore,
    cas: AtomicUsize,
}
#[async_trait]
impl AsyncWorkflowStore for RegistrationRefusal {
    async fn load(&self, id: &str) -> WorkflowPersistenceResult<Option<WorkflowRun>> {
        self.inner.load(id).await
    }
    async fn upsert(&self, row: WorkflowRunUpsert) -> WorkflowPersistenceResult<WorkflowRun> {
        self.inner.upsert(row).await
    }
    async fn claim(
        &self,
        id: &str,
        owner: &str,
        lease: Duration,
    ) -> WorkflowPersistenceResult<WorkflowLeaseClaim> {
        self.inner.claim(id, owner, lease).await
    }
    async fn renew(
        &self,
        id: &str,
        owner: &str,
        lease: Duration,
    ) -> WorkflowPersistenceResult<bool> {
        self.inner.renew(id, owner, lease).await
    }
    async fn compare_and_swap(
        &self,
        row: WorkflowRunUpsert,
        rev: u64,
        owner: &str,
        lease: Duration,
    ) -> WorkflowPersistenceResult<Option<WorkflowRun>> {
        if self.cas.fetch_add(1, Ordering::SeqCst) >= 1 {
            return Err(refusal());
        }
        self.inner.compare_and_swap(row, rev, owner, lease).await
    }
}

#[derive(Default)]
struct IgnoreFailedRegistration {
    cancelled: parking_lot::Mutex<Vec<String>>,
}
#[async_trait]
impl AsyncWorkflowExecutor for IgnoreFailedRegistration {
    async fn execute_async(
        &self,
        _: WorkflowChildRequest,
        _: CancellationToken,
        registration: Arc<dyn AsyncWorkflowChildRegistration>,
    ) -> WorkflowPersistenceResult<WorkflowChildResult> {
        assert!(
            registration
                .register_async("real-child".into())
                .await
                .is_err()
        );
        std::future::pending().await
    }
    async fn cancel_children(&self, ids: &[String]) {
        self.cancelled.lock().extend_from_slice(ids);
    }
}

fn innermost(error: &WorkflowPersistenceError) -> &WorkflowPersistenceError {
    match error {
        WorkflowPersistenceError::Registration(error) => innermost(error),
        other => other,
    }
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn ignored_registration_refusal_cancels_child_and_preserves_source_without_waiting() {
    let ledger_backend = MemoryBackend::new();
    let _workspace = tempfile::tempdir().unwrap();
    let ledger = Arc::new(crate::test_support::ledger_in(&ledger_backend));
    let store = Arc::new(RegistrationRefusal {
        inner: LedgerWorkflowStore::new(ledger.clone()),
        cas: AtomicUsize::new(0),
    });
    let executor = Arc::new(IgnoreFailedRegistration::default());
    let engine = WorkflowEngine::new_async(store, executor.clone());
    let mut definition = tests::definition();
    definition.phases.truncate(1);
    engine
        .initialise_async("run".into(), &definition, json!({}), None)
        .await
        .unwrap();
    let result = tokio::time::timeout(
        Duration::from_secs(1),
        engine.drive_async("run", &definition, CancellationToken::new()),
    )
    .await;
    let error = result
        .expect("registration refusal must interrupt an executor that ignores it")
        .unwrap_err();
    assert!(matches!(
        innermost(&error),
        WorkflowPersistenceError::Ledger(_)
    ));
    assert!(
        error
            .to_string()
            .contains("registration backend unavailable")
    );
    assert_eq!(*executor.cancelled.lock(), vec!["real-child"]);
    assert_ne!(
        ledger
            .get_workflow_run("run")
            .await
            .unwrap()
            .unwrap()
            .status,
        WorkflowRunStatus::Completed
    );
}

struct UnregisteredExecutor;
#[async_trait]
impl AsyncWorkflowExecutor for UnregisteredExecutor {
    async fn execute_async(
        &self,
        _: WorkflowChildRequest,
        _: CancellationToken,
        _: Arc<dyn AsyncWorkflowChildRegistration>,
    ) -> WorkflowPersistenceResult<WorkflowChildResult> {
        Ok(WorkflowChildResult {
            child_id: "unregistered".into(),
            output: json!("uncommitted"),
        })
    }
    async fn cancel_children(&self, _: &[String]) {}
}

#[tokio::test(flavor = "current_thread")]
async fn authoritative_success_cannot_substitute_for_committed_child_registration() {
    let ledger_backend = MemoryBackend::new();
    let _workspace = tempfile::tempdir().unwrap();
    let ledger = Arc::new(crate::test_support::ledger_in(&ledger_backend));
    let engine = WorkflowEngine::new_async(
        Arc::new(LedgerWorkflowStore::new(ledger.clone())),
        Arc::new(UnregisteredExecutor),
    );
    let definition = tests::definition();
    engine
        .initialise_async("run".into(), &definition, json!({}), None)
        .await
        .unwrap();
    assert!(
        matches!(engine.drive_async("run",&definition,CancellationToken::new()).await,
        Err(WorkflowPersistenceError::UnregisteredChild(id)) if id=="unregistered")
    );
    let run = ledger.get_workflow_run("run").await.unwrap().unwrap();
    assert!(run.child_run_ids.is_empty());
    assert_ne!(run.status, WorkflowRunStatus::Completed);
    assert!(
        engine
            .initialise("legacy".into(), &definition, json!({}), None)
            .is_err()
    );
    assert!(
        engine
            .drive("run", &definition, CancellationToken::new())
            .await
            .is_err()
    );
}

struct RefusingChild;
#[async_trait]
impl AsyncWorkflowExecutor for RefusingChild {
    async fn execute_async(
        &self,
        request: WorkflowChildRequest,
        _: CancellationToken,
        registration: Arc<dyn AsyncWorkflowChildRegistration>,
    ) -> WorkflowPersistenceResult<WorkflowChildResult> {
        if request.index_in_phase == 0 {
            registration.register_async("refused-child".into()).await?;
            return Err(refusal());
        }
        std::future::pending().await
    }
    async fn cancel_children(&self, _: &[String]) {}
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn typed_child_failure_interrupts_parallel_waiters_without_string_flattening() {
    let ledger_backend = MemoryBackend::new();
    let _workspace = tempfile::tempdir().unwrap();
    let ledger = Arc::new(crate::test_support::ledger_in(&ledger_backend));
    let engine = WorkflowEngine::new_async(
        Arc::new(LedgerWorkflowStore::new(ledger.clone())),
        Arc::new(RefusingChild),
    );
    let mut definition = tests::definition();
    definition.phases.truncate(1);
    definition.phases[0].agent_ids = vec!["refused".into(), "waiting".into()];
    engine
        .initialise_async("run".into(), &definition, json!({}), None)
        .await
        .unwrap();
    let error = tokio::time::timeout(
        Duration::from_secs(1),
        engine.drive_async("run", &definition, CancellationToken::new()),
    )
    .await
    .expect("typed child error cannot wait for unrelated child")
    .unwrap_err();
    assert!(matches!(
        innermost(&error),
        WorkflowPersistenceError::Ledger(_)
    ));
    assert!(
        error
            .to_string()
            .contains("registration backend unavailable")
    );
    assert_ne!(
        ledger
            .get_workflow_run("run")
            .await
            .unwrap()
            .unwrap()
            .status,
        WorkflowRunStatus::Completed
    );
}

struct SyncRegistrationAttempt;
#[async_trait]
impl AsyncWorkflowExecutor for SyncRegistrationAttempt {
    async fn execute_async(
        &self,
        _: WorkflowChildRequest,
        _: CancellationToken,
        registration: Arc<dyn AsyncWorkflowChildRegistration>,
    ) -> WorkflowPersistenceResult<WorkflowChildResult> {
        assert!(registration.register("sync-child".into()).is_err());
        std::future::pending().await
    }
    async fn cancel_children(&self, _: &[String]) {}
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn ignored_synchronous_registration_refusal_cannot_leave_authoritative_driver_waiting() {
    let ledger_backend = MemoryBackend::new();
    let _workspace = tempfile::tempdir().unwrap();
    let store = Arc::new(LedgerWorkflowStore::new(Arc::new(
        crate::test_support::ledger_in(&ledger_backend),
    )));
    let engine = WorkflowEngine::new_async(store, Arc::new(SyncRegistrationAttempt));
    let definition = tests::definition();
    engine
        .initialise_async("run".into(), &definition, json!({}), None)
        .await
        .unwrap();
    let error = tokio::time::timeout(
        Duration::from_secs(1),
        engine.drive_async("run", &definition, CancellationToken::new()),
    )
    .await
    .expect("synchronous registration refused without blocking")
    .unwrap_err();
    assert!(matches!(
        innermost(&error),
        WorkflowPersistenceError::AsyncRegistrationRequired
    ));
}

#[path = "engine_driver_tests.rs"]
mod driver_tests;

struct RefusedLegacyRegistration;
impl WorkflowChildRegistration for RefusedLegacyRegistration {
    fn register(&self, _: String) -> Result<(), OrchestrationError> {
        Err(OrchestrationError("registration refused".into()))
    }
}

#[tokio::test(flavor = "current_thread")]
async fn authoritative_placeholder_callbacks_cannot_supply_legacy_fallback() {
    let marker = AuthoritativeWorkflow;
    let row = WorkflowRunUpsert {
        id: "run".into(),
        definition_id: "definition".into(),
        parent_thread_id: None,
        input: json!({}),
        phase_states: json!({}),
        child_run_ids: vec![],
        status: WorkflowRunStatus::Running,
        summary: None,
        started_at: None,
        completed_at: None,
    };
    assert!(marker.load("run").is_err());
    assert!(marker.upsert(row.clone()).is_err());
    assert!(
        marker
            .claim("run", "owner", Duration::from_secs(30))
            .is_err()
    );
    assert!(
        marker
            .compare_and_swap(row, 0, "owner", Duration::from_secs(30))
            .is_err()
    );
    assert!(
        marker
            .renew("run", "owner", Duration::from_secs(30))
            .is_err()
    );
    let request = WorkflowChildRequest {
        run_id: "run".into(),
        phase: "phase".into(),
        agent_id: "agent".into(),
        index_in_phase: 0,
        prompt: String::new(),
    };
    assert!(
        marker
            .execute(
                request,
                CancellationToken::new(),
                Arc::new(RefusedLegacyRegistration)
            )
            .await
            .is_err()
    );
    WorkflowExecutor::cancel_children(&marker, &[]).await;
}

struct PendingRegistration {
    inner: tests::MemoryStore,
    started: tokio::sync::Notify,
    pending: std::sync::atomic::AtomicBool,
    refuse_renewal: bool,
}
#[async_trait]
impl AsyncWorkflowStore for PendingRegistration {
    async fn load(&self, id: &str) -> WorkflowPersistenceResult<Option<WorkflowRun>> {
        Ok(self.inner.load(id)?)
    }
    async fn upsert(&self, row: WorkflowRunUpsert) -> WorkflowPersistenceResult<WorkflowRun> {
        Ok(self.inner.upsert(row)?)
    }
    async fn claim(
        &self,
        id: &str,
        owner: &str,
        lease: Duration,
    ) -> WorkflowPersistenceResult<WorkflowLeaseClaim> {
        Ok(self.inner.claim(id, owner, lease)?)
    }
    async fn renew(
        &self,
        id: &str,
        owner: &str,
        lease: Duration,
    ) -> WorkflowPersistenceResult<bool> {
        if self.refuse_renewal && self.pending.load(Ordering::SeqCst) {
            return Err(refusal());
        }
        Ok(self.inner.renew(id, owner, lease)?)
    }
    async fn compare_and_swap(
        &self,
        row: WorkflowRunUpsert,
        rev: u64,
        owner: &str,
        lease: Duration,
    ) -> WorkflowPersistenceResult<Option<WorkflowRun>> {
        if !row.child_run_ids.is_empty() {
            self.pending.store(true, Ordering::SeqCst);
            self.started.notify_one();
            return std::future::pending().await;
        }
        Ok(self.inner.compare_and_swap(row, rev, owner, lease)?)
    }
}
#[derive(Default)]
struct PendingChild {
    cancelled: parking_lot::Mutex<Vec<String>>,
}
#[async_trait]
impl AsyncWorkflowExecutor for PendingChild {
    async fn execute_async(
        &self,
        _: WorkflowChildRequest,
        _: CancellationToken,
        registration: Arc<dyn AsyncWorkflowChildRegistration>,
    ) -> WorkflowPersistenceResult<WorkflowChildResult> {
        registration.register_async("pending-child".into()).await?;
        unreachable!("registration remains pending")
    }
    async fn cancel_children(&self, ids: &[String]) {
        self.cancelled.lock().extend_from_slice(ids);
    }
}
async fn pending_child_case(renewal: bool) {
    let store = Arc::new(PendingRegistration {
        inner: tests::MemoryStore::default(),
        started: tokio::sync::Notify::new(),
        pending: std::sync::atomic::AtomicBool::new(false),
        refuse_renewal: renewal,
    });
    let executor = Arc::new(PendingChild::default());
    let engine = WorkflowEngine::new_async(store.clone(), executor.clone())
        .with_lease_duration(Duration::from_millis(30));
    let mut definition = tests::definition();
    definition.phases.truncate(1);
    definition.phases[0].agent_ids.truncate(1);
    engine
        .initialise_async("run".into(), &definition, json!({}), None)
        .await
        .unwrap();
    let cancel = CancellationToken::new();
    let control = async {
        store.started.notified().await;
        if !renewal {
            cancel.cancel();
        }
    };
    let drive = engine.drive_async("run", &definition, cancel.clone());
    let (result, _) = tokio::time::timeout(Duration::from_secs(1), async {
        tokio::join!(drive, control)
    })
    .await
    .unwrap();
    if renewal {
        assert!(matches!(
            innermost(&result.unwrap_err()),
            WorkflowPersistenceError::Ledger(_)
        ));
    } else {
        result.unwrap();
        let run = store.inner.load("run").unwrap().unwrap();
        assert_eq!(run.status, WorkflowRunStatus::Interrupted);
        assert!(
            run.child_run_ids.is_empty(),
            "uncommitted ids are cancellation targets, not persisted children"
        );
    }
    assert_eq!(*executor.cancelled.lock(), vec!["pending-child"]);
}
#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn heartbeat_loss_cancels_a_child_with_pending_registration() {
    pending_child_case(true).await;
}
#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn host_cancellation_cancels_a_child_with_pending_registration() {
    pending_child_case(false).await;
}
