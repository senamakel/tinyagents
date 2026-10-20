use super::*;
use serde_json::json;
use tinyagents_session::run_ledger::WorkflowRunStatus;
use tinystoragedrivers_testkit::MemoryBackend;

fn row() -> WorkflowRunUpsert {
    WorkflowRunUpsert {
        id: "workflow".into(),
        definition_id: "definition".into(),
        parent_thread_id: None,
        input: json!({"input":1}),
        phase_states: json!({}),
        child_run_ids: vec![],
        status: WorkflowRunStatus::Running,
        summary: None,
        started_at: None,
        completed_at: None,
    }
}

#[tokio::test(flavor = "current_thread")]
async fn authoritative_adapter_loads_persisted_ledger_without_runtime_bridge() {
    let ledger_backend = MemoryBackend::new();
    let _workspace = tempfile::tempdir().unwrap();
    let ledger = Arc::new(crate::test_support::ledger_in(&ledger_backend));
    let expected = ledger.upsert_workflow_run(row()).await.unwrap();
    let store: Arc<dyn AsyncWorkflowStore> = Arc::new(LedgerWorkflowStore::new(ledger));
    assert_eq!(store.load("workflow").await.unwrap(), Some(expected));
}

#[tokio::test(flavor = "current_thread")]
async fn independent_adapters_preserve_claim_revision_and_stale_owner_rules() {
    let ledger_backend = MemoryBackend::new();
    let _workspace = tempfile::tempdir().unwrap();
    let first = LedgerWorkflowStore::new(Arc::new(crate::test_support::ledger_in(&ledger_backend)));
    let second =
        LedgerWorkflowStore::new(Arc::new(crate::test_support::ledger_in(&ledger_backend)));
    let initial = first.upsert(row()).await.unwrap();
    assert_eq!(initial.revision, 0);
    let WorkflowLeaseClaim::Acquired(claimed) = first
        .claim("workflow", "owner-a", Duration::from_secs(60))
        .await
        .unwrap()
    else {
        panic!("claim refused")
    };
    assert!(matches!(
        second
            .claim("workflow", "owner-b", Duration::from_secs(60))
            .await
            .unwrap(),
        WorkflowLeaseClaim::Busy(_)
    ));
    assert!(
        !second
            .renew("workflow", "owner-b", Duration::from_secs(60))
            .await
            .unwrap()
    );
    assert!(
        second
            .compare_and_swap(row(), claimed.revision, "owner-b", Duration::from_secs(60))
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        first
            .renew("workflow", "owner-a", Duration::from_secs(60))
            .await
            .unwrap()
    );
    let updated = first
        .compare_and_swap(row(), claimed.revision, "owner-a", Duration::from_secs(60))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(updated.revision, claimed.revision + 1);
    assert!(
        first
            .compare_and_swap(row(), claimed.revision, "owner-a", Duration::from_secs(60))
            .await
            .unwrap()
            .is_none()
    );
}

struct AuthoritativeRegistration;
#[async_trait]
impl AsyncWorkflowChildRegistration for AuthoritativeRegistration {
    async fn register_async(&self, _: String) -> WorkflowPersistenceResult<()> {
        Ok(())
    }
}

#[tokio::test(flavor = "current_thread")]
async fn authoritative_registration_refuses_synchronous_success() {
    let registration: Arc<dyn AsyncWorkflowChildRegistration> = Arc::new(AuthoritativeRegistration);
    assert!(matches!(
        registration.register("child".into()),
        Err(WorkflowPersistenceError::AsyncRegistrationRequired)
    ));
    registration.register_async("child".into()).await.unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn explicit_driver_scopes_isolate_workflows_without_local_files() {
    use tinyagents_session::run_ledger::DriverRunLedger;
    use tinyagents_storage_kit::{Scope, StorageBackend};
    let backend = MemoryBackend::new();
    let first = LedgerWorkflowStore::new(Arc::new(DriverRunLedger::new(
        backend
            .for_scope(&Scope::new("profile-a").unwrap())
            .unwrap(),
    )));
    let second = LedgerWorkflowStore::new(Arc::new(DriverRunLedger::new(
        backend
            .for_scope(&Scope::new("profile-b").unwrap())
            .unwrap(),
    )));
    first.upsert(row()).await.unwrap();
    assert!(second.load("workflow").await.unwrap().is_none());
    assert!(first.load("workflow").await.unwrap().is_some());
}

#[path = "persistence_refusal_tests.rs"]
mod refusal_tests;

#[tokio::test(flavor = "current_thread")]
async fn invalid_lease_duration_refuses_before_touching_the_ledger() {
    let ledger_backend = MemoryBackend::new();
    let _workspace = tempfile::tempdir().unwrap();
    let store = LedgerWorkflowStore::new(Arc::new(crate::test_support::ledger_in(&ledger_backend)));
    let lease = Duration::MAX;
    assert!(matches!(
        store.claim("workflow", "owner", lease).await,
        Err(WorkflowPersistenceError::LeaseDuration(_))
    ));
    assert!(matches!(
        store.compare_and_swap(row(), 0, "owner", lease).await,
        Err(WorkflowPersistenceError::LeaseDuration(_))
    ));
    assert!(matches!(
        store.renew("workflow", "owner", lease).await,
        Err(WorkflowPersistenceError::LeaseDuration(_))
    ));
    // Nothing reached the ledger: the refused claim created no workflow.
    assert!(store.load("workflow").await.unwrap().is_none());
}

struct LegacyCallback(std::sync::atomic::AtomicUsize);
impl WorkflowChildRegistration for LegacyCallback {
    fn register(&self, child_id: String) -> Result<(), OrchestrationError> {
        if child_id == "refused" {
            return Err(OrchestrationError("legacy callback refused".into()));
        }
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
}

#[tokio::test(flavor = "current_thread")]
async fn explicit_legacy_registration_preserves_callback_results() {
    let callback = Arc::new(LegacyCallback(std::sync::atomic::AtomicUsize::new(0)));
    let adapter = LegacyWorkflowRegistration::new(callback.clone());
    adapter.register("sync-child".into()).unwrap();
    adapter.register_async("async-child".into()).await.unwrap();
    assert_eq!(callback.0.load(std::sync::atomic::Ordering::SeqCst), 2);
    assert!(matches!(
        adapter.register_async("refused".into()).await,
        Err(WorkflowPersistenceError::Legacy(_))
    ));
    assert!(matches!(
        adapter.register("refused".into()),
        Err(WorkflowPersistenceError::Legacy(_))
    ));
    assert_eq!(callback.0.load(std::sync::atomic::Ordering::SeqCst), 2);
}

#[tokio::test(flavor = "current_thread")]
async fn expired_owner_takeover_fences_prior_owner_without_sleeping() {
    let ledger_backend = MemoryBackend::new();
    let _workspace = tempfile::tempdir().unwrap();
    let ledger = Arc::new(crate::test_support::ledger_in(&ledger_backend));
    ledger.upsert_workflow_run(row()).await.unwrap();
    let WorkflowLeaseClaim::Acquired(expired) = ledger
        .try_claim_workflow_run("workflow", "old-owner", chrono::Duration::seconds(-1))
        .await
        .unwrap()
    else {
        panic!("initial claim refused")
    };
    let store = LedgerWorkflowStore::new(ledger);
    let WorkflowLeaseClaim::Acquired(replacement) = store
        .claim("workflow", "new-owner", Duration::from_secs(60))
        .await
        .unwrap()
    else {
        panic!("expired takeover refused")
    };
    assert!(replacement.revision > expired.revision);
    assert!(
        !store
            .renew("workflow", "old-owner", Duration::from_secs(60))
            .await
            .unwrap()
    );
    assert!(
        store
            .compare_and_swap(
                row(),
                replacement.revision,
                "old-owner",
                Duration::from_secs(60)
            )
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        store
            .renew("workflow", "new-owner", Duration::from_secs(60))
            .await
            .unwrap()
    );
}
