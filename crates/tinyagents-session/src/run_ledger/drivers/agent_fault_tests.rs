use super::*;
use crate::testkit::MemoryStorage;
use serde_json::Value;
use std::sync::atomic::{AtomicBool, Ordering};
use tinyagents_storage_kit::{
    CollectionSpec, DocumentStore, Page, Precondition, ReadCheck, Version, Versioned, WriteOp,
    WriteResult,
};

#[derive(Debug)]
struct NoPostCommitReads {
    inner: Arc<dyn DocumentStore>,
    committed: AtomicBool,
    telemetry_race: Option<Value>,
    raced: AtomicBool,
}
type NativeResult<T> = std::result::Result<T, StorageError>;

#[async_trait::async_trait]
impl DocumentStore for NoPostCommitReads {
    fn capabilities(&self) -> tinystoragedrivers_types::Capabilities {
        self.inner.capabilities()
    }
    async fn ensure_collection(&self, spec: &CollectionSpec) -> NativeResult<()> {
        self.inner.ensure_collection(spec).await
    }
    async fn get(&self, collection: &str, id: &str) -> NativeResult<Option<Versioned<Value>>> {
        if collection == storage::TELEMETRY && self.committed.load(Ordering::SeqCst) {
            return Err(StorageError::unavailable(
                "fixture expires reads immediately after committed write",
            ));
        }
        self.inner.get(collection, id).await
    }
    async fn put(
        &self,
        collection: &str,
        id: &str,
        doc: Value,
        precondition: Precondition,
    ) -> NativeResult<Version> {
        let result = self.inner.put(collection, id, doc, precondition).await?;
        if collection == storage::AGENTS {
            self.committed.store(true, Ordering::SeqCst);
        }
        Ok(result)
    }
    async fn delete(
        &self,
        collection: &str,
        id: &str,
        precondition: Precondition,
    ) -> NativeResult<bool> {
        self.inner.delete(collection, id, precondition).await
    }
    async fn query(&self, collection: &str, query: &Query) -> NativeResult<Page<Versioned<Value>>> {
        self.inner.query(collection, query).await
    }
    async fn count(&self, collection: &str, filter: &Filter) -> NativeResult<u64> {
        self.inner.count(collection, filter).await
    }
    async fn delete_where(&self, collection: &str, filter: &Filter) -> NativeResult<u64> {
        self.inner.delete_where(collection, filter).await
    }
    async fn claim(
        &self,
        collection: &str,
        filter: &Filter,
        sort: &[Sort],
        patch: &Value,
    ) -> NativeResult<Option<Versioned<Value>>> {
        self.inner.claim(collection, filter, sort, patch).await
    }
    async fn drop_collection(&self, collection: &str) -> NativeResult<()> {
        self.inner.drop_collection(collection).await
    }
    async fn atomic_batch_checked(
        &self,
        checks: Vec<ReadCheck>,
        writes: Vec<WriteOp>,
    ) -> NativeResult<Vec<WriteResult>> {
        let changed = writes.iter().any(|write| matches!(write, WriteOp::Put { collection, .. } if collection == storage::AGENTS));
        if changed
            && !self.raced.swap(true, Ordering::SeqCst)
            && let Some(value) = &self.telemetry_race
        {
            self.inner
                .put(
                    storage::TELEMETRY,
                    &storage::key(&["run"]).unwrap(),
                    value.clone(),
                    Precondition::None,
                )
                .await?;
        }
        let result = self.inner.atomic_batch_checked(checks, writes).await?;
        if changed {
            self.committed.store(true, Ordering::SeqCst);
        }
        Ok(result)
    }
}

fn wrapped(native: &ScopedStorage) -> DriverRunLedger {
    let storage = tinyagents_storage_kit::Storage::new(
        Arc::new(NoPostCommitReads {
            inner: native.documents().clone(),
            committed: AtomicBool::new(false),
            telemetry_race: None,
            raced: AtomicBool::new(false),
        }),
        native.streams().clone(),
        native.kv().clone(),
        native.blobs().clone(),
        native.queues().clone(),
        native.secrets().clone(),
    );
    DriverRunLedger::new(ScopedStorage::new(storage, native.scope().clone()))
}

#[tokio::test]
async fn agent_upsert_returns_its_commit_without_a_fallible_postcommit_telemetry_read() {
    let backend = MemoryStorage::new();
    let native = backend.for_scope(&Scope::new("alice").unwrap()).unwrap();
    let ledger = wrapped(&native);
    assert!(ledger.upsert_agent_run(run("run")).await.is_ok());
    assert!(
        DriverRunLedger::new(native)
            .get_agent_run("run")
            .await
            .unwrap()
            .is_some()
    );
}

#[tokio::test]
async fn status_and_event_return_their_commit_without_a_fallible_postcommit_telemetry_read() {
    let backend = MemoryStorage::new();
    let native = backend.for_scope(&Scope::new("alice").unwrap()).unwrap();
    DriverRunLedger::new(native.clone())
        .upsert_agent_run(run("run"))
        .await
        .unwrap();
    let ledger = wrapped(&native);
    assert!(
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
                    payload: serde_json::json!({})
                })
            )
            .await
            .is_ok()
    );
    assert_eq!(
        DriverRunLedger::new(native)
            .get_agent_run("run")
            .await
            .unwrap()
            .unwrap()
            .status,
        AgentRunStatus::Completed
    );
}

#[tokio::test]
async fn concurrent_telemetry_change_retries_before_agent_commit_and_returns_validated_rollup() {
    for transition in [false, true] {
        let backend = MemoryStorage::new();
        let native = backend.for_scope(&Scope::new("alice").unwrap()).unwrap();
        let seed = DriverRunLedger::new(native.clone());
        seed.upsert_agent_run(run("run")).await.unwrap();
        seed.upsert_run_telemetry(RunTelemetryUpsert {
            run_id: "run".into(),
            input_tokens: Some(73),
            ..Default::default()
        })
        .await
        .unwrap();
        let newer = native
            .documents()
            .get(storage::TELEMETRY, &storage::key(&["run"]).unwrap())
            .await
            .unwrap()
            .unwrap()
            .doc;
        seed.upsert_run_telemetry(RunTelemetryUpsert {
            run_id: "run".into(),
            input_tokens: Some(1),
            ..Default::default()
        })
        .await
        .unwrap();
        let documents = Arc::new(NoPostCommitReads {
            inner: native.documents().clone(),
            committed: AtomicBool::new(false),
            telemetry_race: Some(newer),
            raced: AtomicBool::new(false),
        });
        let ledger =
            DriverRunLedger::new(crate::testkit::with_documents(&native, documents.clone()));
        let committed = if transition {
            match ledger
                .transition_agent_run_status_from(
                    "run",
                    AgentRunStatus::Running,
                    AgentRunStatus::Completed,
                    None,
                    None,
                    Some(RunEventAppend {
                        run_id: "run".into(),
                        event_type: "done".into(),
                        payload: serde_json::json!({}),
                    }),
                )
                .await
                .unwrap()
            {
                RunTransition::Applied(row) => row,
                other => panic!("expected applied transition, got {other:?}"),
            }
        } else {
            ledger.upsert_agent_run(run("run")).await.unwrap()
        };
        assert!(documents.raced.load(Ordering::SeqCst));
        assert!(documents.committed.load(Ordering::SeqCst));
        assert_eq!(committed.telemetry.unwrap().input_tokens, 73);
        assert_eq!(
            seed.list_recent_run_events(&RunEventListRequest {
                run_id: "run".into(),
                ..Default::default()
            })
            .await
            .unwrap()
            .events
            .len(),
            usize::from(transition)
        );
    }
}
