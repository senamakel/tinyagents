use super::*;
use serde_json::Value;
use tinyagents_session::run_ledger::DriverRunLedger;
use tinyagents_storage_kit::{
    Capabilities, Capability, CollectionSpec, DocumentStore, ErrorKind, Filter, Page, Precondition,
    Query, Scope, ScopedStorage, Sort, Storage, StorageBackend, StorageError, Version, Versioned,
};
use tinystoragedrivers_testkit::MemoryBackend;

#[derive(Debug)]
struct RefusedDocuments(ErrorKind);

impl RefusedDocuments {
    fn refused<T>(&self) -> tinyagents_storage_kit::Result<T> {
        Err(StorageError::new(self.0, "admitted backend refused"))
    }
}

#[async_trait]
impl DocumentStore for RefusedDocuments {
    async fn drop_collection(&self, _: &str) -> tinyagents_storage_kit::Result<()> {
        self.refused()
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }
    async fn ensure_collection(&self, _: &CollectionSpec) -> tinyagents_storage_kit::Result<()> {
        self.refused()
    }
    async fn get(
        &self,
        _: &str,
        _: &str,
    ) -> tinyagents_storage_kit::Result<Option<Versioned<Value>>> {
        self.refused()
    }
    async fn put(
        &self,
        _: &str,
        _: &str,
        _: Value,
        _: Precondition,
    ) -> tinyagents_storage_kit::Result<Version> {
        self.refused()
    }
    async fn delete(
        &self,
        _: &str,
        _: &str,
        _: Precondition,
    ) -> tinyagents_storage_kit::Result<bool> {
        self.refused()
    }
    async fn query(
        &self,
        _: &str,
        _: &Query,
    ) -> tinyagents_storage_kit::Result<Page<Versioned<Value>>> {
        self.refused()
    }
    async fn count(&self, _: &str, _: &Filter) -> tinyagents_storage_kit::Result<u64> {
        self.refused()
    }
    async fn delete_where(&self, _: &str, _: &Filter) -> tinyagents_storage_kit::Result<u64> {
        self.refused()
    }
    async fn claim(
        &self,
        _: &str,
        _: &Filter,
        _: &[Sort],
        _: &Value,
    ) -> tinyagents_storage_kit::Result<Option<Versioned<Value>>> {
        self.refused()
    }
}

#[tokio::test(flavor = "current_thread")]
async fn authoritative_refusals_preserve_native_kind_and_never_create_sqlite() {
    let ambient_db = std::env::current_dir()
        .unwrap()
        .join("session_db/sessions.db");
    let existed_before = ambient_db.exists();
    for kind in [
        ErrorKind::Unavailable,
        ErrorKind::Conflict,
        ErrorKind::Unsupported(Capability::Transactions),
    ] {
        let memory = MemoryBackend::new();
        let scope = Scope::new("explicit-profile").unwrap();
        let handles = memory.for_scope(&scope).unwrap();
        let admitted = ScopedStorage::new(
            Storage::new(
                Arc::new(RefusedDocuments(kind)),
                handles.streams().clone(),
                handles.kv().clone(),
                handles.blobs().clone(),
                handles.queues().clone(),
                handles.secrets().clone(),
            ),
            scope,
        );
        let store = LedgerWorkflowStore::new(Arc::new(DriverRunLedger::new(admitted)));
        let outcomes = [
            store.load("workflow").await.map(|_| ()),
            store.upsert(row()).await.map(|_| ()),
            store
                .claim("workflow", "owner", Duration::from_secs(30))
                .await
                .map(|_| ()),
            store
                .compare_and_swap(row(), 0, "owner", Duration::from_secs(30))
                .await
                .map(|_| ()),
            store
                .renew("workflow", "owner", Duration::from_secs(30))
                .await
                .map(|_| ()),
        ];
        for outcome in outcomes {
            let WorkflowPersistenceError::Ledger(LedgerError::Driver(error)) = outcome.unwrap_err()
            else {
                panic!("native error flattened")
            };
            assert_eq!(error.kind(), kind);
            assert_eq!(error.message(), "admitted backend refused");
        }
    }
    assert_eq!(
        ambient_db.exists(),
        existed_before,
        "refusal created an ambient SQLite database"
    );
}
