use super::*;
use crate::testkit::MemoryStorage;
use serde_json::json;
use tinyagents_storage_kit::{CollectionSpec, Fence, Scope};
use tinystoragedrivers_api::StorageBackend;

fn request(id: &str) -> CreateConversationThread {
    CreateConversationThread {
        id: id.into(),
        title: "Initial".into(),
        created_at: "2026-01-01T00:00:00Z".into(),
        parent_thread_id: None,
        labels: None,
        personality_id: None,
        working_dir: None,
    }
}

async fn missing_declaration(backend: &dyn StorageBackend) {
    backend.prepare_schema().await.unwrap();
    let scope = Scope::new("alice").unwrap();
    let scoped = backend.for_scope(&scope).unwrap();
    let docs = scoped.documents().clone();
    let store = DriverConversationStore::new_validate_only(scoped);
    let error = store
        .get_thread("missing")
        .await
        .expect_err("explicit admission must require a persisted declaration before read");
    assert!(matches!(error, ConversationError::Driver(e) if e.kind() == ErrorKind::NotFound));
    assert_eq!(docs.count(COLLECTION, &Filter::All).await.unwrap(), 0);
}

async fn prepared_fenced(backend: &dyn StorageBackend) {
    backend.prepare_schema().await.unwrap();
    let cluster = Scope::new("cluster").unwrap();
    let profile = Scope::new("alice").unwrap();
    let raw = backend.for_scope(&profile).unwrap();
    raw.documents()
        .ensure_collection(&CollectionSpec::new(COLLECTION))
        .await
        .unwrap();
    let owner = backend.for_scope(&cluster).unwrap();
    owner
        .documents()
        .ensure_collection(&CollectionSpec::new("leases"))
        .await
        .unwrap();
    owner
        .documents()
        .put("leases", "alice", json!({"epoch": 1}), Precondition::None)
        .await
        .unwrap();
    let fence = Fence::epoch(cluster, "leases", "alice", "epoch", 1);
    let actor = backend.existing_schema().unwrap();
    let store = DriverConversationStore::new_validate_only(
        actor.for_scope_fenced(&profile, &fence).unwrap(),
    );
    let thread = store.ensure_thread(request("one")).await.unwrap();
    assert_eq!(store.get_thread("one").await.unwrap(), Some(thread.clone()));
    assert!(store.get_messages("one").await.unwrap().is_empty());
    assert_eq!(store.list_threads().await.unwrap(), vec![thread.clone()]);
    let sibling = DriverConversationStore::new_validate_only(
        actor.for_scope(&Scope::new("bob").unwrap()).unwrap(),
    );
    assert!(sibling.get_thread("one").await.unwrap().is_none());
    owner
        .documents()
        .put("leases", "alice", json!({"epoch": 2}), Precondition::None)
        .await
        .unwrap();
    let error = store
        .update_thread_title("one", "stale", "now")
        .await
        .unwrap_err();
    assert!(matches!(error, ConversationError::Driver(e) if e.kind() == ErrorKind::Unavailable));
    assert_eq!(
        DriverConversationStore::new(raw)
            .get_thread("one")
            .await
            .unwrap(),
        Some(thread)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn memory_explicit_admission_refuses_missing_declaration_before_read() {
    missing_declaration(&MemoryStorage::new()).await;
}
#[tokio::test(flavor = "current_thread")]
async fn sqlite_explicit_admission_refuses_missing_declaration_before_read() {
    let dir = tempfile::tempdir().unwrap();
    let backend = MemoryStorage::new();
    missing_declaration(&backend).await;
}
#[tokio::test(flavor = "current_thread")]
async fn memory_prepared_actor_retains_original_native_fence_and_scope() {
    prepared_fenced(&MemoryStorage::new()).await;
}
#[tokio::test(flavor = "current_thread")]
async fn sqlite_prepared_actor_retains_original_native_fence_and_scope() {
    let dir = tempfile::tempdir().unwrap();
    let backend = MemoryStorage::new();
    prepared_fenced(&backend).await;
}

#[path = "driver_schema_fixtures.rs"]
mod fixtures;

#[tokio::test(flavor = "current_thread")]
async fn validation_is_repeated_and_preserves_native_refusal_source_without_actor_ddl() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let backend = MemoryStorage::new();
    backend.prepare_schema().await.unwrap();
    let scope = Scope::new("alice").unwrap();
    let raw = backend.for_scope(&scope).unwrap();
    DriverConversationStore::prepare_schema(raw.documents().as_ref())
        .await
        .unwrap();
    let restricted = backend
        .existing_schema()
        .unwrap()
        .for_scope(&scope)
        .unwrap();
    let marker = Arc::new(std::io::Error::other("native validation original cause"));
    let docs = Arc::new(fixtures::RecordingDocs {
        inner: restricted.documents().clone(),
        validates: AtomicUsize::new(0),
        declares: AtomicUsize::new(0),
        refuse: AtomicBool::new(false),
        marker: marker.clone(),
    });
    let mut store = DriverConversationStore::new_validate_only(restricted);
    store.docs = docs.clone();
    assert!(store.get_thread("one").await.unwrap().is_none());
    let first = docs.validates.load(Ordering::SeqCst);
    assert!(first > 0);
    docs.refuse.store(true, Ordering::SeqCst);
    let results = [
        store.get_thread("one").await.map(|_| ()),
        store.list_threads().await.map(|_| ()),
        store
            .search_cross_thread_messages("", 0, None)
            .await
            .map(|_| ()),
        store.ensure_thread(request("one")).await.map(|_| ()),
        store.purge_threads().await.map(|_| ()),
    ];
    for result in results {
        let ConversationError::Driver(error) = result.unwrap_err() else {
            panic!("native refusal flattened");
        };
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        let original = std::error::Error::source(error.as_ref())
            .unwrap()
            .downcast_ref::<Arc<std::io::Error>>()
            .unwrap();
        assert!(Arc::ptr_eq(original, &marker));
    }
    assert!(docs.validates.load(Ordering::SeqCst) >= first + 5);
    assert_eq!(docs.declares.load(Ordering::SeqCst), 0);
    assert_eq!(
        raw.documents()
            .count(COLLECTION, &Filter::All)
            .await
            .unwrap(),
        0
    );
}

#[tokio::test(flavor = "current_thread")]
async fn native_required_feature_validation_accepts_compatible_superset_declarations() {
    let backend = MemoryStorage::new();
    let scope = Scope::new("alice").unwrap();
    let raw = backend.for_scope(&scope).unwrap();
    raw.documents()
        .ensure_collection(&CollectionSpec::new(COLLECTION).ttl("until"))
        .await
        .unwrap();
    let store = DriverConversationStore::new_validate_only(
        backend
            .existing_schema()
            .unwrap()
            .for_scope(&scope)
            .unwrap(),
    );
    assert!(store.get_thread("missing").await.unwrap().is_none());
    assert_eq!(
        DriverConversationStore::collection_specs(),
        vec![CollectionSpec::new(COLLECTION)]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn schema_preparation_is_explicit_and_restricted_native_handles_refuse_it() {
    let backend = MemoryStorage::new();
    let raw = backend.for_scope(&Scope::new("local").unwrap()).unwrap();
    DriverConversationStore::prepare_schema(raw.documents().as_ref())
        .await
        .unwrap();
    let actor = backend
        .existing_schema()
        .unwrap()
        .for_scope(&Scope::new("local").unwrap())
        .unwrap();
    let error = DriverConversationStore::prepare_schema(actor.documents().as_ref())
        .await
        .unwrap_err();
    assert_eq!(
        error.kind(),
        ErrorKind::Unsupported(tinystoragedrivers_types::Capability::ExistingSchema)
    );
    actor
        .documents()
        .validate_collection_spec(&CollectionSpec::new(COLLECTION))
        .await
        .unwrap();
    assert_eq!(
        actor
            .documents()
            .count(COLLECTION, &Filter::All)
            .await
            .unwrap(),
        0
    );
}
