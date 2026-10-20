//! Contract tests for Claude Code storage-port persistence and imports.

use super::*;
use tinyagents_storage_kit::{Scope, StorageBackend};
use tinystoragedrivers_api::DocumentStore;
use tinystoragedrivers_testkit::{Failing, FaultPlan, MemoryBackend};
use tinystoragedrivers_types::ErrorKind;

fn new_store() -> (tinyagents_storage_kit::ScopedStorage, Arc<SessionStore>) {
    let storage = MemoryBackend::new()
        .for_scope(&Scope::new("claude-code-tests").unwrap())
        .unwrap();
    let store = Arc::new(SessionStore::new(storage.documents()).unwrap());
    (storage, store)
}

fn another_scope(name: &str) -> (tinyagents_storage_kit::ScopedStorage, Arc<SessionStore>) {
    let storage = MemoryBackend::new()
        .for_scope(&Scope::new(name).unwrap())
        .unwrap();
    let store = Arc::new(SessionStore::new(storage.documents()).unwrap());
    (storage, store)
}

#[test]
fn uuid_v4_format() {
    let id = generate_uuid_v4();
    assert!(is_uuid_v4(&id));
}

#[test]
fn rejects_non_v4() {
    assert!(!is_uuid_v4("not-a-uuid"));
    assert!(!is_uuid_v4("cc_abc123"));
    assert!(!is_uuid_v4("00000000-0000-1000-8000-000000000000"));
}

#[tokio::test]
async fn session_and_delivered_turns_roundtrip_through_host_storage() {
    let (_storage, store) = new_store();
    assert_eq!(store.get("thread").await.unwrap(), None);
    store.set("thread", "session-a").await.unwrap();
    store
        .record_delivered("thread", &["one".into(), "two".into()])
        .await
        .unwrap();
    assert_eq!(
        store.get("thread").await.unwrap().as_deref(),
        Some("session-a")
    );
    assert_eq!(
        store.delivered("thread").await.unwrap(),
        ["one", "two"].into_iter().map(str::to_owned).collect()
    );
}

#[tokio::test]
async fn replacement_session_clears_delivered_fingerprints() {
    let (_storage, store) = new_store();
    store.set("thread", "old").await.unwrap();
    store
        .record_delivered("thread", &["turn".into()])
        .await
        .unwrap();
    store.set("thread", "old").await.unwrap();
    assert!(store.delivered("thread").await.unwrap().contains("turn"));
    store.set("thread", "new").await.unwrap();
    assert!(store.delivered("thread").await.unwrap().is_empty());
}

#[tokio::test]
async fn remove_if_only_clears_the_expected_session_and_its_deliveries() {
    let (_storage, store) = new_store();
    store.set("thread", "session-a").await.unwrap();
    store
        .record_delivered("thread", &["turn".into()])
        .await
        .unwrap();
    assert!(!store.remove_if("thread", "session-b").await.unwrap());
    assert_eq!(
        store.get("thread").await.unwrap().as_deref(),
        Some("session-a")
    );
    assert!(store.remove_if("thread", "session-a").await.unwrap());
    assert_eq!(store.get("thread").await.unwrap(), None);
    assert!(store.delivered("thread").await.unwrap().is_empty());
}

#[tokio::test]
async fn claim_is_exclusive_released_on_failure_and_committed_after_acceptance() {
    let (_storage, store) = new_store();
    let fingerprints = vec!["turn".to_owned()];
    let (already, first) = store.claim_delivered("t", &fingerprints).await.unwrap();
    assert!(already.is_empty());
    let (already, second) = store.claim_delivered("t", &fingerprints).await.unwrap();
    assert!(already.contains("turn"));
    drop(second);
    drop(first);
    assert!(
        store
            .claim_delivered("t", &fingerprints)
            .await
            .unwrap()
            .0
            .is_empty()
    );
    store.record_delivered("t", &fingerprints).await.unwrap();
    assert!(store.delivered("t").await.unwrap().contains("turn"));
}

#[tokio::test]
async fn reservations_coordinate_for_same_handle_but_not_different_scopes() {
    let (storage_a, a) = new_store();
    let b = Arc::new(SessionStore::new(storage_a.documents()).unwrap());
    let (_storage_other, other) = another_scope("separate-claude-code-scope");
    let fp = ["same-fingerprint".to_owned()];
    let (_, claim) = a.claim_delivered("thread", &fp).await.unwrap();
    assert!(
        b.claim_delivered("thread", &fp)
            .await
            .unwrap()
            .0
            .contains(&fp[0])
    );
    assert!(
        other
            .claim_delivered("thread", &fp)
            .await
            .unwrap()
            .0
            .is_empty()
    );
    drop(claim);
}

#[tokio::test]
async fn failed_session_removal_keeps_the_mapping_and_delivered_turns() {
    let storage = MemoryBackend::new()
        .for_scope(&Scope::new("claude-code-remove-failure").unwrap())
        .unwrap();
    let faults = FaultPlan::new();
    let documents: Arc<dyn DocumentStore> = Arc::new(Failing::with_plan(
        Arc::clone(storage.documents()),
        faults.clone(),
    ));
    let store = SessionStore::new(&documents).unwrap();
    store.set("thread", "session-a").await.unwrap();
    store
        .record_delivered("thread", &["turn".to_owned()])
        .await
        .unwrap();

    faults.fail_operation_nth("documents.put_with", 1, ErrorKind::Backend);
    assert!(store.remove_if("thread", "session-a").await.is_err());
    faults.clear();

    assert_eq!(
        store.get("thread").await.unwrap().as_deref(),
        Some("session-a")
    );
    assert!(store.delivered("thread").await.unwrap().contains("turn"));
}

#[tokio::test]
async fn failed_delivery_write_releases_the_reservation_for_retry() {
    let storage = MemoryBackend::new()
        .for_scope(&Scope::new("claude-code-delivery-failure").unwrap())
        .unwrap();
    let faults = FaultPlan::new();
    let documents: Arc<dyn DocumentStore> = Arc::new(Failing::with_plan(
        Arc::clone(storage.documents()),
        faults.clone(),
    ));
    let store = Arc::new(SessionStore::new(&documents).unwrap());
    let fingerprint = ["turn".to_owned()];
    let (already, claim) = store.claim_delivered("thread", &fingerprint).await.unwrap();
    assert!(already.is_empty());

    faults.fail_operation_nth("documents.put_with", 1, ErrorKind::Backend);
    assert!(
        store
            .record_delivered("thread", &fingerprint)
            .await
            .is_err()
    );
    drop(claim);
    faults.clear();

    let (already, retry) = store.claim_delivered("thread", &fingerprint).await.unwrap();
    assert!(already.is_empty());
    assert!(store.delivered("thread").await.unwrap().is_empty());
    drop(retry);
}

#[tokio::test]
async fn legacy_import_merges_matching_session_and_never_overwrites_newer_state() {
    let (_storage, store) = new_store();
    let old_release_fixture = include_bytes!("fixtures/claude-code-sessions.json");
    store.set("same", "uuid-a").await.unwrap();
    store
        .record_delivered("same", &["new-turn".into()])
        .await
        .unwrap();
    store.set("different", "uuid-current").await.unwrap();
    assert_eq!(
        import_legacy_sessions(&store, old_release_fixture)
            .await
            .unwrap(),
        2
    );
    assert_eq!(
        store.get("different").await.unwrap().as_deref(),
        Some("uuid-current")
    );
    assert_eq!(store.get("new").await.unwrap().as_deref(), Some("uuid-new"));
    assert!(store.delivered("same").await.unwrap().contains("old-turn"));
    assert!(store.delivered("same").await.unwrap().contains("new-turn"));
    assert_eq!(store.get("orphan").await.unwrap(), None);
    assert!(
        store
            .delivered("orphan")
            .await
            .unwrap()
            .contains("orphan-turn")
    );
    import_legacy_sessions(&store, old_release_fixture)
        .await
        .unwrap();
    assert_eq!(
        import_legacy_sessions(&store, old_release_fixture)
            .await
            .unwrap(),
        0
    );
    assert_eq!(store.delivered("same").await.unwrap().len(), 2);
}

#[tokio::test]
async fn malformed_legacy_bytes_are_rejected() {
    let (_storage, store) = new_store();
    assert!(import_legacy_sessions(&store, b"not json").await.is_err());
    assert_eq!(store.get("thread").await.unwrap(), None);
}
