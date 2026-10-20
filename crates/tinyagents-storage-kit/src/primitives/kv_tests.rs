use super::*;
use crate::Scope;
use tinystoragedrivers_api::StorageBackend;
use tinystoragedrivers_testkit::memory_backend;

#[tokio::test]
async fn update_reads_and_replaces_a_namespaced_value_atomically() {
    let storage = memory_backend()
        .for_scope(&Scope::new("local").unwrap())
        .unwrap();
    storage
        .documents()
        .ensure_collection(&super::super::collection_specs()[1])
        .await
        .unwrap();
    let kv = Kv::ns(storage.documents(), "harness", "claude-code").unwrap();

    assert_eq!(
        kv.update("session", |current: Option<u64>| Ok(Some(
            current.unwrap_or(0) + 1
        )))
        .await
        .unwrap(),
        Some(1)
    );
    assert_eq!(kv.get::<u64>("session").await.unwrap(), Some(1));
    assert_eq!(
        kv.update("session", |current: Option<u64>| Ok(Some(
            current.unwrap() + 1
        )))
        .await
        .unwrap(),
        Some(2)
    );
}
