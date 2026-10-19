use super::*;
use tinystoragedrivers_core::{MemoryStorage, StorageBackend};

fn provider() -> DriverSessionStores {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemoryStorage::new());
    DriverSessionStores::new(backend, Scope::local()).unwrap()
}

#[tokio::test]
async fn partial_journal_copy_resumes_after_matching_ordered_prefix() {
    let source = provider();
    let destination = provider();
    let source_stores = source.try_for_agent("agent").unwrap();
    let destination_stores = destination.try_for_agent("agent").unwrap();
    let stream = "harness.journal/run";
    for value in ["one", "two", "three"] {
        source_stores
            .journal
            .append(stream, serde_json::json!({"value": value}))
            .await
            .unwrap();
    }
    for value in ["one", "two"] {
        destination_stores
            .journal
            .append(stream, serde_json::json!({"value": value}))
            .await
            .unwrap();
    }

    workspace_migration::copy_agent_storage(&source, &destination, "agent").unwrap();
    let entries = destination_stores
        .journal
        .read_from(stream, 0)
        .await
        .unwrap();
    assert_eq!(
        entries
            .into_iter()
            .map(|(_, value)| value)
            .collect::<Vec<_>>(),
        [
            serde_json::json!({"value": "one"}),
            serde_json::json!({"value": "two"}),
            serde_json::json!({"value": "three"}),
        ]
    );
}

#[tokio::test]
async fn differing_journal_history_refuses_import_without_overwriting_target() {
    let source = provider();
    let destination = provider();
    let source_stores = source.try_for_agent("agent").unwrap();
    let destination_stores = destination.try_for_agent("agent").unwrap();
    let stream = "harness.journal/run";
    source_stores
        .journal
        .append(stream, serde_json::json!({"value": "legacy"}))
        .await
        .unwrap();
    destination_stores
        .journal
        .append(stream, serde_json::json!({"value": "newer"}))
        .await
        .unwrap();

    assert!(workspace_migration::copy_agent_storage(&source, &destination, "agent").is_err());
    assert_eq!(
        destination_stores
            .journal
            .read_from(stream, 0)
            .await
            .unwrap()[0]
            .1,
        serde_json::json!({"value": "newer"})
    );
}
