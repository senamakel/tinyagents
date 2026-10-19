use super::*;
use tinyagents_harness::store::{DriverStore, Store};
use tinystoragedrivers_core::{
    CollectionSpec, DocumentStoreExt, MemoryStorage, Migrations, Precondition, StorageBackend,
};

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

#[tokio::test]
async fn truncated_journal_with_matching_base_resumes_without_recreating_prefix() {
    use tinystoragedrivers_core::StreamStore;

    let source = provider();
    let destination = provider();
    let (source_scope, _) = source.scoped_for("agent").unwrap();
    let (destination_scope, _) = destination.scoped_for("agent").unwrap();
    let stream = "harness.journal/run";
    for value in ["one", "two", "three", "four"] {
        source_scope
            .streams()
            .append(stream, serde_json::json!({"value": value}))
            .await
            .unwrap();
    }
    source_scope
        .streams()
        .truncate_before(stream, 2)
        .await
        .unwrap();
    for value in ["one", "two", "three"] {
        destination_scope
            .streams()
            .append(stream, serde_json::json!({"value": value}))
            .await
            .unwrap();
    }
    destination_scope
        .streams()
        .truncate_before(stream, 2)
        .await
        .unwrap();

    workspace_migration::copy_agent_storage(&source, &destination, "agent").unwrap();
    let info = destination_scope
        .streams()
        .info(stream)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(info.base, 2);
    assert_eq!(info.len, 4);
    assert_eq!(
        destination_scope
            .streams()
            .read_window(stream, 2, 2)
            .await
            .unwrap()
            .into_iter()
            .map(|entry| entry.value)
            .collect::<Vec<_>>(),
        [
            serde_json::json!({"value": "three"}),
            serde_json::json!({"value": "four"}),
        ]
    );
}

#[tokio::test]
async fn copy_imports_only_owned_documents_and_existing_collections() {
    let source = provider();
    let destination = provider();
    let source_stores = source.try_for_agent("agent").unwrap();
    let destination_stores = destination.try_for_agent("agent").unwrap();
    let (source_scope, _) = source.scoped_for("agent").unwrap();
    let (destination_scope, _) = destination.scoped_for("agent").unwrap();

    source_stores
        .kv
        .put("session", "kept", serde_json::json!({"value": 1}))
        .await
        .unwrap();
    let other_owner = DriverStore::with_owner(Arc::clone(source_scope.documents()), "other");
    other_owner
        .put("session", "private", serde_json::json!({"value": 2}))
        .await
        .unwrap();
    source_scope
        .documents()
        .ensure_collection(&CollectionSpec::new("unrelated.collection"))
        .await
        .unwrap();
    source_scope
        .documents()
        .put(
            "unrelated.collection",
            "document",
            serde_json::json!({"value": 3}),
            Precondition::Absent,
        )
        .await
        .unwrap();
    source_scope
        .documents()
        .ensure_collection(&CollectionSpec::new("tsd.migrations"))
        .await
        .unwrap();
    Migrations::new(&source_scope)
        .mark_done("other", "private_step")
        .await
        .unwrap();

    workspace_migration::copy_agent_storage(&source, &destination, "agent").unwrap();

    assert_eq!(
        destination_stores.kv.get("session", "kept").await.unwrap(),
        Some(serde_json::json!({"value": 1}))
    );
    let other_owner = DriverStore::with_owner(Arc::clone(destination_scope.documents()), "other");
    assert_eq!(other_owner.get("session", "private").await.unwrap(), None);
    assert!(
        !Migrations::new(&destination_scope)
            .done("other", "private_step")
            .await
            .unwrap(),
    );
    assert!(
        !destination_scope
            .documents()
            .collections("")
            .await
            .unwrap()
            .iter()
            .any(|collection| collection == "unrelated.collection")
    );
}
