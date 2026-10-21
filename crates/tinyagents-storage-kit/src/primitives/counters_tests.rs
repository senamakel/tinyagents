use super::*;
use tinystoragedrivers_api::{DocumentStore, StorageBackend};
use tinystoragedrivers_testkit::memory_backend;
use tinystoragedrivers_types::{CollectionSpec, ErrorKind, Precondition, ReadCheck, WriteOp};

#[tokio::test]
async fn changed_import_dependency_aborts_rows_and_counter_advances_together() {
    let storage = memory_backend()
        .for_scope(&tinystoragedrivers_types::Scope::new("counter-import").unwrap())
        .unwrap();
    let docs = storage.documents();
    docs.ensure_collection(&CollectionSpec::new(COUNTERS))
        .await
        .unwrap();
    docs.ensure_collection(&CollectionSpec::new("owner.imported"))
        .await
        .unwrap();

    let initial_version = docs
        .put(
            "owner.imported",
            "dependency",
            serde_json::json!({"state":"old"}),
            Precondition::Absent,
        )
        .await
        .unwrap();
    docs.put(
        "owner.imported",
        "dependency",
        serde_json::json!({"state":"changed"}),
        Precondition::Version(initial_version),
    )
    .await
    .unwrap();

    let result = Counters::new(docs)
        .set_many_at_least_with_batch(
            &[("history", "messages", 41), ("history", "tool_calls", 73)],
            vec![ReadCheck {
                collection: "owner.imported".into(),
                id: "dependency".into(),
                expected: tinystoragedrivers_types::ExpectedDocument::Version(initial_version),
            }],
            vec![WriteOp::Put {
                collection: "owner.imported".into(),
                id: "message".into(),
                doc: serde_json::json!({"id":41}),
                precondition: Precondition::Absent,
            }],
        )
        .await;
    assert_eq!(result.unwrap_err().kind(), ErrorKind::Conflict);
    let counters = Counters::new(docs);
    assert_eq!(counters.peek("history", "messages").await.unwrap(), 0);
    assert_eq!(counters.peek("history", "tool_calls").await.unwrap(), 0);
    assert!(
        docs.get("owner.imported", "message")
            .await
            .unwrap()
            .is_none()
    );
}
