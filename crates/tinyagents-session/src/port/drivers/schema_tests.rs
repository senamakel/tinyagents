use super::*;
use crate::port::SessionStoreProvider;
use crate::testkit::MemoryStorage;
use crate::turn_state::TurnState;
use serde_json::{Value, json};
use std::sync::Arc;
use tinyagents_storage_kit::{
    Filter, Item, MigrateOptions, Precondition, Registry, Scope, ScopedStorage, WriteOptions,
};
use tinystoragedrivers_api::StorageBackend;

fn agent_storage(backend: &Arc<dyn StorageBackend>) -> ScopedStorage {
    let scope = Scope::new("test-planner-store").unwrap();
    ScopedStorage::new(backend.for_scope(&scope).unwrap(), scope)
}

async fn put_v1(storage: &ScopedStorage, collection: &str, id: &str, body: Value, created: i64) {
    storage
        .documents()
        .put_with(
            collection,
            id,
            body,
            Precondition::None,
            WriteOptions::new().created_ms(created),
        )
        .await
        .unwrap();
}

async fn migrate(storage: &ScopedStorage) -> tinyagents_storage_kit::MigrationReport {
    for spec in tinyagents_storage_kit::primitives::collection_specs() {
        storage.documents().ensure_collection(&spec).await.unwrap();
    }
    for item in storage_schema().items {
        if let Item::Collection { spec, .. } = item {
            storage.documents().ensure_collection(&spec).await.unwrap();
        }
    }
    Registry::compose([storage_schema()])
        .unwrap()
        .migrate(storage, &MigrateOptions::new())
        .await
        .unwrap()
}

fn v1_turn(thread: &str, request: &str) -> Value {
    let turn = TurnState::started(thread, request, 8, "2026-01-01T00:05:00Z");
    json!({
        "thread_id": thread, "request_id": request, "lifecycle": "started",
        "turn": serde_json::to_value(&turn).unwrap(),
    })
}

#[test]
fn the_schema_names_what_the_session_crate_owns() {
    let schema = storage_schema();
    assert_eq!(schema.owner.as_str(), "session");
    let names: Vec<&str> = schema.items.iter().map(|item| item.name()).collect();
    assert_eq!(
        names,
        [
            "session.transcript_entries",
            "session.transcripts",
            "session.turn_states"
        ]
    );
    assert!(
        schema
            .items
            .iter()
            .all(|item| item.scope() == tinyagents_storage_kit::ScopeClass::Agent)
    );
    Registry::compose([schema]).unwrap();
}

#[tokio::test]
async fn transcripts_move_to_key_of_ids_and_keep_creation_time_in_the_body() {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemoryStorage::new());
    let storage = agent_storage(&backend);
    put_v1(
        &storage,
        "session_transcripts",
        "5:a/b__c",
        json!({
            "stem": "a/b", "subagent": false, "written": true, "thread_id": "t",
            "agent_id": "planner", "agent_name": "planner", "indexed_seq": 3,
            "created_at": "2026-01-01T00:00:01.500000000Z",
        }),
        0,
    )
    .await;
    // A reservation nobody wrote is dropped: it is a lease now.
    put_v1(
        &storage,
        "session_transcripts",
        "6:a.g1",
        json!({
            "stem": "a.g1", "subagent": false, "written": false,
            "created_at": "2026-01-01T00:00:02Z", "reserved_at": 1,
        }),
        0,
    )
    .await;
    put_v1(
        &storage,
        "session_transcript_entries",
        "3:a/b#0000000004",
        json!({ "stem": "a/b", "seq": 4, "written": true, "extend": [] }),
        1_700_000_000_000,
    )
    .await;

    let report = migrate(&storage).await;
    assert!(report.ran.iter().any(|s| s.step == "transcripts_v2"));
    assert!(report.ran.iter().any(|s| s.step == "transcript_entries_v2"));

    let docs = storage.documents();
    let index = docs
        .get("session.transcripts", "a%2Fb")
        .await
        .unwrap()
        .expect("id is Key::of([stem])");
    assert!(
        index.meta.created_ms > 0,
        "the engine assigns new row metadata"
    );
    assert_eq!(index.meta.schema, 2);
    assert!(index.doc.get("created_at").is_none());
    assert_eq!(index.doc["created_ms"], 1_767_225_601_500_i64);
    assert_eq!(index.doc["thread_id"], "t");
    assert_eq!(index.doc["indexed_seq"], 3);
    assert_eq!(
        docs.count("session.transcripts", &Filter::default())
            .await
            .unwrap(),
        1,
        "the reservation row is not carried over"
    );
    let entry = docs
        .get("session.transcript_entries", "a%2Fb/0000000004")
        .await
        .unwrap()
        .expect("id is Key::of([stem, zero-padded seq])");
    assert_eq!(entry.doc["seq"], 4);
    assert!(
        entry.meta.created_ms > 0,
        "the engine assigns new row metadata"
    );
    for v1 in ["session_transcripts", "session_transcript_entries"] {
        assert_eq!(docs.count(v1, &Filter::default()).await.unwrap(), 0, "{v1}");
    }
}

#[tokio::test]
async fn turn_states_are_unwrapped_to_the_snake_case_storage_form() {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemoryStorage::new());
    let storage = agent_storage(&backend);
    put_v1(
        &storage,
        "session_turn_states",
        "1:t/1:r",
        v1_turn("t", "r"),
        0,
    )
    .await;
    put_v1(
        &storage,
        "session_turn_states",
        "broken",
        json!({"thread_id": "t", "turn": "nope"}),
        0,
    )
    .await;

    migrate(&storage).await;

    let docs = storage.documents();
    let row = docs
        .get("session.turn_states", "t/r")
        .await
        .unwrap()
        .expect("id is Key::of([thread, request])");
    assert_eq!(row.doc["started_ms"], 1_767_225_900_000_i64);
    assert_eq!(row.doc["max_iterations"], 8);
    assert!(row.doc.get("turn").is_none() && row.doc.get("startedAt").is_none());
    assert_eq!(row.meta.schema, 2);
    assert_eq!(
        docs.count("session.turn_states", &Filter::default())
            .await
            .unwrap(),
        1,
        "an undecodable row is dropped"
    );
}

#[test]
fn migrated_records_are_what_the_provider_reads() {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemoryStorage::new());
    let storage = agent_storage(&backend);
    let blocking = tinyagents_storage_kit::Blocking::new().unwrap();
    blocking
        .run({
            let storage = storage.clone();
            async move {
                put_v1(
                    &storage,
                    "session_transcripts",
                    "1:s",
                    json!({
                        "stem": "s", "subagent": false, "written": true, "thread_id": "t",
                        "agent_id": "planner", "agent_name": "planner", "indexed_seq": 0,
                        "created_at": "2026-01-01T00:00:00Z",
                    }),
                    0,
                )
                .await;
                put_v1(
                    &storage,
                    "session_transcript_entries",
                    "1:s#0000000000",
                    json!({
                        "stem": "s", "seq": 0, "written": true,
                        "extend": [{"role": "user", "content": "hello"}],
                    }),
                    0,
                )
                .await;
                put_v1(
                    &storage,
                    "session_turn_states",
                    "1:t/1:r",
                    v1_turn("t", "r"),
                    0,
                )
                .await;
                migrate(&storage).await;
            }
        })
        .unwrap();

    let provider = DriverSessionStores::new(crate::testkit::scoped_storage_factory(
        backend,
        [("planner".to_owned(), "test-planner-store".to_owned())]
            .into_iter()
            .collect(),
    ))
    .unwrap();
    let stores = provider.for_agent("planner");
    let root = stores.transcripts.root_for_thread("t").expect("indexed");
    let session = root.read_session().unwrap().expect("written");
    assert_eq!(session.messages[0].content, "hello");
    assert_eq!(
        stores.turn_states.get("t").unwrap().unwrap().started_at,
        "2026-01-01T00:05:00Z"
    );
}
