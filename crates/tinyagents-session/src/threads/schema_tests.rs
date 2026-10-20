use super::*;
use crate::testkit::MemoryStorage;
use crate::threads::{
    ConversationPersistence, CreateConversationThread, DriverConversationStore, ThreadMessage,
    ThreadMessagePatch,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use tinyagents_storage_kit::{
    CollectionSpec, Item, MigrateOptions, Precondition, Registry, Scope, ScopeClass, ScopedStorage,
    WriteOptions,
    schema::{StepBody, StreamStepFn},
};
use tinystoragedrivers_api::StorageBackend;

const INCARNATION: &str = "6f9619ff-8b86-d011-b42d-00c04fc964ff";
const TOKEN: &str = "0b4e8d6c-3c0e-4f5e-9d3f-1d2c3b4a5f60";

fn storage() -> ScopedStorage {
    let backend: Arc<dyn StorageBackend> = Arc::new(MemoryStorage::new());
    let scope = Scope::new("test-conversations").unwrap();
    ScopedStorage::new(backend.for_scope(&scope).unwrap(), scope)
}

/// The v1 document id of a thread: the hex SHA-256 of its id.
fn digest(thread_id: &str) -> String {
    hex::encode(Sha256::digest(thread_id.as_bytes()))
}

fn v1_thread(id: &str, incarnation: &str, visible_end: u64, deleted: bool) -> Value {
    json!({
        "schema": 1, "incarnation": incarnation, "deleted": deleted, "visible_end": visible_end,
        "thread": {
            "id": id, "title": "Title", "isActive": true, "messageCount": visible_end,
            "lastMessageAt": "2026-01-02T00:00:00Z", "createdAt": "2026-01-01T00:00:00Z",
            "labels": ["tasks"], "workingDir": "/work"
        }
    })
}

fn v1_message(id: &str, content: &str, incarnation: &str) -> Value {
    json!({
        "schema": 1, "incarnation": incarnation,
        "message": {
            "id": id, "content": content, "type": "text",
            "extraMetadata": {"camelCase": {"stays": 1}}, "sender": "user",
            "createdAt": "2026-01-02T00:00:00.5Z"
        }
    })
}

async fn put_v1(storage: &ScopedStorage, collection: &str, id: &str, body: Value) {
    let docs = storage.documents();
    docs.ensure_collection(&CollectionSpec::new(collection))
        .await
        .unwrap();
    docs.put_with(
        collection,
        id,
        body,
        Precondition::None,
        WriteOptions::new().created_ms(1_700_000_000_000),
    )
    .await
    .unwrap();
}

async fn put_v1_stream(
    storage: &ScopedStorage,
    thread_id: &str,
    incarnation: &str,
    rows: &[Value],
) {
    let stream = format!(
        "conversation_messages_v1:{}:{incarnation}",
        digest(thread_id)
    );
    for row in rows {
        storage
            .streams()
            .append(&stream, row.clone())
            .await
            .unwrap();
    }
}

async fn migrate(storage: &ScopedStorage) {
    for item in conversations_storage_schema().items {
        if let Item::Collection { spec, .. } = item {
            storage.documents().ensure_collection(&spec).await.unwrap();
        }
    }
    for spec in tinyagents_storage_kit::primitives::collection_specs() {
        storage.documents().ensure_collection(&spec).await.unwrap();
    }
    Registry::compose([conversations_storage_schema()])
        .unwrap()
        .migrate(storage, &MigrateOptions::new())
        .await
        .unwrap();
}

fn message(id: &str, content: &str) -> ThreadMessage {
    ThreadMessage {
        id: id.into(),
        content: content.into(),
        message_type: "text".into(),
        extra_metadata: json!({"camelCase": {"stays": 1}}),
        sender: "user".into(),
        created_at: "2026-01-02T00:00:00.500Z".into(),
    }
}

fn stream_step() -> StreamStepFn {
    let step = conversations_storage_schema()
        .steps
        .into_iter()
        .find(|step| step.id == "messages_v2")
        .expect("a messages_v2 step");
    let StepBody::Streams(run) = step.run else {
        panic!("messages_v2 is a stream step");
    };
    run
}

#[test]
fn the_schema_names_what_the_conversations_owner_declares() {
    let schema = conversations_storage_schema();
    assert_eq!(schema.owner.as_str(), "conversations");
    let mut names: Vec<&str> = schema.items.iter().map(Item::name).collect();
    names.sort_unstable();
    assert_eq!(
        names,
        [
            "conversations.appends",
            "conversations.messages",
            "conversations.threads"
        ]
    );
    assert!(
        schema
            .items
            .iter()
            .all(|item| item.scope() == ScopeClass::Tenant)
    );
    let mut steps: Vec<&str> = schema.steps.iter().map(|step| step.id).collect();
    steps.sort_unstable();
    assert_eq!(steps, ["appends_v2", "messages_v2", "threads_v2"]);
    Registry::compose([schema]).unwrap();
}

#[tokio::test]
async fn v1_threads_appends_and_messages_move_to_the_v2_names_and_shapes() {
    let storage = storage();
    let id = "thread/1";
    put_v1(
        &storage,
        "conversation_threads_v1",
        &digest(id),
        v1_thread(id, INCARNATION, 2, false),
    )
    .await;
    put_v1_stream(
        &storage,
        id,
        INCARNATION,
        &[
            v1_message("m1", "one", INCARNATION),
            v1_message("m2", "two", INCARNATION),
        ],
    )
    .await;
    let pending = v1_message("m3", "three", INCARNATION);
    put_v1(
        &storage,
        "conversation_appends_v1",
        &format!("{}:{INCARNATION}", digest(id)),
        json!({
            "schema": "1", "token": TOKEN, "thread_id": id, "incarnation": INCARNATION,
            "offset": "2", "record": serde_json::to_string(&pending).unwrap()
        }),
    )
    .await;
    migrate(&storage).await;

    let docs = storage.documents();
    let thread = docs
        .get("conversations.threads", "thread%2F1")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(thread.meta.schema, 2);
    assert!(
        thread.meta.created_ms > 0,
        "the engine assigns new row metadata"
    );
    assert_eq!(
        thread.doc,
        json!({
            "id": "thread/1", "title": "Title", "is_active": true,
            "created_ms": 1_767_225_600_000_i64, "last_message_ms": 1_767_312_000_000_i64,
            "labels": ["tasks"], "working_dir": "/work", "incarnation": INCARNATION,
            "deleted": false, "visible_end": 2, "messages_key": digest(id)
        })
    );
    let intent = docs
        .get(
            "conversations.appends",
            &format!("thread%2F1/{INCARNATION}"),
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        intent.doc,
        json!({
            "token": TOKEN, "thread_id": id, "incarnation": INCARNATION, "offset": 2,
            "record": {
                "id": "m3", "content": "three", "message_type": "text",
                "extra_metadata": {"camelCase": {"stays": 1}}, "sender": "user",
                "created_ms": 1_767_312_000_500_i64
            }
        })
    );
    let stream = format!("conversations.messages/{}/{INCARNATION}", digest(id));
    assert_eq!(storage.streams().len(&stream).await.unwrap(), 2);
    let first = &storage.streams().read_window(&stream, 0, 1).await.unwrap()[0];
    assert_eq!(
        first.value,
        json!({
            "id": "m1", "content": "one", "message_type": "text",
            "extra_metadata": {"camelCase": {"stays": 1}}, "sender": "user",
            "created_ms": 1_767_312_000_500_i64
        })
    );
    assert!(
        storage
            .streams()
            .streams("conversation_messages_v1:")
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        docs.count("conversation_threads_v1", &Default::default())
            .await
            .unwrap(),
        0
    );
    assert_eq!(
        docs.count("conversation_appends_v1", &Default::default())
            .await
            .unwrap(),
        0
    );
}

#[tokio::test]
async fn a_migrated_conversation_reads_recovers_its_intent_and_moves_to_canonical_streams() {
    let storage = storage();
    let id = "thread/1";
    put_v1(
        &storage,
        "conversation_threads_v1",
        &digest(id),
        v1_thread(id, INCARNATION, 2, false),
    )
    .await;
    put_v1_stream(
        &storage,
        id,
        INCARNATION,
        &[
            v1_message("m1", "one", INCARNATION),
            v1_message("m2", "two", INCARNATION),
        ],
    )
    .await;
    put_v1(
        &storage,
        "conversation_appends_v1",
        &format!("{}:{INCARNATION}", digest(id)),
        json!({
            "schema": "1", "token": TOKEN, "thread_id": id, "incarnation": INCARNATION,
            "offset": "2",
            "record": serde_json::to_string(&v1_message("m3", "three", INCARNATION)).unwrap()
        }),
    )
    .await;
    put_v1(
        &storage,
        "conversation_threads_v1",
        &digest("gone"),
        v1_thread("gone", "11111111-1111-4111-8111-111111111111", 0, true),
    )
    .await;
    migrate(&storage).await;

    let store = DriverConversationStore::new(storage.clone());
    let thread = store.get_thread(id).await.unwrap().unwrap();
    assert_eq!(thread.message_count, 2);
    assert_eq!(thread.created_at, "2026-01-01T00:00:00Z");
    assert_eq!(thread.working_dir.as_deref(), Some("/work"));
    assert!(store.get_thread("gone").await.unwrap().is_none());
    assert_eq!(
        store.get_messages(id).await.unwrap(),
        vec![message("m1", "one"), message("m2", "two")]
    );

    // The next append finishes the migrated intent first, then lands its own message.
    store
        .append_message(id, message("m4", "four"))
        .await
        .unwrap();
    assert_eq!(
        store.get_messages(id).await.unwrap(),
        vec![
            message("m1", "one"),
            message("m2", "two"),
            message("m3", "three"),
            message("m4", "four"),
        ]
    );
    assert_eq!(
        storage
            .documents()
            .count("conversations.appends", &Default::default())
            .await
            .unwrap(),
        0
    );

    // A rewrite opens a fresh incarnation whose stream carries the thread id.
    let patch = ThreadMessagePatch {
        extra_metadata: Some(json!({"edited": true})),
    };
    store.update_message(id, "m1", patch).await.unwrap();
    let streams = storage
        .streams()
        .streams("conversations.messages/")
        .await
        .unwrap();
    assert_eq!(
        streams.len(),
        1,
        "the legacy-keyed stream is retired: {streams:?}"
    );
    assert!(
        streams[0].starts_with("conversations.messages/thread%2F1/"),
        "{streams:?}"
    );
    assert_eq!(store.get_messages(id).await.unwrap().len(), 4);
    store
        .append_message(id, message("m5", "five"))
        .await
        .unwrap();
    assert_eq!(store.get_messages(id).await.unwrap().len(), 5);

    // Recreating a deleted v1 thread starts a fresh incarnation under the canonical key.
    store
        .ensure_thread(CreateConversationThread {
            id: "gone".into(),
            title: "Back".into(),
            created_at: "2026-02-01T00:00:00Z".into(),
            parent_thread_id: None,
            labels: None,
            personality_id: None,
            working_dir: None,
        })
        .await
        .unwrap();
    assert_eq!(
        store.get_thread("gone").await.unwrap().unwrap().title,
        "Back"
    );
}

#[tokio::test]
async fn migrating_twice_changes_nothing() {
    let storage = storage();
    let id = "thread/1";
    put_v1(
        &storage,
        "conversation_threads_v1",
        &digest(id),
        v1_thread(id, INCARNATION, 1, false),
    )
    .await;
    put_v1_stream(
        &storage,
        id,
        INCARNATION,
        &[v1_message("m1", "one", INCARNATION)],
    )
    .await;
    migrate(&storage).await;
    migrate(&storage).await;
    let store = DriverConversationStore::new(storage.clone());
    assert_eq!(
        store.get_messages(id).await.unwrap(),
        vec![message("m1", "one")]
    );
    assert_eq!(
        storage
            .documents()
            .count("conversations.threads", &Default::default())
            .await
            .unwrap(),
        1
    );
}

#[test]
fn the_stream_step_is_pure_and_keeps_the_v1_thread_key_and_incarnation() {
    let run = stream_step();
    let name = format!(
        "conversation_messages_v1:{}:{INCARNATION}",
        digest("thread/1")
    );
    let entries = vec![
        v1_message("m1", "one", INCARNATION),
        v1_message("m2", "two", INCARNATION),
    ];
    let first = run(name.clone(), entries.clone()).unwrap().unwrap();
    assert_eq!(first, run(name, entries).unwrap().unwrap());
    assert_eq!(
        first.0,
        format!(
            "conversations.messages/{}/{INCARNATION}",
            digest("thread/1")
        )
    );
    assert_eq!(first.1.len(), 2);
    assert_eq!(first.1[1]["id"], "m2");
    assert!(first.1[1].get("incarnation").is_none() && first.1[1].get("schema").is_none());
}

#[test]
fn the_stream_step_ends_a_stream_at_its_first_unreadable_row() {
    let run = stream_step();
    let name = format!("conversation_messages_v1:{}:{INCARNATION}", digest("t"));
    let entries = vec![
        v1_message("m1", "one", INCARNATION),
        json!({"not": "a record"}),
        v1_message("m3", "three", INCARNATION),
    ];
    assert_eq!(run(name.clone(), entries).unwrap().unwrap().1.len(), 1);
    let foreign = vec![v1_message("m1", "one", "other-incarnation")];
    assert!(run(name, foreign).unwrap().unwrap().1.is_empty());
}

#[test]
fn the_stream_step_drops_a_stream_whose_name_is_not_v1() {
    let run = stream_step();
    assert!(
        run("conversation_messages_v1:nocolon".into(), vec![])
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn a_v1_time_that_is_not_rfc_3339_migrates_as_the_epoch() {
    let storage = storage();
    let id = "t";
    let mut body = v1_thread(id, INCARNATION, 1, false);
    body["thread"]["createdAt"] = json!("yesterday");
    put_v1(&storage, "conversation_threads_v1", &digest(id), body).await;
    let mut row = v1_message("m1", "one", INCARNATION);
    row["message"]["createdAt"] = json!("now");
    put_v1_stream(&storage, id, INCARNATION, &[row]).await;
    migrate(&storage).await;
    let store = DriverConversationStore::new(storage);
    assert_eq!(
        store.get_thread(id).await.unwrap().unwrap().created_at,
        "1970-01-01T00:00:00Z"
    );
    assert_eq!(
        store.get_messages(id).await.unwrap()[0].created_at,
        "1970-01-01T00:00:00Z"
    );
}
