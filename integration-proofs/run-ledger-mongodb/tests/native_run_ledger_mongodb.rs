//! Opt-in actual Mongo replica-set proof in an excluded standalone workspace.
//! Ordinary standalone runs ignore this case; ignored results are not native
//! Mongo acceptance evidence. The root CI graph excludes this workspace.
//!
//! Set TINYAGENTS_TEST_MONGO_URL to an unauthenticated literal-loopback URI with
//! an explicit port, then enable mongodb-storage-tests and run this target with
//! --ignored. The case uses a unique database; remove its owned test container
//! after the run. No production endpoint or credential is accepted.
#![cfg(feature = "mongodb-storage-tests")]

use serde_json::json;
use std::sync::Arc;
use tinyagents_session::run_ledger::{
    DriverRunLedger, LedgerError, RunEventAppend, RunEventListRequest, RunLedgerStore,
};
use tinyagents_storage_kit::{Key, ScopedStorage};
use tinystoragedrivers_api::{DocumentStore, Storage, StorageBackend, StreamStore};
use tinystoragedrivers_connect::MongoConnection;
use tinystoragedrivers_document::mongodb::{MongoDocuments, MongoStreams};
use tinystoragedrivers_testkit::MemoryBackend;
use tinystoragedrivers_types::{ErrorKind, Scope};

/// Validate the explicitly opted-in unauthenticated loopback Mongo endpoint.
fn endpoint() -> String {
    let value = std::env::var("TINYAGENTS_TEST_MONGO_URL")
        .expect("opt-in local Mongo tests require TINYAGENTS_TEST_MONGO_URL");
    let url = reqwest::Url::parse(&value).expect("local Mongo URI must be valid");
    assert_eq!(
        url.scheme(),
        "mongodb",
        "SRV and non-Mongo endpoints are forbidden"
    );
    assert!(
        matches!(url.host_str(), Some("127.0.0.1" | "[::1]")) && url.port().is_some(),
        "local Mongo requires literal loopback and an explicit port"
    );
    assert!(
        url.username().is_empty() && url.password().is_none(),
        "credentials are forbidden"
    );
    value
}

/// Open native Mongo document and stream ports in a UUID database.
async fn backend(scope: &Scope) -> (String, String, ScopedStorage) {
    let url = endpoint();
    let database = format!("tinyagents_ledger_{}", uuid::Uuid::new_v4().simple());
    let connection = MongoConnection::connect(&url, &database).await.unwrap();
    let documents =
        Arc::new(MongoDocuments::new(connection.clone(), scope.clone())) as Arc<dyn DocumentStore>;
    let streams = Arc::new(MongoStreams::new(connection, scope.clone())) as Arc<dyn StreamStore>;
    let memory = MemoryBackend::new().for_scope(scope).unwrap();
    let storage = Storage::new(
        documents,
        streams,
        memory.kv().clone(),
        memory.blobs().clone(),
        memory.queues().clone(),
        memory.secrets().clone(),
    );
    (url, database, ScopedStorage::new(storage, scope.clone()))
}

fn event() -> RunEventAppend {
    RunEventAppend {
        run_id: "run".into(),
        event_type: "tick".into(),
        payload: json!({}),
    }
}

#[tokio::test]
#[ignore = "local Mongo replica set: explicit loopback URL and --ignored required"]
async fn terminal_event_append_persists_bson_representable_boundary() {
    let scope = Scope::new("alice").unwrap();
    let (url, database, storage) = backend(&scope).await;
    let documents = storage.documents().clone();
    let ledger = DriverRunLedger::new(storage);
    DriverRunLedger::prepare_schema(documents.as_ref())
        .await
        .unwrap();

    // Seed the public v2 allocation head at the last signed BSON sequence.
    // Appending writes both this nullable terminal head and the event through
    // the actual Mongo converter; i64::MAX + 1 would be rejected by BSON.
    let head_id = Key::of(["run"]).unwrap().as_str().to_owned();
    documents
        .put(
            "ledger.event_heads",
            &head_id,
            json!({"next": i64::MAX}),
            tinystoragedrivers_types::Precondition::None,
        )
        .await
        .unwrap();
    assert_eq!(
        ledger.append_run_event(event()).await.unwrap().sequence,
        i64::MAX as u64
    );
    assert!(matches!(
        ledger.append_run_event(event()).await.unwrap_err(),
        LedgerError::Driver(error) if error.kind() == ErrorKind::Serialization
    ));

    // A second connection proves the terminal event and exhausted head were
    // persisted by Mongo, rather than retained in the first driver's memory.
    let reopened = MongoConnection::connect(&url, &database).await.unwrap();
    let peer_docs =
        Arc::new(MongoDocuments::new(reopened, scope.clone())) as Arc<dyn DocumentStore>;
    let peer_memory = MemoryBackend::new().for_scope(&scope).unwrap();
    let peer_storage = ScopedStorage::new(
        Storage::new(
            peer_docs.clone(),
            Arc::new(MongoStreams::new(
                MongoConnection::connect(&url, &database).await.unwrap(),
                scope.clone(),
            )),
            peer_memory.kv().clone(),
            peer_memory.blobs().clone(),
            peer_memory.queues().clone(),
            peer_memory.secrets().clone(),
        ),
        scope,
    );
    let peer = DriverRunLedger::new(peer_storage);
    let head = peer_docs
        .get("ledger.event_heads", &head_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(head.doc["next"], serde_json::Value::Null);
    let events = peer
        .list_recent_run_events(&RunEventListRequest {
            run_id: "run".into(),
            after_sequence: Some(i64::MAX as u64 - 1),
            limit: Some(10),
        })
        .await
        .unwrap();
    assert_eq!(events.events.len(), 1);
    assert_eq!(events.events[0].sequence, i64::MAX as u64);
    eprintln!("Actual Mongo terminal event proof completed");
}
