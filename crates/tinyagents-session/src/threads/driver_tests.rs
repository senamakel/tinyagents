use super::*;
use crate::testkit::MemoryStorage;
use serde_json::json;
use tinyagents_storage_kit::Scope;

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
fn scoped(backend: &MemoryStorage, scope: &str) -> DriverConversationStore {
    DriverConversationStore::new(backend.for_scope(&Scope::new(scope).unwrap()).unwrap())
}

#[tokio::test(flavor = "current_thread")]
async fn independent_handles_share_metadata_but_other_scope_does_not() {
    let backend = MemoryStorage::new();
    let a = scoped(&backend, "profile-a");
    let b = scoped(&backend, "profile-a");
    let other = scoped(&backend, "profile-b");
    a.ensure_thread(request("same")).await.unwrap();
    assert_eq!(b.list_threads().await.unwrap().len(), 1);
    assert!(other.get_thread("same").await.unwrap().is_none());
    let mut upsert = request("same");
    upsert.created_at = "2026-02-01T00:00:00Z".into();
    upsert.personality_id = Some("persona".into());
    b.ensure_thread(upsert).await.unwrap();
    let (title, labels) = tokio::join!(
        a.update_thread_title("same", "Changed", "now"),
        b.update_thread_labels("same", vec!["worker".into()], "now")
    );
    title.unwrap();
    labels.unwrap();
    let row = a.get_thread("same").await.unwrap().unwrap();
    assert_eq!(row.title, "Changed");
    assert_eq!(row.labels, vec!["tasks"]);
    assert_eq!(row.personality_id.as_deref(), Some("persona"));
    assert_eq!(row.created_at, "2026-01-01T00:00:00Z");
    assert!(
        matches!(other.update_thread_title("same", "x", "now").await, Err(ConversationError::Driver(error)) if error.kind() == ErrorKind::NotFound)
    );
}

#[tokio::test]
async fn unprepared_append_never_approximates_a_committed_message() {
    let backend = MemoryStorage::new();
    let store = scoped(&backend, "profile");
    store.ensure_thread(request("one")).await.unwrap();
    let message = ThreadMessage {
        id: "agent:r".into(),
        content: "private".into(),
        message_type: "text".into(),
        extra_metadata: json!({}),
        sender: "assistant".into(),
        created_at: "2026-01-01T00:00:00Z".into(),
    };
    assert!(
        matches!(
            store.append_message("one", message).await,
            Err(ConversationError::Driver(error)) if error.kind() == ErrorKind::NotFound
        ),
        "plain append requires its own trusted schema preparation"
    );
    assert!(store.get_messages("one").await.unwrap().is_empty());
}

async fn seed(store: &DriverConversationStore, id: &str, content: &[&str]) {
    store.ensure_thread(request(id)).await.unwrap();
    let mut row = store.load(id).await.unwrap().unwrap();
    let stream = row.doc.stream().unwrap();
    for (offset, content) in content.iter().enumerate() {
        let record = MessageDto::from_message(
            &ThreadMessage {
                id: format!("message:{offset}"),
                content: (*content).into(),
                sender: "user".into(),
                message_type: "text".into(),
                extra_metadata: json!({}),
                created_at: format!("2026-01-{:02}T00:00:00Z", offset % 28 + 1),
            },
            to_ms,
        )
        .unwrap();
        store
            .streams
            .append(&stream, serde_json::to_value(record).unwrap())
            .await
            .unwrap();
    }
    row.doc.visible_end = content.len() as u64;
    row.doc.thread.message_count = content.len();
    if !content.is_empty() {
        row.doc.thread.last_message_at = "2026-01-02T00:00:00Z".into();
    }
    store
        .docs
        .put(
            COLLECTION,
            &row.id,
            serde_json::to_value(row.doc.to_dto().unwrap()).unwrap(),
            row.unchanged(),
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn tombstone_and_recreated_incarnation_hide_old_stream_and_search_rows() {
    let backend = MemoryStorage::new();
    let a = scoped(&backend, "profile");
    let b = scoped(&backend, "profile");
    seed(&a, "same", &["old secret 東京"]).await;
    let old = a.load("same").await.unwrap().unwrap();
    assert!(b.delete_thread("same", "now").await.unwrap());
    assert!(!b.delete_thread("same", "now").await.unwrap());
    assert!(a.get_messages("same").await.unwrap().is_empty());
    assert!(
        a.search_cross_thread_messages("secret", 10, None)
            .await
            .unwrap()
            .is_empty()
    );
    let mut recreated = request("same");
    recreated.created_at = "2026-03-01T00:00:00Z".into();
    a.ensure_thread(recreated).await.unwrap();
    let current = b.load("same").await.unwrap().unwrap();
    assert_ne!(current.doc.incarnation, old.doc.incarnation);
    assert_eq!(current.doc.thread.created_at, "2026-03-01T00:00:00Z");
    assert!(b.get_messages("same").await.unwrap().is_empty());
    assert_eq!(a.streams.len(&old.doc.stream().unwrap()).await.unwrap(), 1);
}

#[tokio::test]
async fn exhaustive_paged_search_matches_local_substrings_and_cjk_tied_groups() {
    let backend = MemoryStorage::new();
    let store = scoped(&backend, "profile");
    let mut index = inverted_index::InvertedIndex::new();
    for n in 0..(PAGE_SIZE + 3) {
        let id = format!("thread{n}");
        seed(&store, &id, &["Café alphabet", "東京で日本語", "unrelated"]).await;
        for message in store.get_messages(&id).await.unwrap() {
            index.insert(&id, message);
        }
    }
    assert_eq!(store.list_threads().await.unwrap().len(), PAGE_SIZE + 3);
    for query in [
        "cafe",
        "pha",
        "東京",
        "日",
        "cafe 東京",
        "a",
        "",
        "not-found",
    ] {
        let normalize = |mut hits: Vec<CrossThreadHit>| {
            hits.sort_by(|a, b| {
                a.thread_id
                    .cmp(&b.thread_id)
                    .then(a.message_id.cmp(&b.message_id))
            });
            hits
        };
        assert_eq!(
            normalize(
                store
                    .search_cross_thread_messages(query, 1000, None)
                    .await
                    .unwrap()
            ),
            normalize(index.search(query, 1000, None))
        );
    }
    let hits = store
        .search_cross_thread_messages("東京", 1000, Some("thread0"))
        .await
        .unwrap();
    assert_eq!(hits.len(), PAGE_SIZE + 2);
    assert!(hits.iter().all(|hit| hit.thread_id != "thread0"));
    assert!(
        store
            .search_cross_thread_messages("東京", 0, None)
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn bounded_stream_windows_read_all_committed_rows_and_hide_uncommitted_tail() {
    let backend = MemoryStorage::new();
    let store = scoped(&backend, "profile");
    seed(&store, "one", &vec!["visible"; PAGE_SIZE * 2 + 3]).await;
    let row = store.load("one").await.unwrap().unwrap();
    let stream = row.doc.stream().unwrap();
    store
        .streams
        .append(&stream, json!({"not": "committed"}))
        .await
        .unwrap();
    assert_eq!(
        store.get_messages("one").await.unwrap().len(),
        PAGE_SIZE * 2 + 3
    );
}

#[tokio::test]
async fn invalid_ids_and_malformed_committed_records_keep_native_error_kinds() {
    let backend = MemoryStorage::new();
    let store = scoped(&backend, "profile");
    for id in ["", "null\0id"] {
        assert!(
            matches!(store.ensure_thread(request(id)).await, Err(ConversationError::Driver(e)) if e.kind() == ErrorKind::InvalidInput)
        );
    }
    store
        .ensure_thread(request(&"x".repeat(1024 * 1024 + 1)))
        .await
        .unwrap_err();
    store
        .ensure_thread(request(&"long".repeat(1000)))
        .await
        .unwrap();
    seed(&store, "one", &["valid"]).await;
    let row = store.load("one").await.unwrap().unwrap();
    let stream = row.doc.stream().unwrap();
    store.streams.delete_stream(&stream).await.unwrap();
    assert!(
        matches!(store.get_messages("one").await, Err(ConversationError::Driver(e)) if e.kind() == ErrorKind::Serialization)
    );
    store
        .streams
        .append(
            &stream,
            json!({"id": "x", "content": "bad", "sender": "user", "createdAt": "now"}),
        )
        .await
        .unwrap();
    assert!(
        matches!(store.get_messages("one").await, Err(ConversationError::Driver(e)) if e.kind() == ErrorKind::Serialization)
    );
}
#[tokio::test]
async fn store_reconstruction_reads_same_committed_profile_fixture() {
    let scope = Scope::new("profile").unwrap();
    let backend = MemoryStorage::new();
    let a = DriverConversationStore::new(backend.for_scope(&scope).unwrap());
    seed(&a, "one", &["東京 persisted"]).await;
    drop(a);
    let b = DriverConversationStore::new(backend.for_scope(&scope).unwrap());
    assert_eq!(
        b.get_messages("one").await.unwrap()[0].content,
        "東京 persisted"
    );
}

#[path = "driver_fixtures.rs"]
mod fixtures;

#[tokio::test(flavor = "current_thread")]
async fn conflicting_metadata_edits_reread_version_and_preserve_both_fields() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let backend = MemoryStorage::new();
    let scope = Scope::new("profile").unwrap();
    scoped(&backend, "profile")
        .ensure_thread(request("one"))
        .await
        .unwrap();
    let ports = backend.for_scope(&scope).unwrap();
    let scoped_ports = ports;
    let docs = Arc::new(fixtures::ConflictingDocs {
        inner: scoped_ports.documents().clone(),
        attempts: AtomicUsize::new(0),
        barrier: tokio::sync::Barrier::new(2),
        refuse_get: false,
    });
    let a =
        DriverConversationStore::new(crate::testkit::with_documents(&scoped_ports, docs.clone()));
    let b =
        DriverConversationStore::new(crate::testkit::with_documents(&scoped_ports, docs.clone()));
    let (title, labels) = tokio::join!(
        a.update_thread_title("one", "Changed", "now"),
        b.update_thread_labels("one", vec!["worker".into()], "now")
    );
    title.unwrap();
    labels.unwrap();
    assert!(
        docs.attempts.load(Ordering::SeqCst) >= 3,
        "fixture must force an actual native CAS conflict"
    );
    let row = a.get_thread("one").await.unwrap().unwrap();
    assert_eq!(row.title, "Changed");
    assert_eq!(row.labels, vec!["tasks"]);
}

fn instrumented_streams(
    store: &DriverConversationStore,
    pause_first: bool,
) -> Arc<fixtures::PausedStreams> {
    use std::sync::atomic::AtomicUsize;
    Arc::new(fixtures::PausedStreams {
        inner: store.streams.clone(),
        calls: AtomicUsize::new(0),
        max_limit: AtomicUsize::new(0),
        reached: tokio::sync::Notify::new(),
        release: tokio::sync::Notify::new(),
        pause_first,
    })
}

#[tokio::test]
async fn prefix_reader_bounds_every_window_and_revalidates_recreated_thread() {
    use std::sync::atomic::Ordering;
    let backend = MemoryStorage::new();
    let store = scoped(&backend, "profile");
    seed(&store, "one", &vec!["old"; PAGE_SIZE * 2 + 3]).await;
    let streams = instrumented_streams(&store, false);
    let reader = DriverConversationStore {
        validate_only: false,
        docs: store.docs.clone(),
        streams: streams.clone(),
    };
    assert_eq!(
        reader.get_messages("one").await.unwrap().len(),
        PAGE_SIZE * 2 + 3
    );
    assert_eq!(streams.calls.load(Ordering::SeqCst), 3);
    assert_eq!(streams.max_limit.load(Ordering::SeqCst), PAGE_SIZE);
    let paused = instrumented_streams(&store, true);
    let reader = DriverConversationStore {
        validate_only: false,
        docs: store.docs.clone(),
        streams: paused.clone(),
    };
    let pending = tokio::spawn(async move { reader.get_messages("one").await });
    paused.reached.notified().await;
    store.delete_thread("one", "now").await.unwrap();
    store.ensure_thread(request("one")).await.unwrap();
    paused.release.notify_one();
    assert!(
        pending.await.unwrap().unwrap().is_empty(),
        "old committed prefix must not resurrect after metadata changes"
    );
}

#[tokio::test]
async fn configured_backend_refusal_preserves_native_source_without_selecting_classic_store() {
    use std::sync::atomic::AtomicUsize;
    let backend = MemoryStorage::new();
    let scope = Scope::new("profile").unwrap();
    let ports = backend.for_scope(&scope).unwrap();
    let docs = Arc::new(fixtures::ConflictingDocs {
        inner: ports.documents().clone(),
        attempts: AtomicUsize::new(0),
        barrier: tokio::sync::Barrier::new(2),
        refuse_get: true,
    });
    let store = DriverConversationStore::new(crate::testkit::with_documents(&ports, docs));
    let Err(error) = store.ensure_thread(request("one")).await else {
        panic!("configured refusal became success");
    };
    let source = std::error::Error::source(&error).unwrap();
    let source = source
        .downcast_ref::<StorageError>()
        .or_else(|| source.downcast_ref::<Box<StorageError>>().map(Box::as_ref))
        .expect("native driver source must remain typed");
    assert_eq!(source.kind(), ErrorKind::Unavailable);
    assert!(
        matches!(store.get_messages("one").await, Err(ConversationError::Driver(e)) if e.kind() == ErrorKind::Unavailable)
    );
}

#[tokio::test]
async fn corrupt_stored_identity_reports_serialization_instead_of_caller_input_error() {
    let backend = MemoryStorage::new();
    let store = scoped(&backend, "profile");
    store.ensure_thread(request("one")).await.unwrap();
    let mut row = store.load("one").await.unwrap().unwrap();
    row.doc.thread.id.clear();
    store
        .docs
        .put(
            COLLECTION,
            &row.id,
            serde_json::to_value(row.doc.to_dto().unwrap()).unwrap(),
            row.unchanged(),
        )
        .await
        .unwrap();
    assert!(
        matches!(store.get_thread("one").await, Err(ConversationError::Driver(e)) if e.kind() == ErrorKind::Serialization)
    );
}
