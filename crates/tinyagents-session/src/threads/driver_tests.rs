use super::*;
use async_trait::async_trait;
use serde_json::json;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tinystoragedrivers_core::{
    Capabilities, CollectionSpec, DocumentStore, Filter, MemoryStorage, Precondition, Query,
    ReadCheck, Scope, SearchHit, Sort, StorageBackend, StorageError, Version, Versioned, WriteOp,
    WriteResult,
};

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
    DriverConversationStore::prepare_schema(a.docs.as_ref())
        .await
        .unwrap();
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
    DriverConversationStore::prepare_schema(store.docs.as_ref())
        .await
        .unwrap();
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

#[derive(Debug, Default)]
struct SearchProbe {
    calls: AtomicUsize,
    thread_queries: AtomicUsize,
    without_full_text: AtomicBool,
    fail_search: AtomicBool,
}

#[derive(Debug)]
struct ProbedDocuments {
    inner: Arc<dyn DocumentStore>,
    probe: Arc<SearchProbe>,
}

#[async_trait]
impl DocumentStore for ProbedDocuments {
    fn capabilities(&self) -> Capabilities {
        if self.probe.without_full_text.load(Ordering::Relaxed) {
            self.inner
                .capabilities()
                .without(tinystoragedrivers_core::Capability::FullText)
        } else {
            self.inner.capabilities()
        }
    }

    async fn ensure_collection(&self, spec: &CollectionSpec) -> Result<(), StorageError> {
        self.inner.ensure_collection(spec).await
    }

    async fn validate_collection_spec(
        &self,
        required: &CollectionSpec,
    ) -> Result<(), StorageError> {
        self.inner.validate_collection_spec(required).await
    }

    async fn get(
        &self,
        collection: &str,
        id: &str,
    ) -> Result<Option<Versioned<serde_json::Value>>, StorageError> {
        self.inner.get(collection, id).await
    }

    async fn put(
        &self,
        collection: &str,
        id: &str,
        doc: serde_json::Value,
        precondition: Precondition,
    ) -> Result<Version, StorageError> {
        self.inner.put(collection, id, doc, precondition).await
    }

    async fn delete(
        &self,
        collection: &str,
        id: &str,
        precondition: Precondition,
    ) -> Result<bool, StorageError> {
        self.inner.delete(collection, id, precondition).await
    }

    async fn query(
        &self,
        collection: &str,
        query: &Query,
    ) -> Result<tinystoragedrivers_core::Page<Versioned<serde_json::Value>>, StorageError> {
        if collection == COLLECTION {
            self.probe.thread_queries.fetch_add(1, Ordering::Relaxed);
        }
        self.inner.query(collection, query).await
    }

    async fn count(&self, collection: &str, filter: &Filter) -> Result<u64, StorageError> {
        self.inner.count(collection, filter).await
    }

    async fn delete_where(&self, collection: &str, filter: &Filter) -> Result<u64, StorageError> {
        self.inner.delete_where(collection, filter).await
    }

    async fn claim(
        &self,
        collection: &str,
        filter: &Filter,
        sort: &[Sort],
        patch: &serde_json::Value,
    ) -> Result<Option<Versioned<serde_json::Value>>, StorageError> {
        self.inner.claim(collection, filter, sort, patch).await
    }

    async fn atomic_batch(&self, ops: Vec<WriteOp>) -> Result<Vec<WriteResult>, StorageError> {
        self.inner.atomic_batch(ops).await
    }

    async fn atomic_batch_checked(
        &self,
        checks: Vec<ReadCheck>,
        ops: Vec<WriteOp>,
    ) -> Result<Vec<WriteResult>, StorageError> {
        self.inner.atomic_batch_checked(checks, ops).await
    }

    async fn search(
        &self,
        collection: &str,
        text: &str,
        limit: usize,
    ) -> Result<Vec<SearchHit>, StorageError> {
        self.probe.calls.fetch_add(1, Ordering::Relaxed);
        if self.probe.fail_search.load(Ordering::Relaxed) {
            return Err(StorageError::backend("injected full-text failure"));
        }
        self.inner.search(collection, text, limit).await
    }

    async fn drop_collection(&self, collection: &str) -> Result<(), StorageError> {
        self.inner.drop_collection(collection).await
    }

    async fn collections(&self, prefix: &str) -> Result<Vec<String>, StorageError> {
        self.inner.collections(prefix).await
    }
}

#[tokio::test]
async fn stores_without_full_text_capability_keep_the_scan_fallback() {
    let backend = MemoryStorage::new();
    let mut store = scoped(&backend, "profile");
    DriverConversationStore::prepare_schema(store.docs.as_ref())
        .await
        .unwrap();
    seed(&store, "東京", &["東京で日本語"]).await;
    let probe = Arc::new(SearchProbe {
        without_full_text: AtomicBool::new(true),
        ..SearchProbe::default()
    });
    store.docs = Arc::new(ProbedDocuments {
        inner: store.docs.clone(),
        probe: probe.clone(),
    });

    let hits = store
        .search_cross_thread_messages("東京", 10, None)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].thread_id, "東京");
    assert_eq!(probe.calls.load(Ordering::Relaxed), 0);
    assert!(probe.thread_queries.load(Ordering::Relaxed) > 0);
}

#[tokio::test]
async fn full_text_failures_are_returned_without_scan_fallback() {
    let backend = MemoryStorage::new();
    let mut store = scoped(&backend, "profile");
    DriverConversationStore::prepare_schema(store.docs.as_ref())
        .await
        .unwrap();
    seed(&store, "one", &["東京で日本語"]).await;
    store.backfill_search_index().await.unwrap();
    let probe = Arc::new(SearchProbe {
        fail_search: AtomicBool::new(true),
        ..SearchProbe::default()
    });
    store.docs = Arc::new(ProbedDocuments {
        inner: store.docs.clone(),
        probe: probe.clone(),
    });

    let result = store.search_cross_thread_messages("東京", 10, None).await;
    assert!(matches!(
        result,
        Err(ConversationError::Driver(error))
            if error.to_string().contains("injected full-text failure")
    ));
    assert_eq!(probe.calls.load(Ordering::Relaxed), 1);
    assert_eq!(probe.thread_queries.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn full_text_search_uses_index_without_reading_thread_corpus() {
    let backend = MemoryStorage::new();
    let mut store = scoped(&backend, "profile");
    DriverConversationStore::prepare_schema(store.docs.as_ref())
        .await
        .unwrap();
    let mut expected = inverted_index::InvertedIndex::new();
    for (thread_id, content) in [
        ("tokyo", "東京tokyo"),
        ("kyoto", "京都は美しい"),
        ("other", "a short unrelated note"),
    ] {
        seed(&store, thread_id, &[content]).await;
        for message in store.get_messages(thread_id).await.unwrap() {
            expected.insert(thread_id, message);
        }
    }
    let normalize = |hits: &mut Vec<CrossThreadHit>| {
        hits.sort_by(|a, b| {
            a.thread_id
                .cmp(&b.thread_id)
                .then(a.message_id.cmp(&b.message_id))
        });
    };
    // These messages were seeded directly into the existing thread/stream
    // schema, so there are no derived rows yet. The first ordinary search
    // must backfill them before serving the query.
    let mut migrated = store
        .search_cross_thread_messages("東京", 20, None)
        .await
        .unwrap();
    let mut migration_reference = expected.search("東京", 20, None);
    normalize(&mut migrated);
    normalize(&mut migration_reference);
    assert_eq!(migrated, migration_reference);

    let probe = Arc::new(SearchProbe::default());
    store.docs = Arc::new(ProbedDocuments {
        inner: store.docs.clone(),
        probe: probe.clone(),
    });
    let mut actual = store
        .search_cross_thread_messages("東京", 20, None)
        .await
        .unwrap();
    let mut reference = expected.search("東京", 20, None);
    normalize(&mut actual);
    normalize(&mut reference);
    assert_eq!(
        actual, reference,
        "indexed search must preserve CJK search results"
    );
    assert!(
        probe.calls.load(Ordering::Relaxed) > 0,
        "FullText-capable stores must query DocumentStore::search"
    );
    assert_eq!(
        probe.thread_queries.load(Ordering::Relaxed),
        0,
        "steady-state search must not list the conversation thread corpus"
    );
}

#[tokio::test]
async fn pathological_full_text_queries_keep_the_global_recency_fallback() {
    let backend = MemoryStorage::new();
    let store = scoped(&backend, "profile");
    DriverConversationStore::prepare_schema(store.docs.as_ref())
        .await
        .unwrap();
    seed(&store, "popular", &vec!["popularterm"; 10_050]).await;
    store.ensure_thread(request("recent")).await.unwrap();
    let mut recent = store.load("recent").await.unwrap().unwrap();
    let recent_message = ThreadMessage {
        id: "message:0".into(),
        content: "unrelated recent message".into(),
        message_type: "text".into(),
        extra_metadata: json!({}),
        sender: "user".into(),
        created_at: "2030-01-01T00:00:00Z".into(),
    };
    let stream = recent.doc.stream().unwrap();
    store
        .streams
        .append(
            &stream,
            serde_json::to_value(MessageDto::from_message(&recent_message, to_ms).unwrap())
                .unwrap(),
        )
        .await
        .unwrap();
    recent.doc.visible_end = 1;
    recent.doc.thread.message_count = 1;
    recent.doc.thread.last_message_at = recent_message.created_at.clone();
    store
        .docs
        .put(
            COLLECTION,
            &recent.id,
            serde_json::to_value(recent.doc.to_dto().unwrap()).unwrap(),
            recent.unchanged(),
        )
        .await
        .unwrap();

    let mut scan = inverted_index::InvertedIndex::new();
    for thread_id in ["popular", "recent"] {
        for message in store.get_messages(thread_id).await.unwrap() {
            scan.insert(thread_id, message);
        }
    }
    for excluded in [None, Some("recent")] {
        assert_eq!(
            store
                .search_cross_thread_messages("popularterm", 2, excluded)
                .await
                .unwrap(),
            scan.search("popularterm", 2, excluded),
            "the large-candidate fallback preserves global recency and exclusion"
        );
    }
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
async fn native_scope_fence_rejects_metadata_takeover_without_local_fallback() {
    use tinystoragedrivers_core::Fence;
    let backend = MemoryStorage::new();
    let owner_scope = Scope::new("root-owner").unwrap();
    let owner = backend.for_scope(&owner_scope).unwrap();
    owner
        .documents()
        .put(
            "leases",
            "profile",
            json!({"epoch":1}),
            Precondition::Absent,
        )
        .await
        .unwrap();
    let profile = Scope::new("profile").unwrap();
    let fence = Fence::epoch(owner_scope, "leases", "profile", "epoch", 1);
    let store = DriverConversationStore::new(backend.for_scope_fenced(&profile, &fence).unwrap());
    let before = store.ensure_thread(request("one")).await.unwrap();
    owner
        .documents()
        .put("leases", "profile", json!({"epoch":2}), Precondition::None)
        .await
        .unwrap();
    assert!(
        matches!(store.update_thread_title("one", "stale", "now").await, Err(ConversationError::Driver(e)) if e.kind() == ErrorKind::Fenced)
    );
    assert_eq!(
        scoped(&backend, "profile").get_thread("one").await.unwrap(),
        Some(before)
    );
}

#[tokio::test]
async fn sqlite_reopen_reads_same_committed_profile_fixture() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("conversation.db");
    let scope = Scope::new("profile").unwrap();
    let backend = tinystoragedrivers_sqlite::SqliteStorage::open(&path).unwrap();
    let a = DriverConversationStore::new(backend.for_scope(&scope).unwrap());
    seed(&a, "one", &["東京 persisted"]).await;
    drop(a);
    drop(backend);
    let reopened = tinystoragedrivers_sqlite::SqliteStorage::open(&path).unwrap();
    let b = DriverConversationStore::new(reopened.for_scope(&scope).unwrap());
    assert_eq!(
        b.get_messages("one").await.unwrap()[0].content,
        "東京 persisted"
    );
}

#[tokio::test]
async fn sqlite_full_text_search_matches_scan_for_backfilled_cjk_messages() {
    let dir = tempfile::tempdir().unwrap();
    let backend =
        tinystoragedrivers_sqlite::SqliteStorage::open(dir.path().join("conversation-search.db"))
            .unwrap();
    let scope = Scope::new("profile").unwrap();
    let store = DriverConversationStore::new(backend.for_scope(&scope).unwrap());
    DriverConversationStore::prepare_schema(store.docs.as_ref())
        .await
        .unwrap();
    let mut expected = inverted_index::InvertedIndex::new();
    for (thread_id, content) in [
        ("tokyo", "東京tokyo"),
        ("kyoto", "京都は美しい日本語"),
        ("mixed", "Café 東京 substring"),
        ("other", "unrelated note"),
    ] {
        seed(&store, thread_id, &[content]).await;
        for message in store.get_messages(thread_id).await.unwrap() {
            expected.insert(thread_id, message);
        }
    }
    for query in ["東京", "日", "cafe 東京", "not-found"] {
        let mut actual = store
            .search_cross_thread_messages(query, 20, Some("other"))
            .await
            .unwrap();
        let mut reference = expected.search(query, 20, Some("other"));
        let normalize = |hits: &mut Vec<CrossThreadHit>| {
            hits.sort_by(|a, b| {
                a.thread_id
                    .cmp(&b.thread_id)
                    .then(a.message_id.cmp(&b.message_id))
            });
        };
        normalize(&mut actual);
        normalize(&mut reference);
        assert_eq!(actual, reference, "SQLite parity for {query:?}");
    }
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
    let docs = Arc::new(fixtures::ConflictingDocs {
        inner: ports.documents().clone(),
        attempts: AtomicUsize::new(0),
        barrier: tokio::sync::Barrier::new(2),
        refuse_get: false,
    });
    let a = DriverConversationStore::new(ScopedStorage::new(
        scope.clone(),
        "test",
        docs.clone(),
        ports.streams().clone(),
        ports.blobs().clone(),
    ));
    let b = DriverConversationStore::new(ScopedStorage::new(
        scope,
        "test",
        docs.clone(),
        ports.streams().clone(),
        ports.blobs().clone(),
    ));
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
    let store = DriverConversationStore::new(ScopedStorage::new(
        scope,
        "refused",
        docs,
        ports.streams().clone(),
        ports.blobs().clone(),
    ));
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
