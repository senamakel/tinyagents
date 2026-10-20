use super::*;
use std::collections::BTreeSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use tinystoragedrivers_api::DocumentStore;
use tinystoragedrivers_types::{
    Capabilities, CollectionSpec, ErrorKind, Filter, Page, Precondition, Query, Sort, StorageError,
    Version, Versioned, WriteOptions,
};

struct DeleteProbeDocs {
    inner: Arc<dyn DocumentStore>,
    thread_deletes: AtomicUsize,
    successful_thread_deletes: AtomicUsize,
    fail_on_thread_delete: Option<usize>,
    return_false_after_thread_delete: Option<usize>,
    inject_thread_after_first_delete: Mutex<Option<ThreadDoc>>,
}

impl std::fmt::Debug for DeleteProbeDocs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeleteProbeDocs").finish_non_exhaustive()
    }
}

#[async_trait]
impl DocumentStore for DeleteProbeDocs {
    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }

    async fn validate_collection_spec(
        &self,
        required: &CollectionSpec,
    ) -> tinystoragedrivers_types::Result<()> {
        self.inner.validate_collection_spec(required).await
    }

    async fn ensure_collection(
        &self,
        spec: &CollectionSpec,
    ) -> tinystoragedrivers_types::Result<()> {
        self.inner.ensure_collection(spec).await
    }

    async fn get(
        &self,
        collection: &str,
        id: &str,
    ) -> tinystoragedrivers_types::Result<Option<Versioned<serde_json::Value>>> {
        self.inner.get(collection, id).await
    }

    async fn put(
        &self,
        collection: &str,
        id: &str,
        doc: serde_json::Value,
        precondition: Precondition,
    ) -> tinystoragedrivers_types::Result<Version> {
        self.inner.put(collection, id, doc, precondition).await
    }

    async fn put_with(
        &self,
        collection: &str,
        id: &str,
        doc: serde_json::Value,
        precondition: Precondition,
        options: WriteOptions,
    ) -> tinystoragedrivers_types::Result<Version> {
        self.inner
            .put_with(collection, id, doc, precondition, options)
            .await
    }

    async fn delete(
        &self,
        collection: &str,
        id: &str,
        precondition: Precondition,
    ) -> tinystoragedrivers_types::Result<bool> {
        if collection == COLLECTION {
            let attempt = self.thread_deletes.fetch_add(1, Ordering::SeqCst) + 1;
            if self.fail_on_thread_delete == Some(attempt) {
                return Err(StorageError::unavailable("injected thread delete failure"));
            }
            let deleted = self.inner.delete(collection, id, precondition).await?;
            if deleted && self.return_false_after_thread_delete == Some(attempt) {
                return Ok(false);
            }
            if deleted {
                let successful = self
                    .successful_thread_deletes
                    .fetch_add(1, Ordering::SeqCst);
                if successful == 0 {
                    let thread = {
                        self.inject_thread_after_first_delete
                            .lock()
                            .expect("fixture injection mutex")
                            .take()
                    };
                    if let Some(thread) = thread {
                        let key = DriverConversationStore::key(&thread.thread.id)
                            .expect("fixture thread id is valid");
                        self.inner
                            .put_with(
                                COLLECTION,
                                &key,
                                serde_json::to_value(
                                    thread.to_dto().expect("fixture thread converts to DTO"),
                                )
                                .expect("fixture thread DTO serializes"),
                                Precondition::Absent,
                                WriteOptions::new().schema(SCHEMA_VERSION),
                            )
                            .await?;
                    }
                }
            }
            return Ok(deleted);
        }
        self.inner.delete(collection, id, precondition).await
    }

    async fn query(
        &self,
        collection: &str,
        query: &Query,
    ) -> tinystoragedrivers_types::Result<Page<Versioned<serde_json::Value>>> {
        self.inner.query(collection, query).await
    }

    async fn count(
        &self,
        collection: &str,
        filter: &Filter,
    ) -> tinystoragedrivers_types::Result<u64> {
        self.inner.count(collection, filter).await
    }

    async fn delete_where(
        &self,
        collection: &str,
        filter: &Filter,
    ) -> tinystoragedrivers_types::Result<u64> {
        self.inner.delete_where(collection, filter).await
    }

    async fn claim(
        &self,
        collection: &str,
        filter: &Filter,
        sort: &[Sort],
        patch: &serde_json::Value,
    ) -> tinystoragedrivers_types::Result<Option<Versioned<serde_json::Value>>> {
        self.inner.claim(collection, filter, sort, patch).await
    }

    async fn atomic_batch_checked(
        &self,
        checks: Vec<tinystoragedrivers_types::ReadCheck>,
        ops: Vec<tinystoragedrivers_types::WriteOp>,
    ) -> tinystoragedrivers_types::Result<Vec<tinystoragedrivers_types::WriteResult>> {
        self.inner.atomic_batch_checked(checks, ops).await
    }

    async fn drop_collection(&self, collection: &str) -> tinystoragedrivers_types::Result<()> {
        self.inner.drop_collection(collection).await
    }
}

fn request(id: &str) -> CreateConversationThread {
    CreateConversationThread {
        id: id.into(),
        title: "Purge callback".into(),
        created_at: "2026-01-01T00:00:00Z".into(),
        parent_thread_id: None,
        labels: None,
        personality_id: None,
        working_dir: None,
    }
}

fn message(id: &str) -> ThreadMessage {
    ThreadMessage {
        id: id.into(),
        content: format!("message {id}"),
        message_type: "text".into(),
        extra_metadata: serde_json::json!({}),
        sender: "user".into(),
        created_at: "2026-01-02T00:00:00Z".into(),
    }
}

async fn probed_store(
    fail_on_thread_delete: Option<usize>,
    inject_thread_after_first_delete: Option<&str>,
    return_false_after_thread_delete: Option<usize>,
) -> (DriverConversationStore, Arc<DeleteProbeDocs>) {
    let backend = crate::testkit::MemoryStorage::new();
    let scope =
        tinyagents_storage_kit::Scope::new("purge-callback-probe").expect("fixture scope is valid");
    let scoped = backend.for_scope(&scope).expect("memory ports");
    DriverConversationStore::prepare_schema(scoped.documents().as_ref())
        .await
        .expect("conversation schema");
    DriverConversationStore::prepare_append_schema(scoped.documents().as_ref())
        .await
        .expect("append intent schema");
    let injected_thread = inject_thread_after_first_delete
        .map(|id| DriverConversationStore::ensure(None, &request(id)));
    let docs = Arc::new(DeleteProbeDocs {
        inner: scoped.documents().clone(),
        thread_deletes: AtomicUsize::new(0),
        successful_thread_deletes: AtomicUsize::new(0),
        fail_on_thread_delete,
        return_false_after_thread_delete,
        inject_thread_after_first_delete: Mutex::new(injected_thread),
    });
    let store = DriverConversationStore::new(crate::testkit::with_documents(&scoped, docs.clone()));
    (store, docs)
}

#[tokio::test]
async fn purge_callback_reports_logical_ids_after_each_live_or_tombstoned_delete() {
    let (store, docs) = probed_store(None, None, None).await;
    store.ensure_thread(request("channel/live")).await.unwrap();
    store
        .append_message("channel/live", message("live-1"))
        .await
        .unwrap();
    store
        .append_message("channel/live", message("live-2"))
        .await
        .unwrap();
    store
        .ensure_thread(request("channel/tombstoned"))
        .await
        .unwrap();
    store
        .append_message("channel/tombstoned", message("tombstoned-1"))
        .await
        .unwrap();
    assert!(
        store
            .delete_thread("channel/tombstoned", "now")
            .await
            .unwrap()
    );

    let mut deleted = Vec::new();
    let mut observed_deletes = 0;
    let stats = store
        .purge_threads_with(|thread_id| {
            observed_deletes += 1;
            assert_eq!(
                docs.successful_thread_deletes.load(Ordering::SeqCst),
                observed_deletes,
                "each callback follows exactly one successful thread document delete"
            );
            deleted.push(thread_id.to_owned());
        })
        .await
        .unwrap();

    assert_eq!(
        deleted.into_iter().collect::<BTreeSet<_>>(),
        BTreeSet::from(["channel/live".to_owned(), "channel/tombstoned".to_owned()])
    );
    assert_eq!((stats.thread_count, stats.message_count), (1, 2));
    assert_eq!(docs.inner.count(COLLECTION, &Filter::All).await.unwrap(), 0);
}

#[tokio::test]
async fn purge_callback_keeps_successful_ids_when_a_later_document_delete_fails() {
    let (store, docs) = probed_store(Some(2), None, None).await;
    for id in ["logical/one", "logical/two", "logical/three"] {
        store.ensure_thread(request(id)).await.unwrap();
    }
    let page = store
        .docs
        .query(COLLECTION, &Query::filter(Filter::All).limit(PAGE_SIZE))
        .await
        .unwrap();
    let expected_first = DriverConversationStore::decode(page.items.into_iter().next().unwrap())
        .unwrap()
        .doc
        .thread
        .id;

    let mut callbacks = Vec::new();
    let result = store
        .purge_threads_with(|thread_id| callbacks.push(thread_id.to_owned()))
        .await;

    assert!(matches!(
        result,
        Err(ConversationError::Driver(error)) if error.kind() == ErrorKind::Unavailable
    ));
    assert_eq!(callbacks, vec![expected_first]);
    assert_eq!(docs.successful_thread_deletes.load(Ordering::SeqCst), 1);
    assert_eq!(docs.inner.count(COLLECTION, &Filter::All).await.unwrap(), 2);
}

#[tokio::test]
async fn async_purge_callback_keeps_completed_ids_when_a_later_delete_fails() {
    let (store, docs) = probed_store(Some(2), None, None).await;
    for id in ["logical/one", "logical/two", "logical/three"] {
        store.ensure_thread(request(id)).await.unwrap();
    }
    let page = store
        .docs
        .query(COLLECTION, &Query::filter(Filter::All).limit(PAGE_SIZE))
        .await
        .unwrap();
    let expected_first = DriverConversationStore::decode(page.items.into_iter().next().unwrap())
        .unwrap()
        .doc
        .thread
        .id;

    let mut callbacks = Vec::new();
    let result = store
        .purge_threads_with_async(|thread_id| {
            callbacks.push(thread_id);
            std::future::ready(())
        })
        .await;

    assert!(matches!(
        result,
        Err(ConversationError::Driver(error)) if error.kind() == ErrorKind::Unavailable
    ));
    assert_eq!(callbacks, vec![expected_first]);
    assert_eq!(docs.successful_thread_deletes.load(Ordering::SeqCst), 1);
    assert_eq!(docs.inner.count(COLLECTION, &Filter::All).await.unwrap(), 2);
}

#[tokio::test]
async fn purge_callback_enumerates_paged_and_new_threads() {
    let inserted_id = "channel/inserted-during-purge";
    let (store, docs) = probed_store(None, Some(inserted_id), None).await;
    let mut expected = (0..PAGE_SIZE + 3)
        .map(|index| format!("channel/page-{index:03}"))
        .collect::<BTreeSet<_>>();
    for id in &expected {
        store.ensure_thread(request(id)).await.unwrap();
    }
    let injected_key = DriverConversationStore::key(inserted_id).unwrap();
    assert!(
        docs.inner
            .get(COLLECTION, &injected_key)
            .await
            .unwrap()
            .is_none(),
        "fixture thread must be absent before the purge begins"
    );
    expected.insert(inserted_id.to_owned());

    let mut deleted = BTreeSet::new();
    let stats = store
        .purge_threads_with(|thread_id| {
            assert_eq!(
                docs.successful_thread_deletes.load(Ordering::SeqCst),
                deleted.len() + 1,
                "callback follows each successful delete before another is attempted"
            );
            assert!(
                deleted.insert(thread_id.to_owned()),
                "duplicate callback id"
            );
        })
        .await
        .unwrap();

    assert_eq!(deleted, expected);
    assert_eq!(stats.thread_count, PAGE_SIZE + 4);
    assert_eq!(stats.message_count, 0);
    assert_eq!(docs.inner.count(COLLECTION, &Filter::All).await.unwrap(), 0);
}

#[tokio::test]
async fn async_purge_rechecks_after_last_page_delete_and_callback_insertions() {
    let probe_inserted_id = "channel/probe-inserted-on-final-delete";
    let callback_inserted_id = "channel/async-callback-inserted-on-final-delete";
    let (store, docs) = probed_store(None, Some(probe_inserted_id), None).await;
    store
        .ensure_thread(request("channel/only-original"))
        .await
        .unwrap();

    let callback_store = store.clone();
    let callbacks = Arc::new(Mutex::new(Vec::new()));
    let observed = Arc::clone(&callbacks);
    let stats = store
        .purge_threads_with_async(move |thread_id| {
            let callback_store = callback_store.clone();
            let observed = Arc::clone(&observed);
            async move {
                let was_original = thread_id == "channel/only-original";
                observed.lock().unwrap().push(thread_id);
                if was_original {
                    callback_store
                        .ensure_thread(request(callback_inserted_id))
                        .await
                        .unwrap();
                }
            }
        })
        .await
        .unwrap();

    let callbacks = callbacks
        .lock()
        .unwrap()
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    assert_eq!(
        callbacks,
        BTreeSet::from([
            "channel/only-original".to_owned(),
            probe_inserted_id.to_owned(),
            callback_inserted_id.to_owned(),
        ]),
        "the final-page probe row and async callback row are both purged and reported"
    );
    assert_eq!((stats.thread_count, stats.message_count), (3, 0));
    assert_eq!(docs.successful_thread_deletes.load(Ordering::SeqCst), 3);
    assert_eq!(docs.inner.count(COLLECTION, &Filter::All).await.unwrap(), 0);
}

#[tokio::test]
async fn purge_callback_skips_a_delete_reported_as_not_removed() {
    let thread_id = "logical/concurrent-delete";
    let (store, docs) = probed_store(None, None, Some(1)).await;
    store.ensure_thread(request(thread_id)).await.unwrap();
    store
        .append_message(thread_id, message("one"))
        .await
        .unwrap();
    let key = DriverConversationStore::key(thread_id).unwrap();

    let mut callbacks = Vec::new();
    let stats = store
        .purge_threads_with(|deleted_id| callbacks.push(deleted_id.to_owned()))
        .await
        .unwrap();

    assert!(callbacks.is_empty());
    assert_eq!((stats.thread_count, stats.message_count), (1, 1));
    assert!(docs.inner.get(COLLECTION, &key).await.unwrap().is_none());
    assert_eq!(docs.thread_deletes.load(Ordering::SeqCst), 1);
    assert_eq!(docs.successful_thread_deletes.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn async_purge_callback_is_awaited_before_the_next_document_delete() {
    let (store, docs) = probed_store(None, None, None).await;
    store.ensure_thread(request("logical/live")).await.unwrap();
    store
        .ensure_thread(request("logical/tombstoned"))
        .await
        .unwrap();
    store
        .append_message("logical/tombstoned", message("tombstoned-message"))
        .await
        .unwrap();
    assert!(
        store
            .delete_thread("logical/tombstoned", "now")
            .await
            .unwrap()
    );

    let (callback_tx, mut callback_rx) = tokio::sync::mpsc::unbounded_channel();
    let purge = tokio::spawn(async move {
        store
            .purge_threads_with_async(move |thread_id| {
                let (release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();
                callback_tx
                    .send((thread_id, release_tx))
                    .expect("test is waiting for every callback");
                async move {
                    release_rx
                        .await
                        .expect("test releases each callback before finishing");
                }
            })
            .await
    });

    let (first_id, release_first) = callback_rx.recv().await.unwrap();
    assert_eq!(docs.successful_thread_deletes.load(Ordering::SeqCst), 1);
    assert!(callback_rx.try_recv().is_err());
    release_first.send(()).unwrap();

    let (second_id, release_second) = callback_rx.recv().await.unwrap();
    assert_ne!(first_id, second_id);
    assert_eq!(docs.successful_thread_deletes.load(Ordering::SeqCst), 2);
    assert!(callback_rx.try_recv().is_err());
    release_second.send(()).unwrap();

    let stats = purge.await.unwrap().unwrap();
    assert_eq!(
        BTreeSet::from([first_id, second_id]),
        BTreeSet::from(["logical/live".to_owned(), "logical/tombstoned".to_owned()])
    );
    assert_eq!((stats.thread_count, stats.message_count), (1, 0));
    assert_eq!(docs.inner.count(COLLECTION, &Filter::All).await.unwrap(), 0);
}

#[tokio::test]
async fn cancelling_an_async_purge_callback_does_not_process_the_next_row() {
    let (store, docs) = probed_store(None, None, None).await;
    for id in ["logical/one", "logical/two"] {
        store.ensure_thread(request(id)).await.unwrap();
    }

    let (callback_tx, mut callback_rx) = tokio::sync::mpsc::unbounded_channel();
    let purge = tokio::spawn(async move {
        store
            .purge_threads_with_async(move |thread_id| {
                let (release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();
                callback_tx
                    .send((thread_id, release_tx))
                    .expect("test is waiting for the callback");
                async move {
                    let _ = release_rx.await;
                }
            })
            .await
    });

    let (_deleted_id, _release) = callback_rx.recv().await.unwrap();
    assert_eq!(docs.successful_thread_deletes.load(Ordering::SeqCst), 1);
    purge.abort();
    assert!(purge.await.unwrap_err().is_cancelled());
    assert_eq!(docs.successful_thread_deletes.load(Ordering::SeqCst), 1);
    assert_eq!(docs.inner.count(COLLECTION, &Filter::All).await.unwrap(), 1);
}
