//! Admitted scoped conversation metadata and immutable committed-prefix readers.
//!
//! No host paths/backend lookup, FullText shortcut, or persistent process cache.
//! Plain append uses a recoverable intent; edits and distributed leases remain unsupported.
//!
//! Threads are `conversations.threads` documents (schema v2, see the private `dto` module).
//! A change to one thread document is one [`Repo::update`] loop. The loops that
//! stay hand-written coordinate more than that one document: an append
//! reconciles an intent, a stream row and the thread together, and a rewrite
//! copies a whole log before it swaps the thread onto it.

use super::dto::{MessageDto, ThreadDto, ThreadState, to_ms};
use super::persistence::*;
use super::schema::{SCHEMA_VERSION, THREADS as COLLECTION, append_spec, stream_name, thread_spec};
use super::*;
use async_trait::async_trait;
use std::future::Future;
use std::sync::Arc;
use tinyagents_storage_kit::ScopedStorage;
use tinyagents_storage_kit::primitives::Repo;
use tinyagents_storage_kit::{
    CollectionSpec, ErrorKind, Filter, Key, Precondition, Query, StorageError, Versioned,
    WriteOptions,
};
use tinystoragedrivers_api::{DocumentStore, StreamStore};
mod append;
mod pending;
mod reader;
mod rewrite;

const PAGE_SIZE: usize = 64;
/// Attempts of the loops that reconcile more than one document or stream.
const RETRIES: usize = 32;

/// A thread document decoded: the wire thread beside its stream state.
#[derive(Clone)]
struct ThreadDoc {
    incarnation: String,
    deleted: bool,
    visible_end: u64,
    /// The thread segment of the stream's name when it is not the thread id
    /// (a migrated v1 stream; see [`super::schema`]).
    messages_key: Option<String>,
    thread: ConversationThread,
}
impl ThreadDoc {
    /// The stored body. Times must be RFC 3339; the count is `visible_end`.
    fn to_dto(&self) -> Result<ThreadDto, StorageError> {
        ThreadDto::from_wire(
            &self.thread,
            ThreadState {
                incarnation: self.incarnation.clone(),
                deleted: self.deleted,
                visible_end: self.visible_end,
                messages_key: self.messages_key.clone(),
            },
            to_ms,
        )
    }
    fn from_dto(dto: ThreadDto) -> Result<Self, StorageError> {
        let (thread, state) = dto.into_wire()?;
        Ok(Self {
            incarnation: state.incarnation,
            deleted: state.deleted,
            visible_end: state.visible_end,
            messages_key: state.messages_key,
            thread,
        })
    }
    /// The name of the stream holding this incarnation's committed messages.
    fn stream(&self) -> Result<String, StorageError> {
        stream_name(
            self.messages_key.as_deref().unwrap_or(&self.thread.id),
            &self.incarnation,
        )
    }
}

/// Conversation handle whose scope/native admission has already been bound.
///
/// Schema1 metadata edits commit one document; append atomically publishes its
/// metadata and retires a companion intent. Committed messages
/// live in an immutable per-incarnation prefix. This adapter exposes no message
/// importer or claimed commit; plain append requires its own prepared intent collection.
#[derive(Clone)]
pub struct DriverConversationStore {
    docs: Arc<dyn DocumentStore>,
    streams: Arc<dyn StreamStore>,
    validate_only: bool,
}
impl DriverConversationStore {
    /// Consume explicit trusted scoped ports with unchanged legacy schema behavior.
    ///
    /// This constructor neither declares nor validates application schema and
    /// does not establish actor schema safety. Use explicit trusted preparation
    /// and [`Self::new_validate_only`] for restricted actor admission. No ambient
    /// backend or file fallback is possible.
    pub fn new(scoped: ScopedStorage) -> Self {
        Self {
            docs: scoped.documents().clone(),
            streams: scoped.streams().clone(),
            validate_only: false,
        }
    }
    /// Consume admitted ports with immutable read-only schema validation.
    ///
    /// Every implemented operation validates persisted required features anew;
    /// no successful validation is cached. Compatible native declaration
    /// supersets are accepted. This neither restricts the supplied backend's
    /// implicit initialization nor establishes actor/scope authority: the host
    /// must supply native existing-schema handles and its original fence.
    pub fn new_validate_only(scoped: ScopedStorage) -> Self {
        Self {
            validate_only: true,
            ..Self::new(scoped)
        }
    }

    /// The `conversations.threads` declaration.
    ///
    /// There are no required secondary indexes, TTL or full-text fields.
    /// Native validation accepts compatible supersets, not exact equality.
    pub fn collection_specs() -> Vec<CollectionSpec> {
        vec![thread_spec()]
    }

    /// Declare application metadata under explicitly trusted bootstrap authority.
    ///
    /// This does not prepare native internal schemas, bind an actor, migrate
    /// data or enable message writes. Restricted actor handles must refuse.
    ///
    /// # Errors
    /// The native declaration error, preserving its kind and original cause.
    pub async fn prepare_schema(
        documents: &dyn DocumentStore,
    ) -> tinystoragedrivers_types::Result<()> {
        for spec in Self::collection_specs() {
            documents.ensure_collection(&spec).await?;
        }
        Ok(())
    }

    async fn admit(&self) -> ConversationResult<()> {
        if self.validate_only {
            for spec in Self::collection_specs() {
                self.docs.validate_collection_spec(&spec).await?;
            }
        }
        Ok(())
    }

    /// Purge thread documents and notify the caller after each successful removal.
    ///
    /// The callback receives the logical thread ID from the stored conversation,
    /// including for tombstoned threads. It runs synchronously, immediately after
    /// the document store confirms that row was deleted and before another row is
    /// processed. A `false` delete result does not invoke the callback.
    ///
    /// The callback is infallible by API shape and runs on the caller's executor.
    /// Keep it limited to recording the obligation; do not perform follow-up work,
    /// panic, or call back into this store. If durable recording uses synchronous
    /// I/O, run the purge from an appropriate blocking executor. Completed
    /// callbacks remain observable if a later query, decode, or delete fails; the
    /// purge is not crash-atomic, so a process failure between deletion and the
    /// callback can still leave a deletion obligation unreported. Cancellation
    /// while a document delete is in flight can likewise leave a deletion that
    /// the backend completed without a callback.
    ///
    /// Purging repeatedly reads from the head until a fresh query returns no
    /// rows, so rows inserted during deletion or callbacks are included when
    /// observed before that empty query. This is not a transactional snapshot:
    /// rows inserted after the final empty query are outside the purge.
    ///
    /// The returned counts retain [`ConversationPersistence::purge_threads`]
    /// semantics: tombstones are removed but are not included in the live thread
    /// or message totals.
    ///
    /// # Errors
    /// Returns the same admission and driver errors as
    /// [`ConversationPersistence::purge_threads`].
    pub async fn purge_threads_with<F>(
        &self,
        mut after_delete: F,
    ) -> ConversationResult<ConversationPurgeStats>
    where
        F: FnMut(&str) + Send,
    {
        self.admit().await?;
        self.purge_native_with(move |thread_id| {
            after_delete(&thread_id);
            std::future::ready(())
        })
        .await
    }

    /// Purge threads and await a callback after each successfully deleted thread document.
    ///
    /// The owned logical thread ID is delivered for live and tombstoned threads.
    /// The callback is awaited immediately after the document store confirms the
    /// deletion and before another row is processed. A `false` delete result
    /// does not invoke the callback. This variant is suitable when recording the
    /// deletion obligation requires asynchronous work, including work that must
    /// yield to the storage bridge's executor.
    ///
    /// The callback is infallible by API shape. If a later query, decode, or
    /// delete fails, completed callbacks remain observable and the purge returns
    /// the same error as [`ConversationPersistence::purge_threads`]. The purge
    /// is not crash-atomic: a process failure between document deletion and
    /// callback completion can leave an obligation unreported. Cancellation
    /// while a document delete is in flight can leave a deletion that the
    /// backend completed without a callback; cancellation after deletion can
    /// also stop the purge before an async callback completes.
    ///
    /// Purging repeatedly reads from the head until a fresh query returns no
    /// rows, so rows inserted during deletion or callbacks are included when
    /// observed before that empty query. This is not a transactional snapshot:
    /// rows inserted after the final empty query are outside the purge.
    ///
    /// The returned counts retain [`ConversationPersistence::purge_threads`]
    /// semantics: tombstones are removed but are not included in the live thread
    /// or message totals.
    pub async fn purge_threads_with_async<F, Fut>(
        &self,
        after_delete: F,
    ) -> ConversationResult<ConversationPurgeStats>
    where
        F: FnMut(String) -> Fut + Send,
        Fut: Future<Output = ()> + Send,
    {
        self.admit().await?;
        self.purge_native_with_async(after_delete).await
    }

    fn key(id: &str) -> ConversationResult<String> {
        if id.is_empty() || id.contains('\0') || id.len() > 1024 * 1024 {
            return Err(StorageError::invalid_input(
                "thread id must be nonempty, contain no NUL, and fit 1MiB",
            )
            .into());
        }
        Ok(Key::of([id])?.as_str().to_owned())
    }
    /// A thread document of schema v2 at `id`, checked against its own identity.
    fn thread_doc(id: &str, dto: ThreadDto) -> Result<ThreadDoc, StorageError> {
        let stored_key = Self::key(&dto.id).map_err(|error| {
            StorageError::serialization("invalid stored conversation identity").with_source(error)
        })?;
        if uuid::Uuid::parse_str(&dto.incarnation).is_err() || stored_key != id {
            return Err(StorageError::serialization(
                "conversation metadata identity/incarnation mismatch",
            ));
        }
        ThreadDoc::from_dto(dto)
    }
    fn decode(row: Versioned<serde_json::Value>) -> ConversationResult<Versioned<ThreadDoc>> {
        let dto: ThreadDto = serde_json::from_value(row.doc).map_err(|e| {
            StorageError::serialization("invalid conversation metadata").with_source(e)
        })?;
        Ok(Versioned {
            doc: Self::thread_doc(&row.id, dto)?,
            id: row.id,
            version: row.version,
            meta: row.meta,
        })
    }
    /// Thread documents as a typed handle whose writes declare the v2 body version.
    fn threads(&self) -> Result<Repo<ThreadDto>, StorageError> {
        Ok(Repo::new(&self.docs, COLLECTION)?
            .with_options(WriteOptions::new().schema(SCHEMA_VERSION)))
    }
    async fn load(&self, id: &str) -> ConversationResult<Option<Versioned<ThreadDoc>>> {
        self.admit().await?;
        let key = Self::key(id)?;
        self.docs
            .get(COLLECTION, &key)
            .await?
            .map(Self::decode)
            .transpose()
    }
    /// Replace the thread `id` with what `operation` makes of the current one
    /// (`None` when absent), as one [`Repo::update`] loop: `operation` runs
    /// again after a lost swap, so it must be free of side effects.
    async fn change(
        &self,
        id: &str,
        operation: impl Fn(Option<&ThreadDoc>) -> Result<ThreadDoc, StorageError> + Send + Sync,
    ) -> ConversationResult<ConversationThread> {
        self.admit().await?;
        let key = Self::key(id)?;
        let stored = self
            .threads()?
            .update(&key, |current| {
                let current = current.map(|dto| Self::thread_doc(&key, dto)).transpose()?;
                Ok(Some(operation(current.as_ref())?.to_dto()?))
            })
            .await?;
        let (dto, _) =
            stored.ok_or_else(|| StorageError::backend("conversation update wrote nothing"))?;
        Ok(ThreadDoc::from_dto(dto)?.thread)
    }
    fn live(current: Option<&ThreadDoc>) -> Result<ThreadDoc, StorageError> {
        current
            .filter(|doc| !doc.deleted)
            .cloned()
            .ok_or_else(|| StorageError::not_found("conversation thread not found"))
    }
    fn ensure(current: Option<&ThreadDoc>, request: &CreateConversationThread) -> ThreadDoc {
        if let Some(current) = current.filter(|doc| !doc.deleted) {
            let mut next = current.clone();
            next.thread.title = request.title.clone();
            if let Some(parent) = &request.parent_thread_id {
                next.thread.parent_thread_id = Some(parent.clone());
            }
            if let Some(labels) = &request.labels {
                next.thread.labels = super::labels::normalize_labels(labels.clone());
            }
            if let Some(personality) = &request.personality_id {
                next.thread.personality_id = Some(personality.clone());
            }
            if let Some(dir) = &request.working_dir {
                next.thread.working_dir = (!dir.is_empty()).then(|| dir.clone());
            }
            next
        } else {
            ThreadDoc {
                incarnation: uuid::Uuid::new_v4().to_string(),
                deleted: false,
                visible_end: 0,
                messages_key: None,
                thread: ConversationThread {
                    id: request.id.clone(),
                    title: request.title.clone(),
                    chat_id: None,
                    is_active: true,
                    message_count: 0,
                    last_message_at: request.created_at.clone(),
                    created_at: request.created_at.clone(),
                    parent_thread_id: request.parent_thread_id.clone(),
                    labels: request
                        .labels
                        .clone()
                        .map(super::labels::normalize_labels)
                        .unwrap_or_else(|| super::labels::infer_labels(&request.id)),
                    personality_id: request.personality_id.clone(),
                    working_dir: request.working_dir.clone().filter(|dir| !dir.is_empty()),
                },
            }
        }
    }
}

#[async_trait]
impl ConversationPersistence for DriverConversationStore {
    async fn ensure_thread(
        &self,
        request: CreateConversationThread,
    ) -> ConversationResult<ConversationThread> {
        self.admit().await?;
        self.change(&request.id, |current| Ok(Self::ensure(current, &request)))
            .await
    }
    async fn get_thread(&self, thread_id: &str) -> ConversationResult<Option<ConversationThread>> {
        self.admit().await?;
        Ok(self
            .load(thread_id)
            .await?
            .filter(|row| !row.doc.deleted)
            .map(|row| row.doc.thread))
    }
    async fn list_threads(&self) -> ConversationResult<Vec<ConversationThread>> {
        self.admit().await?;
        let mut rows: Vec<_> = self
            .live_rows()
            .await?
            .into_iter()
            .map(|row| row.doc.thread)
            .collect();
        rows.sort_by(|a, b| {
            timestamp(&b.last_message_at)
                .cmp(&timestamp(&a.last_message_at))
                .then_with(|| timestamp(&b.created_at).cmp(&timestamp(&a.created_at)))
        });
        Ok(rows)
    }
    async fn get_messages(&self, thread_id: &str) -> ConversationResult<Vec<ThreadMessage>> {
        self.admit().await?;
        self.messages(thread_id).await
    }
    async fn update_thread_title(
        &self,
        thread_id: &str,
        title: &str,
        _updated_at: &str,
    ) -> ConversationResult<ConversationThread> {
        self.admit().await?;
        self.change(thread_id, |current| {
            let mut doc = Self::live(current)?;
            doc.thread.title = title.into();
            Ok(doc)
        })
        .await
    }
    async fn update_thread_labels(
        &self,
        thread_id: &str,
        labels: Vec<String>,
        _updated_at: &str,
    ) -> ConversationResult<ConversationThread> {
        self.admit().await?;
        self.change(thread_id, |current| {
            let mut doc = Self::live(current)?;
            doc.thread.labels = super::labels::normalize_labels(labels.clone());
            Ok(doc)
        })
        .await
    }
    async fn update_thread_working_dir(
        &self,
        thread_id: &str,
        working_dir: Option<String>,
        _updated_at: &str,
    ) -> ConversationResult<ConversationThread> {
        self.admit().await?;
        self.change(thread_id, |current| {
            let mut doc = Self::live(current)?;
            doc.thread.working_dir = working_dir
                .as_ref()
                .map(|dir| dir.trim().to_owned())
                .filter(|dir| !dir.is_empty());
            Ok(doc)
        })
        .await
    }
    async fn delete_thread(&self, thread_id: &str, _deleted_at: &str) -> ConversationResult<bool> {
        self.admit().await?;
        let key = Self::key(thread_id)?;
        let mut deleted = false;
        self.threads()?
            .update(&key, |current| {
                deleted = false;
                let Some(current) = current else {
                    return Ok(None);
                };
                let mut doc = Self::thread_doc(&key, current)?;
                if doc.deleted {
                    return Ok(None);
                }
                doc.deleted = true;
                deleted = true;
                Ok(Some(doc.to_dto()?))
            })
            .await?;
        Ok(deleted)
    }
    async fn search_cross_thread_messages(
        &self,
        query: &str,
        limit: usize,
        exclude_thread_id: Option<&str>,
    ) -> ConversationResult<Vec<CrossThreadHit>> {
        self.admit().await?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        let mut index = inverted_index::InvertedIndex::new();
        for row in self.live_rows().await? {
            if exclude_thread_id == Some(row.doc.thread.id.as_str()) {
                continue;
            }
            for message in self.messages(&row.doc.thread.id).await? {
                index.insert(&row.doc.thread.id, message);
            }
        }
        Ok(index.search(query, limit, exclude_thread_id))
    }
    async fn append_message(
        &self,
        thread_id: &str,
        message: ThreadMessage,
    ) -> ConversationResult<ThreadMessage> {
        self.append_native(thread_id, message).await
    }
    async fn update_message(
        &self,
        thread_id: &str,
        message_id: &str,
        patch: ThreadMessagePatch,
    ) -> ConversationResult<ThreadMessage> {
        self.admit().await?;
        self.update_message_native(thread_id, message_id, patch)
            .await
    }
    async fn delete_messages_from(
        &self,
        thread_id: &str,
        message_id: &str,
    ) -> ConversationResult<Option<usize>> {
        self.admit().await?;
        self.delete_messages_from_native(thread_id, message_id)
            .await
    }
    async fn purge_threads(&self) -> ConversationResult<ConversationPurgeStats> {
        self.purge_threads_with(|_| {}).await
    }
}
fn timestamp(value: &str) -> i64 {
    chrono::DateTime::parse_from_rfc3339(value)
        .map(|v| v.timestamp_millis())
        .unwrap_or(i64::MIN)
}
#[cfg(test)]
#[path = "driver_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "driver_append_tests.rs"]
mod append_tests;

#[cfg(test)]
#[path = "driver_purge_callback_tests.rs"]
mod purge_callback_tests;
