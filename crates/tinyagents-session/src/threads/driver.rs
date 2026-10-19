//! Admitted scoped conversation metadata and immutable committed-prefix readers.
//!
//! No host paths/backend lookup; FullText candidates live in the scoped search collection.
//! Plain append uses a recoverable intent; edits and distributed leases remain unsupported.
//!
//! Threads are `conversations.threads` documents (schema v2, see [`super::dto`]).
//! A change to one thread document is one [`Repo::update`] loop. The loops that
//! stay hand-written coordinate more than that one document: an append
//! reconciles an intent, a stream row and the thread together, and a rewrite
//! copies a whole log before it swaps the thread onto it.

use super::dto::{MessageDto, ThreadDto, ThreadState, to_ms};
use super::persistence::*;
use super::schema::{SCHEMA_VERSION, THREADS as COLLECTION, append_spec, stream_name, thread_spec};
use super::*;
use async_trait::async_trait;
use std::sync::Arc;
use tinystoragedrivers_core::primitives::Repo;
use tinystoragedrivers_core::{
    Capability, CollectionSpec, DocumentStore, ErrorKind, Filter, Key, Precondition, Query,
    ScopedStorage, StorageError, StreamStore, Versioned, WriteOptions,
};
mod append;
mod pending;
mod reader;
mod rewrite;
pub(super) mod search;

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

    /// A process-lifetime store over the memory driver, for library hosts that
    /// have no storage backend. Nothing is written to disk and nothing survives
    /// the process; plain append is fully prepared.
    ///
    /// # Panics
    /// Never in practice: the memory driver accepts every declaration.
    pub async fn in_memory() -> Self {
        use tinystoragedrivers_core::{MemoryStorage, Scope, StorageBackend};
        let backend = MemoryStorage::new();
        let scoped = backend
            .for_scope(&Scope::new("local").expect("static scope"))
            .expect("memory scope");
        Self::prepare_append_schema(scoped.documents().as_ref())
            .await
            .expect("memory driver accepts declarations");
        Self::new(scoped)
    }

    /// Thread metadata and the base derived-search collection declaration.
    /// Trusted bootstrap adds its FullText field when the driver supports it.
    /// Native validation accepts compatible supersets, not exact equality.
    pub fn collection_specs() -> Vec<CollectionSpec> {
        vec![thread_spec(), search::collection_spec()]
    }

    pub(super) fn search_collection_spec() -> CollectionSpec {
        search::collection_spec()
    }

    pub(super) fn search_schema_version() -> u32 {
        search::spec_version()
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
    ) -> tinystoragedrivers_core::Result<()> {
        for spec in Self::collection_specs() {
            documents.ensure_collection(&spec).await?;
        }
        Self::prepare_search_schema(documents).await?;
        Ok(())
    }

    async fn admit(&self) -> ConversationResult<()> {
        if self.validate_only {
            for spec in Self::collection_specs() {
                self.docs.validate_collection_spec(&spec).await?;
            }
            if self.docs.capabilities().contains(Capability::FullText) {
                self.docs
                    .validate_collection_spec(&search::full_text_collection_spec())
                    .await?;
            }
        }
        Ok(())
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
        let mut deleted_incarnation = None;
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
                deleted_incarnation = Some(doc.incarnation.clone());
                doc.deleted = true;
                deleted = true;
                Ok(Some(doc.to_dto()?))
            })
            .await?;
        if let Some(incarnation) = deleted_incarnation.filter(|_| deleted) {
            self.remove_search_incarnation(thread_id, &incarnation)
                .await;
        }
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
        if self.docs.capabilities().contains(Capability::FullText) {
            return self.search_indexed(query, limit, exclude_thread_id).await;
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
        self.admit().await?;
        self.purge_native().await
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
#[path = "driver_schema_tests.rs"]
mod schema_tests;

#[cfg(test)]
#[path = "driver_append_tests.rs"]
mod append_tests;
