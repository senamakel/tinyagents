//! Durable full-text candidates and safe one-time backfill for conversations.

use super::*;
use crate::threads::inverted_index::query_terms;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use tinystoragedrivers_core::{Capability, Filter, WriteOp};

pub(super) const SEARCH_COLLECTION: &str = "conversations.search";
const SEARCH_READY: &str = "search-index-ready-v1";
const SEARCH_SCHEMA_VERSION: u32 = 1;

/// Indexed message data. `terms` contains portable opaque tokens, so the
/// storage drivers' different full-text parsers see the same searchable words.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SearchEntry {
    thread_id: String,
    incarnation: String,
    offset: u64,
    message_id: String,
    content: String,
    sender: String,
    created_at: String,
    created_at_ms: i64,
    terms: String,
}

impl SearchEntry {
    fn new(thread_id: &str, incarnation: &str, offset: u64, message: &ThreadMessage) -> Self {
        Self {
            thread_id: thread_id.to_owned(),
            incarnation: incarnation.to_owned(),
            offset,
            message_id: message.id.clone(),
            content: message.content.clone(),
            sender: message.sender.clone(),
            created_at: message.created_at.clone(),
            created_at_ms: chrono::DateTime::parse_from_rfc3339(&message.created_at)
                .map(|value| value.timestamp_millis())
                .unwrap_or(i64::MIN),
            terms: indexed_terms(&message.content),
        }
    }

    fn into_message(self) -> ThreadMessage {
        ThreadMessage {
            id: self.message_id,
            content: self.content,
            message_type: "text".into(),
            extra_metadata: serde_json::Value::Null,
            sender: self.sender,
            created_at: self.created_at,
        }
    }
}

/// Stable id for one immutable message in one thread incarnation.
fn entry_id(thread_id: &str, incarnation: &str, offset: u64) -> String {
    let mut digest = Sha256::new();
    digest.update(thread_id.as_bytes());
    digest.update([0]);
    digest.update(incarnation.as_bytes());
    digest.update([0]);
    digest.update(offset.to_be_bytes());
    hex::encode(digest.finalize())
}

/// Terms use only lowercase ASCII letters and digits. This avoids FTS syntax,
/// punctuation and CJK tokenization differences between SQLite, MongoDB, file
/// and memory drivers. Unigrams preserve the legacy short-term fallback.
fn indexed_terms(content: &str) -> String {
    let normalized = super::super::tokenize::normalize(content);
    let mut features = BTreeSet::new();
    for gram in super::super::tokenize::ngrams(&normalized) {
        features.insert(format!("g{}", hex::encode(gram.as_bytes())));
    }
    for character in normalized.chars() {
        features.insert(format!("c{:x}", character as u32));
    }
    features.into_iter().collect::<Vec<_>>().join(" ")
}

fn query_features(query: &str) -> Vec<String> {
    let mut features = BTreeSet::new();
    for term in query_terms(query) {
        let grams: Vec<_> = super::super::tokenize::ngrams(&term).into_iter().collect();
        if grams.is_empty() {
            for character in term.chars() {
                features.insert(format!("c{:x}", character as u32));
            }
        } else {
            for gram in grams {
                features.insert(format!("g{}", hex::encode(gram.as_bytes())));
            }
        }
    }
    features.into_iter().collect()
}

/// The base declaration registered with the storage schema. Trusted bootstrap
/// adds the full-text field only when the selected driver supports it.
pub(super) fn collection_spec() -> CollectionSpec {
    CollectionSpec::new(SEARCH_COLLECTION)
        .index(tinystoragedrivers_core::IndexSpec::new(
            "thread_incarnation",
            ["thread_id", "incarnation"],
        ))
        .index(tinystoragedrivers_core::IndexSpec::new(
            "recency",
            ["created_at_ms", "thread_id", "offset"],
        ))
}

fn full_text_spec() -> CollectionSpec {
    collection_spec().searchable(["terms"])
}

pub(super) fn spec_version() -> u32 {
    SEARCH_SCHEMA_VERSION
}

fn write_op(
    thread_id: &str,
    incarnation: &str,
    offset: u64,
    message: &ThreadMessage,
) -> Result<WriteOp, StorageError> {
    let entry = SearchEntry::new(thread_id, incarnation, offset, message);
    Ok(WriteOp::Put {
        collection: SEARCH_COLLECTION.into(),
        id: entry_id(thread_id, incarnation, offset),
        doc: serde_json::to_value(entry).map_err(|error| {
            StorageError::serialization("encode conversation search entry").with_source(error)
        })?,
        precondition: Precondition::Absent,
    })
}

impl DriverConversationStore {
    pub(super) async fn prepare_search_schema(
        documents: &dyn DocumentStore,
    ) -> Result<(), StorageError> {
        let spec = if documents.capabilities().contains(Capability::FullText) {
            full_text_spec()
        } else {
            collection_spec()
        };
        documents.ensure_collection(&spec).await
    }

    pub(super) async fn backfill_search_index(&self) -> ConversationResult<()> {
        if self
            .docs
            .get(SEARCH_COLLECTION, SEARCH_READY)
            .await?
            .is_some()
        {
            return Ok(());
        }
        for row in self.live_rows().await? {
            let thread_id = row.doc.thread.id.as_str();
            let mut stable_messages = None;
            let mut stable_incarnation = String::new();
            for _ in 0..RETRIES {
                let Some(current) = self.load(thread_id).await?.filter(|row| !row.doc.deleted)
                else {
                    stable_messages = Some(Vec::new());
                    break;
                };
                let messages = self.messages(thread_id).await?;
                let Some(revalidated) = self.load(thread_id).await? else {
                    stable_messages = Some(Vec::new());
                    break;
                };
                if revalidated.doc.deleted {
                    stable_messages = Some(Vec::new());
                    break;
                }
                if revalidated.version != current.version
                    || revalidated.doc.incarnation != current.doc.incarnation
                {
                    continue;
                }
                stable_incarnation = current.doc.incarnation;
                stable_messages = Some(messages);
                break;
            }
            let messages = stable_messages.ok_or_else(|| {
                StorageError::conflict(
                    "conversation changed repeatedly during search-index backfill",
                )
            })?;
            for (offset, message) in messages.iter().enumerate() {
                let id = entry_id(thread_id, &stable_incarnation, offset as u64);
                let entry =
                    SearchEntry::new(thread_id, &stable_incarnation, offset as u64, message);
                let body = serde_json::to_value(entry).map_err(|error| {
                    StorageError::serialization("encode conversation search entry")
                        .with_source(error)
                })?;
                self.docs
                    .put(SEARCH_COLLECTION, &id, body, Precondition::None)
                    .await?;
            }
            let indexed_incarnation = if stable_incarnation.is_empty() {
                row.doc.incarnation.as_str()
            } else {
                stable_incarnation.as_str()
            };
            let still_live = self.load(thread_id).await?.is_some_and(|current| {
                !current.doc.deleted && current.doc.incarnation == indexed_incarnation
            });
            if !still_live {
                self.remove_search_incarnation(thread_id, indexed_incarnation)
                    .await;
            }
        }
        self.docs
            .put(
                SEARCH_COLLECTION,
                SEARCH_READY,
                serde_json::json!({ "version": SEARCH_SCHEMA_VERSION, "ready": true }),
                Precondition::None,
            )
            .await?;
        Ok(())
    }

    pub(super) async fn index_rewritten_messages(
        &self,
        thread_id: &str,
        incarnation: &str,
        messages: &[ThreadMessage],
    ) -> ConversationResult<()> {
        if !self.docs.capabilities().contains(Capability::FullText) {
            return Ok(());
        }
        for (offset, message) in messages.iter().enumerate() {
            let id = entry_id(thread_id, incarnation, offset as u64);
            let entry = SearchEntry::new(thread_id, incarnation, offset as u64, message);
            let body = serde_json::to_value(entry).map_err(|error| {
                StorageError::serialization("encode conversation search entry").with_source(error)
            })?;
            self.docs
                .put(SEARCH_COLLECTION, &id, body, Precondition::Absent)
                .await?;
        }
        Ok(())
    }

    pub(super) async fn remove_search_incarnation(&self, thread_id: &str, incarnation: &str) {
        if !self.docs.capabilities().contains(Capability::FullText) {
            return;
        }
        let filter = Filter::eq("thread_id", thread_id).and(Filter::eq("incarnation", incarnation));
        if let Err(error) = self.docs.delete_where(SEARCH_COLLECTION, &filter).await {
            tracing::debug!(target: "tinyagents_session::threads", %error, "conversation search entries not removed");
        }
    }

    pub(super) async fn search_indexed(
        &self,
        query: &str,
        limit: usize,
        exclude_thread_id: Option<&str>,
    ) -> ConversationResult<Vec<CrossThreadHit>> {
        let features = query_features(query);
        if features.is_empty() {
            return Ok(Vec::new());
        }
        self.backfill_search_index().await?;
        for _ in 0..RETRIES {
            let candidate_hits = self
                .docs
                .search(SEARCH_COLLECTION, &features.join(" "), usize::MAX)
                .await?;
            let mut cache: HashMap<String, Option<Versioned<ThreadDoc>>> = HashMap::new();
            let mut entries = Vec::new();
            for hit in candidate_hits {
                if hit.id == SEARCH_READY {
                    continue;
                }
                let Some(row) = self.docs.get(SEARCH_COLLECTION, &hit.id).await? else {
                    continue;
                };
                let entry: SearchEntry = serde_json::from_value(row.doc).map_err(|error| {
                    StorageError::serialization("invalid conversation search entry")
                        .with_source(error)
                })?;
                let live = if let Some(cached) = cache.get(&entry.thread_id) {
                    cached.clone()
                } else {
                    let current = self
                        .load(&entry.thread_id)
                        .await?
                        .filter(|current| !current.doc.deleted);
                    cache.insert(entry.thread_id.clone(), current.clone());
                    current
                };
                let Some(live) = live else {
                    continue;
                };
                if entry.incarnation != live.doc.incarnation || entry.offset >= live.doc.visible_end
                {
                    continue;
                }
                entries.push(entry);
            }
            entries.sort_by(|a, b| a.thread_id.cmp(&b.thread_id).then(a.offset.cmp(&b.offset)));
            let mut index = inverted_index::InvertedIndex::new();
            for entry in entries {
                index.insert(&entry.thread_id, entry.into_message());
            }
            let mut changed = false;
            for (thread_id, snapshot) in &cache {
                let current = self
                    .load(thread_id)
                    .await?
                    .filter(|current| !current.doc.deleted);
                if current.as_ref().map(|row| row.version)
                    != snapshot.as_ref().map(|row| row.version)
                {
                    changed = true;
                    break;
                }
            }
            if !changed {
                return match index.search_outcome(query, limit, exclude_thread_id) {
                    inverted_index::SearchOutcome::Hits(hits) => Ok(hits),
                    inverted_index::SearchOutcome::RecencyFallback => {
                        self.search_recency_fallback(limit, exclude_thread_id).await
                    }
                };
            }
        }
        Err(StorageError::conflict("conversation changed repeatedly during indexed search").into())
    }

    async fn search_recency_fallback(
        &self,
        limit: usize,
        exclude_thread_id: Option<&str>,
    ) -> ConversationResult<Vec<CrossThreadHit>> {
        for _ in 0..RETRIES {
            let mut query = Query::filter(Filter::exists("created_at_ms", true))
                .sort(tinystoragedrivers_core::Sort::desc("created_at_ms"))
                .sort(tinystoragedrivers_core::Sort::asc("thread_id"))
                .sort(tinystoragedrivers_core::Sort::asc("offset"))
                .limit(PAGE_SIZE);
            let mut cache: HashMap<String, Option<Versioned<ThreadDoc>>> = HashMap::new();
            let mut hits = Vec::new();
            loop {
                let page = self.docs.query(SEARCH_COLLECTION, &query).await?;
                for row in page.items {
                    let entry: SearchEntry = serde_json::from_value(row.doc).map_err(|error| {
                        StorageError::serialization("invalid conversation search entry")
                            .with_source(error)
                    })?;
                    if exclude_thread_id == Some(entry.thread_id.as_str()) {
                        continue;
                    }
                    let live = if let Some(cached) = cache.get(&entry.thread_id) {
                        cached.clone()
                    } else {
                        let current = self
                            .load(&entry.thread_id)
                            .await?
                            .filter(|current| !current.doc.deleted);
                        cache.insert(entry.thread_id.clone(), current.clone());
                        current
                    };
                    let Some(live) = live else {
                        continue;
                    };
                    if entry.incarnation != live.doc.incarnation
                        || entry.offset >= live.doc.visible_end
                    {
                        continue;
                    }
                    hits.push(CrossThreadHit {
                        thread_id: entry.thread_id,
                        message_id: entry.message_id,
                        role: entry.sender,
                        content: entry.content,
                        created_at: entry.created_at,
                        score: 0.0,
                    });
                    if hits.len() == limit {
                        break;
                    }
                }
                if hits.len() == limit {
                    break;
                }
                let Some(next) = page.next else {
                    break;
                };
                if query.cursor.as_ref() == Some(&next) {
                    return Err(StorageError::backend(
                        "nonadvancing conversation search recency cursor",
                    )
                    .into());
                }
                query.cursor = Some(next);
            }
            let mut changed = false;
            for (thread_id, snapshot) in &cache {
                let current = self
                    .load(thread_id)
                    .await?
                    .filter(|current| !current.doc.deleted);
                if current.as_ref().map(|row| row.version)
                    != snapshot.as_ref().map(|row| row.version)
                {
                    changed = true;
                    break;
                }
            }
            if !changed {
                return Ok(hits);
            }
        }
        Err(StorageError::conflict("conversation changed repeatedly during recency search").into())
    }
}

pub(super) fn append_write_op(
    thread_id: &str,
    incarnation: &str,
    offset: u64,
    message: &ThreadMessage,
) -> Result<WriteOp, StorageError> {
    write_op(thread_id, incarnation, offset, message)
}

pub(super) fn full_text_collection_spec() -> CollectionSpec {
    full_text_spec()
}
