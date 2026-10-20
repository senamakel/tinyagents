//! A [`CompletionStore`] over `tinystoragedrivers` document ports.
//!
//! [`DriverCompletionStore`] keeps the latest [`CompletionRecord`] per task on
//! whatever backend the host opened, so a completion the router reported as
//! stored survives a restart. The document handle is already bound to a tenant
//! scope.
//!
//! # Layout
//!
//! One collection, `tasks.completions` (see [`storage_schema`](crate::storage_schema)),
//! with one document per task id under `Key::of([task_id])`: a bare snake_case
//! body (`task_id`, `parent_key`, `agent_id`, `state`, `result`, `finished_ms`,
//! ...). `parent_key` and `state` are indexed for [`CompletionStore::list`] and
//! [`CompletionStore::compact`]. The record's last-update time is the document's
//! `updated_ms` metadata, which the driver stamps on every write. `put` is an
//! upsert, which is all the contract asks of it: the router serialises its
//! read-modify-write sequences.
//!
//! # Sync trait
//!
//! [`CompletionStore`] is synchronous, so each call runs its port futures on a
//! [`Blocking`] bridge thread; this works from any caller, inside a tokio
//! runtime or not.

use std::sync::Arc;
use std::time::{Duration, SystemTime};

use tinyagents_harness::error::{Result, TinyAgentsError};
use tinyagents_storage_kit::{
    Blocking, CollectionSpec, DocumentStore, DocumentStoreExt, Filter, IndexSpec, Precondition,
    Query, Repo, StorageError, UPDATED_FIELD, Versioned, WriteOptions,
};
use tokio::sync::OnceCell;

use super::store::CompletionStore;
use super::types::{CompletionRecord, CompletionState};
use crate::storage::{
    COMPLETIONS, CompletionDoc, SCHEMA_VERSION, completions_spec, task_id_key, to_ms,
};

const LOG_PREFIX: &str = "[completion-store]";

fn map_error(error: StorageError) -> TinyAgentsError {
    TinyAgentsError::Storage(format!("completion store: {error}"))
}

fn decode(row: Versioned<serde_json::Value>) -> Option<CompletionRecord> {
    let updated_ms = row.meta.updated_ms;
    match serde_json::from_value::<CompletionDoc>(row.doc) {
        Ok(doc) => Some(doc.into_record(updated_ms)),
        Err(error) => {
            // One unreadable record must not block every other completion.
            tracing::warn!(error = %error, "{LOG_PREFIX} skipping unreadable record");
            None
        }
    }
}

fn decode_strict(row: Versioned<serde_json::Value>) -> Result<CompletionRecord> {
    let updated_ms = row.meta.updated_ms;
    serde_json::from_value::<CompletionDoc>(row.doc)
        .map(|doc| doc.into_record(updated_ms))
        .map_err(|error| {
            TinyAgentsError::Storage(format!(
                "completion store contains unreadable record: {error}"
            ))
        })
}

/// A durable [`CompletionStore`] stored in a driver [`DocumentStore`].
#[derive(Debug, Clone)]
pub struct DriverCompletionStore {
    docs: Arc<dyn DocumentStore>,
    bridge: Blocking,
    repo: Repo<CompletionDoc>,
    declared: Arc<OnceCell<()>>,
}

impl DriverCompletionStore {
    /// Store completions in `tasks.completions`, with its own bridge.
    ///
    /// # Errors
    ///
    /// When the bridge's runtime thread cannot start.
    pub fn new(docs: Arc<dyn DocumentStore>) -> std::result::Result<Self, StorageError> {
        Ok(Self::with_bridge(docs, Blocking::new()?))
    }

    /// Store completions in `tasks.completions`, running the synchronous trait
    /// on `bridge` so several stores share one thread.
    ///
    /// # Panics
    ///
    /// Never: the collection name is a valid constant.
    #[allow(clippy::expect_used, reason = "the collection name is a constant")]
    pub fn with_bridge(docs: Arc<dyn DocumentStore>, bridge: Blocking) -> Self {
        Self {
            repo: Repo::new(&docs, COMPLETIONS)
                .expect("tasks.completions is a valid collection")
                .with_options(WriteOptions::new().schema(SCHEMA_VERSION)),
            docs,
            bridge,
            declared: Arc::new(OnceCell::new()),
        }
    }

    /// The declaration a store using `collection` needs.
    pub fn collection_spec(collection: impl Into<String>) -> CollectionSpec {
        CollectionSpec::new(collection)
            .index(IndexSpec::new("by_parent", ["parent_key"]))
            .index(IndexSpec::new("by_state", ["state"]))
    }

    /// Declares the backing collection eagerly so an unavailable backend is
    /// reported before the host accepts work.
    pub fn declare_collections(&self) -> Result<()> {
        let this = self.clone();
        self.run(async move { this.declare().await })
    }

    fn run<T: Send + 'static>(
        &self,
        work: impl std::future::Future<Output = Result<T>> + Send + 'static,
    ) -> Result<T> {
        self.bridge
            .run(work)
            .map_err(|e| TinyAgentsError::Storage(format!("completion store bridge: {e}")))?
    }

    async fn declare(&self) -> Result<()> {
        self.declared
            .get_or_try_init(|| async {
                self.docs
                    .ensure_collection(&completions_spec())
                    .await
                    .map_err(map_error)
            })
            .await
            .map(|_| ())
    }
}

impl CompletionStore for DriverCompletionStore {
    fn get(&self, task_id: &str) -> Option<CompletionRecord> {
        let this = self.clone();
        let task_id = task_id.to_owned();
        self.run(async move {
            this.declare().await?;
            let id = task_id_key(&task_id).map_err(map_error)?;
            let found = this.docs.get(COMPLETIONS, &id).await.map_err(map_error)?;
            Ok(found.and_then(decode))
        })
        .inspect_err(|error| tracing::warn!(error = %error, "{LOG_PREFIX} read failed"))
        .ok()
        .flatten()
    }

    fn put(&self, record: &CompletionRecord) -> Result<()> {
        let this = self.clone();
        let record = record.clone();
        self.run(async move {
            this.declare().await?;
            let id = task_id_key(&record.task_id).map_err(map_error)?;
            this.repo
                .put(&id, &CompletionDoc::from_record(&record))
                .await
                .map(|_| ())
                .map_err(map_error)
        })
    }

    fn list(&self, parent_key: Option<&str>) -> Vec<CompletionRecord> {
        let this = self.clone();
        let parent_key = parent_key.map(str::to_owned);
        self.run(async move {
            this.declare().await?;
            let filter = match &parent_key {
                Some(parent) => Filter::eq("parent_key", parent.as_str()),
                None => Filter::default(),
            };
            let rows = this
                .docs
                .query_all(COMPLETIONS, &Query::filter(filter))
                .await
                .map_err(map_error)?;
            Ok(rows.into_iter().filter_map(decode).collect())
        })
        .inspect_err(|error| tracing::warn!(error = %error, "{LOG_PREFIX} list failed"))
        .unwrap_or_default()
    }

    fn try_list(&self, parent_key: Option<&str>) -> Result<Vec<CompletionRecord>> {
        let this = self.clone();
        let parent_key = parent_key.map(str::to_owned);
        self.run(async move {
            this.declare().await?;
            let filter = match &parent_key {
                Some(parent) => Filter::eq("parent_key", parent.as_str()),
                None => Filter::default(),
            };
            let rows = this
                .docs
                .query_all(COMPLETIONS, &Query::filter(filter))
                .await
                .map_err(map_error)?;
            rows.into_iter().map(decode_strict).collect()
        })
    }

    fn compact(&self, retain: Duration) -> Result<usize> {
        let this = self.clone();
        let cutoff = to_ms(SystemTime::now())
            .saturating_sub(i64::try_from(retain.as_millis()).unwrap_or(i64::MAX));
        self.run(async move {
            this.declare().await?;
            // Settled means anything but pending, last written more than
            // `retain` ago: the document's own `updated_ms`.
            let filter = Filter::ne("state", serde_json::to_value(CompletionState::Pending)?)
                .and(Filter::lt(UPDATED_FIELD, cutoff));
            let rows = this
                .docs
                .query_all(COMPLETIONS, &Query::filter(filter))
                .await
                .map_err(map_error)?;
            let mut dropped = 0;
            for row in rows {
                // Only the version we saw: a record refreshed since is live again.
                match this
                    .docs
                    .delete(COMPLETIONS, &row.id, Precondition::Version(row.version))
                    .await
                {
                    Ok(removed) => dropped += usize::from(removed),
                    Err(error) if error.kind() == tinyagents_storage_kit::ErrorKind::Conflict => {}
                    Err(error) => return Err(map_error(error)),
                }
            }
            tracing::debug!(dropped, "{LOG_PREFIX} compacted");
            Ok(dropped)
        })
    }
}

#[cfg(test)]
#[path = "driver_tests.rs"]
mod tests;
