//! A [`Checkpointer`] over `tinystoragedrivers` document ports.
//!
//! [`DriverCheckpointer`] puts graph checkpoints, their pending writes and the
//! per-thread execution lease on whatever backend the host opened (SQLite on a
//! desktop, MongoDB in the cloud, memory in tests). The document handle is
//! already bound to a scope, so two tenants' threads never meet even when they
//! share a thread id.
//!
//! # Layout
//!
//! There is one checkpointer: every user of it shares the same three
//! collections and tells itself apart by a namespace (`ns`, default `graph`;
//! the delegation graph uses `delegation` and workflows `flows`). The namespace
//! is part of every id and index, so two namespaces never meet.
//!
//! - `graph.checkpoints`: one document per stored checkpoint,
//!   `{ns, thread, subgraph, seq, checkpoint_id, checkpoint}` under
//!   `Key::of([ns, thread, seq])`. `subgraph` is the encoded subgraph
//!   namespace, so scoped reads filter on it; `checkpoint` is the checkpoint
//!   itself. `seq` is a per-thread insertion counter, so listing is a query
//!   sorted on it and duplicate checkpoint ids resolve to the latest write,
//!   exactly like the append-only backends.
//! - `graph.threads`: one counter document per thread, `{ns, thread, next}`,
//!   advanced with [`Repo::update`] so concurrent writers never share a `seq`.
//! - `graph.writes`: one document per `(thread, subgraph, checkpoint)` holding
//!   the merged pending writes (see [`merge_writes`]).
//! - Leases live in `tsd.leases` under owner `graph` and key
//!   `Key::of([ns, thread])`, claimed through [`Leases`]; the lease's epoch is
//!   the write fence (see [`DriverCheckpointer::lease_key`]).

use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tinyagents_harness::error::{Result, TinyAgentsError};
use tinyagents_storage_kit::{
    DocumentStore, DocumentStoreExt, ErrorKind, Filter, Leases, Query, Repo, Sort, StorageError,
    Versioned, WriteOptions,
};
use tokio::sync::OnceCell;

use super::{
    Checkpoint, CheckpointConfig, CheckpointId, CheckpointMetadata, Checkpointer, PendingWrite,
    decode_json_err, merge_writes, require_checkpoint_id,
};
use crate::storage::{
    CHECKPOINTS, CheckpointBody, DEFAULT_NAMESPACE, LEASE_OWNER, SCHEMA_VERSION, THREADS,
    ThreadBody, WRITES, WritesBody, checkpoint_id_key, checkpoints_spec, lease_key, leases_spec,
    subgraph_key, thread_id_key, threads_spec, writes_id_key, writes_spec,
};

/// Map a storage driver failure onto the graph's checkpoint error.
fn map_error(error: StorageError) -> TinyAgentsError {
    TinyAgentsError::Checkpoint(format!("storage driver: {error}"))
}

/// A lease ttl in whole milliseconds; a lease must outlive its own claim, so
/// the shortest is one.
fn ttl_ms(ttl: Duration) -> i64 {
    i64::try_from(ttl.as_millis()).unwrap_or(i64::MAX).max(1)
}

/// A [`Checkpointer`] that stores everything in a driver [`DocumentStore`].
pub struct DriverCheckpointer<State> {
    docs: Arc<dyn DocumentStore>,
    ns: String,
    checkpoints: Repo<CheckpointBody>,
    threads: Repo<ThreadBody>,
    writes: Repo<WritesBody>,
    leases: Leases,
    declared: Arc<OnceCell<()>>,
    _state: PhantomData<fn() -> State>,
}

impl<State> Clone for DriverCheckpointer<State> {
    fn clone(&self) -> Self {
        Self {
            docs: Arc::clone(&self.docs),
            ns: self.ns.clone(),
            checkpoints: self.checkpoints.clone(),
            threads: self.threads.clone(),
            writes: self.writes.clone(),
            leases: self.leases.clone(),
            declared: Arc::clone(&self.declared),
            _state: PhantomData,
        }
    }
}

impl<State> std::fmt::Debug for DriverCheckpointer<State> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DriverCheckpointer")
            .field("ns", &self.ns)
            .finish_non_exhaustive()
    }
}

impl<State> DriverCheckpointer<State> {
    /// Store checkpoints in the `graph.*` collections of `docs`, under the
    /// namespace [`DEFAULT_NAMESPACE`] (`graph`).
    pub fn new(docs: Arc<dyn DocumentStore>) -> Self {
        Self::with_namespace(docs, DEFAULT_NAMESPACE)
    }

    /// Store checkpoints in the `graph.*` collections of `docs` under the
    /// namespace `ns`, so several independent users of the checkpointer (the
    /// delegation graph as `delegation`, workflows as `flows`) can share one
    /// backend without meeting.
    ///
    /// # Panics
    ///
    /// Never: the collection names are valid constants.
    #[allow(clippy::expect_used, reason = "the collection names are constants")]
    pub fn with_namespace(docs: Arc<dyn DocumentStore>, ns: impl Into<String>) -> Self {
        let options = WriteOptions::new().schema(SCHEMA_VERSION);
        fn repo<T>(docs: &Arc<dyn DocumentStore>, name: &str, options: WriteOptions) -> Repo<T> {
            Repo::new(docs, name)
                .expect("graph collections are valid names")
                .with_options(options)
        }
        Self {
            checkpoints: repo(&docs, CHECKPOINTS, options),
            threads: repo(&docs, THREADS, options),
            writes: repo(&docs, WRITES, options),
            leases: Leases::new(&docs),
            docs,
            ns: ns.into(),
            declared: Arc::new(OnceCell::new()),
            _state: PhantomData,
        }
    }

    /// This checkpointer's namespace.
    pub fn namespace(&self) -> &str {
        &self.ns
    }

    /// The `tsd.leases` owner and key of `thread`'s execution lease. A host may
    /// use these identifiers with its own conditional-write policy.
    pub fn lease_key(&self, thread: &str) -> (&'static str, String) {
        (LEASE_OWNER, lease_key(&self.ns, thread))
    }

    async fn declared(&self) -> Result<()> {
        self.declared
            .get_or_try_init(|| async {
                for spec in [
                    checkpoints_spec(),
                    threads_spec(),
                    writes_spec(),
                    leases_spec(),
                ] {
                    self.docs
                        .ensure_collection(&spec)
                        .await
                        .map_err(map_error)?;
                }
                Ok::<(), TinyAgentsError>(())
            })
            .await
            .map(|_| ())
    }

    /// Reserve the next insertion sequence number for `thread`.
    async fn next_seq(&self, thread: &str) -> Result<u64> {
        let (stored, _) = self
            .threads
            .update(&thread_id_key(&self.ns, thread), |current| {
                Ok(Some(ThreadBody {
                    ns: self.ns.clone(),
                    thread: thread.to_owned(),
                    next: current.map_or(1, |counter| counter.next + 1),
                }))
            })
            .await
            .map_err(map_error)?
            .ok_or_else(|| {
                TinyAgentsError::Checkpoint(format!(
                    "could not reserve a checkpoint sequence for thread `{thread}`"
                ))
            })?;
        Ok(stored.next - 1)
    }

    /// The filter selecting `thread`'s documents in this namespace.
    fn thread_filter(&self, thread: &str) -> Filter {
        Filter::eq("ns", self.ns.as_str()).and(Filter::eq("thread", thread))
    }

    /// Every stored checkpoint document of `thread`, in insertion order.
    async fn thread_docs(&self, thread: &str) -> Result<Vec<Versioned<Value>>> {
        self.declared().await?;
        let query = Query::filter(self.thread_filter(thread)).sort(Sort::asc("seq"));
        self.docs
            .query_all(CHECKPOINTS, &query)
            .await
            .map_err(map_error)
    }

    async fn drop_writes(&self, filter: Filter) -> Result<()> {
        self.docs
            .delete_where(WRITES, &filter)
            .await
            .map(|_| ())
            .map_err(map_error)
    }

    /// The newest checkpoint document matching `filter`.
    async fn newest(&self, filter: Filter) -> Result<Option<Versioned<Value>>> {
        let query = Query::filter(filter).sort(Sort::desc("seq")).limit(1);
        let page = self
            .docs
            .query(CHECKPOINTS, &query)
            .await
            .map_err(map_error)?;
        Ok(page.items.into_iter().next())
    }
}

impl<State> DriverCheckpointer<State>
where
    State: DeserializeOwned,
{
    fn decode(stored: Versioned<Value>) -> Result<Checkpoint<State>> {
        let record = stored.doc.get("checkpoint").cloned().unwrap_or(Value::Null);
        // Through the shared classifier, so a `State` that no longer decodes
        // is tagged `[schema]` exactly as the file and SQLite backends tag it
        // (durable delegations prune such a checkpoint and start fresh).
        // (`"record"` names the whole-checkpoint decode to the classifier.)
        let mut checkpoint: Checkpoint<State> = serde_json::from_value(record)
            .map_err(|error| decode_json_err("storage driver checkpointer", "record", error))?;
        checkpoint.normalize();
        Ok(checkpoint)
    }
}

#[async_trait]
impl<State> Checkpointer<State> for DriverCheckpointer<State>
where
    State: Serialize + DeserializeOwned + Send + Sync + 'static,
{
    async fn put(&self, checkpoint: Checkpoint<State>) -> Result<CheckpointId> {
        self.declared().await?;
        let id = CheckpointId::new(checkpoint.checkpoint_id.clone());
        let seq = self.next_seq(&checkpoint.thread_id).await?;
        let body = CheckpointBody {
            ns: self.ns.clone(),
            thread: checkpoint.thread_id.clone(),
            subgraph: subgraph_key(&checkpoint.namespace),
            seq,
            checkpoint_id: checkpoint.checkpoint_id.clone(),
            checkpoint: serde_json::to_value(&checkpoint)?,
        };
        self.checkpoints
            .insert(
                &checkpoint_id_key(&self.ns, &checkpoint.thread_id, seq),
                &body,
            )
            .await
            .map_err(map_error)?;
        Ok(id)
    }

    async fn get(
        &self,
        thread_id: &str,
        checkpoint_id: Option<&str>,
    ) -> Result<Option<Checkpoint<State>>> {
        self.declared().await?;
        let mut filter = self.thread_filter(thread_id);
        if let Some(id) = checkpoint_id {
            filter = filter.and(Filter::eq("checkpoint_id", id));
        }
        self.newest(filter).await?.map(Self::decode).transpose()
    }

    /// One indexed query on the stored subgraph namespace, so a parent run and
    /// a subgraph sharing a thread never load each other's checkpoints, even
    /// when they reuse a checkpoint id.
    async fn get_scoped(
        &self,
        thread_id: &str,
        checkpoint_id: Option<&str>,
        namespace: &[String],
    ) -> Result<Option<Checkpoint<State>>> {
        self.declared().await?;
        let mut filter = self
            .thread_filter(thread_id)
            .and(Filter::eq("subgraph", subgraph_key(namespace)));
        if let Some(id) = checkpoint_id {
            filter = filter.and(Filter::eq("checkpoint_id", id));
        }
        self.newest(filter).await?.map(Self::decode).transpose()
    }

    /// Metadata only: the stored `State` is skipped rather than decoded, so
    /// listing a long thread costs its headers, not its states.
    async fn list(&self, thread_id: &str) -> Result<Vec<CheckpointMetadata>> {
        self.thread_docs(thread_id)
            .await?
            .into_iter()
            .map(|stored| {
                let record = stored.doc.get("checkpoint").cloned().unwrap_or(Value::Null);
                let mut header: Checkpoint<serde::de::IgnoredAny> = serde_json::from_value(record)
                    .map_err(|error| {
                        decode_json_err("storage driver checkpointer", "record", error)
                    })?;
                header.normalize();
                Ok(header.to_metadata())
            })
            .collect()
    }

    async fn get_thread(&self, thread_id: &str) -> Result<Vec<Checkpoint<State>>> {
        self.thread_docs(thread_id)
            .await?
            .into_iter()
            .map(Self::decode)
            .collect()
    }

    async fn list_threads(&self) -> Result<Vec<String>> {
        self.declared().await?;
        let counters = self
            .docs
            .query_all(THREADS, &Query::filter(Filter::eq("ns", self.ns.as_str())))
            .await
            .map_err(map_error)?;
        let mut threads = Vec::new();
        for counter in counters {
            let Some(thread) = counter.doc.get("thread").and_then(Value::as_str) else {
                continue;
            };
            let live = self
                .docs
                .count(CHECKPOINTS, &self.thread_filter(thread))
                .await
                .map_err(map_error)?;
            if live > 0 {
                threads.push(thread.to_owned());
            }
        }
        Ok(threads)
    }

    async fn delete_thread(&self, thread_id: &str) -> Result<()> {
        self.declared().await?;
        self.docs
            .delete_where(CHECKPOINTS, &self.thread_filter(thread_id))
            .await
            .map_err(map_error)?;
        // Keep the sequence counter: a later thread of the same name continues
        // after it, so its records never interleave with stale cursors.
        self.drop_writes(self.thread_filter(thread_id)).await
    }

    async fn delete_checkpoints(&self, thread_id: &str, ids: &[String]) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        self.declared().await?;
        let filter = self.thread_filter(thread_id).and(Filter::one_of(
            "checkpoint_id",
            ids.iter().map(String::as_str),
        ));
        let removed = self
            .docs
            .delete_where(CHECKPOINTS, &filter)
            .await
            .map_err(map_error)?;
        self.drop_writes(filter).await?;
        Ok(usize::try_from(removed).unwrap_or(usize::MAX))
    }

    async fn put_writes(&self, config: &CheckpointConfig, writes: &[PendingWrite]) -> Result<()> {
        let checkpoint_id = require_checkpoint_id(config)?;
        if writes.is_empty() {
            return Ok(());
        }
        self.declared().await?;
        let subgraph = subgraph_key(&config.namespace);
        let id = writes_id_key(&self.ns, &config.thread_id, &subgraph, &checkpoint_id);
        self.writes
            .update(&id, |current| {
                let mut stored = current.map(|body| body.writes).unwrap_or_default();
                merge_writes(&mut stored, writes);
                Ok(Some(WritesBody {
                    ns: self.ns.clone(),
                    thread: config.thread_id.clone(),
                    subgraph: subgraph.clone(),
                    checkpoint_id: checkpoint_id.clone(),
                    writes: stored,
                }))
            })
            .await
            .map(|_| ())
            .map_err(|error| {
                if error.kind() == ErrorKind::Conflict {
                    TinyAgentsError::Checkpoint(format!(
                        "could not record pending writes for thread `{}`",
                        config.thread_id
                    ))
                } else {
                    map_error(error)
                }
            })
    }

    async fn get_writes(&self, config: &CheckpointConfig) -> Result<Vec<PendingWrite>> {
        let Some(checkpoint_id) = self.resolve_write_target(config).await? else {
            return Ok(Vec::new());
        };
        self.declared().await?;
        let id = writes_id_key(
            &self.ns,
            &config.thread_id,
            &subgraph_key(&config.namespace),
            &checkpoint_id,
        );
        Ok(self
            .writes
            .get(&id)
            .await
            .map_err(map_error)?
            .map(|found| found.doc.writes)
            .unwrap_or_default())
    }

    async fn try_claim(&self, thread: &str, owner: &str, ttl: Duration) -> Result<bool> {
        self.declared().await?;
        match self
            .leases
            .acquire(
                LEASE_OWNER,
                &lease_key(&self.ns, thread),
                owner,
                ttl_ms(ttl),
            )
            .await
        {
            Ok(lease) => Ok(lease.is_some()),
            // A claim that kept losing its swap did not get the lease.
            Err(error) if error.kind() == ErrorKind::Conflict => Ok(false),
            Err(error) => Err(map_error(error)),
        }
    }

    async fn renew(&self, thread: &str, owner: &str, ttl: Duration) -> Result<bool> {
        self.declared().await?;
        let key = lease_key(&self.ns, thread);
        let Some(held) = self
            .leases
            .current(LEASE_OWNER, &key)
            .await
            .map_err(map_error)?
        else {
            return Ok(false);
        };
        if held.holder != owner {
            return Ok(false);
        }
        match self
            .leases
            .renew(LEASE_OWNER, &key, owner, held.epoch, ttl_ms(ttl))
            .await
        {
            Ok(lease) => Ok(lease.is_some()),
            Err(error) if error.kind() == ErrorKind::Conflict => Ok(false),
            Err(error) => Err(map_error(error)),
        }
    }

    async fn release(&self, thread: &str, owner: &str) -> Result<()> {
        self.declared().await?;
        let key = lease_key(&self.ns, thread);
        let Some(held) = self
            .leases
            .current(LEASE_OWNER, &key)
            .await
            .map_err(map_error)?
        else {
            return Ok(());
        };
        if held.holder != owner {
            return Ok(());
        }
        match self
            .leases
            .release(LEASE_OWNER, &key, owner, held.epoch)
            .await
        {
            Ok(_) => Ok(()),
            // Someone else reclaimed it in between; it is no longer ours.
            Err(error) if error.kind() == ErrorKind::Conflict => Ok(()),
            Err(error) => Err(map_error(error)),
        }
    }
}

#[cfg(test)]
#[path = "drivers_tests.rs"]
mod tests;
