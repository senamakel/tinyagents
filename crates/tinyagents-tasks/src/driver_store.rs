//! A [`TaskStore`] over `tinystoragedrivers` document ports.
//!
//! [`DriverTaskStore`] keeps orchestration task records and their per-task
//! lifecycle history on whatever backend the host opened (SQLite on a desktop,
//! MongoDB in the cloud, memory in tests). The document handle is already bound
//! to a tenant scope, so two tenants' tasks never meet even when they share a
//! task id.
//!
//! # Layout
//!
//! Two collections of the `tasks` owner (see [`storage_schema`](crate::storage_schema)):
//!
//! - `tasks.records`: the latest record per task under `Key::of([task_id])`, a
//!   bare snake_case body (`task_id`, `status`, `kind_label`, the lineage ids,
//!   `rev`, the result and `*_ms` times). The scalar fields are indexed for
//!   [`TaskStore::list`]; `rev` counts transitions. The last-update time is the
//!   document's `updated_ms` metadata.
//! - `tasks.history`: one immutable snapshot per transition under
//!   `Key::of([task_id, rev])`, so [`TaskStore::history`] returns the whole
//!   timeline oldest to newest.
//!
//! # Transitions
//!
//! Every transition is one [`Repo::update`] compare-and-swap on the task's
//! record: it applies the same pure state machine
//! [`InMemoryTaskStore`](crate::InMemoryTaskStore) uses, and a concurrent writer
//! makes the swap re-run against the winner's record, so two executors racing to
//! settle one task cannot both succeed. Only the winner then writes the
//! snapshot for the revision it committed. The snapshot follows the record, not
//! the same commit, so a crash between the two leaves the history one snapshot
//! short while the record stays correct.
//!
//! # Sync trait
//!
//! [`TaskStore`] is synchronous, so each call runs its port futures on a
//! [`Blocking`] bridge thread; this works from any caller, inside a tokio
//! runtime or not.

use std::sync::{Arc, Mutex};

use tinyagents_harness::error::{Result, TinyAgentsError};
use tinyagents_harness::ids::TaskId;
use tinyagents_storage_kit::{
    Blocking, DocumentStore, DocumentStoreExt, ErrorKind, Filter, Query, Repo, Sort, StorageError,
    Versioned, WriteOptions,
};
use tokio::sync::OnceCell;

use crate::storage::{
    HISTORY, RECORDS, SCHEMA_VERSION, TaskDoc, history_id_key, history_spec, records_spec,
    task_id_key,
};
use crate::store::{control_outcome, orchestration_not_found, transition};
use crate::types::{
    OrchestrationControlOutcome, OrchestrationTaskFilter, OrchestrationTaskRecord,
    OrchestrationTaskResult, OrchestrationTaskSpec,
};

fn map_error(error: StorageError) -> TinyAgentsError {
    TinyAgentsError::Storage(format!("task store: {error}"))
}

fn decode(row: Versioned<serde_json::Value>) -> Result<OrchestrationTaskRecord> {
    let updated_ms = row.meta.updated_ms;
    serde_json::from_value::<TaskDoc>(row.doc)
        .map(|doc| doc.into_record(updated_ms))
        .map_err(|e| TinyAgentsError::Storage(format!("task store: decode record: {e}")))
}

/// A durable [`TaskStore`](crate::TaskStore) stored in a driver [`DocumentStore`].
#[derive(Debug, Clone)]
pub struct DriverTaskStore {
    docs: Arc<dyn DocumentStore>,
    bridge: Blocking,
    records: Repo<TaskDoc>,
    history: Repo<TaskDoc>,
    declared: Arc<OnceCell<()>>,
}

impl DriverTaskStore {
    /// Store tasks in the `tasks.*` collections, with its own bridge.
    ///
    /// # Errors
    ///
    /// When the bridge's runtime thread cannot start.
    pub fn new(docs: Arc<dyn DocumentStore>) -> std::result::Result<Self, StorageError> {
        Ok(Self::with_bridge(docs, Blocking::new()?))
    }

    /// Store tasks in the `tasks.*` collections, running the synchronous trait
    /// on `bridge` so several stores share one thread.
    ///
    /// # Panics
    ///
    /// Never: the collection names are valid constants.
    #[allow(clippy::expect_used, reason = "the collection names are constants")]
    pub fn with_bridge(docs: Arc<dyn DocumentStore>, bridge: Blocking) -> Self {
        let options = WriteOptions::new().schema(SCHEMA_VERSION);
        Self {
            records: Repo::new(&docs, RECORDS)
                .expect("tasks.records is a valid collection")
                .with_options(options),
            history: Repo::new(&docs, HISTORY)
                .expect("tasks.history is a valid collection")
                .with_options(options),
            docs,
            bridge,
            declared: Arc::new(OnceCell::new()),
        }
    }

    /// Declare the collections now, so a backend that cannot hold them is
    /// found at open time rather than at the first task.
    ///
    /// # Errors
    ///
    /// The backend's refusal, or the bridge failing.
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
            .map_err(|e| TinyAgentsError::Storage(format!("task store bridge: {e}")))?
    }

    async fn declare(&self) -> Result<()> {
        self.declared
            .get_or_try_init(|| async {
                for spec in [records_spec(), history_spec()] {
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

    /// Write the snapshot of `record` at `rev`. A snapshot that already exists
    /// is kept: revisions are immutable.
    async fn snapshot(&self, record: &OrchestrationTaskRecord, rev: u64) -> Result<()> {
        let id = history_id_key(record.spec.task_id.as_str(), rev).map_err(map_error)?;
        match self
            .history
            .insert(&id, &TaskDoc::from_record(record, rev))
            .await
        {
            Ok(_) => Ok(()),
            Err(error)
                if matches!(error.kind(), ErrorKind::Conflict | ErrorKind::AlreadyExists) =>
            {
                Ok(())
            }
            Err(error) => Err(map_error(error)),
        }
    }

    /// Read `task_id`, apply `change`, and commit the result with one
    /// compare-and-swap, then record its history snapshot.
    fn modify<R: Send + 'static>(
        &self,
        task_id: &TaskId,
        change: impl Fn(&mut OrchestrationTaskRecord) -> Result<R> + Send + Sync + 'static,
    ) -> Result<R> {
        let this = self.clone();
        let task_id = task_id.clone();
        self.run(async move {
            this.declare().await?;
            let id = task_id_key(task_id.as_str()).map_err(map_error)?;
            // What the closure decided, kept outside it: `update` may run it
            // again after a lost swap, and only the last run's outcome stands.
            let decided: Mutex<Option<Result<(R, OrchestrationTaskRecord)>>> = Mutex::new(None);
            let stored = this
                .records
                .update(&id, |current| {
                    let outcome = match current {
                        None => Err(orchestration_not_found(&task_id)),
                        Some(doc) => {
                            let rev = doc.rev;
                            // The update time is metadata, which `update` does not
                            // hand over; every transition stamps its own.
                            let mut record = doc.into_record(0);
                            change(&mut record).map(|outcome| (outcome, record, rev))
                        }
                    };
                    match outcome {
                        Ok((outcome, record, rev)) => {
                            let doc = TaskDoc::from_record(&record, rev + 1);
                            *decided
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner) =
                                Some(Ok((outcome, record)));
                            Ok(Some(doc))
                        }
                        Err(error) => {
                            *decided
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner) =
                                Some(Err(error));
                            Ok(None)
                        }
                    }
                })
                .await
                .map_err(map_error)?;
            let decided = decided
                .into_inner()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .ok_or_else(|| TinyAgentsError::Storage("task store: no transition ran".into()))?;
            let (outcome, record) = decided?;
            let (doc, _) = stored.ok_or_else(|| {
                TinyAgentsError::Storage("task store: transition was not stored".into())
            })?;
            this.snapshot(&record, doc.rev).await?;
            Ok(outcome)
        })
    }
}

impl crate::store::TaskStore for DriverTaskStore {
    fn insert(&self, spec: OrchestrationTaskSpec) -> Result<OrchestrationTaskRecord> {
        let this = self.clone();
        self.run(async move {
            this.declare().await?;
            let record = OrchestrationTaskRecord::pending(spec);
            let id = task_id_key(record.spec.task_id.as_str()).map_err(map_error)?;
            match this
                .records
                .insert(&id, &TaskDoc::from_record(&record, 1))
                .await
            {
                Ok(_) => {}
                Err(error)
                    if matches!(error.kind(), ErrorKind::Conflict | ErrorKind::AlreadyExists) =>
                {
                    return Err(TinyAgentsError::DuplicateComponent(format!(
                        "orchestration task `{}`",
                        record.spec.task_id
                    )));
                }
                Err(error) => return Err(map_error(error)),
            }
            this.snapshot(&record, 1).await?;
            Ok(record)
        })
    }

    fn get(&self, task_id: &TaskId) -> Option<OrchestrationTaskRecord> {
        let this = self.clone();
        let task_id = task_id.clone();
        self.run(async move {
            this.declare().await?;
            let id = task_id_key(task_id.as_str()).map_err(map_error)?;
            let found = this.docs.get(RECORDS, &id).await.map_err(map_error)?;
            found.map(decode).transpose()
        })
        .inspect_err(|error| tracing::warn!("[orchestration] task store read failed: {error}"))
        .ok()
        .flatten()
    }

    fn list(&self, filter: OrchestrationTaskFilter) -> Vec<OrchestrationTaskRecord> {
        let this = self.clone();
        self.run(async move {
            this.declare().await?;
            let mut query_filter = Filter::default();
            for (field, value) in [
                (
                    "root_run_id",
                    filter.root_run_id.as_ref().map(|id| id.as_str()),
                ),
                (
                    "parent_run_id",
                    filter.parent_run_id.as_ref().map(|id| id.as_str()),
                ),
                ("thread_id", filter.thread_id.as_ref().map(|id| id.as_str())),
                ("node_id", filter.node_id.as_ref().map(|id| id.as_str())),
                ("kind_label", filter.kind.as_deref()),
            ] {
                if let Some(value) = value {
                    query_filter = query_filter.and(Filter::eq(field, value));
                }
            }
            if let Some(status) = filter.status {
                query_filter =
                    query_filter.and(Filter::eq("status", serde_json::to_value(status)?));
            }
            let rows = this
                .docs
                .query_all(RECORDS, &Query::filter(query_filter))
                .await
                .map_err(map_error)?;
            let mut records = Vec::with_capacity(rows.len());
            for row in rows {
                let record = decode(row)?;
                // The time window is the part the projections cannot answer.
                if filter.matches(&record) {
                    records.push(record);
                }
            }
            records.sort_by(|a, b| a.spec.task_id.as_str().cmp(b.spec.task_id.as_str()));
            Ok(records)
        })
        .inspect_err(|error| tracing::warn!("[orchestration] task store list failed: {error}"))
        .unwrap_or_default()
    }

    fn history(&self, task_id: &TaskId) -> Vec<OrchestrationTaskRecord> {
        let this = self.clone();
        let task_id = task_id.as_str().to_owned();
        self.run(async move {
            this.declare().await?;
            let rows = this
                .docs
                .query_all(
                    HISTORY,
                    &Query::filter(Filter::eq("task_id", task_id)).sort(Sort::asc("rev")),
                )
                .await
                .map_err(map_error)?;
            rows.into_iter().map(decode).collect()
        })
        .inspect_err(|error| tracing::warn!("[orchestration] task history read failed: {error}"))
        .unwrap_or_default()
    }

    fn mark_running(&self, task_id: &TaskId) -> Result<OrchestrationTaskRecord> {
        self.modify(task_id, |record| {
            transition::mark_running(record)?;
            Ok(record.clone())
        })
    }

    fn mark_awaiting(&self, task_id: &TaskId) -> Result<OrchestrationTaskRecord> {
        self.modify(task_id, |record| {
            transition::mark_awaiting(record)?;
            Ok(record.clone())
        })
    }

    fn mark_awaiting_with_question(
        &self,
        task_id: &TaskId,
        question: String,
    ) -> Result<OrchestrationTaskRecord> {
        self.modify(task_id, move |record| {
            transition::mark_awaiting_with_question(record, question.clone())?;
            Ok(record.clone())
        })
    }

    fn complete(
        &self,
        task_id: &TaskId,
        result: OrchestrationTaskResult,
    ) -> Result<OrchestrationTaskRecord> {
        self.modify(task_id, move |record| {
            transition::complete(record, result.clone())?;
            Ok(record.clone())
        })
    }

    fn fail(&self, task_id: &TaskId, error: String) -> Result<OrchestrationTaskRecord> {
        self.modify(task_id, move |record| {
            transition::fail(record, error.clone())?;
            Ok(record.clone())
        })
    }

    fn timeout(&self, task_id: &TaskId, error: String) -> Result<OrchestrationTaskRecord> {
        self.modify(task_id, move |record| {
            transition::timeout(record, error.clone())?;
            Ok(record.clone())
        })
    }

    fn request_cancel(&self, task_id: &TaskId) -> Result<OrchestrationControlOutcome> {
        self.modify(task_id, |record| {
            let message = transition::request_cancel(record)?;
            Ok(control_outcome(record, message))
        })
    }

    fn mark_cancelled(&self, task_id: &TaskId) -> Result<OrchestrationTaskRecord> {
        self.modify(task_id, |record| {
            transition::mark_cancelled(record)?;
            Ok(record.clone())
        })
    }

    fn kill(&self, task_id: &TaskId) -> Result<OrchestrationControlOutcome> {
        self.modify(task_id, |record| {
            let message = transition::kill(record)?;
            Ok(control_outcome(record, message))
        })
    }

    fn set_timeout_ms(&self, task_id: &TaskId, timeout_ms: u64) -> Result<OrchestrationTaskRecord> {
        self.modify(task_id, move |record| {
            transition::set_timeout_ms(record, timeout_ms)?;
            Ok(record.clone())
        })
    }
}

#[cfg(test)]
#[path = "driver_store_tests.rs"]
mod tests;
