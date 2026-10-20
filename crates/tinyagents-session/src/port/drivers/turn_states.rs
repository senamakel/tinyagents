//! [`TurnStates`] over a driver [`DocumentStore`].
//!
//! One document per turn in [`COLLECTION`], id `Key::of([thread, request])`.
//! The body is the turn's snake_case storage form (see
//! `turn_states_storage.rs`: `thread_id`, `request_id`, `lifecycle`,
//! `started_ms`, `updated_ms`, …), with the thread indexed so a thread's turns
//! are one query. Every conditional change (a write that must not clobber a
//! completed turn, a settle, an interruption sweep) is one [`Repo::update`]
//! compare-and-swap, so two processes sharing a database cannot lose each
//! other's updates.
//!
//! Semantics match [`InMemoryTurnStates`](crate::port::InMemoryTurnStates):
//! "latest" is the newest `started_at`, completed turns are kept up to
//! [`COMPLETED_RETENTION`] per thread, and a completed turn is never
//! overwritten by a conditional write.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use serde_json::Value;
use tinyagents_storage_kit::primitives::Repo;
use tinyagents_storage_kit::{Blocking, DocumentStoreExt};
use tinyagents_storage_kit::{
    CollectionSpec, ErrorKind, Filter, IndexSpec, Key, Precondition, Query, StorageError,
    Versioned, WriteOptions,
};
use tinystoragedrivers_api::DocumentStore;
use tokio::sync::OnceCell;

use super::super::TurnStates;
use super::super::helpers::{
    COMPLETED_RETENTION, completed_newest_first, is_terminal, newest_first,
};
use crate::turn_state::{TurnLifecycle, TurnState};
use storage::{decode as decode_body, encode};
use tinyagents_harness::store::SchemaAdmission;

/// The collection turn snapshots live in.
pub(super) const COLLECTION: &str = "session.turn_states";

/// The body version the collection is written at.
pub(crate) const SCHEMA_VERSION: u32 = 2;

/// The declaration shared by bootstrap and lazy turn-state initialization.
pub(super) fn collection_spec() -> CollectionSpec {
    CollectionSpec::new(COLLECTION).index(IndexSpec::new("by_thread", ["thread_id"]))
}

/// [`TurnStates`] stored in a driver [`DocumentStore`].
#[derive(Debug, Clone)]
pub struct DriverTurnStates {
    inner: Arc<Inner>,
    bridge: Blocking,
}

#[derive(Debug)]
struct Inner {
    docs: Arc<dyn DocumentStore>,
    /// Writes of [`COLLECTION`] at [`SCHEMA_VERSION`].
    repo: Repo<Value>,
    declared: OnceCell<()>,
    schema_admission: SchemaAdmission,
}

impl DriverTurnStates {
    /// Turn states in `docs`, with synchronous calls run on `bridge`.
    pub fn new(docs: Arc<dyn DocumentStore>, bridge: Blocking) -> Self {
        Self::new_with_schema_admission(docs, bridge, SchemaAdmission::Declare)
    }

    /// Turn states with an immutable schema policy and independent initialization.
    pub fn new_with_schema_admission(
        docs: Arc<dyn DocumentStore>,
        bridge: Blocking,
        schema_admission: SchemaAdmission,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                repo: Repo::new(&docs, COLLECTION)
                    .expect("the turn state collection name is valid")
                    .with_options(WriteOptions::new().schema(SCHEMA_VERSION)),
                docs,
                declared: OnceCell::new(),
                schema_admission,
            }),
            bridge,
        }
    }

    /// Runs `op` against the store on the bridge.
    fn run<T, F, Fut>(&self, op: F) -> Result<T, String>
    where
        F: FnOnce(Arc<Inner>) -> Fut,
        Fut: Future<Output = Result<T, StorageError>> + Send + 'static,
        T: Send + 'static,
    {
        self.run_checked(op).map_err(|error| error.to_string())
    }

    /// Uses the same blocking bridge without projecting native errors to strings.
    fn run_checked<T, F, Fut>(&self, op: F) -> Result<T, StorageError>
    where
        F: FnOnce(Arc<Inner>) -> Fut,
        Fut: Future<Output = Result<T, StorageError>> + Send + 'static,
        T: Send + 'static,
    {
        self.bridge.run(op(Arc::clone(&self.inner)))?
    }
    /// Recovers interrupted turns while retaining native declaration, read and write errors.
    pub(super) fn mark_all_interrupted_checked(
        &self,
        now_rfc3339: &str,
    ) -> Result<usize, StorageError> {
        let now = now_rfc3339.to_string();
        self.run_checked(|inner| async move {
            let mut count = 0;
            for stored in inner.query(Filter::All).await? {
                let turn = stored.doc;
                if is_terminal(&turn) {
                    continue;
                }
                if inner
                    .update(&turn.thread_id, &turn.request_id, |current| {
                        interrupted(current, &now)
                    })
                    .await?
                    .is_some()
                {
                    count += 1;
                }
            }
            Ok(count)
        })
    }
}

fn id(thread_id: &str, request_id: &str) -> Result<String, StorageError> {
    Key::of([thread_id, request_id]).map(|key| key.as_str().to_owned())
}

fn decode(stored: Versioned<Value>) -> Result<Versioned<TurnState>, StorageError> {
    let Versioned {
        id,
        version,
        doc,
        meta,
    } = stored;
    let turn = decode_body(&doc).map_err(|error| {
        StorageError::serialization(format!("turn state {id} is unreadable: {error}"))
    })?;
    Ok(Versioned {
        id,
        version,
        doc: turn,
        meta,
    })
}

impl Inner {
    async fn declared(&self) -> Result<(), StorageError> {
        self.declared
            .get_or_try_init(|| async {
                let spec = collection_spec();
                self.schema_admission.admit(self.docs.as_ref(), &spec).await
            })
            .await
            .map(|_| ())
    }

    async fn get_turn(
        &self,
        thread_id: &str,
        request_id: &str,
    ) -> Result<Option<Versioned<TurnState>>, StorageError> {
        self.declared().await?;
        self.docs
            .get(COLLECTION, &id(thread_id, request_id)?)
            .await?
            .map(decode)
            .transpose()
    }

    async fn query(&self, filter: Filter) -> Result<Vec<Versioned<TurnState>>, StorageError> {
        self.declared().await?;
        self.docs
            .query_all(COLLECTION, &Query::filter(filter))
            .await?
            .into_iter()
            .map(decode)
            .collect()
    }

    async fn thread(&self, thread_id: &str) -> Result<Vec<TurnState>, StorageError> {
        let mut turns: Vec<TurnState> = self
            .query(Filter::eq("thread_id", thread_id))
            .await?
            .into_iter()
            .map(|stored| stored.doc)
            .collect();
        turns.sort_by(newest_first);
        Ok(turns)
    }

    async fn write(&self, state: &TurnState) -> Result<(), StorageError> {
        self.declared().await?;
        self.repo
            .put(&id(&state.thread_id, &state.request_id)?, &encode(state)?)
            .await
            .map(drop)
    }

    /// Drops `thread_id`'s completed turns beyond the newest
    /// [`COMPLETED_RETENTION`].
    ///
    /// Each delete is conditional on the version the listing saw: a turn
    /// rewritten in the meantime is someone else's newer state and stays.
    async fn prune_completed(&self, thread_id: &str) -> Result<(), StorageError> {
        let mut completed: Vec<Versioned<TurnState>> = self
            .query(Filter::eq("thread_id", thread_id))
            .await?
            .into_iter()
            .filter(|stored| stored.doc.lifecycle == TurnLifecycle::Completed)
            .collect();
        completed.sort_by(|a, b| completed_newest_first(&a.doc, &b.doc));
        for stale in completed.iter().skip(COMPLETED_RETENTION) {
            match self
                .docs
                .delete(COLLECTION, &stale.id, stale.unchanged())
                .await
            {
                Ok(_) => {}
                Err(error) if is_race(&error) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    /// Applies `change` to one turn under compare-and-swap; returns the turn
    /// it wrote. `change` answers `None` to leave the turn alone.
    async fn update(
        &self,
        thread_id: &str,
        request_id: &str,
        change: impl Fn(&TurnState) -> Option<TurnState> + Send + Sync,
    ) -> Result<Option<TurnState>, StorageError> {
        self.declared().await?;
        let mut applied = None;
        self.repo
            .update(&id(thread_id, request_id)?, |current| {
                let Some(current) = current else {
                    return Ok(None);
                };
                let Some(next) = change(&decode_body(&current)?) else {
                    return Ok(None);
                };
                let body = encode(&next)?;
                applied = Some(next);
                Ok(Some(body))
            })
            .await?;
        Ok(applied)
    }
}

/// Whether `error` means another writer got there first.
fn is_race(error: &StorageError) -> bool {
    error.kind() == ErrorKind::Conflict
}

/// `turn` marked interrupted as of `now`.
fn interrupted(turn: &TurnState, now: &str) -> Option<TurnState> {
    if is_terminal(turn) {
        return None;
    }
    let mut next = turn.clone();
    next.lifecycle = TurnLifecycle::Interrupted;
    next.updated_at = now.to_string();
    next.active_tool = None;
    next.active_subagent = None;
    Some(next)
}

impl TurnStates for DriverTurnStates {
    fn put(&self, state: &TurnState) -> Result<(), String> {
        let state = state.clone();
        self.run(|inner| async move {
            inner.write(&state).await?;
            if state.lifecycle == TurnLifecycle::Completed {
                inner.prune_completed(&state.thread_id).await?;
            }
            Ok(())
        })
    }

    fn put_unless_completed(&self, state: &TurnState) -> Result<bool, String> {
        let state = state.clone();
        self.run(|inner| async move {
            inner.declared().await?;
            let mut declined = false;
            let written = inner
                .repo
                .update(&id(&state.thread_id, &state.request_id)?, |current| {
                    declined = false;
                    if let Some(current) = current
                        && decode_body(&current)?.lifecycle == TurnLifecycle::Completed
                    {
                        declined = true;
                        return Ok(None);
                    }
                    Ok(Some(encode(&state)?))
                })
                .await?;
            if declined || written.is_none() {
                return Ok(false);
            }
            if state.lifecycle == TurnLifecycle::Completed {
                inner.prune_completed(&state.thread_id).await?;
            }
            Ok(true)
        })
    }

    fn get(&self, thread_id: &str) -> Result<Option<TurnState>, String> {
        Ok(self.list_thread(thread_id)?.into_iter().next())
    }

    fn get_turn(&self, thread_id: &str, request_id: &str) -> Result<Option<TurnState>, String> {
        let (thread_id, request_id) = (thread_id.to_string(), request_id.to_string());
        self.run(|inner| async move {
            Ok(inner
                .get_turn(&thread_id, &request_id)
                .await?
                .map(|stored| stored.doc))
        })
    }

    fn delete(&self, thread_id: &str) -> Result<bool, String> {
        let thread_id = thread_id.to_string();
        self.run(|inner| async move {
            inner.declared().await?;
            let removed = inner
                .docs
                .delete_where(COLLECTION, &Filter::eq("thread_id", thread_id))
                .await?;
            Ok(removed > 0)
        })
    }

    fn delete_turn(&self, thread_id: &str, request_id: &str) -> Result<bool, String> {
        let (thread_id, request_id) = (thread_id.to_string(), request_id.to_string());
        self.run(|inner| async move {
            inner.declared().await?;
            inner
                .docs
                .delete(
                    COLLECTION,
                    &id(&thread_id, &request_id)?,
                    Precondition::None,
                )
                .await
        })
    }

    fn list(&self) -> Result<Vec<TurnState>, String> {
        self.run(|inner| async move {
            let mut latest: HashMap<String, TurnState> = HashMap::new();
            for turn in inner.query(Filter::All).await? {
                let turn = turn.doc;
                match latest.get(&turn.thread_id) {
                    Some(kept) if newest_first(&turn, kept) != Ordering::Less => {}
                    _ => {
                        latest.insert(turn.thread_id.clone(), turn);
                    }
                }
            }
            Ok(latest.into_values().collect())
        })
    }

    fn list_thread(&self, thread_id: &str) -> Result<Vec<TurnState>, String> {
        let thread_id = thread_id.to_string();
        self.run(|inner| async move { inner.thread(&thread_id).await })
    }

    fn clear_all(&self) -> Result<usize, String> {
        self.run(|inner| async move {
            inner.declared().await?;
            let removed = inner.docs.delete_where(COLLECTION, &Filter::All).await?;
            Ok(usize::try_from(removed).unwrap_or(usize::MAX))
        })
    }

    fn mark_all_interrupted(&self, now_rfc3339: &str) -> Result<usize, String> {
        self.mark_all_interrupted_checked(now_rfc3339)
            .map_err(|error| error.to_string())
    }

    fn settle_turn(
        &self,
        thread_id: &str,
        request_id: &str,
        lifecycle: TurnLifecycle,
        now_rfc3339: &str,
    ) -> Result<bool, String> {
        let (thread_id, request_id) = (thread_id.to_string(), request_id.to_string());
        let now = now_rfc3339.to_string();
        self.run(|inner| async move {
            let settled = inner
                .update(&thread_id, &request_id, |current| {
                    if is_terminal(current) {
                        return None;
                    }
                    let mut next = current.clone();
                    next.lifecycle = lifecycle;
                    next.phase = None;
                    next.active_tool = None;
                    next.active_subagent = None;
                    next.updated_at.clone_from(&now);
                    Some(next)
                })
                .await?;
            if settled.is_some() && lifecycle == TurnLifecycle::Completed {
                inner.prune_completed(&thread_id).await?;
            }
            Ok(settled.is_some())
        })
    }
}

#[path = "turn_states_storage.rs"]
pub(super) mod storage;

#[cfg(test)]
#[path = "turn_states_tests.rs"]
mod tests;
