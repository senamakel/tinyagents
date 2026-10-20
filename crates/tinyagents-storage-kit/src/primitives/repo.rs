//! [`Repo`]: a typed collection handle and the one compare-and-swap loop.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use serde::de::DeserializeOwned;

use tinystoragedrivers_api::DocumentStore;
use tinystoragedrivers_types::ErrorKind;
use tinystoragedrivers_types::{
    Page, Precondition, Query, Result, Version, Versioned, WriteOptions, validate_collection,
};

/// How long [`Repo::update`] keeps retrying a lost compare-and-swap.
///
/// The default is the spec's budget: 32 attempts with a jittered backoff that
/// starts at 1 ms and doubles up to 8 ms. A lost swap means another writer
/// made progress, so the loop converges; the backoff only spreads the retries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Total attempts, including the first. At least one is always made.
    pub attempts: u32,
    /// The pause before the second attempt; doubles each attempt.
    pub initial_backoff: Duration,
    /// The longest pause. Each pause is jittered into its upper half.
    pub max_backoff: Duration,
}

impl RetryPolicy {
    /// A policy with the given budget and backoff bounds.
    #[must_use]
    pub const fn new(attempts: u32, initial_backoff: Duration, max_backoff: Duration) -> Self {
        Self {
            attempts,
            initial_backoff,
            max_backoff,
        }
    }

    /// The full budget with no pause between attempts, for tests.
    #[must_use]
    pub const fn immediate() -> Self {
        Self::new(32, Duration::ZERO, Duration::ZERO)
    }

    /// The pause after the failed attempt number `attempt` (counting from 0):
    /// the doubled backoff capped at the maximum, jittered into its upper half.
    fn pause(&self, attempt: u32) -> Duration {
        let doubled = self
            .initial_backoff
            .saturating_mul(1_u32.checked_shl(attempt).unwrap_or(u32::MAX));
        let cap = doubled.min(self.max_backoff);
        let nanos = u64::try_from(cap.as_nanos()).unwrap_or(u64::MAX);
        if nanos == 0 {
            return Duration::ZERO;
        }
        // `RandomState` draws fresh keys per instance: jitter without a
        // random-number dependency.
        let random = RandomState::new().build_hasher().finish();
        let half = nanos / 2;
        Duration::from_nanos(half + random % (nanos - half + 1))
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::new(32, Duration::from_millis(1), Duration::from_millis(8))
    }
}

/// A typed handle on one collection of one scope.
///
/// Bodies are `T` serialized as a JSON object (see
/// [`validate_doc`](crate::validate_doc)). Every write carries the repo's
/// [`WriteOptions`], so a collection declares its schema version (and, for a
/// short-lived one, an expiry) once.
///
/// ```
/// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
/// use serde::{Deserialize, Serialize};
/// use tinystoragedrivers_api::StorageBackend;
/// use tinystoragedrivers_testkit::memory_backend;
/// use tinyagents_storage_kit::primitives::Repo;
/// use tinystoragedrivers_types::{CollectionSpec, Scope};
///
/// #[derive(Serialize, Deserialize, PartialEq, Debug)]
/// struct Job { runs: u32 }
///
/// let storage = memory_backend().for_scope(&Scope::new("local")?)?;
/// storage.documents().ensure_collection(&CollectionSpec::new("jobs")).await?;
/// let jobs = Repo::<Job>::new(storage.documents(), "cron.jobs")?;
/// jobs.update("nightly", |job| Ok(Some(Job { runs: job.map_or(1, |j| j.runs + 1) }))).await?;
/// let (job, _version) = jobs
///     .update("nightly", |job| Ok(Some(Job { runs: job.map_or(1, |j| j.runs + 1) })))
///     .await?
///     .unwrap();
/// assert_eq!(job.runs, 2);
/// # Ok::<(), tinystoragedrivers_types::StorageError>(())
/// # }).unwrap();
/// ```
pub struct Repo<T> {
    docs: Arc<dyn DocumentStore>,
    collection: String,
    options: WriteOptions,
    retry: RetryPolicy,
    // `fn() -> T` keeps the handle `Send + Sync` whatever `T` is.
    marker: PhantomData<fn() -> T>,
}

impl<T> Clone for Repo<T> {
    fn clone(&self) -> Self {
        Self {
            docs: Arc::clone(&self.docs),
            collection: self.collection.clone(),
            options: self.options,
            retry: self.retry,
            marker: PhantomData,
        }
    }
}

impl<T> std::fmt::Debug for Repo<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Repo")
            .field("collection", &self.collection)
            .field("options", &self.options)
            .field("retry", &self.retry)
            .finish_non_exhaustive()
    }
}

impl<T> Repo<T> {
    /// A handle on `collection`, with plain writes and the default
    /// [`RetryPolicy`].
    ///
    /// The collection is not created: declare it with
    /// [`ensure_collection`](DocumentStore::ensure_collection) when it needs
    /// indexes.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidInput`] for an invalid collection name.
    pub fn new(docs: &Arc<dyn DocumentStore>, collection: impl Into<String>) -> Result<Self> {
        let collection = collection.into();
        validate_collection(&collection)?;
        Ok(Self {
            docs: Arc::clone(docs),
            collection,
            options: WriteOptions::default(),
            retry: RetryPolicy::default(),
            marker: PhantomData,
        })
    }

    /// Declare the schema version and expiry every write of this handle carries.
    #[must_use]
    pub fn with_options(mut self, options: WriteOptions) -> Self {
        self.options = options;
        self
    }

    /// Replace the [`RetryPolicy`] of [`update`](Self::update).
    #[must_use]
    pub fn with_retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    /// The collection this handle is bound to.
    #[must_use]
    pub fn collection(&self) -> &str {
        &self.collection
    }
}

impl<T: Serialize + DeserializeOwned + Send + Sync> Repo<T> {
    /// Read one record.
    ///
    /// # Errors
    ///
    /// The read's error, or [`ErrorKind::Serialization`] when the body is not a `T`.
    pub async fn get(&self, id: &str) -> Result<Option<Versioned<T>>> {
        match self.docs.get(&self.collection, id).await? {
            Some(found) => Ok(Some(decode(found)?)),
            None => Ok(None),
        }
    }

    /// Write a record, replacing any earlier one (upsert).
    ///
    /// # Errors
    ///
    /// The write's error, or [`ErrorKind::Serialization`].
    pub async fn put(&self, id: &str, value: &T) -> Result<Version> {
        self.write(id, value, Precondition::None).await
    }

    /// Write a record that must not exist yet.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Conflict`] when the id is taken, or as [`put`](Self::put).
    pub async fn insert(&self, id: &str, value: &T) -> Result<Version> {
        self.write(id, value, Precondition::Absent).await
    }

    /// Remove a record and report whether it existed.
    ///
    /// # Errors
    ///
    /// The delete's error.
    pub async fn delete(&self, id: &str) -> Result<bool> {
        self.docs
            .delete(&self.collection, id, Precondition::None)
            .await
    }

    /// Remove a record only when `matches` accepts its current value, as one
    /// compare-and-swap: the delete is conditional on the version `keep_if`
    /// saw, so a writer that got in between turns it into a retry, not a
    /// lost update. Returns whether a record was removed; `false` when the
    /// record is absent or `matches` declined it.
    ///
    /// `matches` is called again on every retry, so it must be free of side
    /// effects. After [`RetryPolicy::attempts`] lost swaps the last
    /// [`Conflict`](ErrorKind::Conflict) is returned.
    ///
    /// # Errors
    ///
    /// A conflict past the budget, or any other read or delete error.
    pub async fn delete_if<F>(&self, id: &str, mut matches: F) -> Result<bool>
    where
        F: FnMut(&T) -> bool + Send,
    {
        let attempts = self.retry.attempts.max(1);
        let mut attempt = 0;
        loop {
            let Some(found) = self.docs.get(&self.collection, id).await? else {
                return Ok(false);
            };
            let found = decode::<T>(found)?;
            if !matches(&found.doc) {
                return Ok(false);
            }
            match self
                .docs
                .delete(&self.collection, id, Precondition::Version(found.version))
                .await
            {
                Ok(removed) => return Ok(removed),
                Err(error) if error.kind() == ErrorKind::Conflict => {
                    attempt += 1;
                    if attempt >= attempts {
                        return Err(error);
                    }
                    let pause = self.retry.pause(attempt - 1);
                    if !pause.is_zero() {
                        tokio::time::sleep(pause).await;
                    }
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// One page of the records matching `query`.
    ///
    /// # Errors
    ///
    /// The query's error, or [`ErrorKind::Serialization`].
    pub async fn query(&self, query: &Query) -> Result<Page<Versioned<T>>> {
        let page = self.docs.query(&self.collection, query).await?;
        Ok(Page {
            items: page
                .items
                .into_iter()
                .map(decode)
                .collect::<Result<Vec<_>>>()?,
            next: page.next,
        })
    }

    /// Read-modify-write one record as a single compare-and-swap loop.
    ///
    /// `f` receives the current value (`None` when the record is missing) and
    /// returns the value to store, or `None` to leave the record as it is.
    /// The write is conditional on the version `f` saw (or on the record still
    /// being absent). When another writer got in first the loop re-reads and
    /// calls `f` again, so `f` must be free of side effects. After
    /// [`RetryPolicy::attempts`] lost swaps the last
    /// [`Conflict`](ErrorKind::Conflict) is returned.
    ///
    /// Returns the stored value and its version, or `None` when `f` declined.
    ///
    /// # Errors
    ///
    /// `f`'s error (nothing is written), a conflict past the budget, or any
    /// other read or write error.
    pub async fn update<F>(&self, id: &str, mut f: F) -> Result<Option<(T, Version)>>
    where
        F: FnMut(Option<T>) -> Result<Option<T>> + Send,
    {
        let attempts = self.retry.attempts.max(1);
        let mut attempt = 0;
        loop {
            let (precondition, current) = match self.docs.get(&self.collection, id).await? {
                Some(found) => {
                    let found = decode::<T>(found)?;
                    (Precondition::Version(found.version), Some(found.doc))
                }
                None => (Precondition::Absent, None),
            };
            let Some(next) = f(current)? else {
                return Ok(None);
            };
            match self.write(id, &next, precondition).await {
                Ok(version) => return Ok(Some((next, version))),
                Err(error) if error.kind() == ErrorKind::Conflict => {
                    attempt += 1;
                    if attempt >= attempts {
                        return Err(error);
                    }
                    let pause = self.retry.pause(attempt - 1);
                    if !pause.is_zero() {
                        tokio::time::sleep(pause).await;
                    }
                }
                Err(error) => return Err(error),
            }
        }
    }

    async fn write(&self, id: &str, value: &T, precondition: Precondition) -> Result<Version> {
        let body = serde_json::to_value(value)?;
        self.docs
            .put_with(&self.collection, id, body, precondition, self.options)
            .await
    }
}

fn decode<T: DeserializeOwned>(found: Versioned<serde_json::Value>) -> Result<Versioned<T>> {
    Ok(Versioned {
        id: found.id,
        version: found.version,
        doc: serde_json::from_value(found.doc)?,
        meta: found.meta,
    })
}
