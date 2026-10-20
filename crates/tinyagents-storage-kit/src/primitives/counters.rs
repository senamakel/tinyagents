//! [`Counters`]: named monotonic counters.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::COUNTERS;
use super::repo::{Repo, RetryPolicy};
use tinystoragedrivers_api::DocumentStore;
use tinystoragedrivers_types::{Result, StorageError};

/// The largest value every driver can store (`i64::MAX`).
const MAX_VALUE: u64 = i64::MAX as u64;

#[derive(Debug, Serialize, Deserialize)]
struct Body {
    value: u64,
}

/// Atomic counters over [`COUNTERS`], one per `(owner, name)`.
///
/// Each step is one compare-and-swap [`update`](Repo::update), so two tasks
/// (or two processes on a shared backend) never receive the same number.
/// Counts live in a JSON number, so every driver stores the same range: 0 to
/// `i64::MAX`. [`next`](Self::next) and [`set_at_least`](Self::set_at_least)
/// refuse to go past it with `InvalidInput` rather than store a value some
/// backend could not hold.
#[derive(Debug, Clone)]
pub struct Counters {
    repo: Repo<Body>,
}

impl Counters {
    /// Counters on `docs`, which must be bound to the scope that owns them.
    ///
    /// # Panics
    ///
    /// Never: the collection name is a valid constant.
    #[must_use]
    #[allow(clippy::expect_used, reason = "COUNTERS is a valid constant name")]
    pub fn new(docs: &Arc<dyn DocumentStore>) -> Self {
        Self {
            repo: Repo::new(docs, COUNTERS).expect("tsd.counters is a valid collection"),
        }
    }

    /// Replace the retry policy of the underlying compare-and-swap loop.
    #[must_use]
    pub fn with_retry(mut self, retry: RetryPolicy) -> Self {
        self.repo = self.repo.with_retry(retry);
        self
    }

    /// Advance the counter and return the new value. The first call returns 1.
    ///
    /// # Errors
    ///
    /// A storage error, a conflict past the retry budget, or `InvalidInput`
    /// when the counter would pass `i64::MAX`.
    pub async fn next(&self, owner: &str, name: &str) -> Result<u64> {
        let id = id(owner, name)?;
        let stored = self
            .repo
            .update(&id, |body| {
                let value = body.map_or(Some(1), |b| {
                    b.value.checked_add(1).filter(|v| *v <= MAX_VALUE)
                });
                value
                    .map(|value| Some(Body { value }))
                    .ok_or_else(|| StorageError::invalid_input("counter overflow"))
            })
            .await?;
        stored
            .map(|(body, _)| body.value)
            .ok_or_else(unreachable_declined)
    }

    /// The current value, 0 for a counter never advanced.
    ///
    /// # Errors
    ///
    /// A storage error.
    pub async fn peek(&self, owner: &str, name: &str) -> Result<u64> {
        let found = self.repo.get(&id(owner, name)?).await?;
        Ok(found.map_or(0, |found| found.doc.value))
    }

    /// Raise the counter to at least `minimum` and return the resulting value.
    /// A counter already at or above it is left alone: use it after importing
    /// ids from elsewhere so [`next`](Self::next) cannot hand one out again.
    ///
    /// # Errors
    ///
    /// `InvalidInput` when `minimum` is above `i64::MAX`, a storage error, or a
    /// conflict past the retry budget.
    pub async fn set_at_least(&self, owner: &str, name: &str, minimum: u64) -> Result<u64> {
        if minimum > MAX_VALUE {
            return Err(StorageError::invalid_input("counter value out of range"));
        }
        let id = id(owner, name)?;
        let stored = self
            .repo
            .update(&id, |body| {
                Ok(match body {
                    Some(b) if b.value >= minimum => None,
                    _ => Some(Body { value: minimum }),
                })
            })
            .await?;
        match stored {
            Some((body, _)) => Ok(body.value),
            None => self.peek(owner, name).await,
        }
    }
}

fn id(owner: &str, name: &str) -> Result<String> {
    super::super::composite_key([owner, name])
}

fn unreachable_declined() -> StorageError {
    StorageError::backend("counter update declined unexpectedly")
}
