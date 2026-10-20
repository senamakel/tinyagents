//! Stream sealing and retention declarations.

use serde::{Deserialize, Serialize};

/// When the owner of a stream seals it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SealWhen {
    /// The owner calls `StreamStore::seal` when
    /// the stream's producer finishes.
    OnFinish,
    /// Maintenance seals the stream once it has gone this many milliseconds
    /// without an append.
    Idle(u64),
    /// The stream is never sealed.
    Never,
}

/// The sealing and retention policy a `Stream` schema item states.
///
/// This is the declaration; `Registry::maintain`
/// applies it by sealing idle streams and deleting expired ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SealPolicy {
    /// When to seal.
    pub when: SealWhen,
    /// Delete the stream (`delete_stream`) this many milliseconds after it
    /// was sealed. `None` keeps it until the owner deletes it. Unsigned like
    /// [`SealWhen::Idle`]: a negative duration cannot be stated, so a sealed
    /// stream is never due for deletion before the instant it was sealed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retain_ms: Option<u64>,
}

impl SealPolicy {
    /// A policy that seals `when` and keeps sealed streams.
    #[must_use]
    pub const fn new(when: SealWhen) -> Self {
        Self {
            when,
            retain_ms: None,
        }
    }

    /// The same policy, deleting a stream `retain_ms` after it was sealed.
    #[must_use]
    pub const fn retaining_ms(mut self, retain_ms: u64) -> Self {
        self.retain_ms = Some(retain_ms);
        self
    }
}

impl Default for SealPolicy {
    /// Never seal, keep forever.
    fn default() -> Self {
        Self::new(SealWhen::Never)
    }
}
