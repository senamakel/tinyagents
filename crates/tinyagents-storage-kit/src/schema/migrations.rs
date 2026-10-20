//! `tsd.migrations`: where the runner records step progress and where hosts
//! record the completion of legacy-file importers.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::backend::ScopedStorage;
use crate::document::{CollectionSpec, DocumentStore, DocumentStoreExt, Precondition};
use crate::error::Result;
use crate::key::Key;

/// The collection that holds one [`Progress`] document per step.
///
/// It lives in whatever scope the [`ScopedStorage`] given to
/// [`Registry::migrate`](super::Registry::migrate) or [`Migrations::new`] is
/// bound to, so progress is kept per scope: a tenant's scope records the
/// steps run over that tenant's data.
pub const MIGRATIONS_COLLECTION: &str = "tsd.migrations";

pub(super) fn migrations_spec() -> CollectionSpec {
    CollectionSpec::new(MIGRATIONS_COLLECTION)
}

/// The body of a `tsd.migrations` document: `{owner, step, from, to, cursor,
/// written, done}`. A runner step has the id `Key::of([owner, item, step])`;
/// An importer marked by [`MigrationLease::mark_done`](crate::MigrationLease::mark_done)
/// has `Key::of([owner, step])` and `from` and `to` of 0.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Progress {
    /// The owner of the step.
    pub owner: String,
    /// The step id.
    pub step: String,
    /// The version the step reads.
    pub from: u32,
    /// The version the step writes.
    pub to: u32,
    /// The keyset cursor after the last finished batch; `None` before the
    /// first batch and once the step is done.
    pub cursor: Option<String>,
    /// Records the step has written (or found already written).
    pub written: u64,
    /// Whether the step has completed.
    pub done: bool,
}

/// Internal handle on the runner's `tsd.migrations` progress documents.
#[derive(Debug, Clone)]
pub(super) struct Migrations {
    docs: Arc<dyn DocumentStore>,
}

impl Migrations {
    /// A handle over `storage`'s scope. The collection must have been
    /// declared by [`Registry::prepare`](super::Registry::prepare) or
    /// [`Registry::migrate`](super::Registry::migrate).
    #[must_use]
    pub(super) fn new(storage: &ScopedStorage) -> Self {
        Self {
            docs: Arc::clone(storage.documents()),
        }
    }

    pub(super) async fn load(&self, id: &Key) -> Result<Option<Progress>> {
        Ok(self
            .docs
            .get_as::<Progress>(MIGRATIONS_COLLECTION, id.as_str())
            .await?
            .map(|found| found.doc))
    }

    pub(super) async fn save(&self, id: &Key, progress: &Progress) -> Result<()> {
        self.docs
            .put_as(
                MIGRATIONS_COLLECTION,
                id.as_str(),
                progress,
                Precondition::None,
            )
            .await
            .map(|_| ())
    }
}

/// The id of a runner step's progress document.
pub(super) fn step_key(owner: &str, item: &str, step: &str) -> Result<Key> {
    Key::of([owner, item, step])
}
