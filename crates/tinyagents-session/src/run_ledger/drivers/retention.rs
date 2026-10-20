//! Explicit age-based telemetry retention through the bound native document port.

use super::{DriverRunLedger, RunTelemetry, storage};
use chrono::{DateTime, Utc};
use tinyagents_storage_kit::{
    DocumentStoreExt, ErrorKind, Precondition, Query, StorageError, WriteOp, WriteResult,
};

impl DriverRunLedger {
    /// Delete one observed eligible set atomically, retrying only version conflicts.
    ///
    /// A concurrent update that refreshes a candidate is re-read before retry.
    /// New rows arriving after enumeration belong to a later retention pass.
    /// An empty batch still obtains native fence/capability admission. Schema
    /// initialization follows the handle’s declaring or validation policy; a
    /// refusal never authorizes another scope or a local fallback.
    pub(super) async fn prune_telemetry(
        &self,
        older_than: DateTime<Utc>,
    ) -> Result<usize, StorageError> {
        self.declare().await?;
        for _ in 0..storage::CAS_ATTEMPTS {
            let candidates = self
                .scoped
                .documents()
                .query_all(storage::TELEMETRY, &Query::default())
                .await?;
            let mut deletes = Vec::new();
            for candidate in candidates {
                let record: RunTelemetry = storage::decode(candidate.doc)?;
                // Compare instants, not variable-precision RFC3339 strings.
                if record
                    .updated_at
                    .is_some_and(|updated| updated < older_than)
                {
                    deletes.push(WriteOp::Delete {
                        collection: storage::TELEMETRY.into(),
                        id: candidate.id,
                        precondition: Precondition::Version(candidate.version),
                    });
                }
            }
            match self
                .scoped
                .documents()
                .atomic_batch_checked(Vec::new(), deletes)
                .await
            {
                Ok(results) => {
                    return Ok(results
                        .into_iter()
                        .filter(|result| matches!(result, WriteResult::Delete { removed: true }))
                        .count());
                }
                Err(error) if error.kind() == ErrorKind::Conflict => continue,
                Err(error) => return Err(error),
            }
        }
        Err(StorageError::conflict(
            "run-ledger telemetry retention contention budget exhausted",
        ))
    }
}
