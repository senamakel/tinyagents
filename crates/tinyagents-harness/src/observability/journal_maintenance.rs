//! Host-driven maintenance for durable harness run journals.
//!
//! A host calls [`maintain_run_journals`] with its clock and retention policy.
//! The operation can seal journals belonging to inactive runs that were not
//! sealed during normal run completion, then remove expired sealed journals.
//! Active runs are excluded from both actions. A retention value of zero means
//! journals are kept indefinitely.

use std::collections::HashSet;

use crate::error::Result;
use crate::observability::{HarnessStatusStore, is_active};
use crate::store::map_driver_error as map_error;
use tinyagents_storage_kit::StreamStore;

/// Outcome counts from one journal maintenance pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JournalMaintenanceReport {
    /// Journals returned by the stream-family listing.
    pub inspected: usize,
    /// Previously open journals sealed for inactive or unknown runs.
    pub backfilled: usize,
    /// Expired sealed journals successfully deleted.
    pub deleted: usize,
    /// Eligible journals that changed before their conditional deletion.
    pub conflicts: usize,
}

/// Reconcile harness journals using host-supplied time and retention policy.
///
/// When `backfill_inactive` is true, open journals whose run is not active in
/// `statuses` are sealed. Sealed journals for active runs are always retained.
/// For inactive or unknown runs, positive `retention_ms` deletes a journal
/// only when `sealed_ms + retention_ms < now_ms`; equality remains retained.
/// A zero retention keeps journals indefinitely. Stream deletion is fenced by
/// the seal timestamp observed by this call, so a changed incarnation is not
/// accidentally removed. Errors from either owner port are propagated.
///
/// Status and stream stores are independent ports, so this operation does not
/// claim a cross-store atomic snapshot. Hosts should invoke it as background
/// maintenance, not as part of a live run's append path.
pub async fn maintain_run_journals(
    statuses: &dyn HarnessStatusStore,
    streams: &dyn StreamStore,
    retention_ms: u64,
    now_ms: i64,
    backfill_inactive: bool,
) -> Result<JournalMaintenanceReport> {
    let active: HashSet<String> = statuses
        .list_active()
        .await?
        .into_iter()
        .filter(|status| is_active(status))
        .map(|status| status.run_id.as_str().to_owned())
        .collect();

    let prefix = format!("{}/", crate::storage::JOURNAL);
    let names = streams.streams(&prefix).await.map_err(map_error)?;
    let mut report = JournalMaintenanceReport {
        inspected: names.len(),
        ..JournalMaintenanceReport::default()
    };

    for name in names {
        let Some(run_id) = name.strip_prefix(&prefix) else {
            continue;
        };
        if run_id.is_empty() {
            continue;
        }
        let is_active = active.contains(run_id);
        let Some(mut info) = streams.info(&name).await.map_err(map_error)? else {
            continue;
        };

        if backfill_inactive && !is_active && info.sealed_ms.is_none() {
            streams.seal(&name).await.map_err(map_error)?;
            report.backfilled += 1;
            let Some(refreshed) = streams.info(&name).await.map_err(map_error)? else {
                continue;
            };
            info = refreshed;
        }

        if is_active || retention_ms == 0 {
            continue;
        }
        let Some(sealed_ms) = info.sealed_ms else {
            continue;
        };
        let retention_ms = i64::try_from(retention_ms).unwrap_or(i64::MAX);
        if sealed_ms.saturating_add(retention_ms) < now_ms {
            if streams
                .delete_sealed_stream(&name, sealed_ms)
                .await
                .map_err(map_error)?
            {
                report.deleted += 1;
            } else {
                report.conflicts += 1;
            }
        }
    }

    Ok(report)
}

#[cfg(test)]
#[path = "journal_maintenance_tests.rs"]
mod tests;
