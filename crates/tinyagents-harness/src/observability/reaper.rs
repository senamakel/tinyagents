//! Startup reconciliation for orphaned runs.
//!
//! When a host process exits (a crash, a restart, a deploy) any run still in a
//! non-terminal state with an expired, owned lease is eligible for reaping.
//! Unleased or legacy records have unknown ownership and are retained. Stores
//! must implement the conditional transition atomically; the default is a safe
//! no-op. [`reap_orphaned_runs_and_seal`] seals only runs actually settled.

use std::time::SystemTime;

use super::{HarnessEventJournal, HarnessStatusStore};

/// Error recorded on a run reaped by the startup sweep. Stable and
/// grep-friendly so an operator (or a test) can tell a reaped run from a
/// genuinely failed one.
pub const ORPHAN_REAP_REASON: &str = "run orphaned: core restarted while the run was in flight";

/// Reap every run left non-terminal by a previous process.
///
/// Lists the still-active runs in `store` and moves each to `Cancelled`.
/// `Cancelled` (not `Failed`) is deliberate: a restart is not a run failure and
/// `Failed` would render as a red error row. Best-effort: a per-run persistence
/// failure is logged and does not abort the sweep, and a failure to list logs
/// and yields `0`. Returns the number of runs reaped.
pub async fn reap_orphaned_runs(store: &dyn HarnessStatusStore) -> usize {
    sweep(store, None, SystemTime::now()).await
}

/// [`reap_orphaned_runs`], and seal the journal of every run it reaped.
///
/// A reaped run is finished for good (its executor died with the process), so
/// its journal gets no more observations: sealing it lets a durable journal
/// store it compactly and apply its retention period, without changing what a
/// reader sees. A run that never journaled anything has nothing to seal. A
/// failure to seal is logged and does not undo the reap or stop the sweep.
/// Returns the number of runs reaped.
pub async fn reap_orphaned_runs_and_seal(
    store: &dyn HarnessStatusStore,
    journal: &dyn HarnessEventJournal,
) -> usize {
    sweep(store, Some(journal), SystemTime::now()).await
}

async fn sweep(
    store: &dyn HarnessStatusStore,
    journal: Option<&dyn HarnessEventJournal>,
    now: SystemTime,
) -> usize {
    // One id for the whole sweep so every line can be tied to the same boot.
    // Derived from the clock rather than a uuid so it stays ordered in a tail.
    let sweep_id = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    tracing::debug!("[agent] startup run sweep entry sweep_id={sweep_id}");

    let reaped = match store.reap_expired(now).await {
        Ok(runs) => runs,
        Err(err) => {
            tracing::warn!(
                "[agent] startup run sweep exit sweep_id={sweep_id} branch=reap-failed: {err}"
            );
            return 0;
        }
    };

    if reaped.is_empty() {
        tracing::debug!("[agent] startup run sweep exit sweep_id={sweep_id} branch=none-active");
        return 0;
    }
    tracing::debug!(
        "[agent] startup run sweep sweep_id={sweep_id} expired={} reaped",
        reaped.len()
    );

    let count = reaped.len();
    for status in reaped {
        let run_id = status.run_id.as_str().to_string();
        if let Some(journal) = journal
            && let Err(err) = journal.seal_run(&run_id).await
        {
            tracing::warn!(
                "[agent] startup run sweep sweep_id={sweep_id} run_id={run_id} \
                 journal seal failed: {err}"
            );
        }
    }
    // One operator-visible line, and only when the sweep did something.
    if count > 0 {
        tracing::info!(
            "[agent] startup run sweep sweep_id={sweep_id} reaped {count} expired run(s)"
        );
    } else {
        tracing::debug!("[agent] startup run sweep exit sweep_id={sweep_id} branch=nothing-reaped");
    }
    count
}

#[cfg(test)]
#[path = "reaper_tests.rs"]
mod tests;
