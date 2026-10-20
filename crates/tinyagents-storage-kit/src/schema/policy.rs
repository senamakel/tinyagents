//! [`Registry::maintain`]: applying the seal and retention policies the
//! schemas declare.
//!
//! Order matters for a host that also reclaims space: run the policies first
//! (they free rows and blobs) and then
//! [`StorageBackend::maintain`](crate::StorageBackend::maintain) (it gives the
//! freed space back to the filesystem).
//!
//! ```text
//! registry.maintain_all(&backend, now_ms, budget).await?;   // seal, delete
//! backend.maintain(budget).await?;                          // vacuum
//! ```

#[cfg(any(test, doc))]
use super::Registry;
use super::compose::PreparedRegistry;
use super::types::Item;
use crate::backend::{MaintenanceBudget, ScopedStorage, StorageBackend};
use crate::error::{ErrorKind, Result};
use crate::stream::{SealPolicy, SealWhen, StreamInfo};

/// What one [`Registry::maintain`] (or [`Registry::maintain_all`]) call did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct PolicyReport {
    /// Scopes visited (`maintain_all`; 1 for `maintain`).
    pub scopes: u64,
    /// Streams whose info was read.
    pub inspected: u64,
    /// Streams sealed because they were idle.
    pub sealed: u64,
    /// Streams deleted because their retention had passed.
    pub deleted: u64,
    /// Attempts that lost a race (the stream changed or vanished while it was
    /// being sealed or deleted). The next call sees the stream again.
    pub conflicts: u64,
    /// The budget ran out while work remained, so another call may find more.
    pub exhausted: bool,
}

/// How many more streams a call may touch.
struct Allowance(Option<u64>);

impl Allowance {
    fn spent(&self) -> bool {
        self.0 == Some(0)
    }

    fn touch(&mut self) {
        if let Some(left) = &mut self.0 {
            *left = left.saturating_sub(1);
        }
    }
}

#[derive(Clone, Copy)]
enum Action {
    Seal,
    /// Delete the incarnation sealed at this instant.
    Delete(i64),
}

/// What `policy` wants done to a stream with `info` at `now_ms`. A missing
/// time means "unknown": no action.
fn due(policy: &SealPolicy, info: &StreamInfo, now_ms: i64) -> Option<Action> {
    if let Some(sealed) = info.sealed_ms {
        let retain = i64::try_from(policy.retain_ms?).unwrap_or(i64::MAX);
        (sealed.saturating_add(retain) < now_ms).then_some(Action::Delete(sealed))
    } else {
        let SealWhen::Idle(idle_ms) = policy.when else {
            return None;
        };
        let last = info.last_append_ms?;
        let idle_ms = i64::try_from(idle_ms).unwrap_or(i64::MAX);
        (now_ms.saturating_sub(last) > idle_ms).then_some(Action::Seal)
    }
}

/// Whether a policy can ever act: `Never` and `OnFinish` without retention
/// cannot, so such a family is not even listed.
fn acts(policy: &SealPolicy) -> bool {
    matches!(policy.when, SealWhen::Idle(_)) || policy.retain_ms.is_some()
}

impl PreparedRegistry {
    /// Apply every declared stream policy to the streams in `storage`'s scope.
    ///
    /// For each `Item::Stream` family (the base stream `<base>` and every
    /// `<base>/…`), at `now_ms` (epoch milliseconds):
    ///
    /// - a stream that is not sealed, under [`SealWhen::Idle`]`(ms)`, whose
    ///   last append is more than `ms` before `now_ms`, is sealed;
    /// - a sealed stream under a policy with `retain_ms`, sealed more than
    ///   `retain_ms` before `now_ms`, is deleted (`delete_sealed_stream`,
    ///   which removes the sealed blob too, and only if the stream is still
    ///   the one inspected: sealed at the same instant).
    ///
    /// [`SealWhen::OnFinish`] streams are sealed by their owners and
    /// [`SealWhen::Never`] ones never, but retention applies to any sealed
    /// stream whose policy states it. A stream whose driver recorded no time
    /// (written before times existed) is left alone.
    ///
    /// `budget.max_streams` bounds the streams sealed or deleted in this
    /// call; when it runs out the report says `exhausted` and the host calls
    /// again later. A race (another writer sealed or deleted the stream
    /// first, or deleted and recreated it so it is no longer the one
    /// inspected) is counted in `conflicts`, not returned. An idle stream
    /// appended to after inspection is still sealed: idleness was the
    /// trigger, and `seal` keeps every entry.
    ///
    /// The policy applies to every scope class: the host chooses which scopes
    /// to hand in. Run the backend's own
    /// [`maintain`](crate::StorageBackend::maintain) afterwards to return the
    /// freed space.
    ///
    /// # Errors
    ///
    /// `Unsupported(StreamSeal)` when the driver records no stream times and
    /// some policy has work to do, or a backend error.
    pub(super) async fn maintain(
        &self,
        storage: &ScopedStorage,
        now_ms: i64,
        budget: MaintenanceBudget,
    ) -> Result<PolicyReport> {
        let mut allowance = Allowance(budget.max_streams);
        let mut report = PolicyReport {
            scopes: 1,
            ..PolicyReport::default()
        };
        self.apply(storage, now_ms, &mut allowance, &mut report)
            .await?;
        Ok(report)
    }

    /// [`maintain`](Self::maintain) over every scope `backend` holds records
    /// for, sharing one budget across them.
    ///
    /// Needs [`Capability::ScopeEnumeration`](crate::Capability::ScopeEnumeration);
    /// without it, enumerate the scopes yourself and call
    /// [`maintain`](Self::maintain) for each.
    ///
    /// # Errors
    ///
    /// `Unsupported(ScopeEnumeration)` when the backend cannot list scopes,
    /// otherwise as [`maintain`](Self::maintain).
    pub(super) async fn maintain_all(
        &self,
        backend: &dyn StorageBackend,
        now_ms: i64,
        budget: MaintenanceBudget,
    ) -> Result<PolicyReport> {
        let mut allowance = Allowance(budget.max_streams);
        let mut report = PolicyReport::default();
        for scope in backend.scopes("").await? {
            let storage = backend.for_scope(&scope)?;
            report.scopes += 1;
            self.apply(&storage, now_ms, &mut allowance, &mut report)
                .await?;
            if report.exhausted {
                break;
            }
        }
        Ok(report)
    }

    async fn apply(
        &self,
        storage: &ScopedStorage,
        now_ms: i64,
        allowance: &mut Allowance,
        report: &mut PolicyReport,
    ) -> Result<()> {
        let streams = storage.streams();
        for schema in &self.schemas {
            for item in &schema.items {
                let Item::Stream {
                    name: base, seal, ..
                } = item
                else {
                    continue;
                };
                if !acts(seal) {
                    continue;
                }
                let family = format!("{base}/");
                for name in streams.streams(base).await? {
                    if name != *base && !name.starts_with(&family) {
                        continue;
                    }
                    let Some(info) = streams.info(&name).await? else {
                        continue;
                    };
                    report.inspected += 1;
                    let Some(action) = due(seal, &info, now_ms) else {
                        continue;
                    };
                    if allowance.spent() {
                        report.exhausted = true;
                        return Ok(());
                    }
                    allowance.touch();
                    let outcome = match action {
                        Action::Seal => streams.seal(&name).await.map(|_| true),
                        Action::Delete(sealed_ms) => {
                            streams.delete_sealed_stream(&name, sealed_ms).await
                        }
                    };
                    match (outcome, action) {
                        (Ok(true), Action::Seal) => report.sealed += 1,
                        (Ok(true), Action::Delete(_)) => report.deleted += 1,
                        // The stream is no longer the sealed one inspected.
                        (Ok(false), _) => report.conflicts += 1,
                        (Err(error), _)
                            if matches!(
                                error.kind(),
                                ErrorKind::Conflict | ErrorKind::NotFound
                            ) =>
                        {
                            report.conflicts += 1;
                        }
                        (Err(error), _) => return Err(error),
                    }
                }
            }
        }
        Ok(())
    }
}
