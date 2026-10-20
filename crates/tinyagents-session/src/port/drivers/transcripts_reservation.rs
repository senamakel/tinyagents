//! Generation reservations: a `tsd.leases` record per successor stem.
//!
//! A compaction (or a legacy adoption) reserves the stem it is about to write
//! before it seals the predecessor, so two of them never both open the
//! successor. The reservation is a lease of owner [`LEASE_OWNER`] keyed by the
//! stem, held for [`RESERVATION_TTL_MS`]:
//!
//! - a lease that lapsed was left by a process that stopped between reserving
//!   and writing. Another reserver takes it over, and the lease epoch moves up,
//!   so the stalled holder's [`claim`] is refused instead of overwriting the
//!   successor;
//! - the first write of the reserved handle [`claim`]s the lease (a renewal at
//!   the epoch it reserved); it is left to lapse afterwards, because a later
//!   reserver that raced the write is stopped by the written-entries check;
//! - clearing a reserved generation nobody wrote [`release`]s it, so the
//!   compaction can be retried at once instead of after the lease lapses.

use std::sync::Arc;

use tinyagents_storage_kit::primitives::{self, Leases};
use tinyagents_storage_kit::{CollectionSpec, Filter, StorageError};
use tinystoragedrivers_api::DocumentStore;

use super::ENTRIES;

/// The `tsd.leases` owner of the session crate's reservations.
pub(super) const LEASE_OWNER: &str = "session";

/// How long a generation reservation holds the successor stem. A reservation
/// that old and still unwritten was left by a process that stopped; without a
/// takeover its sealed predecessor would refuse every later write.
pub(super) const RESERVATION_TTL_MS: i64 = 30_000;

/// The lease collection reservations live in.
pub(crate) fn leases_spec() -> CollectionSpec {
    primitives::collection_specs()
        .into_iter()
        .find(|spec| spec.name == primitives::LEASES)
        .expect("the primitives always declare tsd.leases")
}

/// A reservation this handle holds: who, and at which lease epoch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Reservation {
    holder: String,
    epoch: u64,
}

/// Reserves `stem` as a new generation: succeeds when nothing is written
/// there and no live reservation holds it.
///
/// # Errors
///
/// A plain error when the generation exists or is reserved; a
/// [`StorageError`] when the store refused a read or write.
pub(super) async fn reserve(
    docs: &Arc<dyn DocumentStore>,
    leases: &Leases,
    stem: &str,
) -> anyhow::Result<Reservation> {
    let written = docs
        .count(
            ENTRIES,
            &Filter::eq("stem", stem).and(Filter::eq("written", true)),
        )
        .await?;
    anyhow::ensure!(written == 0, "session generation already exists");
    let holder = format!("reservation:{}", uuid::Uuid::new_v4());
    match leases
        .acquire(LEASE_OWNER, stem, &holder, RESERVATION_TTL_MS)
        .await?
    {
        Some(lease) => {
            tracing::debug!(
                target: "tinyagents_session::port::drivers",
                stem,
                epoch = lease.epoch,
                "[session-store] generation reserved"
            );
            Ok(Reservation {
                holder,
                epoch: lease.epoch,
            })
        }
        None => anyhow::bail!("session generation already exists or is reserved"),
    }
}

/// Claims `reservation` before its handle's first write.
///
/// # Errors
///
/// A plain error when another writer took the reservation over (this one
/// stalled past [`RESERVATION_TTL_MS`]): the generation is theirs and this
/// handle's writes are refused instead of overwriting it.
pub(super) async fn claim(
    leases: &Leases,
    stem: &str,
    reservation: &Reservation,
) -> anyhow::Result<()> {
    let renewed = leases
        .renew(
            LEASE_OWNER,
            stem,
            &reservation.holder,
            reservation.epoch,
            RESERVATION_TTL_MS,
        )
        .await?;
    anyhow::ensure!(
        renewed.is_some(),
        "session generation {stem} was taken over by another writer; reload the session"
    );
    Ok(())
}

/// Gives `reservation` back. A reservation another process has since taken
/// over is theirs and stays (the release reports `false`).
pub(super) async fn release(
    leases: &Leases,
    stem: &str,
    reservation: &Reservation,
) -> Result<bool, StorageError> {
    leases
        .release(LEASE_OWNER, stem, &reservation.holder, reservation.epoch)
        .await
}
