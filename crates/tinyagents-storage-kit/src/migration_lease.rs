//! Renewable owner-local leases for storage open and migration.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use tinystoragedrivers_types::{
    Capability, CollectionSpec, ErrorKind, ExpectedDocument, Precondition, ReadCheck, Result,
    StorageError, Version, WriteOp,
};
use tokio::sync::Mutex;

use crate::primitives::{self, Acquire, Acquisition, Lease, Leases};
use crate::{Key, MIGRATIONS_COLLECTION, ScopedStorage};

const MIGRATION_KEY: &str = "owner_migrations";
const DEFAULT_TTL_MS: i64 = 30_000;

/// A renewable lease held while one owner prepares and migrates its storage.
#[derive(Clone, Debug)]
pub struct MigrationLease {
    docs: Arc<dyn crate::DocumentStore>,
    leases: Leases,
    owner: String,
    holder: String,
    lease: Lease,
    version: Arc<Mutex<Version>>,
    ttl_ms: i64,
}

impl MigrationLease {
    async fn acquire(
        storage: &ScopedStorage,
        owner: &str,
        ttl_ms: i64,
        leases: Leases,
    ) -> Result<Self> {
        storage
            .documents()
            .ensure_collection(&primitives::leases_collection_spec())
            .await?;
        let holder = uuid::Uuid::new_v4().to_string();
        loop {
            match leases
                .acquire_with(Acquire {
                    owner,
                    key: MIGRATION_KEY,
                    holder: &holder,
                    ttl_ms,
                    endpoint: None,
                    held_epoch: None,
                })
                .await?
            {
                Acquisition::Granted { lease, .. } => {
                    let id = lease_id(owner)?;
                    let version = storage
                        .documents()
                        .get(primitives::LEASES, id.as_str())
                        .await?
                        .ok_or_else(|| StorageError::conflict("migration lease disappeared"))?;
                    ensure_tenure(&version.doc, &holder, lease.epoch)?;
                    return Ok(Self {
                        docs: Arc::clone(storage.documents()),
                        leases,
                        owner: owner.to_owned(),
                        holder,
                        lease,
                        version: Arc::new(Mutex::new(version.version)),
                        ttl_ms,
                    });
                }
                Acquisition::Held(_) => {
                    tokio::time::sleep(Duration::from_millis(25)).await;
                }
            }
        }
    }

    /// Renew this lease and return an error if a later tenure owns the key.
    pub async fn renew(&self) -> Result<()> {
        let lease = self
            .leases
            .renew(
                &self.owner,
                MIGRATION_KEY,
                &self.holder,
                self.lease.epoch,
                self.ttl_ms,
            )
            .await?
            .ok_or_else(|| StorageError::conflict("owner migration lease was lost"))?;
        if lease.holder != self.holder || lease.epoch != self.lease.epoch {
            return Err(StorageError::conflict(
                "owner migration lease was taken over",
            ));
        }
        let id = lease_id(&self.owner)?;
        let current = self
            .docs
            .get(primitives::LEASES, id.as_str())
            .await?
            .ok_or_else(|| StorageError::conflict("owner migration lease disappeared"))?;
        ensure_tenure(&current.doc, &self.holder, self.lease.epoch)?;
        *self.version.lock().await = current.version;
        Ok(())
    }

    /// Record an importer completion against this exact lease tenure.
    ///
    /// When the document backend supports validated batches, a lease takeover
    /// and this marker write cannot pass each other. Backends without that
    /// optional capability still receive a fresh lease check before the write.
    pub async fn mark_done(&self, owner: &str, step: &str) -> Result<()> {
        if owner != self.owner {
            return Err(StorageError::invalid_input(
                "a migration lease can only complete steps for its owner",
            ));
        }
        let id = Key::of([owner, step])?;
        let body = serde_json::to_value(Progress {
            owner: owner.to_owned(),
            step: step.to_owned(),
            from: 0,
            to: 0,
            cursor: None,
            written: 0,
            done: true,
        })?;
        self.docs
            .ensure_collection(&CollectionSpec::new(MIGRATIONS_COLLECTION))
            .await?;
        if !self
            .docs
            .capabilities()
            .contains(Capability::ValidatedBatch)
        {
            self.renew().await?;
            return self
                .docs
                .put(MIGRATIONS_COLLECTION, id.as_str(), body, Precondition::None)
                .await
                .map(|_| ());
        }
        let lease_id = lease_id(&self.owner)?;
        for _ in 0..4 {
            self.renew().await?;
            let expected = *self.version.lock().await;
            match self
                .docs
                .atomic_batch_checked(
                    vec![ReadCheck {
                        collection: primitives::LEASES.to_owned(),
                        id: lease_id.as_str().to_owned(),
                        expected: ExpectedDocument::Version(expected),
                    }],
                    vec![WriteOp::Put {
                        collection: MIGRATIONS_COLLECTION.to_owned(),
                        id: id.as_str().to_owned(),
                        doc: body.clone(),
                        precondition: Precondition::None,
                    }],
                )
                .await
            {
                Ok(_) => return Ok(()),
                Err(error) if error.kind() == ErrorKind::Conflict => continue,
                Err(error) => return Err(error),
            }
        }
        Err(StorageError::conflict(
            "owner migration lease changed while marking completion",
        ))
    }
}

/// Run `work` while holding and renewing the migration lease for `owner`.
///
/// The lease is scoped to the already-bound storage handle. A failed work
/// future releases its lease; a lost lease turns an otherwise successful
/// future into a conflict rather than returning handles opened by a stale
/// migrator.
pub async fn with_migration_lease<T, F, Fut>(
    storage: &ScopedStorage,
    owner: &str,
    work: F,
) -> Result<T>
where
    F: FnOnce(MigrationLease) -> Fut,
    Fut: Future<Output = Result<T>>,
{
    with_migration_lease_using(
        storage,
        owner,
        DEFAULT_TTL_MS,
        Leases::new(storage.documents()),
        work,
    )
    .await
}

async fn with_migration_lease_using<T, F, Fut>(
    storage: &ScopedStorage,
    owner: &str,
    ttl_ms: i64,
    leases: Leases,
    work: F,
) -> Result<T>
where
    F: FnOnce(MigrationLease) -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let lease = MigrationLease::acquire(storage, owner, ttl_ms, leases).await?;
    let heartbeat = lease.clone();
    let (stop_tx, mut stop_rx) = tokio::sync::watch::channel(false);
    let (lease_lost_tx, lease_lost_rx) = tokio::sync::oneshot::channel();
    let stop = StopHeartbeat(stop_tx);
    let renew_every = Duration::from_millis((ttl_ms / 3).max(1) as u64);
    let task = tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = tokio::time::sleep(renew_every) => {},
                changed = stop_rx.changed() => {
                    if changed.is_err() || *stop_rx.borrow() {
                        return;
                    }
                }
            }
            if heartbeat.renew().await.is_err() {
                let _ = lease_lost_tx.send(());
                return;
            }
        }
    });
    let result = tokio::select! {
        result = work(lease.clone()) => result,
        _ = lease_lost_rx => Err(StorageError::conflict(
            "owner migration lease renewal failed",
        )),
    };
    let _ = stop.0.send(true);
    let _ = task.await;
    let final_check = lease.renew().await;
    let _ = lease
        .leases
        .release(
            &lease.owner,
            MIGRATION_KEY,
            &lease.holder,
            lease.lease.epoch,
        )
        .await;
    match (result, final_check) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
    }
}

struct StopHeartbeat(tokio::sync::watch::Sender<bool>);

impl Drop for StopHeartbeat {
    fn drop(&mut self) {
        let _ = self.0.send(true);
    }
}

fn lease_id(owner: &str) -> Result<Key> {
    Key::of([owner, MIGRATION_KEY])
}

fn ensure_tenure(doc: &serde_json::Value, holder: &str, epoch: u64) -> Result<()> {
    if doc.get("holder").and_then(serde_json::Value::as_str) != Some(holder)
        || doc.get("epoch").and_then(serde_json::Value::as_u64) != Some(epoch)
    {
        return Err(StorageError::conflict(
            "owner migration lease was taken over",
        ));
    }
    Ok(())
}

#[derive(Serialize)]
struct Progress {
    owner: String,
    step: String,
    from: u32,
    to: u32,
    cursor: Option<String>,
    written: u64,
    done: bool,
}

#[cfg(test)]
#[path = "migration_lease_tests.rs"]
mod tests;
