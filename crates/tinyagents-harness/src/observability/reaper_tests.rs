//! Tests for the orphaned-run startup sweep.

use super::*;
use crate::events::HarnessRunStatus;
use crate::ids::{ComponentId, ExecutionStatus, HarnessPhase};
use crate::observability::{DriverStatusStore, mint_run_id};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tinystoragedrivers_core::{MemoryStorage, Scope, StorageBackend};

fn store() -> DriverStatusStore {
    let scoped = MemoryStorage::new().for_scope(&Scope::local()).unwrap();
    DriverStatusStore::new(Arc::clone(scoped.documents()))
}

async fn seed_status(store: &DriverStatusStore, kind: ExecutionStatus) -> String {
    let run_id = mint_run_id();
    let mut status =
        HarnessRunStatus::new(run_id.clone(), ComponentId::new("mock-model".to_string()));
    match kind {
        ExecutionStatus::Pending => {}
        ExecutionStatus::Running => status.mark_running(HarnessPhase::Model),
        ExecutionStatus::Interrupted => status.mark_interrupted(),
        other => panic!("seed_status only seeds non-terminal states, got {other:?}"),
    }
    status.lease_owner = Some("dead-process".into());
    status.lease_expires_at = Some(SystemTime::UNIX_EPOCH);
    status.lease_epoch = 1;
    store.put_status(status).await.unwrap();
    run_id.as_str().to_string()
}

#[tokio::test]
async fn reap_cancels_every_active_run_and_spares_terminal_ones() {
    let store = store();
    let pending = seed_status(&store, ExecutionStatus::Pending).await;
    let running = seed_status(&store, ExecutionStatus::Running).await;
    let interrupted = seed_status(&store, ExecutionStatus::Interrupted).await;

    let done = mint_run_id();
    let mut done_status =
        HarnessRunStatus::new(done.clone(), ComponentId::new("mock-model".to_string()));
    done_status.mark_running(HarnessPhase::Model);
    done_status.mark_completed();
    store.put_status(done_status).await.unwrap();

    assert_eq!(reap_orphaned_runs(&store).await, 3);

    for run_id in [&pending, &running, &interrupted] {
        let status = store.get_status(run_id).await.unwrap().expect("present");
        assert_eq!(status.status, ExecutionStatus::Cancelled);
        assert_eq!(status.current_phase, HarnessPhase::Done);
        assert_eq!(status.error.as_deref(), Some(ORPHAN_REAP_REASON));
        assert!(status.ended_at.is_some());
    }
    let done_after = store.get_status(done.as_str()).await.unwrap().unwrap();
    assert_eq!(done_after.status, ExecutionStatus::Completed);
    assert!(done_after.error.is_none());

    assert!(store.list_active().await.unwrap().is_empty());
    assert_eq!(reap_orphaned_runs(&store).await, 0, "idempotent");
}

#[tokio::test]
async fn reap_on_empty_store_is_a_noop() {
    let store = store();
    assert_eq!(reap_orphaned_runs(&store).await, 0);
}

#[tokio::test]
async fn shared_reaper_preserves_active_runs_with_unknown_ownership() {
    let backend = MemoryStorage::new();
    let scoped = backend.for_scope(&Scope::local()).unwrap();
    let owner = DriverStatusStore::new(Arc::clone(scoped.documents()));
    let reaper = DriverStatusStore::new(Arc::clone(scoped.documents()));
    // A row written by another process (or by a pre-lease version) has no
    // trustworthy owner/expiry, so the shared reaper must preserve it.
    let run_id = mint_run_id();
    let mut unowned =
        HarnessRunStatus::new(run_id.clone(), ComponentId::new("mock-model".to_string()));
    unowned.mark_running(HarnessPhase::Model);
    owner.put_status(unowned).await.unwrap();
    assert_eq!(reap_orphaned_runs(&reaper).await, 0);
    let status = owner.get_status(&run_id).await.unwrap().expect("present");
    assert_eq!(status.status, ExecutionStatus::Running);
    assert_eq!(status.current_phase, HarnessPhase::Model);
    assert!(status.ended_at.is_none());
}

#[tokio::test]
async fn shared_reaper_preserves_live_lease_and_settles_expired_lease() {
    let backend = MemoryStorage::new();
    let scoped = backend.for_scope(&Scope::local()).unwrap();
    let owner = DriverStatusStore::new(Arc::clone(scoped.documents()));
    let reaper = DriverStatusStore::new(Arc::clone(scoped.documents()));
    let live_id = mint_run_id();
    let mut live = HarnessRunStatus::new(live_id.clone(), ComponentId::new("model".into()));
    live.mark_running(HarnessPhase::Model);
    live.lease_owner = Some("healthy-owner".into());
    live.lease_epoch = 4;
    live.lease_expires_at = Some(SystemTime::now() + Duration::from_secs(60));
    owner.put_status(live).await.unwrap();
    let expired_id = seed_status(&owner, ExecutionStatus::Running).await;

    assert_eq!(reap_orphaned_runs(&reaper).await, 1);
    assert_eq!(
        owner
            .get_status(live_id.as_str())
            .await
            .unwrap()
            .unwrap()
            .status,
        ExecutionStatus::Running
    );
    let expired = owner.get_status(&expired_id).await.unwrap().unwrap();
    assert_eq!(expired.status, ExecutionStatus::Cancelled);
    assert_eq!(expired.lease_owner, None);
}

#[tokio::test]
async fn status_lease_epoch_fences_heartbeat_takeover_and_finish() {
    let store = store();
    let now = SystemTime::UNIX_EPOCH + Duration::from_secs(100);
    let expires = now + Duration::from_secs(5);
    let run_id = mint_run_id();
    let mut initial = HarnessRunStatus::new(run_id.clone(), ComponentId::new("model".into()));
    initial.mark_running(HarnessPhase::Model);
    let first = store
        .claim_status(initial, "owner-a", now, expires)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(first.lease_epoch, 1);
    assert!(
        store
            .renew_status_lease(
                run_id.as_str(),
                "owner-a",
                1,
                now + Duration::from_secs(1),
                now + Duration::from_secs(8),
            )
            .await
            .unwrap()
    );
    assert!(
        !store
            .renew_status_lease(
                run_id.as_str(),
                "owner-a",
                1,
                now + Duration::from_secs(9),
                now + Duration::from_secs(12),
            )
            .await
            .unwrap()
    );

    let mut takeover = HarnessRunStatus::new(run_id.clone(), ComponentId::new("model".into()));
    takeover.mark_running(HarnessPhase::Model);
    let second = store
        .claim_status(
            takeover,
            "owner-b",
            now + Duration::from_secs(9),
            now + Duration::from_secs(20),
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(second.lease_epoch, 2);

    let mut stale_finish = first;
    stale_finish.mark_completed();
    assert!(
        !store
            .finish_status_lease(stale_finish, "owner-a", 1, now + Duration::from_secs(9))
            .await
            .unwrap()
    );
    let mut current_finish = second;
    current_finish.mark_completed();
    assert!(
        store
            .finish_status_lease(current_finish, "owner-b", 2, now + Duration::from_secs(10))
            .await
            .unwrap()
    );
}

#[test]
fn mark_cancelled_sets_terminal_fields() {
    let mut status = HarnessRunStatus::new(mint_run_id(), ComponentId::new("m".to_string()));
    status.mark_running(HarnessPhase::Model);
    status.mark_cancelled("why");
    assert_eq!(status.status, ExecutionStatus::Cancelled);
    assert_eq!(status.current_phase, HarnessPhase::Done);
    assert_eq!(status.error.as_deref(), Some("why"));
    assert!(status.ended_at.is_some());
}

mod sealing {
    use super::*;
    use crate::events::AgentEvent;
    use crate::ids::RunId;
    use crate::observability::{AgentObservation, HarnessEventJournal, StoreEventJournal};
    use crate::store::{AppendStore, DriverAppendStore};

    async fn journal_one_event(journal: &dyn HarnessEventJournal, run: &str) {
        let id = RunId::new(run);
        journal
            .append(AgentObservation {
                event_id: format!("{run}-evt-0").into(),
                run_id: id.clone(),
                parent_run_id: None,
                root_run_id: id,
                offset: 0,
                ts_ms: 1,
                event: AgentEvent::StreamClosed,
            })
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn reaping_seals_the_journal_of_each_reaped_run_only() {
        let scoped = MemoryStorage::new().for_scope(&Scope::local()).unwrap();
        let status = DriverStatusStore::new(Arc::clone(scoped.documents()));
        let journal = StoreEventJournal::new(Arc::new(DriverAppendStore::journal(Arc::clone(
            scoped.streams(),
        ))) as Arc<dyn AppendStore>);

        let orphan = seed_status(&status, ExecutionStatus::Running).await;
        let no_journal = seed_status(&status, ExecutionStatus::Pending).await;
        let finished = mint_run_id();
        let mut done = HarnessRunStatus::new(finished.clone(), ComponentId::new("m".to_string()));
        done.mark_running(HarnessPhase::Model);
        done.mark_completed();
        status.put_status(done).await.unwrap();
        journal_one_event(&journal, &orphan).await;
        journal_one_event(&journal, finished.as_str()).await;

        assert_eq!(reap_orphaned_runs_and_seal(&status, &journal).await, 2);

        let sealed = |name: String| {
            let streams = Arc::clone(scoped.streams());
            async move {
                streams
                    .info(&format!("harness.journal/{name}"))
                    .await
                    .unwrap()
                    .map(|info| info.sealed_ms.is_some())
            }
        };
        assert_eq!(sealed(orphan.clone()).await, Some(true));
        assert_eq!(sealed(finished.as_str().to_owned()).await, Some(false));
        assert_eq!(sealed(no_journal).await, None, "no stream was invented");
        assert_eq!(journal.read_from(&orphan, 0).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn a_journal_that_fails_to_seal_does_not_stop_the_sweep() {
        struct Refusing;
        #[async_trait::async_trait]
        impl HarnessEventJournal for Refusing {
            async fn append(&self, _: AgentObservation) -> crate::error::Result<u64> {
                Ok(0)
            }
            async fn read_from(
                &self,
                _: &str,
                _: u64,
            ) -> crate::error::Result<Vec<AgentObservation>> {
                Ok(Vec::new())
            }
            async fn seal_run(&self, _: &str) -> crate::error::Result<()> {
                Err(crate::error::TinyAgentsError::Storage("no".into()))
            }
        }
        let status = store();
        let a = seed_status(&status, ExecutionStatus::Running).await;
        let b = seed_status(&status, ExecutionStatus::Running).await;
        assert_eq!(reap_orphaned_runs_and_seal(&status, &Refusing).await, 2);
        for id in [a, b] {
            let s = status.get_status(&id).await.unwrap().unwrap();
            assert_eq!(s.status, ExecutionStatus::Cancelled);
        }
    }
}
