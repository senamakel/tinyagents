//! Tests for the orphaned-run startup sweep.

use super::*;
use crate::events::HarnessRunStatus;
use crate::ids::{ComponentId, ExecutionStatus, HarnessPhase};
use crate::observability::{DriverStatusStore, mint_run_id};
use std::sync::Arc;
use tinyagents_storage_kit::{Scope, StorageBackend};
use tinystoragedrivers_testkit::MemoryBackend;

fn store() -> DriverStatusStore {
    let scoped = MemoryBackend::new()
        .for_scope(&Scope::new("local").unwrap())
        .unwrap();
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
        let scoped = MemoryBackend::new()
            .for_scope(&Scope::new("local").unwrap())
            .unwrap();
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
