//! Owner-level tests for backfill and retention of harness run journals.

use super::*;
use crate::error::{Result, TinyAgentsError};
use crate::events::HarnessRunStatus;
use crate::ids::{ComponentId, ExecutionStatus, HarnessPhase};
use crate::observability::{DriverStatusStore, HarnessStatusStore, mint_run_id};
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;
use tinyagents_storage_kit::{Scope, StorageBackend, StreamStore};
use tinystoragedrivers_testkit::{Failing, FaultPlan, MemoryBackend};
use tinystoragedrivers_types::{
    Capabilities, ErrorKind, Result as StorageResult, SealOutcome, StreamEntry, StreamInfo,
};

fn setup() -> (DriverStatusStore, Arc<dyn StreamStore>) {
    let scoped = MemoryBackend::new()
        .for_scope(&Scope::new("local").unwrap())
        .unwrap();
    (
        DriverStatusStore::new(Arc::clone(scoped.documents())),
        Arc::clone(scoped.streams()),
    )
}

async fn status(store: &dyn HarnessStatusStore, state: ExecutionStatus) -> String {
    let run_id = mint_run_id();
    let mut status = HarnessRunStatus::new(run_id.clone(), ComponentId::new("test".to_owned()));
    match state {
        ExecutionStatus::Completed => status.mark_completed(),
        ExecutionStatus::Running => status.mark_running(HarnessPhase::Model),
        other => panic!("expected completed or running status, got {other:?}"),
    }
    store.put_status(status).await.unwrap();
    run_id.as_str().to_owned()
}

async fn journal(streams: &dyn StreamStore, run_id: &str) {
    streams
        .append(
            &format!("harness.journal/{run_id}"),
            serde_json::json!({ "event": "test" }),
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn backfills_only_inactive_journals_and_is_idempotent() {
    let (status_store, streams) = setup();
    let active = status(&status_store, ExecutionStatus::Running).await;
    let inactive = status(&status_store, ExecutionStatus::Completed).await;
    journal(streams.as_ref(), &active).await;
    journal(streams.as_ref(), &inactive).await;
    journal(streams.as_ref(), "not-active").await;
    let now_ms = 10_000;

    let first = maintain_run_journals(&status_store, streams.as_ref(), 0, now_ms, true)
        .await
        .unwrap();
    assert_eq!(first.backfilled, 2);
    assert_eq!(
        streams
            .info(&format!("harness.journal/{active}"))
            .await
            .unwrap()
            .unwrap()
            .sealed_ms,
        None,
        "an active run's journal stays open"
    );
    assert!(
        streams
            .info(&format!("harness.journal/{inactive}"))
            .await
            .unwrap()
            .unwrap()
            .sealed_ms
            .is_some()
    );
    assert!(
        streams
            .info("harness.journal/not-active")
            .await
            .unwrap()
            .unwrap()
            .sealed_ms
            .is_some()
    );

    let again = maintain_run_journals(&status_store, streams.as_ref(), 0, now_ms, true)
        .await
        .unwrap();
    assert_eq!(again.backfilled, 0);
    assert_eq!(
        again.deleted, 0,
        "zero retention keeps sealed journals forever"
    );
}

#[tokio::test]
async fn retention_deletes_only_after_the_exact_cutoff() {
    let (status_store, streams) = setup();
    journal(streams.as_ref(), "retained-at-cutoff").await;
    streams
        .seal("harness.journal/retained-at-cutoff")
        .await
        .unwrap();
    let sealed_ms = streams
        .info("harness.journal/retained-at-cutoff")
        .await
        .unwrap()
        .unwrap()
        .sealed_ms
        .unwrap();
    let retention_ms = 5_000;

    let at_cutoff = maintain_run_journals(
        &status_store,
        streams.as_ref(),
        retention_ms,
        sealed_ms + retention_ms as i64,
        false,
    )
    .await
    .unwrap();
    assert_eq!(at_cutoff.deleted, 0);
    assert!(
        streams
            .info("harness.journal/retained-at-cutoff")
            .await
            .unwrap()
            .is_some()
    );

    let past_cutoff = maintain_run_journals(
        &status_store,
        streams.as_ref(),
        retention_ms,
        sealed_ms + retention_ms as i64 + 1,
        false,
    )
    .await
    .unwrap();
    assert_eq!(past_cutoff.deleted, 1);
    assert!(
        streams
            .info("harness.journal/retained-at-cutoff")
            .await
            .unwrap()
            .is_none()
    );
    let repeated = maintain_run_journals(
        &status_store,
        streams.as_ref(),
        retention_ms,
        sealed_ms + retention_ms as i64 + 1,
        false,
    )
    .await
    .unwrap();
    assert_eq!(repeated.deleted, 0);
}

#[tokio::test]
async fn active_run_journals_are_never_removed_by_retention() {
    let (status_store, streams) = setup();
    let active = status(&status_store, ExecutionStatus::Running).await;
    let name = format!("harness.journal/{active}");
    journal(streams.as_ref(), &active).await;
    streams.seal(&name).await.unwrap();

    let report = maintain_run_journals(&status_store, streams.as_ref(), 1, i64::MAX, true)
        .await
        .unwrap();

    assert_eq!(report.backfilled, 0);
    assert_eq!(report.deleted, 0);
    assert!(streams.info(&name).await.unwrap().is_some());
}

struct RefusingStatusStore;

#[async_trait::async_trait]
impl HarnessStatusStore for RefusingStatusStore {
    async fn put_status(&self, _: HarnessRunStatus) -> Result<()> {
        Ok(())
    }

    async fn get_status(&self, _: &str) -> Result<Option<HarnessRunStatus>> {
        Ok(None)
    }

    async fn list_by_thread(&self, _: &str) -> Result<Vec<HarnessRunStatus>> {
        Ok(Vec::new())
    }

    async fn list_active(&self) -> Result<Vec<HarnessRunStatus>> {
        Err(TinyAgentsError::Storage("status listing failed".to_owned()))
    }
}

#[tokio::test]
async fn status_enumeration_failures_are_propagated() {
    let (_, streams) = setup();
    let error = maintain_run_journals(&RefusingStatusStore, streams.as_ref(), 0, 0, true)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("status listing failed"));
}

#[tokio::test]
async fn stream_listing_failures_are_propagated() {
    let (status_store, streams) = setup();
    let faults = FaultPlan::new();
    faults.fail_operation_nth("streams.streams", 1, ErrorKind::Backend);
    let streams: Arc<dyn StreamStore> = Arc::new(Failing::with_plan(streams, faults));

    let error = maintain_run_journals(&status_store, streams.as_ref(), 0, 0, true)
        .await
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("injected failure for streams.streams")
    );
}

#[derive(Debug)]
struct RefusingConditionalDelete {
    inner: Arc<dyn StreamStore>,
}

#[async_trait]
impl StreamStore for RefusingConditionalDelete {
    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }

    async fn append(&self, stream: &str, value: Value) -> StorageResult<u64> {
        self.inner.append(stream, value).await
    }

    async fn append_batch(&self, stream: &str, values: Vec<Value>) -> StorageResult<u64> {
        self.inner.append_batch(stream, values).await
    }

    async fn read_window(
        &self,
        stream: &str,
        from: u64,
        limit: usize,
    ) -> StorageResult<Vec<StreamEntry>> {
        self.inner.read_window(stream, from, limit).await
    }

    async fn len(&self, stream: &str) -> StorageResult<u64> {
        self.inner.len(stream).await
    }

    async fn truncate_before(&self, stream: &str, offset: u64) -> StorageResult<u64> {
        self.inner.truncate_before(stream, offset).await
    }

    async fn delete_stream(&self, stream: &str) -> StorageResult<bool> {
        self.inner.delete_stream(stream).await
    }

    async fn streams(&self, prefix: &str) -> StorageResult<Vec<String>> {
        self.inner.streams(prefix).await
    }

    async fn seal(&self, stream: &str) -> StorageResult<SealOutcome> {
        self.inner.seal(stream).await
    }

    async fn delete_sealed_stream(&self, _stream: &str, _sealed_ms: i64) -> StorageResult<bool> {
        Ok(false)
    }

    async fn info(&self, stream: &str) -> StorageResult<Option<StreamInfo>> {
        self.inner.info(stream).await
    }
}

#[tokio::test]
async fn changed_sealed_incarnation_is_not_counted_as_deleted() {
    let (status_store, streams) = setup();
    journal(streams.as_ref(), "changed").await;
    streams.seal("harness.journal/changed").await.unwrap();
    let streams: Arc<dyn StreamStore> = Arc::new(RefusingConditionalDelete { inner: streams });

    let report = maintain_run_journals(&status_store, streams.as_ref(), 1, i64::MAX, false)
        .await
        .unwrap();

    assert_eq!(report.deleted, 0);
    assert!(
        streams
            .info("harness.journal/changed")
            .await
            .unwrap()
            .is_some()
    );
}
