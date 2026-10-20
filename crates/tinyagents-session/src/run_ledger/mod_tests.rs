use crate::testkit::MemoryStorage;
use std::sync::Arc;

use tinyagents_harness::tool::{
    ToolEffectLedger, ToolEffectSettle as HarnessSettle, ToolEffectStart as HarnessStart,
    ToolEffectStatus as HarnessStatus,
};
use tinyagents_storage_kit::{Scope, ScopedStorage};
use tinystoragedrivers_api::StorageBackend;

use super::*;

fn ledger_on(backend: &dyn StorageBackend, scope: &str) -> Arc<dyn RunLedgerStore> {
    let scope = Scope::new(scope).unwrap();
    Arc::new(DriverRunLedger::new(ScopedStorage::new(
        backend.for_scope(&scope).unwrap(),
        scope,
    )))
}

fn started(run: &str, call: &str) -> HarnessStart {
    HarnessStart {
        run_id: run.into(),
        call_id: call.into(),
        tool: "send_email".into(),
        idempotency_key: format!("key-{call}"),
        effect_summary: Some("to bob".into()),
    }
}

#[tokio::test]
async fn run_ledger_tool_effects_implements_the_harness_ledger_trait() {
    let ledger = RunLedgerToolEffects::new(ledger_on(&MemoryStorage::new(), "agent"));

    ledger.started(started("run-1", "call-1")).await.unwrap();
    ledger.started(started("run-1", "call-2")).await.unwrap();
    ledger.started(started("run-2", "elsewhere")).await.unwrap();

    let unresolved = ledger.unresolved("run-1").await.unwrap();
    assert_eq!(unresolved.len(), 2, "only the run's own calls");
    assert!(
        unresolved
            .iter()
            .all(|e| e.status == HarnessStatus::Started)
    );
    let call_one = unresolved.iter().find(|e| e.call_id == "call-1").unwrap();
    assert_eq!(call_one.tool, "send_email");
    assert_eq!(call_one.idempotency_key.as_deref(), Some("key-call-1"));
    assert_eq!(call_one.effect_summary.as_deref(), Some("to bob"));
    assert!(call_one.settled_at.is_none());

    ledger
        .settled(HarnessSettle {
            run_id: "run-1".into(),
            call_id: "call-1".into(),
            status: HarnessStatus::Completed,
            effect_summary: Some("sent".into()),
        })
        .await
        .unwrap();
    let after = ledger.unresolved("run-1").await.unwrap();
    assert_eq!(after.len(), 1, "a settled call is no longer unresolved");
    assert_eq!(after[0].call_id, "call-2");

    // A call left `started` across a resume settles as interrupted.
    ledger
        .settled(HarnessSettle {
            run_id: "run-1".into(),
            call_id: "call-2".into(),
            status: HarnessStatus::Interrupted,
            effect_summary: None,
        })
        .await
        .unwrap();
    assert!(ledger.unresolved("run-1").await.unwrap().is_empty());
}

#[tokio::test]
async fn tool_effects_survive_store_reconstruction_and_stay_per_scope() {
    let backend = MemoryStorage::new();
    let ledger = RunLedgerToolEffects::new(ledger_on(&backend, "agent"));
    ledger.started(started("run-1", "call-1")).await.unwrap();
    let reopened = RunLedgerToolEffects::new(ledger_on(&backend, "agent"));
    let unresolved = reopened.unresolved("run-1").await.unwrap();
    assert_eq!(
        unresolved.len(),
        1,
        "the in-flight call is still there after a restart"
    );
    let peer = RunLedgerToolEffects::new(ledger_on(&backend, "peer"));
    assert!(peer.unresolved("run-1").await.unwrap().is_empty());
}

#[tokio::test]
async fn the_ledger_conformance_suite_passes_on_independent_memory_backends() {
    use crate::testkit::conformance::run_ledger_conformance;
    run_ledger_conformance(ledger_on(&MemoryStorage::new(), "agent")).await;
    let backend = MemoryStorage::new();
    run_ledger_conformance(ledger_on(&backend, "agent")).await;
}
