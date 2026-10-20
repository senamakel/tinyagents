//! Session-persistence conformance for the run ledger and transcript history.
//!
//! Mirrors `persistence_conformance.rs`'s pattern for the graph checkpointer:
//! the same contract suite is run against every backend the crate owns, so a
//! defect in one cannot hide behind another. The run ledger is certified on the
//! memory and sqlite drivers. `TranscriptHistory` genuinely has two
//! implementations — the production driver-backed one and the in-memory test
//! double — and both are certified here.

mod common;

use std::sync::Arc;

use common::storage;
use tinyagents_session::run_ledger::drivers::DriverRunLedger;
use tinyagents_session::testkit::InMemoryTranscriptHistory;
use tinyagents_session::testkit::conformance::{
    run_ledger_conformance, transcript_history_conformance,
};
use tinyagents_session::transcript::TranscriptMeta;

#[tokio::test]
async fn run_ledger_satisfies_the_conformance_suite_on_memory() {
    run_ledger_conformance(Arc::new(DriverRunLedger::new(storage::memory()))).await;
}

#[tokio::test]
async fn run_ledger_satisfies_the_conformance_suite_on_sqlite() {
    let dir = tempfile::tempdir().unwrap();
    let ledger = DriverRunLedger::new(storage::sqlite(dir.path(), "ledger.sqlite"));
    run_ledger_conformance(Arc::new(ledger)).await;
}

fn contract_meta() -> TranscriptMeta {
    TranscriptMeta {
        session_id: None,
        parent_session_id: None,
        agent_name: "contract-agent".to_string(),
        agent_id: Some("contract-agent-id".to_string()),
        agent_type: Some("root".to_string()),
        dispatcher: "native".to_string(),
        provider: None,
        model: None,
        created: String::new(),
        updated: String::new(),
        turn_count: 0,
        prefix_message_count: None,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: None,
        task_id: None,
    }
}

#[test]
fn driver_transcript_history_satisfies_the_conformance_suite() {
    use tinyagents_session::port::{DriverSessionStores, SessionStoreProvider};
    let stores = DriverSessionStores::new(storage::fixed_factory(storage::memory())).unwrap();
    let history = stores
        .for_agent("contract")
        .transcripts
        .open_stem("session", contract_meta())
        .unwrap();
    transcript_history_conformance(history.as_ref());
}

#[test]
fn in_memory_transcript_history_satisfies_the_conformance_suite() {
    let history = InMemoryTranscriptHistory::new("contract", contract_meta());
    transcript_history_conformance(&history);
}
