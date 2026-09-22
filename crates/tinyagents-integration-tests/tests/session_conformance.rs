//! Session-persistence conformance for the run ledger and transcript history.
//!
//! Mirrors `persistence_conformance.rs`'s pattern for the graph checkpointer:
//! the same contract suite is run against every backend the crate owns, so a
//! defect in one cannot hide behind another. The run ledger has exactly one
//! backend (SQLite via a workspace directory), so the suite is run against
//! two independent workspaces to pin that its state does not leak through the
//! path-keyed connection cache. `TranscriptHistory` genuinely has two
//! implementations — the production file-backed one and the in-memory test
//! double — and both are certified here.

use tinyagents_session::testkit::InMemoryTranscriptHistory;
use tinyagents_session::testkit::conformance::{
    run_ledger_conformance, transcript_history_conformance,
};
use tinyagents_session::transcript::{FileTranscriptHistory, TranscriptMeta};

#[test]
fn run_ledger_satisfies_the_conformance_suite_on_a_fresh_workspace() {
    let dir = tempfile::tempdir().unwrap();
    run_ledger_conformance(dir.path());
}

#[test]
fn run_ledger_satisfies_the_conformance_suite_on_a_second_independent_workspace() {
    // A second, independent workspace must behave identically and must not
    // observe any state from the first (the connection cache is keyed by
    // resolved database path, so this also pins that key derivation is
    // actually per-workspace).
    let dir = tempfile::tempdir().unwrap();
    run_ledger_conformance(dir.path());
}

fn contract_meta() -> TranscriptMeta { session_id: None, parent_session_id: None,
    TranscriptMeta { session_id: None, parent_session_id: None,
        agent_name: "contract-agent".to_string(),
        agent_id: Some("contract-agent-id".to_string()),
        agent_type: Some("root".to_string()),
        dispatcher: "native".to_string(),
        provider: None,
        model: None,
        created: String::new(),
        updated: String::new(),
        turn_count: 0,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: None,
        task_id: None,
    }
}

#[test]
fn file_transcript_history_satisfies_the_conformance_suite() {
    let dir = tempfile::tempdir().unwrap();
    let history = FileTranscriptHistory::new(dir.path(), "session", contract_meta()).unwrap();
    transcript_history_conformance(&history);
}

#[test]
fn in_memory_transcript_history_satisfies_the_conformance_suite() {
    let history = InMemoryTranscriptHistory::new("contract", contract_meta());
    transcript_history_conformance(&history);
}
