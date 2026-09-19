//! Reusable session-persistence **conformance** (contract) suites.
//!
//! Two suites, one per storage seam this crate owns:
//!
//! * [`run_ledger_conformance`] exercises the run-ledger's free-function API
//!   (append/read, parent-run lineage, idempotent re-import of the same run
//!   id, event append + ordered listing, and a concurrent lease-claim race)
//!   against a workspace directory. The ledger has exactly one backend today
//!   ([`crate::store`]'s SQLite-backed connection cache, keyed by workspace
//!   path) — there is no second implementation to parameterize this suite
//!   over — so it is run here against two *independent* workspaces to pin
//!   that the contract holds without any shared, path-keyed cache state
//!   leaking between them.
//! * [`transcript_history_conformance`] exercises
//!   [`crate::transcript::TranscriptHistory`]/[`crate::transcript::TranscriptRead`],
//!   which — unlike the ledger — genuinely has two independent
//!   implementations: the production
//!   [`FileTranscriptHistory`](crate::transcript::FileTranscriptHistory) and
//!   the in-memory [`super::InMemoryTranscriptHistory`] double. Both are
//!   certified against the same suite in this crate's own tests.
//!
//! Each function panics with a descriptive message on the first violation, so
//! call it from a `#[test]`.

use std::path::Path;

use serde_json::json;

use crate::run_ledger::{
    AgentRunKind, AgentRunListRequest, AgentRunStatus, AgentRunUpsert, RunEventAppend,
    RunEventListRequest, WorkflowLeaseClaim, WorkflowRunStatus, WorkflowRunUpsert,
    append_run_event, get_agent_run, list_agent_runs, list_recent_run_events,
    try_claim_workflow_run, upsert_agent_run, upsert_workflow_run,
};
use crate::transcript::{TranscriptHistory, TranscriptMessage, TranscriptMeta, TranscriptTurn};

fn contract_run(id: &str, parent: Option<&str>) -> AgentRunUpsert {
    AgentRunUpsert {
        id: id.to_string(),
        kind: AgentRunKind::Subagent,
        parent_run_id: parent.map(str::to_string),
        parent_thread_id: Some("contract-thread".to_string()),
        agent_id: Some("contract-agent".to_string()),
        status: AgentRunStatus::Running,
        prompt_ref: None,
        worker_thread_id: None,
        task_board_id: None,
        task_card_id: None,
        checkpoint_path: None,
        checkpoint: None,
        summary: None,
        error: None,
        metadata: json!({}),
        started_at: None,
        completed_at: None,
    }
}

/// Runs the run-ledger contract against the SQLite-backed store rooted at
/// `workspace_dir`.
///
/// Covers append/read (`upsert_agent_run` + `get_agent_run`), parent-run
/// lineage (`list_agent_runs` filtered by `parent_run_id`), idempotent
/// re-import (re-upserting the same run id updates the row in place rather
/// than duplicating it), event append + ordered listing
/// (`append_run_event` + `list_recent_run_events`), and a concurrent
/// lease-claim race (`try_claim_workflow_run` from two racing threads,
/// exactly one of which must win).
pub fn run_ledger_conformance(workspace_dir: &Path) {
    // ── Append / read ────────────────────────────────────────────────
    let created = upsert_agent_run(workspace_dir, contract_run("run-a", None))
        .expect("upsert run-a succeeds");
    assert_eq!(created.id, "run-a", "upsert returns the row it just wrote");
    assert_eq!(created.status, AgentRunStatus::Running);

    let fetched = get_agent_run(workspace_dir, "run-a")
        .expect("get run-a succeeds")
        .expect("run-a exists after being upserted");
    assert_eq!(fetched.id, "run-a");

    assert!(
        get_agent_run(workspace_dir, "does-not-exist")
            .expect("get on a missing id succeeds")
            .is_none(),
        "a missing run id must resolve to None, not an error"
    );

    // ── Lineage ──────────────────────────────────────────────────────
    upsert_agent_run(workspace_dir, contract_run("run-b", Some("run-a")))
        .expect("upsert run-b succeeds");
    upsert_agent_run(workspace_dir, contract_run("run-c", None)).expect("upsert run-c succeeds");
    let children = list_agent_runs(
        workspace_dir,
        &AgentRunListRequest {
            parent_run_id: Some("run-a".to_string()),
            ..AgentRunListRequest::default()
        },
    )
    .expect("list children of run-a succeeds");
    assert_eq!(
        children.runs.len(),
        1,
        "the parent_run_id filter must return exactly the lineage under run-a, \
         not every run in the workspace"
    );
    assert_eq!(children.runs[0].id, "run-b");

    // ── Idempotent re-import ─────────────────────────────────────────
    // Re-upserting the same run id (as a resumed host replaying its own
    // durable state would) must update the existing row, not create a
    // second one.
    let mut resumed = contract_run("run-a", None);
    resumed.status = AgentRunStatus::Completed;
    resumed.summary = Some("finished".to_string());
    upsert_agent_run(workspace_dir, resumed).expect("re-upsert run-a succeeds");

    let after = get_agent_run(workspace_dir, "run-a")
        .expect("get run-a succeeds")
        .expect("run-a still exists");
    assert_eq!(after.status, AgentRunStatus::Completed);
    assert_eq!(after.summary.as_deref(), Some("finished"));

    let all = list_agent_runs(workspace_dir, &AgentRunListRequest::default())
        .expect("list all runs succeeds");
    assert_eq!(
        all.runs.iter().filter(|run| run.id == "run-a").count(),
        1,
        "re-importing the same run id must update the row in place, never duplicate it"
    );

    // ── Event append + ordered listing ──────────────────────────────
    for (event_type, n) in [("step", 1), ("step", 2), ("finished", 3)] {
        append_run_event(
            workspace_dir,
            RunEventAppend {
                run_id: "run-a".to_string(),
                event_type: event_type.to_string(),
                payload: json!({ "n": n }),
            },
        )
        .expect("append_run_event succeeds");
    }
    let events = list_recent_run_events(
        workspace_dir,
        &RunEventListRequest {
            run_id: "run-a".to_string(),
            after_sequence: None,
            limit: None,
        },
    )
    .expect("list_recent_run_events succeeds");
    assert_eq!(events.events.len(), 3, "every appended event is listed");
    let sequence_ns: Vec<i64> = events
        .events
        .iter()
        .map(|event| event.payload["n"].as_i64().expect("n is present"))
        .collect();
    assert_eq!(
        sequence_ns,
        vec![1, 2, 3],
        "events list back in append (sequence) order"
    );

    // Polling with an `after_sequence` cursor returns only what is newer.
    let cursor = events.events[0].sequence;
    let after = list_recent_run_events(
        workspace_dir,
        &RunEventListRequest {
            run_id: "run-a".to_string(),
            after_sequence: Some(cursor),
            limit: None,
        },
    )
    .expect("cursor-bounded list succeeds");
    assert_eq!(
        after.events.len(),
        2,
        "the cursor excludes what was already seen"
    );

    // ── Concurrency ──────────────────────────────────────────────────
    // A workflow lease is the ledger's one compare-and-swap primitive: two
    // concurrent claims for the same id must serialize so exactly one wins.
    upsert_workflow_run(
        workspace_dir,
        WorkflowRunUpsert {
            id: "workflow-contract".to_string(),
            definition_id: "contract".to_string(),
            parent_thread_id: None,
            input: json!({}),
            phase_states: json!({}),
            child_run_ids: vec![],
            status: WorkflowRunStatus::Running,
            summary: None,
            started_at: None,
            completed_at: None,
        },
    )
    .expect("seed workflow run succeeds");

    let workspace = workspace_dir.to_path_buf();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let mut workers = Vec::new();
    for owner in ["first", "second"] {
        let workspace = workspace.clone();
        let barrier = barrier.clone();
        workers.push(std::thread::spawn(move || {
            barrier.wait();
            try_claim_workflow_run(
                &workspace,
                "workflow-contract",
                owner,
                chrono::Duration::minutes(1),
            )
            .expect("try_claim_workflow_run does not error")
        }));
    }
    let claims: Vec<_> = workers.into_iter().map(|w| w.join().unwrap()).collect();
    let acquired = claims
        .iter()
        .filter(|claim| matches!(claim, WorkflowLeaseClaim::Acquired(_)))
        .count();
    assert_eq!(
        acquired, 1,
        "exactly one of two concurrent claims for the same lease must win"
    );
}

fn contract_message(role: &str, content: &str) -> TranscriptMessage {
    TranscriptMessage::new(role, content)
}

/// Projects a message list onto its `(role, content)` pairs.
///
/// Used to compare the *logical view* `append_turn`/`replace` establish
/// against a freshly-constructed expectation. A raw [`TranscriptMessage`]
/// equality would be backend-specific here: a file-backed history
/// legitimately stamps `preserve_request_id: true` on every message it reads
/// back (a row read from a transcript owns its recorded correlation id — see
/// `jsonl::message_from_line`), while a freshly-built [`contract_message`]
/// defaults it to `false`. That divergence is a correct, documented
/// round-trip behavior of the file format, not a logical-view difference, so
/// the conformance suite must not assert byte-identical metadata across
/// backends — only that the same messages, in the same order, are present.
fn content_view(messages: &[TranscriptMessage]) -> Vec<(String, String)> {
    messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect()
}

fn contract_meta() -> TranscriptMeta {
    TranscriptMeta {
        agent_name: "contract-agent".to_string(),
        agent_id: Some("contract-agent-id".to_string()),
        agent_type: Some("root".to_string()),
        dispatcher: "native".to_string(),
        provider: Some("contract-provider".to_string()),
        model: Some("contract-model".to_string()),
        created: "2026-01-01T00:00:00Z".to_string(),
        updated: "2026-01-01T00:00:00Z".to_string(),
        turn_count: 0,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: Some("contract-thread".to_string()),
        task_id: None,
    }
}

/// Runs the [`TranscriptHistory`] contract against `history`.
///
/// Covers: a fresh handle has no session yet; `append` grows the logical set
/// one message at a time; `append_turn` sets the logical set to exactly its
/// `next` argument (the extension-vs-compaction *diff* is a storage-format
/// optimization some backends make and others do not, but the resulting
/// logical view must always equal `next`); idempotent re-import (calling
/// `append_turn` again with the same `next` — as a resumed host replaying its
/// last turn would — leaves the logical set unchanged, never duplicated);
/// `replace` overwrites the logical set outright; and `clear` empties it
/// without erroring on a handle that was never written.
pub fn transcript_history_conformance(history: &dyn TranscriptHistory) {
    // ── Fresh handle ─────────────────────────────────────────────────
    assert!(
        history
            .read_session()
            .expect("read_session on a fresh handle succeeds")
            .is_none(),
        "a handle that was never written must report no session yet"
    );
    assert!(
        history
            .messages()
            .expect("messages on a fresh handle succeeds")
            .is_empty(),
        "a fresh handle has no messages"
    );
    // Clearing a handle that was never written must be a no-op, not an error.
    history
        .clear()
        .expect("clearing a never-written handle succeeds");

    // ── Append ───────────────────────────────────────────────────────
    history
        .append(contract_message("user", "hello"))
        .expect("append succeeds");
    history
        .append(contract_message("assistant", "hi there"))
        .expect("append succeeds");
    let after_appends = history.messages().expect("messages succeeds");
    assert_eq!(
        after_appends.len(),
        2,
        "each append grows the logical set by one"
    );
    assert_eq!(after_appends[0].content, "hello");
    assert_eq!(after_appends[1].content, "hi there");
    assert!(
        history
            .read_session()
            .expect("read_session succeeds")
            .is_some(),
        "a written handle reports a session"
    );

    // ── append_turn: extension ──────────────────────────────────────
    let meta = contract_meta();
    let turn1_next = vec![
        contract_message("user", "hello"),
        contract_message("assistant", "hi there"),
        contract_message("user", "and then?"),
    ];
    history
        .append_turn(TranscriptTurn {
            prev: &after_appends,
            next: &turn1_next,
            meta: &meta,
            turn_usage: None,
            request_id: Some("turn-1"),
        })
        .expect("append_turn (extension) succeeds");
    assert_eq!(
        history.messages().expect("messages succeeds"),
        turn1_next,
        "append_turn's logical view is exactly its `next` argument"
    );

    // ── append_turn: idempotent re-import ────────────────────────────
    // A resumed host replaying the same last turn calls `append_turn` again
    // with the same logical set; the result must be unchanged, not a
    // duplicated tail.
    history
        .append_turn(TranscriptTurn {
            prev: &turn1_next,
            next: &turn1_next,
            meta: &meta,
            turn_usage: None,
            request_id: Some("turn-1"),
        })
        .expect("re-importing the same turn succeeds");
    assert_eq!(
        history.messages().expect("messages succeeds"),
        turn1_next,
        "re-importing an already-persisted turn must not duplicate it"
    );

    // ── replace ──────────────────────────────────────────────────────
    let reduced = vec![contract_message("user", "compacted summary")];
    history.replace(&reduced).expect("replace succeeds");
    assert_eq!(
        history.messages().expect("messages succeeds"),
        reduced,
        "replace overwrites the logical set outright"
    );

    // ── clear ────────────────────────────────────────────────────────
    history.clear().expect("clear succeeds");
    assert!(
        history.messages().expect("messages succeeds").is_empty(),
        "clear empties the logical set"
    );
}
