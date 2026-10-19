//! Reusable session-persistence **conformance** (contract) suites.
//!
//! Two suites, one per storage seam this crate owns:
//!
//! * [`run_ledger_conformance`] exercises any
//!   [`RunLedgerStore`] (append/read, parent-run lineage, idempotent
//!   re-import of the same run id, event append + ordered listing, and the
//!   concurrent lease-claim and team-task-claim races). Every backend a host
//!   opens goes through the one driver implementation, so this suite is run
//!   here against the memory and SQLite drivers. FileStorage is tested against
//!   the operations its capability set supports (workflow CRUD and leases),
//!   and separately verifies that atomic agent/event/task operations return
//!   `Unsupported(ValidatedBatch)` without partial writes; its document driver
//!   does not provide the multi-document transaction boundary those operations
//!   require. The full MongoDB contracts run from the excluded
//!   `integration-proofs/run-ledger-mongodb` workspace because BSON enables
//!   `serde_json/preserve_order`, which must not alter the main workspace's
//!   shared feature graph.
//! * [`transcript_history_conformance`] exercises
//!   [`crate::transcript::TranscriptHistory`]/[`crate::transcript::TranscriptRead`],
//!   which — unlike the ledger — genuinely has two independent
//!   implementations: the production
//!   [`DriverTranscriptHistory`](crate::port::DriverTranscriptHistory) and
//!   the in-memory [`super::InMemoryTranscriptHistory`] double. Both are
//!   certified against the same suite in this crate's own tests.
//!
//! * [`session_store_conformance`] and
//!   [`session_store_isolation_conformance`] exercise a whole
//!   [`SessionStoreProvider`] — the port a host implements over its own
//!   database — against [`crate::port::InMemorySessionStores`] here, and
//!   against a host's provider in the host's own tests.
//!
//! Each function panics with a descriptive message on the first violation, so
//! call it from a `#[test]`.

use serde_json::json;

use crate::port::SessionStoreProvider;
use crate::run_ledger::{
    AgentRunKind, AgentRunListRequest, AgentRunStatus, AgentRunUpsert, AgentTeamTaskStatus,
    AgentTeamTaskUpsert, ClaimOutcome, RunEventAppend, RunEventListRequest, RunLedgerStore,
    WorkflowLeaseClaim, WorkflowRunStatus, WorkflowRunUpsert,
};
use crate::transcript::{
    SessionRef, TranscriptHistory, TranscriptMessage, TranscriptMeta, TranscriptPartial,
    TranscriptTurn,
};
use crate::turn_state::{TurnLifecycle, TurnState};

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
        checkpoint_path: None,
        checkpoint: None,
        summary: None,
        error: None,
        metadata: json!({}),
        started_at: None,
        completed_at: None,
    }
}

/// Runs the run-ledger contract against any [`RunLedgerStore`].
///
/// Covers append/read (`upsert_agent_run` + `get_agent_run`), parent-run
/// lineage (`list_agent_runs` filtered by `parent_run_id`), idempotent
/// re-import (re-upserting the same run id updates the row in place rather
/// than duplicating it), event append + ordered listing
/// (`append_run_event` + `list_recent_run_events`), a concurrent lease-claim
/// race (`try_claim_workflow_run` from racing tasks, exactly one of which
/// must win), and the atomic team-task claim race (exactly one member wins a
/// contested task).
///
/// The ledger is a port: certify a backend by passing a fresh, empty handle.
pub async fn run_ledger_conformance(ledger: std::sync::Arc<dyn RunLedgerStore>) {
    // ── Append / read ────────────────────────────────────────────────
    let created = ledger
        .upsert_agent_run(contract_run("run-a", None))
        .await
        .expect("upsert run-a succeeds");
    assert_eq!(created.id, "run-a", "upsert returns the row it just wrote");
    assert_eq!(created.status, AgentRunStatus::Running);

    let fetched = ledger
        .get_agent_run("run-a")
        .await
        .expect("get run-a succeeds")
        .expect("run-a exists after being upserted");
    assert_eq!(fetched.id, "run-a");

    assert!(
        ledger
            .get_agent_run("does-not-exist")
            .await
            .expect("get on a missing id succeeds")
            .is_none(),
        "a missing run id must resolve to None, not an error"
    );

    // ── Lineage ──────────────────────────────────────────────────────
    ledger
        .upsert_agent_run(contract_run("run-b", Some("run-a")))
        .await
        .expect("upsert run-b succeeds");
    ledger
        .upsert_agent_run(contract_run("run-c", None))
        .await
        .expect("upsert run-c succeeds");
    let children = ledger
        .list_agent_runs(&AgentRunListRequest {
            parent_run_id: Some("run-a".to_string()),
            ..AgentRunListRequest::default()
        })
        .await
        .expect("list children of run-a succeeds");
    assert_eq!(
        children.runs.len(),
        1,
        "the parent_run_id filter must return exactly the lineage under run-a, \
         not every run in the scope"
    );
    assert_eq!(children.runs[0].id, "run-b");

    // ── Idempotent re-import ─────────────────────────────────────────
    // Re-upserting the same run id (as a resumed host replaying its own
    // durable state would) must update the existing row, not create a
    // second one.
    let mut resumed = contract_run("run-a", None);
    resumed.status = AgentRunStatus::Completed;
    resumed.summary = Some("finished".to_string());
    ledger
        .upsert_agent_run(resumed)
        .await
        .expect("re-upsert run-a succeeds");

    let after = ledger
        .get_agent_run("run-a")
        .await
        .expect("get run-a succeeds")
        .expect("run-a still exists");
    assert_eq!(after.status, AgentRunStatus::Completed);
    assert_eq!(after.summary.as_deref(), Some("finished"));

    let all = ledger
        .list_agent_runs(&AgentRunListRequest::default())
        .await
        .expect("list all runs succeeds");
    assert_eq!(
        all.runs.iter().filter(|run| run.id == "run-a").count(),
        1,
        "re-importing the same run id must update the row in place, never duplicate it"
    );

    // ── Event append + ordered listing ──────────────────────────────
    for (event_type, n) in [("step", 1), ("step", 2), ("finished", 3)] {
        ledger
            .append_run_event(RunEventAppend {
                run_id: "run-a".to_string(),
                event_type: event_type.to_string(),
                payload: json!({ "n": n }),
            })
            .await
            .expect("append_run_event succeeds");
    }
    let events = ledger
        .list_recent_run_events(&RunEventListRequest {
            run_id: "run-a".to_string(),
            after_sequence: None,
            limit: None,
        })
        .await
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
    assert_eq!(
        events.events.iter().map(|e| e.sequence).collect::<Vec<_>>(),
        vec![1, 2, 3],
        "sequences are dense and start at one"
    );

    // Polling with an `after_sequence` cursor returns only what is newer.
    let cursor = events.events[0].sequence;
    let after = ledger
        .list_recent_run_events(&RunEventListRequest {
            run_id: "run-a".to_string(),
            after_sequence: Some(cursor),
            limit: None,
        })
        .await
        .expect("cursor-bounded list succeeds");
    assert_eq!(
        after.events.len(),
        2,
        "the cursor excludes what was already seen"
    );

    // ── Concurrency ──────────────────────────────────────────────────
    // A workflow lease is the ledger's compare-and-swap primitive: concurrent
    // claims for the same id must serialize so exactly one wins.
    ledger
        .upsert_workflow_run(WorkflowRunUpsert {
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
        })
        .await
        .expect("seed workflow run succeeds");

    let mut claimers = tokio::task::JoinSet::new();
    for owner in ["first", "second", "third"] {
        let ledger = ledger.clone();
        claimers.spawn(async move {
            ledger
                .try_claim_workflow_run("workflow-contract", owner, chrono::Duration::minutes(1))
                .await
                .expect("try_claim_workflow_run does not error")
        });
    }
    let mut acquired = 0;
    while let Some(claim) = claimers.join_next().await {
        if matches!(claim.expect("claim task"), WorkflowLeaseClaim::Acquired(_)) {
            acquired += 1;
        }
    }
    assert_eq!(
        acquired, 1,
        "exactly one of the concurrent claims for the same lease must win"
    );

    // Likewise a contested team task goes to exactly one member.
    ledger
        .upsert_agent_team_task(AgentTeamTaskUpsert {
            id: "contract-task".to_string(),
            team_id: "contract-team".to_string(),
            title: "contested".to_string(),
            objective: None,
            status: AgentTeamTaskStatus::Todo,
            owner_member_id: None,
            depends_on: vec![],
            gate_status: None,
            gate_reason: None,
            evidence: vec![],
            source_run_id: None,
            order_index: 0,
            created_at: None,
        })
        .await
        .expect("seed team task succeeds");
    let mut members = tokio::task::JoinSet::new();
    for n in 0..4 {
        let ledger = ledger.clone();
        members.spawn(async move {
            ledger
                .claim_agent_team_task(
                    "contract-team",
                    "contract-task",
                    &format!("member-{n}"),
                    &format!("token-{n}"),
                )
                .await
                .expect("claim_agent_team_task does not error")
        });
    }
    let mut claimed = 0;
    while let Some(outcome) = members.join_next().await {
        match outcome.expect("member task") {
            ClaimOutcome::Claimed(_) => claimed += 1,
            ClaimOutcome::AlreadyClaimed => {}
            other => panic!("unexpected claim outcome {other:?}"),
        }
    }
    assert_eq!(claimed, 1, "exactly one member wins a contested task");
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
        session_id: None,
        parent_session_id: None,
        agent_name: "contract-agent".to_string(),
        agent_id: Some("contract-agent-id".to_string()),
        agent_type: Some("root".to_string()),
        dispatcher: "native".to_string(),
        provider: Some("contract-provider".to_string()),
        model: Some("contract-model".to_string()),
        created: "2026-01-01T00:00:00Z".to_string(),
        updated: "2026-01-01T00:00:00Z".to_string(),
        turn_count: 0,
        prefix_message_count: None,
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
/// logical view — via [`content_view`], role/content only — must always equal
/// `next`; per-message bookkeeping like `preserve_request_id` is legitimately
/// backend-specific, since a file-backed history stamps it `true` on every
/// row it reads back); idempotent re-import (calling `append_turn` again with
/// the same `next` — as a resumed host replaying its last turn would —
/// leaves the logical set unchanged, never duplicated); `replace` overwrites
/// the logical set outright; and `clear` empties it without erroring on a
/// handle that was never written.
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
            tools: None,
        })
        .expect("append_turn (extension) succeeds");
    assert_eq!(
        content_view(&history.messages().expect("messages succeeds")),
        content_view(&turn1_next),
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
            tools: None,
        })
        .expect("re-importing the same turn succeeds");
    assert_eq!(
        content_view(&history.messages().expect("messages succeeds")),
        content_view(&turn1_next),
        "re-importing an already-persisted turn must not duplicate it"
    );

    // ── replace ──────────────────────────────────────────────────────
    let reduced = vec![contract_message("user", "compacted summary")];
    history.replace(&reduced).expect("replace succeeds");
    assert_eq!(
        content_view(&history.messages().expect("messages succeeds")),
        content_view(&reduced),
        "replace overwrites the logical set outright"
    );

    // ── clear ────────────────────────────────────────────────────────
    history.clear().expect("clear succeeds");
    assert!(
        history.messages().expect("messages succeeds").is_empty(),
        "clear empties the logical set"
    );
}

/// Runs the [`SessionStoreProvider`] contract against `provider`.
///
/// Uses agent `contract-store-agent` and thread `contract-store-thread`, so
/// run it against a provider with nothing stored for them. Covers, through
/// one agent's [`AgentStores`](crate::port::AgentStores):
///
/// - transcripts: a session has no transcript until written; a written turn
///   reads back through the session, its thread and its agent; compaction
///   generations chain and the head moves; an interrupted partial is
///   appended only once a root transcript exists, and never enters the
///   model-context replay;
/// - turn states: write, conditional write, lookup, settling, and
///   interrupted-marking with the on-disk store's semantics;
/// - key-value and journal: round trips.
///
/// Panics with a descriptive message on the first violation.
pub async fn session_store_conformance(provider: &dyn SessionStoreProvider) {
    let stores = provider.for_agent("contract-store-agent");

    // ── Transcripts ──────────────────────────────────────────────────
    let session = SessionRef::scoped("contract-store-thread", "contract-store-agent");
    let locator = &stores.transcripts;
    assert!(
        !locator.session_exists(&session),
        "a session has no transcript before its first write"
    );
    assert!(
        !locator
            .append_interrupted_partial(
                "contract-store-thread",
                None,
                &TranscriptPartial::new("too early"),
                None,
            )
            .expect("a partial with no transcript is not an error"),
        "a partial has nowhere to go before the first turn"
    );
    let mut meta = contract_meta();
    meta.thread_id = Some("contract-store-thread".to_string());
    meta.agent_id = Some("contract-agent".to_string());
    meta.session_id = Some(session.session_id());
    let history = locator
        .open_session(&session, meta.clone())
        .expect("open_session binds a handle");
    let turn = vec![
        contract_message("user", "book a table"),
        contract_message("assistant", "for how many?"),
    ];
    history
        .append_turn(TranscriptTurn {
            prev: &[],
            next: &turn,
            meta: &meta,
            turn_usage: None,
            request_id: Some("contract-turn-1"),
            tools: None,
        })
        .expect("append_turn succeeds");
    assert!(locator.session_exists(&session), "a written session exists");
    let read = locator
        .read_session_transcript(&session)
        .expect("a written session reads back")
        .read_session()
        .expect("read_session succeeds")
        .expect("a written session has a transcript");
    assert_eq!(content_view(&read.messages), content_view(&turn));
    let by_thread = locator
        .root_for_thread_scoped("contract-store-thread", Some("contract-agent"))
        .expect("the root transcript is found by its thread")
        .read_session()
        .expect("read_session succeeds")
        .expect("the thread's root has a transcript");
    assert_eq!(content_view(&by_thread.messages), content_view(&turn));
    let by_agent = locator
        .latest_for_agent("contract-agent")
        .expect("the root transcript is found by its agent")
        .read_session()
        .expect("read_session succeeds")
        .expect("the agent's transcript exists");
    assert_eq!(content_view(&by_agent.messages), content_view(&turn));
    assert!(
        locator
            .root_for_thread_scoped("contract-store-thread", Some("someone-else"))
            .is_none(),
        "a thread lookup scoped to another agent finds nothing"
    );
    assert!(
        locator.root_for_thread("no-such-thread").is_none(),
        "an unknown thread has no root transcript"
    );

    // A compaction seals generation 0 and opens generation 1.
    let (successor, next) = locator
        .begin_generation(&session, meta.clone())
        .expect("begin_generation opens a successor");
    assert_eq!(successor.generation, 1);
    let compacted = vec![contract_message("user", "summary: a table for two")];
    next.replace(&compacted).expect("replace succeeds");
    assert_eq!(locator.head_generation(&session), successor);
    assert_eq!(
        locator.session_chain(&session),
        vec![session.clone(), successor.clone()],
        "the chain walks every generation, oldest first"
    );
    let sealed = locator
        .read_session_transcript(&session)
        .expect("the sealed generation still reads back")
        .read_session()
        .expect("read_session succeeds")
        .expect("the sealed generation keeps its transcript");
    assert_eq!(
        content_view(&sealed.messages),
        content_view(&turn),
        "a compaction leaves the sealed generation untouched"
    );

    assert!(
        locator
            .append_interrupted_partial(
                "contract-store-thread",
                Some("contract-agent"),
                &TranscriptPartial::new("half an ans"),
                Some("contract-turn-2"),
            )
            .expect("a partial append is checked"),
        "the current head accepts a display partial"
    );
    let replay = locator
        .read_session_transcript(&session)
        .expect("the session reads back")
        .read_session()
        .expect("read_session succeeds")
        .expect("the session has a transcript");
    assert!(
        replay
            .messages
            .iter()
            .all(|message| message.content != "half an ans"),
        "a display partial never enters the model-context replay"
    );
    let display = locator
        .root_for_thread_scoped("contract-store-thread", Some("contract-agent"))
        .expect("the display transcript remains discoverable")
        .read_session()
        .expect("read_session succeeds")
        .expect("the display transcript exists");
    assert_eq!(content_view(&display.messages), content_view(&compacted));

    // ── Turn states ──────────────────────────────────────────────────
    let turns = &stores.turn_states;
    let earlier = TurnState::started(
        "contract-store-thread",
        "contract-turn-1",
        8,
        "2026-01-01T00:00:00Z",
    );
    let later = TurnState::started(
        "contract-store-thread",
        "contract-turn-2",
        8,
        "2026-01-01T00:01:00Z",
    );
    turns.put(&earlier).expect("put succeeds");
    turns.put(&later).expect("put succeeds");
    assert_eq!(
        turns
            .get("contract-store-thread")
            .expect("get succeeds")
            .map(|turn| turn.request_id),
        Some("contract-turn-2".to_string()),
        "get answers the newest turn"
    );
    assert_eq!(
        turns
            .list_thread("contract-store-thread")
            .expect("list_thread succeeds")
            .iter()
            .map(|turn| turn.request_id.as_str())
            .collect::<Vec<_>>(),
        vec!["contract-turn-2", "contract-turn-1"],
        "list_thread is newest first"
    );
    assert!(
        turns
            .settle_turn(
                "contract-store-thread",
                "contract-turn-1",
                TurnLifecycle::Completed,
                "2026-01-01T00:02:00Z",
            )
            .expect("settle_turn succeeds"),
        "an in-flight turn settles"
    );
    assert!(
        !turns
            .put_unless_completed(&earlier)
            .expect("put_unless_completed succeeds"),
        "a completed turn is never overwritten conditionally"
    );
    assert_eq!(
        turns
            .mark_all_interrupted("2026-01-01T00:03:00Z")
            .expect("mark_all_interrupted succeeds"),
        1,
        "only the turn still in flight is interrupted"
    );
    assert_eq!(
        turns
            .get_turn("contract-store-thread", "contract-turn-2")
            .expect("get_turn succeeds")
            .map(|turn| turn.lifecycle),
        Some(TurnLifecycle::Interrupted)
    );
    assert!(
        turns
            .delete_turn("contract-store-thread", "contract-turn-2")
            .expect("delete_turn succeeds")
    );
    assert_eq!(turns.list().expect("list succeeds").len(), 1);
    assert_eq!(turns.clear_all().expect("clear_all succeeds"), 1);
    assert!(
        turns
            .get("contract-store-thread")
            .expect("get succeeds")
            .is_none()
    );

    // ── Key-value and journal ────────────────────────────────────────
    stores
        .kv
        .put("contract", "key", json!({"n": 1}))
        .await
        .expect("kv put succeeds");
    assert_eq!(
        stores
            .kv
            .get("contract", "key")
            .await
            .expect("kv get succeeds"),
        Some(json!({"n": 1}))
    );
    assert_eq!(
        stores.kv.list("contract").await.expect("kv list succeeds"),
        vec!["key".to_string()]
    );
    stores
        .kv
        .delete("contract", "key")
        .await
        .expect("kv delete succeeds");
    assert_eq!(
        stores
            .kv
            .get("contract", "key")
            .await
            .expect("kv get succeeds"),
        None
    );
    stores
        .journal
        .append("contract-run", json!("first"))
        .await
        .expect("journal append succeeds");
    stores
        .journal
        .append("contract-run", json!("second"))
        .await
        .expect("journal append succeeds");
    let entries = stores
        .journal
        .read_from("contract-run", 0)
        .await
        .expect("journal read succeeds");
    assert_eq!(
        entries
            .into_iter()
            .map(|(_, value)| value)
            .collect::<Vec<_>>(),
        vec![json!("first"), json!("second")],
        "a journal reads back in append order"
    );
}

/// Checks that `provider` keeps agents apart: whatever one agent writes,
/// another agent's stores cannot see.
///
/// Only for providers that serve several users from one backend; a
/// single-user host that shares one store between its agents does not claim
/// this. Uses agents `contract-iso-a` and `contract-iso-b`.
pub async fn session_store_isolation_conformance(provider: &dyn SessionStoreProvider) {
    let alice = provider.for_agent("contract-iso-a");
    let bob = provider.for_agent("contract-iso-b");

    let session = SessionRef::scoped("shared-thread", "contract-iso-a");
    let mut meta = contract_meta();
    meta.thread_id = Some("shared-thread".to_string());
    meta.agent_id = Some("contract-iso-a".to_string());
    let turn = vec![contract_message("user", "alice's secret")];
    alice
        .transcripts
        .open_session(&session, meta.clone())
        .expect("open_session binds a handle")
        .append_turn(TranscriptTurn {
            prev: &[],
            next: &turn,
            meta: &meta,
            turn_usage: None,
            request_id: None,
            tools: None,
        })
        .expect("append_turn succeeds");
    alice
        .turn_states
        .put(&TurnState::started(
            "shared-thread",
            "r1",
            8,
            "2026-01-01T00:00:00Z",
        ))
        .expect("put succeeds");
    alice
        .kv
        .put("goals", "shared-thread", json!("alice"))
        .await
        .expect("kv put succeeds");
    alice
        .journal
        .append("shared-run", json!("alice"))
        .await
        .expect("journal append succeeds");

    assert!(
        !bob.transcripts.session_exists(&session),
        "another agent cannot see a transcript by its session"
    );
    assert!(
        bob.transcripts.root_for_thread("shared-thread").is_none(),
        "another agent cannot find a transcript by a shared thread id"
    );
    assert!(
        bob.transcripts.latest_for_agent("contract-agent").is_none(),
        "another agent cannot find a transcript by agent name"
    );
    assert!(
        bob.turn_states
            .get("shared-thread")
            .expect("get succeeds")
            .is_none(),
        "another agent cannot see turn states"
    );
    assert_eq!(
        bob.kv
            .get("goals", "shared-thread")
            .await
            .expect("kv get succeeds"),
        None,
        "another agent cannot read key-value records"
    );
    assert_eq!(
        bob.journal
            .len("shared-run")
            .await
            .expect("journal len succeeds"),
        0,
        "another agent cannot read journal streams"
    );
}
