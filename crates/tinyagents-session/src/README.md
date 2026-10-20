# `session` — durable session history, run ledger, and transcripts

Driver-backed history for agent sessions (`DriverRunLedger` over any
`tinystoragedrivers` backend), a restart-survivable ledger for background
agent/workflow execution, a store-port transcript seam for KV-cache-stable resume, and a driver-backed chat
thread/message store for the product-facing chat log.

## Why this is a top-level module

Session history is a persistence domain in its own right, not part of the agent
loop. Nothing in `harness` reads from it, and a host can use it without running
a harness at all — indexing sessions produced elsewhere, or recovering
orchestration state at boot before any agent exists. Filing it under `harness::`
would imply a dependency that exists in neither direction.

## How it differs from the other persistence layers

| Layer | Question it answers | Lifetime |
| --- | --- | --- |
| `harness::store` | "what is this run working with right now?" | during a run |
| `graph::checkpoint` | "how do I resume this interrupted run?" | until resumed |
| **`session` (driver ledger)** | "what happened, what did it cost, how did runs nest?" | indefinitely |
| **`session::transcript` (store port)** | "what exact messages did the model see?" | indefinitely, resumed verbatim |
| **`session::threads` (driver)** | "which chats does the user have, and what is in them?" | indefinitely |

The driver-backed history (`run_ledger`) is queryable history that
nothing resumes *from*: cross-session search, cost attribution, and
orchestration recovery. `session::transcript` is the exception — it exists
specifically so a restarted run can resume with the byte-identical message
stream a provider (and its prompt cache) already saw; see its own
[README](./transcript/README.md) for the format.

## Layout

Everything lives wherever the host's storage backend puts it:

```text
{host storage backend}                          ← sessions, run ledger, transcripts, chat threads (collections)
```

Classic workspace files are outside this crate's layout. The host owns their
discovery, import and any rewrite; see the legacy import handoff.

## Public surface

Re-exported from the crate root (see `src/lib.rs`); the full surface stays
reachable under `session::`, `session::run_ledger::`, and
`session::transcript::`.

- **Session history** — the `SessionHistoryStore` trait on `DriverRunLedger`:
  `record_session_start`, `record_message`, `record_tool_call`,
  `record_session_end`, `get_session`, `list_sessions`, `search_sessions`,
  `list_messages`, `list_tool_calls`, `list_children`, `mark_interrupted`
- **Retention** — `apply_retention`, `prune_sessions_before`,
  `prune_tool_calls_before`, `prune_run_events_before`,
  `prune_run_telemetry_before`, `trim_session_messages`, `history_reindex`
- **Run ledger** — agent runs, workflow runs, teams, members, tasks, run events,
  and telemetry, with the claim/completion coordination primitives — see its
  own [README](./run_ledger/README.md)
- **Transcripts** — the `TranscriptLocator`/`TranscriptHistory` store seam,
  background appends, the per-session turn lock, thread spend and the display
  view — see its own [README](./transcript/README.md)
- **Chat threads** — `DriverConversationStore` and its wire types (re-exported at
  the root) under `session::threads` — see its own [README](./threads/README.md)
- **Store port** — `SessionStoreProvider`, `AgentStores`, `TurnStates`,
  `InMemorySessionStores`, `DriverSessionStores`, and `RebindableSessionStores` over a `tinystoragedrivers` backend — see
  [`docs/modules/session/store-port.md`](../../../docs/modules/session/store-port.md)
- **Legacy import handoff** — workspace discovery and parsing moved to the
  host; see [`docs/legacy-import-handoff.md`](../docs/legacy-import-handoff.md).
- **Testkit** — `testkit::conformance::run_ledger_conformance` and
  `transcript_history_conformance`: contract suites certifying that a run
  ledger or a `TranscriptHistory` implementation behaves like the bundled
  ones. Run against the testkit memory backend and both the driver-backed and the in-memory
  `testkit::InMemoryTranscriptHistory` double in
  `crates/tinyagents-integration-tests/tests/session_conformance.rs`.

## Collections

`DriverRunLedger` declares its collections on first use, idempotently:

| Collection | Holds |
| --- | --- |
| `history.sessions` | one document per session; lineage via `parent_session_id`, times in `started_ms` / `ended_ms` |
| `history.messages` | per-message content, optional reasoning, model, tokens, cost |
| `history.tool_calls` | tool name, input, bounded output, status, duration |
| `history.search` | search rows over session name, message content, tool name |
| `tsd.counters` (`history`, `messages` / `tool_calls`) | dense message and tool-call id counters; a failed commit leaves a gap |
| run-ledger collections | agent and workflow runs, run events and telemetry, teams, members, tasks |

Classic `sessions.db`, `session_raw/` transcript and turn-state import is a
host responsibility. Session storage declarations still migrate records
already held by the admitted backend.

## Operational constraints

These are the non-obvious rules; each is pinned by a test in `run_ledger/drivers/history_tests.rs`.

**Search input is plain text, not FTS5 syntax.** `SessionSearchParams::query` is
tokenised into plain terms before it reaches the backend search, so ordinary
strings (`C++`, `foo-bar`, `file.rs`) search instead of failing on syntax.

**Indexed content is truncated on a character boundary.** Slicing at a raw byte
offset panics on multi-byte input, and because the message insert has already
committed, the row would survive with no FTS entry — silently unsearchable.

**Tool output is bounded** to `MAX_TOOL_OUTPUT_BYTES`, truncated on a character
boundary with a marker appended.

**Nothing is deleted unless a host asks.** `retention.rs` owns the only delete
paths (sessions, messages, tool calls, run events, telemetry) plus
`history_reindex`, which rebuilds the search index from the stored rows. Retention is a policy decision,
so none of it runs on a schedule of its own.

**A claim is meaningful only while a task is `in_progress`.** An upsert that
moves a task off that status clears `claimed_by_member_id` and `claim_token`;
leaving them set strands the task, since a new claim sees `AlreadyClaimed`,
completion sees `NotClaimed`, and release/shutdown skip it.

**Evidence accumulates across completion attempts,** including attempts whose
gate fails — otherwise a retry after fixing an unrelated gate would fail
`require_evidence` on evidence already submitted.

## Layout of this module

| File | Role |
| --- | --- |
| `lib.rs` | crate docs and public surface |
| `types.rs` | serde record types for sessions/messages/tool calls |
| `fts.rs` | search-snippet cap and snippet helper |
| `run_ledger/` | background run + team coordination — see its own [README](./run_ledger/README.md) |
| `threads/` | driver-backed chat thread/message store with a cross-thread search index — see its own [README](./threads/README.md) |
| `transcript.rs` + `transcript/` | durable, provider-neutral transcripts for KV-cache-stable resume — see its own [README](./transcript/README.md) |
| `*_tests.rs` | module-local unit tests |
