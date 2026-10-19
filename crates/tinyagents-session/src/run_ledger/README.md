# `session::run_ledger` — background run and team coordination

A durable, restart-survivable ledger for background agent/workflow execution
state, sharing the session database with `crate::store`. See the top-level
`session` crate README for how this fits next to session history and the
graph checkpointer.

## Why this is separate from session history

`crate::ops` (session history) records what a conversation *said*. This
module records what the runtime was *doing*: which background agent runs,
worker threads, and workflows were in flight, what state they held, and how
they coordinate. A run and the session that produced it are queryable
together (both go through `crate::store`), but they answer different
questions and have different write patterns — session history is append-only
per turn; the run ledger is upserted and compare-and-swapped as a run's
state changes.

## Public surface

Re-exported from `run_ledger::` (see `mod.rs`); also reachable through the
crate root as `session::run_ledger::`.

### Agent runs

- `upsert_agent_run` / `get_agent_run` / `list_agent_runs` — CRUD for
  `AgentRun` (subagent, worker-thread, background-agent, team-member, or
  workflow-child runs).
- `transition_agent_run_status` — the only path that can *clear* `error` /
  `completed_at` (the upsert can only set them); used by control verbs like
  "retry".
- `interrupt_orphaned_agent_runs` — startup sweep that settles any run still
  `running`/`pending` from a dead process to `interrupted`.

### Workflow runs

- `upsert_workflow_run` / `get_workflow_run` / `list_workflow_runs` — CRUD for
  `WorkflowRun`.
- `try_claim_workflow_run` — acquires (or observes) the compare-and-swap
  driver lease that ensures at most one process drives a given workflow.
- `compare_and_swap_workflow_run` / `compare_and_swap_workflow_run_lifecycle`
  — revision-fenced writes for a live driver (the former renews the lease and
  checks the caller still owns it; the latter is for host lifecycle commands
  that may fence an in-flight driver and always clears the lease).
- `renew_workflow_run_lease` — heartbeat that extends a lease without
  advancing `revision`.

### Events and telemetry

- `append_run_event` / `list_recent_run_events` — an append-only,
  atomically-sequenced per-run event log.
- `upsert_run_telemetry` — partial-update token/cost/status rollup per run.

### Agent teams

- `upsert_agent_team` / `get_agent_team` / `list_agent_teams` — CRUD for
  `AgentTeam`.
- `upsert_agent_team_member` / `get_agent_team_member` /
  `list_agent_team_members` — CRUD for `AgentTeamMember`.
- `mark_agent_team_member_running` / `mark_agent_team_member_idle` /
  `mark_agent_team_member_idle_if_current` / `shutdown_agent_team_member` —
  member lifecycle transitions; conditional idle fences stale worker cleanup,
  and shutdown also releases any task the member had claimed.
- `upsert_agent_team_task` / `get_agent_team_task` / `list_agent_team_tasks`
  — CRUD for `AgentTeamTask`.
- `claim_agent_team_task` / `complete_agent_team_task` /
  `complete_agent_team_task_if_claimed` / `release_agent_team_task` /
  `release_agent_team_task_if_claimed` — the
  claim/completion protocol: an atomic compare-and-swap claim, a quality-gated
  completion, and an explicit release for a member that abandons a task without
  shutting down. Conditional completion and release fence worker operations by
  their claim token.

### Restart recovery classification (`recovery.rs`)

- `classify_recovery(tail, tool_effects)` — pure and advisory. Given the
  transcript tail's dangling tool calls (`DanglingToolCall`, no recorded
  result) and the per-call `tool_effects` rows, returns one `CallRecovery` per
  call, matched on `(run_id, call_id)`:

  | Effect record | Class | Meaning |
  | --- | --- | --- |
  | none | `Resume` | the call never began; continue normally |
  | `completed`, `failed` | `ResumeReportOnly` | status is recorded (the result/error itself may not be persisted); report it, do not re-execute |
  | `deferred` | `AwaitingAnswer` | the call paused for approval/a result; resume by answering the deferral, never re-execute |
  | `started` only, `interrupted` | `NeedsVerification` | may have committed; verify real-world state, never blindly re-run |

- When several effect rows match one call, the most cautious class wins.
- `overall_recovery(&classified)` — the most cautious class (`Resume <
  ResumeReportOnly < AwaitingAnswer < NeedsVerification`), `Resume` when nothing dangles.

The effect ledger writes `started` before a call executes, which is why "no
row" means "never began"; hosts that did not attach the ledger must not rely on
`Resume`. Nothing here re-runs a tool or writes to the ledger. For the
parent-facing note about interrupted *children* see
`tinyagents_tasks::build_restart_recovery_note`.

## Layout of this module

| File | Role |
| --- | --- |
| `mod.rs` | Module docs and public surface (re-exports). |
| `types.rs` | Serde record types, one `*Upsert` companion per persisted type, and list request/response shapes. |
| `ops.rs` | CRUD, listing, and coordination primitives (see below). |
| `store.rs` | Schema entry point — a no-op now that all DDL lives in `crate::migrations`; kept as the conventional call site. |
| `recovery.rs` | `classify_recovery` / `overall_recovery`: advisory restart-recovery verdicts for dangling tool calls (tests in `recovery_tests.rs`). |
| `mod_tests.rs` | Module-local unit tests. |

## Operational constraints

These are the non-obvious rules; most are pinned by a test in `mod_tests.rs`.

**An upsert reads its own write back inside the same transaction.** Every
`upsert_*` function opens one `crate::store::with_transaction` call that
both writes the row and re-fetches it via a connection-scoped `*_inner`
helper — never a fresh `with_connection` after the write, which could
observe a concurrent writer's state instead of the caller's own.

**Workflow-run upserts favor `COALESCE` on plain `upsert_workflow_run`, but
compare-and-swap for a live driver.** `upsert_workflow_run` has no revision
fencing and can clobber a concurrent driver's write; a driver holding a
lease must use `compare_and_swap_workflow_run` (revision + lease-owner
checked) or `compare_and_swap_workflow_run_lifecycle` (revision only, always
clears the lease) instead.

**A workflow lease is cleared whenever `status` becomes terminal.** Both the
plain upsert and the compare-and-swap paths null out `lease_owner` /
`lease_expires_at` on a transition to `completed` / `failed` / `cancelled` /
`interrupted`, since a finished run has nothing left to drive.

**Run-event sequences are allocated by the INSERT itself**, via a
sub-`SELECT MAX(sequence) + 1` inside the same statement — never a separate
read followed by an insert, which would race two concurrent appenders onto
the same sequence number and lose one event to a primary-key conflict.

**Telemetry counters are `Option` for partial updates.** The columns are
`NOT NULL DEFAULT`, and SQLite does not apply that default to an explicit
`NULL`, so `upsert_run_telemetry`'s insert side coalesces to the column
default while its update side coalesces to the *stored* value — `excluded.*`
cannot serve the update side, since it observes the already-coalesced insert
row and would read a caller's `None` as `0`.

**A team-task claim is meaningful only while `status = "in_progress"`.**
`claimed_by_member_id` and `claim_token` are cleared whenever
`upsert_agent_team_task` moves a task off that status; leaving them set
strands the task — a new claim sees `AlreadyClaimed`, completion sees
`NotClaimed`, and release/shutdown skip it.

**Completion evidence accumulates across attempts, including failed ones.**
`complete_agent_team_task` merges submitted evidence into what is already
stored rather than replacing it, so a retry after fixing one gate failure
does not have to resubmit evidence already accepted.

**`shutdown_agent_team_member` releases the member's in-progress tasks back
to `todo`** (clearing claimant + token) in the same transaction that marks
the member `stopped` — the bulk analogue of `release_agent_team_task`.

`status_map.rs` holds the total `From` conversions between `AgentRunStatus` and the canonical `tinyagents_tasks::OrchestrationTaskStatus` (`Paused` reads as `Awaiting`, `Interrupted` as `Abandoned`); the `AgentRunStatus` wire strings are unchanged. Mapping table: `tinyagents-tasks` README.


## Storage provider port

`RunLedgerStore` is the object-safe, async form of the existing ledger
API. A handle binds its workspace or namespace once; payloads never choose
another namespace. It covers agent runs, workflow leases/revisions, events,
telemetry, teams/members/tasks, effects, recovery classification,
command-center controls, and the existing retention operations. The three pure
recovery helpers remain synchronous. Storage methods can await driver ports
directly; no blocking bridge or caller-runtime reentry is required. The SQLite
adapter polls the original short synchronous transactions; it keeps their
existing execution behavior rather than adding a driver client.

`SqliteRunLedger::new(workspace)` implements the port by delegating to the
existing SQLite functions. Status CAS plus event remains one transaction;
workflow ownership, expiry and revision are checked together; task dependency
reads and claims retain their current transaction. Constructors do not open or
migrate the database. Existing workspace-based APIs and all SQLite tables remain
available, including the still-active session/message/tool-call tables.

Providers may override `SessionStoreProvider::run_ledger_for_agent` to return an
`Arc<dyn RunLedgerStore>`. Its default is `Ok(None)`, independent of the existing
`AgentStores` bundle. `Ok(None)` does not authorize a local fallback. Admission or backend
errors use the typed error result and are never disguised as absence. Hosts must
select their policy explicitly; this new port does not route existing callers
or migrate their data by itself. Shared hosts must supply handles bound to the
admitted tenant and agent, and must not use the classic startup orphan sweep to
interrupt another live process.

The SQLite adapter preserves current task completion semantics: completion is
validated against the member identity, not a takeover claim token. Retention
preserves its existing scope: sessions/messages/tool calls, events and telemetry;
it does not add pruning of workflows, teams or effects.

### Typed errors

The port returns `LedgerResult<T>`. `LedgerError` retains legacy SQLite and
control errors and, with `storage-drivers`, native `StorageError` values. Native
admission, fencing, conflict and capability kinds remain available to callers.

### Scoped driver adapter

With `storage-drivers`, `DriverRunLedger::new(scoped)` binds an explicitly
admitted `ScopedStorage`. It opens no workspace database. The supplied native
fence remains attached to its document handles.

Implemented native families include agent upsert/read/filter/page, telemetry joins,
status compare-and-swap, events and command-center controls; workflow
upsert/read/list and lease/revision/lifecycle CAS; telemetry field deltas;
team/member/task upsert/read/list; tool-effect start/settle/unresolved/interrupted;
member running/idle transitions; and individual task release. Conflicts alone
are retried. Native fencing, generation and capability refusals retain their
typed sources. Agent writes validate the captured telemetry version in the same
document batch and return the committed body without a fallible postcommit read.

Dependency-aware team claims/completion, bulk member shutdown and retention
across sessions/messages/events still return typed Unsupported before writing.
The orphan sweep is explicitly for an exclusive owner; it is not a shared-process
lease reaper.
Existing host callers are not routed to this adapter automatically.

DriverSessionStores caches one optional native ledger handle per logical agent.
Independent factories and actor bindings keep independent caches while sharing
the same durable namespace. The provider currently constructs declaring ledger
handles; the direct validation-only constructor below does not prove that a
provider actor factory propagates validation-only admission.

| Scoped collection | Records and query indexes |
| --- | --- |
| `ledger.workflow_runs` | Definition, status, parent thread, updated time; the driver lease is a `tsd.leases` record (`ledger`, `Key::of(["workflow", run])`) |
| `ledger.telemetry` | Run telemetry, updated time |
| `ledger.agent_runs` | Runs; kind, status, parent/worker identities and updated time |
| `ledger.teams` | Parent thread, status and updated time |
| `ledger.members` | Team, creation time and unique `(team_id, name)` |
| `ledger.tasks` | Team, status, claimant and `(team_id, order_index, created_ms)` |
| `ledger.tool_effects` | Run, status and start time |
| `ledger.event_outbox` | Authoritative events, unique per-run sequence |
| `ledger.event_heads` | Next per-run event ordinal (recomputed from the events when absent) |
| stream `ledger.events/<run>` | The native projection of a run's events |
| `tsd.migrations` (`ledger`, `agent_import/<operation id>`) | Agent-import receipt; the source fingerprint is its `cursor` |

Records are bare snake_case storage DTOs with `*_ms` times (the camelCase wire
types are unchanged) under `Key::of` ids. The writer-set times stay body
fields: an import or a caller supplies them, and paging and retention order by
them. Times are whole milliseconds. Durable metadata has no TTL.
Single-record updates are one `Repo::update` loop. A workflow claim, renewal or
revision-fenced write is a lease operation plus a record write, so unlike the
embedded lease it is no longer one atomic document change.

See [`docs/modules/session/ledger-history-storage-schema.md`](../../../../docs/modules/session/ledger-history-storage-schema.md)
for the v1 to v2 mapping, the steps and the host importer.

### Status, event receipts and stream projection

A native checked document batch commits agent status, event allocation and its
durable receipt together. Failed status CAS writes no event. Receipts are the
authoritative event history; list cursors retain the classic signed event
ordinals, including gaps imported after legacy pruning. Native queries request
only the remaining result limit and follow driver cursors; zero-limit calls
still obtain native read admission.

Document and stream ports have no combined transaction.
DriverRunLedger::reconcile_run_events explicitly projects receipts into the
per-run stream using conditional append. It verifies the existing prefix
against committed receipts, refuses corruption, and retries competing appends
without duplicates. Successful batches extend the already-validated prefix;
conflicting appends revalidate it. Only consecutive native conflicts consume
the contention budget, so successful progress can project arbitrarily many
batches. Stream offsets count projected entries; they are distinct
from legacy event ordinals. Projection refusal preserves the successful
document commit and returns a typed error only from the explicit reconciliation
call. Reconstructed handles can replay outstanding receipts. Neither receipts
nor projection currently implement retention.

### Explicit telemetry retention

`prune_run_telemetry_before` removes only telemetry whose non-null updated instant
is strictly before the supplied cutoff. It enumerates the bound scope, then
conditionally deletes all eligible observed versions in one validated native
batch. A concurrent update conflicts and triggers re-enumeration; an update
that refreshes the timestamp survives. New rows after enumeration belong to a
later pass. The operation does not promise a collection-wide snapshot or invoke
retention automatically. Timestamp comparison uses instants, avoiding variable
RFC3339 precision changing eligibility.

Empty passes still require native batch/fence admission. Typed refusal changes
no candidate records; a retained retired-generation handle cannot prune records
in a newly issued generation. Schema admission stays the supplied handle’s
policy. Session/message/tool-call/event retention and whole-ledger retention are
still unsupported; no event receipts, streams, import receipts, or run records
are removed by telemetry pruning.

### Schema preparation and validation

DriverRunLedger::collection_specs exposes the shared declarations.
prepare_schema is an explicit trusted-bootstrap declaration operation.
new_existing_schema validates those declarations rather than ensuring them;
missing or incompatible schemas refuse before ledger records are written.
The supplied ScopedStorage must also use the native ExistingSchema contract to
restrict driver-internal initialization. The constructor does not infer
authority from capabilities or turn arbitrary ports into restricted ports.

### Source admission before migration

`SourceOwner::asserted_by_trusted_host(scope)` and
`QuiescentSource::asserted_by_trusted_host(path, owner)` represent trusted host
assertions. Hosts must validate the path-to-owner mapping and stop source writers
for the complete inspection/import. Model input, stored rows and path digests
cannot establish that authority.

`preflight_source` compares source ownership with the target scope before source
access. It rejects missing databases without creating directories, and refuses
WAL or journal sidecars with `LedgerError::SourceRequiresSnapshot`. The host must
supply a checkpointed database or consistent snapshot; the importer never
checkpoints it. Sources are opened through SQLite immutable read-only access,
which also avoids writing auxiliary shared-memory files. The retained snapshot
discovers historical schema and columns without migrations. import_agent_family supports the known agent_runs, run_events and run_telemetry
schema. It validates all source rows before target declarations, commits all
absent target records and a durable operation receipt in one native batch, and
retains source bytes. An identical operation retry returns its receipt without
overwriting later native edits; changed source bytes or target collisions
refuse. The source fingerprint identifies content and grants no authority.
Original event ordinals survive import, and allocation continues above their
maximum.

Hosts must still arrange source ownership, quiescence, snapshots and the
migration checkpoint. Older incomplete layouts can be inspected but are not
silently upgraded. Workflow/team/effect/session import, automatic host
migration, shared-owner recovery and generation-aware retention remain separate
work.
