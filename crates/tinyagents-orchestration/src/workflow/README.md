# `orchestration::workflow` — durable phase DAG execution

A **workflow** is a directed acyclic graph of phases, each with associated
agents, dependencies, and concurrency limits. The [`WorkflowEngine`] schedules
phases topologically, spawns bounded-concurrent child tasks per phase,
collects results, and manages state transitions durably.

## Public surface

- **`WorkflowEngine<E: WorkflowExecutor>`** — the execution engine: schedules
  runnable phases, spawns child tasks, handles concurrency limits, collects
  results, and persists state. Generic over a host-supplied executor.
- **`WorkflowExecutor`** — the trait for host-supplied work: create and
  monitor child tasks, cancel them, and query their status.
- **`WorkflowStore`, `SessionWorkflowStore`** — durable run state: loads,
  saves, claims (lease), and compare-and-swap updates. [`SessionWorkflowStore`]
  wraps the session run ledger.
- **`WorkflowDefinition`, `WorkflowPhase`** — the declarative phase DAG: phase
  names, descriptions, agent ids, dependencies, concurrency settings.
- **`PhaseStatus`** — phase state machine: Pending → Running → (Completed |
  Failed). Interrupted phases can reset to Pending for retry.
- **State projection functions** — query and mutate the JSON phase-state
  document: `phase_status()`, `next_runnable_phase()`, `all_phases_completed()`,
  `reset_running_phases()`, `phase_prompt()`, `synthesize_summary()`.

## Design and invariants

- **Durable at every boundary:** Workflow runs and phase states are durably
  persisted via [`WorkflowStore`]. The engine is safe to interrupt and resume.
- **Topological scheduling:** Phases are scheduled in dependency order. The
  engine finds the next runnable phase (all dependencies met), preventing
  partial execution and cycles.
- **Bounded parallelism:** The engine respects `default_concurrency` (agents
  per phase in parallel) and `max_children` (total spawned children at once).
  Phase work is fan-out (parallel agents) + fan-in (collect results).
- **Deterministic retry:** If the engine is interrupted while running phases,
  all running phases are reset to Pending (outputs cleared). Completed phases
  remain immutable and are never retried.
- **Result aggregation:** Upstream outputs (from dependency phases) are
  collected and passed to downstream phases' prompts, allowing workflows to
  reason over prior results.
- **JSON phase state:** The phase-state document is JSON (a BTreeMap of phase
  names to `{ status, outputs, reason, ... }`). This projection is durable,
  queryable, and renderable in UIs.

## File map

- **`engine.rs`** — [`WorkflowEngine`] implementation: phase scheduling, child
  task spawning and monitoring, state persistence, concurrency enforcement.
  [`WorkflowStore`], [`SessionWorkflowStore`], [`WorkflowExecutor`] trait.
  `drive()` dispatches to `drive_via_graph()` (the lowered graph, see
  `lower.rs` below) or `drive_legacy()` (the original scheduler loop).
- **`types.rs`** — data types: [`WorkflowDefinition`], [`WorkflowPhase`],
  [`DefinitionError`], [`WorkflowDefinitionListResponse`].
- **`state.rs`** — phase-state projection: JSON document queries and mutations.
  [`PhaseStatus`], `next_runnable_phase()`, `phase_prompt()`,
  `synthesize_summary()`, etc.
- **`graph.rs`** — scheduler *preview* DAG (`scheduler_topology_preview`):
  a fixed, never-executed `dispatch -> run_phase -> dispatch -> ... -> done`
  shape used purely for diagnostics. Not the graph that runs a workflow.
- **`lower.rs`** *(behind the `graph-workflows` Cargo feature)* — the real
  `WorkflowDefinition -> CompiledGraph` lowering (Phase 4 of
  `docs/runtime-comparison/feature-gaps.md`). See below.
- **`validate.rs`** — structural validation: no duplicate phases, valid
  dependencies, no cycles, valid concurrency settings, etc.
- **`tests.rs`** — tests for scheduling, phase transitions, concurrency,
  result aggregation, and interruption/retry behavior; `lowering_tests`
  (behind `graph-workflows`) tests the lowering itself.

## The `graph-workflows` feature: lowering to a `CompiledGraph`

With the `graph-workflows` Cargo feature enabled, `lower.rs` provides two
things:

- **`lowered_topology(definition)`** — a pure, never-executed structural
  export: one `tinyagents_graph` node per phase, `depends_on` expressed as
  literal `add_waiting_edge` barriers. This is "the DAG the definition
  declares", inspectable via `CompiledGraph::topology()`/`export::to_json`,
  analogous to `scheduler_topology_preview` above.
- **`lower_workflow(..)`** — the *executable* lowering `WorkflowEngine::drive`
  actually runs (`drive_via_graph`) when the feature is on and the engine's
  `use_graph` flag is set — `true` by default under the feature; override
  per instance with `WorkflowEngine::with_graph_execution(bool)`. It is
  **not** the literal waiting-edge topology: it is a small `dispatch ->
  <phase> -> dispatch -> ...` graph, where `dispatch` picks the next
  runnable phase via the same `next_runnable_phase` the legacy scheduler
  uses. `lower.rs`'s module doc explains why: `run_phase`'s durable
  compare-and-swap persistence assumes it is never raced by a sibling phase
  reading the same pre-step snapshot, which literal per-phase waiting edges
  would violate for two independent phases that become ready in the same
  superstep. The router shape keeps exactly one phase node active per
  superstep, so `run_phase` — durability, child registration, lease
  heartbeat, cancellation, agent fan-out via
  `tinyagents_graph::parallel::map_reduce` — is reused **completely
  unchanged** from the legacy path.

`WorkflowStore`/the run ledger remains the single source of truth (the
status projection) and the durable lease remains the lock, on both paths —
the graph only changes how phases are *sequenced*, not how they are
*persisted*.

Because `run_phase` is reused unchanged, the entire `tests.rs` suite
(`cargo test -p tinyagents-orchestration`, no feature: legacy path; `cargo
test -p tinyagents-orchestration --features graph-workflows`: graph path,
since the feature defaults `use_graph` to `true`) passes on **both** paths
with no test needing to be marked graph-path-legacy-only. `lowering_tests`
in `tests.rs` additionally tests the lowering itself: topology export
matches the definition, per-phase fan-out count matches `agent_ids`,
`depends_on` ordering is respected, and a resumed-after-interrupt run
skips already-completed phases.

## Relationship to other modules

- **Depends on:** `tinyagents-graph` (DAG validation), `tinyagents-session`
  (run ledger), `tinyagents-harness` (cancellation, error types).
- **Used by:** host orchestration logic (workflow management and execution).
- **Integration:** Can model multi-agent phases as teams (via
  `orchestration::teams`).

## Typical usage

1. Define a [`WorkflowDefinition`] with phases and dependencies.
2. Create a [`WorkflowEngine`] with a host-supplied [`WorkflowExecutor`] and
   [`WorkflowStore`].
3. Call `engine.run()`: the engine schedules phases, spawns bounded child
   tasks, collects results, and persists state.
4. On interruption, retry `engine.run()`: running phases reset to Pending;
   completed phases remain done.
