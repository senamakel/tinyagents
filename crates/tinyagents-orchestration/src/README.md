# `tinyagents-orchestration` — host-neutral agent team and workflow composition

This crate coordinates durable agent work at two scales: **teams** (groups of
collaborative members working on shared tasks) and **workflows** (phases of
work with dependencies, bounded parallelism, and result aggregation). The
crate is deliberately host-free — hosts supply persistence, worker execution,
and policy; this crate owns scheduling, state machine transitions, and
invariant enforcement.

## Public surface

- **`teams::TeamService`** — team creation, member lifecycle, task coordination,
  and message delivery. Enforces duplicate-free member names, valid task
  dependencies (no cycles), and durable task claims.
- **`teams::TeamLedger`** — the trait for durable team state; hosts implement
  or use the built-in [`SessionTeamLedger`] backed by `tinyagents-session`.
- **`teams::NewMember`, `TeamView`, `MemberShutdown`** — data types for team
  operations and snapshots.
- **`workflow::WorkflowEngine`** — drives phase scheduling, spawns child tasks,
  collects results, and manages state transitions. Bounded concurrency via
  `default_concurrency` and `max_children`.
- **`workflow::WorkflowStore`, `SessionWorkflowStore`** — durable workflow run
  state (analogous to [`TeamLedger`]).
- **`workflow::WorkflowDefinition`, `WorkflowPhase`** — declarative phase DAG:
  phases, agents per phase, dependencies, concurrency limits.
- **`workflow::PhaseStatus`**, state projection functions — JSON phase-state
  document and queries (which phases are runnable? which are complete?).

## Design and invariants

- **Host-free composition:** The crate never makes decisions about
  authorization, credentials, model selection, or policy. Hosts supply
  persistence layers, execution callbacks, and policy enforcement. The crate
  enforces structure and state machine safety only.
- **Durable at every boundary:** Team members, tasks, events, and workflow
  runs are all durably persisted. The crate reads and writes state via
  [`TeamLedger`] and [`WorkflowStore`] traits, allowing hosts to use their own
  storage.
- **Event-sourced messaging:** Team messages are durably logged in the run
  event stream. Members read their undelivered messages and durably advance a
  delivery watermark — making repeated calls idempotent.
- **Dependency-aware scheduling:** Teams and workflows both enforce directed
  acyclic dependencies. The scheduler finds the next runnable task/phase (all
  dependencies met), preventing cycles and partial execution.
- **Bounded parallelism:** Workflows respect `default_concurrency` and
  `max_children`. The engine limits concurrent child tasks to prevent resource
  exhaustion.
- **Retry on interruption:** If a workflow is interrupted while running phases,
  running phases are reset to Pending for deterministic retry (with evidence
  cleared). Completed phases remain immutable.

## File and module map

- **`teams/`** — durable, dependency-aware agent-team composition.
  - `service.rs` — [`TeamService`], the public API; [`TeamLedger`] trait;
    [`SessionTeamLedger`] implementation.
  - `types.rs` — [`NewMember`], [`TeamView`], [`MemberShutdown`], [`TeamError`].
  - `graph.rs` — member worker execution DAG (execute → complete/fail → done).
  - `runtime.rs` — prompt composition, message delivery, and event draining.
  - `tests.rs` — tests for team creation, member lifecycle, and messaging.
  - `README.md` — (to be created) detailed team design and lifecycle.
- **`workflow/`** — durable, host-neutral workflow definitions and execution.
  - `engine.rs` — [`WorkflowEngine`], phase scheduling, child task management,
    state persistence, and concurrency bounding.
  - `types.rs` — [`WorkflowDefinition`], [`WorkflowPhase`], [`DefinitionError`].
  - `state.rs` — phase state projection (JSON document), status queries, and
    transitions.
  - `graph.rs` — workflow scheduler DAG and topological ordering.
  - `validate.rs` — structural and host-specific validation.
  - `tests.rs` — tests for scheduling, phase transitions, and concurrency.
  - `README.md` — (to be created) detailed workflow design and execution.
- **`lib.rs`** — crate public surface; dependency direction enforced by tests.

## Relationship to other modules

- **Depends on:** `tinyagents-graph` (DAG operations, graph executor),
  `tinyagents-harness` (cancellation tokens, error types), `tinyagents-session`
  (run ledger, durable storage APIs), `tinyagents-definition` (agent definitions).
- **Used by:** Host orchestration logic (team and workflow management and
  coordination).
- **Test coverage:** Boundary tests ensure the crate depends only on lower
  layers; integration tests cover team coordination, workflow scheduling, and
  concurrency limits.

## Typical usage

1. **Teams:** Create a team with members, add tasks with dependencies, claim
   and complete tasks as members work. Message flow is durable and delivered
   via watermarks.
2. **Workflows:** Define a phase DAG, instantiate the engine with a custom
   executor, and call `run()`. The engine schedules phases, spawns child tasks
   (bounded by concurrency), collects results, and persists state.
