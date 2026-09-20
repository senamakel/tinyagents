# SDK Gaps: Orchestration And Control

> Part of [SDK Gaps](README.md). Covers graph fanout and parallel-agent
> ergonomics, sub-agent steering/waiting/reuse, workspace isolation, and
> middleware control outcomes.

## Backlog

### 10. Graph Fanout And Parallel Agent Ergonomics

Status: partially present.

TinyAgents graph has `Send`, `Command`, reducers, interrupts, parallel execution,
and max concurrency. OpenHuman still added `run_parallel_fanout` to provide an
ordered, bounded map/reduce helper for council runs and `spawn_parallel_agents`.

Implement:

- Add a generic SDK helper for parallel map/reduce:
  - preserve input order
  - limit concurrency
  - collect per-item success/failure
  - support cancellation
  - support reducer updates
  - support timeout per item and total timeout
  - expose graph lifecycle events
- Add a higher-level parallel-agent builder:
  - validate task specs
  - dispatch workers through `Send`
  - collect result envelopes
  - merge usage/cost/events
  - detect worker failure policy: fail-fast, collect-all, quorum, best-effort

Acceptance criteria:

- OpenHuman can delete most of `run_parallel_fanout` and use the SDK helper.
- `spawn_parallel_agents` can be expressed as graph configuration plus
  OpenHuman policy adapters.
- Results remain deterministic in input order even when workers complete out of
  order.

### 11. Sub-Agent Steering, Waiting, And Reuse

Status: partially present; queued steering/follow-ups shipped (A4).

TinyAgents has sub-agent and steering primitives, but OpenHuman still owns
session reuse, wait handles, detached run tracking, user-facing cancellation,
early-exit handling, and parent-child progress aggregation. The generic
`DetachedTaskRegistry` owns owner checks, wait/timeout, cancel-before-abort,
steering lookup, and bounded terminal cleanup.

A4 (`docs/runtime-comparison/plan.md` Phase 2) put `RunQueue<Message>` on the
loop path: `RunContext::with_run_queue`; `Steer` drained after each tool batch
and at a natural finish, `Followup` at a natural finish (one more turn),
`Collect` onto `AgentRun::collected`; `RunPolicy::queue_mode` (`All` |
`OneAtATime`); `AgentEvent::QueuedMessageApplied`. `SteeringHandle` is
unchanged; OpenHuman's `agent/harness/run_queue/` can go. See
[`runtime.md`](modules/harness/runtime.md#queued-steering-and-follow-ups-a4).

Implement:

- First-class detached sub-agent sessions.
- `wait`, `cancel`, `kill`, `resume`, `steer`, and `close` controls backed by
  `TaskStore`.
- Reusable child sessions with explicit lifecycle state.
- Parent/root event correlation for every child run.
- Early-exit policy that can pause a run and surface a structured payload.

Acceptance criteria:

- Callers can spawn a detached child run, wait for it later, and survive process
  restart if durable stores are configured.
- Parent and child usage/cost/events roll up without bespoke registries.
- OpenHuman can reduce `running_subagents.rs` to policy and UI projection code.

### 12. Workspace Isolation And Sandbox Hooks

Status: missing as an SDK-owned abstraction.

OpenHuman has workspace/action-root policy, internal workspace protection,
trusted roots, worktree isolation, sandbox modes, and command permission tiers.
TinyAgents should not own OpenHuman's policy, but it needs generic hooks for
agents that run tools over real files or command executors.

Implement:

- A `WorkspaceIsolation` or `ExecutionEnvironment` interface.
- Hooks for preparing per-agent worktrees/sandboxes and cleaning them up.
- Tool execution context fields for workspace root, logical task root, sandbox
  descriptor, and policy identity.
- Events for isolation setup, violation, cleanup, and failure.

Acceptance criteria:

- Parallel agents can run with isolated workspaces using SDK lifecycle hooks.
- Tools can discover their allowed root from context instead of app globals.
- Policy engines can block unsafe paths before tool execution.

### 13. Middleware Control Outcomes

Status: shipped (harness loop control), partial (graph/sub-agent defer).

Landed as part of `docs/runtime-comparison/plan.md` Phase 2 item A1. Every
lifecycle `Middleware` hook has a `_control`-suffixed counterpart
(`before_model_control`, `after_model_control`, `before_tool_control`,
`after_tool_control`, `before_agent_control`, `after_agent_control`) that the
`MiddlewareStack` actually drives, returning `MiddlewareControl::{Continue,
JumpTo(LoopTarget::{Model,Tools,End}), UpdateState(StateUpdate),
StopWithFinal, Interrupt}`; a default shim forwards to the pre-existing plain
hook and returns `Continue`, so no existing `Middleware` impl needed to
change. `MiddlewareModelOutcome`/`MiddlewareToolOutcome` gained a `Command`
variant for a `wrap_model`/`wrap_tool` short-circuit. A canonical tool's own
`ToolResult.control` (vendored `tinytools::ToolControl`:
`return_direct`/`terminate`/`goto`/`state_update`) is translated into the
same vocabulary. `Middleware::should_stop_after_turn` covers a turn-boundary
aggregate stop. Precedence is `Interrupt > StopWithFinal > JumpTo/UpdateState
> Continue`; within one phase the first non-`Continue` outcome wins and later
non-observer hooks are skipped (`Middleware::is_observer`).
`BudgetMiddleware`/`HumanApprovalMiddleware` are rebased on `JumpTo(End)` and
`Interrupt` respectively. See
`crates/tinyagents-harness/src/{context,middleware,agent_loop}/*` and
[`docs/modules/harness/middleware.md`](modules/harness/middleware.md#middleware-control-a1).

Still open: routing a harness-level control outcome onto a graph `Command`
node when the loop runs as a graph node, and "defer to task/sub-agent" as a
control outcome — both remain graph/A5 (loop-as-`CompiledGraph`) territory,
not yet implemented.

Acceptance criteria (harness scope):

- [x] Early-exit tools and budget stop hooks do not require adapter-local
      steering side channels (`ToolResult::return_direct`/`terminate`,
      `BudgetMiddleware`).
- [ ] Graph and harness middleware use compatible control vocabulary (harness
      side only; graph `Command`/`Interrupt` remain a separate vocabulary).
- [x] Control decisions are visible in journals for audit/replay
      (`AgentEvent::ControlApplied`).

