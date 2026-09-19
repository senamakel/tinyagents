# TinyAgents SDK Gaps

> **Internal migration backlog.** This is a working document tracking an
> internal OpenHuman-to-TinyAgents migration effort, not a general public
> roadmap or API reference. See [`ROADMAP.md`](../ROADMAP.md) for the
> project's public-facing roadmap.

This document lists TinyAgents SDK features that are missing or only partially
available from the perspective of migrating OpenHuman's Rust agent core onto
TinyAgents.

Scope:

- Source baseline: local TinyAgents checkout at `6f898fb`.
- OpenHuman evidence: `src/openhuman/tinyagents/*`,
  `src/openhuman/agent/*`, `src/openhuman/cost/*`, and
  `src/openhuman/tokenjuice/*`.
- This is not the OpenHuman migration plan. That plan lives in
  `docs/tinyagents-migration-spec.md`.
- Items here are upstream TinyAgents implementation candidates.
- Tests should be implemented last, after the API and storage surfaces settle.

## Executive Summary

TinyAgents already has strong primitives for harness runs, graph execution,
middleware, event streams, model profiles, usage/cost accounting, checkpointers,
and sub-agent orchestration. The biggest remaining gaps are production-grade
policy metadata, durable orchestration stores, richer streaming events,
recoverable tool-call behavior, graph fanout ergonomics, and SDK-owned adapters
for the lifecycle controls OpenHuman currently implements around the SDK.

OpenHuman can migrate more of `src/openhuman/agent/` if TinyAgents grows these
features:

- First-class reasoning and tool-call argument streaming events (tool
  metadata, unknown-tool recovery, and deferred approvals have since shipped).
- Durable `TaskStore` and event/status stores with replay, lineage, cursors,
  redaction, and cancellation semantics.
- Storage compatibility options for SQLite users that already depend on a
  different `rusqlite` / `libsqlite3-sys` version.
- Higher-level map/reduce and parallel-agent orchestration helpers on top of
  graph `Send`.
- Budget enforcement and provider/model catalog metadata that can drive
  preflight, fallback, and reconciliation.
- Conformance suites for providers, tools, middleware, graph stores, and
  checkpointers.

## Backlog

### 1. Rich Tool Policy Metadata

Status: partially present.

TinyAgents has `ToolSchema { name, description, parameters, format }` and
`ToolExecutionContext { run_id, thread_id, depth, max_turn_output_tokens,
events }`. That is enough for model-visible tool calls, but not enough for
OpenHuman's approval gate, command classifier, workspace policy, sandbox
handoff, or tool-result budgeting.

OpenHuman currently keeps that metadata outside TinyAgents in domain tool
registries and adapters. That means the SDK cannot make fail-closed decisions
about whether a tool should be exposed, approved, retried, timed out, or allowed
to touch the filesystem/network.

Implement (shipped as vendored `tinytools::ToolPolicy` — side effects,
runtime requirements, access requirements — plus `ToolPolicyMiddleware`;
`access.approval_required` now also drives the A2 deferral in §14):

- Add helper middleware for policy enforcement before model-visible exposure and
  before execution.

Acceptance criteria:

- Callers can build a dynamic per-run tool set from policy metadata.
- Unknown or under-classified tools fail closed by default.
- Tool policy can be serialized for registry introspection and audit logs.
- Existing plain `ToolSchema` remains supported as the model-visible projection.

### 2. Recoverable Unknown Tool Calls

Status: shipped.

TinyAgents now has `UnknownToolPolicy::{Fail, ReturnToolError, Rewrite}` on
`RunPolicy` (`crates/tinyagents-harness/src/runtime/types.rs`), applied in
`crates/tinyagents-harness/src/agent_loop/tools.rs` (~305-371). The default is
`ReturnToolError`: an unregistered tool call is injected back as a tool-error
result naming the requested tool and the valid tools, so the loop continues
and the model can self-correct, instead of aborting the run. `Fail` restores
the old abort behavior, and `Rewrite { tool_name }` retargets the call to a
fixed compatibility tool. OpenHuman's `UNKNOWN_TOOL_SENTINEL` workaround can
be retired in favor of this policy.

Still open: a `RepairWithMiddleware` variant letting a tool middleware
transform the call. Events preserve the requested name, arguments, and call id.

Acceptance criteria:

- OpenHuman can delete `UNKNOWN_TOOL_SENTINEL`.
- Harness events distinguish "tool not found" from "tool executed and failed".
- The policy can vary by run, sub-agent, or tool allowlist.

### 3. Reasoning And Tool-Argument Streaming

Status: partial.

`MessageDelta { text, reasoning, tool_call }`
(`vendor/tinyinference/crates/tinyinference-llm/src/message/types.rs`) now
carries a dedicated `reasoning` fragment alongside visible text, and
`ModelDelta` events carry that delta — so reasoning streaming exists.
What is still missing is explicit block start/end channels: there is no
tool-call-start or tool-call-argument-delta / tool-call-completed signal
separate from the accumulated `tool_call` fragment, so a consumer cannot tell
when a tool-call block begins or ends without inferring it from delta
content. OpenHuman providers also emit tool-call argument fragments that need
that boundary information; the current adapter still uses an out-of-band
`ThinkingForwarder` for parts of this.

Implement:

- Extend streaming deltas with explicit channels:
  - visible text delta
  - reasoning/thinking delta
  - tool call start
  - tool call argument delta
  - tool call completed/assembled
  - provider metadata/raw event summary
- Keep channel semantics provider-neutral.
- Emit the same data through `AgentEvent`, `AgentObservation`, journals, and live
  stream items.
- Attribute every delta to run id, model call id, optional thread id, parent run
  id, and root run id.

Acceptance criteria:

- OpenHuman can delete `ThinkingForwarder`.
- UI consumers can render visible text, reasoning, and tool argument assembly
  from TinyAgents events alone.
- Non-streaming providers can still emit post-hoc reasoning as one event.

### 4. Durable Orchestration Task Store

Status: partially present.

TinyAgents defines a `TaskStore` trait and an `InMemoryTaskStore`. OpenHuman
still owns durable detached-sub-agent state, cancellation handles, wait/reuse
semantics, tombstones, and task lifecycle persistence around that store.

Implement:

- Add durable `TaskStore` implementations:
  - JSONL append store.
  - SQLite store behind a storage feature.
  - Optional caller-supplied store adapter.
- Persist task spec, status, timestamps, result, error, parent/root run ids,
  cancellation requests, timeouts, and control decisions.
- Add lifecycle history, not only latest state.
- Support replay/listing by parent run, root run, thread id, task kind, status,
  and created-at window.

Acceptance criteria:

- A process restart does not lose detached or awaiting orchestration tasks.
- Supervisors can list, wait, cancel, kill, and inspect tasks through the SDK
  store contract.
- OpenHuman can retire most bespoke task status/tombstone persistence in
  `running_subagents.rs`.

### 5. SQLite Storage Compatibility

Status: partially present.

TinyAgents has a `SqliteCheckpointer`, but enabling the `sqlite` feature pulls a
specific `rusqlite` / `libsqlite3-sys` version. OpenHuman already depends on a
different SQLite native-link version, so it cannot enable that feature and had
to implement `SqlRunLedgerCheckpointer`.

Implement one or more compatibility paths:

- Make SQLite support trait-first and allow external connection adapters.
- Provide a version-flexible storage layer, possibly via `sqlx` or a separate
  crate feature matrix.
- Split schema helpers from dependency ownership so apps can create the tables
  using their own SQLite connection.
- Expose a small `CheckpointStore` persistence trait below `Checkpointer`.

Acceptance criteria:

- Applications that already own SQLite can use TinyAgents durable checkpoints
  without native-link conflicts.
- OpenHuman can replace `SqlRunLedgerCheckpointer` with an SDK-supported adapter
  or a thin schema integration.
- Storage features remain opt-in and keep the default crate dependency-light.

### 6. Production Event And Status Journals

Status: partially present.

TinyAgents has `HarnessEventJournal`, `StoreEventJournal`, `HarnessStatusStore`,
and `HarnessRunStatus`. OpenHuman still bridges TinyAgents events into its own
progress system, cost tracker, run ledger, and UI status stream.

Implement:

- Durable event journals with cursors, replay windows, filters, compaction, and
  redaction hooks.
- Status stores with parent/root lineage, thread-scoped listing, phase details,
  active tool/model call ids, usage totals, cost totals, and terminal summaries.
- Event filters for UI surfaces: text stream only, tool timeline, cost updates,
  graph lifecycle, errors, task lifecycle.
- Redaction policies for prompts, tool args, tool results, PII, secrets, and
  provider payloads.
- Stable event ids and offset semantics across process restarts.

Acceptance criteria:

- A UI can attach late and reconstruct a run without subscribing at start time.
- A supervisor can query every active descendant of a root run.
- OpenHuman event bridges become mostly format adapters, not state owners.

### 7. Cost, Usage, And Budget Enforcement

Status: partially present.

TinyAgents has `Usage`, `UsageTotals`, `CostTotals`, and accounting middleware.
OpenHuman still owns richer budget behavior, global cost trackers, per-session
rollups, budget stop hooks, and token/cost dashboard data.

Implement:

- A budget middleware that can preflight, enforce, and reconcile costs.
- Per-run and recursive root-run budgets for input, output, cached input,
  reasoning tokens, total tokens, and money.
- Distinguish provider-reported usage from estimated usage.
- Track cached-token pricing, reasoning pricing, embeddings, image/audio usage,
  and tool/provider fees where present.
- Add budget events: preflight, reservation, spend, refund/reconcile, warn,
  exceeded, blocked.

Acceptance criteria:

- A caller can stop a recursive harness/graph run when a root budget is
  exhausted.
- Budget totals roll up from child/sub-agent runs without custom side channels.
- OpenHuman cost UI can read TinyAgents-normalized records or a thin projection.

### 8. Model Catalog And Provider Resolution

Status: partially present.

TinyAgents has `ModelProfile`, including provider, model, modalities, tool
calling, streaming, structured output, reasoning, and token windows. OpenHuman
still has provider catalog logic and local model capability inference that drive
fallback, token budgeting, and routing.

Implement:

- SDK-owned model catalog snapshots with provider, model id, display name,
  lifecycle status, context windows, modalities, streaming support, reasoning,
  structured-output support, and pricing keys.
- Capability-driven model resolution: required capabilities, fallback chains,
  local/cloud preferences, and provider health.
- Runtime profile discovery hooks for local models.
- Pricing table integration that maps `ModelProfile` to `CostTotals`.

Acceptance criteria:

- Model selection can be expressed in TinyAgents policy instead of
  OpenHuman-only routing code.
- Fallback can reject models that lack required tool, vision, structured-output,
  context-window, or reasoning capabilities.
- Token budgeting can use the resolved model's real context window.

### 9. Dynamic Tool Exposure And Allowlist Policy

Status: partially present.

TinyAgents can run with a provided tool registry, but OpenHuman needs per-agent,
per-tier, per-sub-agent, and per-task allowlists. Tool visibility depends on
security tier, workspace roots, parent/child delegation policy, model
capabilities, and whether the run is background or interactive.

Implement:

- A tool selection middleware that receives run context, agent identity, task
  kind, parent policy, and model profile.
- Allowlist/denylist composition with explicit inheritance rules.
- Explainable exposure decisions for audit/debugging.
- Fail-closed behavior when policy metadata is missing.

Acceptance criteria:

- Sub-agents inherit only the tools they are allowed to call.
- Tool exposure decisions are visible in run events or observations.
- OpenHuman can remove adapter-local allowlist enforcement from most call paths.

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
`DetachedTaskRegistry` owns process-local owner checks, wait/timeout,
cooperative-cancel-before-abort, steering lookup, and bounded terminal cleanup.

A4 (`docs/runtime-comparison/plan.md` Phase 2) put `RunQueue<Message>` on the
loop path: `RunContext::with_run_queue(RunQueueHandle)`; `Steer` drained after
each tool batch and at a natural finish, `Followup` at a natural finish (one
more turn), `Collect` onto `AgentRun::collected`; `RunPolicy::queue_mode`
(`All` | `OneAtATime`); `AgentEvent::QueuedMessageApplied { lane, count }`.
`SteeringHandle` is unchanged. OpenHuman's `agent/harness/run_queue/` can be
deleted in favour of the SDK's. See
[`docs/modules/harness/runtime.md`](modules/harness/runtime.md#queued-steering-and-follow-ups-a4).

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

### 14. Deferred Tool Calls (A2)

Status: shipped (harness); durability stays host-owned.

Landed as `docs/runtime-comparison/plan.md` Phase 2 item A2. A tool call now
leaves the loop as a typed, resumable output instead of `Err(Interrupted)`:
`ToolPolicy.access.approval_required`, `Err(TinyAgentsError::ApprovalRequired
{ metadata })` / `CallDeferred { metadata }` (from a tool or a `before_tool`
middleware), or a `ToolRegistry::register_external(schema)` tool all produce
`AgentRun::deferred = Some(DeferredToolRequests { calls, approvals,
metadata })` after the batch's other calls run. Resume with
`AgentHarness::resume_deferred` / `AgentTurnRequest::with_deferred_results`
and `DeferredToolResults { approvals: ToolApprovalDecision::{Approve,
ApproveWithArgs, Deny}, calls: DeferredCallResult::{Result, Retry, Failed} }`;
`remaining()` reports unresolved ids. A `DeferredToolHandler` on the harness
resolves inline; `HumanApprovalMiddleware::with_approval_outcome` returns
`ApprovalOutcome::{Allow, Deny, Defer}`. Events: `ToolDeferred`,
`ToolApproved`, `ToolDenied`. OpenHuman's `security/approval::ApprovalGate`
becomes a `DeferredToolHandler`. Persistence of `run.messages` +
`run.deferred` is the host's (the session ledger depends on the harness, so
the loop cannot write it); see
[`docs/modules/harness/tool.md`](modules/harness/tool.md#deferred-tool-calls-approval-and-external-execution-a2).

### 15. Registry Diagnostics And Introspection

Status: partially present.

TinyAgents has registry primitives. OpenHuman still needs richer diagnostics for
duplicate components, alias resolution, component health, model/provider/tool
capabilities, and event listener wiring.

Implement:

- Registry snapshot export with models, tools, middleware, graph nodes,
  checkpointers, task stores, event listeners, and aliases.
- Duplicate and shadowing diagnostics.
- Health/status probes for registered providers and stores.
- Machine-readable component dependency graph.
- Optional DOT/JSON graph export for runtime components, not only graph nodes.

Acceptance criteria:

- A CLI or UI can show exactly what TinyAgents components are active.
- Registry failures are actionable without inspecting app-specific logs.
- OpenHuman dead-code audits can map old modules to SDK-owned registry entries.

### 17. Storage And Graph Conformance

Status: missing as a standardized SDK suite.

Durable graphs and task stores are hard to migrate safely without a shared
contract test suite.

Implement:

- Checkpointer conformance for memory, file, SQLite, and caller-supplied stores.
- TaskStore conformance for lifecycle transitions, filters, cancellation,
  timeout, kill, restart/replay, and concurrent writes.
- Graph conformance for `Send`, reducers, interrupts, resume, max concurrency,
  dynamic routing, fanout failure policy, and deterministic result collection.

Acceptance criteria:

- Storage adapters can be swapped without changing graph behavior.
- Durable interrupt/resume semantics are proven across backends.
- Parallel-agent helpers have regression tests for order, failure, timeout, and
  cancellation.

## Implementation Order

1. Define API contracts for tool policy, unknown-tool handling, streaming delta
   channels, durable task storage, storage adapters, and control outcomes.
2. Implement the lowest-level data types and traits behind non-breaking
   defaults.
3. Add in-memory implementations first.
4. Add durable stores and compatibility adapters second.
5. Add middleware helpers and high-level graph helpers.
6. Migrate OpenHuman adapters to the new SDK surfaces.
7. Remove OpenHuman-specific compatibility shims once the SDK behavior is
   equivalent.
8. Implement conformance and regression tests last.
