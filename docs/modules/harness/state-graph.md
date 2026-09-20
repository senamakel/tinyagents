# Harness State Graph Runtime

This document describes what actually exists in the workspace today. An
earlier draft of this file sketched an aspirational `StateGraph<S>` design
inspired by an OpenHuman PR; none of those types (`GraphState`, `Node<S>`,
`Command::Fork`, a `StateGraph` builder) were ever implemented under
`tinyagents-harness`. The real state-graph runtime lives in the
`tinyagents-graph` crate (`crates/tinyagents-graph/src/`), documented in
`docs/modules/graph/`, and this page now only covers the one harness-facing
surface that is genuinely new: the compiled-graph rendition of the agent loop
(A5, `docs/runtime-comparison/langgraph.md` §4 and
`docs/runtime-comparison/pydantic-ai.md` §4's `iter` API).

## Where the graph runtime actually lives

`tinyagents-graph`'s `GraphBuilder`/`CompiledGraph` (`crates/tinyagents-graph/src/builder/`,
`crates/tinyagents-graph/src/compiled/`) is the durable, typed, Pregel-style
state graph: partial updates and reducers (`reducer`), commands and
interrupts (`command`), checkpointing (`checkpoint`, with `InMemoryCheckpointer`,
`FileCheckpointer`, and an optional `SqliteCheckpointer`), streaming/events
(`stream`), subgraph embedding (`subgraph`), and recursion tracking
(`recursion`). See `docs/modules/graph/` for that design in full; nothing on
this page duplicates it.

`tinyagents-harness` does not implement its own competing graph engine. What
it exposes, in `tinyagents_harness::agent_loop::phases`, is a **seam**: typed
phase contracts (`TurnPlan`, `ModelOutcome`, `ToolBatchOutcome`, `Settlement`)
and a `LoopDriver` trait that lets `AgentHarness` delegate its loop execution
to an alternate engine. `tinyagents-graph` is the only implementor of that
seam, because the dependency direction in this workspace is graph → harness
(never the reverse) — a compiled-graph rendition of the harness's own loop
therefore has to live in the graph crate, wired back in through this trait.

## The agent loop as a `CompiledGraph` (A5)

`tinyagents_graph::agent_loop` (`crates/tinyagents-graph/src/agent_loop/`)
compiles the harness's default `plan -> model -> tools -> settle` loop into
an ordinary `CompiledGraph<LoopState, LoopUpdate>`. `LoopState` is a
whole-state graph value (`LoopUpdate = LoopState`, built with
`GraphBuilder::overwrite()`): every node returns the complete next state
rather than a partial patch. It is `Serialize`/`Deserialize`, unlike the
harness's own `RunContext` (deliberately not serializable), so a
graph-driven run can be checkpointed mid-run — including at an interrupt —
and resumed.

Three entry points, sharing one set of node bodies
(`runtime::plan_node`/`model_node`/`tools_node`/`settle_node`):

- **`compile_loop(rt: Arc<LoopRuntime<State, Ctx>>) -> Result<CompiledGraph<LoopState, LoopUpdate>>`**
  builds the real compiled graph over an owned `LoopRuntime` (harness/state
  `Arc`'d, `RunContext`/`AgentRun`/`HarnessRunStatus` owned). This is the
  graph that gets checkpointing, `resume`, and step-by-step control.
- **`AgentLoopGraphExt::iter(harness: Arc<AgentHarness<..>>, app_state, ctx, input) -> LoopIter`**
  mirrors pydantic-ai's `Agent.iter`: `LoopIter::next()` runs exactly one
  node activation and reports a `LoopStep` (which node ran, what runs next,
  whether it interrupted); `override_next(node)` redirects the next
  activation; `state()` reads the committed `LoopState`; `run_to_end()` steps
  until completion or the first unresolved interrupt.
- **`GraphLoopDriver`** implements `phases::LoopDriver` and is installed with
  `AgentHarness::with_loop_driver` plus
  `RunPolicy::execution = LoopExecution::Graph` (the default,
  `LoopExecution::Direct`, is the original `run_loop` body, byte-for-byte
  unchanged). This makes `AgentHarness::invoke` itself run the graph-rendition
  node bodies.

### Why `GraphLoopDriver` does not build a `CompiledGraph`

`phases::LoopDriver::drive` is handed a **borrowed** `&AgentHarness`/`&State`
and `&mut RunContext`/`&mut AgentRun`/`&mut HarnessRunStatus` — the same
shape `run_loop` itself uses. `GraphBuilder::add_node`'s closures must be
`Send + Sync + 'static`, so a `CompiledGraph`'s nodes can only close over
genuinely owned (`Arc`'d) data. Building an `Arc<AgentHarness>` from a bare
`&AgentHarness` is not possible in safe Rust, and this workspace denies
`unsafe_code`. Forcing every `with_loop_driver` caller to already hold an
`Arc<AgentHarness>` before installing a driver would also be circular (the
harness does not exist as an `Arc` until after it is fully built, including
its driver).

So `GraphLoopDriver::drive` instead calls the exact same node bodies
directly, in a hand-rolled loop, against the real borrowed `&mut` state — no
`Arc`, no `Mutex`, no `CompiledGraph`. This is sound with zero unsafe code
because a borrowed async call needs no `'static` bound. The trade-off: an
interrupt raised through `AgentHarness::invoke` (`RunPolicy::execution ==
Graph`) still ends the run cleanly (mirroring a steering pause, or
`TinyAgentsError::Interrupted` for a `MiddlewareControl::Interrupt` — see
below) but is **not** a resumable `CompiledGraph` checkpoint. A caller that
wants graph-level checkpoint/resume across an interrupt should drive the loop
through `compile_loop`/`LoopIter` directly instead of through
`AgentHarness::invoke`.

### Routing and interrupts

Every node routes explicitly (`mark_command_routing` on all four nodes) via
`Command::goto`, driven by `tinyagents_harness::context::MiddlewareControl`:

| `MiddlewareControl` | Graph routing |
| --- | --- |
| `Continue` / `UpdateState` | falls through to whatever the calling node had already determined the turn's natural next step to be (tool routing from `model`, `plan` from `tools`) |
| `JumpTo(Model)` | routes to `plan` (closing any unanswered tool calls first) |
| `JumpTo(Tools)` | routes to `tools` if there are pending calls, else `settle` |
| `JumpTo(End)` / `StopWithFinal` | routes to `settle` with `finished = true` |
| `Interrupt { node, message }` | `NodeResult::Interrupt` — a real, checkpointable `crate::Interrupt` returned directly by the `plan`/`model`/`tools` node bodies through `compile_loop`/`LoopIter` (these nodes are marked as interrupt points for the export only — via the `NodeMeta` flag directly, not `GraphBuilder::mark_interrupt`, which now aliases the real `interrupt_before` pause and would double-pause every activation); through `GraphLoopDriver` this instead surfaces as `TinyAgentsError::Interrupted`, matching `run_loop`'s own behavior for this control |

A steering pause (`SteeringOutcome::Pause`, checked in `plan_node`) also
produces a `NodeResult::Interrupt`, distinguished from a middleware interrupt
by its id suffix (`-steering-pause`); `GraphLoopDriver` treats that one as a
clean pause (`run.paused`, `Ok(())`), exactly like `run_loop`'s
`LoopExit::Paused`.

`RunLimits` (model/tool call caps) are enforced through the same
`RunContext::limits`/`LimitTracker` the direct loop uses, reconciled against
`RunPolicy::limits` with the identical (`runtime::reconcile_call_limits`)
logic `run_loop_body` applies at the top of a run, so
`RunConfig::with_max_model_calls` and `RunPolicy::limits` interact
identically under both engines. A cap hit surfaces as
`TinyAgentsError::LimitExceeded` (or a clean `LimitStop`-style finish under
`LimitBehavior::StopWithPartial`) either way.

### Tool batch execution

The `tools` node runs a turn's whole tool-call batch in **one** node
activation, via `tinyagents_harness::agent_loop::phases::execute_tool_batch`,
rather than one graph node (or one `Send` fan-out branch) per call. The
direct loop's serial-admission / serial-or-concurrent-dispatch decision (see
`tinyagents_harness::agent_loop::tools`) is call-count- and
middleware-dependent; re-deriving it as graph topology would either duplicate
that decision as a router (two sources of truth) or lose the exact
ordering/budget guarantees the direct loop promises. Reusing the harness
function as-is keeps ordering, concurrency-eligibility, and budget/limit
semantics identical to the direct loop by construction — at the cost of a
coarser graph: a `tools` activation is atomic from the executor's point of
view, so resuming after an interrupt mid-batch re-runs the whole batch, not
just its unfinished calls.

### Documented scope (not a byte-for-byte reimplementation)

`tinyagents_graph::agent_loop` covers the common path: tool calling,
structured output (`ResponseFormat::Auto`/`JsonSchema`, provider-schema and
tool-call-fallback strategies), the output-validation retry loop (A3), run
limits, `MiddlewareControl` routing, and steering. It intentionally does
**not** cover: host-model routing (`HostCapabilities`), cross-provider
handoff transforms, the deferred-tool discovery bridge, truncated-empty-
response recovery/retry, `RunPolicy::retry`/`fallback`'s built-in retry loop
(a registered `ModelMiddleware` still runs), response caching, and
`StructuredStrategy::Prompted`/`ToolCallUnion` or
`EndStrategy::Early`/`Exhaustive` (A6). `resolve_structured_plan` also
resolves a profile-driven `ResponseFormat::Auto` choice against the
*default* model binding rather than the turn's actually-resolved model.
`RunPolicy::execution` defaults to `LoopExecution::Direct`, so no existing
caller is affected unless it opts in.

Equivalence between the two engines is covered by
`crates/tinyagents-integration-tests/tests/loop_as_graph.rs`: scripted tool
call, structured output, output-validation retry, limit stop, approval
interrupt, and steering-inject scenarios run through both `Direct` and
`Graph`, asserting matching transcript/structured output/usage and that the
direct run's `AgentEvent` kind sequence is a subsequence of the graph run's
(the graph engine may emit extra events — for example `ToolsAdvertised` once
per turn instead of once per run — but never fewer or reordered ones); plus
checkpoint+resume across an interrupt (`InMemoryCheckpointer` and
`FileCheckpointer`) and `LoopIter` stepping with `override_next`.

## See also

- `docs/modules/graph/` — the underlying `GraphBuilder`/`CompiledGraph`
  runtime this rendition is built from.
- `docs/modules/graph/subagents-recursion.md` — how a compiled graph (this
  one included) nests inside the recursive sub-agent/subgraph architecture.
- `docs/runtime-comparison/langgraph.md` §4 and
  `docs/runtime-comparison/pydantic-ai.md` §4 — the comparative design notes
  A5 responds to.
