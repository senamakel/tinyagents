# harness

Provider-neutral model calls, tools, middleware, and streaming — the harness
runtime is where a single model call becomes a recursive system: it runs the
agent loop (model ⇄ tools), and because a whole harness agent can be wrapped
as a tool via `subagent`, an agent calling a tool *is* an agent calling
another agent. Parent/child run lineage, depth limits, usage/cost roll-up,
steering, and cancellation all flow through here.

The crate is intentionally split by feature: each module directory owns one
substantial part of model/tool orchestration. Within a module, type
definitions live in `types.rs`, behavior in `mod.rs`, unit tests in
`test.rs`, and complex modules additionally carry their own `README.md` — see
the module map below.

## Module map

| Module | Concern |
| --- | --- |
| `agent_loop` | The default model-tool-model agent loop — one model call driven to completion, plus tool dispatch. |
| `artifacts` | Filesystem offload for oversized worker artifacts on long-horizon runs. |
| `cache` | Prompt, response, and layout caches for repeated/replayed requests. |
| `cancel` | Cooperative, runtime-agnostic run cancellation (`CancellationToken`). See [`cancel/README.md`](cancel/README.md). |
| `config` | Crate-owned session configuration, decoupled from any host's config schema. |
| `context` | `RunContext` — the unit of recursion: run configuration and runtime context threaded through every nested layer. |
| `cost` | Additive cost accounting (`CostTotals`) that rolls a child run's cost up into its parent. |
| `error` | The crate-wide `TinyAgentsError` type and `Result` alias every fallible surface funnels through. |
| `events` | The typed observability layer (`AgentEvent`, sinks, listeners) — the live in-process event surface. |
| `handoff` | Progressive-disclosure handoff cache for oversized tool results, keeping bloated payloads out of sub-agent history. |
| `host` | Host capability traits — the seams a host implements to supply product-specific behavior to a generic runtime. |
| `ids` | Identifier newtypes (`RunId`, `CallId`, …) and lifecycle enums used to correlate a recursive run tree. |
| `limits` | Run-scoped limit enforcement (model/tool call caps, wall clock) that keeps recursion bounded. |
| `middleware` | The middleware stack wrapping every level of the recursion identically. See [`middleware/README.md`](middleware/README.md). |
| `model_registry` | Runtime-owned executable model registry, name resolution, and fallback ordering. |
| `multimodal` (feature `multimodal`) | Attachment resolution for `[IMAGE:…]` / `[FILE:…]` markers into model-readable bytes. |
| `no_progress` | Detects a stuck turn (identical failing/successful tool calls) and escalates through a nudge/halt ladder. See [`no_progress/README.md`](no_progress/README.md). |
| `observability` | Durable observability — journals, status stores, sinks — making the live event stream persistent. See [`observability/README.md`](observability/README.md). |
| `prompt` | Prompt assembly — templates and `PromptBuilder` turning runtime values into the final request. |
| `providers` | Model adapters whose behavior depends on TinyAgents-specific prompt dialects (e.g. Claude Code/Agent SDK). |
| `retriever` | Provider-neutral retrieval contracts (`Retriever`) for injecting ranked context into a prompt. See [`retriever/README.md`](retriever/README.md). |
| `retry` | Retry/backoff, model fallback, and rate-limiting policies applied uniformly to every model call. See [`retry/README.md`](retry/README.md). |
| `run_queue` | A generic multi-lane FIFO queue (steer/followup/collect) for messages arriving during an active run. See [`run_queue/README.md`](run_queue/README.md). |
| `runtime` | The harness runtime facade (`AgentHarness`) and invocation-local runtime wiring. See [`runtime/README.md`](runtime/README.md). |
| `steering` | Policy-checked, observable orchestrator → sub-agent steering commands. See [`steering/README.md`](steering/README.md). |
| `store` | Long-term key-value and append-only stream storage backends (`Store`, `AppendStore`, `NamespacedStore`). See [`store/README.md`](store/README.md). |
| `stream` | Higher-level streaming projections (`StreamChunk`, `StreamSink`) from the raw event stream. See [`stream/README.md`](stream/README.md). |
| `structured` | Structured (typed, JSON-schema-validated) output extraction from a model call. |
| `subagent` | First-class sub-agents with recursion-depth tracking — the harness's flagship recursion surface. |
| `summarization` | Explicit message trimming, summarization, and compression policies (the harness's answer to context rot). |
| `testkit` | Deterministic model/tool doubles, event recorder, and trajectory assertions for testing without a live provider. See [`testkit/README.md`](testkit/README.md). |
| `token_estimation` | The crate's shared, structurally-complete token estimator (a port of LangChain's `count_tokens_approximately`). |
| `tool` | Harness-side registration and execution support for canonical (`tinytools`) tools. See [`tool/README.md`](tool/README.md). |
| `tools` (feature `tools`) | Optional builtin harness tools implementing the canonical `tinytools::Tool` interface. |

## How the pieces fit together

1. **Configure**: a host builds a `RunContext` (`context`) carrying its
   `StoreRegistry` (`store`), `ModelRegistry` (`model_registry`),
   `ToolRegistry` (`tool`), limits (`limits`), cancellation token (`cancel`),
   and event sink (`events`).
2. **Run**: `agent_loop` drives the model ⇄ tool cycle for one turn,
   consulting `retry`/`model_registry` on provider failure, `no_progress` and
   `summarization` to keep the loop productive and in-budget, `handoff` to
   keep oversized tool results out of history, and `middleware` around every
   step.
3. **Recurse**: `subagent` lets a tool wrap a whole nested `AgentHarness`
   invocation, so the same loop runs again one level deeper, tracked by
   `ids`/`limits`/`cost` roll-up.
4. **Observe**: every step emits an `AgentEvent` (`events`); `stream` projects
   those onto consumer-facing `StreamChunk`s, and `observability` durably
   journals them.
5. **Extend**: host-owned `AgentMiddleware` wraps the complete run to load and
   save memory, prepare and clean up workspaces, or attach other product policy.
   The harness keeps only the execution seam.
6. **Test**: `testkit` supplies deterministic doubles for every seam above so
   the whole loop is testable without a live provider.

## Feature flags

Cargo features are crate-local: `sqlite`, `tools`, `multimodal`, `tracing`.
Tracing instrumentation is compiled out by default; enable the `tracing`
feature to get it. See the workspace `CLAUDE.md` for the full list across all
crates.
