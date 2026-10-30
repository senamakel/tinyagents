# Harness Observability And Events

Observability is a first-class harness surface. Every important lifecycle step
should emit typed events that can drive logs, traces, streaming UIs, tests, and
durable run replay.

## Source Inspiration

LangChain uses callbacks, tracers, event streams, usage callbacks, and LangSmith
integration:

- callbacks:
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/callbacks>
- tracers:
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/tracers>
- usage callback:
  <https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/callbacks/usage.py>
- runnable event stream tests:
  <https://github.com/langchain-ai/langchain/tree/master/libs/core/tests/unit_tests/runnables>

TinyAgents should use typed Rust events rather than stringly callback names.

## Responsibilities

- Emit typed lifecycle events.
- Support multiple event sinks.
- Support redaction before persistence or external export.
- Support streaming subscribers.
- Support deterministic test collectors.
- Support durable event journals through `store`.
- Preserve parent/root run relationships.
- Include usage, cost, cache, retry, fallback, and latency data.
- Capture model-call and tool-call latency as first-class metrics.


## Execution Status Store

Harness runs need a readable status surface. Direct model calls and model-tool
loops publish their current execution status so external listeners can inspect runs
without holding an in-process stream.

```rust
pub struct HarnessRunStatus {
    pub run_id: RunId,
    pub parent_run_id: Option<RunId>,
    pub root_run_id: RunId,
    pub thread_id: Option<ThreadId>,
    pub component: ComponentId,
    pub status: ExecutionStatus,
    pub current_phase: HarnessPhase,
    pub model_calls: usize,
    pub tool_calls: usize,
    pub active_model_call: Option<CallId>,
    pub active_tool_calls: Vec<CallId>,
    pub last_event_id: Option<EventId>,
    pub usage: UsageTotals,
    pub cost: CostTotals,
    pub started_at: SystemTime,
    pub updated_at: SystemTime,
    pub ended_at: Option<SystemTime>,
    pub error: Option<HarnessErrorSummary>,
    pub metadata: serde_json::Value,
}

pub enum HarnessPhase {
    Starting,
    LoadingMemory,
    BuildingPrompt,
    CallingModel,
    ValidatingStructuredOutput,
    CallingTools,
    Summarizing,
    SavingMemory,
    Completed,
    Failed,
    Cancelled,
}
```

Status should move through `Pending`, `Running`, `Waiting`, `Completed`,
`Failed`, or `Cancelled` states while `current_phase` describes the active
harness operation. Nested harness calls can read child status through `parent_run_id` and
`root_run_id`.

Run-level latency can be derived from `started_at` and either `ended_at` or
`updated_at`. This status-derived view is compact and cheap, but it only reports
end-to-end elapsed time; per-call latency comes from the event journal.

```rust
#[async_trait]
pub trait HarnessStatusStore: Send + Sync {
    async fn put_status(&self, status: HarnessRunStatus) -> Result<()>;
    async fn get_status(&self, run_id: RunId) -> Result<Option<HarnessRunStatus>>;
    async fn list_by_thread(&self, thread_id: ThreadId) -> Result<Vec<HarnessRunStatus>>;
}
```

The status store should be pluggable and can be backed by in-memory state,
JSONL, MongoDB, SQLite, or another store backend. Status writes must be compact:
they should include counters, ids, phase, error summaries, and timestamps, not
full prompts, tool outputs, or provider payloads.

## Event Shape

```rust
pub struct HarnessEvent {
    pub id: EventId,
    pub run_id: RunId,
    pub parent_run_id: Option<RunId>,
    pub root_run_id: RunId,
    pub thread_id: Option<ThreadId>,
    pub component: ComponentId,
    pub time: SystemTime,
    pub tags: Vec<String>,
    pub metadata: serde_json::Value,
    pub kind: HarnessEventKind,
}
```

Event kinds should include:

- `run.started`
- `run.completed`
- `turn.started` / `turn.completed` / `message.appended` (see [terminal outcome and turn lifecycle](terminal-outcome.md))
- `run.failed`
- `model.started`
- `model.delta`
- `model.completed`
- `model.failed`
- `tool.started`
- `tool.progress`
- `tool.completed`
- `tool.failed`
- `middleware.started`
- `middleware.completed`
- `middleware.failed`
- `memory.loaded`
- `memory.saved`
- `summary.created`
- `cache.hit`
- `cache.miss`
- `cache.prompt_miss`
- `usage.recorded`
- `cost.recorded`
- `retry.scheduled`
- `fallback.selected`
- `rate_limit.waited`
- `limit.reached`
- `stream.closed`

## Physical Provider Dispatches

`observability::physical_dispatch::ObservedChatModel` wraps a concrete provider
and calls `PhysicalDispatchObserver::on_start` only after the caller's budget
wrapper has admitted the request. Its terminal callback reports provider
usage, model, finish reason, the serving provider, the response id and the time
to first output, without prompts, outputs, raw errors, or other provider
metadata. Dropping an in-flight invocation or unfinished stream reports
cancellation exactly once. Streaming calls remain streaming and finish only on
a terminal stream item.

`ModelBudget::wrap_model` installs a task-local adapter automatically. A scoped
turn observer receives one `TurnObservation::ModelDispatch` for each provider
invocation while the existing logical `TurnObservation::Model` event remains
available for compatibility. Calls refused before provider admission produce
no physical dispatch event.

### Who served a call, and when its first token arrived

Both `TurnObservation::Model` and `TurnObservation::ModelDispatch` (and
`PhysicalDispatchOutcome`) carry three provider-neutral facts besides usage
(which already includes `reasoning_tokens`):

| Field | Source | `None` when |
| --- | --- | --- |
| `served_by` | a top-level `provider` string on the response's `raw` (OpenRouter names the upstream provider that served the call there, for example `"DeepInfra"`) | the response does not name one, as for a provider that serves calls itself |
| `response_id` | the provider's response id, which the adapters keep on `AssistantMessage::id` for buffered and streamed responses alike (`chatcmpl-…`, OpenRouter's `gen-…`, `msg_…`); a top-level `raw.id` is the fallback | the provider sent no id |
| `first_token_ms` (`first_output` on the outcome) | the time from the call's start to its first streamed output: text, reasoning or a tool-call fragment; stream markers, usage updates and empty deltas do not count | the call was buffered, or the stream produced no output |

For `Model`, the agent loop records the first output of each streaming attempt
on the run context, so with retries the time belongs to the attempt whose
response the call returned, measured from the start of the logical call. For
`ModelDispatch`, it is measured within the one physical invocation. A streamed
OpenAI-compatible response keeps `provider` in `raw` because it is outside the
chunk schema; the chunk `id` is not kept in `raw`, which is why `response_id`
reads the message id first.

## Latency Metrics

The observability module exposes `AgentLatencyMetrics` for durable latency
rollups. Metrics are derived from `AgentObservation` timestamps rather than
payload metadata, so they work for in-memory journals, JSONL-backed journals,
and redacted sinks that preserve structural ids.

Agent latency metrics include:

- `run_elapsed_ms`: first `run.started` to first terminal `run.completed` or
  `run.failed`
- model call latency: `model.started` to `model.completed`, correlated by
  `CallId`
- tool call latency: `tool.started` to `tool.completed`, correlated by `CallId`
- total, max, and average latency helpers for completed model calls
- total, max, and average latency helpers for completed tool calls

Tool-call latency is required because tools are often the slowest part of an
agent turn: retrieval, browser work, RPC calls, file processing, and sub-agent
tools all need to be visible independently from provider latency.

```rust
let observations = journal.read_from(run_id, 0).await?;
let metrics = AgentLatencyMetrics::from_observations(&observations);

let run_ms = metrics.run_elapsed_ms;
let slowest_tool_ms = metrics.max_tool_ms;
let average_tool_ms = metrics.average_tool_ms();
```

Incomplete calls are ignored. A `model.started` without `model.completed` or a
`tool.started` without `tool.completed` does not produce a latency record
because there is no terminal timestamp. Failed runs still report
`run_elapsed_ms` when `run.failed` is observed.

## Event Journal And Listener Replay

The harness should support both live event subscribers and durable replay. A UI
or supervisor can then connect after a run has started and reconstruct what
happened from the last known offset.

```rust
#[async_trait]
pub trait HarnessEventJournal: Send + Sync {
    async fn append(&self, event: HarnessEvent) -> Result<EventOffset>;
    async fn read_from(
        &self,
        run_id: RunId,
        offset: EventOffset,
    ) -> Result<Vec<HarnessEventRecord>>;
}
```

Listener filters should include:

- run id, root run id, parent run id, thread id, component id, model id, or tool
  name
- event families such as run, model, tool, middleware, memory, summary, cache,
  usage, cost, retry, fallback, rate-limit, store, or stream
- minimum severity for errors and warnings
- replay offset
- redaction profile

Live event delivery should not block model or tool execution unless the run
policy explicitly requires a mandatory observer. Durable listeners should replay
from the journal rather than relying on in-memory broadcast delivery.

## Observability Cache

The harness cache feature can store derived observability projections. These
records help dashboards and tests read status quickly, but they are not the
authoritative source of run history.

Cacheable observability projections include:

- latest run status by run id
- latest thread status rollup
- usage and cost rollups
- model-call and tool-call timing summaries
- agent latency rollups from durable observations
- cache-hit summaries by model, tool, or prompt version
- redacted prompt/message previews for UIs
- event replay cursors for test collectors

Every cached projection must include a source event offset and projection
version. When the event offset advances, the cached projection is stale.

## Event Sinks

```rust
#[async_trait]
pub trait EventSink: Send + Sync {
    async fn emit(&self, event: HarnessEvent) -> Result<()>;
}
```

Built-in sinks:

- no-op sink
- in-memory collector
- stdout/debug sink
- JSONL append sink
- broadcast stream sink
- redacting sink wrapper
- fan-out sink

Event emission should not make the model or tool call fail unless policy says
observability is required. When best-effort sinks fail, the harness should emit
or record a sink error where possible.

## Stable Event Ids And Delta Attribution

Emitted `EventId`s are scoped by a stream prefix so they are stable across
process restarts and never collide between concurrent runs. An `EventSink`
built with `EventSink::with_stream_id(prefix)` mints ids of the form
`{prefix}-evt-{offset}`, where `offset` is the sink's monotonic emission
counter. `RunContext` seeds the prefix from the run id
(`EventSink::with_stream_id(config.run_id.as_str())`), so replaying the same run
re-mints identical ids for the same `(stream_id, offset)`. `EventSink::new()`
(the default) instead uses a process-unique prefix (`s<n>`) so two default sinks
never collide at offset 0.

```rust
use tinyagents_harness::events::{AgentEvent, EventSink};

// Two "processes" replaying the same run re-mint identical, stable ids.
let first = EventSink::with_stream_id("run-42");
let second = EventSink::with_stream_id("run-42");
let a = first.emit(AgentEvent::StateUpdate);
assert_eq!(a.id.as_str(), "run-42-evt-0");
assert_eq!(a.id, second.emit(AgentEvent::StateUpdate).id);

// A different run never collides even though both restart at offset 0.
let other = EventSink::with_stream_id("run-99");
assert_ne!(other.emit(AgentEvent::StateUpdate).id, a.id);
```

Because a sink is shared across a recursive run tree, a streamed delta must be
attributable to *its* run without depending on which sink instance delivered it.
`AgentEvent::ModelDelta` therefore carries `run_id` alongside `call_id` and the
`MessageDelta`, so a UI can route each streamed chunk to its run/thread lineage
directly. The `invoke_streaming*` agent-loop methods emit one
`ModelDelta { run_id, .. }` per streamed delta, each tagged with the emitting
run's id.

## Redaction

Redaction should happen at sink boundaries. Internal typed events may carry
full details while a persistent or remote sink can receive redacted payloads.

Redaction policies should handle:

- message content
- tool arguments
- tool results
- provider raw payloads
- metadata
- store keys
- secrets
- PII
- artifact paths

Redacted events should preserve structural fields such as ids, event kinds,
timings, and counters so traces remain useful.

## Payload Capture

The event stream is **payload-free by default**: `AgentEvent::ModelCompleted`
and `AgentEvent::ToolCompleted` carry only ids and usage, never the prompt,
completion, tool arguments, or tool result. This keeps journals and exporters
safe for privacy-sensitive deployments without any configuration.

`RunPolicy::capture` (a `PayloadCapture { model_io, tool_io }`) opts a run into
carrying those payloads onto the completion events:

- `model_io` — attaches the request messages (`input`) and the model completion
  (`output`) to every `ModelCompleted`.
- `tool_io` — attaches the tool arguments (`input`) and the tool result
  (`output`) to every `ToolCompleted`.

Capture is snapshotted inside the agent loop before the request/call value is
moved into the model or tool wrap onion, so it never perturbs execution. Because
captured payloads ride the ordinary event pipeline, a `RedactingSink` still
masks them before they reach a journal or exporter, and the durable journal
replays them verbatim.

Downstream, the Langfuse exporter reads these fields to populate the
Input/Output panels of a generation (`generation-create`) and a tool
observation (`span-create`). With capture disabled the fields are absent and the
observation renders with ids, timing, and usage only. The harness exporter also
defaults the trace-level metadata from the run lineage (root/first run ids and
parent), so a trace is correlatable even when the caller passes no metadata.

## Langfuse Observation Tree

The harness exporter mirrors LangChain's callback **run tree** rather than a
flat event list. For each distinct `run_id` in a batch it emits one
`span-create` (`{trace_id}:run:{run_id}`, named `agent` for the top-level run
and `sub-agent` for any spawned child), folding the `RunStarted`/`RunCompleted`/
`RunFailed` events into that span's start/end window and ERROR status. Every
model generation, tool span, and remaining event then nests under its run span
via `parentObservationId`.

Because run-span ids are deterministic and trace-namespaced, the tree spans
**separately exported batches**: a sub-agent run whose observations are shipped
in their own batch carries `parent_run_id`, so its run span resolves to
`{trace_id}:run:{parent_run_id}` and nests under the parent agent's span in the
same trace, preserving the nested agent-run lineage.

## Scores

`LangfuseClient::create_score` (and the offline `build_score_batch`) mirrors
Langfuse's `createScore` — the way a LangChain/LangGraph trace is graded after
the fact (human ratings, LLM-as-judge, regression metrics). A `LangfuseScore`
is numeric, categorical, or boolean; scoped to a `trace_id` and optionally to a
single observation id; and carries an optional comment. Score ids default to a
deterministic value derived from the target and metric name, so re-scoring the
same target upserts rather than duplicating. Point the score's `trace_id` at an
exported trace (recall both exporters default the trace id to the run's
`root_run_id`) to attach the evaluation to that run.
