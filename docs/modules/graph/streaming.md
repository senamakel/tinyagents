# Graph Streaming And Events

The graph stream should support low-level event streams and high-level
projection streams.

Core stream modes:

- `values`: full state values after each step
- `updates`: per-node/per-task state updates
- `messages`: harness message or token deltas emitted by model nodes
- `custom`: arbitrary user stream writes from inside nodes
- `checkpoints`: checkpoint payloads
- `tasks`: task start and task result payloads
- `debug`: checkpoints plus task internals
- `events`: all graph lifecycle events

Typed stream part:

```rust
pub enum StreamPart<State, Output> {
    Values {
        namespace: Vec<String>,
        data: Output,
        interrupts: Vec<Interrupt>,
    },
    Updates {
        namespace: Vec<String>,
        data: IndexMap<NodeId, StateUpdate>,
    },
    Messages {
        namespace: Vec<String>,
        message: Message,
        metadata: StreamMetadata,
    },
    Custom {
        namespace: Vec<String>,
        data: serde_json::Value,
    },
    Checkpoint {
        namespace: Vec<String>,
        data: CheckpointPayload<State>,
    },
    Tasks {
        namespace: Vec<String>,
        data: TaskStreamPayload,
    },
    Debug {
        namespace: Vec<String>,
        data: DebugPayload<State>,
    },
}
```

Event stream:

```rust
pub enum GraphEvent {
    RunStarted { run_id: RunId, graph_id: GraphId },
    RunStreamingStarted { run_id: RunId },
    StepStarted { step: usize, active: Vec<NodeId> },
    TaskStarted { task_id: TaskId, node: NodeId, triggers: Vec<String> },
    TaskCompleted { task_id: TaskId, node: NodeId },
    TaskCached { task_id: TaskId, node: NodeId },
    TaskFailed { task_id: TaskId, node: NodeId, error: String },
    StateUpdated { node: NodeId, update: serde_json::Value },
    RouteSelected { node: NodeId, routes: Vec<RouteTarget> },
    ContextForked { parent_task_id: TaskId, child_task_id: TaskId },
    ContextForkJoined { parent_task_id: TaskId, child_task_id: TaskId },
    SubgraphStarted { node: NodeId, child_run_id: RunId, namespace: Vec<String> },
    SubgraphCompleted { node: NodeId, child_run_id: RunId },
    SubAgentStarted { node: NodeId, agent: ComponentId, child_run_id: RunId },
    SubAgentCompleted { node: NodeId, agent: ComponentId, child_run_id: RunId },
    RecursionDepthChanged { depth: usize },
    CheckpointSaved { checkpoint_id: CheckpointId },
    InterruptEmitted { interrupt: Interrupt },
    RunDraining { run_id: RunId, reason: String },
    RunCompleted { run_id: RunId },
    RunFailed { run_id: RunId, error: String },
    Custom { name: String, payload: serde_json::Value },
}
```

Streaming requirements:

- graph runs can be consumed as an async stream
- streaming does not require waiting for final state
- every streamed event carries run id, thread id, namespace, step, and node/task
  metadata when available
- subgraph streams preserve nested namespaces
- harness streams from model/tool/sub-agent nodes are forwarded with graph node
  context
- subscribers can filter graph events, harness events, sub-agent events, state
  updates, task payloads, messages, and checkpoints
- a typed run stream should expose final output, interrupted status, and pending
  interrupts even when the caller only subscribed to a subset of projections

## Implementation status (runtime-comparison Phase 3, C3)

The design above is the target shape; the current implementation is a
scoped-down but real subset built around `crate::stream`:

- **`GraphEventEnvelope { run_id, task_id, ns, seq, event }`**
  (`stream/types.rs`) wraps every `GraphEvent` the executor emits.
  `GraphEventSink::emit` takes the envelope, not the bare event — `ns` is the
  emitting graph instance's checkpoint namespace (empty at the top level, one
  segment deeper per level of subgraph embedding) and `seq` is a per-instance
  monotonic counter (`CompiledGraph::sequence`, an `Arc<AtomicU64>`). The
  counter is shared across a clone that only swaps `event_sink` (journal
  wrapping keeps counting where the plain run left off) but is reset to a
  fresh one for a subgraph embedded as a node
  (`subgraph::namespaced`/`CompiledGraph::with_fresh_sequence`) — the deeper
  `ns` already disambiguates that stream, so nothing is gained by chaining
  the parent's sequence into it. `task_id` is `None` until per-task
  correlation ids exist end to end (`docs/runtime-comparison/feature-gaps.md`
  D4); adding it later is additive.
- **`StreamMode::{Tasks, Checkpoints}`** are implemented. `GraphEvent::mode()`
  maps `TaskScheduled`/`TaskStarted`/`TaskCompleted`/`NodeStarted`/
  `NodeCompleted`/`NodeFailed`/`NodeRetryScheduled` onto `Tasks`,
  `CheckpointSaved`/`CheckpointRestored` onto `Checkpoints`; every other kind
  (including the plain run/step lifecycle events) is debug-only detail.
  `stream::project::project_graph_event(event, modes)` applies that mapping.
  `GraphEvent::TaskStarted`/`TaskCompleted { cached }` are new variants
  emitted alongside `NodeStarted`/`NodeCompleted`/`NodeFailed` at the same
  boundary; `cached` is always `false` today (no per-node task cache exists
  yet — D2).
- **`StreamProjection`** (`stream/project.rs`) is the cross-source fold this
  doc's "harness streams … are forwarded with graph node context" bullet
  calls for: `fold_graph_event`/`fold_agent_event` take a `GraphEventEnvelope`
  and a harness `AgentEvent` respectively and append into `messages`,
  `tool_calls`, and `subagents` — each a `Vec<Cursored<T>>` sharing one
  monotonic `cursor` across all three views. `StreamProjection::since(cursor)`
  is the late-attach replay primitive: a consumer that connects after a run
  is already underway asks for everything past the cursor it last saw
  instead of re-reading history. Only `GraphEvent::SubgraphStarted`/
  `SubgraphCompleted` project from the graph side today (as `subagents`
  entries); the harness side covers model deltas
  (`AgentEvent::ModelDelta` → `messages`), tool lifecycle
  (`ToolStarted`/`ToolCompleted`/`ToolFailed` → `tool_calls`), and sub-agent
  lifecycle (`SubAgentStarted`/`SubAgentCompleted` → `subagents`).
- **Not yet implemented**: the typed `StreamPart<State, Output>` enum, the
  `values`/`updates`/`custom` projections from live graph state (those still
  require the graph-state side channel this doc's header note already calls
  out), `RunStreamingStarted`, `ContextForkJoined`, and `RunDraining`. A
  subgraph node does not automatically inherit its parent's `event_sink` —
  nested observability today requires configuring the same sink on both
  explicitly (see `subgraph::test::nested_subgraph_run_yields_envelopes_with_correct_namespace_depth_and_seq`
  for the pattern); full automatic propagation is future work alongside D4.
