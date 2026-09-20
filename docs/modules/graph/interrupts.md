# Graph Interrupts And Resume

Interrupts pause execution and return control to the caller. Everything in
this section — the `Interrupt` type, `Command::resume`/`resume_tasks`,
`CompiledGraph::resume`/`resume_from`/`retry`, the `interrupt_before` /
`interrupt_after` selectors, and `response_schema` validation — is
implemented today (`crates/tinyagents-graph/src/command/`,
`crates/tinyagents-graph/src/builder/mod.rs`,
`crates/tinyagents-graph/src/compiled/{step,boundary,resume}.rs`). Only the
"Targeted Human Steering" section at the end remains a target.

The struct as shipped — `id` is a bare `String` (not `InterruptId`) and
there is no `order` field; `task_id` is stamped by the interrupt boundary
with the pausing branch's task id, and `response_schema` is optional:

```rust
pub struct Interrupt {
    pub id: String,
    pub node: NodeId,
    pub payload: serde_json::Value,
    pub task_id: Option<TaskId>,
    pub response_schema: Option<serde_json::Value>,
}
```

Resume API:

```rust
compiled_graph
    .resume("support-123", Command::resume(json!({ "approved": true })))
    .await?;
```

Rules:

- interrupts require both a checkpointer and a `thread_id`
- if a node emits an interrupt without resumable durability, the run returns a
  resume error instead of an interrupted execution
- interrupted executions are returned only after the checkpoint needed for
  resume has been persisted
- the interrupted node restarts from the beginning (a node-emitted interrupt
  and an `interrupt_before` pause); an `interrupt_after` pause is the one
  exception — the node already ran and its stored result is replayed instead
- every branch of a step that interrupts is surfaced (`GraphExecution::interrupts`
  carries all of them, not just the lowest-index one) — a `Send` fan-out of
  one node interrupting on several concurrent activations is matched by task
  id, each stamped onto its own `Interrupt::task_id`, rather than by an
  `order` field
- resume values as a map from task id to value: `Command::resume_tasks(..)` /
  `Command::resume_by_task` deliver a distinct value per interrupted task in
  one resume call, keyed by `TaskId` — `Command::resume(value)` (one value,
  fanned to every task named by the checkpoint's stamped `interrupted_nodes`
  or, absent that, to every pending task) still works and is consulted as the
  fallback for any task the map does not name
- **Target (not implemented):** resume values as a map keyed by interrupt id
  specifically (rather than task id)
- node code before an interrupt must be deterministic or idempotent, or
  wrapped in `NodeContext::durable_task` (see
  [builder.md](builder.md#nodecontextdurable_task)) so a side effect ahead of
  the pause is memoised rather than repeated on the re-run

## `interrupt_before` / `interrupt_after` selectors

```rust
GraphBuilder::<State, Update>::new()
    // ...
    .interrupt_before(["approve"])
    .interrupt_after(["plan", "act"])   // needs Update: Serialize + DeserializeOwned
    .mark_interrupt("review")           // alias for interrupt_before(["review"])
```

Executor-level pauses at named nodes, for debugging, approvals, and human
review at arbitrary graph boundaries without editing node code. They are
distinct from a node returning `NodeResult::Interrupt` itself: the executor
injects the `Interrupt`, with payload `{"phase": "before"}` or
`{"phase": "after"}`, stamped with the activation's task id, and persists the
usual interrupt-boundary checkpoint (`has_interrupts: true` in
`get_state_history`). Both selectors are validated at `compile()`
(`TinyAgentsError::MissingNode` for an unknown node) and mark the node as an
interrupt point in the topology export.

`interrupt_before(nodes)` pauses *instead of* invoking a listed node's
handler. The checkpoint's pending set is that activation; `resume` /
`retry` re-schedules it and runs the handler normally, delivering any
`Command::resume` value on `NodeContext::resume`. The handler runs exactly
once overall.

`interrupt_after(nodes)` lets the handler run to completion, then holds its
`Update`/`Command` back from committed state: the result is serialized as a
deferred-result write (`PendingWrite::interrupt_after`, control-plane idx
`WRITES_IDX_INTERRUPT_AFTER`) in the checkpoint's `pending_writes`, the
checkpoint's `state` and the returned `GraphExecution::state` are the
*pre-update* state, and the node stays in the pending set. On resume the
executor **replays** the stored result — decoding the update through the
`Update` codec and honouring the node's `goto` — without invoking the
handler again, so it still ran exactly once and nothing is lost. This is
why `interrupt_after` requires `Update: Serialize + DeserializeOwned` (the
same bound `CompiledGraph::with_cached_node` takes); `interrupt_before` has
no such bound. A node that returns its own `NodeResult::Interrupt` is not
paused a second time, and an `Err` passes through to the failure boundary.

Acknowledgement: resuming a checkpoint acknowledges the executor-injected
pauses it recorded (per `(phase, task_id)`), so the re-run skips that phase.
The acks of a still-pending task are carried forward in checkpoint metadata
(`acknowledged_interrupts`) across any later pause of the same task — an
`interrupt_before` node that then pauses at `interrupt_after`, or emits its
own interrupt, is not paused *before* again. `update_state` carries both the
acks and the deferred-result write forward for tasks it leaves pending.

`mark_interrupt(node)` is `interrupt_before([node])`: earlier versions set
only the export marker, the runtime pause is now real.

## `response_schema`

```rust
Interrupt::new("approve", json!({ "ask": "approve?" }))
    .with_response_schema(json!({
        "type": "object",
        "required": ["approved"],
        "properties": { "approved": { "type": "boolean" } }
    }))
```

An interrupt can declare the JSON-schema subset (`type`, `required`,
`properties`, `additionalProperties`, `items`, `enum`) its resume value must
satisfy. `resume`/`resume_from` validate **fail-closed, before any
checkpoint write**: the value each schema-bearing pending interrupt would
receive — from `Command::resume_tasks` by task id, or `Command::resume`
fanned out exactly as `NodeContext::resume` would be filled — is checked
with `tinyagents_harness::tool::validate_against_schema` before the resumed
run claims the thread. A mismatch returns `TinyAgentsError::Validation`
naming the interrupt and the offending path, and the thread's latest
checkpoint is untouched (same checkpoint id, same pending interrupt, no new
history entry). A bare `retry()` delivers no value and validates nothing.
`response_schema` is `None` by default and omitted from checkpoint JSON when
unset, so legacy records decode unchanged.

## Targeted Human Steering

**Target (not implemented; see `docs/runtime-comparison/plan.md`).** Nothing
below this point — `ResumeTarget` as a targeted-steering struct,
`resume_targeted`, or per-run/per-task/per-namespace resume routing — exists
in `crates/tinyagents-graph/src` today. The `ResumeTarget` type that does
exist (`crates/tinyagents-graph/src/compiled/types.rs`) is unrelated: it is a
`Latest`/`Checkpoint(CheckpointId)` enum selecting which checkpoint a resume
replays from, not a target-selection struct with `run_id`/`task_id`/
`interrupt_id`/`namespace` fields.

Human input during an interrupt is one form of steering. A control surface
should be able to target:

- the parent orchestrator run
- a specific child sub-agent run
- a graph task id
- a node namespace inside a subgraph
- a specific interrupt id

Targeted resume shape:

```rust
pub struct ResumeTarget {
    pub run_id: RunId,
    pub task_id: Option<TaskId>,
    pub interrupt_id: Option<InterruptId>,
    pub namespace: Vec<String>,
}

compiled_graph
    .resume_targeted(
        ResumeTarget {
            run_id,
            task_id: Some(child_task),
            interrupt_id: Some(approval_interrupt),
            namespace: vec!["supervisor".into(), "research_agent".into()],
        },
        Command::resume(json!({ "approved": true })),
    )
    .await?;
```

Rules:

- resuming a child interrupt resumes that child task, not all paused siblings
- resuming the parent orchestrator may leave child interrupts pending unless
  policy cancels or resolves them
- a human can add steering instructions while resuming, but those instructions
  must be recorded separately from the interrupt answer
- stale resume targets are rejected with the latest run/checkpoint metadata
- UI clients should present pending interrupts with run tree path, node id,
  task id, sub-agent id, and checkpoint id so humans can steer the intended
  target
