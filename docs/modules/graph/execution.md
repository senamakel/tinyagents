# Graph Execution Model And Parallelization

Milestone 1 executor:

- sequential
- one active node at a time
- whole-state updates
- direct or string conditional routes
- recursion limit

Target executor:

- Pregel-like bulk synchronous parallel supersteps
- multiple active tasks per step
- immutable state snapshot during a step
- writes visible in the next step only
- deterministic write ordering before reducer application
- reducer/channel updates at step boundaries
- checkpoint after step completion
- pending writes from completed tasks
- cached writes replay without rerunning nodes
- resume from checkpoint
- recursive call tracking
- child graph and child agent run tracking

Superstep lifecycle:

1. Load checkpoint, active tasks, and pending writes. On resume, a checkpoint
   left mid-step by an interrupt/failure (see step 6 below and
   `code-review-graph.md` C1/C2) reloads its `completed_tasks` too, not just
   its pending set — those already-completed siblings are not re-run.
2. Emit step started event.
3. **Target (not implemented):** match cached task writes when cache policy
   allows it — no `cache_policy`/cached-writes-replay mechanism (an explicit,
   opt-in policy for replaying a *stored write's payload* instead of
   re-running a node) exists in `crates/tinyagents-graph/src` today; every
   *active* task still re-runs. What does exist, unconditionally, is a
   narrower but load-bearing guarantee: a task that already **completed**
   within the same still-in-progress superstep is never re-run just because
   an interrupt/failure boundary was hit — see step 6.
4. Run active tasks under concurrency, timeout, retry, and cancellation policy.
5. Collect writes, commands, sends, interrupts, and errors — every result,
   not only the ones before the first stalled (errored/interrupted) branch
   (`crates/tinyagents-graph/src/compiled/step.rs::fold_step`).
6. Persist task writes as pending writes when checkpointing supports it. At an
   interrupt/failure boundary this includes every branch that completed this
   step (`Checkpoint::completed_tasks`), regardless of its index relative to
   the stalled branch; only the stalled branch(es) become the resumed run's
   pending set. A completed branch's *routing* (step 8) is deferred rather
   than resolved here — resolving it before the stalled branches are known
   would let a downstream node observe a state missing their eventual
   updates — and is carried forward until the whole step finishes
   (`crates/tinyagents-graph/src/compiled/boundary.rs::advance`'s
   `carried_completed` handling).
7. Apply channel reducers at the step boundary. An additive channel-per-field
   state model does exist (`crates/tinyagents-graph/src/channel/`:
   `Channel`, `ChannelSet`, `ChannelState`, with `LastValue`/`Topic`/
   `Delta`/`Messages`/`Barrier`/`BinaryAggregate` merge strategies and
   concurrent-write conflict detection), but there is no per-channel
   **version** tracking (see checkpointing.md) — conflict detection is
   step-stamped, not versioned.
8. Select next active tasks from routing commands and `Send` fan-out.
   (**Target:** selection specifically by "channel version changes" does not
   apply, since channel versions are not tracked — see above.)
9. Persist the checkpoint according to durability mode.
10. Emit checkpoint, update, task, and step completion events.

Checkpointing mid-node should be avoided. Async Rust stack suspension is not a
stable persistence primitive; rerunning a node from the beginning is easier to
reason about and matches interrupt semantics. A side effect a node performs
*before* it pauses or fails can be made re-run-safe with
`NodeContext::durable_task(key, fut)`, which memoises the sub-step's output
per `(task_id, key)` in the checkpoint write ledger (see
[builder.md](builder.md#nodecontextdurable_task)).

## Stopping a run: cancellation and graceful drain

Two run-level stop signals travel in `RunOptions` (accepted by
`run_with_options`, `run_with_thread_options`, and `resume_with_options`);
they differ only in what happens to the superstep in flight.

```rust
pub struct RunOptions {
    pub cancellation: Option<tinyagents_harness::CancellationToken>,
    pub drain: Option<tinyagents_graph::DrainSignal>,
}

let (handle, signal) = DrainSignal::new();
let run = graph
    .run_with_thread_options("thread-1", state, RunOptions::with_drain(signal))
    .await?;
// elsewhere, e.g. a shutdown hook or a node handler: handle.drain();
```

| | cancellation | drain |
|---|---|---|
| checked | before every superstep **and** raced against the step's handlers | before every superstep only |
| step in flight | abandoned (nothing from it is trusted to have applied) | runs to completion; reducer and boundary checkpoint commit as normal |
| checkpoint | pending = the whole active set | pending = the *next* step's activations, metadata `"drained": true` |
| status / event | `ExecutionStatus::Cancelled`, `GraphEvent::RunCancelled` | `ExecutionStatus::Drained`, `GraphEvent::RunDrained { run_id, steps }` |
| result | `Ok(GraphExecution)`, `drained: false` | `Ok(GraphExecution)`, `drained: true` |

A drain is the graceful counterpart to cancellation: raise it from a shutdown
hook (or from inside a node via a cloned `DrainHandle`) and the run finishes
the work it has started, persists a checkpoint naming exactly what it did not
start, and returns. `GraphExecution::drained` is `true` only on that path and
`false` on every other outcome (completed, interrupted, failed, cancelled);
`GraphRunStatus::is_terminal()` treats `Drained` like `Cancelled` — the run
id will not advance on its own. `resume`/`retry` continue the thread from the
drained checkpoint like any other pending-tasks checkpoint, reaching the same
final state an undrained run would. A drain raised before the first step
persists the entry activation and runs nothing; without a thread id the run
still stops with `Drained` but has no checkpoint to persist against. The
`Drop` guard and cancellation details are in
[fault-tolerance.md](fault-tolerance.md#cooperative-cancellation).

## Parallelization

Parallel execution is represented as multiple active tasks in one superstep. A
node can route to more than one next node through conditional routing, a
command, or `Send` packets.

```rust
Command::new()
    .update(update)
    .goto([
        Send::new("retrieve_docs", json!({ "query": "billing" })),
        Send::new("retrieve_docs", json!({ "query": "refund" })),
        Send::new("score_risk", json!({ "account_id": 42 })),
    ])
```

Parallel execution rules:

- all active nodes in a superstep read the same committed state snapshot
- node-local reads may optionally include that node's own pending writes for
  branch decisions
- each node returns partial updates, commands, interrupts, sends, or errors
- channels merge successful writes at the step boundary
- conflicting writes produce reducer/channel errors, not arbitrary last-writer
  behavior
- a failed required node fails the step unless an error handler or policy routes
  it elsewhere
- completed writes can be preserved as pending writes when other nodes fail
- concurrency is bounded by graph defaults and run config

Parallelism must be visible in events. The `GraphEvent` enum actually shipped
(`crates/tinyagents-graph/src/stream/types.rs`) uses `Node*` naming, not
`Task*`, and has no cached-task variant:

- `StepStarted { active: [...] }`
- `TaskScheduled` (not `TaskStarted`)
- `NodeStarted` / `NodeCompleted` / `NodeFailed` (not `TaskCompleted`/`TaskFailed`)
- `NodeRetryScheduled`
- **Target (not implemented):** `TaskCached` — no cached-write replay exists
- `StateUpdated`
- `RouteSelected`
- `CheckpointSaved` / `CheckpointRestored`
- `StepCompleted`

For agent-specific fanout, forked runtime context, and shared-cache semantics,
see [Parallel agents and context forking](parallel-agents-forking.md).
