# Graph Runtime Context, Node Defaults, And Policies

This page describes the per-task runtime context and the per-node execution
policy surface as implemented in `crates/tinyagents-graph/src/builder/`
(`types.rs`, `policy.rs`) and `crates/tinyagents-graph/src/compiled/`
(`run_ctx.rs`, `step.rs`, `boundary.rs`). It replaces an older sketch of a
`GraphContext<Ctx>` / `TimeoutPolicy` / `error_handler: NodeId` design that
was never built; nothing below is a target.

## `NodeContext`

Every node handler receives `(State, NodeContext)`. The context is created
per activation by the executor (`RunCtx::node_context`), scoped to that one
task, and never a global. It exposes:

- identity: `graph_id`, `node_id`, `run_id`, `thread_id` (when
  checkpointing), `root_run_id`, `recursion_frames` (the live recursion
  stack, root-first), `task_id` (stable within the superstep; distinguishes
  `Send` fan-out activations of one node), `siblings` (how many activations
  of this node share the step), and `fork` (the branch in a parallel step)
- `step`: the 1-based superstep number
- `resume`: the value a `Command::resume`/`resume_tasks` delivered to this
  task, `None` otherwise
- `send_arg`: the per-invocation argument of a `Send` packet or `GraphInput`
- `channel_versions` / `versions_seen` and `changed_since_last_run(channel)`
  for channel-model graphs
- `child_runs`: the sink a subgraph node reports spawned child runs to
- `agent_binding`: the host-supplied recursive-agent binding for this run
- `heartbeat()`: restarts the node's idle-timeout window
- `durable_task(key, fut)`: memoises a side-effecting sub-step per
  `(task_id, key)` across re-runs of the task — see
  [builder.md](builder.md#nodecontextdurable_task)

Deadline and cancellation are *not* on the context. The whole-run deadline
(`CompiledGraph::with_run_deadline`) and the run-level stop signals
(`RunOptions::cancellation`, `RunOptions::drain`) are enforced by the
executor between and around supersteps (see
[execution.md](execution.md) and [fault-tolerance.md](fault-tolerance.md)).
There is no store registry or custom stream writer on the context; events
flow through the graph's `GraphEventSink`.

## Node policies

```rust
pub struct NodePolicy<State, Update> {
    pub retry: Option<RetryPolicy>,
    pub timeout: Option<Duration>,
    pub idle_timeout: Option<Duration>,
    pub cache: Option<NodeCachePolicy<State>>,
    pub on_error: Option<Arc<OnErrorFn<State, Update>>>,
    pub defer: bool,
}

pub struct NodeCachePolicy<State> {
    pub key: Arc<CacheKeyFn<State>>,   // Fn(&State, Option<&send_arg>) -> String
    pub ttl: Option<Duration>,
}
```

Attach a policy per node with `GraphBuilder::with_node_policy(node, policy)`
or graph-wide with `GraphBuilder::set_node_defaults(policy)`. Every field is
optional and additive; `NodePolicy::default()` changes nothing. Builder
sugar: `NodePolicy::default().with_retry(..).with_timeout(..)
.with_idle_timeout(..).with_cache(..).with_on_error(..).deferred()`.

### Resolution

The executor resolves an *effective* policy per activation
(`CompiledGraph::effective_policy` → `NodePolicy::resolve`), field by field:

1. the node's own policy, when that field is `Some`;
2. else the `set_node_defaults` policy's field;
3. else the legacy graph-wide setting — `CompiledGraph::with_node_retry`
   for `retry`, `GraphBuilder::with_node_timeout` / `GraphDefaults
   { node_timeout }` for `timeout`;
4. else nothing.

`defer` is the logical OR of the per-node and default flags. Per-node values
therefore always win; graph-level values only fill gaps.

### Retry

A handler that fails with a retryable error (`tinyagents_harness::retry::is_retryable`
— the transient model/tool class) is re-run from its start, with a fresh
handler future and a cloned context, up to the policy's attempt cap,
emitting `GraphEvent::NodeRetryScheduled` and sleeping the backoff only when
the policy opts in (`RetryPolicy::with_backoff_sleep`). A cloned context
shares the task's `durable_task` memos, so a retried attempt does not repeat
a side effect the first attempt already memoised. Non-retryable errors and
exhausted budgets escalate to the failure boundary (a resumable checkpoint
on a checkpointed thread; see fault-tolerance.md).

### Timeouts

Two independent ceilings race every attempt: `timeout` (flat wall-clock time
per attempt) and `idle_timeout` (the maximum gap between two
`NodeContext::heartbeat()` calls, or between start and the first one). Either
firing fails the attempt with `TinyAgentsError::Timeout`, which is not
retryable. Timeout cancellation is cooperative in the sense that the handler
future is dropped at the race, not pre-empted mid-instruction. There is no
"refresh on graph progress" mode; only an explicit heartbeat re-arms the
idle window.

### `on_error`

`on_error: Fn(&State, &TinyAgentsError) -> Option<Command<Update>>` is
consulted once retries are exhausted (or the error is not retryable).
Returning `Some(command)` makes the node complete with that command — its
update applied, its `goto` honoured — instead of failing the run. There is no
separate "error-handler node" concept; a policy that wants one routes to it
via the returned `Command::goto`.

### Caching

Caching is a two-part opt-in:

- `NodeCachePolicy` on the node (via `NodePolicy::cache`, or directly with
  `CompiledGraph::with_cached_node(node, policy)`, which also installs the
  `Update: Serialize + DeserializeOwned` codec) derives a key from the state
  snapshot and the activation's `send_arg`, with an optional TTL;
- a backend on the graph: `CompiledGraph::with_task_cache(Arc<dyn TaskCache>)`
  (`InMemoryTaskCache`, or `SqliteTaskCache` under the `sqlite` feature).

Without a backend every cache policy is inert. With one, a hit replays the
stored `Update` without invoking the handler and emits
`GraphEvent::TaskCompleted { cached: true }`; a miss runs the handler and,
on an `Update`/`Command` result, stores it (`TaskCompleted { cached: false }`).
Errors and interrupts are never cached. The key function is called once per
activation. The cache key is entirely the policy's to compute — include
whatever inputs, config version, or namespace make two activations
interchangeable — and keys are scoped by `(graph_id, node_id, hash)`. Cache
errors are treated as misses; caching is an optimisation, never a
correctness requirement. This is distinct from `NodeContext::durable_task`,
which is per-task memoisation inside one thread's checkpoint ledger, not a
cross-run cache.

### Defer

`defer: true` (or `GraphBuilder::mark_deferred(node)`, which also sets the
export marker) holds an activation back from the frontier while any
non-deferred activation is also ready, accumulating held activations across
supersteps and releasing them all at once the first time a boundary would
otherwise route to an empty frontier — a "run once everything else is done"
synthesis join without an explicit barrier (`boundary::apply_defer`). Held
deferred activations are not persisted: a run resumed mid-hold starts with
an empty hold.

## Graph defaults

`GraphBuilder::set_defaults(GraphDefaults { .. })` applies only the `Some`
fields of `recursion_limit`, `parallel`, `max_concurrency`, and
`node_timeout`; `with_max_concurrency(n)` bounds in-flight handlers per
parallel superstep (`0` = unbounded); `with_node_timeout(d)` is the legacy
graph-wide flat timeout that `NodePolicy::resolve` falls back to.
