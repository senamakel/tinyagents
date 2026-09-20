# Graph Node Model

Nodes are async units of work. They receive a state view plus runtime context
and return an update, command, interrupt, or no-op.

```rust
#[async_trait]
pub trait GraphNode<State, Ctx = ()>: Send + Sync {
    async fn run(
        &self,
        state: StateView<'_, State>,
        ctx: &mut GraphContext<Ctx>,
    ) -> Result<NodeResult>;
}

pub enum NodeResult {
    Update(StateUpdate),
    Command(Command),
    Interrupt(Interrupt),
    None,
}
```

Closure-backed nodes stay as an ergonomic adapter:

```rust
Node::new("agent", |state| async move {
    Ok(NodeOutput::continue_with(state))
})
```

Target node spec:

```rust
pub struct NodeSpec<State, Ctx = ()> {
    pub id: NodeId,
    pub node: Arc<dyn GraphNode<State, Ctx>>,
    pub input: ChannelSelection,
    pub destinations: Option<DestinationHints>,
    pub metadata: serde_json::Value,
    pub defer: bool,
    pub retry: Option<RetryPolicy>,
    pub cache: Option<CachePolicy>,
    pub timeout: Option<TimeoutPolicy>,
    pub error_handler: Option<NodeId>,
    pub is_error_handler: bool,
}
```

Important node kinds:

- closure node
- trait-backed node
- harness model/tool/agent-loop node
- sub-agent node
- subgraph node
- router node
- error-handler node
- test node

Deferred nodes run near graph termination, after normal active work is drained.
Use them for cleanup, final scoring, final summarization, or output shaping.

## Actual handler signature and state cloning (M2)

The shipped `NodeHandler<State, Update>` (`crates/tinyagents-graph/src/builder/types.rs`)
is `Fn(Arc<State>, NodeContext) -> NodeFuture<Update>` — every handler
receives the step's committed state as an `Arc<State>`, not an owned
`State`. This is `docs/runtime-comparison/code-review-graph.md` finding M2:
before it, the executor cloned `State` once per handler invocation (every
retry attempt, every parallel `Send`/fan-out branch), so an N-way fan-out
with retries could clone the whole state many times in a single superstep.
Now `StepRunner::run_step` (`crates/tinyagents-graph/src/compiled/step.rs`)
clones `State` **at most once per superstep**, into that `Arc`; every
branch/attempt within the step shares it via a cheap `Arc::clone`.

Two builder entry points wire a closure into a node:

- `GraphBuilder::add_node(id, |state: State, ctx| async { .. })` — the
  by-value convenience form kept for source compatibility. It is a thin
  adapter that clones out of the `Arc` once per invocation, so every
  existing handler written against the pre-M2 signature still compiles
  unchanged.
- `GraphBuilder::add_node_shared(id, |state: Arc<State>, ctx| async { .. })`
  — the zero-clone form. Prefer it for a node with a large `State` (e.g. one
  carrying message history) or one that only reads its state; it forwards
  the step's `Arc<State>` directly with no clone at all.

`NodeContext::send_arg` (the per-invocation `Send` argument) is likewise
`Option<Arc<serde_json::Value>>` rather than `Option<serde_json::Value>`, so
a repeated `Send` fan-out of the same node and every retry of one activation
share the argument's allocation too. It serializes exactly like a bare
`serde_json::Value` in checkpoint records (serde's `Arc<T>` impl is
transparent), so on-disk format is unaffected.
