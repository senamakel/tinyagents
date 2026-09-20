# Graph Builder And Compile Contract

The builder supports:

- `add_node`
- `add_node_shared`
- `add_sequence`
- `add_edge`
- `add_waiting_edge`
- `add_conditional_edges`
- `set_entry_point`
- `set_conditional_entry_point`
- `set_finish_point`
- `set_node_defaults`
- `set_defaults`
- `with_max_concurrency`
- `with_node_timeout`
- `interrupt_before` / `interrupt_after` / `mark_interrupt`
- `compile`

`add_conditional_edges` accepts route labels and a router return value that are
any `impl ToString`, so a user-defined route enum that implements `Display` (or
the `Route` newtype) can label edges directly; plain `&str`/`String` labels keep
working unchanged.

`add_sequence([a, b, c])` is convenience sugar for a chain of direct edges
(`add_edge(a, b).add_edge(b, c)`). `add_waiting_edge(from, to)` is a barrier edge:
`to` activates only once *all* of its registered predecessors have completed,
even across supersteps.

`add_edge`/`add_waiting_edge` accumulate: calling `add_edge("a", "b")` then
`add_edge("a", "c")` schedules **both** `b` and `c` as static fan-out targets
of `a` (deduplicated — adding the same edge twice does not schedule the
target twice), matching the "one or more node names" routing contract. This
is a change from earlier versions, where a second `add_edge` call from the
same source silently overwrote the first.

`add_conditional_edges_checked(from, router, all_labels)` is
`add_conditional_edges` plus an exhaustive `all_labels` list tied to the
router's own return type. `compile()` cross-checks every declared label
against the node's route table and rejects a mismatch with
`TinyAgentsError::MissingRoute` **at build time**, instead of only at run
time when the router happens to return the mistyped label. Plain
`add_conditional_edges` (no exhaustive label list) still only fails at run
time, since an opaque closure's possible outputs can't be enumerated ahead of
time.

Graph defaults are settable in one call:

```rust
pub struct GraphDefaults {
    pub recursion_limit: Option<usize>,
    pub parallel: Option<bool>,
    pub max_concurrency: Option<usize>,
    pub node_timeout: Option<Duration>,
}
```

`set_defaults(GraphDefaults { .. })` applies only the `Some` fields.
`with_max_concurrency(n)` bounds the number of node handlers in flight per
parallel superstep (the active set runs in chunks of at most `n`).
`with_node_timeout(d)` fails the run with `TinyAgentsError::Timeout` if any node
handler does not resolve within `d`.

`CompiledGraph::with_run_deadline(d)` bounds the *whole run* by a wall-clock
`d`, checked at every super-step boundary: when the elapsed run time first
reaches `d` the run stops *between* super-steps with `TinyAgentsError::Timeout`,
leaving the last committed boundary checkpoint intact and resumable. Prefer this
over wrapping `run` in an external `tokio::time::timeout`, which aborts
mid-super-step and cannot leave a clean checkpoint. It bounds scheduling, not a
single in-flight node — pair it with `with_node_timeout` to also bound
individual handlers.

## Interrupt selectors

```rust
GraphBuilder::<State, Update>::new()
    .interrupt_before(["approve"])
    .interrupt_after(["plan"])      // requires Update: Serialize + DeserializeOwned
    .mark_interrupt("review")       // == interrupt_before(["review"])
```

`interrupt_before(nodes)` pauses the run instead of invoking a listed node;
`interrupt_after(nodes)` lets the node run, then pauses *before* its result
is applied, persisting the result as a deferred write that resume replays
without re-running the handler. Both inject an `Interrupt` with payload
`{"phase": "before" | "after"}`, need a checkpointer and thread like any
interrupt, are validated at `compile()` (`MissingNode`), and set the
export's `NodeInfo::interrupt` marker. `mark_interrupt` used to set only that
marker; it is now an alias for `interrupt_before` — a node that already
pauses itself should not be listed (it would pause twice; annotate it with
`with_node_metadata` for the export instead). Full semantics, including how
acknowledgements survive repeated pauses, are in
[interrupts.md](interrupts.md#interrupt_before--interrupt_after-selectors).

## `NodeContext::durable_task`

```rust
.add_node("charge", |state, ctx: NodeContext| async move {
    let receipt: Receipt = ctx
        .durable_task("charge-card", async { payments.charge(&state).await })
        .await?;
    if ctx.resume.is_none() {
        return Ok(NodeResult::Interrupt(Interrupt::new("charge", json!(receipt))));
    }
    Ok(NodeResult::Update(state.with_receipt(receipt)))
})
```

A node is re-run from its start after an interrupt/resume, a failure/retry,
or an in-process node retry. `durable_task(key, fut)` runs `fut` at most
once per `(task_id, key)`: the first execution awaits it and records its
`Ok` output (`T: Serialize + DeserializeOwned`) as a
`PendingWrite::durable_task` memo in the task's checkpoint write ledger; a
re-run of the same task returns the stored value without polling `fut`. An
`Err` is returned unmemoised. Keys are independent — a handler that memoised
`"a"` then failed before `"b"` replays `"a"` and runs `"b"` fresh on retry.
Memos are scoped to one task in one thread (not a cross-run cache; see
`CompiledGraph::with_cached_node` for that) and are durable across process
restarts on a checkpointed thread; a graph without a checkpointer still
dedupes within one run. The executor pre-seeds each re-run's context from
the checkpoint and persists new memos at whichever boundary the task
stalls at (interrupt or failure); a task that completes drops its memos.

## Not implemented: compile-time options

An earlier draft of this page described a `CompileOptions` struct
(`checkpointer: CheckpointerChoice`, `cache`, `store`, `debug`,
`stream_transformers`) mirroring LangGraph's compile-time bag. It does not
exist: checkpointers, caches, sinks, and policies are attached to the
`CompiledGraph` with `with_*` methods after `compile()`, and the interrupt
selectors live on the builder as above.

Validation rules:

- graph must have at least one `START` path
- `START` cannot be an edge target
- `END` cannot be an edge source
- every edge source exists, except `START`
- every edge target exists, except `END`
- duplicate node ids are rejected
- duplicate branch names from a source are rejected
- conditional route targets are validated at compile time when known
- interrupt targets must exist
- waiting-edge sources and targets must exist
- command destinations used only for rendering are marked as such
- node additions after compile do not mutate an existing compiled graph
