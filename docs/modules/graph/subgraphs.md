# Graph Subgraphs

Subgraphs are compiled graphs used as nodes.

Two state modes:

```text
shared-state subgraph
parent State == child State

adapter subgraph
parent State -> child Input -> child Output -> parent Update
```

Subgraph requirements (implemented unless marked target; verified against
`crates/tinyagents-graph/src/subgraph/`):

- namespace checkpoint ids
- preserve `root_run_id`
- set child `parent_run_id`
- propagate thread id by default
- **Target (not implemented):** allow isolated child thread ids by explicit
  configuration — today an embedded child always runs on the parent's thread
  when one is set, or unthreaded (no checkpoints) when it is not; there is no
  opt-in for a child to have its own independent thread id
- **Target (not implemented):** inherit, override, or disable the parent
  checkpointer per subgraph — today the child always inherits the parent's
  checkpointer
- emit nested events with parent node id and namespace
- stream child values, updates, messages, tasks, and checkpoints when requested
- **Target (not implemented):** allow `Command::Parent` handoff from child
  graph to parent graph — no `Parent` variant exists on `Command`
- **Target (not implemented):** expose child state in parent checkpoint task
  metadata — today the parent tracks only lineage (`ChildRun` entries: child
  run id, node, and a `child_runs` array in boundary-checkpoint metadata),
  not the child's state

Subgraph persistence must be explicit. Inherited checkpointing is convenient for
shared-state subgraphs; isolated checkpointing is safer for reusable child
graphs that may also run independently.

## Run hierarchy

When a subgraph node runs its embedded `CompiledGraph`, the child run is wired
into the parent's recursion tree:

- the child gets its own `run_id`, preserves the enclosing run's `root_run_id`,
  and sets `parent_run_id` to the parent run;
- the child run extends the parent's recursion frames (seeded from
  `NodeContext::recursion_frames`) and its root frame names the embedding node,
  so depth tracking is correct without mutating the parent's live stack;
- the spawned child is reported back through `NodeContext::child_runs` and
  surfaces on `GraphExecution::child_runs` (a `ChildRun` list keyed by node) and
  in every boundary checkpoint's metadata under a `child_runs` array;
- callers read the parent/child lineage after a run via
  `GraphExecution::run_tree()` (a `RunTree`: this run's id, the shared root, the
  parent run, and the spawned children).

When the parent `NodeContext` carries a thread id, embedded child graphs run on
that same thread by default. The child checkpoint namespace is still extended
with the embedding node id, so parent and child checkpoints share a thread but
remain separately inspectable:

```text
thread: t
parent checkpoint namespace: []
child checkpoint namespace: ["child_node"]
```

If the parent run has no thread id, the child keeps the unthreaded execution
behavior and does not persist checkpoints.
