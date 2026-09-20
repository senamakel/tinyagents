# Graph State, Channels, And Updates

The current scaffold returns whole state from each node. Durable graphs should
move to partial state updates applied through channels. A channel owns the
current value, the accepted update type, checkpoint representation, reducer, and
step-boundary merge behavior.

```rust
pub trait Channel: Send + Sync {
    type Value;
    type Update;
    type Checkpoint;

    fn get(&self) -> Result<Self::Value>;
    fn update(&mut self, values: Vec<Self::Update>) -> Result<ChannelChange>;
    fn checkpoint(&self) -> Result<Self::Checkpoint>;
    fn restore(checkpoint: Self::Checkpoint) -> Result<Self>
    where
        Self: Sized;
    fn consume(&mut self) -> Result<ChannelChange> {
        Ok(ChannelChange::Unchanged)
    }
    fn finish(&mut self) -> Result<ChannelChange> {
        Ok(ChannelChange::Unchanged)
    }
}
```

Required channel policies:

- `LastValue`: accepts one update per step and overwrites the value.
- `Overwrite`: explicit overwrite marker for aggregate channels.
- `BinaryAggregate`: applies a binary reducer such as append, add, min, or max.
- `Topic`: pub/sub collection, optionally accumulating across steps.
- `Ephemeral`: value exists only for one step or one trigger.
- `Barrier`: waits until named sources have all arrived.
- `NamedBarrier`: tracks named arrivals for join semantics.
- `Messages`: message merge by id for chat histories.
- `Delta`: stores compact deltas plus periodic snapshots for large append-heavy
  channels.
- `Untracked`: excluded from checkpointing when safe.

Why channel-level reducers matter:

- parallel branches can write different fields safely
- map-reduce fanout can aggregate many outputs in one step
- checkpoints can store pending writes instead of only final whole-state values
- failed parallel nodes can rerun without discarding completed writes
- tests can assert exact writes and reducer behavior
- generated/language-defined nodes can use simple partial update contracts

The graph should support both root state and object state. A single root channel
is useful for scalar workflows; multi-key state is the default for agent graphs.

## Implemented additive model (`graph::channel`)

The channel model is shipped **additively**: the monolithic `State` +
`StateReducer` path is unchanged, and channels are an opt-in alternative that
runs on the *existing* executor. The implementation lives in
`crates/tinyagents-graph/src/channel/` and is `serde_json::Value`-backed for generality.

- `Channel` (object-safe trait): per-key `merge(current, incoming) -> value`
  plus `allows_concurrent`, `is_ephemeral`, `is_tracked`, and `is_ready`
  (barrier) hooks. Concrete channels: `LastValue` (overwrite), `Topic`
  (append into an array), `Delta` (numeric accumulate), `Messages` (merge by
  `id`), `Ephemeral` (overwrite, cleared next step), `Untracked` (overwrite,
  excluded from snapshots), `Barrier`/`NamedBarrier` (count/name fan-in with
  readiness), and `BinaryAggregate` (fold via a closure or any
  `Reducer<Value>`).
- `ChannelSet`: a named map of `Box<dyn Channel>` plus their current values,
  with `add_channel`/`with_channel`, `apply_update(name, value)`, `get`,
  `is_ready`, and `snapshot()` (the tracked, durable view).
- `ChannelState`: a graph `State` wrapping a `ChannelSet`. It implements
  `StateReducer<ChannelState, ChannelUpdate>` for itself (the `&self` reducer
  receiver is unused; merge rules travel inside the running state), so a
  channel graph is built directly with
  `GraphBuilder::<ChannelState, ChannelUpdate>::new().set_reducer(ChannelState::new())`.
- `ChannelUpdate`: a batch of `(name, value)` writes a node returns
  (`ChannelUpdate::new().set(..).set(..)`). Stamp it with the producing node's
  superstep via `.at_step(ctx.step)`.

### Concurrent-write conflict detection

The executor folds a step's branch updates one at a time. Stamping each
`ChannelUpdate` with `ctx.step` lets the reducer group a step's writes: when the
step number advances it resets its per-step bookkeeping and clears `Ephemeral`
channels. A second same-step write to a non-aggregate channel (`LastValue`,
`Ephemeral`, `Untracked` — `allows_concurrent == false`) raises
`TinyAgentsError::InvalidConcurrentUpdate`; aggregate channels
(`allows_concurrent == true`) merge both writes in deterministic active-set
index order. Cross-step overwrites and repeated writes inside one update are
last-wins, not conflicts. Unstamped updates are treated as independent steps
(no conflict detection, no ephemeral clearing), preserving the simplest path.

## Serializable channels

`Channel` carries a `config(&self) -> serde_json::Value` hook (default
`Value::Null`) alongside `kind()`. A `ChannelSet` serializes as an ordered map
of `{ kind, config, value }` entries — no `dyn Channel` trait objects on the
wire — and its `Deserialize` impl reconstructs each channel from `kind` +
`config` via an internal `channel_from_config` dispatcher, with no external
context. `ChannelState` derives `Serialize`/`DeserializeOwned` on top of that,
so `Checkpoint<ChannelState>` satisfies every bundled `Checkpointer`'s durable
bounds and round-trips through `FileCheckpointer`/`SqliteCheckpointer` like any
other graph state.

Built-in channels need no config beyond `kind` (`LastValue`, `Topic`, `Delta`,
`Messages`, `Ephemeral`, `Untracked`), or a small literal payload
(`Barrier`/`NamedBarrier` persist their `expected` set). `BinaryAggregate` is
the one exception: its merge rule is a closure, which cannot serialize. Instead
of persisting the closure, a `BinaryAggregate` channel persists a **reducer
name**, resolved through a process-wide `ReducerRegistry`:

```rust
// Register once, anywhere before the checkpoint is decoded (typically at
// startup, alongside the graph builder that will use it):
let graph = GraphBuilder::<ChannelState, ChannelUpdate>::new()
    .register_reducer("double", |current: Value, incoming: Value| {
        Ok(json!(current.as_i64().unwrap() * incoming.as_i64().unwrap()))
    })
    // ...
    .set_reducer(ChannelState::new());

// Build the channel from the registered name:
let initial = ChannelState::new()
    .with_channel("product", BinaryAggregate::named("double")?);
```

`BinaryAggregate::named(name)` looks `name` up in the registry immediately
(returning `TinyAgentsError::Checkpoint` if it is not registered yet) and
records `name` in the channel's `config()`. Decoding a `binary_aggregate`
channel repeats that lookup against whatever the *decoding* process has
registered; a name nobody registered decodes to
`TinyAgentsError::Checkpoint("unknown reducer ...")` rather than silently
losing the merge rule. `GraphBuilder::register_reducer` is the intended entry
point — it just delegates to `ReducerRegistry::register`, and the registry is
global (not scoped to one builder) because a checkpoint decode has no builder
in scope at all.

The built-ins `"append"`, `"last"`, `"sum"`, `"max"`, `"min"`, and
`"set_union"` are always registered, needing no call to `register_reducer`.
`BinaryAggregate::new`/`from_reducer` (a bare closure, no name) still work for
a channel that is never checkpointed — they just cannot round-trip.

## Channel versions

Every `ChannelState` tracks a cumulative per-channel version counter,
`channel_versions: BTreeMap<String, u64>`, bumped once per distinct channel
name touched by each folded `ChannelUpdate` (a channel written twice in one
update still bumps once). It is exposed via `ChannelState::channel_versions()`
and persisted on `Checkpoint::channel_versions` at every boundary — the
executor's normal/failure/cancel boundaries and the manual `update_state`
write all go through the same `channel::channel_bookkeeping` extraction
function, so they cannot disagree about what they persist. A plain whole-state
graph (any `State` that is not `ChannelState`) reports a single `"state"`
channel, bumped once per checkpoint, so `channel_versions` is always populated
regardless of which state model a graph uses.

`NodeContext` carries two views built from this: `channel_versions` (what the
node's invocation currently observes) and `versions_seen` (this node's own
snapshot from the last time it ran — empty on its first-ever activation).
`NodeContext::changed_since_last_run(channel)` compares the two, so a node can
skip work when nothing it depends on has changed:

```rust
.add_node("summarize", |state: ChannelState, ctx: NodeContext| async move {
    if !ctx.changed_since_last_run("messages") {
        return Ok(NodeResult::Update(ChannelUpdate::new()));
    }
    // ... recompute the summary ...
})
```

The per-node snapshot is itself persisted (`Checkpoint::versions_seen`, keyed
by node id string — `NodeId` has no `Ord` impl to key a `BTreeMap` on
directly) and restored on resume, so this bookkeeping survives a restart
instead of resetting.

## Delta-channel history and `Overwrite`

`ChannelSet::with_delta(name, snapshot_every)` marks an already-registered
append-style channel (typically `Topic`, or an `"append"`/`"set_union"`
`BinaryAggregate`) for **delta tracking**: every write to `name` also records
its raw incoming value into `ChannelState::step_deltas()` for the current
step, which the checkpoint-construction call sites copy into
`Checkpoint::channel_deltas`. Unlike `channel_versions`/the channel's own
`value`, this field is **not cumulative** — each checkpoint carries only its
own step's writes to a delta-tracked channel, which is what keeps a single
checkpoint's size bounded no matter how long the channel's full value grows
(a 200-step append thread's checkpoint bytes grow roughly linearly with step
count, not quadratically). `Checkpointer::delta_history(config, channel)`
replays the full per-step write sequence for a delta-tracked channel; the
default implementation walks `state_history` oldest-first and concatenates
each checkpoint's own `channel_deltas` entry.

`ChannelUpdate::overwrite(name, value)` (backed by `ChannelWrite::Overwrite`)
bypasses the channel's merge rule entirely and replaces its value outright.
This is what lets a delta/append channel reset its baseline: the overwritten
value becomes what subsequent `.set(name, ..)` merges build on, and it also
rebases the channel's `step_deltas` entry (prior accumulated deltas for that
step are cleared before the overwrite's own value is recorded), so a consumer
walking `delta_history` sees the overwrite as a fresh starting point rather
than an append onto stale history.

`update_state` and `fork_state` never bypass this: `update_state`'s manual
write folds through the exact same `ChannelState::merge` → `ChannelSet::apply_
channel_write` dispatch the executor boundary uses (`apply_channel_write` is
the single write-path function every channel-graph write funnels through), and
`fork_state` copies a source checkpoint's `channel_versions`/`channel_deltas`/
`versions_seen` verbatim rather than re-deriving them — so replay and a manual
write can never diverge on what a checkpoint records.
