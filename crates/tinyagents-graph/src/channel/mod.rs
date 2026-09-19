//! Channel-per-field state model (additive).
//!
//! See `types` for the type definitions and the high-level model. This file
//! supplies the concrete [`Channel`] merge rules, the [`ChannelSet`] map
//! operations, and the [`ChannelState`] ⇒ [`StateReducer`] bridge that lets a
//! channel graph run on the existing executor.
//!
//! ## How a channel graph runs on the unchanged executor
//!
//! The executor folds a superstep's branch results one at a time:
//! `state = reducer.apply(state, update)` for each branch's
//! [`ChannelUpdate`]. [`ChannelState`] is its own reducer, so each `apply`
//! dispatches every write in the update to the owning channel's
//! [`Channel::merge`].
//!
//! ## Concurrent-write conflict detection
//!
//! When two fan-out branches write the *same* channel in *one* superstep, the
//! merge must decide whether that is legal:
//!
//! - **Aggregate channels** ([`Topic`], [`BinaryAggregate`], [`Delta`],
//!   [`Messages`], [`Barrier`], [`NamedBarrier`]) set
//!   [`Channel::allows_concurrent`] to `true`; both writes fold in
//!   deterministic active-set index order.
//! - **Overwrite channels** ([`LastValue`], [`Ephemeral`], [`Untracked`])
//!   return `false`; a second same-step write to such a channel raises
//!   [`TinyAgentsError::InvalidConcurrentUpdate`] because there is no
//!   deterministic winner.
//!
//! Because the executor applies a step's updates as a contiguous batch, "same
//! step" is tracked by stamping each [`ChannelUpdate`] with the node's
//! `ctx.step` via [`ChannelUpdate::at_step`]. When updates are stamped, the
//! reducer resets its per-step bookkeeping (and clears [`Ephemeral`] channels)
//! whenever the step number advances. Unstamped updates are each treated as
//! their own step (last-value writes always win, no conflict detection and no
//! ephemeral clearing) — so existing whole-state habits keep working and
//! conflict detection is strictly opt-in.

mod registry;
mod types;

pub use registry::ReducerRegistry;
pub use types::{
    Barrier, BinaryAggregate, Channel, ChannelSet, ChannelState, ChannelUpdate, ChannelWrite,
    Delta, Ephemeral, LastValue, Messages, NamedBarrier, Topic, Untracked,
};

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::reducer::StateReducer;
use crate::{Result, TinyAgentsError};

// --- Channel merge rules ---

impl Channel for LastValue {
    fn kind(&self) -> &'static str {
        "last_value"
    }

    fn merge(&self, _current: Option<Value>, incoming: Value) -> Result<Value> {
        Ok(incoming)
    }

    fn clone_box(&self) -> Box<dyn Channel> {
        Box::new(*self)
    }
}

impl Channel for Topic {
    fn kind(&self) -> &'static str {
        "topic"
    }

    fn merge(&self, current: Option<Value>, incoming: Value) -> Result<Value> {
        // Reuse the existing array in place instead of cloning it per merge.
        let mut list = match current {
            Some(Value::Array(items)) => items,
            Some(other) => vec![other],
            None => Vec::new(),
        };
        match incoming {
            Value::Array(items) => list.extend(items),
            other => list.push(other),
        }
        Ok(Value::Array(list))
    }

    fn allows_concurrent(&self) -> bool {
        true
    }

    fn clone_box(&self) -> Box<dyn Channel> {
        Box::new(*self)
    }
}

impl Channel for Delta {
    fn kind(&self) -> &'static str {
        "delta"
    }

    fn merge(&self, current: Option<Value>, incoming: Value) -> Result<Value> {
        let add_err =
            || TinyAgentsError::Graph("Delta channel only accepts numeric writes".to_string());
        let incoming_num = incoming.as_f64().ok_or_else(add_err)?;
        let Some(current) = current else {
            return Ok(incoming);
        };
        let current_num = current.as_f64().ok_or_else(add_err)?;

        // Stay in integer space when both operands are integers.
        if current.is_i64() && incoming.is_i64() {
            let sum = current.as_i64().unwrap() + incoming.as_i64().unwrap();
            return Ok(Value::from(sum));
        }
        Ok(Value::from(current_num + incoming_num))
    }

    fn allows_concurrent(&self) -> bool {
        true
    }

    fn clone_box(&self) -> Box<dyn Channel> {
        Box::new(*self)
    }
}

impl Channel for Messages {
    fn kind(&self) -> &'static str {
        "messages"
    }

    fn merge(&self, current: Option<Value>, incoming: Value) -> Result<Value> {
        // Reuse the existing array in place instead of cloning it per merge.
        let mut list = match current {
            Some(Value::Array(items)) => items,
            Some(_) => {
                return Err(TinyAgentsError::Graph(
                    "Messages channel value must be a JSON array".to_string(),
                ));
            }
            None => Vec::new(),
        };
        let incoming = match incoming {
            Value::Array(items) => items,
            other => vec![other],
        };
        // Build an id -> index map over the existing list once (O(existing)) so
        // each incoming message is an O(1) lookup instead of a linear scan.
        // Previously this dedup was O(existing x incoming), which bit at a few
        // thousand messages.
        let mut index: HashMap<String, usize> = list
            .iter()
            .enumerate()
            .filter_map(|(i, existing)| {
                existing
                    .get("id")
                    .and_then(Value::as_str)
                    .map(|id| (id.to_string(), i))
            })
            .collect();
        for msg in incoming {
            match msg.get("id").and_then(Value::as_str).map(str::to_string) {
                // Keyed message: replace the same id in place, or append and
                // remember its position for later incoming writes.
                Some(id) => match index.get(&id) {
                    Some(&i) => list[i] = msg,
                    None => {
                        index.insert(id, list.len());
                        list.push(msg);
                    }
                },
                // Unkeyed message: always appended (unchanged behavior).
                None => list.push(msg),
            }
        }
        Ok(Value::Array(list))
    }

    fn allows_concurrent(&self) -> bool {
        true
    }

    fn clone_box(&self) -> Box<dyn Channel> {
        Box::new(*self)
    }
}

impl Channel for Ephemeral {
    fn kind(&self) -> &'static str {
        "ephemeral"
    }

    fn merge(&self, _current: Option<Value>, incoming: Value) -> Result<Value> {
        Ok(incoming)
    }

    fn is_ephemeral(&self) -> bool {
        true
    }

    fn clone_box(&self) -> Box<dyn Channel> {
        Box::new(*self)
    }
}

impl Channel for Untracked {
    fn kind(&self) -> &'static str {
        "untracked"
    }

    fn merge(&self, _current: Option<Value>, incoming: Value) -> Result<Value> {
        Ok(incoming)
    }

    fn is_tracked(&self) -> bool {
        false
    }

    fn clone_box(&self) -> Box<dyn Channel> {
        Box::new(*self)
    }
}

impl Barrier {
    /// Creates a count-based barrier that is ready after `expected` arrivals.
    pub fn new(expected: usize) -> Self {
        Self { expected }
    }
}

impl Channel for Barrier {
    fn kind(&self) -> &'static str {
        "barrier"
    }

    fn merge(&self, current: Option<Value>, incoming: Value) -> Result<Value> {
        // Reuse the accumulated array in place instead of cloning it per merge.
        let mut list = match current {
            Some(Value::Array(items)) => items,
            Some(other) => vec![other],
            None => Vec::new(),
        };
        match incoming {
            Value::Array(items) => list.extend(items),
            other => list.push(other),
        }
        Ok(Value::Array(list))
    }

    fn config(&self) -> Value {
        serde_json::json!({ "expected": self.expected })
    }

    fn allows_concurrent(&self) -> bool {
        true
    }

    fn is_ready(&self, current: Option<&Value>) -> bool {
        current
            .and_then(Value::as_array)
            .map(|items| items.len() >= self.expected)
            .unwrap_or(self.expected == 0)
    }

    fn clone_box(&self) -> Box<dyn Channel> {
        Box::new(*self)
    }
}

impl NamedBarrier {
    /// Creates a name-based barrier that is ready once every name has arrived.
    pub fn new(expected: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            expected: expected.into_iter().map(Into::into).collect(),
        }
    }
}

impl Channel for NamedBarrier {
    fn kind(&self) -> &'static str {
        "named_barrier"
    }

    fn merge(&self, current: Option<Value>, incoming: Value) -> Result<Value> {
        // Reuse the accumulated object in place instead of cloning it per merge.
        let mut map = match current {
            Some(Value::Object(map)) => map,
            Some(_) => {
                return Err(TinyAgentsError::Graph(
                    "NamedBarrier channel value must be a JSON object".to_string(),
                ));
            }
            None => serde_json::Map::new(),
        };
        let Value::Object(incoming) = incoming else {
            return Err(TinyAgentsError::Graph(
                "NamedBarrier writes must be JSON objects of named arrivals".to_string(),
            ));
        };
        for (key, value) in incoming {
            map.insert(key, value);
        }
        Ok(Value::Object(map))
    }

    fn config(&self) -> Value {
        serde_json::json!({ "expected": self.expected })
    }

    fn allows_concurrent(&self) -> bool {
        true
    }

    fn is_ready(&self, current: Option<&Value>) -> bool {
        let Some(Value::Object(map)) = current else {
            return self.expected.is_empty();
        };
        self.expected.iter().all(|name| map.contains_key(name))
    }

    fn clone_box(&self) -> Box<dyn Channel> {
        Box::new(self.clone())
    }
}

impl BinaryAggregate {
    /// Creates an aggregate channel from a binary fold closure. The first write
    /// becomes the value directly; later writes are `fold(current, incoming)`.
    ///
    /// Unnamed: [`Channel::config`] carries no reducer name, so a channel
    /// built this way merges correctly at runtime but cannot round-trip
    /// through a durable checkpointer. Use [`BinaryAggregate::named`] (backed
    /// by [`ReducerRegistry`]) for a channel that must survive a checkpoint
    /// decode.
    pub fn new<F>(fold: F) -> Self
    where
        F: Fn(Value, Value) -> Result<Value> + Send + Sync + 'static,
    {
        Self {
            fold: Arc::new(fold),
            reducer_name: None,
        }
    }

    /// Builds an aggregate channel from a [`crate::Reducer<Value>`]. Also
    /// unnamed — see [`BinaryAggregate::new`].
    pub fn from_reducer<R>(reducer: R) -> Self
    where
        R: crate::Reducer<Value> + 'static,
    {
        Self::new(move |current, incoming| reducer.reduce(current, incoming))
    }

    /// Builds an aggregate channel from the reducer registered under `name`
    /// in the process-wide [`ReducerRegistry`] (register it first with
    /// [`crate::GraphBuilder::register_reducer`], or use one of the built-ins
    /// — `"append"`, `"last"`, `"sum"`, `"max"`, `"min"`, `"set_union"`).
    ///
    /// Unlike [`BinaryAggregate::new`], this channel's [`Channel::config`]
    /// persists `name`, so it round-trips through a durable checkpointer:
    /// decoding looks `name` back up in the registry (present in the
    /// resuming process — the same call site that ran this graph before must
    /// have registered it) and fails with
    /// `TinyAgentsError::Checkpoint("unknown reducer ...")` if it is not
    /// there.
    pub fn named(name: impl Into<String>) -> Result<Self> {
        let name = name.into();
        let fold = ReducerRegistry::require(&name)?;
        Ok(Self {
            fold,
            reducer_name: Some(name),
        })
    }
}

impl Channel for BinaryAggregate {
    fn kind(&self) -> &'static str {
        "binary_aggregate"
    }

    fn merge(&self, current: Option<Value>, incoming: Value) -> Result<Value> {
        match current {
            Some(current) => (self.fold)(current, incoming),
            None => Ok(incoming),
        }
    }

    fn config(&self) -> Value {
        ReducerRegistry::config_for(self.reducer_name.as_deref())
    }

    fn allows_concurrent(&self) -> bool {
        true
    }

    fn clone_box(&self) -> Box<dyn Channel> {
        Box::new(self.clone())
    }
}

/// Reconstructs a boxed [`Channel`] from its persisted `{kind, config}` pair
/// (the counterpart of [`Channel::config`]), used by [`ChannelSet`]'s
/// [`serde::Deserialize`] impl to hydrate a checkpoint's channel schema with
/// no external context — see `channel/registry.rs`'s module docs for why
/// `binary_aggregate` alone needs the process-wide [`ReducerRegistry`] to do
/// this.
fn channel_from_config(kind: &str, config: &Value) -> Result<Box<dyn Channel>> {
    match kind {
        "last_value" => Ok(Box::new(LastValue)),
        "topic" => Ok(Box::new(Topic)),
        "delta" => Ok(Box::new(Delta)),
        "messages" => Ok(Box::new(Messages)),
        "ephemeral" => Ok(Box::new(Ephemeral)),
        "untracked" => Ok(Box::new(Untracked)),
        "barrier" => {
            let expected = config
                .get("expected")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            Ok(Box::new(Barrier::new(expected)))
        }
        "named_barrier" => {
            let expected: Vec<String> = config
                .get("expected")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            Ok(Box::new(NamedBarrier::new(expected)))
        }
        "binary_aggregate" => {
            let name = config.get("reducer").and_then(Value::as_str).ok_or_else(|| {
                TinyAgentsError::Checkpoint(
                    "binary_aggregate channel requires a named reducer to decode; build it \
                     with `BinaryAggregate::named` so its config persists a reducer name"
                        .to_string(),
                )
            })?;
            Ok(Box::new(BinaryAggregate::named(name)?))
        }
        other => Err(TinyAgentsError::Checkpoint(format!(
            "unknown channel kind `{other}`"
        ))),
    }
}

// --- ChannelSet ---

impl ChannelSet {
    /// Creates an empty channel set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `channel` under `name`, returning the set for chaining.
    pub fn with_channel(
        mut self,
        name: impl Into<String>,
        channel: impl Channel + 'static,
    ) -> Self {
        self.add_channel(name, channel);
        self
    }

    /// Registers `channel` under `name`.
    pub fn add_channel(&mut self, name: impl Into<String>, channel: impl Channel + 'static) {
        self.channels.insert(name.into(), Box::new(channel));
    }

    /// Marks an already-registered append-style channel (typically [`Topic`]
    /// or a `"append"`/`"set_union"` [`BinaryAggregate`]) for delta-history
    /// tracking: every write to `name` also records the raw incoming value
    /// into [`ChannelState::step_deltas`] for that step, which the
    /// checkpoint-construction call sites persist into
    /// [`crate::checkpoint::Checkpoint::channel_deltas`]. Every
    /// `snapshot_every` writes (minimum `1`) an additional full-value
    /// snapshot marker (`{"$snapshot": <value>}`) is recorded alongside the
    /// delta, so a consumer walking the history can fast-forward without
    /// replaying every write from genesis.
    ///
    /// Returns the set for chaining. A no-op marker on a channel name that
    /// is never registered with [`ChannelSet::with_channel`]/
    /// [`ChannelSet::add_channel`] has no effect (there is nothing to track
    /// writes for).
    pub fn with_delta(mut self, name: impl Into<String>, snapshot_every: u32) -> Self {
        self.delta_channels
            .insert(name.into(), snapshot_every.max(1));
        self
    }

    /// Returns the current value of `name`, if any has been written.
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.values.get(name)
    }

    /// Whether `name` is a registered channel.
    pub fn contains(&self, name: &str) -> bool {
        self.channels.contains_key(name)
    }

    /// Whether the channel `name` permits concurrent same-step writes. Errors
    /// if `name` is not a registered channel.
    pub fn allows_concurrent(&self, name: &str) -> Result<bool> {
        self.channel(name).map(|c| c.allows_concurrent())
    }

    /// Whether the barrier (or other) channel `name` has received everything it
    /// is waiting for. Non-barrier channels are always ready. Errors if `name`
    /// is not registered.
    pub fn is_ready(&self, name: &str) -> Result<bool> {
        let channel = self.channel(name)?;
        Ok(channel.is_ready(self.values.get(name)))
    }

    /// Folds `value` into the channel `name` via its merge rule. Errors with
    /// [`TinyAgentsError::Graph`] if `name` is not a registered channel.
    ///
    /// The unknown-channel check runs *before* any state is touched. If a
    /// registered channel's [`Channel::merge`] rejects the write (e.g. a
    /// [`Delta`] receiving a non-numeric value), the channel's prior value is
    /// dropped — a rejected write leaves the channel unset. This matches the
    /// executor's reducer contract, where a merge error discards the whole
    /// [`ChannelState`] for that step regardless.
    pub fn apply_update(&mut self, name: &str, value: Value) -> Result<()> {
        // Field-level borrows (channels immutable, values mutable) so the
        // current value can be *moved* into `merge` — accumulating channels then
        // fold in place rather than cloning the whole accumulated value.
        let channel = self
            .channels
            .get(name)
            .map(AsRef::as_ref)
            .ok_or_else(|| TinyAgentsError::Graph(format!("unknown channel `{name}`")))?;
        let current = self.values.remove(name);
        let merged = channel.merge(current, value)?;
        self.values.insert(name.to_string(), merged);
        Ok(())
    }

    /// The single dispatch point for one channel write, folding an ordinary
    /// [`ChannelWrite::Merge`] through [`ChannelSet::apply_update`] or
    /// replacing the value outright for a [`ChannelWrite::Overwrite`] (which
    /// bypasses the channel's merge rule and becomes the new baseline for
    /// any merge/delta tracking that follows). Returns the channel's value
    /// after the write.
    ///
    /// This is the one write path every channel-graph write funnels through
    /// — a normal executor superstep boundary
    /// ([`ChannelState::merge`]/[`crate::channel::ChannelUpdate`]),
    /// `CompiledGraph::update_state`, and `CompiledGraph::fork_state`'s copy
    /// — so replay and a manual update can never disagree about what a
    /// write means (I5/R3; see `docs/modules/graph/state-channels.md`).
    pub fn apply_channel_write(&mut self, name: &str, write: &ChannelWrite) -> Result<Value> {
        match write {
            ChannelWrite::Merge(value) => {
                self.apply_update(name, value.clone())?;
                Ok(self.values.get(name).cloned().unwrap_or(Value::Null))
            }
            ChannelWrite::Overwrite(value) => {
                // Validate the channel exists (same contract as `apply_update`)
                // before mutating.
                self.channel(name)?;
                self.values.insert(name.to_string(), value.clone());
                Ok(value.clone())
            }
        }
    }

    /// Returns the tracked channel values as an ordered map, excluding
    /// [`Untracked`] channels. This is the durable/inspectable state view.
    pub fn snapshot(&self) -> BTreeMap<String, Value> {
        self.values
            .iter()
            .filter(|(name, _)| {
                self.channels
                    .get(*name)
                    .map(|c| c.is_tracked())
                    .unwrap_or(true)
            })
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect()
    }

    /// Clears the value of every [`Ephemeral`] channel. Called at the start of a
    /// new step by [`ChannelState`].
    pub(crate) fn clear_ephemeral(&mut self) {
        let ephemeral: Vec<String> = self
            .channels
            .iter()
            .filter(|(_, c)| c.is_ephemeral())
            .map(|(name, _)| name.clone())
            .collect();
        for name in ephemeral {
            self.values.remove(&name);
        }
    }

    fn channel(&self, name: &str) -> Result<&dyn Channel> {
        self.channels
            .get(name)
            .map(AsRef::as_ref)
            .ok_or_else(|| TinyAgentsError::Graph(format!("unknown channel `{name}`")))
    }
}

/// One channel's wire representation: `{ kind, config, value }` (the
/// counterpart of [`Channel::config`]/[`channel_from_config`]).
#[derive(Serialize, Deserialize)]
struct ChannelEntry {
    kind: String,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    config: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    value: Option<Value>,
}

/// [`ChannelSet`]'s full wire representation: its channel schema/values plus
/// the [`ChannelSet::with_delta`] registrations, so a decoded set round-trips
/// which channels are delta-tracked (not just their current values).
#[derive(Serialize, Deserialize)]
struct ChannelSetWire {
    channels: BTreeMap<String, ChannelEntry>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    delta: HashMap<String, u32>,
}

impl serde::Serialize for ChannelSet {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let channels: BTreeMap<String, ChannelEntry> = self
            .channels
            .iter()
            .map(|(name, channel)| {
                (
                    name.clone(),
                    ChannelEntry {
                        kind: channel.kind().to_string(),
                        config: channel.config(),
                        value: self.values.get(name).cloned(),
                    },
                )
            })
            .collect();
        ChannelSetWire {
            channels,
            delta: self.delta_channels.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for ChannelSet {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let wire = ChannelSetWire::deserialize(deserializer)?;
        let mut channels: HashMap<String, Box<dyn Channel>> = HashMap::new();
        let mut values: HashMap<String, Value> = HashMap::new();
        for (name, entry) in wire.channels {
            let channel =
                channel_from_config(&entry.kind, &entry.config).map_err(serde::de::Error::custom)?;
            channels.insert(name.clone(), channel);
            if let Some(value) = entry.value {
                values.insert(name, value);
            }
        }
        Ok(ChannelSet {
            channels,
            values,
            delta_channels: wire.delta,
        })
    }
}

// --- ChannelUpdate ---

impl ChannelUpdate {
    /// Creates an empty update.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a `(name, value)` merged write, returning the update for
    /// chaining.
    pub fn set(mut self, name: impl Into<String>, value: impl Into<Value>) -> Self {
        self.writes
            .push((name.into(), ChannelWrite::Merge(value.into())));
        self
    }

    /// Adds a `(name, value)` write that bypasses the channel's merge rule
    /// and replaces its value outright (see [`ChannelWrite::Overwrite`]).
    pub fn overwrite(mut self, name: impl Into<String>, value: impl Into<Value>) -> Self {
        self.writes
            .push((name.into(), ChannelWrite::Overwrite(value.into())));
        self
    }

    /// Stamps the update with the producing node's superstep (`ctx.step`),
    /// enabling same-step concurrent-write conflict detection and ephemeral
    /// clearing. Without a stamp each update is treated as its own step.
    pub fn at_step(mut self, step: usize) -> Self {
        self.step = Some(step);
        self
    }

    /// Whether the update carries no writes.
    pub fn is_empty(&self) -> bool {
        self.writes.is_empty()
    }
}

// --- ChannelState ---

impl ChannelState {
    /// Creates a state with no channels.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `channel` under `name`, returning the state for chaining. Use
    /// this to declare a graph's channel schema before running.
    pub fn with_channel(
        mut self,
        name: impl Into<String>,
        channel: impl Channel + 'static,
    ) -> Self {
        self.set.add_channel(name, channel);
        self
    }

    /// Marks an already-registered channel for delta-history tracking; see
    /// [`ChannelSet::with_delta`].
    pub fn with_delta(mut self, name: impl Into<String>, snapshot_every: u32) -> Self {
        self.set = self.set.with_delta(name, snapshot_every);
        self
    }

    /// Borrows the underlying [`ChannelSet`].
    pub fn channels(&self) -> &ChannelSet {
        &self.set
    }

    /// Returns the current value of channel `name`, if written.
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.set.get(name)
    }

    /// Returns the tracked channel values (see [`ChannelSet::snapshot`]).
    pub fn snapshot(&self) -> BTreeMap<String, Value> {
        self.set.snapshot()
    }

    /// Whether channel `name` is a satisfied barrier (see
    /// [`ChannelSet::is_ready`]).
    pub fn is_ready(&self, name: &str) -> Result<bool> {
        self.set.is_ready(name)
    }

    /// Folds a [`ChannelUpdate`] into this state, dispatching each write to its
    /// channel's merge rule. This is the core reducer step.
    ///
    /// When the update is stamped (via [`ChannelUpdate::at_step`]) with a step
    /// number that differs from the last one seen, the per-step write tracking
    /// is reset and [`Ephemeral`] channels are cleared before the writes apply.
    /// A second write to a non-aggregate channel within the same stamped step
    /// raises [`TinyAgentsError::InvalidConcurrentUpdate`].
    pub fn merge(mut self, update: ChannelUpdate) -> Result<Self> {
        match update.step {
            Some(step) if step != self.current_step => {
                self.current_step = step;
                self.step_writes.clear();
                self.step_deltas.clear();
                self.set.clear_ephemeral();
            }
            Some(_) => {}
            None => {
                // Unstamped updates are independent: no cross-update detection.
                self.step_writes.clear();
                self.step_deltas.clear();
            }
        }

        // Distinct channels touched by this single update (a node writing the
        // same channel twice in one update is last-wins, not a conflict).
        let mut distinct: Vec<&str> = Vec::new();
        for (name, _) in &update.writes {
            if !distinct.contains(&name.as_str()) {
                distinct.push(name.as_str());
            }
        }

        // Validate before mutating so a conflicting step never commits partial
        // writes.
        for name in &distinct {
            let allows = self.set.allows_concurrent(name)?;
            let count = self.step_writes.get(*name).copied().unwrap_or(0) + 1;
            if count > 1 && !allows {
                return Err(TinyAgentsError::InvalidConcurrentUpdate(format!(
                    "channel `{name}` received {count} concurrent writes in one step but is not an aggregate channel"
                )));
            }
        }

        let touched: HashSet<String> = distinct.iter().map(|n| n.to_string()).collect();
        for name in &touched {
            *self.step_writes.entry(name.clone()).or_insert(0) += 1;
            // Channel versions (I5/R3): bumped once per distinct channel
            // name touched by this update, regardless of write kind
            // (Merge/Overwrite) — see the module docs on
            // `Checkpoint::channel_versions`.
            *self.channel_versions.entry(name.clone()).or_insert(0) += 1;
        }
        // `apply_channel_write` (on `ChannelSet`) is the single write-path
        // dispatch point every channel-graph write funnels through — see its
        // docs. This loop is that path's boundary-fold caller; `update_state`
        // and `fork_state` reach the same dispatch point through
        // `compiled::channel_bookkeeping`/direct `ChannelSet` access so
        // replay and a manual write cannot diverge.
        for (name, write) in update.writes {
            let is_overwrite = write.is_overwrite();
            let value = write.value().clone();
            self.set.apply_channel_write(&name, &write)?;
            if let Some(&snapshot_every) = self.set.delta_channels.get(&name) {
                let entry = self.step_deltas.entry(name.clone()).or_default();
                if is_overwrite {
                    // Overwrite rebases the delta/append history: prior
                    // accumulated deltas for this channel no longer describe
                    // the current baseline.
                    entry.clear();
                }
                entry.push(value);
                let version = self.channel_versions.get(&name).copied().unwrap_or(0);
                if snapshot_every > 0 && version % u64::from(snapshot_every) == 0 {
                    let full = self.set.get(&name).cloned().unwrap_or(Value::Null);
                    entry.push(serde_json::json!({ "$snapshot": full }));
                }
            }
        }
        Ok(self)
    }

    /// Cumulative per-channel version counters (I5/R3). See
    /// [`crate::checkpoint::Checkpoint::channel_versions`].
    pub fn channel_versions(&self) -> &BTreeMap<String, u64> {
        &self.channel_versions
    }

    /// This step's accumulated raw write values for every
    /// [`ChannelSet::with_delta`]-tracked channel, reset when the stamped
    /// step advances. See [`crate::checkpoint::Checkpoint::channel_deltas`].
    pub fn step_deltas(&self) -> &BTreeMap<String, Vec<Value>> {
        &self.step_deltas
    }
}

/// Extracts `(channel_versions, channel_deltas)` to embed into a freshly
/// built [`crate::checkpoint::Checkpoint`], downcasting `state` to
/// [`ChannelState`] when the graph uses the channel model.
///
/// For any other `State` type (a plain whole-state graph) a single
/// `"state"` channel is reported at `fallback_version`, with no deltas — see
/// the module docs on [`crate::checkpoint::Checkpoint::channel_versions`].
///
/// Shared by every checkpoint-construction call site (the executor's normal/
/// failure/cancel boundaries in `compiled::boundary`, and
/// `compiled::state_api`'s `update_state`) so a normal superstep boundary
/// and a manual write can never disagree about what they persist here (the
/// "one write path" contract — I5/R3).
pub fn channel_bookkeeping<State: 'static>(
    state: &State,
    fallback_version: u64,
) -> (
    BTreeMap<String, u64>,
    BTreeMap<String, Vec<serde_json::Value>>,
) {
    match (state as &dyn std::any::Any).downcast_ref::<ChannelState>() {
        Some(channel_state) => (
            channel_state.channel_versions().clone(),
            channel_state.step_deltas().clone(),
        ),
        None => {
            let mut versions = BTreeMap::new();
            versions.insert("state".to_string(), fallback_version);
            (versions, BTreeMap::new())
        }
    }
}

/// `ChannelState` is its own [`StateReducer`]: the `&self` receiver is unused
/// (merge rules live in the running `state`'s [`ChannelSet`]), so any
/// `ChannelState` may be passed to `set_reducer`.
impl StateReducer<ChannelState, ChannelUpdate> for ChannelState {
    fn apply(&self, state: ChannelState, update: ChannelUpdate) -> Result<ChannelState> {
        state.merge(update)
    }
}

#[cfg(test)]
mod test;
