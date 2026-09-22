//! Builder types for the durable graph.
//!
//! Everything here is data accumulated by a [`GraphBuilder`]; the behavior
//! that acts on it (adding nodes/edges, validation, `compile`) lives in
//! `builder/mod.rs`. [`GraphBuilder::compile`](super::GraphBuilder::compile)
//! consumes a [`GraphBuilder`] and produces a [`crate::CompiledGraph`], whose
//! own fields largely mirror the ones declared here (see
//! `compiled::types::CompiledGraph`).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use crate::Result;
use crate::checkpoint::PendingWrite;
use crate::command::NodeResult;
use crate::reducer::StateReducer;
use tinyagents_harness::ids::{GraphId, NodeId, RunId, TaskId, ThreadId};

/// The reserved virtual entry node.
pub const START: &str = "__start__";
/// The reserved virtual terminal node.
pub const END: &str = "__end__";

/// Boxed future produced by a durable node handler.
pub type NodeFuture<Update> = Pin<Box<dyn Future<Output = Result<NodeResult<Update>>> + Send>>;

/// A durable node handler: receives a state snapshot and per-task context,
/// returns a [`NodeResult`].
///
/// Internally every handler receives the step's committed state as an
/// `Arc<State>` (M2 in `docs/runtime-comparison/code-review-graph.md`): a
/// superstep clones `State` at most once (building this `Arc`), and every
/// branch/attempt within that step shares it via a cheap `Arc::clone`
/// instead of re-cloning the whole state. [`super::GraphBuilder::add_node`]
/// (the by-value convenience entry point) is a thin adapter over this
/// signature that clones out of the `Arc` on every invocation; callers that
/// want the zero-clone path use
/// [`super::GraphBuilder::add_node_shared`], whose closure receives the
/// `Arc<State>` directly.
pub type NodeHandler<State, Update> =
    dyn Fn(Arc<State>, NodeContext) -> NodeFuture<Update> + Send + Sync;

/// A conditional routing function over committed state. Returns a route label
/// resolved against the node's route table at the step boundary.
///
/// Internally this returns a typed [`Route`] rather than a bare `String` —
/// `Route` is `From<String>`/cheaply stringifies, so this is purely a
/// representation change and does not affect
/// [`super::GraphBuilder::add_conditional_edges`]'s public signature, which
/// still accepts any router closure returning `impl ToString`.
pub type RouterFn<State> = dyn Fn(&State) -> Route + Send + Sync;

/// Identifies one branch of a concurrent (fan-out) superstep.
///
/// When a graph compiled with [`crate::GraphBuilder::with_parallel`]
/// runs more than one active node in a single superstep, every branch executes
/// against its own cloned `State` snapshot and receives a distinct `ForkId` on
/// its [`NodeContext`]. The `branch_index` is the branch's position in the
/// deterministically-ordered active set, so a handler can tell which fork it is
/// (e.g. to seed per-fork randomness or pick a strategy) and the executor can
/// keep reducer application reproducible regardless of completion order.
///
/// In sequential mode (the default), and in a parallel step that happens to
/// have a single active node, `NodeContext::fork` is `None`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForkId {
    /// The branch's index in the superstep's active set (0-based, stable).
    pub branch_index: usize,
    /// The node executing on this branch.
    pub node: NodeId,
}

impl ForkId {
    /// Creates a fork id for `node` at `branch_index` within the active set.
    pub fn new(branch_index: usize, node: NodeId) -> Self {
        Self { branch_index, node }
    }
}

/// The heartbeat channel between a running node handler and the executor's
/// idle-timeout watcher (see [`NodeContext::heartbeat`]).
///
/// Cheap to clone; every clone of a [`NodeContext`] shares the same clock,
/// which is how a `heartbeat()` call made *inside* the handler future is
/// observed by the timeout race wrapped *around* it. A fresh clock is built
/// per activation; a hand-built context can use the [`Default`].
#[derive(Clone, Default)]
pub struct IdleClock {
    notify: Arc<tokio::sync::Notify>,
    beats: Arc<std::sync::atomic::AtomicU64>,
}

impl IdleClock {
    /// Records a heartbeat, waking the idle-timeout watcher so it re-arms.
    pub fn touch(&self) {
        self.beats
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.notify.notify_one();
    }

    /// Total heartbeats recorded so far.
    pub fn beats(&self) -> u64 {
        self.beats.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Resolves once `idle` elapses with no heartbeat in between; every
    /// [`Self::touch`] restarts the window. Never resolves if heartbeats keep
    /// arriving inside the window. With no heartbeat at all this resolves
    /// exactly `idle` after it is first polled — a flat timeout.
    pub(crate) async fn idle_elapsed(&self, idle: Duration) {
        loop {
            let sleep = tokio::time::sleep(idle);
            tokio::pin!(sleep);
            tokio::select! {
                _ = &mut sleep => return,
                _ = self.notify.notified() => continue,
            }
        }
    }
}

impl std::fmt::Debug for IdleClock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IdleClock")
            .field("beats", &self.beats())
            .finish()
    }
}

/// Per-task runtime context passed to a durable node handler.
///
/// The context exposes run identity, the current step, and — crucially — an
/// optional `resume` value. On a normal run `resume` is `None`; when a run is
/// resumed after an interrupt, the interrupted node is re-run with `resume` set
/// to the value carried by the resume command.
#[derive(Clone)]
pub struct NodeContext {
    /// The graph currently executing this node.
    pub graph_id: GraphId,
    /// The node being executed.
    pub node_id: NodeId,
    /// The current run id.
    pub run_id: RunId,
    /// The thread id when checkpointing is enabled.
    pub thread_id: Option<ThreadId>,
    /// The 1-based superstep number.
    pub step: usize,
    /// Resume value supplied by `CompiledGraph::resume`, if any.
    pub resume: Option<serde_json::Value>,
    /// The branch identity when this node runs as one fork of a concurrent
    /// (fan-out) superstep; `None` in sequential mode or single-node steps.
    pub fork: Option<ForkId>,
    /// The per-invocation argument when this activation was scheduled by a
    /// [`crate::Send`] packet or seeded through
    /// [`crate::GraphInput`]; `None` for normal edge/`goto` activations.
    /// This is how map-reduce / search-fanout branches and external graph
    /// inputs receive custom data that differs from the graph's shared
    /// committed state.
    ///
    /// `Arc`-wrapped (M2) so a repeated `Send` fan-out of the same node, and
    /// every retry attempt of one activation, share the same allocation
    /// instead of deep-cloning the argument per attempt. Serializes
    /// transparently as the underlying `serde_json::Value` (serde's blanket
    /// `Arc<T>` impl), so on-disk checkpoint records are unaffected.
    pub send_arg: Option<Arc<serde_json::Value>>,
    /// The root run id of the recursion tree this node executes within. For a
    /// top-level run this equals `run_id`; for a subgraph/sub-agent child run it
    /// is the shared ancestor, so a child a node spawns can preserve the root.
    pub root_run_id: Option<RunId>,
    /// The enclosing run's live recursion stack (root-first). A subgraph node
    /// seeds an embedded child graph with these frames so the child extends the
    /// parent's recursion tree instead of starting a fresh one.
    pub recursion_frames: Vec<crate::recursion::RecursionFrame>,
    /// A per-run collector the executor provides so a subgraph node can report
    /// the [`ChildRun`](crate::ChildRun) it spawned back to the enclosing
    /// run; `None` when no executor sink is attached (e.g. a hand-built context).
    pub child_runs: Option<crate::recursion::ChildRunSink>,
    /// Complete host-owned recursive-agent binding for this execution, if one
    /// was supplied at the graph entry point.
    pub agent_binding: Option<crate::subagent_node::AgentInvocationBinding>,
    /// Stable identity of this scheduled activation within its superstep
    /// (R5). Distinguishes repeated `Send` fan-out activations of the same
    /// node — a subgraph node consults this (with [`Self::siblings`]) to
    /// namespace its child checkpoint per fan-out branch instead of sharing
    /// one namespace across every concurrent activation of the node (I1).
    pub task_id: TaskId,
    /// The number of activations of [`Self::node_id`] in this same
    /// superstep's active set (I1). `1` for an ordinary (non-fan-out)
    /// activation; greater than `1` means a `Send` fan-out scheduled several
    /// concurrent activations of this node this step.
    pub siblings: usize,
    /// The channel versions (I5/R3) as this node's state snapshot sees them
    /// — [`crate::channel::ChannelState::channel_versions`] for a channel
    /// graph, or a single `{"state": n}` entry for a plain whole-state
    /// graph. Compared against [`Self::versions_seen`] by
    /// [`Self::changed_since_last_run`].
    pub channel_versions: BTreeMap<String, u64>,
    /// This node's own snapshot of [`Self::channel_versions`] as of the last
    /// time it ran (empty on a node's first-ever activation in the thread).
    pub versions_seen: BTreeMap<String, u64>,
    /// Heartbeat channel for the node's idle timeout
    /// ([`crate::NodePolicy::idle_timeout`]); see [`Self::heartbeat`].
    pub idle_clock: IdleClock,
    /// This task's [`Self::durable_task`] memo buffer, shared by every clone
    /// of the context (a retried attempt sees the first attempt's memos).
    /// Pre-seeded by the executor with the durable-task writes the task's
    /// checkpoint already holds (so a re-run after an interrupt, crash, or
    /// retry hits instead of re-executing), and appended to on every miss;
    /// the executor folds it back into the boundary checkpoint's
    /// `pending_writes` when the task stalls. Empty on a hand-built context.
    pub(crate) durable_writes: Arc<std::sync::Mutex<Vec<PendingWrite>>>,
}

impl NodeContext {
    /// Runs `fut` at most once per `(task, key)` across re-runs of this task.
    ///
    /// A node handler is re-run from its start after an interrupt/resume,
    /// a failure/retry, or an in-process node retry, so any side effect it
    /// performs (an API call, a payment, a counter increment) would repeat.
    /// Wrapping that side effect in `durable_task` memoises its output in
    /// this task's checkpoint write ledger, keyed by the task id and `key`:
    /// the first execution awaits `fut`, serializes its `Ok` output as a
    /// [`PendingWrite::durable_task`] memo, and returns it; a later re-run
    /// of the same task finds the memo and returns the stored value
    /// **without polling `fut` at all**. Memos are only ever recorded for a
    /// successful `fut`; an `Err` is returned unmemoised so a retry re-runs
    /// the step. Keys are independent: a handler that memoised `"a"` and
    /// then failed before `"b"` replays `"a"` and runs `"b"` fresh.
    ///
    /// The memo is scoped to this task in this thread — it is not a
    /// cross-run cache (see [`crate::TaskCache`] for that) — and it is only
    /// durable across process restarts when the graph runs on a
    /// checkpointed thread; without a checkpointer it still dedupes within
    /// one run (in-process retries). Two calls with the same `key` inside
    /// one handler execution return the same stored value.
    pub async fn durable_task<T, F>(&self, key: &str, fut: F) -> Result<T>
    where
        T: serde::Serialize + serde::de::DeserializeOwned,
        F: Future<Output = Result<T>>,
    {
        let hit = self
            .lock_durable_writes()
            .iter()
            .find(|w| w.durable_task_key() == Some(key))
            .map(|w| w.payload.clone());
        if let Some(payload) = hit {
            return serde_json::from_value(payload).map_err(|err| {
                crate::TinyAgentsError::Serialization(serde::de::Error::custom(format!(
                    "durable_task `{key}` of task `{}` (node `{}`) holds a memo that does not                      decode as the requested type: {err}",
                    self.task_id.as_str(),
                    self.node_id
                )))
            });
        }
        let value = fut.await?;
        let payload = serde_json::to_value(&value)?;
        let mut writes = self.lock_durable_writes();
        let idx = writes.iter().map(|w| w.idx).max().unwrap_or(0).max(0) + 1;
        writes.push(PendingWrite::durable_task(
            self.node_id.clone(),
            self.task_id.clone(),
            idx,
            key,
            payload,
        ));
        Ok(value)
    }

    /// Locks the durable-task memo buffer, tolerating a poisoned lock (the
    /// buffer is a plain `Vec` push/scan, so a panic mid-hold leaves it
    /// consistent).
    pub(crate) fn lock_durable_writes(&self) -> std::sync::MutexGuard<'_, Vec<PendingWrite>> {
        self.durable_writes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// A snapshot of this task's durable-task memo writes (pre-seeded plus
    /// any recorded by [`Self::durable_task`] so far).
    pub(crate) fn durable_writes_snapshot(&self) -> Vec<PendingWrite> {
        self.lock_durable_writes().clone()
    }

    /// Signals liveness to the executor's idle-timeout watcher, restarting
    /// the node's [`crate::NodePolicy::idle_timeout`] window. Cheap (an
    /// atomic increment plus a notify); a no-op for a node with no idle
    /// timeout configured. Does not affect the flat
    /// [`crate::NodePolicy::timeout`] ceiling.
    pub fn heartbeat(&self) {
        self.idle_clock.touch();
    }

    /// This activation's stable task identity (R5). See the field docs on
    /// [`Self::task_id`].
    pub fn task_id(&self) -> &TaskId {
        &self.task_id
    }

    /// Whether `channel` has been written since the last time this node ran
    /// (I5/R3): compares [`Self::channel_versions`] (current) against
    /// [`Self::versions_seen`] (this node's own last-observed snapshot). A
    /// channel this node has never seen before (including a node's very
    /// first activation) counts as changed whenever it has ever been
    /// written at all.
    pub fn changed_since_last_run(&self, channel: &str) -> bool {
        self.channel_versions.get(channel).copied().unwrap_or(0)
            != self.versions_seen.get(channel).copied().unwrap_or(0)
    }
}

impl std::fmt::Debug for NodeContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NodeContext")
            .field("graph_id", &self.graph_id)
            .field("node_id", &self.node_id)
            .field("run_id", &self.run_id)
            .field("thread_id", &self.thread_id)
            .field("step", &self.step)
            .field("resume", &self.resume)
            .field("fork", &self.fork)
            .field("send_arg", &self.send_arg)
            .field("root_run_id", &self.root_run_id)
            .field("recursion_frames", &self.recursion_frames)
            .field("has_child_runs", &self.child_runs.is_some())
            .field("has_agent_binding", &self.agent_binding.is_some())
            .field("task_id", &self.task_id)
            .field("siblings", &self.siblings)
            .field("channel_versions", &self.channel_versions)
            .field("versions_seen", &self.versions_seen)
            .field("idle_clock", &self.idle_clock)
            .field("durable_writes", &self.lock_durable_writes().len())
            .finish()
    }
}

/// Behavior-free, introspectable metadata attached to a node by the builder.
///
/// Markers and free-form metadata recorded here never affect execution; they
/// exist so the [topology export](crate::export) can describe what a node
/// *is* (a subgraph embedding, an interrupt point, a deferred join, …) without
/// inspecting the node's opaque handler closure. All fields are optional and
/// additive: an unset [`NodeMeta`] (the [`Default`]) contributes nothing.
#[derive(Clone, Debug, Default)]
pub(crate) struct NodeMeta {
    /// A human-readable node kind (e.g. `model`, `tool`, `subgraph`).
    pub(crate) kind: Option<String>,
    /// The node pauses the run before/at execution (an interrupt point).
    pub(crate) interrupt: bool,
    /// The node is a deferred join — it activates only after the rest of the
    /// active frontier has drained (a barrier-style synthesis node).
    pub(crate) deferred: bool,
    /// The node embeds and runs a child graph (a subgraph node).
    pub(crate) subgraph: bool,
    /// Declared `goto` destination hints for a command-routing node, in the
    /// order they were registered. Purely advisory: the runtime resolves the
    /// real target from the emitted [`crate::Command`] at runtime.
    pub(crate) command_destinations: Vec<NodeId>,
    /// Arbitrary, sorted key/value annotations carried into the export.
    pub(crate) metadata: BTreeMap<String, String>,
}

/// Encodes an `Update` for checkpoint storage.
pub(crate) type EncodeUpdateFn<Update> =
    dyn Fn(&Update) -> serde_json::Result<serde_json::Value> + Send + Sync;
/// Decodes a stored value back into an `Update`.
pub(crate) type DecodeUpdateFn<Update> =
    dyn Fn(serde_json::Value) -> serde_json::Result<Update> + Send + Sync;

/// A type-erased `Update` serde codec, installed by
/// [`GraphBuilder::interrupt_after`](super::GraphBuilder::interrupt_after)
/// (the one builder entry point that requires `Update: Serialize +
/// DeserializeOwned`) so the executor can persist a node's deferred result
/// in the checkpoint write ledger and replay it on resume, while
/// [`GraphBuilder`]/[`crate::CompiledGraph`] themselves stay bound-free
/// over `Update`. Same pattern as
/// [`crate::CompiledGraph::with_cached_node`].
pub(crate) struct UpdateCodec<Update> {
    pub(crate) encode: Arc<EncodeUpdateFn<Update>>,
    pub(crate) decode: Arc<DecodeUpdateFn<Update>>,
}

impl<Update> UpdateCodec<Update>
where
    Update: serde::Serialize + serde::de::DeserializeOwned + 'static,
{
    /// Builds the serde codec for `Update`.
    pub(crate) fn serde() -> Self {
        Self {
            encode: Arc::new(|update: &Update| serde_json::to_value(update)),
            decode: Arc::new(|value: serde_json::Value| serde_json::from_value(value)),
        }
    }
}

impl<Update> Clone for UpdateCodec<Update> {
    fn clone(&self) -> Self {
        Self {
            encode: self.encode.clone(),
            decode: self.decode.clone(),
        }
    }
}

/// A compiled-in node: its handler. The node's id lives as the key of the
/// `nodes` map it is stored in ([`crate::compiled::CompiledGraph::nodes`]).
pub(crate) struct BuilderNode<State, Update> {
    pub(crate) handler: Arc<NodeHandler<State, Update>>,
}

impl<State, Update> Clone for BuilderNode<State, Update> {
    fn clone(&self) -> Self {
        Self {
            handler: self.handler.clone(),
        }
    }
}

/// A small newtype wrapper for a conditional-route label.
///
/// Routers may return any `impl ToString` label (a plain `&str`/`String`, or a
/// user-defined route enum that implements `Display`). `Route` is an optional
/// ergonomic helper for building route tables and for routers that prefer to
/// return a typed value instead of a bare string; it stringifies via
/// [`ToString`] at the route-table boundary.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Route(pub String);

impl Route {
    /// Wraps any `impl ToString` (e.g. a route enum with `Display`) as a label.
    pub fn new(label: impl ToString) -> Self {
        Self(label.to_string())
    }

    /// The label as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Route {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<Route> for String {
    fn from(route: Route) -> Self {
        route.0
    }
}

impl From<String> for Route {
    fn from(label: String) -> Self {
        Self(label)
    }
}

impl From<&str> for Route {
    fn from(label: &str) -> Self {
        Self(label.to_string())
    }
}

/// Tunable per-graph defaults applied to a [`GraphBuilder`] in one call via
/// [`GraphBuilder::set_defaults`].
///
/// Every field is optional; only the `Some` fields override the builder's
/// current configuration, so partial defaults compose with explicit
/// `with_*` calls. All fields are opt-in and additive — an unset
/// [`GraphDefaults`] (the [`Default`]) changes nothing.
#[derive(Clone, Debug, Default)]
pub struct GraphDefaults {
    /// Maximum number of supersteps before [`crate::TinyAgentsError::RecursionLimit`].
    pub recursion_limit: Option<usize>,
    /// Whether the active node set of a superstep runs concurrently.
    pub parallel: Option<bool>,
    /// Upper bound on the number of branches run concurrently within one step
    /// (only meaningful when `parallel` is enabled). `None` means unbounded.
    pub max_concurrency: Option<usize>,
    /// Default wall-clock timeout applied to every node handler; on elapse the
    /// run fails with [`crate::TinyAgentsError::Timeout`]. `None` means no
    /// per-node timeout.
    pub node_timeout: Option<Duration>,
}

/// Conditional routing for a node: a router function plus its route table.
pub(crate) struct Branch<State> {
    pub(crate) router: Arc<RouterFn<State>>,
    pub(crate) routes: HashMap<String, NodeId>,
}

impl<State> Clone for Branch<State> {
    fn clone(&self) -> Self {
        Self {
            router: self.router.clone(),
            routes: self.routes.clone(),
        }
    }
}

/// A mutable, ergonomic builder for a durable state graph.
///
/// `State` is the committed graph state; `Update` is the partial-update type
/// merged through the configured [`StateReducer`]. For whole-state graphs use
/// `Update == State` together with the overwrite reducer (see
/// [`super::GraphBuilder::overwrite`]).
pub struct GraphBuilder<State, Update> {
    pub(crate) graph_id: GraphId,
    /// Optional human-readable graph name carried into the topology export
    /// (the `graph_id` remains the stable identifier).
    pub(crate) name: Option<String>,
    pub(crate) nodes: HashMap<NodeId, BuilderNode<State, Update>>,
    /// Static/waiting edges: source node -> its ordered, deduplicated list of
    /// successor targets. A node may have more than one static successor
    /// (fan-out): every target in the list activates, not just one.
    pub(crate) edges: HashMap<NodeId, Vec<NodeId>>,
    pub(crate) branches: HashMap<NodeId, Branch<State>>,
    /// Exhaustive route-label declarations registered via
    /// [`super::GraphBuilder::add_conditional_edges_checked`]: node -> every
    /// label its router may produce. [`super::GraphBuilder::validate_routes`]
    /// cross-checks these against the node's actual route table at build
    /// time, catching a typo'd label before it can fail a run with
    /// [`crate::TinyAgentsError::MissingRoute`].
    pub(crate) route_label_checks: HashMap<NodeId, Vec<String>>,
    pub(crate) command_nodes: HashSet<NodeId>,
    /// Barrier/waiting edges: target node -> set of predecessor nodes that must
    /// all have completed (across steps) before the target activates.
    pub(crate) waiting: HashMap<NodeId, HashSet<NodeId>>,
    /// Mixed fan-in barrier relief registrations; see
    /// [`super::BarrierRelief`].
    pub(crate) barrier_reliefs: Vec<super::BarrierRelief>,
    pub(crate) reducer: Option<Arc<dyn StateReducer<State, Update>>>,
    pub(crate) recursion_limit: usize,
    /// When true, active nodes in a superstep run concurrently (fan-out).
    pub(crate) parallel: bool,
    /// Upper bound on concurrently-running branches per step (`None` = unbounded).
    pub(crate) max_concurrency: Option<usize>,
    /// Default per-node handler timeout (`None` = no timeout).
    pub(crate) node_timeout: Option<Duration>,
    /// Behavior-free per-node markers/metadata surfaced by the topology export.
    pub(crate) node_meta: HashMap<NodeId, NodeMeta>,
    /// Per-node execution policies (retry/timeout/cache/on_error/defer); see
    /// [`super::NodePolicy`].
    pub(crate) node_policies: HashMap<NodeId, super::NodePolicy<State, Update>>,
    /// Graph-wide default execution policy every node falls back to, field
    /// by field, when it has no per-node override.
    pub(crate) node_defaults: Option<super::NodePolicy<State, Update>>,
    /// Nodes the executor pauses *before* running (see
    /// [`super::GraphBuilder::interrupt_before`]).
    pub(crate) interrupt_before: HashSet<NodeId>,
    /// Nodes the executor pauses *after* running, before applying their
    /// result (see [`super::GraphBuilder::interrupt_after`]).
    pub(crate) interrupt_after: HashSet<NodeId>,
    /// The `Update` codec `interrupt_after` needs to persist a deferred
    /// result; `None` until the first `interrupt_after` call.
    pub(crate) update_codec: Option<UpdateCodec<Update>>,
}
