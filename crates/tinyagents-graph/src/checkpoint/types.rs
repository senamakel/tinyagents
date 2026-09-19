//! Checkpoint records and metadata — the persisted snapshots that make every
//! level of a recursive graph run resumable and forkable.
//!
//! Checkpoints are graph-runtime persistence, separate from harness memory and
//! long-term stores. They are written at superstep boundaries only — never
//! mid-node — because rerunning a node from its start is far easier to reason
//! about than suspending an async Rust stack, and it matches interrupt/resume
//! semantics exactly.
//!
//! Each record carries a `thread_id` lineage key, a `parent_checkpoint_id`
//! chain (the spine that time-travel and forking walk), and a `namespace` that
//! scopes nested subgraph checkpoints so a parent run and the child graphs it
//! embeds never overwrite each other.

use std::collections::BTreeMap;
use std::fmt;

use crate::command::{Interrupt, RouteTarget};
use tinyagents_harness::ids::{NodeId, TaskId};

/// Default value for a `TaskId` field carrying `#[serde(default = "..")]`:
/// `TaskId` is a foreign newtype (from `tinyagents_harness`), so it cannot
/// implement `Default` here (orphan rule) — this free function stands in for
/// it. An empty task id is exactly what a checkpoint written before task
/// identities existed decodes to.
fn empty_task_id() -> TaskId {
    TaskId::from(String::new())
}

/// `#[serde(skip_serializing_if = "..")]` predicate pairing with
/// [`empty_task_id`].
fn task_id_is_empty(id: &TaskId) -> bool {
    id.as_str().is_empty()
}

/// The current on-disk checkpoint record shape (checkpoint format v2): a
/// single `tasks`/`completed` pair replaces the four overlapping v1
/// projections of pending work (`next_nodes`, `completed_tasks` +
/// `completed_routes`, `pending_activations`). See the module docs on
/// [`Checkpoint`] and `docs/modules/graph/checkpointing.md` for the full
/// decode story.
pub const CHECKPOINT_FORMAT_VERSION: u32 = 2;

/// `#[serde(default = "..")]` for [`Checkpoint::version`]: a record with no
/// `version` field on disk predates the field entirely, which is exactly
/// what checkpoint format v1 (the shape before this constant existed) looked
/// like.
fn checkpoint_version_v1() -> u32 {
    1
}

/// Why a checkpoint was written.
///
/// Mirrors the documented metadata `source` taxonomy: a checkpoint is produced
/// by the initial graph `input`, a normal superstep `loop` boundary, a manual
/// `update` (a state write attributed through the reducers), or a `fork` that
/// branches a thread for time-travel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckpointSource {
    /// The initial state supplied when a run starts.
    Input,
    /// A normal superstep boundary in the execution loop.
    Loop,
    /// A manual state update written through the channel reducers.
    Update,
    /// A fork that branches a thread for time-travel/replay.
    Fork,
}

impl CheckpointSource {
    /// The lowercase wire/string form used in checkpoint metadata.
    pub fn as_str(&self) -> &'static str {
        match self {
            CheckpointSource::Input => "input",
            CheckpointSource::Loop => "loop",
            CheckpointSource::Update => "update",
            CheckpointSource::Fork => "fork",
        }
    }

    /// Parses a source string, returning `None` for unknown values.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "input" => Some(CheckpointSource::Input),
            "loop" => Some(CheckpointSource::Loop),
            "update" => Some(CheckpointSource::Update),
            "fork" => Some(CheckpointSource::Fork),
            _ => None,
        }
    }
}

impl fmt::Display for CheckpointSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// When committed checkpoints are persisted relative to graph execution.
///
/// The default is [`DurabilityMode::Sync`], which preserves today's behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DurabilityMode {
    /// Persist a checkpoint before the next step starts. The boundary state is
    /// durable before any successor node runs — the strongest guarantee.
    #[default]
    Sync,
    /// Persist off the critical path: the boundary checkpoint write is handed
    /// to a spawned background task while the next step executes, so superstep
    /// latency does not pay for checkpoint I/O.
    ///
    /// Failure semantics: a background write error is **not** silently lost.
    /// The executor records it and fails the run at the next durability
    /// boundary that observes it; at the terminal boundary (and at any
    /// interrupt boundary) every in-flight write is awaited, so a completed
    /// run's result always reflects its checkpoints' persistence. The final
    /// (terminal) and interrupt checkpoints themselves are always written
    /// synchronously. The `CheckpointSaved` event for a background write is
    /// emitted when the write completes, so its ordering relative to later
    /// step events is not deterministic. Outside a tokio runtime this mode
    /// degrades to [`DurabilityMode::Sync`].
    Async,
    /// Persist only the final checkpoint when the graph exits (or pauses on an
    /// interrupt). Intermediate boundaries are not written, trading
    /// resumability granularity for fewer writes.
    Exit,
}

/// Coordinates that address a checkpoint within a thread.
///
/// `checkpoint_id` of `None` selects the latest checkpoint for the thread;
/// `namespace` scopes nested subgraph checkpoints so a parent run and its
/// embedded child graphs never collide.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CheckpointConfig {
    /// Thread lineage key.
    pub thread_id: String,
    /// Specific checkpoint to address, or `None` for the latest.
    pub checkpoint_id: Option<String>,
    /// Namespace scoping for nested subgraph checkpoints.
    pub namespace: Vec<String>,
}

impl CheckpointConfig {
    /// Builds a config addressing the latest checkpoint of `thread_id` at the
    /// root namespace.
    pub fn latest(thread_id: impl Into<String>) -> Self {
        Self {
            thread_id: thread_id.into(),
            checkpoint_id: None,
            namespace: Vec::new(),
        }
    }
}

/// The documented core persistence unit: a checkpoint together with its config,
/// the config of its parent, and the per-task pending writes preserved with it.
///
/// Backends compose this from `get` + `list` via
/// [`Checkpointer::get_tuple`](crate::Checkpointer::get_tuple).
#[derive(Clone, Debug)]
pub struct CheckpointTuple<State> {
    /// Config that addresses this checkpoint.
    pub config: CheckpointConfig,
    /// The checkpoint record itself.
    pub checkpoint: Checkpoint<State>,
    /// Config addressing the parent checkpoint, when one exists.
    pub parent_config: Option<CheckpointConfig>,
    /// Pending writes carried by the checkpoint.
    pub pending_writes: Vec<PendingWrite>,
}

/// A persisted snapshot of a graph run at a superstep boundary.
///
/// Derives `Serialize`/`Deserialize` with serde's conditional bounds: a
/// `Checkpoint<State>` is (de)serializable exactly when `State` is, which is
/// what lets file-backed backends such as
/// [`FileCheckpointer`](crate::FileCheckpointer) round-trip whole records
/// through JSON. The in-memory path never needs it.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Checkpoint<State> {
    /// The on-disk record shape. `1` (the implicit shape before this field
    /// existed — `#[serde(default = "checkpoint_version_v1")]`) or
    /// [`CHECKPOINT_FORMAT_VERSION`] (`2`). Every writer in this crate stamps
    /// `2`; a `1` is only ever seen decoding a record written by an older
    /// build. See [`Checkpoint::normalize`].
    #[serde(default = "checkpoint_version_v1")]
    pub version: u32,
    /// Wall-clock time this checkpoint was written, in milliseconds since the
    /// Unix epoch (see [`tinyagents_harness::ids::now_ms`]).
    ///
    /// `#[serde(default)]` (`0`) for a v1 record, which never carried a
    /// timestamp at all — `0` is a visibly-unset sentinel, not a plausible
    /// wall-clock value.
    #[serde(default)]
    pub created_at: u64,
    /// Checkpoint lineage key for a conversation/workflow/tenant run series.
    pub thread_id: String,
    /// This checkpoint's id within the thread.
    pub checkpoint_id: String,
    /// The run that produced this checkpoint, when known.
    ///
    /// Optional and back-compatible: pre-existing records and manual snapshots
    /// may leave it `None`. The executor stamps it so checkpoints can be deleted
    /// by run id via [`Checkpointer::delete_by_run`](crate::Checkpointer::delete_by_run).
    pub run_id: Option<String>,
    /// The previous checkpoint id in the thread lineage.
    pub parent_checkpoint_id: Option<String>,
    /// Namespace scoping for nested subgraph checkpoints.
    pub namespace: Vec<String>,
    /// Committed graph state at this boundary.
    pub state: State,
    /// The single source of truth for what runs when this checkpoint is
    /// resumed: every pending activation, preserving each one's
    /// per-invocation [`Send`](crate::Send) argument and task identity.
    ///
    /// Checkpoint format v2 (see [`Checkpoint::version`]). Replaces the v1
    /// pair of `next_nodes` (a node-id-only projection) and
    /// `pending_activations` (an `Option`-wrapped superset that was the same
    /// information, just optional) with exactly one field that is never
    /// ambiguous with anything else on the record. A v1 record decodes with
    /// this empty; call [`Checkpoint::normalize`] (every bundled backend's
    /// decode path does) to populate it from the legacy fields.
    #[serde(default)]
    pub tasks: Vec<PendingActivation>,
    /// The single source of truth for what completed in the step that
    /// produced this checkpoint, and how each task explicitly routed (if it
    /// returned a `Command::goto`).
    ///
    /// Checkpoint format v2. Replaces the v1 pair of parallel vectors
    /// `completed_tasks: Vec<NodeId>` and
    /// `completed_routes: Vec<Vec<RouteTarget>>`, which had to stay
    /// positionally aligned by convention rather than by type. A v1 record
    /// decodes with this empty; [`Checkpoint::normalize`] zips the legacy
    /// pair back into this shape.
    #[serde(default)]
    pub completed: Vec<CompletedTask>,
    /// Per-task partial writes preserved when a step partially completes.
    pub pending_writes: Vec<PendingWrite>,
    /// Interrupts that paused the run at this boundary.
    pub interrupts: Vec<Interrupt>,
    /// Barrier (waiting-edge) arrivals accumulated across supersteps, persisted
    /// so a join node's precondition survives an interrupt/failure + resume.
    ///
    /// `#[serde(default)]` for back-compat: older checkpoints load with an
    /// empty set (the pre-field behavior, where arrivals were run-local).
    #[serde(default)]
    pub barrier_arrivals: Vec<BarrierArrivals>,
    /// Cumulative per-channel version counters as of this boundary (I5/R3):
    /// name -> a monotonically-increasing count of writes to that channel.
    ///
    /// For a [`crate::channel::ChannelState`] graph this is
    /// [`crate::channel::ChannelState::channel_versions`] verbatim. For a
    /// plain whole-`State` graph (any other `State` type) it is the single
    /// entry `{"state": <boundary count>}`, bumped once per checkpoint —
    /// see `compiled::channel_bookkeeping`, the one function every
    /// checkpoint-construction call site (a normal superstep boundary,
    /// `update_state`, `fork_state`) uses to fill this field, so replay and
    /// a manual write cannot disagree about it.
    ///
    /// `#[serde(default)]` for back-compat: a checkpoint written before this
    /// field existed decodes with it empty.
    #[serde(default)]
    pub channel_versions: BTreeMap<String, u64>,
    /// Per-node snapshot of [`Checkpoint::channel_versions`] as of the last
    /// time each node ran, keyed by node id string (`NodeId` itself has no
    /// `Ord` impl to key a `BTreeMap` on). Backs
    /// [`crate::builder::NodeContext::changed_since_last_run`): a node
    /// compares its own entry here (what it last observed) against the
    /// checkpoint's live `channel_versions` (what is current) to tell
    /// whether a channel changed since it last ran.
    ///
    /// `#[serde(default)]` for back-compat.
    #[serde(default)]
    pub versions_seen: BTreeMap<String, BTreeMap<String, u64>>,
    /// Per-step delta-channel write history (I5/R3): for every channel
    /// registered with [`crate::channel::ChannelSet::with_delta`], the raw
    /// values written to it *in the step that produced this checkpoint*
    /// (not cumulative — each checkpoint carries only its own step's
    /// writes, which is what keeps per-checkpoint size bounded for a
    /// long-running append channel). Replayed across a thread's lineage by
    /// [`crate::Checkpointer::delta_history`].
    ///
    /// `#[serde(default)]` for back-compat and for every checkpoint of a
    /// graph that declares no delta-tracked channels (always empty there).
    #[serde(default)]
    pub channel_deltas: BTreeMap<String, Vec<serde_json::Value>>,
    /// Free-form metadata (source, step, etc.).
    pub metadata: serde_json::Value,

    // ---- Checkpoint format v1 fields (decode-only) -------------------------
    //
    // Every writer in this crate leaves these at their empty default, so a
    // freshly-written record serializes with none of them present
    // (`skip_serializing_if`) — only [`Checkpoint::tasks`]/
    // [`Checkpoint::completed`] above carry pending/completed work going
    // forward. They exist purely so a record written by a build that
    // predates checkpoint format v2 still deserializes; [`Checkpoint::normalize`]
    // is the single place that reads them and folds them into the v2 shape.
    // Every reader elsewhere in this crate (`compiled::{resume,boundary,
    // state_api,mod}`) reads `tasks`/`completed` only.
    /// v1: nodes that should run when resuming from this checkpoint. Decode-only.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub next_nodes: Vec<NodeId>,
    /// v1: nodes that completed in the step that produced this checkpoint.
    /// Decode-only.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub completed_tasks: Vec<NodeId>,
    /// v1: the explicit `Command::goto` routing for each entry of
    /// [`completed_tasks`](Self::completed_tasks), positionally aligned.
    /// Decode-only.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub completed_routes: Vec<Vec<RouteTarget>>,
    /// v1: pending activations superset of `next_nodes`. Decode-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending_activations: Option<Vec<PendingActivation>>,
}

/// One pending node activation persisted in a checkpoint: the node to run on
/// resume plus the optional per-invocation [`Send`](crate::Send)
/// argument that scheduled it.
///
/// The durable counterpart of the executor's in-flight activation. Persisting
/// the `send_arg` is what lets a map-reduce fanout survive an interrupt/failure
/// boundary — without it every pending worker re-runs with no argument.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PendingActivation {
    /// The node scheduled to run on resume.
    pub node: NodeId,
    /// The per-invocation `Send` argument, when the activation was a `Send`
    /// packet (plain edge/goto activations carry `None`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub send_arg: Option<serde_json::Value>,
    /// Stable identity of this scheduled task within its superstep.
    ///
    /// Unlike `node`, this distinguishes repeated `Send` fan-out activations
    /// targeting the same node. Empty on checkpoints written before task
    /// identities were persisted. Serializes transparently as the underlying
    /// string, so on-disk records are unaffected by the `String` -> `TaskId`
    /// type change (R5).
    #[serde(default = "empty_task_id", skip_serializing_if = "task_id_is_empty")]
    pub task_id: TaskId,
}

/// One task that completed in the step a checkpoint's boundary closes,
/// checkpoint format v2's replacement for the v1
/// `completed_tasks: Vec<NodeId>` / `completed_routes: Vec<Vec<RouteTarget>>`
/// pair (see [`Checkpoint::completed`]).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CompletedTask {
    /// The task that completed, unique within the superstep that produced
    /// it. Empty (`TaskId::from(String::new())`) for a task carried forward
    /// from a checkpoint written before task identities existed, or derived
    /// from a v1 record's `completed_tasks` (which carried no task id at
    /// all).
    #[serde(default = "empty_task_id", skip_serializing_if = "task_id_is_empty")]
    pub task_id: TaskId,
    /// The node that completed.
    pub node: NodeId,
    /// The explicit `Command::goto` routing this task returned, or empty when
    /// it returned none (route via static/conditional edges instead).
    #[serde(default)]
    pub routes: Vec<RouteTarget>,
}

impl CompletedTask {
    /// Builds a completed-task record with no explicit `Command::goto`
    /// routing (route via static/conditional edges).
    pub fn new(task_id: impl Into<TaskId>, node: impl Into<NodeId>) -> Self {
        Self {
            task_id: task_id.into(),
            node: node.into(),
            routes: Vec::new(),
        }
    }

    /// Builds a completed-task record carrying an explicit `Command::goto`
    /// routing.
    pub fn with_routes(
        task_id: impl Into<TaskId>,
        node: impl Into<NodeId>,
        routes: Vec<RouteTarget>,
    ) -> Self {
        Self {
            task_id: task_id.into(),
            node: node.into(),
            routes,
        }
    }
}

/// The persisted arrivals recorded against one barrier (waiting-edge) join node:
/// the predecessors that have already routed to it but whose join has not yet
/// fired.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BarrierArrivals {
    /// The waiting/join node.
    pub node: NodeId,
    /// The predecessor nodes that have arrived so far.
    pub arrived: Vec<NodeId>,
}

impl<State> Checkpoint<State> {
    /// Builds a fresh checkpoint format v2 record.
    ///
    /// Sensible defaults for everything except `state` and `tasks`: a
    /// freshly-minted [`checkpoint_id`](Self::checkpoint_id) (collision-free
    /// across process restarts, matching what every executor-driven write
    /// already used — see
    /// [`tinyagents_harness::ids::new_checkpoint_id`]), the current
    /// [`created_at`](Self::created_at), [`version`](Self::version) ==
    /// [`CHECKPOINT_FORMAT_VERSION`], and empty `thread_id`/`completed`/
    /// `pending_writes`/`interrupts`/`barrier_arrivals`/`namespace`, with
    /// `metadata` left `null`. Chain the `with_*` setters below to fill in
    /// the rest; every field is also directly `pub` for call sites that
    /// prefer plain field assignment.
    pub fn new(state: State, tasks: Vec<PendingActivation>) -> Self {
        Self {
            version: CHECKPOINT_FORMAT_VERSION,
            created_at: tinyagents_harness::ids::now_ms(),
            thread_id: String::new(),
            checkpoint_id: tinyagents_harness::ids::new_checkpoint_id()
                .as_str()
                .to_string(),
            run_id: None,
            parent_checkpoint_id: None,
            namespace: Vec::new(),
            state,
            tasks,
            completed: Vec::new(),
            pending_writes: Vec::new(),
            interrupts: Vec::new(),
            barrier_arrivals: Vec::new(),
            channel_versions: BTreeMap::new(),
            versions_seen: BTreeMap::new(),
            channel_deltas: BTreeMap::new(),
            metadata: serde_json::Value::Null,
            next_nodes: Vec::new(),
            completed_tasks: Vec::new(),
            completed_routes: Vec::new(),
            pending_activations: None,
        }
    }

    /// Alias for [`Checkpoint::new`] with no pending tasks yet — the start of
    /// a fluent build, e.g. `Checkpoint::builder(state).with_tasks(pending)`.
    pub fn builder(state: State) -> Self {
        Self::new(state, Vec::new())
    }

    /// Sets [`Checkpoint::thread_id`].
    pub fn with_thread_id(mut self, thread_id: impl Into<String>) -> Self {
        self.thread_id = thread_id.into();
        self
    }

    /// Sets [`Checkpoint::checkpoint_id`], overriding the freshly-minted
    /// default from [`Checkpoint::new`].
    pub fn with_checkpoint_id(mut self, checkpoint_id: impl Into<String>) -> Self {
        self.checkpoint_id = checkpoint_id.into();
        self
    }

    /// Sets [`Checkpoint::run_id`].
    pub fn with_run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = Some(run_id.into());
        self
    }

    /// Sets [`Checkpoint::parent_checkpoint_id`].
    pub fn with_parent_checkpoint_id(mut self, parent: Option<String>) -> Self {
        self.parent_checkpoint_id = parent;
        self
    }

    /// Sets [`Checkpoint::namespace`].
    pub fn with_namespace(mut self, namespace: Vec<String>) -> Self {
        self.namespace = namespace;
        self
    }

    /// Sets [`Checkpoint::tasks`].
    pub fn with_tasks(mut self, tasks: Vec<PendingActivation>) -> Self {
        self.tasks = tasks;
        self
    }

    /// Sets [`Checkpoint::completed`].
    pub fn with_completed(mut self, completed: Vec<CompletedTask>) -> Self {
        self.completed = completed;
        self
    }

    /// Sets [`Checkpoint::pending_writes`].
    pub fn with_pending_writes(mut self, writes: Vec<PendingWrite>) -> Self {
        self.pending_writes = writes;
        self
    }

    /// Sets [`Checkpoint::interrupts`].
    pub fn with_interrupts(mut self, interrupts: Vec<Interrupt>) -> Self {
        self.interrupts = interrupts;
        self
    }

    /// Sets [`Checkpoint::barrier_arrivals`].
    pub fn with_barrier_arrivals(mut self, barrier_arrivals: Vec<BarrierArrivals>) -> Self {
        self.barrier_arrivals = barrier_arrivals;
        self
    }

    /// Sets [`Checkpoint::metadata`].
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Sets [`Checkpoint::channel_versions`].
    pub fn with_channel_versions(mut self, channel_versions: BTreeMap<String, u64>) -> Self {
        self.channel_versions = channel_versions;
        self
    }

    /// Sets [`Checkpoint::versions_seen`].
    pub fn with_versions_seen(
        mut self,
        versions_seen: BTreeMap<String, BTreeMap<String, u64>>,
    ) -> Self {
        self.versions_seen = versions_seen;
        self
    }

    /// Sets [`Checkpoint::channel_deltas`].
    pub fn with_channel_deltas(
        mut self,
        channel_deltas: BTreeMap<String, Vec<serde_json::Value>>,
    ) -> Self {
        self.channel_deltas = channel_deltas;
        self
    }

    /// The effective pending-task set: [`Checkpoint::tasks`] directly on a
    /// v2 record (`version >= 2`), or derived from the v1 fields
    /// (preferring `pending_activations`, falling back to `next_nodes`) on a
    /// v1 record. Non-mutating — shared by [`Checkpoint::normalize`] (which
    /// writes the result back) and [`Checkpoint::to_metadata`] (which only
    /// needs to read it).
    fn effective_tasks(&self) -> Vec<PendingActivation> {
        if self.version >= CHECKPOINT_FORMAT_VERSION {
            return self.tasks.clone();
        }
        match &self.pending_activations {
            Some(pending) if !pending.is_empty() => pending.clone(),
            _ => self
                .next_nodes
                .iter()
                .cloned()
                .map(|node| PendingActivation {
                    node,
                    send_arg: None,
                    task_id: empty_task_id(),
                })
                .collect(),
        }
    }

    /// The effective completed-task set: [`Checkpoint::completed`] directly
    /// on a v2 record, or zipped from the v1 `completed_tasks`/
    /// `completed_routes` pair (padding a shorter/missing `completed_routes`
    /// with empty routing — the pre-`completed_routes` behavior) on a v1
    /// record. Non-mutating, mirroring [`Checkpoint::effective_tasks`].
    fn effective_completed(&self) -> Vec<CompletedTask> {
        if self.version >= CHECKPOINT_FORMAT_VERSION {
            return self.completed.clone();
        }
        self.completed_tasks
            .iter()
            .cloned()
            .zip(
                self.completed_routes
                    .iter()
                    .cloned()
                    .chain(std::iter::repeat(Vec::new())),
            )
            .map(|(node, routes)| CompletedTask {
                task_id: empty_task_id(),
                node,
                routes,
            })
            .collect()
    }

    /// Folds a checkpoint format v1 record into the current (v2) shape,
    /// in place: populates [`Checkpoint::tasks`]/[`Checkpoint::completed`]
    /// from whichever legacy fields the record carries (see
    /// [`Checkpoint::effective_tasks`]/[`Checkpoint::effective_completed`]),
    /// clears the legacy fields (so a subsequent `put` of the same value
    /// re-serializes as clean v2), and stamps [`Checkpoint::version`] to
    /// [`CHECKPOINT_FORMAT_VERSION`].
    ///
    /// A no-op on an already-v2 record. Every bundled [`Checkpointer`]
    /// backend calls this on every decode path (`get`/`get_scoped`/`list`/
    /// `state_history`/`get_thread`), so callers outside this module never
    /// observe a v1 record — see `docs/modules/graph/checkpointing.md`.
    pub fn normalize(&mut self) {
        if self.version >= CHECKPOINT_FORMAT_VERSION {
            return;
        }
        self.tasks = self.effective_tasks();
        self.completed = self.effective_completed();
        self.next_nodes = Vec::new();
        self.completed_tasks = Vec::new();
        self.completed_routes = Vec::new();
        self.pending_activations = None;
        self.version = CHECKPOINT_FORMAT_VERSION;
    }

    /// Builds the lightweight [`CheckpointMetadata`] summary for this checkpoint.
    ///
    /// The single source of truth for projecting a stored checkpoint onto its
    /// listing record: it parses the `source`/`step` out of the free-form
    /// `metadata` (falling back to [`CheckpointSource::Loop`]/`0`), projects
    /// [`Checkpoint::effective_tasks`] onto its node ids for
    /// [`CheckpointMetadata::next_nodes`], and copies the lineage fields. Both
    /// `Checkpointer::list` and the state-inspection API
    /// (`get_state`/`get_state_history`) use it so a snapshot's metadata always
    /// matches what listing reports. Correct on an un-normalized v1 record too
    /// (it never mutates `self`), which is what lets a header-only listing
    /// path (no full-record decode) project it without first normalizing.
    pub fn to_metadata(&self) -> CheckpointMetadata {
        let source = self
            .metadata
            .get("source")
            .and_then(|v| v.as_str())
            .and_then(CheckpointSource::parse)
            .unwrap_or(CheckpointSource::Loop);
        let step = self
            .metadata
            .get("step")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let next_nodes = self.effective_tasks().into_iter().map(|t| t.node).collect();
        CheckpointMetadata {
            thread_id: self.thread_id.clone(),
            checkpoint_id: self.checkpoint_id.clone(),
            run_id: self.run_id.clone(),
            parent_checkpoint_id: self.parent_checkpoint_id.clone(),
            namespace: self.namespace.clone(),
            next_nodes,
            has_interrupts: !self.interrupts.is_empty(),
            source,
            step,
        }
    }
}

/// The `idx` reserved for a task's **resume** control-plane write.
///
/// LangGraph reserves *negative* indices for control-plane channels
/// (`WRITES_IDX_MAP`), which is what distinguishes an upsert from an append:
/// see [`PendingWrite::is_control_plane`].
pub const WRITES_IDX_RESUME: i64 = -1;

/// The `idx` reserved for a task's **error** control-plane write.
pub const WRITES_IDX_ERROR: i64 = -2;

/// The `idx` reserved for a task's **interrupt** control-plane write.
pub const WRITES_IDX_INTERRUPT: i64 = -3;

/// The `idx` reserved for a task's **deferred result** control-plane write:
/// the `Update`/`Command::goto` a node produced but that an
/// [`interrupt_after`](crate::GraphBuilder::interrupt_after) pause held back
/// from committed state. At most one per task (a task completes once), and
/// re-put on a repeated pause replaces it — the control-plane upsert rule.
/// Replayed by the executor on resume instead of re-running the handler.
pub const WRITES_IDX_INTERRUPT_AFTER: i64 = -4;

/// Channel prefix for a [`crate::NodeContext::durable_task`] memo write:
/// the full channel is this prefix followed by the caller's `key`.
pub const DURABLE_TASK_CHANNEL_PREFIX: &str = "__durable_task__:";

/// Channel of a task's deferred `interrupt_after` result write (see
/// [`WRITES_IDX_INTERRUPT_AFTER`]).
pub const INTERRUPT_AFTER_CHANNEL: &str = "__interrupt_after__";

/// A partial write produced by a completed task, preserved across reruns.
///
/// # Why writes are recorded separately from the checkpoint
///
/// A superstep can fail *after* some of its tasks have already run. The
/// boundary checkpoint records that those tasks completed, but without a record
/// of what they wrote, a resume has no way to tell "this task already ran" from
/// "this task has not run yet" — so it re-runs them, and any side effect they
/// performed happens twice. Writes are therefore persisted per task through
/// [`Checkpointer::put_writes`](crate::Checkpointer::put_writes) and read
/// back into [`CheckpointTuple::pending_writes`], which is what resume consults
/// to skip already-completed work.
///
/// # Identity
///
/// A write is addressed by `(thread_id, namespace, checkpoint_id, task_id,
/// idx)`, mirroring the primary key LangGraph's SQL checkpointers use. Within
/// one checkpoint the `(task_id, idx)` pair is unique: re-putting the same pair
/// never produces a second row.
///
/// # Control-plane writes upsert; data writes are append-once
///
/// `idx >= 0` is an ordinary data write, emitted once per task in emission
/// order. Re-putting it is **ignored** (insert-or-ignore), so a retried
/// `put_writes` is idempotent.
///
/// `idx < 0` marks a control-plane write — resume values, errors, interrupts —
/// which by construction there is at most one of per task and whose value
/// legitimately changes on a retry. Re-putting it **replaces** the stored value
/// (insert-or-replace). Use the `WRITES_IDX_*` constants rather than raw
/// negative numbers.
///
/// # Back-compatibility
///
/// `task_id`, `idx` and `channel` carry `#[serde(default)]`, so checkpoint
/// records written before the write protocol existed still deserialize (as an
/// anonymous data write at index `0`).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PendingWrite {
    /// The node that produced the write.
    pub node: NodeId,
    /// The task that produced the write, unique within a superstep.
    ///
    /// A plain node id is not enough on its own: a fan-out step runs the same
    /// node several times with different [`Send`](crate::Send) args, and
    /// each of those is a separately resumable task.
    #[serde(default = "empty_task_id")]
    pub task_id: TaskId,
    /// Position of this write within its task's emission order, or one of the
    /// `WRITES_IDX_*` constants for a control-plane write.
    #[serde(default)]
    pub idx: i64,
    /// The channel (state field / node output slot) the write targets.
    ///
    /// Free-form and backend-opaque; it exists so writes stay distinguishable
    /// per channel, and because write isolation is asserted per channel *and*
    /// namespace in the conformance suite.
    #[serde(default)]
    pub channel: String,
    /// The serialized write payload.
    ///
    /// May be [`serde_json::Value::Null`] when the producing runtime cannot
    /// serialize its update type. The graph executor is in exactly that
    /// position — a graph's `Update` carries no `Serialize` bound — so it
    /// records writes as *completion markers*: the applied value is already
    /// durable in the checkpoint's `state`, and the write record's job is to
    /// answer "did this task already run?".
    pub payload: serde_json::Value,
}

impl PendingWrite {
    /// Builds an ordinary data write for `task_id` at position `idx`.
    pub fn data(
        node: impl Into<NodeId>,
        task_id: impl Into<TaskId>,
        idx: i64,
        channel: impl Into<String>,
        payload: serde_json::Value,
    ) -> Self {
        Self {
            node: node.into(),
            task_id: task_id.into(),
            idx,
            channel: channel.into(),
            payload,
        }
    }

    /// Builds a completion marker: a data write at index `0` whose payload is
    /// `null`, recording only that `task_id` ran to completion.
    pub fn completion_marker(node: impl Into<NodeId>, task_id: impl Into<TaskId>) -> Self {
        let node = node.into();
        let channel = node.as_str().to_string();
        Self {
            node,
            task_id: task_id.into(),
            idx: 0,
            channel,
            payload: serde_json::Value::Null,
        }
    }

    /// Builds a [`crate::NodeContext::durable_task`] memo write: the
    /// serialized output of the task's durable sub-step `key`, stored as an
    /// ordinary data write (`idx >= 1`, append-once) on the
    /// [`DURABLE_TASK_CHANNEL_PREFIX`]`key` channel. `idx` must be unique
    /// among the task's writes (the executor allocates it past every write
    /// the task already holds); `0` is reserved for the completion marker.
    pub fn durable_task(
        node: impl Into<NodeId>,
        task_id: impl Into<TaskId>,
        idx: i64,
        key: &str,
        payload: serde_json::Value,
    ) -> Self {
        debug_assert!(idx >= 1, "durable-task writes use idx >= 1");
        Self {
            node: node.into(),
            task_id: task_id.into(),
            idx,
            channel: format!("{DURABLE_TASK_CHANNEL_PREFIX}{key}"),
            payload,
        }
    }

    /// Builds a task's deferred `interrupt_after` result write (see
    /// [`WRITES_IDX_INTERRUPT_AFTER`]): `payload` is
    /// `{"update": <encoded update or null>, "goto": [<RouteTarget>...]}`.
    pub fn interrupt_after(
        node: impl Into<NodeId>,
        task_id: impl Into<TaskId>,
        payload: serde_json::Value,
    ) -> Self {
        Self {
            node: node.into(),
            task_id: task_id.into(),
            idx: WRITES_IDX_INTERRUPT_AFTER,
            channel: INTERRUPT_AFTER_CHANNEL.to_string(),
            payload,
        }
    }

    /// Whether this is a control-plane write (`idx < 0`), which upserts rather
    /// than appends. See the type docs.
    pub fn is_control_plane(&self) -> bool {
        self.idx < 0
    }

    /// Whether this is a [`crate::NodeContext::durable_task`] memo write
    /// (see [`Self::durable_task`]).
    pub fn is_durable_task(&self) -> bool {
        self.channel.starts_with(DURABLE_TASK_CHANNEL_PREFIX)
    }

    /// The caller's `key` of a durable-task memo write, or `None` for any
    /// other write.
    pub fn durable_task_key(&self) -> Option<&str> {
        self.channel.strip_prefix(DURABLE_TASK_CHANNEL_PREFIX)
    }

    /// Whether this is a task's deferred `interrupt_after` result write (see
    /// [`Self::interrupt_after`]).
    pub fn is_interrupt_after(&self) -> bool {
        self.idx == WRITES_IDX_INTERRUPT_AFTER && self.channel == INTERRUPT_AFTER_CHANNEL
    }

    /// Whether this write is a per-task *replay memo* — a durable-task memo
    /// or a deferred `interrupt_after` result — that a re-run of the same
    /// (still pending) task consumes, as opposed to a completion marker or
    /// any other write that records the task as already done.
    ///
    /// Resume keys "which pending tasks already ran" off the writes that are
    /// *not* replay memos: a replay memo belongs to a task that has *not*
    /// completed yet (that is the whole point of memoising it), so counting
    /// it as a completion marker would wrongly skip the task.
    pub fn is_task_replay(&self) -> bool {
        self.is_durable_task() || self.is_interrupt_after()
    }

    /// The `(task_id, idx)` identity pair this write is deduplicated on within
    /// a checkpoint.
    pub fn identity(&self) -> (&str, i64) {
        (self.task_id.as_str(), self.idx)
    }
}

/// Merges `incoming` into `existing`, applying the replace-vs-ignore rule.
///
/// Shared by every backend so the three of them cannot drift on the one part of
/// the write protocol that is easy to get subtly wrong:
///
/// - an incoming **control-plane** write (`idx < 0`) replaces any stored write
///   with the same `(task_id, idx)`;
/// - an incoming **data** write (`idx >= 0`) is ignored when that pair is
///   already stored.
///
/// Returns the number of entries that were actually inserted or replaced, which
/// backends use for logging.
pub fn merge_writes(existing: &mut Vec<PendingWrite>, incoming: &[PendingWrite]) -> usize {
    let mut changed = 0;
    for write in incoming {
        match existing
            .iter_mut()
            .find(|w| w.identity() == write.identity())
        {
            Some(slot) => {
                if write.is_control_plane() {
                    *slot = write.clone();
                    changed += 1;
                }
                // Data writes are append-once: a duplicate is a no-op.
            }
            None => {
                existing.push(write.clone());
                changed += 1;
            }
        }
    }
    changed
}

/// Lightweight checkpoint summary returned by `Checkpointer::list`.
///
/// Listing must not require deserializing full graph state, so metadata is kept
/// separate from the [`Checkpoint`] state payload.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CheckpointMetadata {
    /// Thread lineage key.
    pub thread_id: String,
    /// Checkpoint id.
    pub checkpoint_id: String,
    /// The run that produced this checkpoint, when known.
    pub run_id: Option<String>,
    /// Parent checkpoint id.
    pub parent_checkpoint_id: Option<String>,
    /// Namespace scoping.
    pub namespace: Vec<String>,
    /// Nodes to run on resume.
    pub next_nodes: Vec<NodeId>,
    /// Whether the checkpoint carries pending interrupts.
    pub has_interrupts: bool,
    /// Checkpoint source: `input`, `loop`, `update`, or `fork`.
    pub source: CheckpointSource,
    /// The superstep number that produced the checkpoint.
    pub step: usize,
}
