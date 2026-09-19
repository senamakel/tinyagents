//! Low-level graph events and high-level stream modes — the wire vocabulary the
//! recursive executor uses to narrate its own execution.
//!
//! [`GraphEvent`] is the fine-grained, per-node/per-step lifecycle signal the
//! durable executor emits at every boundary; [`StreamMode`] is the LangGraph-
//! style selection of *which projection* of that stream a caller wants (full
//! values, per-node updates, model messages, debug detail, interrupts, or
//! custom node writes). Together they let observers — including a parent run
//! consuming a subgraph — follow a run without inspecting its internal state.

use serde::{Deserialize, Serialize};

use crate::command::Interrupt;
use tinyagents_harness::ids::{CheckpointId, NodeId, RunId, TaskId};

/// A low-level graph lifecycle event emitted through a [`super::GraphEventSink`].
///
/// These are the durable-executor analogues of the observability event model in
/// the graph spec, reduced to the set the milestone executor actually emits.
///
/// The variants are serde-serializable so a single event can be wrapped into a
/// durable [`crate::observability::GraphObservation`] envelope, journaled,
/// and replayed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum GraphEvent {
    /// The run began (emitted once before the first superstep).
    RunStarted {
        /// The run that started.
        run_id: RunId,
    },
    /// The run finished successfully.
    RunCompleted {
        /// The run that completed.
        run_id: RunId,
        /// Total supersteps executed.
        steps: usize,
    },
    /// The run aborted with an error.
    RunFailed {
        /// The run that failed.
        run_id: RunId,
        /// Rendered error.
        error: String,
    },
    /// The run was cooperatively cancelled via a [`tinyagents_harness::CancellationToken`]
    /// (I4 part 2), either between supersteps or while a superstep's node
    /// handlers were still in flight.
    RunCancelled {
        /// The run that was cancelled.
        run_id: RunId,
    },
    /// A superstep started with the given active node set.
    StepStarted {
        /// 1-based step number.
        step: usize,
        /// Nodes scheduled to run this step.
        active: Vec<NodeId>,
    },
    /// A superstep finished and its boundary work (reducer, checkpoint) ran.
    StepCompleted {
        /// 1-based step number.
        step: usize,
    },
    /// A task was scheduled for a node in the active set.
    TaskScheduled {
        /// Target node.
        node: NodeId,
        /// Step number.
        step: usize,
    },
    /// A task began executing (the [`StreamMode::Tasks`] counterpart of
    /// [`GraphEvent::NodeStarted`], emitted alongside it at the same
    /// boundary).
    TaskStarted {
        /// Target node.
        node: NodeId,
        /// Step number.
        step: usize,
    },
    /// A task finished, successfully or not (the [`StreamMode::Tasks`]
    /// counterpart of [`GraphEvent::NodeCompleted`]/[`GraphEvent::NodeFailed`],
    /// emitted alongside them at the same boundary).
    TaskCompleted {
        /// Target node.
        node: NodeId,
        /// Step number.
        step: usize,
        /// Whether this result was served from a task cache rather than
        /// executed. Always `false` today — per-node task caching
        /// (`docs/runtime-comparison/feature-gaps.md` D2) is not yet
        /// implemented; the field exists so [`StreamMode::Tasks`] consumers
        /// do not need a breaking change once it lands.
        cached: bool,
    },
    /// A node handler began executing.
    NodeStarted {
        /// Node id.
        node: NodeId,
        /// Step number.
        step: usize,
    },
    /// A node handler completed successfully.
    NodeCompleted {
        /// Node id.
        node: NodeId,
        /// Step number.
        step: usize,
    },
    /// A node handler returned an error.
    NodeFailed {
        /// Node id.
        node: NodeId,
        /// Step number.
        step: usize,
        /// Rendered error.
        error: String,
    },
    /// A node handler failed with a retryable error and a retry was scheduled
    /// under the graph's node-retry policy. Emitted before the (opt-in) backoff
    /// wait and the re-run of the node from its start.
    NodeRetryScheduled {
        /// Node id.
        node: NodeId,
        /// Step number.
        step: usize,
        /// The 1-based retry attempt about to be made.
        attempt: usize,
    },
    /// A node produced a state update applied at the boundary.
    StateUpdated {
        /// Node id.
        node: NodeId,
        /// Step number.
        step: usize,
    },
    /// A route was selected for a node.
    RouteSelected {
        /// Source node.
        node: NodeId,
        /// Selected next node.
        target: NodeId,
    },
    /// A checkpoint was persisted at a superstep boundary.
    CheckpointSaved {
        /// Persisted checkpoint id.
        checkpoint_id: CheckpointId,
        /// The superstep this checkpoint was saved at, when the save site
        /// knows it (`None` for saves outside the ordinary superstep boundary,
        /// such as a resume-time bootstrap checkpoint).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        step: Option<usize>,
    },
    /// A checkpoint was loaded to resume/replay a run (a read, not a write).
    CheckpointRestored {
        /// The checkpoint id that was loaded.
        checkpoint_id: CheckpointId,
    },
    /// A node emitted an interrupt and the run paused.
    InterruptEmitted {
        /// The emitted interrupt.
        interrupt: Interrupt,
    },
    /// An embedded subgraph began executing under a child namespace.
    SubgraphStarted {
        /// The parent node hosting the subgraph.
        node: NodeId,
        /// The child checkpoint namespace.
        namespace: Vec<String>,
    },
    /// An embedded subgraph finished executing.
    SubgraphCompleted {
        /// The parent node hosting the subgraph.
        node: NodeId,
        /// The child checkpoint namespace.
        namespace: Vec<String>,
    },
    /// A parallel superstep forked an execution branch for a node.
    ContextForked {
        /// The node whose branch was forked.
        node: NodeId,
        /// The branch (fork) index within the active set.
        fork: usize,
        /// Step number.
        step: usize,
    },
    /// The effective recursion/namespace depth changed.
    RecursionDepthChanged {
        /// The new depth (number of enclosing namespaces).
        depth: usize,
    },
    /// An arbitrary user-defined event written from inside a node.
    Custom {
        /// A stable name for the custom event.
        name: String,
        /// Free-form structured payload.
        data: serde_json::Value,
    },
}

impl GraphEvent {
    /// Returns a stable, dot-separated string that names the kind of event.
    ///
    /// The returned string is a static literal, suitable for logging,
    /// filtering, and serde-independent routing. Examples: `"run.started"`,
    /// `"step.started"`, `"node.completed"`.
    pub fn kind(&self) -> &'static str {
        match self {
            GraphEvent::RunStarted { .. } => "run.started",
            GraphEvent::RunCompleted { .. } => "run.completed",
            GraphEvent::RunFailed { .. } => "run.failed",
            GraphEvent::RunCancelled { .. } => "run.cancelled",
            GraphEvent::StepStarted { .. } => "step.started",
            GraphEvent::StepCompleted { .. } => "step.completed",
            GraphEvent::TaskScheduled { .. } => "task.scheduled",
            GraphEvent::TaskStarted { .. } => "task.started",
            GraphEvent::TaskCompleted { .. } => "task.completed",
            GraphEvent::NodeStarted { .. } => "node.started",
            GraphEvent::NodeCompleted { .. } => "node.completed",
            GraphEvent::NodeFailed { .. } => "node.failed",
            GraphEvent::NodeRetryScheduled { .. } => "node.retry_scheduled",
            GraphEvent::StateUpdated { .. } => "state.updated",
            GraphEvent::RouteSelected { .. } => "route.selected",
            GraphEvent::CheckpointSaved { .. } => "checkpoint.saved",
            GraphEvent::CheckpointRestored { .. } => "checkpoint.restored",
            GraphEvent::InterruptEmitted { .. } => "interrupt.emitted",
            GraphEvent::SubgraphStarted { .. } => "subgraph.started",
            GraphEvent::SubgraphCompleted { .. } => "subgraph.completed",
            GraphEvent::ContextForked { .. } => "context.forked",
            GraphEvent::RecursionDepthChanged { .. } => "recursion.depth_changed",
            GraphEvent::Custom { .. } => "custom",
        }
    }

    /// Returns the superstep number this event is associated with, when the
    /// variant carries one. Used to stamp the `step` field of a durable
    /// [`crate::observability::GraphObservation`].
    pub fn step(&self) -> Option<usize> {
        match self {
            GraphEvent::StepStarted { step, .. }
            | GraphEvent::StepCompleted { step }
            | GraphEvent::TaskScheduled { step, .. }
            | GraphEvent::TaskStarted { step, .. }
            | GraphEvent::TaskCompleted { step, .. }
            | GraphEvent::NodeStarted { step, .. }
            | GraphEvent::NodeCompleted { step, .. }
            | GraphEvent::NodeFailed { step, .. }
            | GraphEvent::NodeRetryScheduled { step, .. }
            | GraphEvent::StateUpdated { step, .. }
            | GraphEvent::ContextForked { step, .. } => Some(*step),
            GraphEvent::RunCompleted { steps, .. } => Some(*steps),
            _ => None,
        }
    }
}

/// High-level projection modes for a graph run stream.
///
/// These mirror the LangGraph stream modes. [`GraphEvent::mode`] maps every
/// event kind onto one of these (or `None` for the lifecycle events every
/// mode should still see); [`super::project::project_graph_event`] applies
/// that mapping to filter a raw [`GraphEventEnvelope`] stream the way
/// [`tinyagents_harness::stream::project_event_for_modes`] does for
/// [`tinyagents_harness::events::AgentEvent`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StreamMode {
    /// Full state values after each step.
    Values,
    /// Per-node/per-task state updates.
    Updates,
    /// Harness message or token deltas from model nodes.
    Messages,
    /// Checkpoints plus task internals.
    Debug,
    /// Pending interrupts only.
    Interrupts,
    /// Arbitrary user stream writes from inside nodes.
    Custom,
    /// Task-level lifecycle: [`GraphEvent::TaskScheduled`],
    /// [`GraphEvent::TaskStarted`], and [`GraphEvent::TaskCompleted`] — the
    /// LangGraph `"tasks"` mode, narrower than [`StreamMode::Debug`] (no
    /// step/checkpoint/routing internals, just task start/end).
    Tasks,
    /// Checkpoint lifecycle only: [`GraphEvent::CheckpointSaved`] and
    /// [`GraphEvent::CheckpointRestored`] — the LangGraph `"checkpoints"`
    /// mode.
    Checkpoints,
}

impl GraphEvent {
    /// Returns the [`StreamMode`] this event projects onto, when it belongs
    /// to a narrower mode than [`StreamMode::Debug`] (which every event kind
    /// still counts toward — see
    /// [`super::project::project_graph_event`]).
    ///
    /// Run/step lifecycle events (`RunStarted`, `StepStarted`, …) have no
    /// narrower home and return `None`: they surface only under
    /// [`StreamMode::Debug`].
    pub fn mode(&self) -> Option<StreamMode> {
        match self {
            GraphEvent::TaskScheduled { .. }
            | GraphEvent::TaskStarted { .. }
            | GraphEvent::TaskCompleted { .. }
            | GraphEvent::NodeStarted { .. }
            | GraphEvent::NodeCompleted { .. }
            | GraphEvent::NodeFailed { .. }
            | GraphEvent::NodeRetryScheduled { .. } => Some(StreamMode::Tasks),
            GraphEvent::StateUpdated { .. } => Some(StreamMode::Updates),
            GraphEvent::CheckpointSaved { .. } | GraphEvent::CheckpointRestored { .. } => {
                Some(StreamMode::Checkpoints)
            }
            GraphEvent::InterruptEmitted { .. } => Some(StreamMode::Interrupts),
            GraphEvent::Custom { .. } => Some(StreamMode::Custom),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// GraphEventEnvelope
// ---------------------------------------------------------------------------

/// A [`GraphEvent`] wrapped with the run/task correlation and ordering
/// metadata every emission site needs to be attributable in a merged,
/// multi-run stream.
///
/// `run_id` and `ns` (the checkpoint namespace) identify which run — and
/// which level of subgraph nesting within it — emitted the event, so a
/// parent run's observer can tell its own events apart from a nested
/// subgraph's. `seq` is a monotonic counter scoped to the emitting
/// [`crate::compiled::CompiledGraph`] instance (shared across a clone that
/// only changes `event_sink`, such as journal wrapping, but **not** shared
/// between a parent graph and a subgraph embedded as a node — the subgraph's
/// [`Self::ns`] already distinguishes its stream). `task_id` is `None` until
/// per-task correlation ids land end-to-end
/// (`docs/runtime-comparison/feature-gaps.md` D4); the field exists now so
/// adding that id later is additive.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphEventEnvelope {
    /// The run that emitted this event.
    pub run_id: RunId,
    /// Correlation id for the task this event belongs to, when task ids are
    /// wired end to end. `None` today.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<TaskId>,
    /// Checkpoint namespace of the emitting graph instance (empty for a
    /// top-level run; one segment deeper per level of subgraph nesting).
    pub ns: Vec<String>,
    /// Monotonically increasing sequence number, scoped as described on
    /// [`Self`].
    pub seq: u64,
    /// The wrapped event.
    pub event: GraphEvent,
}

impl GraphEventEnvelope {
    /// Wraps `event` in a minimal envelope (empty run id/namespace, `seq:
    /// 0`, no task id) for tests that only care about the event payload
    /// reaching a sink, not its attribution.
    #[cfg(test)]
    pub(crate) fn for_test(event: GraphEvent) -> Self {
        Self {
            run_id: RunId::from(String::new()),
            task_id: None,
            ns: Vec::new(),
            seq: 0,
            event,
        }
    }
}
