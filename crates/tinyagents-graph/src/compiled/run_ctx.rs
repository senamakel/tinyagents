//! Per-run execution context threaded through the superstep loop.
//!
//! [`RunCtx`] bundles run identity (ids, namespace), clocks/deadlines,
//! recursion bookkeeping, the accumulators a superstep loop carries forward
//! (`node_visits`, `barrier_arrivals`, `visited`, `all_child_runs`,
//! `steps`/checkpoint lineage), and the handles for background checkpoint
//! writes and status/event I/O.
//!
//! It exists so the step-running, boundary, and resume helpers split out of
//! `executor.rs` stop threading a dozen positional parameters between them:
//! every one of those helpers takes `&RunCtx`/`&mut RunCtx` plus the handful
//! of values that are genuinely local to that call (the active set, the
//! state snapshot, a step's folded outcome). `RunCtx` is created once per
//! `execute_run` call and never outlives it — it borrows the owning
//! [`CompiledGraph`] for that duration.

use super::*;

/// Run-scoped state for one `execute_run` call.
///
/// Fields fall into three groups: identity that never changes for the run
/// (`run_id`, `thread_id`, `root_run_id`, `parent_run_id`, `started_at`,
/// `live_frames`, `recursion_meta`, `binding`), accumulators the superstep
/// loop updates every iteration (`recursion`, `node_visits`,
/// `barrier_arrivals`, `resume_map`, `visited`, `all_child_runs`, `steps`,
/// `last_checkpoint`, `parent_checkpoint`), and I/O handles
/// (`child_sink`, `async_writes`). `graph` is the owning [`CompiledGraph`],
/// kept here so the convenience methods below (`emit`, `save_status`,
/// `base_status`, `node_context`) don't need a separate receiver.
pub(super) struct RunCtx<'a, State, Update> {
    pub(super) graph: &'a CompiledGraph<State, Update>,
    pub(super) run_id: RunId,
    pub(super) thread_id: Option<ThreadId>,
    pub(super) root_run_id: RunId,
    pub(super) parent_run_id: Option<RunId>,
    pub(super) started_at: SystemTime,
    pub(super) live_frames: Vec<RecursionFrame>,
    pub(super) recursion_meta: serde_json::Value,
    pub(super) recursion: RecursionStack,
    pub(super) binding: Option<crate::subagent_node::AgentInvocationBinding>,
    pub(super) child_sink: ChildRunSink,
    pub(super) node_visits: HashMap<NodeId, usize>,
    pub(super) barrier_arrivals: HashMap<NodeId, HashSet<NodeId>>,
    pub(super) async_writes: AsyncCheckpointWrites,
    /// Keyed by task id, falling back to node id (I1/R5); see
    /// [`super::executor::RunSeed::resume_map`].
    pub(super) resume_map: HashMap<String, serde_json::Value>,
    pub(super) visited: Vec<NodeId>,
    pub(super) all_child_runs: Vec<ChildRun>,
    pub(super) steps: usize,
    pub(super) last_checkpoint: Option<CheckpointId>,
    pub(super) parent_checkpoint: Option<String>,
    /// Nodes (with their persisted explicit `Command::goto`, R1) carried
    /// forward from a resumed mid-step checkpoint (an interrupt/failure
    /// boundary whose completed siblings were never routed) — see
    /// [`super::boundary::CompiledGraph::advance`]'s doc. `None` for a fresh
    /// run or a resume from a fully-routed (normal) boundary. Consumed
    /// (`take`n) by the first `advance` call of this run;
    /// [`super::boundary`]'s failure/interrupt boundaries read it (without
    /// consuming it) to keep carrying it forward across a step that
    /// interrupts or fails more than once in a row.
    pub(super) carried_completed: Option<Vec<(NodeId, Vec<RouteTarget>)>>,
}

/// Everything a resumed run seeds `RunCtx` with beyond a fresh run's
/// defaults, bundled into one optional parameter so [`RunCtx::start`] does
/// not grow a positional argument per resume-only field.
///
/// A fresh run (`resume_from_inner` was never called) passes `None`, which
/// is equivalent to `ResumeSeed::default()`.
#[derive(Default)]
pub(super) struct ResumeSeed {
    /// The loaded checkpoint's own step number (`to_metadata().step`), so
    /// this run's `ctx.steps` continues counting up from it instead of
    /// restarting at `0` — see the I3 finding in
    /// `docs/runtime-comparison/code-review-graph.md`: without this,
    /// `metadata.step` (and so `get_state_history`) goes non-monotonic
    /// across a resume, and per-node visit caps
    /// (`RecursionPolicy::max_visits_per_node`) reset every resume rather
    /// than bounding the whole thread's lifetime.
    pub(super) initial_steps: usize,
    /// The loaded checkpoint's persisted `node_visits` metadata (see
    /// [`super::boundary`]'s checkpoint builders), so per-node visit counts
    /// accumulate across a resume instead of resetting.
    pub(super) initial_node_visits: HashMap<NodeId, usize>,
    /// Nodes (with their persisted goto, R1) carried forward from a
    /// mid-step (interrupt/failure) checkpoint whose completed siblings
    /// were never routed — see [`RunCtx::carried_completed`].
    pub(super) carried_completed: Option<Vec<(NodeId, Vec<RouteTarget>)>>,
}

impl<'a, State, Update> RunCtx<'a, State, Update>
where
    State: Clone + Send + Sync + 'static,
    Update: Send + 'static,
{
    /// Forwards to the owning graph's event sink (a no-op without one).
    pub(super) fn emit(&self, event: GraphEvent) {
        self.graph.emit(event);
    }

    /// Forwards to the owning graph's status store (a no-op without one).
    pub(super) async fn save_status(&self, status: GraphRunStatus) {
        self.graph.save_status(status).await;
    }

    /// Builds a fresh [`GraphRunStatus`] for this run at `Running` status,
    /// stamped with this context's identity and start time.
    pub(super) fn base_status(&self) -> GraphRunStatus {
        self.graph
            .base_status(&self.run_id, &self.thread_id, self.started_at)
    }

    /// Builds this run's `RunCtx`: constructs the recursion stack from the
    /// inherited parent frames and pushes the frame for this graph call (a
    /// push that would exceed `max_depth` fails the run — emitting
    /// `RunStarted` and a terminal `Failed` status — before any node
    /// executes), then emits `RunStarted`/`RecursionDepthChanged` for a
    /// successful push.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn start(
        graph: &'a CompiledGraph<State, Update>,
        run_id: RunId,
        thread_id: Option<ThreadId>,
        resume_map: HashMap<NodeId, serde_json::Value>,
        initial_barriers: HashMap<NodeId, HashSet<NodeId>>,
        initial_parent: Option<String>,
        binding: Option<crate::subagent_node::AgentInvocationBinding>,
        resume_seed: ResumeSeed,
    ) -> Result<Self> {
        let ResumeSeed {
            initial_steps,
            initial_node_visits,
            carried_completed,
        } = resume_seed;
        let started_at = SystemTime::now();
        // Graph-call depth (the stack) is tracked separately from node-loop
        // visits (`node_visits`, below).
        let mut recursion =
            RecursionStack::with_frames(graph.recursion_frames.clone(), graph.recursion_policy);
        let root_run_id = graph
            .recursion_frames
            .first()
            .map(|f| f.run_id.clone())
            .unwrap_or_else(|| run_id.clone());
        let parent_run_id = graph.recursion_frames.last().map(|f| f.run_id.clone());
        let this_frame = RecursionFrame {
            graph_id: graph.graph_id.clone(),
            node_id: graph.recursion_node.clone(),
            run_id: run_id.clone(),
            task_id: None,
            namespace: graph.namespace.clone(),
            depth: recursion.depth(),
            parent: parent_run_id.clone(),
        };
        if let Err(err) = recursion.push(this_frame) {
            graph.emit(GraphEvent::RunStarted {
                run_id: run_id.clone(),
            });
            graph
                .fail_run(&run_id, &thread_id, started_at, 0, &err, None)
                .await;
            return Err(err);
        }
        // Serialized once per run for embedding in every checkpoint's metadata.
        let recursion_meta =
            serde_json::to_value(recursion.frames()).unwrap_or(serde_json::Value::Null);
        let live_frames = recursion.frames().to_vec();

        let ctx = Self {
            graph,
            run_id,
            thread_id,
            root_run_id,
            parent_run_id,
            started_at,
            live_frames,
            recursion_meta,
            recursion,
            binding,
            child_sink: ChildRunSink::new(),
            node_visits: initial_node_visits,
            barrier_arrivals: initial_barriers,
            async_writes: AsyncCheckpointWrites::default(),
            resume_map,
            visited: Vec::new(),
            all_child_runs: Vec::new(),
            steps: initial_steps,
            last_checkpoint: None,
            parent_checkpoint: initial_parent,
            carried_completed,
        };
        ctx.emit(GraphEvent::RunStarted {
            run_id: ctx.run_id.clone(),
        });
        // Surface this run's recursion depth so observers can attribute
        // nested runs without reconstructing the tree from logs.
        ctx.emit(GraphEvent::RecursionDepthChanged {
            depth: ctx.recursion.depth(),
        });
        Ok(ctx)
    }

    /// Drains this step's child-run sink into `all_child_runs` and returns
    /// its serialized form for embedding into this boundary's checkpoint
    /// metadata.
    pub(super) fn take_step_child_runs(&mut self) -> serde_json::Value {
        let step_child_runs = self.child_sink.drain();
        self.all_child_runs.extend(step_child_runs.iter().cloned());
        serde_json::to_value(&step_child_runs).unwrap_or(serde_json::Value::Null)
    }

    /// Builds the per-task [`NodeContext`] for `node_id`, consuming its entry
    /// from `resume_map` (a node can only be handed its resume value once).
    ///
    /// `fork` carries the branch identity in a concurrent step (`None` in
    /// sequential mode or single-node steps).
    pub(super) fn node_context(
        &mut self,
        node_id: &NodeId,
        step: usize,
        fork: Option<ForkId>,
        send_arg: Option<serde_json::Value>,
    ) -> NodeContext {
        NodeContext {
            graph_id: self.graph.graph_id.clone(),
            node_id: node_id.clone(),
            run_id: self.run_id.clone(),
            thread_id: self.thread_id.clone(),
            step,
            resume: self.resume_map.remove(node_id),
            fork,
            send_arg,
            root_run_id: Some(self.root_run_id.clone()),
            recursion_frames: self.live_frames.clone(),
            child_runs: Some(self.child_sink.clone()),
            agent_binding: self.binding.clone(),
        }
    }
}
