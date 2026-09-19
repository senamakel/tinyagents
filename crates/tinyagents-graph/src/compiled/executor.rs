//! Public run/resume entry points and the superstep execution engine.
//!
//! Split out of `compiled/mod.rs`; see that module's doc comment for the
//! full executor design (superstep loop, concurrency, and resumable-failure
//! semantics). The superstep loop itself is a thin wire-up over three
//! sibling modules: [`step`] runs a superstep's active node set and folds
//! its results ([`step::StepRunner`]), [`boundary`] applies the reducer,
//! routes, and persists a checkpoint at each of the three boundary shapes a
//! step can end at ([`boundary::StepBoundary`]), and [`resume`] loads a
//! checkpoint back into a fresh run. [`run_ctx::RunCtx`] carries the run
//! identity and bookkeeping all three share.

use super::*;

use crate::compiled::boundary::StepBoundary;
use crate::compiled::run_ctx::RunCtx;
use crate::compiled::step::StepRunner;

/// Everything a fresh or resumed run is seeded with, bundled so
/// [`CompiledGraph::execute`]/[`CompiledGraph::execute_run`] take one
/// parameter instead of positional state/thread/resume/barrier/binding
/// arguments.
struct RunSeed<State, Update> {
    state: State,
    active: Vec<Activation>,
    thread_id: Option<ThreadId>,
    resume_map: HashMap<NodeId, serde_json::Value>,
    barriers: HashMap<NodeId, HashSet<NodeId>>,
    parent: Option<String>,
    binding: Option<crate::subagent_node::AgentInvocationBinding>,
    _update: std::marker::PhantomData<Update>,
}

impl<State, Update> RunSeed<State, Update> {
    fn fresh(state: State, active: Vec<Activation>, thread_id: Option<ThreadId>) -> Self {
        Self {
            state,
            active,
            thread_id,
            resume_map: HashMap::new(),
            barriers: HashMap::new(),
            parent: None,
            binding: None,
            _update: std::marker::PhantomData,
        }
    }

    fn with_binding(mut self, binding: crate::subagent_node::AgentInvocationBinding) -> Self {
        self.binding = Some(binding);
        self
    }
}

impl<State, Update> CompiledGraph<State, Update>
where
    State: Clone + Send + Sync + 'static,
    Update: Send + 'static,
{
    /// Runs the graph to completion (or to an interrupt) without a thread.
    ///
    /// Without a thread id no checkpoints are persisted even if a checkpointer
    /// is configured, since checkpoints are keyed by thread.
    pub async fn run(&self, state: State) -> Result<GraphExecution<State>> {
        self.execute(
            state,
            vec![Activation::node(self.entry.clone())],
            None,
            HashMap::new(),
            HashMap::new(),
            None,
            None,
        )
        .await
    }

    /// Runs one execution with an explicit host-bound recursive-agent binding.
    ///
    /// The binding is scoped to this run and descendants spawned from it; it is
    /// never retained by this reusable graph value.
    pub async fn run_with_agent_binding(
        &self,
        state: State,
        binding: crate::subagent_node::AgentInvocationBinding,
    ) -> Result<GraphExecution<State>> {
        self.execute(
            state,
            vec![Activation::node(self.entry.clone())],
            None,
            HashMap::new(),
            HashMap::new(),
            None,
            Some(binding),
        )
        .await
    }

    /// Runs the graph with one or more external inputs in the first superstep.
    ///
    /// [`GraphInput::start`] targets the graph's compiled entry node, preserving
    /// the usual `START -> entry` contract for user input. Additional inputs may
    /// target any real node directly, so separate LLM/tool loops can be seeded
    /// together. Inputs are not deduplicated: two inputs aimed at the same node
    /// produce two separate activations, each with its own
    /// [`NodeContext::send_arg`](crate::NodeContext::send_arg).
    pub async fn run_with_inputs(
        &self,
        state: State,
        inputs: impl IntoIterator<Item = GraphInput>,
    ) -> Result<GraphExecution<State>> {
        let active = self.initial_inputs(inputs)?;
        self.execute(
            state,
            active,
            None,
            HashMap::new(),
            HashMap::new(),
            None,
            None,
        )
        .await
    }

    /// Runs the graph under a thread id, persisting checkpoints at every
    /// superstep boundary when a checkpointer is configured.
    pub async fn run_with_thread(
        &self,
        thread_id: impl Into<ThreadId>,
        state: State,
    ) -> Result<GraphExecution<State>> {
        self.execute(
            state,
            vec![Activation::node(self.entry.clone())],
            Some(thread_id.into()),
            HashMap::new(),
            HashMap::new(),
            None,
            None,
        )
        .await
    }

    /// Runs one threaded execution with an explicit recursive-agent binding.
    pub async fn run_with_thread_agent_binding(
        &self,
        thread_id: impl Into<ThreadId>,
        state: State,
        binding: crate::subagent_node::AgentInvocationBinding,
    ) -> Result<GraphExecution<State>> {
        self.execute(
            state,
            vec![Activation::node(self.entry.clone())],
            Some(thread_id.into()),
            HashMap::new(),
            HashMap::new(),
            None,
            Some(binding),
        )
        .await
    }

    /// Runs the graph under a thread id with one or more external inputs in the
    /// first superstep, persisting checkpoints at every boundary when a
    /// checkpointer is configured.
    pub async fn run_with_thread_inputs(
        &self,
        thread_id: impl Into<ThreadId>,
        state: State,
        inputs: impl IntoIterator<Item = GraphInput>,
    ) -> Result<GraphExecution<State>> {
        let active = self.initial_inputs(inputs)?;
        self.execute(
            state,
            active,
            Some(thread_id.into()),
            HashMap::new(),
            HashMap::new(),
            None,
            None,
        )
        .await
    }

    /// Resumes an interrupted run from its latest checkpoint, re-running the
    /// interrupted node(s) with the resume value supplied by `command`.
    ///
    /// Requires a checkpointer and an existing checkpoint for the thread;
    /// otherwise returns [`TinyAgentsError::Resume`].
    pub async fn resume(
        &self,
        thread_id: impl Into<ThreadId>,
        command: Command<Update>,
    ) -> Result<GraphExecution<State>> {
        self.resume_from(thread_id, ResumeTarget::Latest, command)
            .await
    }

    /// Resumes an interrupted run with a host-bound recursive-agent binding.
    ///
    /// Like [`Self::resume`], this reloads the latest checkpoint for `thread_id`.
    /// The binding is scoped solely to the resumed execution and is propagated
    /// to every resumed node and nested subgraph; it is never retained by this
    /// reusable graph value.
    pub async fn resume_with_agent_binding(
        &self,
        thread_id: impl Into<ThreadId>,
        command: Command<Update>,
        binding: crate::subagent_node::AgentInvocationBinding,
    ) -> Result<GraphExecution<State>> {
        self.resume_from_with_agent_binding(thread_id, ResumeTarget::Latest, command, binding)
            .await
    }

    /// Retries a failed run from its latest (failure-boundary) checkpoint,
    /// re-running the node that failed and the not-yet-run tail of that step.
    ///
    /// This is the resume counterpart for the *failure* path (as opposed to a
    /// human interrupt): after a node handler aborts a checkpointed run — a
    /// transient outage that outlived the node-retry policy, or a hard crash —
    /// the run leaves a resumable checkpoint (see
    /// [`CompiledGraph::with_node_retry`]). Calling `retry` re-runs exactly what
    /// did not complete, carrying no resume value. It is shorthand for
    /// [`CompiledGraph::resume`] with an empty [`Command`].
    ///
    /// To continue on *user feedback* instead of a bare retry, first inspect the
    /// committed state with
    /// [`get_state`](CompiledGraph::get_state), edit it with
    /// [`update_state`](CompiledGraph::update_state), then call `retry` (or
    /// `resume`) — the edited state is what the re-run sees.
    pub async fn retry(&self, thread_id: impl Into<ThreadId>) -> Result<GraphExecution<State>> {
        self.resume_from(thread_id, ResumeTarget::Latest, Command::new())
            .await
    }

    /// Retries a failed run with a host-bound recursive-agent binding.
    ///
    /// This is the binding-aware counterpart to [`Self::retry`]. The supplied
    /// binding is available only to this retry and any descendants it spawns.
    pub async fn retry_with_agent_binding(
        &self,
        thread_id: impl Into<ThreadId>,
        binding: crate::subagent_node::AgentInvocationBinding,
    ) -> Result<GraphExecution<State>> {
        self.resume_from_with_agent_binding(
            thread_id,
            ResumeTarget::Latest,
            Command::new(),
            binding,
        )
        .await
    }

    /// Resumes a run from a specific checkpoint (time-travel resume).
    ///
    /// [`ResumeTarget::Latest`] behaves exactly like [`CompiledGraph::resume`];
    /// [`ResumeTarget::Checkpoint`] replays forward from an older checkpoint's
    /// config — re-running its pending nodes (and applying `command`'s resume
    /// value to any interrupted node) without mutating the original record. The
    /// addressed checkpoint is read-only; the replay appends new boundary
    /// checkpoints to the thread rather than rewriting history.
    ///
    /// Requires a checkpointer and a matching checkpoint with pending nodes;
    /// otherwise returns [`TinyAgentsError::Resume`].
    pub async fn resume_from(
        &self,
        thread_id: impl Into<ThreadId>,
        target: ResumeTarget,
        command: Command<Update>,
    ) -> Result<GraphExecution<State>> {
        self.resume_from_inner(thread_id.into(), target, command, None)
            .await
    }

    /// Resumes a run from `target` with a host-bound recursive-agent binding.
    ///
    /// This is the binding-aware counterpart to [`Self::resume_from`]. It is
    /// useful when a durable continuation reaches a
    /// [`SubAgentNode`](crate::SubAgentNode) after an interrupt or retry.
    /// The binding remains execution-scoped, including for resumed nested
    /// subgraphs, and is not stored on [`CompiledGraph`](crate::CompiledGraph).
    pub async fn resume_from_with_agent_binding(
        &self,
        thread_id: impl Into<ThreadId>,
        target: ResumeTarget,
        command: Command<Update>,
        binding: crate::subagent_node::AgentInvocationBinding,
    ) -> Result<GraphExecution<State>> {
        self.resume_from_inner(thread_id.into(), target, command, Some(binding))
            .await
    }

    fn initial_inputs(
        &self,
        inputs: impl IntoIterator<Item = GraphInput>,
    ) -> Result<Vec<Activation>> {
        let mut active = Vec::new();
        for input in inputs {
            let node = if input.node.as_str() == START {
                self.entry.clone()
            } else if input.node.as_str() == END {
                return Err(TinyAgentsError::Graph(
                    "graph input cannot target END".to_string(),
                ));
            } else {
                if !self.nodes.contains_key(&input.node) {
                    return Err(TinyAgentsError::MissingNode(input.node.to_string()));
                }
                input.node
            };
            active.push(Activation {
                node,
                send_arg: input.payload,
                task_id: String::new(),
            });
        }
        if active.is_empty() {
            return Err(TinyAgentsError::Validation(
                "run_with_inputs requires at least one input".to_string(),
            ));
        }
        Ok(active)
    }

    // ---- State inspection & time travel ------------------------------------

    /// Returns the configured checkpointer or a [`TinyAgentsError::Checkpoint`]
    /// when inspection is attempted on a graph without durability.
    #[allow(clippy::too_many_arguments)]
    async fn execute(
        &self,
        state: State,
        initial_active: Vec<Activation>,
        thread_id: Option<ThreadId>,
        resume_map: HashMap<NodeId, serde_json::Value>,
        initial_barriers: HashMap<NodeId, HashSet<NodeId>>,
        initial_parent: Option<String>,
        binding: Option<crate::subagent_node::AgentInvocationBinding>,
    ) -> Result<GraphExecution<State>> {
        let run_id = tinyagents_harness::ids::new_run_id();
        // When a durable journal is configured, run against a clone whose event
        // sink wraps every emitted event into a `GraphObservation` and appends
        // it (while still forwarding to any pre-existing live sink). The journal
        // sink carries this graph's checkpoint namespace so subgraph runs record
        // their nested path. Default (no journal) leaves `self` untouched.
        if self.journal.is_some() {
            let this = self.clone_with_journal_sink(&run_id, &thread_id);
            this.execute_run(
                run_id,
                state,
                initial_active,
                thread_id,
                resume_map,
                initial_barriers,
                initial_parent,
                binding,
            )
            .await
        } else {
            self.execute_run(
                run_id,
                state,
                initial_active,
                thread_id,
                resume_map,
                initial_barriers,
                initial_parent,
                binding,
            )
            .await
        }
    }

    /// Builds a clone whose `event_sink` is a [`JournalGraphSink`] for `run_id`,
    /// wrapping any existing sink as the live downstream. Returns a plain clone
    /// when no journal is configured.
    fn clone_with_journal_sink(&self, run_id: &RunId, thread_id: &Option<ThreadId>) -> Self {
        let Some(journal) = &self.journal else {
            return self.clone();
        };
        let mut sink = crate::observability::JournalGraphSink::new(
            journal.clone(),
            run_id.clone(),
            self.graph_id.clone(),
        )
        .with_namespace(self.namespace.clone())
        .with_thread(thread_id.clone());
        if let Some(inner) = &self.event_sink {
            sink = sink.with_inner(inner.clone());
        }
        let mut this = self.clone();
        this.event_sink = Some(Arc::new(sink));
        this
    }

    /// Drives one run's superstep loop to completion, an interrupt, or a
    /// failure.
    ///
    /// Builds this run's [`RunCtx`] (identity, recursion stack, and the
    /// accumulators the loop carries forward) and its [`StepRunner`], then
    /// loops: check the recursion/deadline/visit-count guards, run the
    /// active set's node handlers ([`StepRunner::run_sequential`] or
    /// [`StepRunner::run_parallel`]), fold the results
    /// ([`StepRunner::fold_step`]), apply updates through the reducer
    /// ([`CompiledGraph::apply_updates`]), and dispatch to whichever
    /// boundary the step ended at — failure
    /// ([`CompiledGraph::handle_failure_boundary`]), interrupt
    /// ([`CompiledGraph::handle_interrupt_boundary`]), or the normal boundary
    /// ([`CompiledGraph::advance`], which returns the next active set).
    #[allow(clippy::too_many_arguments)]
    async fn execute_run(
        &self,
        run_id: RunId,
        mut state: State,
        initial_active: Vec<Activation>,
        thread_id: Option<ThreadId>,
        resume_map: HashMap<NodeId, serde_json::Value>,
        initial_barriers: HashMap<NodeId, HashSet<NodeId>>,
        initial_parent: Option<String>,
        binding: Option<crate::subagent_node::AgentInvocationBinding>,
    ) -> Result<GraphExecution<State>> {
        let started_at = SystemTime::now();

        // Build this run's recursion stack from the inherited parent frames and
        // push the frame for this graph call. A push that would exceed
        // `max_depth` fails the run with a clear recursion error before any
        // node executes. Graph-call depth (the stack) is tracked separately
        // from node-loop visits (`RunCtx::node_visits`, below).
        let mut recursion =
            RecursionStack::with_frames(self.recursion_frames.clone(), self.recursion_policy);
        let root_run_id = self
            .recursion_frames
            .first()
            .map(|f| f.run_id.clone())
            .unwrap_or_else(|| run_id.clone());
        let parent_run_id = self.recursion_frames.last().map(|f| f.run_id.clone());
        let this_frame = RecursionFrame {
            graph_id: self.graph_id.clone(),
            node_id: self.recursion_node.clone(),
            run_id: run_id.clone(),
            task_id: None,
            namespace: self.namespace.clone(),
            depth: recursion.depth(),
            parent: parent_run_id.clone(),
        };
        if let Err(err) = recursion.push(this_frame) {
            self.emit(GraphEvent::RunStarted {
                run_id: run_id.clone(),
            });
            self.fail_run(&run_id, &thread_id, started_at, 0, &err, None)
                .await;
            return Err(err);
        }
        // Serialized once per run for embedding in every checkpoint's metadata.
        let recursion_meta =
            serde_json::to_value(recursion.frames()).unwrap_or(serde_json::Value::Null);
        let live_frames = recursion.frames().to_vec();

        let mut ctx = RunCtx {
            graph: self,
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
            node_visits: HashMap::new(),
            barrier_arrivals: initial_barriers,
            async_writes: AsyncCheckpointWrites::default(),
            resume_map,
            visited: Vec::new(),
            all_child_runs: Vec::new(),
            steps: 0,
            last_checkpoint: None,
            parent_checkpoint: initial_parent,
        };
        let runner = StepRunner { graph: self };

        ctx.emit(GraphEvent::RunStarted {
            run_id: ctx.run_id.clone(),
        });
        // Surface this run's recursion depth so observers can attribute nested
        // runs without reconstructing the tree from logs.
        ctx.emit(GraphEvent::RecursionDepthChanged {
            depth: ctx.recursion.depth(),
        });
        // Record the run as live before the first superstep is scheduled.
        let mut running = ctx.base_status();
        running.active_nodes = activation_nodes(&initial_active);
        ctx.save_status(running).await;

        let mut active = initial_active;
        while !active.is_empty() {
            // The effective step cap is the smaller of the builder's recursion
            // limit and the policy's `max_total_steps`, so a policy never
            // loosens an existing limit. Both surface a `RecursionLimit`.
            let step_limit = self
                .recursion_limit
                .min(self.recursion_policy.max_total_steps);
            if ctx.steps >= step_limit {
                let err = TinyAgentsError::RecursionLimit(step_limit);
                return self.fail_and_return(&mut ctx, err).await;
            }
            // Whole-run wall-clock deadline: stop *between* super-steps once the
            // elapsed run time reaches it, leaving the last committed boundary
            // checkpoint intact (unlike an external `tokio::time::timeout`, which
            // aborts mid-super-step and cannot). The already-completed super-steps
            // and their checkpoints are preserved; the run fails with `Timeout`.
            if let Some(deadline) = self.run_deadline {
                let elapsed = ctx.started_at.elapsed().unwrap_or_default();
                if elapsed >= deadline {
                    let err = TinyAgentsError::Timeout(format!(
                        "graph run exceeded its {deadline:?} deadline after {} super-step(s) \
                         ({elapsed:?} elapsed)",
                        ctx.steps
                    ));
                    return self.fail_and_return(&mut ctx, err).await;
                }
            }
            // Node-loop recursion: enforce `max_visits_per_node` per activation.
            for activation in &active {
                if let Err(err) = ctx
                    .recursion
                    .record_node_visit(&mut ctx.node_visits, &activation.node)
                {
                    return self.fail_and_return(&mut ctx, err).await;
                }
            }
            ctx.steps += 1;
            // Assign identities before any branch runs. A failure checkpoint
            // carries these identities with its pending activations, letting a
            // later resume skip only the completed fan-out task.
            for (index, activation) in active.iter_mut().enumerate() {
                if activation.task_id.is_empty() {
                    activation.task_id = format!("{}:{}:{}", ctx.steps, index, activation.node);
                }
            }
            ctx.emit(GraphEvent::StepStarted {
                step: ctx.steps,
                active: activation_nodes(&active),
            });

            let outcome = if self.parallel && active.len() > 1 {
                runner.run_parallel(&mut ctx, &active, &state, ctx.steps).await
            } else {
                runner.run_sequential(&mut ctx, &active, &state, ctx.steps).await
            };
            let outcome = match outcome {
                Ok(outcome) => outcome,
                Err(err) => return self.fail_and_return(&mut ctx, err).await,
            };
            let step_run = runner.fold_step(outcome, ctx.steps, &mut ctx.visited);

            // Apply collected updates through the reducer at the boundary. A
            // reducer error here must still fail the run (not just unwind
            // leaving it `Running`).
            state = match self.apply_updates(state, step_run.updates) {
                Ok(state) => state,
                Err(err) => return self.fail_and_return(&mut ctx, err).await,
            };

            // Collect any child runs spawned by subgraph nodes this step. They
            // are embedded into this boundary's checkpoint metadata (keyed by
            // node) and accumulated onto the final `GraphExecution`.
            let step_child_runs = ctx.child_sink.drain();
            ctx.all_child_runs.extend(step_child_runs.iter().cloned());
            let child_runs_meta =
                serde_json::to_value(&step_child_runs).unwrap_or(serde_json::Value::Null);
            let sb = StepBoundary {
                active: &active,
                goto_map: &step_run.goto_map,
                child_runs_meta: &child_runs_meta,
                step: ctx.steps,
            };

            // Node-handler failure (survived any node-retry policy) or an
            // interrupt: both are terminal for this run, persisting a
            // resumable boundary checkpoint before returning.
            if let Some(fail) = step_run.failure {
                return self.handle_failure_boundary(&mut ctx, sb, &state, fail).await;
            }
            if let Some((index, emitted)) = step_run.interrupt {
                return self
                    .handle_interrupt_boundary(&mut ctx, sb, state, index, emitted)
                    .await;
            }

            active = match self.advance(&mut ctx, sb, &state).await {
                Ok(next) => next,
                Err(err) => return self.fail_and_return(&mut ctx, err).await,
            };
        }

        let mut status = ctx.base_status();
        status.status = ExecutionStatus::Completed;
        status.current_step = ctx.steps;
        status.checkpoint_id = ctx.last_checkpoint.clone();
        status.ended_at = Some(SystemTime::now());
        ctx.save_status(status.clone()).await;
        ctx.emit(GraphEvent::RunCompleted {
            run_id: ctx.run_id.clone(),
            steps: ctx.steps,
        });

        Ok(GraphExecution {
            state,
            run_id: ctx.run_id.clone(),
            graph_id: self.graph_id.clone(),
            root_run_id: ctx.root_run_id.clone(),
            parent_run_id: ctx.parent_run_id.clone(),
            child_runs: ctx.all_child_runs,
            visited: ctx.visited,
            steps: ctx.steps,
            interrupts: Vec::new(),
            status,
            checkpoint_id: ctx.last_checkpoint,
        })
    }
}
