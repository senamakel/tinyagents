//! Resume: loading a checkpoint, filtering out already-completed tasks, and
//! building the resume-value map handed to the re-run node(s).
//!
//! Split out of `executor.rs`; see that module's doc comment for the public
//! `resume`/`resume_from`/`retry` entry points that call into
//! [`CompiledGraph::resume_from_inner`].

use super::*;

use crate::compiled::executor::RunSeed;

impl<State, Update> CompiledGraph<State, Update>
where
    State: Clone + Send + Sync + 'static,
    Update: Send + 'static,
{
    pub(super) async fn resume_from_inner(
        &self,
        thread_id: ThreadId,
        target: ResumeTarget,
        command: Command<Update>,
        binding: Option<crate::subagent_node::AgentInvocationBinding>,
        options: RunOptions,
    ) -> Result<GraphExecution<State>> {
        let checkpointer = self
            .checkpointer
            .as_ref()
            .ok_or_else(|| TinyAgentsError::Resume("no checkpointer configured".to_string()))?;

        let checkpoint_id = match &target {
            ResumeTarget::Latest => None,
            ResumeTarget::Checkpoint(id) => Some(id.as_str()),
        };
        let checkpoint = checkpointer
            .get_scoped(thread_id.as_str(), checkpoint_id, &self.namespace)
            .await?
            .ok_or_else(|| match &target {
                ResumeTarget::Latest => {
                    TinyAgentsError::Resume(format!("no checkpoint found for thread `{thread_id}`"))
                }
                ResumeTarget::Checkpoint(id) => TinyAgentsError::Resume(format!(
                    "no checkpoint `{id}` found for thread `{thread_id}`"
                )),
            })?;
        // Resume *loads* this checkpoint — it is a read, not a write — so emit a
        // restore event, not `CheckpointSaved` (which would falsely inflate
        // persisted-checkpoint counts and mislead durability observers).
        self.emit(GraphEvent::CheckpointRestored {
            checkpoint_id: CheckpointId::new(checkpoint.checkpoint_id.clone()),
        });

        // `checkpoint` was already normalized on read (every backend's decode
        // path calls `Checkpoint::normalize`), so `tasks` is always the
        // single source of truth here, regardless of the stored record's
        // original format version.
        let active: Vec<Activation> = checkpoint.tasks.iter().map(Activation::from).collect();
        if active.is_empty() {
            return Err(TinyAgentsError::Resume(
                "checkpoint has no pending nodes to resume".to_string(),
            ));
        }

        // Partial-failure guard. The boundary that produced this checkpoint
        // recorded a completion marker per task that had already finished; a
        // node named by *both* the pending set and that ledger has therefore
        // already run, and re-running it would repeat its side effects. On a
        // checkpoint the executor itself wrote the two sets are disjoint, so
        // this is a no-op — it earns its keep on a checkpoint that was
        // hand-built, time-travelled to, or edited through `update_state`,
        // where `next_nodes` can legitimately disagree with what ran.
        let completed_config = CheckpointConfig {
            thread_id: thread_id.to_string(),
            checkpoint_id: Some(checkpoint.checkpoint_id.clone()),
            namespace: self.namespace.clone(),
        };
        let recorded = checkpointer.get_writes(&completed_config).await?;
        let ledger: &[crate::checkpoint::PendingWrite] = if recorded.is_empty() {
            &checkpoint.pending_writes
        } else {
            &recorded
        };
        // Only a completion marker (or any other non-replay write) says a
        // task ran. A replay memo — a `durable_task` write or a deferred
        // `interrupt_after` result — belongs to a task that is still
        // *pending* and must re-run (consuming the memo), so it is excluded
        // here and grouped per task below instead.
        let done: HashSet<String> = ledger
            .iter()
            .filter(|w| !w.is_task_replay())
            .map(|w| w.task_id.as_str().to_string())
            .collect();
        let mut task_writes: HashMap<String, Vec<crate::checkpoint::PendingWrite>> = HashMap::new();
        for write in ledger.iter().filter(|w| w.is_task_replay()) {
            task_writes
                .entry(write.task_id.as_str().to_string())
                .or_default()
                .push(write.clone());
        }
        // Executor-injected `interrupt_before`/`interrupt_after` pauses this
        // checkpoint recorded: resuming acknowledges them, so the re-run of
        // that task skips the same phase instead of pausing again.
        let acknowledged_interrupts: HashSet<String> = checkpoint
            .interrupts
            .iter()
            .filter_map(|interrupt| {
                let phase = interrupt.payload.get("phase")?.as_str()?;
                if phase != "before" && phase != "after" {
                    return None;
                }
                let task_id = interrupt.task_id.as_ref()?;
                Some(
                    crate::compiled::run_ctx::RunCtx::<State, Update>::interrupt_ack_key(
                        phase, task_id,
                    ),
                )
            })
            .collect();
        let active: Vec<Activation> = if done.is_empty() {
            active
        } else {
            let filtered: Vec<Activation> = active
                .iter()
                // A node name is not a task identity: a Send fan-out can have
                // several live activations of one node. Legacy checkpoints
                // have no persisted task id, so leave them runnable.
                .filter(|a| a.task_id.as_str().is_empty() || !done.contains(a.task_id.as_str()))
                .cloned()
                .collect();
            if filtered.is_empty() {
                // Every pending node claims to have run. Trust the pending set
                // rather than turning a resumable checkpoint into a hard error:
                // a wrong re-run is recoverable, a stuck thread is not.
                tracing::warn!(
                    "[graph:resume] every pending node of checkpoint `{}` has a completion \
                     marker; resuming them anyway rather than stranding the thread",
                    checkpoint.checkpoint_id
                );
                active
            } else {
                if filtered.len() != active.len() {
                    tracing::debug!(
                        "[graph:resume] checkpoint `{}`: skipping {} already-completed task(s)",
                        checkpoint.checkpoint_id,
                        active.len() - filtered.len()
                    );
                }
                filtered
            }
        };

        // The resume value belongs to the node(s) that actually interrupted.
        // Interrupt/failure boundaries persist `pending` as exactly the
        // stalled (interrupted/failed) branches of that step — a completed
        // sibling's routing is deferred rather than folded into `pending`
        // (see `boundary::advance`'s `carried_completed` handling) — but a
        // checkpoint could still carry a wider pending set (a hand-built one,
        // or one written before this policy), so this still keys off
        // `interrupted_nodes` rather than assuming `active` is exactly the
        // interrupted set. A boundary that recorded no interrupt (a failure
        // boundary, resumed via `retry` with no value) keeps the old
        // fan-across-pending behaviour.
        // I1/R5: keyed by task id (falling back to node id) so a `Send`
        // fan-out of the same node — several live activations sharing one
        // `NodeId` — each receive their own resume value instead of every
        // same-node activation racing for a single node-keyed slot. Prefer
        // the persisted interrupts' own `task_id` (stamped by the interrupt
        // boundary, R5) when present; fall back to `interrupted_nodes` (node
        // names only — a checkpoint written before task identity existed, or
        // a re-emitted subgraph interrupt whose task id was not stamped),
        // keying by every active activation of that node.
        let mut resume_map: HashMap<String, serde_json::Value> = HashMap::new();
        // A caller-supplied per-task map (I1) always wins: it names its
        // targets explicitly, so there is nothing to infer.
        for (task_id, value) in &command.resume_by_task {
            resume_map.insert(task_id.as_str().to_string(), value.clone());
        }
        if let Some(value) = command.resume {
            let task_targets: Vec<String> = checkpoint
                .interrupts
                .iter()
                .filter_map(|i| i.task_id.as_ref())
                .map(|t| t.as_str().to_string())
                .filter(|t| active.iter().any(|a| a.task_id.as_str() == t))
                .collect();
            if !task_targets.is_empty() {
                for task_id in task_targets {
                    resume_map.entry(task_id).or_insert_with(|| value.clone());
                }
            } else {
                let interrupted = interrupted_nodes(&checkpoint, &active);
                if interrupted.is_empty() {
                    for activation in &active {
                        resume_map
                            .entry(resume_key(activation))
                            .or_insert_with(|| value.clone());
                    }
                } else {
                    for node in interrupted {
                        for activation in active.iter().filter(|a| a.node == node) {
                            resume_map
                                .entry(resume_key(activation))
                                .or_insert_with(|| value.clone());
                        }
                    }
                }
            }
        }

        // Fail closed on `Interrupt::response_schema`: every value this resume
        // would hand to a schema-bearing interrupt's task is validated *now*,
        // before `execute` claims the thread or any boundary writes a
        // checkpoint, so a rejected value leaves the thread's checkpoint
        // exactly as it was.
        self.validate_resume_values(&checkpoint, &active, &resume_map)?;

        // Restore accumulated barrier arrivals so a join's precondition survives
        // the interrupt/failure boundary this checkpoint recorded.
        let initial_barriers = barriers_from_persisted(&checkpoint.barrier_arrivals);
        // Chain the first post-resume boundary onto the checkpoint we loaded so
        // the lineage spine stays connected across the resume.
        let initial_parent = Some(checkpoint.checkpoint_id.clone());

        // A mid-step checkpoint (an interrupt/failure `loop`-source boundary
        // — stamped with `interrupted_nodes` or `failed_node`) leaves its
        // completed siblings unrouted (see `boundary::advance`'s
        // `carried_completed` doc, the C2 fix): carry their node ids forward
        // so this resumed run's *first* boundary routes the whole original
        // step together, rather than routing only the freshly re-run
        // pending set in isolation (which would let a successor observe a
        // state missing whatever the other, already-completed siblings
        // wrote).
        //
        // The `source == "loop"` check matters: `update_state` (I2) can
        // *also* stamp `interrupted_nodes` onto an `update`-sourced
        // checkpoint, purely to preserve resume-value provenance — but
        // `update_state` always fully resolves every carried completion's
        // routing itself before writing (see `state_api::update_state`), so
        // its `completed_tasks` never represents owed work. Treating it as
        // mid-step here would route those already-resolved completions a
        // second time, scheduling their successors twice.
        let source_is_loop = checkpoint
            .metadata
            .get("source")
            .and_then(serde_json::Value::as_str)
            == Some("loop");
        let mid_step = source_is_loop
            && (checkpoint.metadata.get("interrupted_nodes").is_some()
                || checkpoint.metadata.get("failed_node").is_some());
        let carried_completed = if mid_step && !checkpoint.completed.is_empty() {
            Some(
                checkpoint
                    .completed
                    .iter()
                    .map(|c| (c.node.clone(), c.routes.clone()))
                    .collect(),
            )
        } else {
            None
        };
        // I3: continue this thread's step counter and per-node visit counts
        // from the loaded checkpoint instead of restarting at zero, so
        // `metadata.step` (and `get_state_history`) stays monotonic and
        // `RecursionPolicy::max_visits_per_node` bounds the whole thread's
        // lifetime rather than resetting every resume.
        let initial_steps = checkpoint.to_metadata().step;
        let initial_node_visits = node_visits_from_persisted(&checkpoint.metadata);
        // I5/R3: carry the per-node channel-versions bookkeeping forward
        // across the resume, same reasoning as `initial_node_visits`.
        let initial_versions_seen = checkpoint.versions_seen.clone().into_iter().collect();

        self.execute(RunSeed {
            state: checkpoint.state,
            active,
            thread_id: Some(thread_id),
            resume_map,
            barriers: initial_barriers,
            parent: initial_parent,
            binding,
            resume_seed: crate::compiled::run_ctx::ResumeSeed {
                initial_steps,
                initial_node_visits,
                initial_versions_seen,
                carried_completed,
                task_writes,
                acknowledged_interrupts,
            },
            options,
            _update: std::marker::PhantomData,
        })
        .await
    }

    /// Validates the resume value each schema-bearing pending interrupt
    /// would receive (see [`Interrupt::response_schema`]) against that
    /// schema, via [`tinyagents_harness::tool::validate_against_schema`].
    ///
    /// The value is looked up exactly as [`RunCtx::node_context`] will hand
    /// it out — by the interrupt's task id, falling back to its node id (the
    /// legacy/whole-node key) — so what is validated is what the node would
    /// see. An interrupt whose task receives no value at all (a bare
    /// `retry`) has nothing to validate. Must be called before any
    /// checkpoint write.
    ///
    /// [`RunCtx::node_context`]: crate::compiled::run_ctx::RunCtx::node_context
    fn validate_resume_values(
        &self,
        checkpoint: &Checkpoint<State>,
        active: &[Activation],
        resume_map: &HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        for interrupt in &checkpoint.interrupts {
            let Some(schema) = &interrupt.response_schema else {
                continue;
            };
            let value = interrupt
                .task_id
                .as_ref()
                .and_then(|task| resume_map.get(task.as_str()))
                .or_else(|| {
                    // Node-keyed fallback, only when the interrupt's node is
                    // actually pending (matching `node_context`'s lookup).
                    active
                        .iter()
                        .any(|a| a.node == interrupt.node)
                        .then(|| resume_map.get(interrupt.node.as_str()))
                        .flatten()
                });
            let Some(value) = value else {
                continue;
            };
            tinyagents_harness::tool::validate_against_schema(schema, value).map_err(|err| {
                TinyAgentsError::Validation(format!(
                    "resume value for interrupt `{}` (node `{}`) rejected by its response_schema: \
                     {err}",
                    interrupt.id, interrupt.node
                ))
            })?;
        }
        Ok(())
    }
}

/// The key an activation's resume value is looked up under: its task id
/// when known (I1/R5 — distinguishes concurrent same-node activations),
/// falling back to its node id (legacy checkpoints, or a value fanned across
/// every pending node with no interrupt provenance).
fn resume_key(activation: &Activation) -> String {
    if activation.task_id.as_str().is_empty() {
        activation.node.to_string()
    } else {
        activation.task_id.as_str().to_string()
    }
}

/// Parses a checkpoint's persisted `metadata.node_visits` object (see
/// `boundary`'s checkpoint builders) back into the live per-node visit-count
/// map. Missing/malformed metadata (checkpoints written before this field
/// existed) yields an empty map — the pre-I3 behavior for that checkpoint.
fn node_visits_from_persisted(metadata: &serde_json::Value) -> HashMap<NodeId, usize> {
    metadata
        .get("node_visits")
        .and_then(serde_json::Value::as_object)
        .map(|obj| {
            obj.iter()
                .filter_map(|(node, count)| {
                    count
                        .as_u64()
                        .map(|c| (NodeId::from(node.as_str()), c as usize))
                })
                .collect()
        })
        .unwrap_or_default()
}
