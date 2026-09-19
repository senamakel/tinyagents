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

        // Prefer the persisted pending activations (which preserve each pending
        // node's `Send` arg); fall back to the node-id projection for
        // checkpoints written before that field existed.
        let active: Vec<Activation> = match &checkpoint.pending_activations {
            Some(pending) if !pending.is_empty() => pending.iter().map(Activation::from).collect(),
            _ => checkpoint
                .next_nodes
                .iter()
                .cloned()
                .map(Activation::node)
                .collect(),
        };
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
        let done: HashSet<String> = if recorded.is_empty() {
            checkpoint
                .pending_writes
                .iter()
                .map(|w| w.task_id.clone())
                .collect()
        } else {
            recorded.iter().map(|w| w.task_id.clone()).collect()
        };
        let active: Vec<Activation> = if done.is_empty() {
            active
        } else {
            let filtered: Vec<Activation> = active
                .iter()
                // A node name is not a task identity: a Send fan-out can have
                // several live activations of one node. Legacy checkpoints
                // have no persisted task id, so leave them runnable.
                .filter(|a| a.task_id.is_empty() || !done.contains(&a.task_id))
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
        let mut resume_map = HashMap::new();
        if let Some(value) = command.resume {
            let interrupted = interrupted_nodes(&checkpoint, &active);
            if interrupted.is_empty() {
                for activation in &active {
                    resume_map.insert(activation.node.clone(), value.clone());
                }
            } else {
                for node in interrupted {
                    resume_map.insert(node, value.clone());
                }
            }
        }

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
        let carried_completed = if mid_step && !checkpoint.completed_tasks.is_empty() {
            Some(checkpoint.completed_tasks.clone())
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
                carried_completed,
            },
            _update: std::marker::PhantomData,
        })
        .await
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
