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

        // The resume value belongs to the node(s) that actually interrupted. The
        // pending set is deliberately wider than that at an interrupt boundary
        // (it also carries the successors of branches that completed before the
        // interrupt), so fanning the value across it would hand `ctx.resume` to
        // nodes that have never run. A boundary that recorded no interrupt (a
        // failure boundary, resumed via `retry` with no value) keeps the old
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

        self.execute(RunSeed {
            state: checkpoint.state,
            active,
            thread_id: Some(thread_id),
            resume_map,
            barriers: initial_barriers,
            parent: initial_parent,
            binding,
            _update: std::marker::PhantomData,
        })
        .await
    }
}
