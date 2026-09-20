//! Lowers a [`WorkflowDefinition`] to a `tinyagents_graph::CompiledGraph`
//! (Phase 4 of `docs/runtime-comparison/feature-gaps.md`; see
//! `docs/runtime-comparison/code-review-graph.md` I12/R6).
//!
//! Two distinct things live here, on purpose:
//!
//! - [`lowered_topology`]: a pure, *never-executed* structural export — one
//!   graph node per phase, `depends_on` expressed as literal
//!   [`GraphBuilder::add_waiting_edge`] barriers — used only so a host or a
//!   test can inspect "the DAG this workflow defines" (mirrors the
//!   `scheduler_topology_preview`/`build_scheduler_graph` split already in
//!   `workflow::graph`, whose doc makes the same "structure-only preview,
//!   never the graph that ran the workflow" distinction).
//! - [`lower_workflow`]: the *executable* lowering [`WorkflowEngine::drive`]
//!   actually runs under the `graph-workflows` feature. It builds a small
//!   `dispatch -> <phase> -> dispatch -> ... ` graph — one real node per
//!   phase plus a `dispatch` router — rather than literal per-phase
//!   waiting-edge topology.
//!
//! ## Why the executable graph doesn't just run the waiting-edge topology
//!
//! [`WorkflowEngine::run_phase`] persists a phase's Running/Completed/Failed
//! transition with an optimistic compare-and-swap keyed to the durable
//! run's revision. Every branch active in one graph superstep is handed the
//! *same* pre-step state snapshot (this holds for both sequential and
//! parallel supersteps — see `tinyagents_graph::compiled`'s module docs), so
//! two phases that became ready in the same superstep (e.g. two independent
//! phases that both depend only on a common upstream phase) would both
//! start `run_phase` from the same stale revision and race the same CAS.
//! Only one would win; the loser's `persist` would report a spurious
//! "lease lost" failure even though the lease is fine — a real correctness
//! hazard, not a cosmetic one.
//!
//! The `dispatch` router node sidesteps this by construction: it is the
//! *only* thing that decides which phase runs next
//! ([`next_runnable_phase`], the same function the legacy scheduler uses),
//! and it always routes to exactly one phase node before looping back to
//! itself. Exactly one phase node is ever active in any given superstep, so
//! `run_phase`'s CAS is never raced and can be reused completely unchanged.
//! `depends_on` ordering is therefore enforced the same way it always was —
//! algorithmically, by `next_runnable_phase` — not by the executable
//! graph's own edges (which is why [`lowered_topology`]'s literal
//! waiting-edge shape is a separate, non-executed structure).
//!
//! Each phase's own agent fan-out (bounded by `WorkflowDefinition`'s single
//! `default_concurrency`, since [`WorkflowPhase`] carries no per-phase
//! override) still goes through `run_phase`'s existing
//! `tinyagents_graph::parallel::map_reduce` call — the graph crate's own
//! bounded fan-out primitive — completely unchanged.

use std::collections::HashSet;
use std::sync::Arc;

use tinyagents_graph::export::GraphTopology;
use tinyagents_graph::recursion::RecursionPolicy;
use tinyagents_graph::{
    ClosureStateReducer, Command, CompiledGraph, GraphBuilder, NodeContext, NodeResult, START,
};
use tinyagents_harness::CancellationToken;
use tinyagents_harness::TinyAgentsError;
use tinyagents_session::run_ledger::{WorkflowRun, WorkflowRunStatus};

use super::engine::{OrchestrationError, PersistRequest, WorkflowEngine, WorkflowExecutor, WorkflowStore};
use super::state::{all_phases_completed, next_runnable_phase, reset_running_phases, synthesize_summary};
use super::WorkflowDefinition;

/// The lowered graph's `State`/`Update` type: the durable run row plus a
/// running count of how many children this drive has spawned so far (the
/// `max_children` budget). An overwrite reducer (`Update == State`) is safe
/// here because — per the module doc — exactly one node is ever active per
/// superstep, so there is never a sibling update to merge.
#[derive(Clone)]
pub(crate) struct SchedulerState {
    pub(crate) run: WorkflowRun,
    pub(crate) total_spawned: u32,
}

/// Builds the *executable* lowered graph: a `dispatch` router plus one node
/// per phase in `definition`. See the module doc for why this shape (rather
/// than literal per-phase waiting edges) is what actually runs.
pub(crate) fn lower_workflow<S, E>(
    engine: Arc<WorkflowEngine<S, E>>,
    definition: Arc<WorkflowDefinition>,
    run_id: String,
    owner: String,
    cancel: CancellationToken,
) -> Result<CompiledGraph<SchedulerState, SchedulerState>, OrchestrationError>
where
    S: WorkflowStore + 'static,
    E: WorkflowExecutor + 'static,
{
    let mut builder = GraphBuilder::<SchedulerState, SchedulerState>::overwrite();

    {
        let engine = engine.clone();
        let definition = definition.clone();
        let run_id = run_id.clone();
        let owner = owner.clone();
        let cancel = cancel.clone();
        builder = builder.add_node("dispatch", move |state: SchedulerState, _ctx: NodeContext| {
            let engine = engine.clone();
            let definition = definition.clone();
            let run_id = run_id.clone();
            let owner = owner.clone();
            let cancel = cancel.clone();
            async move { dispatch(engine, definition, run_id, owner, cancel, state).await }
        });
    }
    builder = builder.set_entry("dispatch").mark_command_routing("dispatch");

    for phase in &definition.phases {
        let engine = engine.clone();
        let definition = definition.clone();
        let run_id = run_id.clone();
        let owner = owner.clone();
        let cancel = cancel.clone();
        let phase = phase.clone();
        let node_id = phase.name.clone();
        builder = builder
            .add_node(node_id.clone(), move |state: SchedulerState, _ctx: NodeContext| {
                let engine = engine.clone();
                let definition = definition.clone();
                let run_id = run_id.clone();
                let owner = owner.clone();
                let cancel = cancel.clone();
                let phase = phase.clone();
                async move {
                    run_phase_node(engine, definition, run_id, owner, cancel, phase, state).await
                }
            })
            .mark_command_routing(node_id);
    }

    let phase_count = definition.phases.len();
    let graph = builder
        .compile()
        .map_err(|error| OrchestrationError(format!("workflow graph lowering failed: {error}")))?
        .with_recursion_policy(RecursionPolicy {
            max_visits_per_node: Some(phase_count + 2),
            max_total_steps: (phase_count + 1) * 3 + 16,
            ..RecursionPolicy::default()
        });
    Ok(graph)
}

/// The `dispatch` node: picks the next runnable phase
/// ([`next_runnable_phase`], identical to the legacy scheduler), or settles
/// the run when nothing is left to run — either every phase completed, or a
/// cancellation was observed with no phase in flight, or no phase is
/// runnable and the workflow is not done (a `depends_on` deadlock).
async fn dispatch<S, E>(
    engine: Arc<WorkflowEngine<S, E>>,
    definition: Arc<WorkflowDefinition>,
    run_id: String,
    owner: String,
    cancel: CancellationToken,
    state: SchedulerState,
) -> tinyagents_graph::Result<NodeResult<SchedulerState>>
where
    S: WorkflowStore + 'static,
    E: WorkflowExecutor + 'static,
{
    if cancel.is_cancelled() {
        let mut phase_states = state.run.phase_states.clone();
        reset_running_phases(
            &mut phase_states,
            "workflow interrupted; phase will retry on resume",
        );
        return match engine
            .persist(
                &state.run,
                PersistRequest {
                    phase_states,
                    child_run_ids: state.run.child_run_ids.clone(),
                    status: WorkflowRunStatus::Interrupted,
                    summary: None,
                    terminal: false,
                },
                &owner,
            )
            .await
        {
            Ok(updated) => {
                engine.finish_cancelled(&run_id);
                Ok(NodeResult::Update(SchedulerState {
                    run: updated,
                    ..state
                }))
            }
            Err(error) => settle_infra_error(&engine, &run_id, &owner, state, error).await,
        };
    }

    if let Some(phase) = next_runnable_phase(&definition, &state.run.phase_states) {
        let target = phase.name.clone();
        return Ok(NodeResult::Command(
            Command::default().with_update(state).with_goto([target]),
        ));
    }

    if all_phases_completed(&definition, &state.run.phase_states) {
        let summary = synthesize_summary(&definition, &state.run.phase_states);
        return match engine
            .persist(
                &state.run,
                PersistRequest {
                    phase_states: state.run.phase_states.clone(),
                    child_run_ids: state.run.child_run_ids.clone(),
                    status: WorkflowRunStatus::Completed,
                    summary,
                    terminal: true,
                },
                &owner,
            )
            .await
        {
            Ok(updated) => {
                engine.finish_completed(&run_id, state.total_spawned as usize);
                Ok(NodeResult::Update(SchedulerState {
                    run: updated,
                    ..state
                }))
            }
            Err(error) => settle_infra_error(&engine, &run_id, &owner, state, error).await,
        };
    }

    let reason = "no runnable phase (dependency deadlock)".to_owned();
    match engine
        .persist(
            &state.run,
            PersistRequest {
                phase_states: state.run.phase_states.clone(),
                child_run_ids: state.run.child_run_ids.clone(),
                status: WorkflowRunStatus::Failed,
                summary: Some(reason.clone()),
                terminal: true,
            },
            &owner,
        )
        .await
    {
        Ok(updated) => {
            engine.finish_failed(&run_id, reason);
            Ok(NodeResult::Update(SchedulerState {
                run: updated,
                ..state
            }))
        }
        Err(error) => settle_infra_error(&engine, &run_id, &owner, state, error).await,
    }
}

/// One phase's node body: reuses [`WorkflowEngine::run_phase`] unchanged,
/// then either loops back to `dispatch` (the phase completed and the run is
/// still `Running`) or stops (a failure/interrupt already durably
/// persisted, and its terminal event already emitted, by `run_phase`
/// itself).
#[allow(clippy::too_many_arguments)]
async fn run_phase_node<S, E>(
    engine: Arc<WorkflowEngine<S, E>>,
    definition: Arc<WorkflowDefinition>,
    run_id: String,
    owner: String,
    cancel: CancellationToken,
    phase: super::WorkflowPhase,
    state: SchedulerState,
) -> tinyagents_graph::Result<NodeResult<SchedulerState>>
where
    S: WorkflowStore + 'static,
    E: WorkflowExecutor + 'static,
{
    match engine
        .run_phase(
            &state.run,
            &definition,
            &phase,
            state.total_spawned,
            cancel,
            &owner,
        )
        .await
    {
        Ok((updated, spawned)) => {
            let next = SchedulerState {
                run: updated,
                total_spawned: state.total_spawned + spawned,
            };
            if next.run.status == WorkflowRunStatus::Running {
                Ok(NodeResult::Command(
                    Command::default().with_update(next).with_goto(["dispatch"]),
                ))
            } else {
                // `run_phase` already persisted this transition and it is a
                // terminal one (Failed/Interrupted/Cancelled) — surface the
                // matching event and stop the loop (no `goto`).
                match next.run.status {
                    WorkflowRunStatus::Interrupted | WorkflowRunStatus::Cancelled => {
                        engine.finish_cancelled(&run_id)
                    }
                    WorkflowRunStatus::Failed => engine.finish_failed(
                        &run_id,
                        next.run
                            .summary
                            .clone()
                            .unwrap_or_else(|| "workflow phase failed".to_owned()),
                    ),
                    WorkflowRunStatus::Completed | WorkflowRunStatus::Pending => {}
                    WorkflowRunStatus::Running => unreachable!("handled above"),
                }
                Ok(NodeResult::Update(next))
            }
        }
        Err(error) => settle_infra_error(&engine, &run_id, &owner, state, error).await,
    }
}

/// A `run_phase`/`persist` call failed for infrastructure reasons (not an
/// ordinary phase outcome). Mirrors `drive_legacy`'s own
/// `owner_lost`/`emit_recorded_terminal` escape hatches exactly: a
/// stop/resume hand-off or a lease takeover must not manufacture a stale
/// terminal event *or* a hard error for a driver that has already been
/// fenced — that case stops the loop silently (`Ok`, no `goto`), matching
/// `drive_legacy`'s `return Ok(())`. A genuine, unfenced infrastructure
/// failure instead propagates as an `Err`, matching `drive_legacy`'s
/// `return Err(error)`; `WorkflowEngine::drive_via_graph`'s single
/// catch-all around `graph.run(..)` is what emits `finish_failed` for it
/// (once, regardless of which node's `Err` bubbled up).
async fn settle_infra_error<S, E>(
    engine: &Arc<WorkflowEngine<S, E>>,
    run_id: &str,
    owner: &str,
    state: SchedulerState,
    error: OrchestrationError,
) -> tinyagents_graph::Result<NodeResult<SchedulerState>>
where
    S: WorkflowStore + 'static,
    E: WorkflowExecutor + 'static,
{
    if engine.owner_lost(run_id, owner).await
        || engine
            .emit_recorded_terminal(run_id, state.total_spawned as usize)
            .await
    {
        return Ok(NodeResult::Update(state));
    }
    // A genuine, unfenced infrastructure failure: propagate it and let
    // `WorkflowEngine::drive_via_graph`'s single catch-all emit
    // `finish_failed` exactly once, rather than emitting it here too.
    Err(TinyAgentsError::Graph(error.to_string()))
}

/// A pure, never-executed structural export: one node per phase, with
/// `depends_on` expressed as literal [`GraphBuilder::add_waiting_edge`]
/// barriers — "the DAG this workflow defines", not the graph that actually
/// runs it (see the module doc). Requires a single root phase (a phase with
/// an empty `depends_on`); `tinyagents_graph`'s topology only records one
/// `entry` node, so a definition with several independent root phases would
/// lose all but one of them here.
pub fn lowered_topology(definition: &WorkflowDefinition) -> Result<GraphTopology, OrchestrationError> {
    let mut builder = GraphBuilder::<(), ()>::new()
        .set_reducer(ClosureStateReducer::new(|state: (), _update: ()| Ok(state)));
    for phase in &definition.phases {
        builder = builder.add_node(phase.name.clone(), |state: (), _ctx: NodeContext| async move {
            Ok(NodeResult::Update(state))
        });
    }
    let depended_on: HashSet<&str> = definition
        .phases
        .iter()
        .flat_map(|phase| phase.depends_on.iter().map(String::as_str))
        .collect();
    for phase in &definition.phases {
        if phase.depends_on.is_empty() {
            builder = builder.add_edge(START, phase.name.clone());
        } else {
            for dependency in &phase.depends_on {
                builder = builder.add_waiting_edge(dependency.clone(), phase.name.clone());
            }
        }
        if !depended_on.contains(phase.name.as_str()) {
            builder = builder.set_finish(phase.name.clone());
        }
    }
    let graph = builder
        .compile()
        .map_err(|error| OrchestrationError(format!("workflow topology lowering failed: {error}")))?;
    Ok(graph.topology())
}
