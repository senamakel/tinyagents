//! Workflow scheduler DAG: phase scheduling, dependency resolution, and
//! topological ordering.
//!
//! Builds a directed acyclic graph representing the workflow's phase
//! dependencies, computes runnable phases, and projects the schedule into
//! the graph layer for topology introspection.

use anyhow::{Result, anyhow};
use tinyagents_graph::export::GraphTopology;
use tinyagents_graph::recursion::RecursionPolicy;
use tinyagents_graph::{
    ClosureStateReducer, Command, CompiledGraph, GraphBuilder, NodeContext, NodeResult,
};

#[derive(Clone, Default)]
pub(crate) struct SchedulerState;

pub(crate) enum SchedulerUpdate {
    Noop,
}

/// Structure-only *preview* of the scheduler topology.
///
/// `WorkflowEngine` is the effectful scheduler. This helper deliberately does
/// not execute that engine; it only supplies a stable topology to diagnostic
/// UIs, so callers must never present it as the graph that ran a workflow.
pub fn scheduler_topology_preview() -> Result<GraphTopology> {
    Ok(build_scheduler_graph(1)?.topology())
}

pub(crate) fn build_scheduler_graph(
    phase_count: usize,
) -> Result<CompiledGraph<SchedulerState, SchedulerUpdate>> {
    let mut builder = GraphBuilder::<SchedulerState, SchedulerUpdate>::new().set_reducer(
        ClosureStateReducer::new(|state: SchedulerState, update| {
            match update {
                SchedulerUpdate::Noop => {}
            }
            Ok(state)
        }),
    );
    // Effects are supplied at invocation time through the execution context.
    // These nodes only make the validated dispatch/run/done topology inspectable;
    // `WorkflowEngine` installs its effectful variants below.
    builder = builder.add_node(
        "dispatch",
        |_state: SchedulerState, _context: NodeContext| async move {
            Ok(NodeResult::Command(Command::default().with_goto(["done"])))
        },
    );
    let graph = builder
        .add_node(
            "run_phase",
            |_state: SchedulerState, _context: NodeContext| async move {
                Ok(NodeResult::Command(
                    Command::default().with_goto(["dispatch"]),
                ))
            },
        )
        .add_node(
            "done",
            |_state: SchedulerState, _context: NodeContext| async move {
                Ok(NodeResult::Update(SchedulerUpdate::Noop))
            },
        )
        .set_entry("dispatch")
        .mark_command_routing("dispatch")
        .mark_command_routing("run_phase")
        .set_finish("done")
        .compile()
        .map_err(|error| anyhow!("workflow scheduler graph compile failed: {error}"))?
        .with_recursion_policy(RecursionPolicy {
            max_visits_per_node: Some(phase_count + 2),
            max_total_steps: (phase_count + 1) * 3 + 16,
            ..RecursionPolicy::default()
        });
    Ok(graph)
}
