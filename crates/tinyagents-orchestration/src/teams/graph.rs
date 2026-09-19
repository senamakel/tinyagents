//! Member worker graph execution: a generic three-step DAG that invokes a host
//! worker, routes on success/failure, and calls back to the host.
//!
//! This module owns the structure of member execution as seen by the graph
//! layer (entry → execute → complete/fail → done), while letting the host
//! supply the actual work (`run_worker`) and post-work effects (`on_complete`
//! and `on_failed`). It is the bridge between durable team coordination
//! (membership, tasks, event delivery) and the graph's lifecycle and event
//! streaming.

use std::future::Future;
use std::sync::Arc;

use anyhow::Result;
use tinyagents_graph::export::GraphTopology;
use tinyagents_graph::stream::GraphEventSink;
use tinyagents_graph::{
    ClosureStateReducer, Command, CompiledGraph, GraphBuilder, NodeContext, NodeResult,
};

/// Terminal classification of a host worker run.
///
/// Returned by a host's `run_worker` callback to signal whether the member
/// completed its work successfully or failed. The member graph routes on this
/// outcome and calls the appropriate host callback (`on_complete` or
/// `on_failed`).
pub enum MemberOutcome {
    /// Member completed successfully; carries the output to be recorded.
    Completed { output: String },
    /// Member failed; carries the failure reason to be recorded.
    Failed { reason: String },
}

#[derive(Clone, Default)]
struct MemberState {
    payload: Option<String>,
}

enum MemberUpdate {
    Payload(String),
    Noop,
}

fn graph_err(error: anyhow::Error) -> tinyagents_harness::TinyAgentsError {
    tinyagents_harness::TinyAgentsError::Graph(error.to_string())
}

/// Run the generic complete-or-fail member graph with host supplied effects.
///
/// `event_sink` is optional because observability belongs to the embedding host;
/// when supplied it receives the graph executor's lifecycle events unchanged.
pub async fn run_member_graph<W, WF, C, CF, F, FF>(
    event_sink: Option<Arc<dyn GraphEventSink>>,
    run_worker: W,
    on_complete: C,
    on_failed: F,
) -> Result<()>
where
    W: Fn() -> WF + Clone + Send + Sync + 'static,
    WF: Future<Output = Result<MemberOutcome>> + Send + 'static,
    C: Fn(String) -> CF + Clone + Send + Sync + 'static,
    CF: Future<Output = Result<()>> + Send + 'static,
    F: Fn(String) -> FF + Clone + Send + Sync + 'static,
    FF: Future<Output = Result<()>> + Send + 'static,
{
    let mut graph = build_member_graph(run_worker, on_complete, on_failed)?;
    if let Some(event_sink) = event_sink {
        graph = graph.with_event_sink(event_sink);
    }
    graph
        .run(MemberState::default())
        .await
        .map_err(|error| anyhow::anyhow!("member graph run failed: {error}"))?;
    Ok(())
}

fn build_member_graph<W, WF, C, CF, F, FF>(
    run_worker: W,
    on_complete: C,
    on_failed: F,
) -> Result<CompiledGraph<MemberState, MemberUpdate>>
where
    W: Fn() -> WF + Clone + Send + Sync + 'static,
    WF: Future<Output = Result<MemberOutcome>> + Send + 'static,
    C: Fn(String) -> CF + Clone + Send + Sync + 'static,
    CF: Future<Output = Result<()>> + Send + 'static,
    F: Fn(String) -> FF + Clone + Send + Sync + 'static,
    FF: Future<Output = Result<()>> + Send + 'static,
{
    let mut builder = GraphBuilder::<MemberState, MemberUpdate>::new().set_reducer(
        ClosureStateReducer::new(|mut state: MemberState, update: MemberUpdate| {
            if let MemberUpdate::Payload(payload) = update {
                state.payload = Some(payload);
            }
            Ok(state)
        }),
    );
    builder = builder.add_node(
        "execute",
        move |_state: MemberState, _context: NodeContext| {
            let run_worker = run_worker.clone();
            async move {
                match run_worker().await.map_err(graph_err)? {
                    MemberOutcome::Completed { output } => Ok(NodeResult::Command(
                        Command::default()
                            .with_update(MemberUpdate::Payload(output))
                            .with_goto(["complete"]),
                    )),
                    MemberOutcome::Failed { reason } => Ok(NodeResult::Command(
                        Command::default()
                            .with_update(MemberUpdate::Payload(reason))
                            .with_goto(["fail"]),
                    )),
                }
            }
        },
    );
    builder = builder.add_node(
        "complete",
        move |state: MemberState, _context: NodeContext| {
            let on_complete = on_complete.clone();
            async move {
                on_complete(state.payload.unwrap_or_default())
                    .await
                    .map_err(graph_err)?;
                Ok(NodeResult::Update(MemberUpdate::Noop))
            }
        },
    );
    builder = builder.add_node("fail", move |state: MemberState, _context: NodeContext| {
        let on_failed = on_failed.clone();
        async move {
            on_failed(state.payload.unwrap_or_default())
                .await
                .map_err(graph_err)?;
            Ok(NodeResult::Update(MemberUpdate::Noop))
        }
    });
    builder
        .add_node(
            "done",
            |_state: MemberState, _context: NodeContext| async move {
                Ok(NodeResult::Update(MemberUpdate::Noop))
            },
        )
        .add_edge("complete", "done")
        .add_edge("fail", "done")
        .set_entry("execute")
        .mark_command_routing("execute")
        .set_finish("done")
        .compile()
        .map_err(|error| anyhow::anyhow!("member graph compile failed: {error}"))
}

/// Structure-only view of the generic member execution graph.
pub fn member_graph_topology() -> Result<GraphTopology> {
    Ok(build_member_graph(
        || async {
            Ok(MemberOutcome::Completed {
                output: String::new(),
            })
        },
        |_| async { Ok(()) },
        |_| async { Ok(()) },
    )?
    .topology())
}

#[cfg(test)]
mod tests;
