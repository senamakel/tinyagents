//! [`GraphLoopDriver`]: plugs the compiled loop's node bodies into
//! [`AgentHarness::invoke`] (and friends) via
//! [`tinyagents_harness::agent_loop::phases::LoopDriver`] +
//! [`AgentHarness::with_loop_driver`], selected by
//! [`tinyagents_harness::runtime::RunPolicy::execution`]`::Graph`.
//!
//! See the module doc on [`super`] ("`GraphLoopDriver` vs.
//! `compile_loop`/`LoopIter`") for why this drives
//! [`super::runtime::plan_node`]/`model_node`/`tools_node`/`settle_node`
//! directly over borrowed `&mut` state in a hand-rolled loop instead of
//! building a [`crate::CompiledGraph`].

use async_trait::async_trait;

use tinyagents_harness::agent_loop::phases::LoopDriver;
use tinyagents_harness::context::RunContext;
use tinyagents_harness::error::{Result, TinyAgentsError};
use tinyagents_harness::events::{AgentEvent, HarnessRunStatus};
use tinyagents_harness::ids::HarnessPhase;
use tinyagents_harness::middleware::AgentRun;
use tinyagents_harness::runtime::AgentHarness;
use tinyagents_harness::steering::PauseState;
use tinyinference_llm::message::Message;

use crate::command::{NodeResult, RouteTarget};

use super::runtime;
use super::types::{node, LoopState};

/// Drives [`AgentHarness::invoke`] through the same node bodies
/// [`super::compile_loop`] wires into a [`crate::CompiledGraph`], without
/// itself building one. Install with
/// [`AgentHarness::with_loop_driver`]`(Arc::new(GraphLoopDriver::new()))` and
/// [`tinyagents_harness::runtime::RunPolicy::execution`]`::Graph`.
///
/// Stateless — one instance can be shared (via `Arc`) across every harness
/// that wants the graph engine.
#[derive(Debug, Default, Clone, Copy)]
pub struct GraphLoopDriver;

impl GraphLoopDriver {
    /// Creates a driver. Stateless: nothing to configure.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl<State, Ctx> LoopDriver<State, Ctx> for GraphLoopDriver
where
    State: Send + Sync,
    Ctx: Send + Sync,
{
    async fn drive(
        &self,
        harness: &AgentHarness<State, Ctx>,
        state: &State,
        ctx: &mut RunContext<Ctx>,
        run: &mut AgentRun,
        status: &mut HarnessRunStatus,
        input: Vec<Message>,
        streaming: bool,
    ) -> Result<()> {
        // Mirrors `run_loop`'s own top-of-run bookkeeping (see that
        // function's docs on why the limit tracker restarts here rather
        // than at `RunContext::new`).
        ctx.limits.restart();
        runtime::reconcile_call_limits(ctx, harness.policy());
        ctx.streaming = streaming;

        let record = ctx.emit(AgentEvent::RunStarted {
            run_id: ctx.run_id().clone(),
            thread_id: ctx.thread_id().cloned(),
        });
        status.set_last_event(record.id);
        status.mark_running(HarnessPhase::Idle);

        harness.middleware().run_before_agent(ctx, state).await?;

        let mut loop_state = LoopState {
            messages: input,
            ..LoopState::default()
        };
        let mut current: &str = node::PLAN;

        let outcome = loop {
            // Keeps `run.messages` a running snapshot of the transcript as
            // of the start of each node call, so a node that errors or
            // interrupts mid-call (and so never hands `loop_state` back)
            // still leaves `run.messages` close to current — mirroring
            // `run_loop`'s "transcript survives every exit path" guarantee
            // as closely as this driver's by-value node contract allows.
            // The one gap: mutations a node makes to its *own* copy of
            // `loop_state.messages` before erroring/interrupting (for
            // example `plan_node`'s steering-injected message on a pause)
            // are not reflected until that node returns successfully.
            run.messages = loop_state.messages.clone();
            let result = match current {
                node::PLAN => runtime::plan_node(harness, ctx, loop_state).await,
                node::MODEL => runtime::model_node(harness, state, ctx, run, status, loop_state).await,
                node::TOOLS => runtime::tools_node(harness, state, ctx, run, status, loop_state).await,
                node::SETTLE => runtime::settle_node(harness, run, loop_state).await,
                other => {
                    break Err(TinyAgentsError::Validation(format!(
                        "GraphLoopDriver: unknown loop node `{other}`"
                    )));
                }
            };

            match result {
                Ok(NodeResult::Interrupt(interrupt)) if interrupt.id.ends_with("-steering-pause") => {
                    // No `CompiledGraph` is in play here, so there is no
                    // checkpoint to pause against — mirror the direct loop's
                    // steering pause instead: latch `run.paused` and finish
                    // this call with `Ok(())`, exactly like
                    // `run_loop`'s `LoopExit::Paused` handling. See the
                    // module doc on `super` for why this is not a resumable
                    // graph interrupt.
                    break Ok(Some(interrupt));
                }
                Ok(NodeResult::Interrupt(interrupt)) => {
                    // A `MiddlewareControl::Interrupt` (an approval gate, not
                    // a steering pause). The direct loop surfaces this as
                    // `TinyAgentsError::Interrupted` (see
                    // `agent_loop::run_loop`'s `apply_pending_control`), not
                    // a pause — matched here so `RunPolicy::execution ==
                    // Graph` behaves identically to `Direct` for this
                    // control. Only `compile_loop`/`LoopIter`'s real
                    // `CompiledGraph` upgrades this into a resumable graph
                    // interrupt (A5's "approvals surfacing as graph
                    // interrupts").
                    let message = interrupt
                        .payload
                        .get("message")
                        .and_then(|value| value.as_str())
                        .unwrap_or("interrupted")
                        .to_string();
                    break Err(TinyAgentsError::Interrupted {
                        node: interrupt.node.to_string(),
                        message,
                    });
                }
                Ok(NodeResult::Update(updated)) => {
                    // Every node body returns `Command`/`Interrupt` (see
                    // `runtime`'s node docs); a bare `Update` would mean the
                    // loop cannot determine where to go next.
                    let _ = updated;
                    break Err(TinyAgentsError::Validation(
                        "GraphLoopDriver: loop node returned an un-routed update".to_string(),
                    ));
                }
                Ok(NodeResult::Command(command)) => {
                    loop_state = match command.update {
                        Some(update) => update,
                        None => {
                            break Err(TinyAgentsError::Validation(
                                "GraphLoopDriver: loop node's command carried no update"
                                    .to_string(),
                            ));
                        }
                    };
                    let Some(target) = command.goto.first() else {
                        break Err(TinyAgentsError::Validation(
                            "GraphLoopDriver: loop node's command carried no route".to_string(),
                        ));
                    };
                    let RouteTarget::Node(node_id) = target else {
                        break Err(TinyAgentsError::Validation(
                            "GraphLoopDriver: loop node routed via `Send`, which this driver \
                             does not support"
                                .to_string(),
                        ));
                    };
                    if node_id.as_str() == crate::builder::END {
                        break Ok(None);
                    }
                    current = match node_id.as_str() {
                        node::PLAN => node::PLAN,
                        node::MODEL => node::MODEL,
                        node::TOOLS => node::TOOLS,
                        node::SETTLE => node::SETTLE,
                        other => {
                            break Err(TinyAgentsError::Validation(format!(
                                "GraphLoopDriver: unknown loop node `{other}`"
                            )));
                        }
                    };
                    continue;
                }
                Err(error) => break Err(error),
            }
        };

        status.mark_running(HarnessPhase::Middleware);
        harness.middleware().run_after_agent(ctx, state, run).await?;

        // `status.mark_completed`/`mark_interrupted`/`mark_failed` and (on
        // error) `AgentEvent::RunFailed` are applied centrally by
        // `agent_loop::entry::drive_collecting` after this call returns,
        // identically for the direct loop and this driver — see that
        // function's doc comment. This driver only emits the terminal event
        // that (like the direct loop's `run_loop_body`) is its own
        // responsibility to raise: `RunCompleted` on a clean finish, or
        // latching `run.paused` (mirroring a steering pause) on an
        // interrupt.
        match outcome {
            Ok(None) => {
                let record = ctx.emit(AgentEvent::RunCompleted {
                    run_id: ctx.run_id().clone(),
                });
                status.set_last_event(record.id);
                Ok(())
            }
            Ok(Some(interrupt)) => {
                let reason = interrupt
                    .payload
                    .get("reason")
                    .or_else(|| interrupt.payload.get("message"))
                    .and_then(|value| value.as_str())
                    .map(str::to_string);
                let record = ctx.emit(AgentEvent::ControlApplied {
                    control: "paused".to_string(),
                    detail: reason
                        .clone()
                        .unwrap_or_else(|| format!("paused at node `{}`", interrupt.node)),
                });
                status.set_last_event(record.id);
                run.paused = Some(PauseState {
                    reason,
                    paused_at_checkpoint: 0,
                });
                Ok(())
            }
            Err(error) => Err(error),
        }
    }
}
