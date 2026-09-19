//! [`StreamMode`] filtering for [`GraphEvent`]s, and [`StreamProjection`] — a
//! cursor-ordered fold of both [`GraphEventEnvelope`]s and harness
//! [`AgentEvent`]s into the three views a UI actually renders: messages, tool
//! calls, and subagent activity.
//!
//! Graph events narrate *structure* (which node/task ran, which checkpoint
//! saved); harness events narrate *content* (what the model said, which tool
//! ran, which subagent was invoked). A consumer watching a recursive run —
//! a graph whose nodes drive harness agent loops, some of which spawn
//! subagents or embed subgraphs — wants both folded into one ordered view.
//! [`StreamProjection`] is that fold. Its [`StreamProjection::cursor`] is a
//! single monotonic counter shared by every view, so a consumer that
//! attaches after a run has already produced output can request only what it
//! missed with [`StreamProjection::since`] instead of re-reading everything.

use tinyagents_harness::events::AgentEvent;
use tinyagents_harness::ids::{CallId, RunId};
use tinyinference_llm::message::MessageDelta;

use super::{GraphEvent, GraphEventEnvelope, StreamMode};

/// Returns `true` when `event` should be delivered to a consumer subscribed
/// to `modes`.
///
/// [`GraphEvent::mode`] gives the single narrow mode most event kinds belong
/// to; the run/step lifecycle events that have none (`RunStarted`,
/// `StepStarted`, …) are debug-only detail and pass only when
/// [`StreamMode::Debug`] is active — mirroring
/// [`tinyagents_harness::stream::project_event_for_modes`]'s treatment of its
/// own lifecycle events.
pub fn project_graph_event(event: &GraphEvent, modes: &[StreamMode]) -> bool {
    match event.mode() {
        Some(mode) => modes.contains(&mode) || modes.contains(&StreamMode::Debug),
        None => modes.contains(&StreamMode::Debug),
    }
}

// ---------------------------------------------------------------------------
// StreamProjection
// ---------------------------------------------------------------------------

/// One item in a [`StreamProjection`] view, tagged with the projection's
/// monotonic [`StreamProjection::cursor`] value at the moment it was folded
/// in.
#[derive(Clone, Debug, PartialEq)]
pub struct Cursored<T> {
    /// This item's position in the projection's global fold order.
    pub cursor: u64,
    /// The projected value.
    pub value: T,
}

/// One entry in [`StreamProjection::messages`]: an assistant message
/// fragment attributed to its run and model call.
#[derive(Clone, Debug, PartialEq)]
pub struct MessageEntry {
    /// The run that produced this fragment.
    pub run_id: RunId,
    /// The model call this fragment belongs to.
    pub call_id: CallId,
    /// The incremental text/reasoning/tool-call fragment.
    pub delta: MessageDelta,
}

/// Lifecycle phase of a tool call, folded from [`AgentEvent::ToolStarted`] /
/// [`AgentEvent::ToolCompleted`] / [`AgentEvent::ToolFailed`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolCallPhase {
    /// [`AgentEvent::ToolStarted`].
    Started,
    /// [`AgentEvent::ToolCompleted`], successful (`error` was `None`).
    Completed,
    /// [`AgentEvent::ToolCompleted`] with `error: Some(_)`, or
    /// [`AgentEvent::ToolFailed`].
    Failed {
        /// The failure message.
        error: String,
    },
}

/// One entry in [`StreamProjection::tool_calls`].
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCallEntry {
    /// Correlates with the call's `Started`/terminal pair.
    pub call_id: CallId,
    /// The tool's name.
    pub tool_name: String,
    /// The call's current lifecycle phase.
    pub phase: ToolCallPhase,
}

/// Lifecycle phase of a subagent or subgraph activation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubagentPhase {
    /// Started (harness [`AgentEvent::SubAgentStarted`] or graph
    /// [`GraphEvent::SubgraphStarted`]).
    Started,
    /// Finished (harness [`AgentEvent::SubAgentCompleted`] or graph
    /// [`GraphEvent::SubgraphCompleted`]).
    Completed,
}

/// One entry in [`StreamProjection::subagents`].
#[derive(Clone, Debug, PartialEq)]
pub struct SubagentEntry {
    /// The sub-agent's name (harness activations) or hosting node id (graph
    /// subgraph activations).
    pub name: String,
    /// The activation's current lifecycle phase.
    pub phase: SubagentPhase,
}

/// Folds a run's [`GraphEventEnvelope`]s and harness [`AgentEvent`]s into
/// three consumer-facing views (`messages`, `tool_calls`, `subagents`) under
/// one monotonic cursor.
///
/// Feed events as they arrive with [`Self::fold_graph_event`] /
/// [`Self::fold_agent_event`], in the order they were emitted (across both
/// sources merged by real time — the projection does not reorder). A
/// consumer that attaches late replays with [`Self::since`] instead of
/// re-reading the full history.
#[derive(Clone, Debug, Default)]
pub struct StreamProjection {
    next_cursor: u64,
    /// Assistant message fragments, in fold order.
    pub messages: Vec<Cursored<MessageEntry>>,
    /// Tool-call lifecycle entries, in fold order. A call's `Started` and
    /// terminal phase are two separate entries sharing `call_id`, not one
    /// mutated in place, so [`Self::since`] replay never has to reconstruct
    /// history a consumer already saw.
    pub tool_calls: Vec<Cursored<ToolCallEntry>>,
    /// Subagent/subgraph lifecycle entries, in fold order (same
    /// two-entries-per-activation shape as `tool_calls`).
    pub subagents: Vec<Cursored<SubagentEntry>>,
}

impl StreamProjection {
    /// Creates an empty projection.
    pub fn new() -> Self {
        Self::default()
    }

    /// The next cursor value that will be assigned. Equal to the total number
    /// of items folded in across every view so far.
    pub fn cursor(&self) -> u64 {
        self.next_cursor
    }

    fn next(&mut self) -> u64 {
        let cursor = self.next_cursor;
        self.next_cursor += 1;
        cursor
    }

    /// Folds one graph event. Only [`GraphEvent::SubgraphStarted`] /
    /// [`GraphEvent::SubgraphCompleted`] currently project onto a view (as
    /// `subagents`); every other kind is structural and is not part of the
    /// three content views this projection exposes (subscribe to the raw
    /// envelope stream directly for those).
    pub fn fold_graph_event(&mut self, envelope: &GraphEventEnvelope) {
        match &envelope.event {
            GraphEvent::SubgraphStarted { node, .. } => {
                self.push_subagent(node.to_string(), SubagentPhase::Started);
            }
            GraphEvent::SubgraphCompleted { node, .. } => {
                self.push_subagent(node.to_string(), SubagentPhase::Completed);
            }
            _ => {}
        }
    }

    /// Folds one harness agent event.
    pub fn fold_agent_event(&mut self, event: &AgentEvent) {
        match event {
            AgentEvent::ModelDelta {
                run_id,
                call_id,
                delta,
            } => {
                let cursor = self.next();
                self.messages.push(Cursored {
                    cursor,
                    value: MessageEntry {
                        run_id: run_id.clone(),
                        call_id: call_id.clone(),
                        delta: delta.clone(),
                    },
                });
            }
            AgentEvent::ToolStarted { call_id, tool_name } => {
                self.push_tool_call(call_id.clone(), tool_name.clone(), ToolCallPhase::Started);
            }
            AgentEvent::ToolCompleted {
                call_id,
                tool_name,
                error,
                ..
            } => {
                let phase = match error {
                    Some(error) => ToolCallPhase::Failed {
                        error: error.clone(),
                    },
                    None => ToolCallPhase::Completed,
                };
                self.push_tool_call(call_id.clone(), tool_name.clone(), phase);
            }
            AgentEvent::ToolFailed {
                call_id,
                tool_name,
                error,
                ..
            } => {
                self.push_tool_call(
                    call_id.clone(),
                    tool_name.clone(),
                    ToolCallPhase::Failed {
                        error: error.clone(),
                    },
                );
            }
            AgentEvent::SubAgentStarted { name, .. } => {
                self.push_subagent(name.clone(), SubagentPhase::Started);
            }
            AgentEvent::SubAgentCompleted { name, .. } => {
                self.push_subagent(name.clone(), SubagentPhase::Completed);
            }
            _ => {}
        }
    }

    fn push_tool_call(&mut self, call_id: CallId, tool_name: String, phase: ToolCallPhase) {
        let cursor = self.next();
        self.tool_calls.push(Cursored {
            cursor,
            value: ToolCallEntry {
                call_id,
                tool_name,
                phase,
            },
        });
    }

    fn push_subagent(&mut self, name: String, phase: SubagentPhase) {
        let cursor = self.next();
        self.subagents.push(Cursored {
            cursor,
            value: SubagentEntry { name, phase },
        });
    }

    /// Returns every item across all three views with `cursor > since`, each
    /// still tagged with its view, in cursor order — what a late-attaching
    /// consumer replays instead of re-reading the full projection.
    pub fn since(&self, since: u64) -> Vec<ProjectedSince> {
        let mut items: Vec<ProjectedSince> = self
            .messages
            .iter()
            .filter(|item| item.cursor > since)
            .map(|item| ProjectedSince::Message(item.clone()))
            .chain(
                self.tool_calls
                    .iter()
                    .filter(|item| item.cursor > since)
                    .map(|item| ProjectedSince::ToolCall(item.clone())),
            )
            .chain(
                self.subagents
                    .iter()
                    .filter(|item| item.cursor > since)
                    .map(|item| ProjectedSince::Subagent(item.clone())),
            )
            .collect();
        items.sort_by_key(ProjectedSince::cursor);
        items
    }
}

/// One replayed item from [`StreamProjection::since`], tagged by which view
/// it belongs to.
#[derive(Clone, Debug, PartialEq)]
pub enum ProjectedSince {
    /// A [`StreamProjection::messages`] entry.
    Message(Cursored<MessageEntry>),
    /// A [`StreamProjection::tool_calls`] entry.
    ToolCall(Cursored<ToolCallEntry>),
    /// A [`StreamProjection::subagents`] entry.
    Subagent(Cursored<SubagentEntry>),
}

impl ProjectedSince {
    /// The item's cursor value, regardless of which view it came from.
    pub fn cursor(&self) -> u64 {
        match self {
            ProjectedSince::Message(item) => item.cursor,
            ProjectedSince::ToolCall(item) => item.cursor,
            ProjectedSince::Subagent(item) => item.cursor,
        }
    }
}

/// Distinct node ids observed in a set of graph events, in first-seen order.
///
/// Small helper used by tests asserting namespace/task attribution; kept here
/// (rather than duplicated per test) since it is generic enough to be useful
/// beyond this module's own tests.
#[cfg(test)]
pub(crate) fn distinct_nodes(envelopes: &[GraphEventEnvelope]) -> Vec<NodeId> {
    let mut seen = HashSet::new();
    let mut order = Vec::new();
    for envelope in envelopes {
        let node = match &envelope.event {
            GraphEvent::NodeStarted { node, .. } => node.clone(),
            _ => continue,
        };
        if seen.insert(node.clone()) {
            order.push(node);
        }
    }
    order
}

#[cfg(test)]
mod test;
