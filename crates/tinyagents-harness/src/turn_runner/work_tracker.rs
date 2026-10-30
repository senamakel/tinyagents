//! Folds generic agent progress into balanced work-liveness signals.

use std::collections::HashSet;

use crate::progress::AgentProgress;

use super::TurnSignal;

/// Tracks whether model or tool work is currently open for a turn.
///
/// The tracker emits [`TurnSignal::WorkStarted`] when progress first opens
/// work and [`TurnSignal::WorkFinished`] when the last open operation closes.
/// This lets a turn runner distinguish a quiet but healthy provider/tool call
/// from an idle turn without depending on a channel-specific event format.
#[derive(Debug, Default)]
pub struct WorkTracker {
    model_call_open: bool,
    open_tool_calls: HashSet<String>,
}

impl WorkTracker {
    fn open(&self) -> bool {
        self.model_call_open || !self.open_tool_calls.is_empty()
    }

    /// Folds one progress event into the open-work state.
    ///
    /// `IterationStarted` closes tool calls whose completion may have been
    /// dropped, while a parent `ToolCallStarted` closes the model call that
    /// requested it. Subagent model completions leave parent work unchanged.
    pub fn observe(&mut self, event: &AgentProgress) -> Option<TurnSignal> {
        let was_open = self.open();
        match event {
            AgentProgress::IterationStarted { .. } => {
                self.open_tool_calls.clear();
                self.model_call_open = true;
            }
            AgentProgress::ModelCallCompleted {
                subagent_task_id: None,
                ..
            } => self.model_call_open = false,
            AgentProgress::ToolCallStarted { call_id, .. } => {
                self.model_call_open = false;
                self.open_tool_calls.insert(call_id.clone());
            }
            AgentProgress::ToolCallCompleted { call_id, .. } => {
                self.open_tool_calls.remove(call_id);
            }
            AgentProgress::TurnCompleted { .. } => {
                self.open_tool_calls.clear();
                self.model_call_open = false;
            }
            _ => {}
        }
        match (was_open, self.open()) {
            (false, true) => Some(TurnSignal::WorkStarted),
            (true, false) => Some(TurnSignal::WorkFinished),
            _ => None,
        }
    }
}

#[cfg(test)]
#[path = "work_tracker_tests.rs"]
mod tests;
