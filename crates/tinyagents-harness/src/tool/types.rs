//! Harness-owned execution context for canonical `tinytools::Tool` calls.
//!
//! Tool vocabulary belongs to `tinytools`. This module deliberately contains
//! only the narrow bridge from a live harness run to `ToolRunContext`; it does
//! not redeclare a tool, result, policy, timeout, or call type.

use crate::cancel::CancellationToken;
use crate::context::RunContext;
use crate::events::EventSink;
use crate::ids::{RunId, ThreadId};

/// The run facts a canonical tool may inspect through
/// [`tinytools::ToolRunContext`].
///
/// Cancellation, event emission, and run identity deliberately remain owned by
/// the harness. Recursive tools that need an actual child `RunContext` use the
/// explicit dispatch seam at registration rather than widening TinyTools'
/// portable context trait.
#[derive(Clone)]
pub struct ToolExecutionContext {
    /// Run that invoked the tool.
    pub run_id: RunId,
    /// Caller thread id, when the parent run is threaded.
    pub thread_id: Option<ThreadId>,
    /// Caller recursion depth.
    pub depth: usize,
    /// Maximum output tokens requested for each model turn in the caller run.
    pub max_turn_output_tokens: Option<u32>,
    /// Shared event sink for nested-run observability.
    pub events: EventSink,
    /// The caller run's cancellation token.
    pub cancellation: CancellationToken,
    /// Whether the caller is driven through the streaming loop path.
    pub streaming: bool,
    /// The isolated workspace/sandbox available to the tool.
    pub workspace: Option<tinytools::WorkspaceDescriptor>,
}

impl ToolExecutionContext {
    /// Captures the non-generic, tool-visible parts of a live run.
    pub fn from_run_context<Ctx>(ctx: &RunContext<Ctx>) -> Self {
        Self {
            run_id: ctx.config.run_id.clone(),
            thread_id: ctx.config.thread_id.clone(),
            depth: ctx.depth(),
            max_turn_output_tokens: ctx.config.max_turn_output_tokens,
            events: ctx.events.clone(),
            cancellation: ctx.cancellation.clone(),
            streaming: ctx.streaming,
            workspace: ctx.workspace.clone(),
        }
    }

    /// Attaches an isolated workspace descriptor the tool may operate in.
    #[must_use]
    pub fn with_workspace(mut self, workspace: tinytools::WorkspaceDescriptor) -> Self {
        self.workspace = Some(workspace);
        self
    }
}

impl tinytools::ToolRunContext for ToolExecutionContext {
    fn workspace(&self) -> Option<&tinytools::WorkspaceDescriptor> {
        self.workspace.as_ref()
    }

    fn thread_id(&self) -> Option<&str> {
        self.thread_id.as_ref().map(ThreadId::as_str)
    }

    fn max_turn_output_tokens(&self) -> Option<u32> {
        self.max_turn_output_tokens
    }
}
