//! Harness-owned execution context for canonical `tinytools::Tool` calls.
//!
//! Tool vocabulary belongs to `tinytools`. This module deliberately contains
//! only the narrow bridge from a live harness run to `ToolRunContext`; it does
//! not redeclare a tool, result, policy, timeout, or call type.

use std::any::Any;
use std::sync::Arc;

use crate::cancel::CancellationToken;
use crate::context::RunContext;
use crate::events::{AgentEvent, EventSink};
use crate::ids::{CallId, RunId, ThreadId};
use crate::store::namespaced::NamespacedStore;

/// The run facts a canonical tool may inspect through
/// [`tinytools::ToolRunContext`].
///
/// Cancellation, event emission, and run identity deliberately remain owned by
/// the harness. Recursive tools that need an actual child `RunContext` use the
/// explicit dispatch seam at registration rather than widening TinyTools'
/// portable context trait.
///
/// # Reaching this from a `tinytools::Tool`
///
/// A canonical tool receives `Option<&dyn ToolRunContext>`, whose typed
/// methods cover only the portable facts (workspace, thread id, output cap).
/// The rest of this struct — the call id, the store, the typed state view,
/// [`Self::custom`] — is reachable by downcasting the erased host extension
/// (B1):
///
/// ```ignore
/// let harness = context
///     .and_then(tinytools::ToolRunContext::host_extension)
///     .and_then(|any| any.downcast_ref::<ToolExecutionContext>());
/// ```
#[derive(Clone)]
pub struct ToolExecutionContext {
    /// Run that invoked the tool.
    pub run_id: RunId,
    /// The id of the tool call being executed — the same id the transcript's
    /// tool-result row and the `ToolStarted`/`ToolCompleted` events carry, so
    /// a tool can correlate anything it records or emits with the call.
    pub call_id: CallId,
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
    /// The run's hierarchical long-term store, when the host attached one
    /// with [`RunContext::with_namespaced_store`]; `None` otherwise.
    pub store: Option<Arc<dyn NamespacedStore>>,
    /// Type-erased application state snapshot, when the host attached one
    /// with [`RunContext::with_state_view`]. Read it through [`Self::state`].
    pub state_view: Option<Arc<dyn Any + Send + Sync>>,
}

impl ToolExecutionContext {
    /// Captures the non-generic, tool-visible parts of a live run for the
    /// tool call `call_id`.
    pub fn from_run_context<Ctx>(ctx: &RunContext<Ctx>, call_id: CallId) -> Self {
        Self {
            run_id: ctx.config.run_id.clone(),
            call_id,
            thread_id: ctx.config.thread_id.clone(),
            depth: ctx.depth(),
            max_turn_output_tokens: ctx.config.max_turn_output_tokens,
            events: ctx.events.clone(),
            cancellation: ctx.cancellation.clone(),
            streaming: ctx.streaming,
            workspace: ctx.workspace.clone(),
            store: ctx.namespaced_store.clone(),
            state_view: ctx.state_view.clone(),
        }
    }

    /// Attaches an isolated workspace descriptor the tool may operate in.
    #[must_use]
    pub fn with_workspace(mut self, workspace: tinytools::WorkspaceDescriptor) -> Self {
        self.workspace = Some(workspace);
        self
    }

    /// The application state the host attached, as `S`.
    ///
    /// `None` when no view was attached **or** when it was attached as a
    /// different type — a mismatch is never a panic, matching how
    /// [`crate::context::StateUpdate::apply`] treats a type it was not built
    /// for.
    #[must_use]
    pub fn state<S: 'static>(&self) -> Option<&S> {
        self.state_view
            .as_deref()
            .and_then(|view| view.downcast_ref::<S>())
    }

    /// Emits an [`AgentEvent::Custom`] carrying `payload` on the run's event
    /// stream, correlated to this call.
    ///
    /// This is the tool's structured progress channel: it lands on the same
    /// ordered stream as the loop's own events (between this call's
    /// `ToolStarted` and `ToolCompleted`), so a UI or journal sees it in
    /// context. The harness attaches no meaning to the payload.
    pub fn custom(&self, payload: serde_json::Value) {
        self.events.emit(AgentEvent::Custom {
            call_id: Some(self.call_id.clone()),
            payload,
        });
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

    fn host_extension(&self) -> Option<&(dyn Any + Send + Sync)> {
        Some(self)
    }
}
