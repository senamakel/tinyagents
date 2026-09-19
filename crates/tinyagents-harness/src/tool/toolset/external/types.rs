//! Type definitions for [`super::ExternalToolSet`].

use tinytools::ToolSpec;

/// [`super::ToolSet`][crate::tool::ToolSet] adaptor for schema-only tools
/// **executed by the host**, not this process.
///
/// Mirrors Pydantic AI's `defer_loading`/deferred-tools model
/// (`docs/runtime-comparison/pydantic-ai.md` §3.4, `pi.md` §4.1's
/// transcript-carried tool-loadout changes). It advertises
/// [`Self::schemas`] to the model like any other toolset, but
/// [`ToolSet::call`][crate::tool::ToolSet::call] never runs them locally —
/// it always fails with
/// [`crate::error::TinyAgentsError::CallDeferred`].
///
/// # Host-integration contract
///
/// This branch's agent loop has no built-in "pause the run, hand the call to
/// the host, resume with the result" exit mechanism yet (there is no
/// existing `Deferred`/deferred-call handshake to integrate with in
/// `crate::agent_loop` beyond
/// [`tinytools::ToolExposure::Deferred`]'s discovery bridge, which is a
/// different concept — it defers *advertising* a tool the harness can
/// still execute, not *executing* one only the host can run). Until that
/// exists, a caller wiring an `ExternalToolSet` into a run must catch
/// [`crate::error::TinyAgentsError::CallDeferred`] itself — for example from
/// a custom [`crate::tool::ToolDispatch`] or by driving the toolset chain
/// directly rather than through [`crate::runtime::AgentHarness`]'s default
/// loop — execute the call out of process, and resume the run by appending
/// an ordinary tool result to the transcript, exactly as the loop already
/// does for a locally executed call. The variant exists so that failure mode
/// is a typed, matchable error instead of an opaque one.
pub struct ExternalToolSet {
    pub(crate) schemas: Vec<ToolSpec>,
}
