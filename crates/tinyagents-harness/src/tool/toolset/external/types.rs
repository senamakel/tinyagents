//! Type definitions for [`super::ExternalToolSet`].

use tinytools::ToolSpec;

/// [`super::ToolSet`][crate::tool::ToolSet] adaptor for schema-only tools
/// **executed by the host**, not this process.
///
/// Mirrors Pydantic AI's `defer_loading`/deferred-tools model
/// (`docs/runtime-comparison/pydantic-ai.md` §3.4, `pi.md` §4.1's
/// transcript-carried tool-loadout changes). It advertises
/// [`Self::schemas`] to the model like any other toolset, but never runs one
/// of them locally.
///
/// # Host-integration contract
///
/// Every advertised tool declaration carries the same
/// [`crate::tool::deferred::ExternalToolMarker`] host extension as
/// [`crate::tool::ToolRegistry::register_external`]'s declaration. Bridged
/// into a harness's registry with
/// [`crate::tool::toolset::ToolSetDispatchBridge`] and
/// [`crate::runtime::AgentHarness::register_tool_dispatch`] (see
/// [`crate::runtime::AgentHarness::with_toolset`]'s doc comment), an
/// admitted call is recognized proactively — before
/// [`ToolSet::call`][crate::tool::ToolSet::call] is ever reached — and the
/// agent loop's ordinary A2 deferred-call exit
/// ([`crate::tool::DeferredToolRequests`], surfaced as `AgentRun::deferred`,
/// or resolved inline through a registered
/// [`crate::tool::DeferredToolHandler`]) takes over from there, identically
/// to a tool registered through `register_external`.
///
/// A caller that reaches [`ToolSet::call`][crate::tool::ToolSet::call]
/// directly instead — driving the toolset chain itself rather than through
/// [`crate::runtime::AgentHarness`]'s default loop, or through some other
/// [`crate::tool::ToolDispatch`] that does not preserve the marker — still
/// gets the same typed
/// [`crate::error::TinyAgentsError::CallDeferred`] signal to catch, execute
/// the call out of process, and resume with the result appended to the
/// transcript as an ordinary tool result.
pub struct ExternalToolSet {
    pub(crate) schemas: Vec<ToolSpec>,
}
