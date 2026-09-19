//! Type definitions for the composable toolset module.
//!
//! [`ToolExposureExplanation`] is the audit payload: *why* a tool did not
//! reach the model unchanged this turn. It is additive on
//! [`crate::events::AgentEvent::ToolsFiltered`] so existing consumers of that
//! event keep working unchanged (`docs/sdk-gaps.md` §9 asks for exactly this
//! explainability, and `docs/runtime-comparison/pydantic-ai.md` §4 notes
//! TinyAgents' middleware-based filtering makes "why was this tool hidden"
//! hard to answer without it).

use serde::{Deserialize, Serialize};

/// Why a [`super::ToolSet`] adaptor changed or withheld one tool this turn.
///
/// Each variant corresponds to one adaptor in `crate::tool::toolset`. A
/// [`super::CombinedToolSet`] does not itself produce an explanation — it
/// only aggregates the ones its members already reported.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ToolExposureExplanation {
    /// [`super::FilteredToolSet`] dropped the tool: its predicate returned
    /// `false`.
    FilteredOut,
    /// [`super::RenamedToolSet`] exposed the tool under a different name.
    Renamed {
        /// The name the inner toolset declared.
        from: String,
        /// The name advertised to the model.
        to: String,
    },
    /// [`super::PrefixedToolSet`] exposed the tool with a name prefix
    /// applied (also used for collision avoidance when combining toolsets
    /// via [`super::CombinedToolSet`]).
    Prefixed {
        /// The name the inner toolset declared.
        from: String,
        /// The prefixed name advertised to the model.
        to: String,
    },
    /// [`super::PreparedToolSet`]'s per-step transform removed or rewrote
    /// the tool's declaration for this turn.
    Prepared,
    /// [`super::ApprovalRequiredToolSet`] marked the tool as requiring
    /// explicit human approval before it may execute.
    ApprovalRequired,
    /// [`super::ExternalToolSet`] advertises the tool for the model but its
    /// execution is deferred to the host — see
    /// [`crate::error::TinyAgentsError::CallDeferred`].
    Deferred,
    /// The tool was hidden entirely (never reached the model this turn) for
    /// a reason not covered by a more specific variant above.
    Hidden,
}
