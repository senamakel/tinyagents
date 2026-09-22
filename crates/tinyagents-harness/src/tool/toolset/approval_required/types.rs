//! Type definitions for [`super::ApprovalRequiredToolSet`].

use std::sync::Arc;

use tinytools::Tool;

use crate::tool::toolset::ToolSet;

/// A predicate deciding whether a declared [`Tool`] needs explicit human
/// approval.
pub type ApprovalPredicate = Arc<dyn Fn(&dyn Tool) -> bool + Send + Sync>;

/// [`ToolSet`] adaptor that marks matching tools as requiring approval.
///
/// Mirrors Pydantic AI's `.approval_required(pred)`
/// (`docs/runtime-comparison/pydantic-ai.md` §3.4). Rather than inventing a
/// second boolean flag, this sets the vendor `tinytools` declaration
/// already meant for it —
/// [`tinytools::ToolAccess::approval_required`][tinytools::policy::ToolAccess]
/// via [`tinytools::ToolPolicy::requiring_approval`] — so a host already
/// consulting [`Tool::policy`] (for example
/// [`crate::middleware::library::ToolPolicyMiddleware`]) enforces this
/// without a second code path to keep in sync. This adaptor does not gate
/// execution itself; it only edits the declaration a host's approval gate
/// reads.
pub struct ApprovalRequiredToolSet<State: Send + Sync, Ctx: Send + Sync> {
    pub(crate) inner: Arc<dyn ToolSet<State, Ctx>>,
    pub(crate) predicate: ApprovalPredicate,
}
