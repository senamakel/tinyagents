//! Type definitions for [`super::FilteredToolSet`].

use std::sync::Arc;

use tinytools::Tool;

use crate::tool::toolset::ToolSet;

/// A predicate deciding whether a declared [`Tool`] should be exposed.
pub type ToolFilterPredicate = Arc<dyn Fn(&dyn Tool) -> bool + Send + Sync>;

/// [`super::ToolSet`][crate::tool::ToolSet] adaptor that keeps only the
/// tools an inner toolset exposes for which `predicate` returns `true`.
///
/// Mirrors Pydantic AI's `.filtered(pred)` (`docs/runtime-comparison/
/// pydantic-ai.md` §3.4). A call for a tool the predicate rejects fails with
/// [`crate::error::TinyAgentsError::ToolNotFound`], exactly like an
/// unregistered name — a filtered-out tool must not be reachable just
/// because a model guesses or is told its name.
pub struct FilteredToolSet<State: Send + Sync, Ctx: Send + Sync> {
    pub(crate) inner: Arc<dyn ToolSet<State, Ctx>>,
    pub(crate) predicate: ToolFilterPredicate,
}
