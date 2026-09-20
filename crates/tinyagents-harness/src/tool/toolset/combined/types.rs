//! Type definitions for [`super::CombinedToolSet`].

use std::sync::Arc;

use crate::tool::toolset::ToolSet;

/// [`ToolSet`] adaptor that merges several toolsets into one.
///
/// Mirrors Pydantic AI's `CombinedToolset` (`docs/runtime-comparison/
/// pydantic-ai.md` §3.4). [`super::CombinedToolSet::tools`] concatenates
/// every member's tools in member order; [`super::CombinedToolSet::call`]
/// dispatches to the **first** member (in registration order) that
/// currently exposes the requested name. Name collisions across members are
/// the caller's responsibility to avoid — wrap a member in
/// [`super::PrefixedToolSet`] first when its names might clash with
/// another's.
pub struct CombinedToolSet<State: Send + Sync, Ctx: Send + Sync> {
    pub(crate) members: Vec<Arc<dyn ToolSet<State, Ctx>>>,
}
