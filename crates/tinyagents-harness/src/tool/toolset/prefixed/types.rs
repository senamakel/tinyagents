//! Type definitions for [`super::PrefixedToolSet`].

use std::sync::Arc;

use crate::tool::toolset::ToolSet;

/// [`ToolSet`] adaptor that prefixes every tool name an inner toolset
/// exposes, and strips the prefix again before delegating a call.
///
/// Mirrors Pydantic AI's `.prefixed('weather')` (`docs/runtime-comparison/
/// pydantic-ai.md` §3.4): the primary use is collision avoidance when
/// [`super::CombinedToolSet`] merges toolsets whose member names might
/// otherwise clash (two MCP servers each exposing a `search` tool, say).
pub struct PrefixedToolSet<State: Send + Sync, Ctx: Send + Sync> {
    pub(crate) inner: Arc<dyn ToolSet<State, Ctx>>,
    pub(crate) prefix: String,
}
