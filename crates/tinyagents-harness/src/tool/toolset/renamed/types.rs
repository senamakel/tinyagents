//! Type definitions for [`super::RenamedToolSet`].

use std::collections::HashMap;
use std::sync::Arc;

use crate::tool::toolset::ToolSet;

/// [`ToolSet`] adaptor that renames tools per an explicit `old -> new` map.
///
/// Mirrors Pydantic AI's `.renamed({...})` (`docs/runtime-comparison/
/// pydantic-ai.md` §3.4). A tool whose declared name is not a key in the map
/// is exposed under its original name unchanged.
pub struct RenamedToolSet<State: Send + Sync, Ctx: Send + Sync> {
    pub(crate) inner: Arc<dyn ToolSet<State, Ctx>>,
    /// Declared (original) name -> advertised (renamed) name.
    pub(crate) renames: HashMap<String, String>,
}
