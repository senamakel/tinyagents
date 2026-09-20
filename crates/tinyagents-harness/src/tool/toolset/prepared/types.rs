//! Type definitions for [`super::PreparedToolSet`].

use std::sync::Arc;

use tinyinference_llm::tool::ToolSchema;

use crate::context::RunContext;
use crate::tool::toolset::ToolSet;

/// A per-step schema transform: given the run context and the declared
/// schemas an inner toolset exposes, returns the schemas that should
/// actually be advertised this turn.
///
/// Omitting a schema from the returned `Vec` hides that tool for this turn
/// (it becomes uncallable through the [`super::PreparedToolSet`], mirroring
/// Pydantic AI's per-tool `prepare` returning `None`); editing a returned
/// schema's `description`/`parameters` rewrites the declaration the model
/// sees without touching the inner toolset's own tool. A schema whose name
/// was never in the input is ignored — this adaptor transforms an existing
/// declared set, it does not mint new tools.
pub type SchemaTransform<Ctx> =
    Arc<dyn Fn(&RunContext<Ctx>, Vec<ToolSchema>) -> Vec<ToolSchema> + Send + Sync>;

/// [`ToolSet`] adaptor that applies a caller-supplied, per-step
/// [`SchemaTransform`] to an inner toolset's declared schemas.
///
/// This is the same seam [`crate::tool::SchemaPreparation`] occupies for
/// provider projection, generalised to a caller-supplied closure that is
/// re-consulted every turn (so it may read [`RunContext`] — the run's tags,
/// depth, or anything else threaded through `Ctx`) rather than being fixed
/// at construction. Mirrors Pydantic AI's `.prepared(fn)`
/// (`docs/runtime-comparison/pydantic-ai.md` §3.4).
pub struct PreparedToolSet<State: Send + Sync, Ctx: Send + Sync> {
    pub(crate) inner: Arc<dyn ToolSet<State, Ctx>>,
    pub(crate) transform: SchemaTransform<Ctx>,
}
