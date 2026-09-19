//! Capability bundles (gap G3, `docs/runtime-comparison/plan.md`): instructions
//! + toolset + middleware + model defaults + exposure + `defer_loading`,
//! composed as one named unit instead of wired separately.
//!
//! `docs/runtime-comparison/pydantic-ai.md` §4 "Capabilities as the unit of
//! composition" is the design source: Pydantic AI's v2 `AbstractCapability` is
//! a bigger idea than middleware alone — it is what a "skill" or "plugin" is.
//! [`Capability`] is that bundle for TinyAgents,
//! [`crate::runtime::AgentHarness::with_capability`] is the harness-side
//! consumer, and [`CapabilityToolSet`]/[`LoadCapabilityTool`] implement the
//! `defer_loading` / `load_capability` on-demand loading mechanic.
//!
//! # Where this type lives, and why
//!
//! `Capability<State, Ctx>` composes [`crate::tool::toolset::ToolSet<State,
//! Ctx>`] and [`crate::middleware::Middleware<State, Ctx>`] trait objects, both
//! native to this crate. It cannot live in `tinyagents-definition` (the
//! lower crate both `tinyagents-harness` and `tinyagents-registry` already
//! depend on): `tinyagents-definition` has zero dependency on
//! `tinyagents-harness` by design — that is what keeps the dependency graph
//! acyclic — so a definition-crate `Capability` could not name a `ToolSet` or
//! `Middleware` type without creating one. `tinyagents-registry` *does*
//! already depend on `tinyagents-harness`, so it can (and does) reference this
//! type — see `tinyagents_registry::CapabilityRegistry::register_capability`
//! — but `CapabilityRegistry<State>` itself carries no `Ctx` type parameter
//! (none of its other stored kinds — models, tools, graphs, agents — need
//! one either, since they are all `Ctx`-free harness/tinytools types), so
//! registry storage for this specific, `Ctx`-generic bundle goes through
//! type-erased `Box<dyn Any>` storage instead of adding a `Ctx` parameter to
//! the whole registry for one feature. See that method's doc comment for the
//! erasure mechanics.

mod types;

#[cfg(test)]
mod test;

use std::sync::Arc;

use serde_json::Value;
use tinytools::ToolExposure;

use crate::error::{Result, TinyAgentsError};
use crate::middleware::Middleware;
use crate::tool::toolset::ToolSet;

pub use types::{
    Capability, CapabilityToolSet, LOAD_CAPABILITY_TOOL_NAME, LoadCapabilityTool,
    ModelRequestDefaults,
};
pub(crate) use types::{CapabilitySpec, ModelDefaultsSpec};

impl<State: Send + Sync, Ctx: Send + Sync> Capability<State, Ctx> {
    /// Creates a capability with `name` and every optional field unset:
    /// no instructions, no toolset, no middleware, no model defaults,
    /// [`ToolExposure::Direct`] exposure, and `defer_loading: false`.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            instructions: None,
            toolset: None,
            middleware: Vec::new(),
            model_defaults: None,
            exposure: ToolExposure::Direct,
            defer_loading: false,
        }
    }

    /// Sets the instructions contributed to the system prompt while this
    /// capability is loaded. Returns `self` for chaining.
    #[must_use]
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Sets the toolset this capability contributes. Returns `self` for
    /// chaining.
    #[must_use]
    pub fn with_toolset(mut self, toolset: Arc<dyn ToolSet<State, Ctx>>) -> Self {
        self.toolset = Some(toolset);
        self
    }

    /// Appends one middleware instance, applied in declaration order.
    /// Returns `self` for chaining.
    #[must_use]
    pub fn with_middleware(mut self, middleware: Arc<dyn Middleware<State, Ctx>>) -> Self {
        self.middleware.push(middleware);
        self
    }

    /// Sets the model-request defaults applied when this capability is
    /// installed. Returns `self` for chaining.
    #[must_use]
    pub fn with_model_defaults(mut self, defaults: ModelRequestDefaults) -> Self {
        self.model_defaults = Some(defaults);
        self
    }

    /// Sets the [`ToolExposure`] applied to every tool this capability
    /// contributes. Returns `self` for chaining.
    #[must_use]
    pub fn with_exposure(mut self, exposure: ToolExposure) -> Self {
        self.exposure = exposure;
        self
    }

    /// Marks this capability as loaded on demand only, via
    /// [`LOAD_CAPABILITY_TOOL_NAME`]. Returns `self` for chaining.
    #[must_use]
    pub fn with_defer_loading(mut self, defer_loading: bool) -> Self {
        self.defer_loading = defer_loading;
        self
    }

    /// Builds a capability from a JSON spec: `{"name", "instructions"?,
    /// "exposure"? ("direct"|"deferred"|"hidden", default "direct"),
    /// "defer_loading"? (default `false`), "model_defaults"?
    /// {"response_format"?, "fallback_models"?}}`.
    ///
    /// Only the declarative fields round-trip through JSON (see
    /// [`Self::to_spec`]): the built capability's `toolset` and `middleware`
    /// are always empty, since neither can be represented in JSON. A host
    /// parsing a `.rag` `capability "name"` reference (or any other
    /// JSON-declared capability) wires those in afterward with
    /// [`Self::with_toolset`]/[`Self::with_middleware`] before installing it
    /// via [`crate::runtime::AgentHarness::with_capability`].
    ///
    /// # Errors
    ///
    /// Returns [`TinyAgentsError::Capability`] if `value` does not match the
    /// spec shape, or if `name` is missing or blank.
    pub fn from_spec(value: Value) -> Result<Self> {
        let spec: CapabilitySpec = serde_json::from_value(value).map_err(|err| {
            TinyAgentsError::Capability(format!("invalid capability spec: {err}"))
        })?;
        if spec.name.trim().is_empty() {
            return Err(TinyAgentsError::Capability(
                "capability spec is missing a non-blank `name`".to_string(),
            ));
        }
        let model_defaults = spec.model_defaults.map(|defaults| ModelRequestDefaults {
            default_response_format: defaults.response_format,
            fallback: if defaults.fallback_models.is_empty() {
                None
            } else {
                Some(crate::retry::FallbackPolicy {
                    models: defaults.fallback_models,
                })
            },
        });
        Ok(Self {
            name: spec.name,
            instructions: spec.instructions,
            toolset: None,
            middleware: Vec::new(),
            model_defaults,
            exposure: spec.exposure.into(),
            defer_loading: spec.defer_loading,
        })
    }

    /// Renders this capability's declarative fields (name, instructions,
    /// exposure, defer_loading, model defaults) as the JSON shape
    /// [`Self::from_spec`] parses. The toolset and middleware are not
    /// representable in JSON and are omitted; round-tripping a capability
    /// through `to_spec`/`from_spec` therefore preserves every field except
    /// those two.
    pub fn to_spec(&self) -> Value {
        let spec = CapabilitySpec {
            name: self.name.clone(),
            instructions: self.instructions.clone(),
            exposure: self.exposure.into(),
            defer_loading: self.defer_loading,
            model_defaults: self
                .model_defaults
                .as_ref()
                .map(|defaults| ModelDefaultsSpec {
                    response_format: defaults.default_response_format.clone(),
                    fallback_models: defaults
                        .fallback
                        .as_ref()
                        .map(|fallback| fallback.models.clone())
                        .unwrap_or_default(),
                }),
        };
        serde_json::to_value(spec).expect("CapabilitySpec always serializes")
    }
}
