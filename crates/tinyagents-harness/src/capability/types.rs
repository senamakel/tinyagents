//! Type definitions for the capability bundle module (gap G3,
//! `docs/runtime-comparison/plan.md`, `docs/runtime-comparison/pydantic-ai.md`
//! §4 "Capabilities as the unit of composition").
//!
//! [`Capability`] is the bundle: instructions + a composable
//! [`crate::tool::toolset::ToolSet`] + middleware + model-request defaults +
//! [`ToolExposure`] + `defer_loading`, mirroring Pydantic AI's
//! `AbstractCapability`. [`CapabilityToolSet`] is the [`ToolSet`] adaptor that
//! makes a list of capabilities composable like any other toolset —
//! [`crate::runtime::AgentHarness::with_capability`] installs one — and
//! [`LoadCapabilityTool`] is the synthetic tool it auto-registers whenever any
//! capability declares `defer_loading: true`, so a model can bring a deferred
//! capability's tools and instructions into scope mid-run (Pydantic AI's
//! `load_capability`).

use std::collections::HashSet;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tinytools::{Tool, ToolExposure, ToolResult};

use crate::context::RunContext;
use crate::error::{Result, TinyAgentsError};
use crate::middleware::Middleware;
use crate::retry::FallbackPolicy;
use crate::tool::toolset::{OverrideTool, ToolSet};
use tinyinference_llm::model::ResponseFormat;

/// Name of the synthetic tool [`CapabilityToolSet`] auto-registers whenever
/// at least one of its capabilities declares [`Capability::defer_loading`].
pub const LOAD_CAPABILITY_TOOL_NAME: &str = "load_capability";

/// Default per-capability overrides applied to the harness's
/// [`crate::runtime::RunPolicy`] by [`crate::runtime::AgentHarness::with_capability`].
///
/// Deliberately narrower than the full [`crate::runtime::RunPolicy`]: only
/// the fields that are both meaningfully "this capability's preference" and
/// cheaply serializable (for [`Capability::to_spec`]/[`Capability::from_spec`])
/// are included. [`crate::retry::RetryPolicy`] is not — it carries a
/// non-serializable predicate closure — so a capability wanting a custom
/// retry policy must be built programmatically with
/// [`Capability::with_middleware`]/[`crate::runtime::AgentHarness::with_policy`]
/// instead.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelRequestDefaults {
    /// Overrides [`crate::runtime::RunPolicy::default_response_format`] when
    /// set.
    pub default_response_format: Option<ResponseFormat>,
    /// Overrides [`crate::runtime::RunPolicy::fallback`] when set.
    pub fallback: Option<FallbackPolicy>,
}

impl ModelRequestDefaults {
    /// Applies every set field onto `policy` in place, leaving the fields
    /// this bundle leaves `None` untouched.
    pub fn apply_to(&self, policy: &mut crate::runtime::RunPolicy) {
        if let Some(format) = &self.default_response_format {
            policy.default_response_format = Some(format.clone());
        }
        if let Some(fallback) = &self.fallback {
            policy.fallback = Some(fallback.clone());
        }
    }
}

/// A composable capability bundle: the unit an `AgentDefinition` or host
/// session references as one named thing instead
/// of wiring a toolset, middleware, and model defaults separately.
///
/// Generic over the same `State`/`Ctx` pair as
/// [`crate::runtime::AgentHarness<State, Ctx>`] and
/// [`crate::tool::toolset::ToolSet<State, Ctx>`] — a `Capability` is meant to
/// be installed directly onto a harness via
/// [`crate::runtime::AgentHarness::with_capability`], not stored inside a
/// registry that has no `Ctx` dimension of its own (see
/// `tinyagents-registry`'s `CapabilityRegistry::register_capability`, which
/// type-erases this value through `Box<dyn Any>` for exactly that reason).
pub struct Capability<State: Send + Sync, Ctx: Send + Sync> {
    /// The capability's stable, unique name.
    pub name: String,
    /// Instructions this capability contributes to the system prompt while
    /// loaded (immediately, unless [`Self::defer_loading`] is set).
    pub instructions: Option<String>,
    /// The tools this capability contributes, if any.
    pub toolset: Option<Arc<dyn ToolSet<State, Ctx>>>,
    /// Middleware appended to the harness's stack when this capability is
    /// installed.
    pub middleware: Vec<Arc<dyn Middleware<State, Ctx>>>,
    /// Model-request defaults applied to the harness's [`crate::runtime::RunPolicy`]
    /// when this capability is installed.
    pub model_defaults: Option<ModelRequestDefaults>,
    /// The [`ToolExposure`] applied uniformly to every tool
    /// [`Self::toolset`] contributes, overriding each tool's own declared
    /// exposure. Defaults to [`ToolExposure::Direct`].
    pub exposure: ToolExposure,
    /// When `true`, this capability's tools and instructions are withheld
    /// until a model calls [`LOAD_CAPABILITY_TOOL_NAME`] with this
    /// capability's [`Self::name`] (Pydantic AI's `defer_loading`).
    pub defer_loading: bool,
}

impl<State: Send + Sync, Ctx: Send + Sync> Clone for Capability<State, Ctx> {
    /// Manual `Clone`, not `#[derive(Clone)]`: a derive would add spurious
    /// `State: Clone, Ctx: Clone` bounds even though neither type parameter
    /// is stored by value here (only inside already-`Clone` `Arc`s).
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            instructions: self.instructions.clone(),
            toolset: self.toolset.clone(),
            middleware: self.middleware.clone(),
            model_defaults: self.model_defaults.clone(),
            exposure: self.exposure,
            defer_loading: self.defer_loading,
        }
    }
}

impl<State: Send + Sync, Ctx: Send + Sync> std::fmt::Debug for Capability<State, Ctx> {
    /// Renders the declarative fields; the toolset and middleware are opaque
    /// trait objects, so only their presence/count is shown.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Capability")
            .field("name", &self.name)
            .field("instructions", &self.instructions)
            .field("has_toolset", &self.toolset.is_some())
            .field("middleware_count", &self.middleware.len())
            .field("model_defaults", &self.model_defaults)
            .field("exposure", &self.exposure)
            .field("defer_loading", &self.defer_loading)
            .finish()
    }
}

/// The JSON-facing shape [`Capability::from_spec`]/[`Capability::to_spec`]
/// round-trip. Only the declarative fields survive: a `toolset` and
/// `middleware` are live trait objects and cannot be represented in JSON, so
/// a capability built from a spec always has an empty toolset/middleware —
/// a host wanting either wires them in afterward with
/// [`Capability::with_toolset`]/[`Capability::with_middleware`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct CapabilitySpec {
    pub(crate) name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) instructions: Option<String>,
    #[serde(default)]
    pub(crate) exposure: ExposureSpec,
    #[serde(default)]
    pub(crate) defer_loading: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) model_defaults: Option<ModelDefaultsSpec>,
}

/// JSON-serializable mirror of [`ToolExposure`], which does not itself
/// derive `Serialize`/`Deserialize`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ExposureSpec {
    #[default]
    Direct,
    Deferred,
    Hidden,
}

impl From<ExposureSpec> for ToolExposure {
    fn from(value: ExposureSpec) -> Self {
        match value {
            ExposureSpec::Direct => ToolExposure::Direct,
            ExposureSpec::Deferred => ToolExposure::Deferred,
            ExposureSpec::Hidden => ToolExposure::Hidden,
        }
    }
}

impl From<ToolExposure> for ExposureSpec {
    fn from(value: ToolExposure) -> Self {
        match value {
            ToolExposure::Direct => ExposureSpec::Direct,
            ToolExposure::Deferred => ExposureSpec::Deferred,
            ToolExposure::Hidden => ExposureSpec::Hidden,
        }
    }
}

/// JSON-serializable mirror of [`ModelRequestDefaults`]: `fallback_models`
/// stands in for [`FallbackPolicy`] (which does not derive
/// `Serialize`/`Deserialize`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct ModelDefaultsSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) response_format: Option<ResponseFormat>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) fallback_models: Vec<String>,
}

/// [`ToolSet`] adaptor composing a list of [`Capability`] bundles, gating a
/// `defer_loading` capability's tools and instructions behind a
/// [`LoadCapabilityTool`] call (gap G3).
///
/// Installed by [`crate::runtime::AgentHarness::with_capability`], which
/// composes it with any toolset already installed via
/// [`crate::runtime::AgentHarness::with_toolset`]. The load state is shared
/// interior-mutable state (`Arc<RwLock<HashSet<String>>>`) so a call to
/// [`LOAD_CAPABILITY_TOOL_NAME`] on one turn is visible to
/// [`ToolSet::tools`]/[`ToolSet::instructions`] on the very next turn — which
/// is exactly what lets the agent loop's existing tool-change diff (gap B6,
/// `crate::agent_loop::tool_changes`) pick up the change automatically: no
/// patch message needs to be hand-constructed here.
pub struct CapabilityToolSet<State: Send + Sync, Ctx: Send + Sync> {
    pub(crate) capabilities: Vec<Capability<State, Ctx>>,
    pub(crate) loaded: Arc<RwLock<HashSet<String>>>,
    pub(crate) load_tool: Option<Arc<LoadCapabilityTool>>,
}

impl<State: Send + Sync, Ctx: Send + Sync> CapabilityToolSet<State, Ctx> {
    /// Builds a toolset over `capabilities`. Every non-`defer_loading`
    /// capability starts loaded; every `defer_loading` capability starts
    /// unloaded and — since at least one is present — a
    /// [`LoadCapabilityTool`] is auto-registered to bring it into scope.
    pub fn new(capabilities: Vec<Capability<State, Ctx>>) -> Self {
        let mut loaded_names = HashSet::new();
        let mut deferred_names = Vec::new();
        for capability in &capabilities {
            if capability.defer_loading {
                deferred_names.push(capability.name.clone());
            } else {
                loaded_names.insert(capability.name.clone());
            }
        }
        let loaded = Arc::new(RwLock::new(loaded_names));
        let load_tool = if deferred_names.is_empty() {
            None
        } else {
            deferred_names.sort();
            Some(Arc::new(LoadCapabilityTool::new(
                deferred_names,
                loaded.clone(),
            )))
        };
        Self {
            capabilities,
            loaded,
            load_tool,
        }
    }

    /// Names of every capability currently loaded (every non-deferred
    /// capability, plus every deferred one a [`LoadCapabilityTool`] call has
    /// loaded), sorted for deterministic assertions.
    pub fn loaded_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .loaded
            .read()
            .expect("capability load state lock poisoned")
            .iter()
            .cloned()
            .collect();
        names.sort();
        names
    }

    fn is_loaded(&self, capability: &Capability<State, Ctx>, loaded: &HashSet<String>) -> bool {
        !capability.defer_loading || loaded.contains(&capability.name)
    }

    /// The synthetic [`LoadCapabilityTool`] this toolset auto-registered, if
    /// any capability declared `defer_loading: true`. `None` when every
    /// capability loads eagerly.
    pub fn load_tool(&self) -> Option<Arc<dyn Tool>> {
        self.load_tool
            .as_ref()
            .map(|tool| tool.clone() as Arc<dyn Tool>)
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ToolSet<State, Ctx> for CapabilityToolSet<State, Ctx> {
    async fn tools(&self, ctx: &RunContext<Ctx>) -> Result<Vec<Arc<dyn Tool>>> {
        let loaded = self
            .loaded
            .read()
            .expect("capability load state lock poisoned")
            .clone();
        let mut out = Vec::new();
        for capability in &self.capabilities {
            if !self.is_loaded(capability, &loaded) {
                continue;
            }
            if let Some(toolset) = &capability.toolset {
                for tool in toolset.tools(ctx).await? {
                    out.push(
                        Arc::new(OverrideTool::new(tool).with_exposure(capability.exposure))
                            as Arc<dyn Tool>,
                    );
                }
            }
        }
        if let Some(load_tool) = &self.load_tool {
            out.push(load_tool.clone() as Arc<dyn Tool>);
        }
        Ok(out)
    }

    async fn call(&self, name: &str, args: Value, ctx: &RunContext<Ctx>) -> Result<ToolResult> {
        if let Some(load_tool) = &self.load_tool
            && name == LOAD_CAPABILITY_TOOL_NAME
        {
            return load_tool
                .execute(args)
                .await
                .map_err(|err| TinyAgentsError::Tool(err.to_string()));
        }
        let loaded = self
            .loaded
            .read()
            .expect("capability load state lock poisoned")
            .clone();
        for capability in &self.capabilities {
            if !self.is_loaded(capability, &loaded) {
                continue;
            }
            let Some(toolset) = &capability.toolset else {
                continue;
            };
            let owns = toolset
                .tools(ctx)
                .await?
                .iter()
                .any(|tool| tool.name() == name);
            if owns {
                return toolset.call(name, args, ctx).await;
            }
        }
        Err(TinyAgentsError::ToolNotFound(name.to_string()))
    }

    fn instructions(&self) -> Option<String> {
        let loaded = self
            .loaded
            .read()
            .expect("capability load state lock poisoned")
            .clone();
        let mut parts = Vec::new();
        let mut pending = Vec::new();
        for capability in &self.capabilities {
            if self.is_loaded(capability, &loaded) {
                if let Some(instructions) = &capability.instructions {
                    parts.push(instructions.clone());
                }
            } else {
                pending.push(capability.name.clone());
            }
        }
        if !pending.is_empty() {
            pending.sort();
            parts.push(format!(
                "Additional capabilities are available via `{LOAD_CAPABILITY_TOOL_NAME}`: {}.",
                pending.join(", ")
            ));
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join("\n\n"))
        }
    }

    async fn for_run(&self, ctx: &RunContext<Ctx>) -> Result<()> {
        let loaded = self
            .loaded
            .read()
            .expect("capability load state lock poisoned")
            .clone();
        for capability in &self.capabilities {
            if !self.is_loaded(capability, &loaded) {
                continue;
            }
            if let Some(toolset) = &capability.toolset {
                toolset.for_run(ctx).await?;
            }
        }
        Ok(())
    }
}

/// The synthetic tool [`CapabilityToolSet`] auto-registers whenever at least
/// one of its capabilities declares `defer_loading: true`.
///
/// Calling it with a known deferred capability name marks that capability
/// loaded in the shared state every [`CapabilityToolSet`] method reads, so
/// the very next turn's [`ToolSet::tools`]/[`ToolSet::instructions`] reflect
/// it — which is what lets the agent loop's existing per-turn tool-change
/// diff (gap B6) record the change as an ordinary transcript patch, with no
/// bespoke wiring needed here.
pub struct LoadCapabilityTool {
    pub(crate) available: Vec<String>,
    pub(crate) loaded: Arc<RwLock<HashSet<String>>>,
}

impl LoadCapabilityTool {
    pub(crate) fn new(available: Vec<String>, loaded: Arc<RwLock<HashSet<String>>>) -> Self {
        Self { available, loaded }
    }
}

#[async_trait]
impl Tool for LoadCapabilityTool {
    fn name(&self) -> &str {
        LOAD_CAPABILITY_TOOL_NAME
    }

    fn description(&self) -> &str {
        "Loads a deferred capability bundle by name, making its tools and \
         instructions available for the rest of this run."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "capability": {
                    "type": "string",
                    "description": "The deferred capability name to load.",
                    "enum": self.available,
                }
            },
            "required": ["capability"],
        })
    }

    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        let Some(name) = args.get("capability").and_then(Value::as_str) else {
            return Ok(ToolResult::error(
                "`capability` argument is required".to_string(),
            ));
        };
        if !self.available.iter().any(|available| available == name) {
            return Ok(ToolResult::error(format!(
                "unknown deferred capability `{name}`"
            )));
        }
        self.loaded
            .write()
            .expect("capability load state lock poisoned")
            .insert(name.to_string());
        Ok(ToolResult::success(format!("capability `{name}` loaded")))
    }
}
