//! Harness runtime facade.
//!
//! [`AgentHarness`] is the durable runtime facade. Hosted roots may provide an
//! invocation-local [`InvocationRuntime`] for models, tools, and middleware;
//! authorized children inherit that exact overlay and never substitute their
//! own durable registries.
//! Hosted roots may attach an [`InvocationRuntime`] for their model, tool, and
//! middleware surface. It is invocation-local and every hosted child must
//! inherit it; a missing overlay is rejected rather than falling back to a
//! child's durable harness.
//!
//! Per-tool deadlines are enabled separately from [`RunPolicy`] through
//! [`AgentHarness::with_tool_timeout_settings`]. Expiry becomes a recoverable
//! tool-error result so the model can repair its plan, while the run wall-clock
//! limit remains the outer hard abort.
//!
//! Owns the high-level [`AgentHarness`] builder and the [`RunPolicy`] bundle
//! that wires registries, middleware, and run policy into a single ergonomic
//! entry point. The agent loop driven by this facade lives in the sibling
//! [`crate::agent_loop`] module.
//!
//! # Layout
//!
//! - `types` holds the public type definitions ([`RunPolicy`] and
//!   [`AgentHarness`]).
//! - This file holds the builder, registration, and accessor methods.
//! - `test.rs` holds focused tests for construction and registration.

mod agent;
mod types;

#[cfg(test)]
pub(crate) use agent::HostInvocationAuthority;
pub use agent::{AgentInvocation, AgentStream, AgentTurnRequest, HostedError, HostedErrorKind};
pub(crate) use agent::{ErasedHostAuthority, emit_host_progress, host_invocation_binding};
pub use types::*;

use std::sync::Arc;

use crate::cache::ResponseCache;
use crate::middleware::{Middleware, MiddlewareStack, ModelMiddleware, ToolMiddleware};
use crate::model_registry::ModelRegistry;
use crate::tool::{ToolDispatch, ToolRegistry, ToolTimeoutSettings};
use tinyinference_llm::model::ChatModel;
use tinytools::Tool;

impl<State: Send + Sync, Ctx: Send + Sync> AgentHarness<State, Ctx> {
    /// Creates an empty harness with default policy and no models, tools, or
    /// middleware registered.
    pub fn new() -> Self {
        Self {
            models: ModelRegistry::new(),
            tools: ToolRegistry::new(),
            middleware: MiddlewareStack::new(),
            policy: RunPolicy::default(),
            tool_timeouts: None,
            response_cache: None,
            output_validator: None,
            toolset: None,
            capabilities: Vec::new(),
            capability_base_toolset: None,
        }
    }

    /// Registers a model under `name`. The first registered model becomes the
    /// registry default unless one is already set. Returns `&mut Self` for
    /// chaining.
    pub fn register_model(
        &mut self,
        name: impl Into<String>,
        model: Arc<dyn ChatModel<State>>,
    ) -> &mut Self {
        self.models.register(name, model);
        self
    }

    /// Sets the default model name used when a request specifies no override.
    pub fn set_default_model(&mut self, name: impl Into<String>) -> &mut Self {
        self.models.set_default(name);
        self
    }

    /// Registers a tool, keyed by its [`Tool::name`]. Returns `&mut Self` for
    /// chaining.
    pub fn register_tool(&mut self, tool: Arc<dyn Tool>) -> &mut Self {
        self.tools.register(tool);
        self
    }

    /// Registers a tool whose execution needs the typed parent run.
    pub fn register_tool_dispatch(
        &mut self,
        dispatch: Arc<dyn ToolDispatch<State, Ctx>>,
    ) -> &mut Self {
        self.tools.register_dispatch(dispatch);
        self
    }

    /// Appends a lifecycle middleware to the stack. Registration order is the
    /// onion order: the first pushed middleware is the outermost layer.
    pub fn push_middleware(&mut self, middleware: Arc<dyn Middleware<State, Ctx>>) -> &mut Self {
        self.middleware.push(middleware);
        self
    }

    /// Appends an around-model wrap middleware ([`ModelMiddleware`]). The
    /// first-registered wrap middleware is the outermost layer; the real model
    /// call (cache + retry + fallback core) is the innermost. Returns
    /// `&mut Self` for chaining.
    pub fn push_model_middleware(
        &mut self,
        middleware: Arc<dyn ModelMiddleware<State, Ctx>>,
    ) -> &mut Self {
        self.middleware.push_model_middleware(middleware);
        self
    }

    /// Appends an around-tool wrap middleware ([`ToolMiddleware`]). The
    /// first-registered wrap middleware is the outermost layer; the real tool
    /// call is the innermost. Returns `&mut Self` for chaining.
    pub fn push_tool_middleware(
        &mut self,
        middleware: Arc<dyn ToolMiddleware<State, Ctx>>,
    ) -> &mut Self {
        self.middleware.push_tool_middleware(middleware);
        self
    }

    /// Replaces the run policy. Returns `&mut Self` for chaining.
    pub fn with_policy(&mut self, policy: RunPolicy) -> &mut Self {
        self.policy = policy;
        self
    }

    /// Installs the shared resolver used for per-tool timeout policies.
    ///
    /// Keeping this runtime concern on the harness leaves [`RunPolicy`]
    /// source-compatible for callers that construct it with struct literals.
    pub fn with_tool_timeout_settings(&mut self, settings: ToolTimeoutSettings) -> &mut Self {
        self.tool_timeouts = Some(settings);
        self
    }

    /// Returns the installed per-tool timeout settings, if any.
    pub fn tool_timeout_settings(&self) -> Option<&ToolTimeoutSettings> {
        self.tool_timeouts.as_ref()
    }

    /// Attaches a [`ResponseCache`] shared across every run this harness drives.
    ///
    /// Once attached, the agent loop computes a stable
    /// [`cache_key`][crate::cache::cache_key] for each model request
    /// and consults the cache before calling the provider. On a hit the
    /// provider is **not** invoked and the cached
    /// [`tinyinference_llm::model::ModelResponse`] is reused; on a miss the
    /// provider is called and the successful response is stored back. Whether
    /// caching is active for a given call is governed by the effective
    /// [`CachePolicy`](tinyinference_llm::cache::CachePolicy) (the per-request
    /// [`tinyinference_llm::model::ModelRequest::cache_policy`] overriding
    /// [`RunPolicy::cache`]).
    ///
    /// Because the cache lives on the harness rather than a single run, two
    /// identical requests issued across separate runs share a key, so the
    /// second run can be served entirely from cache. Returns `&mut Self` for
    /// chaining.
    pub fn with_response_cache(&mut self, cache: Arc<dyn ResponseCache>) -> &mut Self {
        self.response_cache = Some(cache);
        self
    }

    /// Returns a reference to the attached response cache, if any.
    pub fn response_cache(&self) -> Option<&Arc<dyn ResponseCache>> {
        self.response_cache.as_ref()
    }

    /// Registers an [`crate::structured::OutputValidator`] consulted after
    /// the final turn's structured extraction succeeds (A3's
    /// output-validation retry loop).
    ///
    /// The validator sees the *already schema-valid* extracted value; a
    /// `TinyAgentsError::ModelRetry` it returns is treated exactly like a
    /// schema-validation failure — re-asked, bounded by
    /// [`RunPolicy::output_retry`]. Only one validator may be installed;
    /// calling this again replaces it. Returns `&mut Self` for chaining.
    pub fn with_output_validator(
        &mut self,
        validator: Arc<dyn crate::structured::OutputValidator<State, Ctx>>,
    ) -> &mut Self {
        self.output_validator = Some(validator);
        self
    }

    /// Installs a composable [`crate::tool::toolset::ToolSet`] chain (gap
    /// B3) as an additional source of tools, consulted alongside
    /// [`Self::tools`].
    ///
    /// # What this changes
    ///
    /// - **Advertisement**: the agent loop's per-turn model-visible tool
    ///   catalogue is built by projecting this toolset's
    ///   [`crate::tool::toolset::ToolSet::tools`] (re-consulted every turn,
    ///   so a [`crate::tool::toolset::PreparedToolSet`] or
    ///   [`crate::tool::toolset::ApprovalRequiredToolSet`] in the chain can
    ///   vary what is advertised turn to turn) **in addition to** the
    ///   registry's own `Direct` schemas — a name the toolset does not
    ///   mention falls back to the registry unchanged.
    /// - **Dispatch is not automatically wired to this toolset.** The agent
    ///   loop's admission path (`agent_loop/tools.rs`) resolves calls through
    ///   [`Self::tools`] only, exactly as before this field existed. A tool
    ///   that only the toolset chain exposes must also be reachable through
    ///   the registry to be *callable* (not just advertised) — bridge it
    ///   explicitly with
    ///   [`crate::tool::toolset::ToolSetDispatchBridge`] and
    ///   [`Self::register_tool_dispatch`]. See that bridge's doc comment for
    ///   why: it requires `State: 'static, Ctx: 'static`, a bound the loop's
    ///   generic admission path deliberately does not carry (recursive
    ///   sub-agent dispatch stays callable with a borrowed, non-`'static`
    ///   `State`/`Ctx`).
    ///
    /// A caller building a fresh [`crate::tool::ToolRegistry`] separately
    /// (rather than through [`Self::register_tool`]) can pass it here
    /// directly — [`crate::tool::ToolRegistry`] implements
    /// [`crate::tool::toolset::ToolSet`] — or compose it with other
    /// toolsets via [`crate::tool::toolset::CombinedToolSet`].
    ///
    /// `None` (never calling this) leaves every existing harness's turn
    /// behavior exactly as before this field existed. Returns `&mut Self`
    /// for chaining.
    pub fn with_toolset(
        &mut self,
        toolset: Arc<dyn crate::tool::toolset::ToolSet<State, Ctx>>,
    ) -> &mut Self {
        self.toolset = Some(toolset);
        self
    }

    /// Returns the installed toolset chain, if any. See
    /// [`Self::with_toolset`].
    pub fn toolset(&self) -> Option<&Arc<dyn crate::tool::toolset::ToolSet<State, Ctx>>> {
        self.toolset.as_ref()
    }

    /// Installs a [`crate::capability::Capability`] bundle (gap G3): its
    /// toolset, middleware, and model-request defaults are applied to this
    /// harness, and its [`crate::capability::Capability::exposure`]/
    /// [`crate::capability::Capability::defer_loading`] settings are honored
    /// by the [`crate::capability::CapabilityToolSet`] this method installs.
    ///
    /// May be called more than once; every installed capability accumulates
    /// (see [`Self::capabilities`]) and [`Self::toolset`] is rebuilt each
    /// time from the complete list, so a `defer_loading` capability's
    /// [`crate::capability::LoadCapabilityTool`] always covers every deferred
    /// capability installed so far, under one shared load state.
    ///
    /// # What this changes
    ///
    /// - **Toolset**: composes a fresh
    ///   [`crate::capability::CapabilityToolSet`] over every installed
    ///   capability with whatever toolset was already installed via
    ///   [`Self::with_toolset`] *before* the first `with_capability` call
    ///   (captured once, in [`Self::capability_base_toolset`]) through
    ///   [`crate::tool::toolset::CombinedToolSet`]. As with
    ///   [`Self::with_toolset`], dispatch for a capability's own tools is not
    ///   automatically bridged into [`Self::tools`] — bridge explicitly with
    ///   [`crate::tool::toolset::ToolSetDispatchBridge`] and
    ///   [`Self::register_tool_dispatch`] for a tool that must be callable,
    ///   not just advertised. The synthetic `load_capability` tool is the one
    ///   exception: it is registered directly into [`Self::tools`] (it needs
    ///   no `RunContext`/`State` to execute), so it is callable immediately.
    /// - **Middleware**: each capability's middleware is appended, in
    ///   installation order, via [`Self::push_middleware`].
    /// - **Model defaults**: each capability's
    ///   [`crate::capability::ModelRequestDefaults`], if set, is applied onto
    ///   [`Self::policy`] via
    ///   [`crate::capability::ModelRequestDefaults::apply_to`] — a later
    ///   capability's set fields win over an earlier one's.
    ///
    /// Returns `&mut Self` for chaining.
    pub fn with_capability(&mut self, capability: crate::capability::Capability<State, Ctx>) -> &mut Self
    where
        State: 'static,
        Ctx: 'static,
    {
        if self.capabilities.is_empty() {
            self.capability_base_toolset = self.toolset.take();
        }
        for middleware in capability.middleware.clone() {
            self.push_middleware(middleware);
        }
        if let Some(defaults) = &capability.model_defaults {
            defaults.apply_to(&mut self.policy);
        }
        self.capabilities.push(capability);

        let capability_toolset = crate::capability::CapabilityToolSet::new(self.capabilities.clone());

        // The `load_capability` tool needs no `RunContext`/`State` to run, so
        // it is registered directly into `self.tools` — the one part of a
        // capability's contribution that is callable, not just advertised,
        // without a caller-supplied dispatch bridge (see the doc comment
        // above). `Self::register_tool` silently replaces a prior
        // registration under the same name, so re-registering on every call
        // keeps it in sync with the full, still-accumulating capability list.
        if let Some(load_tool) = capability_toolset.load_tool() {
            self.register_tool(load_tool);
        }

        let capability_toolset: Arc<dyn crate::tool::toolset::ToolSet<State, Ctx>> =
            Arc::new(capability_toolset);
        self.toolset = Some(match &self.capability_base_toolset {
            Some(base) => Arc::new(crate::tool::toolset::CombinedToolSet::new(vec![
                base.clone(),
                capability_toolset,
            ])),
            None => capability_toolset,
        });

        self
    }

    /// Returns a reference to the model registry.
    pub fn models(&self) -> &ModelRegistry<State> {
        &self.models
    }

    /// Returns a reference to the tool registry.
    pub fn tools(&self) -> &ToolRegistry<State, Ctx> {
        &self.tools
    }

    /// Returns a reference to the middleware stack.
    pub fn middleware(&self) -> &MiddlewareStack<State, Ctx> {
        &self.middleware
    }

    /// Returns a reference to the active run policy.
    pub fn policy(&self) -> &RunPolicy {
        &self.policy
    }
}

impl<State: Send + Sync, Ctx: Send + Sync> Default for AgentHarness<State, Ctx> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test;
