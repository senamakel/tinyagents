# Registry Implementation Status

`design.md`, `events.md`, and `operations.md` describe a target design for the
registry module. This page describes what actually exists in
`crates/tinyagents-registry/src` today, verified against the code (2026-09-19).
Treat this page as the ground truth for "is X implemented"; treat the other
three as a proposal for where the module is headed.

## What exists

- **`CapabilityRegistry<State = ()>`** (`capability/types.rs`) — the
  name-addressable capability catalog. Partitioned by `ComponentKind` into
  models (`Arc<dyn ChatModel<State>>`), tools (`Arc<dyn Tool>`), and
  declarative agent definitions (`AgentDefinition`); routers and reducers are
  name-only descriptors for now.
  Tracks presence/discovery metadata per `(kind, name)` and an alias map per
  `(kind, alias)`.
- **`ModelCatalog`** (`catalog.rs`) — a deterministic, offline snapshot of
  provider model prices (including `ModelPricing::tiers` context-size-tiered
  rates), context windows, modalities, and capability flags, embedded at
  compile time from `crates/tinyagents-registry/model-catalog.snapshot.json`
  (the sole copy — the old byte-identical duplicate under
  `docs/modules/registry/` was removed) and looked up by `(provider,
  model_id)` or alias. `ModelCatalogSnapshot::validate()` /
  `validate_with_providers(Some(allowed))` reject a malformed snapshot
  (duplicate `(provider, model_id)`, negative price, missing `source`, an
  `max_output_tokens` exceeding `max_input_tokens`, an alias collision, an
  invalid date, or — only when an allowlist is passed — an unrecognized
  provider id) before `ModelCatalog::from_json`/`try_from_snapshot` accept
  it; `from_snapshot` itself stays non-validating for already-trusted data.
  `cargo run -p tinyagents-registry --bin catalog_gen` (`src/bin/catalog_gen.rs`)
  refreshes the checked-in snapshot from `https://models.dev/api.json`
  (curated provider list, tiered pricing, modalities, reasoning flags),
  validating its own output against `catalog::KNOWN_PROVIDERS` before
  writing. `ModelCatalogSnapshot`, `ModelCatalogSource`, `ModelCatalogEntry`,
  and `ModelCapabilities` are its supporting types.
- **`WorkloadRouter`** (`router/mod.rs`, `router/types.rs`; renamed from
  `ModelRouter`, which remains as a `#[deprecated]` type alias) — a
  declarative, name-addressable router that maps workload-tier aliases
  (`chat-v1`, `vision-v1`, …) onto concrete registered model names, with
  per-tier capability gates (`required_capabilities`) and same-family
  fallback ordering (`fallback_policy`). Holds no models and drives no I/O;
  it is pure policy read while wiring a registry + run policy.
  `WorkloadRoute` is its route type. `CapabilityRegistry` now holds one
  (`with_router`/`set_router`/`router()`) and projects it through
  `route_workload(tier) -> Option<&Arc<dyn ChatModel<State>>>`, which
  resolves a tier through the router *and* checks the resolved model name is
  actually registered in that registry (a route naming an unregistered model
  returns `None` rather than panicking).
- **`RegistrySnapshot` / `RegistryDiagnostic`** (`diagnostics.rs`) — a
  serializable, point-in-time projection of a registry's presence metadata
  (`RegistrySnapshot`, with `AliasBinding` entries) for CLIs/UIs/audit logs,
  and `RegistryDiagnostic`/`DiagnosticSeverity` for alias-collision and
  dangling-alias health checks the registration-time duplicate check cannot
  catch on its own.
- **`component`** (`component/types.rs`) — `ComponentId`, `ComponentKind`,
  `ComponentMetadata`: the discovery types every registered component is
  described by, shared across the pieces above.

## What does not exist

Grepping `crates/tinyagents-registry/src` for the following design-doc types
turns up no matches — they are proposed, not implemented:

- `RegistryEvent`, `EventBus`, and the event/listener/filter model described
  in `events.md`.
- `SharedRegistry`, the static/dynamic component split, stream transformers,
  and the redaction/testkit/discovery machinery described in `operations.md`.
- Store/checkpointer registration as registry components (`events.md`).
- Most of the runtime coordination surface described in `design.md` beyond
  the capability catalog itself (e.g. registry-owned middleware/listener
  wiring, distributed-supervisor integration).

~~There is also no `impl DefinitionRegistry for CapabilityRegistry` yet (see
`docs/runtime-comparison/plan.md`, Phase 1c, `W-I8`/`W-I9`), and no
`set_metadata` / `remove` mutation API on `CapabilityRegistry`.~~ Implemented
in Phase 1c — see below.

## Phase 1c additions (W-I3, W-I8, W-I9)

- **`impl tinyagents_definition::DefinitionRegistry for CapabilityRegistry<State>`**
  (`capability/mod.rs`) — `resolve`/`list`/`delegates_for` read straight from
  the registry's `agents` map, so a host that already registers agents in the
  `CapabilityRegistry` no longer has to build a second, separately populated
  `InMemoryDefinitionRegistry` by hand to satisfy
  `HostCapabilities.definitions: Arc<dyn DefinitionRegistry>` (W-I9). Written
  out by hand matching the exact signature `#[async_trait]` expands to,
  rather than applying the macro here: `tinyagents-registry` only has
  `async-trait` as a *dev*-dependency, so the macro is unavailable to
  non-test library code without a `Cargo.toml` edit outside this change's
  file boundary.
- **`CapabilityRegistry::set_metadata(kind, name, ComponentMetadata)`** —
  overwrites the metadata recorded for an already-registered `(kind, name)`,
  making `ComponentMetadata::with_description`/`with_tag` actually reach a
  registered component instead of being dead on arrival (W-I8).
  **`register_model_with`/`register_tool_with`** attach metadata atomically
  at registration instead of needing a follow-up `set_metadata` call.
- **`CapabilityRegistry::remove(kind, name) -> bool`** — drops a registered
  component and its metadata (a no-op, not an error, if absent). This is what
  makes `diagnostics()`'s `alias_shadows_component`/`dangling_alias` checks
  reachable through the public API: `alias()` is fail-closed against both at
  insertion time, so before `remove` existed only `name_reused_across_kinds`
  could ever fire.
- **Deterministic default model (W-I3)** — `CapabilityRegistry` now tracks
  `model_order: Vec<String>` alongside its model `HashMap`, appended the
  first time a name is registered (`register_model`/`replace_model`;
  re-registering an existing name via `replace_model` does not move it).
  `to_model_registry()` builds the harness `ModelRegistry` by iterating
  `model_order` instead of the `HashMap`, so the "first-registered model
  becomes the default" rule
  (`tinyagents_harness::model_registry::ModelRegistry::register`) is
  reproducible across runs instead of following `HashMap` iteration order.
  `to_model_registry_with_default(name)` (new) builds the same registry with
  an explicit default instead, returning `TinyAgentsError::ModelNotFound` if
  `name` is not registered.

## Capability bundle (gap G3)

`CapabilityRegistry::register_capability(name, Capability<State, Ctx>)`
(`capability/mod.rs`) registers a
`tinyagents_harness::capability::Capability<State, Ctx>` — instructions +
toolset + middleware + model defaults + exposure + `defer_loading`, the
runtime-level unit Pydantic AI v2's `AbstractCapability` occupies
(`docs/runtime-comparison/pydantic-ai.md` §4). The bundle type itself lives
in `tinyagents-harness::capability` rather than in `tinyagents-registry` or
the dependency-free `tinyagents-definition` crate: it composes
`tool::toolset::ToolSet<State, Ctx>` and `middleware::Middleware<State, Ctx>`
trait objects that are native to `tinyagents-harness`, and
`tinyagents-definition` has zero dependency on `tinyagents-harness` by
design (that asymmetry is what keeps the crate graph acyclic). Since
`CapabilityRegistry<State>` carries no `Ctx` type parameter (none of its
other stored kinds need one), registry storage for this specific
`Ctx`-generic bundle goes through type-erased `Box<dyn Any>` storage instead
of adding a `Ctx` parameter to the whole registry for one feature; see
`capability/mod.rs`'s `register_capability`/`capability` doc comments for the
erasure mechanics.

`Capability::from_spec(serde_json::Value)` builds a bundle from data (no
toolset/middleware — those are still wired programmatically). Hosts choose
which JSON-declared capabilities to install and retain control of the
available tool, model, and subagent handles.

On the harness side, `AgentHarness::with_capability(capability)` installs a
bundle: its middleware is appended in installation order, its model defaults
are applied onto the harness's `RunPolicy`, and its toolset is
folded into a `CapabilityToolSet` combined with whatever toolset was already
installed via `with_toolset` before the first `with_capability` call. When
any installed capability has `defer_loading: true`, a synthetic
`load_capability` tool (`LOAD_CAPABILITY_TOOL_NAME`) is auto-registered so a
model can bring a deferred capability's tools into scope on demand; the
resulting mid-run tool-set change is recorded through the B6 transcript
patch mechanism (`agent_loop/tool_changes.rs`) rather than silently
reshaping the next request.

## Why the gap

`design.md`/`events.md`/`operations.md` were written as a forward-looking
specification before implementation started; the capability catalog, model
catalog, router, and diagnostics shipped first because they support harness
model resolution today. The
event/lifecycle/listener layer is still design-stage work.
