# SDK Gaps: Cost, Model Catalog, And Registry

> Part of [SDK Gaps](README.md). Covers cost/usage/budget enforcement,
> model catalog and provider resolution, and registry diagnostics.

## Backlog

### 7. Cost, Usage, And Budget Enforcement

Status: partially present.

TinyAgents has `Usage`, `UsageTotals`, `CostTotals`, and accounting middleware.
`ModelPricing` now supports context-size-tiered rates (`ModelPricing::tiers` /
`PriceTier`, selected by `harness::cost::estimate_cost` against a call's
input-token count) for providers that price differently above a context
threshold (see `docs/modules/harness/cost.md`). OpenHuman still owns richer
budget behavior, global cost trackers, per-session rollups, budget stop hooks,
and token/cost dashboard data.

Implement:

- A budget middleware that can preflight, enforce, and reconcile costs.
- Per-run and recursive root-run budgets for input, output, cached input,
  reasoning tokens, total tokens, and money.
- Distinguish provider-reported usage from estimated usage.
- Track cached-token pricing, reasoning pricing, embeddings, image/audio usage,
  and tool/provider fees where present.
- Add budget events: preflight, reservation, spend, refund/reconcile, warn,
  exceeded, blocked.

Acceptance criteria:

- A caller can stop a recursive harness/graph run when a root budget is
  exhausted.
- Budget totals roll up from child/sub-agent runs without custom side channels.
- OpenHuman cost UI can read TinyAgents-normalized records or a thin projection.

### 8. Model Catalog And Provider Resolution

Status: partially present.

TinyAgents has `ModelProfile`, including provider, model, modalities, tool
calling, streaming, structured output, reasoning, and token windows, plus
behavioral fields (`schema_transform`, `default_structured_mode`,
`prompted_output_template`, `thinking_tags`, `thinking_level_map`, `compat:
ProviderCompat`) so an adapter's request/response shaping can be driven by
data instead of hand-written per-model branches. The registry now owns a
generator (`cargo run -p tinyagents-registry --bin catalog_gen`) that refreshes
`crates/tinyagents-registry/model-catalog.snapshot.json` from
`https://models.dev/api.json` with tiered pricing, and
`ModelCatalogSnapshot::validate`/`validate_with_providers` reject a malformed
snapshot (duplicate ids, negative prices, missing source, an output limit
exceeding the input context, alias collisions, bad dates, and — opt-in — an
unrecognized provider id) before it is loaded. `ModelRouter` was renamed to
`WorkloadRouter` (deprecated alias kept) and is now projectable through
`CapabilityRegistry::route_workload(tier)`. OpenHuman still has provider
catalog logic and local model capability inference that drive fallback, token
budgeting, and routing; `ModelCatalog::available_for(auth)` credential-aware
filtering and the generalized `CredentialStore`/OAuth flow remain OpenHuman's
to own (credential handling was scoped out of this SDK pass).

Implement:

- SDK-owned model catalog snapshots with provider, model id, display name,
  lifecycle status, context windows, modalities, streaming support, reasoning,
  structured-output support, and pricing keys.
- Capability-driven model resolution: required capabilities, fallback chains,
  local/cloud preferences, and provider health.
- Runtime profile discovery hooks for local models.
- Pricing table integration that maps `ModelProfile` to `CostTotals`.

Acceptance criteria:

- Model selection can be expressed in TinyAgents policy instead of
  OpenHuman-only routing code.
- Fallback can reject models that lack required tool, vision, structured-output,
  context-window, or reasoning capabilities.
- Token budgeting can use the resolved model's real context window.

### 15. Registry Diagnostics And Introspection

Status: partially present.

TinyAgents has registry primitives. OpenHuman still needs richer diagnostics for
duplicate components, alias resolution, component health, model/provider/tool
capabilities, and event listener wiring. Implement:

- Registry snapshot export with models, tools, middleware, graph nodes,
  checkpointers, task stores, event listeners, and aliases.
- Duplicate and shadowing diagnostics.
- Health/status probes for registered providers and stores.
- Machine-readable component dependency graph.
- Optional DOT/JSON graph export for runtime components, not only graph nodes.

Acceptance criteria:

- A CLI or UI can show exactly what TinyAgents components are active.
- Registry failures are actionable without inspecting app-specific logs.
- OpenHuman dead-code audits can map old modules to SDK-owned registry entries.

