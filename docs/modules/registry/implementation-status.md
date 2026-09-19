# Registry Implementation Status

`design.md`, `events.md`, and `operations.md` describe a target design for the
registry module. This page describes what actually exists in
`crates/tinyagents-registry/src` today, verified against the code (2026-09-19).
Treat this page as the ground truth for "is X implemented"; treat the other
three as a proposal for where the module is headed.

## What exists

- **`CapabilityRegistry<State = ()>`** (`capability/types.rs`) — the
  name-addressable capability catalog. Partitioned by `ComponentKind` into
  models (`Arc<dyn ChatModel<State>>`), tools (`Arc<dyn Tool>`), graph
  blueprints (`Blueprint`), and declarative agent definitions
  (`AgentDefinition`); routers and reducers are name-only descriptors for now.
  Tracks presence/discovery metadata per `(kind, name)` and an alias map per
  `(kind, alias)`. This is the type `.rag` sources bind against.
- **`ModelCatalog`** (`catalog.rs`) — a deterministic, offline snapshot of
  provider model prices, context windows, and capability flags, embedded at
  compile time from `docs/modules/registry/model-catalog.snapshot.json` and
  looked up by `(provider, model_id)` or alias. `ModelCatalogSnapshot`,
  `ModelCatalogSource`, `ModelCatalogEntry`, and `ModelCapabilities` are its
  supporting types.
- **`ModelRouter`** (`router/mod.rs`, `router/types.rs`) — a declarative,
  name-addressable router that maps workload-tier aliases (`chat-v1`,
  `vision-v1`, …) onto concrete registered model names, with per-tier
  capability gates (`required_capabilities`) and same-family fallback
  ordering (`fallback_policy`). Holds no models and drives no I/O; it is pure
  policy read while wiring a registry + run policy. `WorkloadRoute` is its
  route type.
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

There is also no `impl DefinitionRegistry for CapabilityRegistry` yet (see
`docs/runtime-comparison/plan.md`, Phase 1c, `W-I8`/`W-I9`), and no
`set_metadata` / `remove` mutation API on `CapabilityRegistry`.

## Why the gap

`design.md`/`events.md`/`operations.md` were written as a forward-looking
specification before implementation started; the capability catalog, model
catalog, router, and diagnostics shipped first because they are the pieces
`.rag` compilation and harness model resolution depend on today. The
event/lifecycle/listener layer is still design-stage work.
