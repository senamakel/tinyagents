# `tinyagents-tracing` — shared opt-in tracing macros

A minimal re-export of `tracing` macros (`debug!`, `info!`, `warn!`, `error!`,
`trace!`) for use across the TinyAgents crate family. When the `tracing`
feature is enabled, the macros wire to the real `tracing` crate. When disabled,
they expand to no-ops that stringify their arguments and discard them, so
instrumentation code compiles out with zero overhead.

## Public surface

- **`debug!`, `info!`, `warn!`, `error!`, `trace!`** — tracing event macros.
  Enabled when `feature = "tracing"`, no-ops otherwise.

## Design

- **Feature-gated:** Instrumentation is opt-in. By default, all tracing code
  compiles away. This keeps the default build deterministic (no dependency on
  a tracing subscriber).
- **Shared vocabulary:** All crates in the workspace import from this module,
  ensuring consistent tracing calls across boundaries.
- **Zero overhead:** The no-op implementation (`stringify!` + discard) is
  optimized away by the compiler, leaving no runtime cost.
