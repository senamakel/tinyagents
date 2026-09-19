# harness::tools

Optional builtin canonical tools, gated behind the `tools` Cargo feature so a
host that brings its own tool surface does not pull in the extra dependencies
(`chrono`, `chrono-tz`) by default.

Every tool here implements `tinytools::Tool` directly — none needs the
recursive `ToolDispatch` seam in `harness::tool`, since none needs to see the
typed parent run. They register into a `harness::tool::ToolRegistry` like any
other canonical tool.

## Public surface

- `CurrentTimeTool` (`time.rs`) — returns the current UTC and local time,
  optionally converted to a requested IANA timezone.
- `ResolveTimeTool` (`time.rs`) — resolves a relative or absolute time
  expression (`"in 10 minutes"`, `"7d"`, `"2024-01-01"`, RFC-3339, ...) to an
  exact timestamp, so a model produces date/time arguments for other tools
  without hand-computing Unix seconds.
- `time_tools() -> Vec<Arc<dyn Tool>>` — the builtin time tool set as boxed
  trait objects.
- `register_time_tools(&mut ToolRegistry<State, Ctx>)` — registers both time
  tools into an existing registry.

## Files

| File | Role |
| --- | --- |
| `mod.rs` | Feature gate, submodule wiring, and the `time` re-exports. |
| `time.rs` | `CurrentTimeTool`, `ResolveTimeTool`, and the expression parser (`resolve_expr`, `parse_relative_duration`, `ResolveZone`) backing them. |
| `time_test.rs` | Unit tests for both tools and the underlying expression parser. |

## Operational notes

- Both time tools declare `ToolPolicy::read_only()` — they have no side
  effects and are safe to run without confirmation.
- `resolve_expr` / `parse_relative_duration` / `ResolveZone` are `pub(crate)`
  so they are unit-testable directly, but they are implementation details of
  `ResolveTimeTool`; a caller should go through the tool, not these functions.
- Adding a new builtin tool here should follow the same shape: a small
  `Tool` impl with a declarative `parameters_schema`, `ToolPolicy::read_only()`
  or the appropriate policy, and pure-function payload building kept separate
  from the `Tool` trait plumbing so it stays unit-testable.
