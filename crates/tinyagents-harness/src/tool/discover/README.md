# `tool::discover`

On-demand tool discovery for the agent loop. The design doc is
[`docs/modules/harness/tool-discovery.md`](../../../../../docs/modules/harness/tool-discovery.md);
this file is the map of the module.

| File          | Owns                                                                 |
|---------------|----------------------------------------------------------------------|
| `types.rs`    | `ToolDiscoveryPolicy` (the knobs), `DeferredCatalog` (a run's deferred schemas, BM25-indexed, name-sorted), `DeferredTool` |
| `index.rs`    | `Bm25Index` + `tokenize` — ranking over `(sort_key, text)` pairs, knows nothing about tools |
| `manifest.rs` | `render_manifest` — the budgeted listing inside `tool_search`'s description: full → names → count |
| `bridge.rs`   | The two intrinsic tools: `bridge_schemas`, `answer_tool_search`, `unwrap_tool_call` |
| `test.rs`     | Unit tests for all of the above                                      |

The agent loop (`agent_loop/run_loop.rs`, `agent_loop/tools.rs`) is the only
consumer: it builds the catalogue once per run, appends the bridge schemas
after the name-sorted direct set, answers `tool_search` from the catalogue
without running a tool, and rewrites a `tool_call` to its target *before*
`before_tool` so every gate sees the real tool.

Invariants worth keeping:

- The `tools` array never changes within a run because of discovery. Revealed
  schemas travel in a tool result, not in `tools` (cache stability).
- Everything rendered from the catalogue is name-sorted and deterministic.
- Discovery only subtracts: a deferred tool is callable by name whether or not
  the bridge is enabled; a `Hidden` tool is never callable by the model.
- The manifest is bounded by `manifest_token_budget`; the search answer clips
  descriptions to 500 chars and `limit` to `max_limit`.
