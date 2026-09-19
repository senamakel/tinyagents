# Tool Exposure, Discovery, and Schema Budgets

Tool schemas are re-sent with every model call. A host that registers a
hundred tools pays their full JSON Schema on every turn — measured on one
consumer, 31–112 KB of `tools` against a 34 KB system prompt — before the model
has read a single message. This feature keeps that cost proportional to what a
run actually uses, following the pattern Codex, Hermes, and OpenClaw all
converged on: **register everything, advertise little, let the model discover
the rest.**

## Exposure

`tinytools::Tool::exposure()` classifies every registration:

| Exposure   | In `tools` array | Callable by the model | Findable via `tool_search` |
|------------|------------------|-----------------------|----------------------------|
| `Direct`   | yes              | yes                   | n/a                        |
| `Deferred` | **no**           | yes (by name or via `tool_call`) | yes            |
| `Hidden`   | no               | **no** — answered as an unknown tool | no          |

`ToolRegistry::schemas()` returns only `Direct` tools (name-sorted, as before);
`deferred_schemas()` returns the `Deferred` set. Host code can still reach a
`Hidden` tool through `ToolRegistry::dispatch`; the model cannot, and the
"valid tools" list in an unknown-tool answer never names one.

Mark rarely-needed tools `Deferred`: third-party catalogues (MCP servers,
plugins, connector actions), event-triggered utilities, anything a model reaches
for on a handful of turns a week. Keep the core primitives `Direct` — file,
shell, reply, and above all any *ask-the-user* tool. Hermes measured that
deferring its clarification tool collapsed its use; a model does not search for
a tool it does not know it needs.

## The bridge

When a run has at least one deferred tool (after the host allow-list), the
agent loop appends two intrinsic tools **after** the name-sorted direct set:

- `tool_search { query, limit }` — BM25 over name, split identifier, description
  and top-level property names. Returns up to `limit` matches (default 5, max
  20) as `{name, description, parameters}` with the **full** schema, or a
  "no match" note. Its description embeds a manifest of every deferred tool:
  `- name: first sentence (≤ 60 chars)`, degrading to names only, then to a
  bare count, until it fits `ToolDiscoveryPolicy::manifest_token_budget`
  (default 4,000 tokens).
- `tool_call { name, arguments }` — invokes a deferred tool. The loop
  **unwraps it before admission**, so every `before_tool` hook, allow-list,
  policy middleware and the host authorization gate see the real name and
  arguments; `ToolStarted` names the real tool. A revealed tool is also
  callable directly by its own name — the registry already holds it.

Neither is a registered tool; a host that registers its own `tool_search` or
`tool_call` keeps it. Both are governed by `RunPolicy::discovery`
(`ToolDiscoveryPolicy { enabled, manifest_token_budget, default_limit,
max_limit }`). With `enabled: false` deferred tools are neither advertised nor
searchable, but a direct call by name still runs: deferral only ever subtracts
from the wire, never from what the host registered.

### Why a bridge and not hydration

OpenClaw appends a revealed schema to the request's `tools` for the rest of the
run. That is simpler for the model but every reveal changes the `tools` array —
and Anthropic caches *tools → system → messages* as one prefix, OpenAI's
`prompt_cache_key` likewise hashes the tools. A run with `protect_prompt_prefix`
would lose its cache on every discovery. With the bridge the `tools` array is
byte-identical for the whole run (asserted by
`tests/tool_deferral.rs`); a reveal costs one tool result. Hermes and Codex
made the same trade.

### What the bridge does not cover

The deferred catalogue is fixed at run start from the registry and the host
allow-list. Exposure-only middleware (`ContextualToolSelectionMiddleware`,
`DynamicToolSelectionMiddleware`, `ToolPolicyMiddleware::before_model`) shapes
`request.tools` per turn and therefore never sees a deferred tool. Execution-
time gates (`before_tool`, `ToolAllowlistMiddleware`, host authorization) do,
because `tool_call` is unwrapped first. A host that needs per-turn exposure
narrowing of deferred tools should apply it at registration or via the
allow-list.

## Events

- `ToolsAdvertised { direct, deferred, schema_bytes }` — once per run after
  `before_agent`: what went on the wire and what it costs in compact-JSON
  bytes. This is the number a prompt-budget ratchet should track.
- `ToolSearched { call_id, query, matched }` and
  `DeferredToolCall { call_id, tool_name }` — every discovery, auditable.

## Schema budgets

`RunPolicy::tool_schemas: Option<SchemaPreparation>` (default `None` = verbatim)
projects every advertised and deferred schema before it is sent: `$ref`
resolution, provider keyword stripping, optional strict mode, and now an
optional `SchemaCompaction`:

- `max_description_bytes` — clip the description (char boundary, `…`).
- `max_schema_bytes` — run the ladder until the parameters fit: prune
  unreachable `$defs` → strip nested property descriptions → strip all property
  descriptions → drop definition tables → collapse objects deeper than 3 →
  drop `anyOf`/`oneOf`/`allOf`. Every rung keeps the top-level argument surface.

`SchemaCompaction::THIRD_PARTY` (5,000 B / 1,000 B) is Codex's budget for MCP
tools. Admission still validates arguments against the *declared* schema, which
is never looser than the projected one.

## Budgets that count schemas

`SummarizationPolicy::should_summarize_with_tools` and the
`MicrocompactMiddleware` token gate charge `count_tool_schema_tokens` against
their thresholds, so a run whose schemas eat a quarter of the window compacts a
quarter earlier instead of overflowing.

## Prompt-guided models

For models without native tool calling the tool list is rendered into the
system prompt. It now uses a compact TypeScript-style signature per tool —
`Arguments: {path: string, limit?: integer}` plus one note per described
top-level argument — instead of the raw JSON Schema (`tool::type_signature`,
`tool::argument_notes`). Constraints and nested descriptions are dropped; the
native path is unaffected.

## Live proof

`tests/live_tool_deferral.rs` runs the same task against a real model twice
over a 41-tool registry — all `Direct`, then the 40-tool long tail `Deferred`
— and requires both runs to reach `stock_quote`, the deferred run to spend
fewer prompt tokens on its first call, and its `tools` array to be
byte-identical throughout. Measured over OpenRouter (2026-09-19):

| model                       | first-call prompt tokens | total prompt tokens | route  |
|-----------------------------|--------------------------|---------------------|--------|
| `openai/gpt-4.1-mini`       | 3,814 → 878 (−77%)       | 7,692 → 3,275       | search |
| `anthropic/claude-haiku-4.5`| 7,923 → 1,582 (−80%)     | 15,944 → 5,370      | bridge |
| `google/gemini-2.5-flash`   | 2,557 → 837 (−67%)       | 5,161 → 3,743       | search |

Schema bytes on the wire went from 24,725 (41 tools) to 3,791 (3 tools + a
40-entry manifest). "Route" is how the model reached the tool: through
`tool_search`, or straight off the manifest via `tool_call` (Haiku read the
name in the description and skipped the search). The extra model call the
deferred run spends is already paid for on the first turn.

```text
TOOL_DEFERRAL_LIVE=1 cargo test -p tinyagents-integration-tests \
    --test live_tool_deferral -- --nocapture
```

## Marking a tool deferred

```rust
use tinytools::{Tool, ToolExposure};

#[async_trait::async_trait]
impl Tool for StockQuote {
    fn name(&self) -> &str { "stock_quote" }
    fn description(&self) -> &str { "Fetch the latest price for a ticker symbol." }
    fn parameters_schema(&self) -> serde_json::Value { /* … */ }
    fn exposure(&self) -> ToolExposure { ToolExposure::Deferred }
    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<tinytools::ToolResult> { /* … */ }
}
```

Nothing else changes: register it as usual and the loop advertises the bridge.
