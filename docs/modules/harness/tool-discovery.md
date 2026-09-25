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

- `tool_search { query, limit }` — ranks the catalogue's name, split
  identifier, description, top-level property names and `Tool::family`.
  Returns up to `limit` matches (default 5, max 20) as `{name, description,
  parameters}` with the **full** schema, or a "no match" note. See
  [Ranking](#ranking) for what does the ranking. Its description embeds a manifest of every deferred tool:
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

### Ranking

`DeferredCatalog::rank` answers as `ToolDiscoveryPolicy` says:

- With no `ranker` installed (the default), BM25 from `tinytools::rank` —
  free, deterministic, no network. `Bm25Index` and `tokenize` moved to that
  crate so a host ranks with the same arithmetic the bridge does.
- With a host `ranker: Arc<dyn tinytools::ToolRanker>` (a decision model such
  as `tinytools-jev`, or an embedding index), `rank_mode` decides:
  `Ranker` serves it; `Bm25` ignores it; `Compare` serves it and reports the
  BM25 ranking alongside in `ToolSearched.shadow_matched` so the two can be
  judged on live traffic without changing what the model sees.
- A host ranker that fails or returns nothing **falls back to BM25** and
  the reason lands in `ToolSearched.fallback`. A search never errors: an
  error would leave every deferred tool unreachable for the turn.

Every hit the ranker names is resolved through `catalog.get`, so a key the
ranker invented never reaches the model. The catalogue is what the ranker
sees; the model's `query` is the only intent, with an empty `RankContext` —
the model already distilled the turn into it.

### Search and typed promotion

The bridge keeps the first request small. After `tool_search`, each matched
tool's schema is added to subsequent provider requests so the model can call
it with typed arguments. The harness records the added declarations as a
transcript tool-change patch; resume restores only tools still admitted by
the current host allow-list. A new match changes the provider's tools prefix
once. Later requests keep that prefix until another tool is discovered.

### What the bridge does not cover

The deferred catalogue is fixed at run start from the registry and the host
allow-list. Exposure-only middleware (`ContextualToolSelectionMiddleware`,
`DynamicToolSelectionMiddleware`, `ToolPolicyMiddleware::before_model`) shapes
`request.tools` per turn and therefore never sees a deferred tool. Execution-
time gates (`before_tool`, `ToolAllowlistMiddleware`, host authorization) do,
because `tool_call` is unwrapped first. A host that needs per-turn exposure
narrowing of deferred tools should apply it at registration or via the
allow-list.

### Using `ToolPolicyMiddleware::strict` with discovery

The bridge tools (`tool_search`/`tool_call`) are never registered, so a
fail-closed `ToolPolicyMiddleware::strict(policies)` rejects them by default
like any other unclassified name — which would make every deferred tool
undiscoverable. Call
`.exempt_discovery_bridge(true)` to exempt the two reserved names from
classification/side-effect checks:

```rust,ignore
let policy = ToolPolicyMiddleware::strict(registry.policies())
    .exempt_discovery_bridge(true);
```

This is opt-in rather than automatic because it is only safe when `policies`
is the *complete* registry snapshot (as `ToolRegistry::policies()` is): the
exemption only fires for a name with no entry in `policies`, so an incomplete
or stale snapshot could otherwise let a real, side-effecting host tool that
happens to be registered under `tool_search`/`tool_call` bypass strict mode's
fail-closed checks. A host-registered tool under either name always wins over
the intrinsic bridge and is evaluated by its own policy entry regardless of
this flag.

## Events

- `ToolsAdvertised { direct, deferred, schema_bytes }` — once per run after
  `before_agent`: the run's pre-middleware tool surface (the registry-derived
  set before any `before_model` middleware narrows it and before a
  structured-output tool-call fallback, if any, is appended) and what that
  baseline costs in compact-JSON bytes. `direct` counts only `Direct`-exposure
  tool schemas — the two intrinsic bridge schemas are implied by `deferred`
  being nonzero, not folded into `direct` — while `schema_bytes` covers the
  actual wire set (direct schemas plus the bridge, when present). Track it as
  the ceiling a run started with, not as a live per-request wire metric —
  exposure-narrowing middleware (`ToolPolicyMiddleware::before_model`,
  dynamic/contextual selection) can still shrink an individual request below
  it.
- `ToolSearched { call_id, query, matched, ranker, top_confidence, fallback,
  shadow_matched, latency_ms }` and `DeferredToolCall { call_id, tool_name }`
  — every discovery, auditable: which ranker answered, how sure it was, why
  a host ranker was not served, and what BM25 would have said in compare
  mode. `query` is recorded only under `RunPolicy::capture.tool_io`.

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
tools. Admission always validates arguments against the canonical *declared*
schema, never the projected wire schema — the two can diverge in either
direction: compaction only ever removes constraints from what is advertised
(looser on the wire), while `SchemaPreparation::strict` can *add* constraints
(`required` forced, `additionalProperties: false`), making the projected
schema stricter than what admission actually enforces.

## Budgets that count schemas

`SummarizationPolicy::should_summarize_with_tools` and the
`MicrocompactMiddleware` token gate charge `count_tool_schema_tokens` against
their thresholds, so a run whose schemas eat a quarter of the window compacts a
quarter earlier instead of overflowing.

## Prompt-guided models

For models without native tool calling the tool list is rendered into the
system prompt by `tinytools-agent` (see [tool-dialect.md](tool-dialect.md)),
not by this crate: the JSON-in-tag dialect lists each tool's parameter
schema, and the P-Format dialect lists a compact positional call signature
(`read_file[0|<path>|1|<limit>]`). The bridge schemas above go through the same
renderer as any other tool, so `tool_search` / `tool_call` are callable from
either text dialect. `tool::type_signature` / `tool::argument_notes` remain
available as compact TypeScript-style formatters
(`{path: string, limit?: integer}`) for hosts that build their own prompt
text. The native path is unaffected.

## Live proof

`tests/live_tool_deferral.rs` runs the same task against a real model twice
over a 41-tool registry — all `Direct`, then the 40-tool long tail `Deferred`
— and requires both runs to reach `stock_quote`, the deferred run to spend
fewer prompt tokens on its first call, and a searched `stock_quote` to gain
a typed declaration. The earlier baseline measured over OpenRouter (2026-09-19):

| model                       | first-call prompt tokens | total prompt tokens | route  |
|-----------------------------|--------------------------|---------------------|--------|
| `openai/gpt-4.1-mini`       | 3,814 → 878 (−77%)       | 7,692 → 3,275       | search |
| `anthropic/claude-haiku-4.5`| 7,923 → 1,582 (−80%)     | 15,944 → 5,370      | bridge |
| `google/gemini-2.5-flash`   | 2,557 → 837 (−67%)       | 5,161 → 3,743       | search |

Schema bytes on the wire went from 24,725 (41 tools) to 3,791 (3 tools + a
40-entry manifest). "Route" is how the model reached the tool: through
`tool_search`, or straight off the manifest via `tool_call` (Haiku read the
name in the description and skipped the search). A 2026-09-25 live run of
`openai/gpt-4.1-mini` after typed promotion used 3,814 → 904 first-call
tokens and 7,692 → 3,592 total tokens. The separate live promotion case
searched, received a typed `stock_quote` declaration, and invoked it with
an integer `options.limit` in three model calls.

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
