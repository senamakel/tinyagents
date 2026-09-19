# harness::tool

Harness-side registration, projection, and execution support for canonical
tools. Tool *vocabulary* (`Tool`, `ToolCall`, `ToolResult`, `ToolPolicy`,
`ToolTimeout`, `WorkspaceDescriptor`, ...) belongs to the `tinytools` crate;
this module owns only the host concerns that sit between a declared tool and
a live agent run: name lookup, provider-schema projection, prompt-guided
(text-mode) tool calling, injected-argument enforcement, timeout resolution,
and the explicit recursive-dispatch handoff for tools that must see the typed
parent run.

## Public surface

### Registration and dispatch (`mod.rs`)

- `ToolRegistry<State, Ctx>` — a name-keyed registry of canonical tools.
  `register` wraps a plain `tinytools::Tool` in `dispatch`-based execution;
  `register_dispatch` accepts an explicit `ToolDispatch` for the rare tool
  that needs the typed parent state/context (currently sub-agents). `names`,
  `schemas`, `declared_specs`, and `policies` project the registry for the
  model, the transcript, and the policy layer.
- `ToolDispatch<State, Ctx>` — the trait a typed recursive dispatcher
  implements: its own `tool()` declaration, `output_origin()` for the host's
  content-screening boundary, `injected_arguments()` for host-authoritative
  values, `call_options()`, and the `execute()` entry point that receives the
  full typed `State` and `RunContext<Ctx>`.
- `provider_schema` — converts a canonical `ToolSpec` into the inference
  provider's `ToolSchema`, stripping injected arguments first.

### Execution context (`types.rs`)

- `ToolExecutionContext` — the harness-owned bridge from a live run to
  `tinytools::ToolRunContext`: run id, thread id, depth, output-token budget,
  event sink, cancellation token, streaming flag, and an optional workspace
  descriptor. Cancellation, events, and run identity stay owned by the
  harness; a tool that needs a real child `RunContext` goes through the
  explicit dispatch seam instead.

### Injected arguments (`injected.rs`)

- `strip_injected_arguments` — removes host-only keys from model-supplied
  arguments before validation, logging any that were actually present (a
  forgery attempt).
- `project_injected_arguments` — removes injected keys from a schema's
  `properties` **and** `required` so a model is never told to supply an
  argument it cannot see.

The security-critical ordering (strip → validate → inject → invoke) is
documented on the module and enforced by the agent loop's tool-execution
path, not by this module itself.

### Schema cleaning (`schema.rs`)

- `SchemaCleanr` — normalizes a JSON Schema for a target provider: resolves
  local `$ref`/`$defs`, strips provider-unsupported keywords, flattens
  same-typed literal unions, drops nullable variants, converts `const` to
  `enum`, and breaks circular local refs.
- `CleaningStrategy` — `Gemini` / `Anthropic` / `OpenAI` / `Conservative`,
  each with its own `unsupported_keywords()`.
- `GEMINI_UNSUPPORTED_KEYWORDS` — the (most restrictive) keyword list Gemini
  rejects.

### Provider projection seam (`schema_prepare.rs`)

- `prepare_tool_schemas` / `prepare_tool_schema` / `prepare_parameters` — the
  seam a provider adapter uses instead of reading `Tool::schema()` directly:
  normalize → clean (via `SchemaCleanr`) → optional strict-mode sanitizer, in
  that order.
- `SchemaPreparation` — `strategy` + `strict`; `gemini()`/`anthropic()`/
  `openai()`/`conservative()` constructors plus `with_strict()`.
- `normalize_parameters` — replaces a missing/non-object parameter schema
  with an empty object schema so a tool that takes no arguments cannot break
  a request.
- `require_all_properties` / `set_additional_properties_false` — the two
  halves of OpenAI strict-mode sanitization.

### Prompt-guided (text-mode) tool calling (`prompt.rs`)

For provider adapters whose model profile has no native tool calling:

- `prompt_tool_instructions` / `with_prompt_tool_instructions` — embed the
  `<tool_call>` protocol and the tool catalogue into the system prompt.
- `coalesce_prompt_tool_results` — renders structured assistant tool calls
  back into `<tool_call>` text and folds consecutive tool results into one
  `[Tool results]` user turn.
- `ensure_resolvable_user_turn` — inserts a content-free continuation user
  turn when none is present, so chat templates that hard-require a locatable
  user query (e.g. Qwen 3's) don't reject the request outright.
- `parse_prompt_tool_calls_from_text` — extracts `<tool_call>` blocks (and
  DeepSeek's native delimiter) from completed text into `ToolCall`s.
- `ToolCallStreamScrubber` — the streaming counterpart: scrubs `<tool_call>`
  markup out of live text deltas as fragments arrive, holding back any tail
  that could still grow into an opening delimiter.
- `should_recover` / `apply_prompt_tool_calls` — decide whether text-mode
  recovery should run over a completed `ModelResponse` (always for
  prompt-guided models, as a fallback for native models that returned no
  structured calls) and perform it.
- `SYNTHETIC_CALL_ID_PREFIX` / `next_synthetic_call_id` — mint
  process-unique, human-readable ids (`ptc_{sequence}_{slot}`) for a recovered
  call, since a per-response counter collides across turns.

### Timeouts (`timeout.rs`)

- `ToolTimeoutSettings` — shared, atomically-updateable timeout policy: an
  inherited default (`ToolTimeout::Inherit`), clamped `min_ms..=max_ms`
  bounds, and `grace_ms` scheduling slack. `set_inherited_timeout_ms` lets a
  host apply a config/operator override to every harness sharing the
  settings without rebuilding them.
- `ResolvedToolTimeout` — the `resolve()` output: enforced `deadline` plus
  unpadded `budget_ms` reported to callers and observability.

### Tool selection (`select/`)

See `select/README.md` (or the module doc on `select/mod.rs`) for the
prompt-driven ranker that narrows a large tool catalogue before it reaches
the model. Re-exported here as `pub mod select` and via `pub use select::*`.

## Files

| File | Role |
| --- | --- |
| `mod.rs` | `ToolRegistry`, `ToolDispatch`, `provider_schema`; wires the submodules together. |
| `types.rs` | `ToolExecutionContext`. |
| `injected.rs` | Injected (host-only) argument stripping and schema projection. |
| `schema.rs` | `SchemaCleanr`, `CleaningStrategy`; low-level JSON Schema cleaning. |
| `schema_prepare.rs` | Provider projection seam built on `schema.rs`; strict-mode sanitizer. |
| `prompt.rs` | Prompt-guided tool-call protocol: instructions, coalescing, parsing, streaming scrub. |
| `timeout.rs` | `ToolTimeoutSettings`, `ResolvedToolTimeout`. |
| `select/` | Prompt-driven tool ranking (own submodule; see its README/module doc). |
| `*_test.rs`, `test.rs` | Unit tests colocated by concern, listed via `#[path = "..."]` or `mod ..._test;`. |

## Operational constraints

- **Injected-argument ordering is security-critical.** Strip must run before
  validation, which must run before injection, which must run before
  invocation — see `injected.rs`'s module doc for why the order cannot be
  relaxed.
- **Schema cleaning must run before strict-mode sanitization**
  (`prepare_parameters`), so `required` is computed from the resolved
  property set rather than one still hidden behind an unresolved `$ref`.
- **`ToolCallStreamScrubber` is stateful and per-stream.** Create one per
  response stream, feed every fragment through `feed`, and call `flush` once
  at the end to drain the final safe remainder; reusing one across streams or
  skipping `flush` will misplace or drop trailing text.
- **Synthetic call ids are process-global, not per-response**, specifically
  so two recovered calls in different turns of the same run never collide;
  do not reset or shard `SYNTHETIC_CALL_SEQUENCE`.
- Canonical tool vocabulary, execution, and policy enforcement itself remain
  in `tinytools`; this module never redeclares them, only bridges them to a
  live harness run.
