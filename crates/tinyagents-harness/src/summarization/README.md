# harness::summarization

Explicit message trimming, summarization, and compression policies — the
harness's direct answer to "context rot." Context-window-aware gating
(`SummarizationPolicy`) decides *when* a run's transcript has grown large
enough to compress; the trimming/summarization primitives decide *what* to
keep verbatim versus fold into a summary. Every policy decision is an
explicit, inspectable data type: callers choose when to call, what to pass,
and how to handle the result — nothing here runs implicitly inside the agent
loop.

## Design

- **Token estimation** (`estimate_tokens`, `TokenEstimate`) is a cheap
  `chars / 4` heuristic, not a real tokenizer — good for budget checks with a
  ~30% error margin, not for exact accounting.
- **Trimming** (`trim.rs`) is synchronous and LLM-free: it drops messages
  from a slice according to a `TrimStrategy` (`KeepLast`, `KeepFirstAndLast`,
  `MaxTokens`). Every strategy is pairing-safe by default via
  `TrimOptions`/`TokenTrimPolicy`.
- **Summarization** (`Summarizer` trait, `ConcatSummarizer`,
  `SummarizationPolicy`) is async and typically LLM-backed.
  `SummarizationPolicy::plan` decides the split between `to_summarize` and
  `to_keep`; a `Summarizer` then condenses the former into a `SummaryRecord`
  carrying `CompressionProvenance`.
- **Tool-call pairing** (`pairing.rs`) is the structural safety net both of
  the above rely on: a naive length-based cut point routinely separates an
  assistant tool-call turn from the tool results answering it, producing a
  transcript OpenAI/Anthropic reject outright. `find_safe_cutoff_point` and
  friends move the cut to preserve pairing (or repair an already-broken one).
- **Rendering** (`render.rs`) turns any `Message` — including tool calls,
  tool results, and reasoning blocks that `Message::text()` drops — into
  summarizable text, so `ConcatSummarizer`'s default output isn't a column of
  bare role labels for a tool-driven run.

## Public surface

### Token estimation
- `estimate_tokens(text) -> u64` / `TokenEstimate` — the heuristic counter.

### Trimming (`trim.rs`)
- `trim_messages(messages, strategy) -> Vec<Message>` — the common case,
  equivalent to `trim_messages_with(.., &TrimOptions::default())`.
- `trim_messages_with` / `trim_messages_to_token_budget_with` — the
  fully-configurable entry points (`TrimOptions`, `TokenTrimPolicy`).
- `TrimStrategy` — `KeepLast(n)` / `KeepFirstAndLast { first, last }` /
  `MaxTokens(limit)`.
- `TrimOptions` — `repair_tool_pairs` (default on — keep it on),
  `start_on`/`end_on` role boundaries (`starting_on`/`ending_on`/
  `without_pair_repair` builders).
- `TokenTrimPolicy` — `strict`/`preserve_system`/`drop_leading_orphan_tools`
  builders for `trim_messages_to_token_budget_with`.
- `MessageRole` — role-only view of a `Message`, used by the `start_on`/
  `end_on` boundaries.

### Summarization
- `Summarizer` (trait, object-safe) — `summarize(&self, messages) ->
  Result<SummaryRecord>`.
- `ConcatSummarizer` — deterministic, LLM-free default: concatenates
  `render_message_for_summary` output with positional `msg-N` ids.
- `SummarizationPolicy` — `from_profile`/`with_context_window`/
  `with_threshold_fraction` builders; `trigger_budget()`,
  `should_summarize(messages)`, and `plan(messages) -> (to_summarize,
  to_keep)`.
- `SummaryRecord` — `summary: Message` + `provenance: CompressionProvenance`.
- `CompressionProvenance` — `source_ids`, `original_token_estimate`,
  `summary_token_estimate`, `reason`.

### Tool-call pairing (`pairing.rs`, `pub mod pairing`)
- `find_safe_cutoff_point`, `tool_pairing_is_intact`,
  `retract_orphan_tool_calls`, `advance_past_orphan_tools`,
  `is_tool_calling_assistant` — re-exported at `summarization::*` as well as
  reachable via `summarization::pairing::*`.

### Rendering
- `render_message_for_summary(&Message) -> String`.

## Files

| File | Role |
| --- | --- |
| `mod.rs` | `estimate_tokens`, `Summarizer for ConcatSummarizer`, `SummarizationPolicy` impls; wires the submodules together. |
| `types.rs` | All public struct/enum/trait definitions. |
| `pairing.rs` | Tool-call-pairing-safe cut-point repair. |
| `render.rs` | `render_message_for_summary`. |
| `trim.rs` | `trim_messages`/`trim_messages_with`/`trim_messages_to_token_budget_with`. |
| `test.rs` | Coverage for token estimation, trim strategies, pairing repair, policy triggering/planning, and `ConcatSummarizer`. |

## Key invariants

- **`repair_tool_pairs` (on `TrimOptions`) should stay on.** Turning it off
  restores transcripts providers reject outright (`400` on a stray
  `role:"tool"`, or a `tool_result` with no matching `tool_use`); it exists
  only for callers that repair pairing themselves or tests observing the
  unrepaired cut.
- **`SummarizationPolicy::plan`'s split point is never a blind
  `len - keep_last` index.** It always routes through
  `find_safe_cutoff_point`, so `keep_last` is a *minimum*, not an exact
  count, and `to_keep` is always a provider-acceptable slice.
- **System messages are never placed in `to_summarize`** and are (by
  default) never dropped by trimming either — they carry persistent
  instructions.
- **`ConcatSummarizer` renders through `render_message_for_summary`, not
  `Message::text()`**, specifically so a tool-driven run's compacted history
  keeps its tool calls/results/reasoning instead of collapsing to empty
  strings.
- This module never calls into the agent loop or decides *when* to invoke a
  `Summarizer` on its own — `should_summarize`/`plan` are pure decisions the
  caller (the agent loop, a middleware) acts on.

## Relation to neighbouring modules

`SummarizationPolicy::from_profile` reads
`tinyinference_llm::model::ModelProfile::max_input_tokens`. `token_estimation::estimate_slice_tokens`
(`crate::token_estimation`) backs the whole-slice estimates used by
`should_summarize`/`trigger_budget`. Consumers are the agent loop / context
middleware that decide when a run's transcript needs compacting before the
next model call.
