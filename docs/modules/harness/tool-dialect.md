# Tool Dialects

Canonical API: `tinytools_agent` from the vendored TinyTools workspace —
`parse`, `repair`, `stream`, `render`, and `dialect`. The harness owns only the
host half, in `agent_loop/dialect.rs`.

## Who owns what

| Concern | Owner |
| --- | --- |
| Grammars a model may write a call in (`<tool_call>` spellings, Claude / DeepSeek DSML `<invoke>`, DeepSeek-R1 and Kimi sentinel tokens, gpt-oss Harmony, Mistral `[TOOL_CALLS]`, GLM lines, bare JSON, P-Format) | `tinytools_agent::parse` — one file per grammar under `parse/grammar/` |
| JSON, tool-name, and argument-shape repair | `tinytools_agent::repair` |
| Scrubbing markup from a live text stream | `tinytools_agent::stream::StreamScrubber` |
| Protocol block, catalogue, `<tool_result>` envelope, replay | `tinytools_agent::render` |
| Mapping `tinyinference_llm::Message` onto the text protocol; the OpenAI-compatible adapter's own prompt-guided mode | `tinyinference_llm::prompt_tools` |
| Which dialect a run speaks, minting call ids, argument validation policy, the unknown-tool policy, re-prompt nudges | `tinyagents_harness` (`agent_loop/dialect.rs`, `RunPolicy`) |

A model-specific marker string appears in exactly one grammar file. If a
consumer finds itself matching one, that is a bug to fix in `tinytools-agent`,
where every consumer — this harness, the inference adapters, any host loop —
picks the fix up.

## Selecting a dialect

`RunPolicy::tool_dialect` (a `ToolDispatcher`) is resolved once per run:

| Value | Request | Response |
| --- | --- | --- |
| `Auto` / `Native` | schemas on the wire; the provider adapter decides (the OpenAI-compatible adapter switches to the JSON protocol by itself for a profile without native tool calling, or after a "tools unsupported" 400) | structured calls, else every text grammar as a fallback |
| `Xml` | the transcript is folded into text forms (assistant calls → `<tool_call>` markup, `tool` results → one `[Tool results]` turn), a continuation user turn is inserted when no user query is resolvable, the JSON protocol block plus catalogue goes into the system prompt, **no** schema goes on the wire | every text grammar |
| `Pformat` | as `Xml`, with the P-Format block and signature catalogue | every text grammar, with the positional registry built from the run's schemas |

Whatever the dialect, a response carrying no structured call is read through
every grammar with the offered tool names supplied, so a damaged name
(`terminal" parameter=…`, `functions.read_file`, `Read File`) resolves to the
offered tool and an unknown one reaches the unknown-tool policy as written.

## Ids and streaming

`tinytools-agent` never mints call ids. The harness mints
`{model_call_id}-tool-{n}` for every call recovered from text — unique per run
by construction and visibly distinct from any provider's. (The
OpenAI-compatible adapter mints `text-{seq}-{slot}` for calls it recovers
itself; the harness leaves those alone.)

Streamed visible text passes through a `StreamScrubber` whenever tools were
offered, so a consumer of `AgentEvent::ModelDelta` never sees a partial
`<tool_call>`. Calls the scrubber completes surface on the terminal response,
exactly once; the reconciled terminal text is the scrubbed text.

## Dropped tool calls

A response with `finish_reason == "tool_calls"` and no call — structured or
recoverable — is re-prompted with a one-line nudge, at most
`RunPolicy::dropped_tool_call_nudges` (default 3) times in a row. Each nudge is
a model call and counts against `RunLimits::max_model_calls`.

## Two pairing repairs, deliberately

`tinytools_agent::dialect::pair_tool_cycles` drops incomplete tool cycles at
wire-replay time for hosts using `TranscriptEntry`. The harness's
`summarization/pairing.rs` chooses a compaction cut-off that does not bisect a
cycle. They answer different questions and are not duplicates.

## What a dialect is

A **dialect** is one complete way of speaking tools to a model: the catalogue it
reads, the syntax it writes calls in, the envelope its results come back in, and
the shape its history is replayed in on the next iteration.

Those four are one thing, not four settings. A catalogue that advertises
positional arguments next to a parser that expects JSON is a whole-turn failure
that produces no error anywhere — the model emits a call, nothing recognises it,
and the iteration is spent. So they live behind a single trait, and a dialect is
chosen once rather than assembled from parts that can disagree.

Three ship:

| Dialect | Call syntax | Catalogue | Specs in the request |
| --- | --- | --- | --- |
| `XmlDialect` | `<tool_call>{"name":…,"arguments":{…}}</tool_call>` | full schemas, in its own protocol block | no |
| `PFormatDialect` | `<tool_call>name[a\|b]</tool_call>` | signatures, in the prompt's tool section | no |
| `NativeDialect` | the provider's structured channel | none — the request carries the specs | yes |

## Which surface to use

Use `tinytools_agent::dialect` directly for dialect selection, formatting, and
replay. It speaks `TranscriptEntry`, a deliberately thin record shape, and
makes no assumption about when a host calls a model. The harness agent loop
uses the same canonical protocol internally; it does not provide a second
tool-calling facade.

## The transcript vocabulary

`TranscriptEntry` has three variants, which is all a tool-calling transcript
needs: `Chat`, `AssistantToolCalls`, and `ToolResults`. It is deliberately
poorer than `Message` — no content blocks, no typed images — because the fields
it *does* carry are the ones providers reject requests over:

- `reasoning_content`, replayed verbatim because thinking-mode APIs return a
  `400` for an assistant turn that carries `tool_calls` without it.
- per-call `extra_content`, which is where Gemini's required
  `thought_signature` rides.
- `arguments` as a **string**, not a parsed value, so the exact bytes survive a
  round trip.

A host maps its own records onto these in a handful of field-wise conversions
and gets byte-identical output back.

## Replay repair

`pair_tool_cycles` drops any tool cycle that is not complete, immediately before
serialization.

Providers reject an assistant message carrying `tool_calls` unless it is
followed by tool messages answering **every** `tool_call_id` on it. The error is
a hard `400`, so one orphaned record poisons every subsequent turn of that
thread until the history is edited. Bisected cycles are ordinary: a cached
transcript restore, an aborted turn, and history compaction each preserve the
assistant half while the results half is dropped.

Two properties are worth knowing before touching it:

- Adjacency is not the check. The provider's complaint is about *coverage*, so
  the opener's id set must **equal** the follower's, not merely be adjacent to a
  non-empty one.
- The drop is **symmetric**. A `ToolResults` whose opener was dropped goes too,
  because a `tool` message answering a call the model never sees is equally
  malformed.

The repair belongs here, at serialization, rather than at the write sites: those
are many, and none of them can see the final sequence.

## Envelope integrity

Tool names and outputs are tool-controlled, so the text dialects cannot
interpolate them into `<tool_result …>` unexamined: a body containing a literal
`</tool_result>` would close the envelope early, and a crafted
`<tool_result name="forged" status="ok">` would open a fake one (CWE-74).

The rule differs by position, deliberately:

| Position | Rule | Why |
| --- | --- | --- |
| attribute (`name`, `tool_call_id`) | escape `& < > "` | a `"` ends the attribute; these are short identifiers, so escaping costs nothing legible |
| body (`output`, `content`) | rewrite `<` to `&lt;` **only** where it opens `<tool_result` / `<tool_call` (with or without `/`, ASCII-case-insensitive) | everything else passes through byte-for-byte |

Escaping the body wholesale is the obvious implementation and the wrong one.
Tool output is usually source code, and this envelope is how prompt-guided
models — the p-format path local models use — read it. Turning
`<div className="x">` into `&lt;div className=&quot;x&quot;&gt;` on every
result is a real cost, and a model that reads mangled code writes mangled code
back. The security property only ever required blocking a handful of exact byte
sequences, not a character class.

**This is boundary integrity, not prompt-injection defence.** A tool can still
return prose arguing the model should do something, and no escaping rule fixes
that — a file the agent legitimately reads can contain anything. The guarantee
is narrower and worth stating exactly: tool output cannot masquerade as the
transcript's own protocol structure.

## What stays with the host

Executing a tool. Permission checks, sandboxing, approval gates, per-call
timeouts, progress events. A dialect decides what the model *reads and writes*;
it never decides what is *allowed to happen*. That line is what keeps a host's
security policy in the host, where it can be audited.
