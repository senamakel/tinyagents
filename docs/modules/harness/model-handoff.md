# Cross-Provider Handoff

> Part of the harness [Model And Provider Feature](model.md) module doc.

A single run's transcript can span more than one provider or model: an
explicit `ModelRequest::model` override, a fallback chain, or a host routing
decision can each hand the next model call a transcript whose assistant
messages a *different* provider produced. Replayed verbatim, that transcript
can carry content the new target rejects outright or cannot make sense of:
a provider-encrypted `ContentBlock::RedactedThinking` block, a signed
`ContentBlock::Thinking` block whose signature only the origin provider can
verify, tool-call ids shaped for the origin provider (Anthropic rejects a
`tool_use`/`tool_result` id outside `^[a-zA-Z0-9_-]{1,64}$`), or an image
block when the target model has no vision input.

Two pieces of vendor (`tinyinference-llm`) state make this detectable:
`AssistantMessage::origin: Option<MessageOrigin>` — the `{provider, api,
model}` that produced the message, stamped by every provider adapter (OpenAI
Chat Completions, OpenAI Responses, every OpenAI-compatible local preset, and
Anthropic) on both a unary response and a stream's terminal `Completed` item
(`None` means a host-authored turn or a pre-`origin` journal replay) — and
`ModelProfile::tool_call_id_pattern` / `max_tool_call_id_len`, the target's
accepted tool-call id shape, when it constrains one (Anthropic's default
profile populates both; OpenAI leaves them `None`).

`tinyagents_harness::agent_loop::handoff_transform::prepare_for_model`
consumes both immediately before a `ModelRequest` is dispatched — a pure pass
that never touches `Turn`/`RunQueue` bookkeeping. An assistant message is
*foreign* when its stamped origin differs from the call's target origin in
`provider`, `api`, or `model`; a message with no origin is foreign only when
it structurally carries content the target cannot accept (so a legacy
journal with plain text is left alone). `Message::User`/`Message::Tool`
carry no origin at all, so an image block in either is downgraded whenever
the target lacks vision input, regardless of which turn produced it.

For a foreign assistant message the transform drops
`RedactedThinking`, converts a signed `Thinking` block to plain text (or
drops it when empty), downgrades images, and rewrites any tool-call id that
does not conform to the target's pattern/length to a sanitized,
de-duplicated replacement — recorded in one id map for the call so every
`Message::Tool` answering a remapped call is rewritten with the same id.
The rewritten assistant message's own `origin` is cleared, since it no
longer verbatim-replays what the source provider produced.

A transcript with nothing foreign (the common case: a run that never
switches provider) costs nothing — the pass returns a borrowed slice with no
allocation. When something does change, the loop emits
`AgentEvent::HandoffTransformApplied { changes }` with the number of
rewritten messages, so an exporter or test can observe exactly when a
handoff rewrite happened.
