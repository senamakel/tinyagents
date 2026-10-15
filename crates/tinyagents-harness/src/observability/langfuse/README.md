# langfuse

Langfuse ingestion exporter for durable [`AgentObservation`](../README.md)s.
Converts a slice of observations for one run (or run tree) into a Langfuse
`/api/public/ingestion` batch and posts it — either directly to a
self-hosted/cloud Langfuse instance, or through the TinyHumans backend proxy
that injects Langfuse credentials server-side.

This is a plain library call, not an `EventListener`: nothing in
`observability::mod` wires it into the live event path automatically. A
caller reads observations back out of a `HarnessEventJournal` (or collects
them from a `JournalSink`) and exports them explicitly, typically at a
natural export boundary (end of run, end of turn, or on a timer).

## Public surface

- `LangfuseClient` — the exporter. Construct with `::direct` (Basic Auth
  against a Langfuse origin), `::proxy` (Bearer token against the TinyHumans
  backend), or `::from_env` (reads `LANGFUSE_BASE_URL`/`LANGFUSE_PUBLIC_KEY`/
  `LANGFUSE_SECRET_KEY`, or `TINYHUMANS_LANGFUSE_PROXY_URL`/
  `TINYHUMANS_AUTH_TOKEN` when the proxy URL is set).
- `LangfuseClient::send_observations` — builds and sends a trace's legacy
  ingestion batch (not accepted for traces by Langfuse v4); `build_ingestion_batch` does the same without sending, for callers
  that want to inspect or batch the payload themselves.
- `LangfuseClient::create_score` / `build_score_batch` — attaches a
  post-hoc evaluation (`LangfuseScore`) to an already-exported trace or
  observation.
- `LangfuseClient::send_otlp` / `otlp_endpoint` — posts an OTLP/HTTP JSON
  trace request (for example from `trace_export::otlp::otlp_requests`) to
  `/api/public/otel/v1/traces`, or `/telemetry/langfuse/otel/v1/traces`
  through the proxy, with `x-langfuse-ingestion-version: 4`
  (`LANGFUSE_INGESTION_VERSION_HEADER`) to select Langfuse v4's write path.
  A 2xx answer whose OTLP `partialSuccess.rejectedSpans` is non-zero is an
  error. OTLP is Langfuse's supported trace path; the legacy ingestion API is
  sunset on Langfuse Cloud on 2026-11-16.
- `LangfuseClient::send_observations_otlp` / `send_batch_otlp` — the v4
  export: build the same trace as `build_ingestion_batch`, convert it with
  `trace_export::ingestion_otlp` and send each request through `send_otlp`.
  Use these instead of `send_observations`.
- `LangfuseClient::send_batch` — the shared HTTP transport (auth header,
  endpoint normalization, `207 Multi-Status` handling); reused by other
  exporters (e.g. `graph::observability`) that build their own `{"batch":
  [...]}` payload.
- `LangfuseAuth`, `LangfuseTraceConfig`, `LangfuseScore`, `LangfuseScoreValue`
  — configuration and score value types.
- `clean_nulls`, `iso_ms` (`#[doc(hidden)]`, re-exported through
  `super::observability`) — payload helpers shared with the graph
  observability exporter so null-pruning and ISO-8601 timestamp formatting
  stay identical across both.

## File map

| File | Role |
| --- | --- |
| `mod.rs` | `LangfuseClient` impl: endpoint resolution, batch building (trace/run-span/observation-event projection), score batches, the shared `send_batch` transport, and the `clean_nulls`/`iso_ms` helpers. |
| `types.rs` | Plain data types: `LangfuseAuth`, `LangfuseTraceConfig`, `LangfuseScoreValue`, `LangfuseScore` (with builder methods), `LangfuseClient`'s fields. |
| `mod_tests.rs` | Unit tests for batch shape, trace-id resolution, run-span nesting, and score payloads. |

## Trace shape

One export batch always contains: a `trace-create`, one `span-create` per
distinct `run_id` present in the observations (first-seen order), and one
observation event per non-lifecycle event. `RunStarted`/`RunCompleted`/
`RunFailed` are folded into their run's span (start/end time, error status)
rather than emitted as standalone events, so the trace renders a nested
agent → sub-agent → generation/tool tree that mirrors LangChain's callback
run hierarchy — a sub-agent run exported in a *separate* batch still nests
under its parent's span, because run-span ids are deterministic
(`{trace_id}:run:{run_id}`) rather than server-assigned.

## Operational constraints

- **Observation ids must be trace-namespaced.** Langfuse upserts
  observations by `id` project-wide, but a call's `call_id` is only unique
  within one run and is reused every turn. `scoped_observation_id` prefixes
  it with the trace id so re-ingesting the same run is idempotent without
  colliding across turns or threads; the raw `call_id` still rides in
  `metadata` for in-run correlation.
- **`generation-create` needs the model name from a sibling event.**
  `ModelCompleted` does not carry the model id; `collect_call_models`
  pre-scans the batch for the matching `ModelStarted` so the generation's
  `body["model"]` is populated — without it Langfuse cannot price the call.
- **Tool calls are `span-create`, not `tool-create`.** Older/self-hosted
  Langfuse rejects an unrecognized `tool-create` type and silently drops the
  observation.
- **Payload-free mode still produces a usable trace.** Only lineage, offset,
  and event kind are embedded in generic event metadata — never the full
  `AgentEvent` — to avoid roughly doubling batch bytes and growing
  quadratically with a long run (risking Langfuse's ~3.5MB batch cap).
- `send_batch` treats HTTP `207 Multi-Status` as a partial failure when the
  response carries a non-empty `errors` array, rather than reporting success
  because the outer request returned 2xx/207.
- This client performs real network I/O; like the other persisting sinks in
  `observability`, a failed export is surfaced as an `Err` to the caller —
  it is the caller's responsibility to decide whether that should block a
  run (it normally should not).
