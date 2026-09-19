# Feature and E2E Test Matrix

This matrix maps user-facing TinyAgents features to their integration coverage.
It was audited against the public exports in every workspace package and the
tests in `crates/tinyagents-integration-tests/tests/`. A feature is marked
**covered** only when an external-crate test exercises the public API; unit
tests remain valuable but do not close an E2E gap.

## Current coverage

| Surface | Existing integration coverage | Status |
| --- | --- | --- |
| Graph construction, routing, fan-out, checkpoints, interrupts, export, streaming, subgraphs, parallelism | `graph_durable.rs`, `e2e_complex_graph.rs`, `e2e_durable_interrupt.rs`, `e2e_graph_export.rs`, `feature_graph_{routing,fanout,parallel,streaming}.rs` | Covered |
| Graph goals, todo board, task dispatch, subagents, observability | `e2e_graph_{goals,todos,task_dispatch,subagent_node}.rs`, `e2e_observability.rs` | Covered; delegation durability remains unit-only |
| Harness loop, tools, structured output, middleware, retry, cache, streams, subagents, hosted execution | `harness_agent_loop.rs`, `feature_harness_*`, `e2e_{middleware,streaming_cancel,tool_policy,subagents}.rs`, `wave2_*` | Covered; optional features remain gaps |
| Language parsing, compilation, resolution, binding, RAG | `language_pipeline.rs`, `e2e_language_contracts.rs`, `e2e_rag_pipeline.rs`, `feature_language_*` | Covered; extended grammar is parser/compiler-only |
| Registry catalog, capability binding, diagnostics, observability | `e2e_registry_binding.rs`, `e2e_registry_observability_contracts.rs`, `feature_registry_{catalog,diagnostics}.rs` | Covered; router-to-runtime fallback is a gap |
| Session records, retention, search, lifecycle, transcripts | `e2e_session_lifecycle.rs`, `feature_session_retention.rs`, `feature_session_transcript.rs`, persistence contracts | Covered; run-ledger workflows remain gaps |
| Orchestration workflow and teams | `e2e_orchestration_workflow.rs`, `e2e_orchestration_teams.rs` | Happy paths covered; workflow recovery remains a gap |

## Prioritized execution backlog

### Completed P0 scenarios

| Feature | Evidence of gap | Test to add |
| --- | --- | --- |
| Durable orchestration teams | `e2e_orchestration_teams.rs` | Covers a dependency chain, claim/complete evidence, direct and broadcast messages, shutdown claim release, durable reload, and lifecycle events. |
| Hosted harness invocation | `feature_harness_hosted_invocation.rs` | Covers composed/screened input, host observer attribution, and public streamed failure sanitization. |
| Session transcript persistence | `feature_session_transcript.rs` | Covers append, compaction, interrupted partials, replay/display projections, Markdown, thread summaries, and `FileTranscriptHistory` reopen. |
| Registry runtime router | `ModelRouter`/`WorkloadRoute` only have crate-local tests | `feature_registry_router.rs`: route default and capability-gated workloads into harness model selection and verify primary failure falls back to the configured alternate. |
| Definition-host boundary | `tinyagents-definition` has one inline unit test | `feature_definition_runtime.rs`: validate/serialize definitions and use them in a hosted delegated harness run, proving declared children succeed and undeclared children fail. |

### P0: next execution batch

| Feature | Test to add |
| --- | --- |
| Workflow cancellation, lease recovery, concurrency cap | Extend `e2e_orchestration_workflow.rs` with a blocking registered child, cancellation/resume, competing drivers, and a multi-agent capped phase. |
| Durable reviewed delegation | `e2e_graph_delegation.rs` with `FileCheckpointer`, approval interrupt/resume, revision, and denial. |
| Typed graph channels/barriers | `e2e_graph_channels.rs` for aggregate, barrier, ephemeral, conflict, events, and checkpoints. |
| Workspace dispatch claims | A mixed tool-side-effect batch proving permitted parallelism, writer serialization, and unsafe-path rejection. |
| Goal tools in a model loop | A scripted model invoking registered goal tools and an over-budget preflight block. |
| SQLite cache across harness restart | Reopen a disk cache with a new harness and prove a cache hit avoids model call and billing. |
| Provider adapters through `AgentHarness` | Fake Claude Agent SDK and Claude Code subprocesses driven through a real harness. |
| Full-loop context compression and tool timeouts | Assert preserved assistant/tool pairs, compression provenance, timeout-as-tool-result, and model recovery. |
| Session run ledger lifecycle | Exercise lease/CAS/takeover, ordered events, telemetry, team claims, reopen, and orphan interruption. |

### P2: standalone public contracts

| Feature | Test to add |
| --- | --- |
| Optional `tools` feature | Scripted loop using `register_time_tools`, run under integration crate `tools`. |
| Optional `multimodal` feature | Markers through attachment resolution, MIME/size policy, text fallback, and profile rehydration. |
| Public stream iterator | Consume `AgentHarness::invoke_stream` through terminal and early-drop paths. |
| Artifact, handoff, and workspace utilities | Temp-root artifact-offload, handoff, and workspace lifecycle API contract. |
| Namespaced stores and control primitives | Namespace/TTL/batch/search, `RunQueue`, and no-progress/repeat tracking as direct feature contracts. |
| Extended language grammar | Execute materialized command/send/join/retry/checkpoint constructs only where graph runtime supports them. |

## Feature flags

`sqlite` is integration-tested through graph/session persistence. `tools` and
`multimodal` are compiled and unit-tested in CI but need the P2 behavioral
integration tests above. `tracing` is compiled under all-features; adding
behavioral tracing tests is lower priority because instrumentation is opt-in.
The integration crate should forward `tinyagents-orchestration/tracing` when a
new orchestration tracing scenario is added.

## Test design rules

- Use public APIs from `tinyagents-integration-tests`; do not reach into
  private modules merely to raise coverage.
- Keep provider tests offline with deterministic fake subprocesses/models.
- Separate feature-contract tests for standalone APIs from E2E tests for flows
  actually wired into the runtime.
- Every test must make an observable assertion about durable state, transcript,
  event stream, output, or safety boundary.
