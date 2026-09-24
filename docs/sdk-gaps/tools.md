# SDK Gaps: Tool Policy And Execution

> Part of [SDK Gaps](README.md). Covers tool policy metadata, unknown-tool
> recovery, dynamic tool exposure, deferred tool calls, and tool execution
> context.

## Backlog

### 1. Rich Tool Policy Metadata

Status: partially present.

TinyAgents has `ToolSchema { name, description, parameters, format }` and
`ToolExecutionContext` (run/call/thread identity, limits, events, workspace,
store, state view — see §16). Without policy metadata the SDK could not make
fail-closed decisions about whether a tool should be exposed, approved,
retried, timed out, or allowed to touch the filesystem/network.

Shipped as vendored `tinytools::ToolPolicy` (side effects, runtime and
access requirements) plus `ToolPolicyMiddleware`, which enforces it before
model-visible exposure and before execution; `access.approval_required` also
drives the A2 deferral in §14.

Acceptance criteria:

- Callers can build a dynamic per-run tool set from policy metadata.
- Unknown or under-classified tools fail closed by default.
- Tool policy can be serialized for registry introspection and audit logs.
- Existing plain `ToolSchema` remains supported as the model-visible projection.

### 2. Recoverable Unknown Tool Calls

Status: shipped.

TinyAgents now has `UnknownToolPolicy::{Fail, ReturnToolError, Rewrite}` on
`RunPolicy` (`crates/tinyagents-harness/src/runtime/types.rs`), applied in
`crates/tinyagents-harness/src/agent_loop/tools.rs`. The default is
`ReturnToolError`: an unregistered tool call is injected back as a tool-error
result naming the requested tool and the valid tools, so the model can
self-correct instead of the run aborting. `Fail` restores the old abort
behavior; `Rewrite { tool_name }` retargets the call to a fixed compatibility
tool. OpenHuman's `UNKNOWN_TOOL_SENTINEL` workaround can be retired.

Still open: a `RepairWithMiddleware` variant letting a tool middleware
transform the call. Events preserve the requested name, arguments, and call id.
Acceptance: OpenHuman can delete `UNKNOWN_TOOL_SENTINEL`; events distinguish
"tool not found" from "tool executed and failed"; the policy can vary by run,
sub-agent, or tool allowlist.

### 9. Dynamic Tool Exposure And Allowlist Policy

Status: present (B3 composable toolsets, `docs/runtime-comparison/pydantic-ai.md`
§3.4/§4; see `docs/modules/harness/toolsets.md` for the full design).

`tinyagents_harness::tool::toolset::ToolSet<State, Ctx>` (`tools`/`call`/
`instructions`/`for_run`) is the composition unit; `ToolRegistry` implements
it directly, and `Combined`/`Filtered`/`Prefixed`/`Renamed`/`Prepared`/
`ApprovalRequired`/`External` are independently testable value-level
adaptors, wired in via `AgentHarness::with_toolset`. Every adaptor that
changes or withholds a tool records a `ToolExposureExplanation`
(`FilteredOut`, `Renamed`, `Prefixed`, `Prepared`, `ApprovalRequired`,
`Deferred`, `Hidden`), additive on `AgentEvent::ToolsFiltered` — a concrete,
inspectable answer to "why was this tool hidden" instead of depending on
middleware ordering. `ToolAllowlistMiddleware`/`DynamicToolSelectionMiddleware`
are kept as public types but are now thin wrappers sharing predicate logic
with `FilteredToolSet`/`PreparedToolSet` so the two cannot drift.

OpenHuman-specific per-tier/per-sub-agent/per-task allowlist *policy*
composition, and MCP-backed tool sources, still live in OpenHuman: this gap
closes the composition primitive and the host seam (`ExternalToolSet` +
`TinyAgentsError::CallDeferred`) that policy is built on, not OpenHuman's own
policy tables.

Acceptance criteria:

- [x] Sub-agents inherit only the tools they are allowed to call (`FilteredToolSet`/`PrefixedToolSet` chains per sub-agent).
- [x] Tool exposure decisions are visible in run events (`ToolExposureExplanation` on `AgentEvent::ToolsFiltered`).
- [ ] OpenHuman can remove adapter-local allowlist enforcement from most call paths — OpenHuman-side migration, not tracked here.

### 14. Deferred Tool Calls (A2)

Status: shipped (harness); durability stays host-owned.

Landed as `docs/runtime-comparison/plan.md` Phase 2 item A2. A tool call now
leaves the loop as a typed, resumable output instead of `Err(Interrupted)`:
`ToolPolicy.access.approval_required`, `Err(TinyAgentsError::ApprovalRequired
{ metadata })` / `CallDeferred { metadata }` (from a tool or a `before_tool`
middleware), or a `ToolRegistry::register_external(schema)` tool all produce
`AgentRun::deferred = Some(DeferredToolRequests { calls, approvals,
metadata })` after the batch's other calls run. Resume with
`AgentHarness::resume_deferred` / `AgentTurnRequest::with_deferred_results`
and `DeferredToolResults { approvals: ToolApprovalDecision::{Approve,
ApproveWithArgs, Deny}, calls: DeferredCallResult::{Result, Retry, Failed} }`;
`remaining()` reports unresolved ids. A `DeferredToolHandler` on the harness
resolves inline; `HumanApprovalMiddleware::with_approval_outcome` returns
`ApprovalOutcome::{Allow, Deny, Defer}`. Events: `ToolDeferred`,
`ToolApproved`, `ToolDenied`. OpenHuman's `security/approval::ApprovalGate`
becomes a `DeferredToolHandler`. Persistence of `run.messages` +
`run.deferred` is the host's (the session ledger depends on the harness, so
the loop cannot write it); see
[`docs/modules/harness/tool.md`](modules/harness/tool.md#deferred-tool-calls-approval-and-external-execution-a2).

### 16. Tool Execution Context Parity And Rich Returns (B1/B2)

Status: shipped (harness); OpenHuman adapters still to migrate.

`ToolExecutionContext` gained `call_id`, `store` (`NamespacedStore`),
`state::<S>()`, and `custom()` → `AgentEvent::Custom`, reachable from a
`tinytools::Tool` via the new vendored `ToolRunContext::host_extension()`
downcast; `ToolDispatch::execute` takes `call_id`. `ToolResult::follow_up`
becomes a user message after the batch's tool rows; `ToolResult::metadata`
goes to `ToolCompleted { metadata }` / `AgentRun::tool_metadata`, never the
transcript, so OpenHuman's `artifact_offload` JSON-stuffing can move there.
See [`tool-context.md`](modules/harness/tool-context.md). Still open: an
approval flag on the context; a native file block in the message model.

