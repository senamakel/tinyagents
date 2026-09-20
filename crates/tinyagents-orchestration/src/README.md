# `tinyagents-orchestration` — subagent orchestration

This crate is the single home for TinyAgents subagent behavior. It builds on
the lower-level harness and runtime crates without making those crates depend
on orchestration policy.

## Public surface

- `subagent::SubAgent` runs an `AgentHarness` as a child with inherited
  lineage, cancellation, events, host authority, and recursion limits.
- `subagent::SubAgentTool` starts a child asynchronously through typed
  parent-context tool dispatch and immediately returns a stable job id.
- `subagent::SubAgentJobRegistry` records queued, running, completed, failed,
  and cancelled jobs. `SubAgentJobsTool` queries them and
  `SubAgentMessageTool` sends messages to live children.
- `subagent::SubAgentSession` reuses one child and its transcript across turns.
- `subagent::SubagentDriver` coordinates durable resume, preparation,
  execution, pause, and terminal persistence through host-supplied traits.

The direct invocation implementation and its tool-focused tests live in
`subagent/invocation/`. Durable lifecycle files live beside it in `subagent/`.
End-to-end and live subagent tests live in the crate-level `tests/` directory,
and `tests/live_orchestrator_subagents.rs` exercises network-backed job-based
delegation.

## Boundaries

The crate does not own teams or workflow DAGs. Graph-specific node lowering
remains in `tinyagents-graph`, while provider calls, tool dispatch mechanics,
events, and run contexts remain in `tinyagents-harness`. This crate composes
those primitives into child-agent behavior.

Hosts remain responsible for agent definitions, model selection, credentials,
workspace policy, durable storage implementations, and authorization. Hosted
child invocations always re-enter through the exact capability bundle carried
by their parent context.
