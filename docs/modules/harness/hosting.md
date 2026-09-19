# Host Authorization and Tool Timeouts

Split out of `README.md` (see that module's overview for architecture
context); covers the two host-facing pieces of the `Package Shape` runtime
surface: how a host capability bundle is bound to one invocation, and how
per-tool timeouts are resolved.

## Host-authorized invocations

`AgentHarness` is reusable process infrastructure: it owns durable model and
tool registries, middleware, policy, and caches. A host capability bundle is
instead supplied for each root through `runtime::AgentInvocation`:

```rust,no_run
use tinyagents_harness::{
    context::{RunConfig, RunContext},
    runtime::{AgentHarness, AgentInvocation, AgentTurnRequest},
};

# async fn example<State: Send + Sync + 'static>(
#     harness: &AgentHarness<State>,
#     host: tinyagents_harness::host::HostCapabilities<State>,
#     state: &State,
# ) -> tinyagents_harness::Result<()> {
let invocation = AgentInvocation::new(
    host,
    AgentTurnRequest::new("assistant", vec![]),
    RunContext::new(RunConfig::new("run-42"), ()),
);
let _run = harness.invoke_agent(invocation, state).await?;
# Ok(())
# }
```

This prevents concurrent roots from replacing one another's progress,
security, approval, or other host authority. The harness never stores a live
capability bundle (not even in a run-id map): it lives only in the
non-serializable `RunContext`, is never checkpointed, and is dropped with that
invocation. Recursive children inherit the exact parent bundle through their
live context and cannot select a bundle from their own harness. The lower-level
explicit-model `invoke*` APIs remain separate for SDK callers that intentionally
assemble a run without host capabilities.

Hosted invocations require `State: 'static` because their live capability
authority must be retained in the recursive context. The explicit-model
`invoke*`, streaming, and direct `SubAgent` paths do not install or inspect
that authority and continue to support borrowed state.

For a child of a hosted parent, call
`SubAgent::invoke_hosted_in_parent`; it rechecks the parent's delegate
allowlist and inherits the exact bundle. The borrowed-state-compatible
`SubAgent::invoke_in_parent` is explicit-only and rejects a hosted parent
context before it can start a child.

