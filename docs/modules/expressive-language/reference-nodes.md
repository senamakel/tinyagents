# Expressive Language Reference: Node Kinds

> Part of the expressive-language reference. Continues from
> [`README.md`](README.md); see [`reference.md`](reference.md) for the
> full reference index, and [`reference-graph.md`](reference-graph.md)
> for binding, state, routes, policies, examples, and milestones.

## Node Kinds

Initial built-in node kinds:

### `agent`

Uses the harness agent loop or one model call depending on config.

Supported fields:

- `model`
- `system`
- `prompt`
- `tools`
- `capability` — see [Capability references](#capability-references)
- `routes`
- `retry`
- `timeout`

### `model`

Single model invocation. Does not automatically execute tools.

Supported fields:

- `model`
- `system`
- `prompt`
- `routes`
- `retry`
- `timeout`

### `tool_executor`

Executes tool calls already present in state.

Supported fields:

- `tools`
- `next`
- `retry`
- `timeout`

### `router`

Routes based on a named route function provided from Rust.

Supported fields:

- `router` — the registered route-function name (e.g. `router "classify"`),
  parallel to `subgraph`'s `graph "name"` and `subagent`'s `agent "name"`.
  `model` is still accepted as a deprecated fallback for the same value (the
  convention before `router` existed as its own item).
- `routes`
- `metadata`

### `subgraph`

Calls another compiled graph.

Supported fields:

- `graph`
- `next`
- `routes`

### `subagent`

Calls a registered harness agent as a graph node.

Supported fields:

- `agent`
- `input`
- `next`
- `routes`
- `retry`
- `timeout`

Example:

```tinyagents
node research {
  kind subagent
  agent "researcher"
  next synthesize
}
```

#### `steering` — reserved, rejected by the compiler

```tinyagents
steering {
  parent allow ["add_instruction", "request_status", "cancel"]
  human allow ["add_instruction", "pause", "resume", "cancel"]
  delivery "safe_boundary"
}
```

This block **parses** — the grammar reserves the shape above — but `compile`
**rejects** any node that carries it, with a `TinyAgentsError::Compile`
diagnostic. It is not enforced, and it is deliberately not accepted-and-ignored:
a silently discarded policy would let an operator deploy a blueprint believing a
child agent's steering is restricted when the runtime receives no restriction at
all.

There is no faithful lowering yet. `harness::steering::SteeringPolicy` is a
single flat allowlist of `SteeringCommandKind`s (`pause`, `resume`, `cancel`,
`inject_message`, `redirect`, `set_metadata`); it has no `parent`/`human` actor
separation, no delivery policy, and no `add_instruction` or `request_status`
command — so three of the four elements in the block above have no runtime
counterpart.

Until declarative steering is implemented end to end, restrict a child agent by
building the `SteeringPolicy` in the Rust `NodeFactory` that materialises the
node, where the policy is actually attached to the run's `SteeringHandle`.

### `repl_agent`

Runs a host-provided script node, bound by name to a registered `Script`
component. The node implementation is supplied by the host; this crate ships
no interpreter.

Supported fields:

- `model`
- `script`
- `tools`
- `routes`
- `retry`
- `timeout`

### `interrupt`

Emits a resumable human-in-the-loop interrupt.

Supported fields:

- `prompt`
- `options`
- `routes`
- `metadata`

### `join`

Waits for named upstream nodes or barrier channels before continuing.

Supported fields:

- `sources`
- `next`
- `timeout`

### Capability references

Any node kind that accepts a `capability` field can name a registered
[`Capability` bundle](../registry/implementation-status.md#capability-bundle-gap-g3)
(gap G3) — instructions, toolset, middleware, model defaults, and exposure
composed as one unit:

```rag
agent "researcher" {
  model "fast"
  capability "web_research"
}
```

`capability` is a single string, like `model`; specifying the same node's
`capability` twice is a compile error (`dup(..., "capability")`). The name is
resolved the same way `tool`/`model`/`subgraph` references are: unconditional
membership in the host-registered capability allowlist. A `.rag` source that
names a capability the host never registered fails to bind, exactly like an
unregistered tool or model would — see the resolver contract tests in
`crates/tinyagents-integration-tests/tests/e2e_graph_resolver_contracts.rs`.

