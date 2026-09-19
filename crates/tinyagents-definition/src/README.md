# `tinyagents-definition` — host-owned agent definition vocabulary

A minimal, host-owned vocabulary for agent definitions: identity, description,
preferred model, delegates, and tools. The runtime uses this to query what
agents are available and what capabilities they declare, but authorization,
prompting, and execution remain with the host.

## Public surface

- **`AgentDefinition`** — a durable agent descriptor: id, name, description,
  role, preferred model, subagent ids, and tool names.
- **`DefinitionRegistry`** — the async trait for definition lookup: `resolve()`
  by id, `list()` all definitions, `delegates_for()` to list authorized
  subagents for an agent.
- **`InMemoryDefinitionRegistry`** — simple in-memory registry backed by a
  vector; first-write-wins for duplicate ids, stable insertion order.
- **`AgentDefinitionDiagnostic`** — validation finding: field, error code, and
  explanation.

## Design

- **Host-owned vocabulary:** The definition crate deliberately stays minimal.
  It defines only what the runtime may ask, not how agents are configured,
  authorized, or invoked.
- **Deterministic validation:** `AgentDefinition::diagnostics()` returns a
  stable list of problems: missing required fields, duplicates in tool/subagent
  lists, empty entries.
- **Async registry trait:** `DefinitionRegistry` is async so hosts can fetch
  definitions from a database or service. `Ok(None)` is the normal absence
  outcome; errors are for catalogue failures.
- **No built-in enforcement:** Hosts decide whether to enforce role, model pin,
  or subagent delegation.
