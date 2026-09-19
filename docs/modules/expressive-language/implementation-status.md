# Implementation Status

This tracks what the `crates/tinyagents-language/src/` pipeline implements today against the
grammar and AST sketched in [README.md](README.md). The runtime stays
declarative: the compiler captures topology and policy in an inspectable
[`Blueprint`], while runnable behaviour is supplied by a Rust-side `NodeFactory`.

## Package shape

Flat files under `crates/tinyagents-language/src/`:

| file | role |
| --- | --- |
| `span.rs` | byte+line/column source spans |
| `source.rs` | source files and the source map |
| `diagnostic.rs` | structured diagnostics and the caret renderer |
| `ast.rs` | source AST node types (`Program`, `GraphDecl`, `NodeDecl`, …) |
| `lexer.rs` | source text into spanned tokens |
| `parser.rs` | tokens into the AST |
| `compiler.rs` | AST into one `Blueprint` per graph, capability binding, graph build |
| `types.rs` | tokens + compiled `Blueprint`/`*Spec` types (re-exports `ast`) |

`ast.rs` is re-exported from `types.rs`, so existing
`crate::language::types::{Program, NodeDecl, …}` paths keep resolving.

## Implemented grammar

### Graph-level items

- `start <ident>` — entry node.
- `defaults { key value … }` — graph defaults.
- `input { name type … }` / `output { name type … }` — graph I/O shape
  (lowered to `Blueprint::input` / `Blueprint::output` as `IoFieldSpec`s).
- `checkpoint <ident>` / `interrupt <ident>` — graph-level checkpoint and
  interrupt policies (`Blueprint::checkpoint` / `Blueprint::interrupt`).
- `channel <name> <reducer> <arg>*` — state channel bound to a reducer policy.
  Arguments are string/number literals (e.g. a named aggregate reducer or a
  barrier arrival count) captured in `ChannelSpec::args`.
- `node <name> { … }` — node declarations (see below).
- `from -> to` — static edges.
- `join [a, b] -> c` — top-level barrier (`Blueprint::joins`, `JoinSpec`).

### Node-level items

Common: `kind`, `model`, `system`/`prompt`, `tools [..]`, `next`,
`routes { label -> target }`.

Extended (H2):

- `agent "name"` — sub-agent reference for a `subagent` node (`NodeSpec::agent`).
- `graph "name"` — subgraph reference for a `subgraph` node
  (`NodeSpec::subgraph`; binding prefers it over the legacy `model` field).
- `router "name"` — router-function reference for a `router` node, parallel
  to `agent`/`graph`/`script` (added in Phase 1c). `model` is still accepted
  as a deprecated fallback when `router` is absent — `router` nodes
  previously had no dedicated item and overloaded `model` for their
  route-function name. The dedicated `router` value is folded into
  `NodeSpec::model` at compile time, so the compiled `Blueprint` shape is
  unchanged; the AST-level `Resolver` and `CapabilityResolver::bind_blueprint`
  both validate whichever value was used.
- `script "name"` — host script capability for a `repl_agent` node
  (`NodeSpec::script`). Declaration only — never inline code.
- `input "mapping"` — input mapping for sub-agent / subgraph nodes.
- `command { goto <target> update { key value … } }` — typed command
  (`NodeSpec::command`, `CommandSpec`). A bare `goto` also lowers into the
  node's routing (precedence: `routes` > `next` > command `goto` > edge >
  terminal).
- `sends [ send <node> ["input"] … ]` — fanout (`NodeSpec::sends`, `SendSpec`).
- `sources [a, b]` — upstream nodes for a `join` node (`NodeSpec::join_sources`).
- `options ["approve", "reject"]` — choices for an `interrupt` node.
- `checkpoint <ident>`, `timeout <literal>`, `retry { … }`, `metadata { … }`
  — node-level policies.

### Node kinds

The registry-backed binding path (`DEFAULT_NODE_KINDS`) accepts `agent`,
`model`, `tool_executor`, `subgraph`, `graph`, `subagent`, `repl_agent`,
`router`, `interrupt`, `join`, and `human`.

## Validation

`compile` rejects: duplicate nodes, missing/undefined `start`, unknown
`next`/`route`/`edge`/`command goto`/`send`/`join` targets, duplicate route
labels, and mixing static routing with `routes`. Registry binding additionally
checks model/tool/subgraph/router/agent/script/reducer references and node
kinds. A single shared policy (`CapabilityResolver::classify_reference`) maps
each node kind to the reference it must resolve, so the compiler blueprint gate
and both `Resolver` paths cannot drift: `subagent` binds its `agent` reference
against the registered agents and `repl_agent` binds its `script` reference
against the registered scripts.

## Not yet implemented

- State-schema declarations (`state Name { … }`).
- Steering policy lowering for `subagent` nodes. The `steering { … }` block
  parses (the grammar reserves the shape), but the compiler **rejects** any node
  that declares one rather than discarding it silently: the runtime
  `harness::steering::SteeringPolicy` is a single flat command allowlist with no
  `parent`/`human` actor separation, no delivery policy, and no
  `add_instruction`/`request_status` commands, so no faithful lowering exists.
  Build the `SteeringPolicy` in the Rust `NodeFactory` instead. See
  `reference.md`, `subagent` section.
- Duration literals like `60s` (write timeouts as a number or quoted string).
- Formatter and round-trip golden tests (milestone L8).
- Agent-authored review gates (milestone L7). Blueprint provenance itself is
  implemented: `compile_with_provenance` (`compiler.rs:442`) exists alongside
  `compile`.

Note: an earlier draft of this list also said the `CapabilityResolver`
agent-name allowlist was unimplemented and sub-agent names were not
registry-validated. That is stale — `CapabilityResolver::agent_allowed`
(`capability_resolver.rs:101-102`) and the `subagent` binding path
(`capability_resolver.rs:282-284`) do validate `subagent` node agent
references against the registered agents, matching the "Validation" section
above.

`build_graph` (`crates/tinyagents-graph/src/language.rs`) currently lowers
only `blueprint.start`, node names, and each node's `Routing`
(`Next`/`Conditional`/`Terminal`) into the executable graph. Every other
populated blueprint field — channels, checkpoint/interrupt policy, joins,
sends, input/output shape, node metadata/timeout/retry — is parsed and
validated by the compiler but inert once `build_graph` runs: it neither
applies nor rejects them. (Phase 1c of `docs/runtime-comparison/plan.md`
plans to make `build_graph` fail closed — `Compile` error — on any populated
field it still ignores.)
