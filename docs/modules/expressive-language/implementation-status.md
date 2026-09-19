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
labels, mixing static routing with `routes`, and (Phase 1c) a duplicate
single-value node item (`model`/`kind`/`prompt`/`agent`/`graph`/`script`/
`router`/`input`/`command`/`checkpoint`/`timeout`/`steering`) or graph item
(`start`/`checkpoint`/`interrupt`) — previously the second occurrence silently
overwrote the first. Registry binding additionally checks
model/tool/subgraph/router/agent/script/reducer references and node kinds. A
single shared policy (`CapabilityResolver::classify_reference`) maps each node
kind to the reference it must resolve, so the compiler blueprint gate and both
`Resolver` paths cannot drift: `subagent` binds its `agent` reference against
the registered agents and `repl_agent` binds its `script` reference against
the registered scripts.

List separators (`[a, b, c]` in `sources`/`tools`/`options`/`sends`/`join`)
share one rule since Phase 1c: comma-separated with an optional trailing
comma. `sends` previously accepted a comma between entries as optional even
mid-list (`[send a send b]` parsed the same as `[send a, send b]`); it now
requires the comma, matching `parse_ident_list`/`parse_string_list`.

`Literal` has a `Bool` variant since Phase 1c (`Literal::Bool(bool)`), so
`defaults { streaming true }` lowers to a real boolean instead of
`Literal::Ident("true")`. `true`/`false` are recognised in `parse_literal`
before falling back to a bare `Ident`.

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

## `build_graph`: lowered vs rejected fields (Phase 1c)

`build_graph` (`crates/tinyagents-graph/src/language.rs`) still lowers only
`blueprint.start`, node names, each node's Rust-side handler (via
`NodeFactory`), and each node's `Routing` (`Next` → a static edge,
`Conditional` → `mark_command_routing` plus `with_command_destinations` for
the declared route table, `Terminal` → `set_finish`). As of Phase 1c it now
**fails loudly** instead of silently ignoring every other populated field: it
inspects the blueprint before touching the factory or the builder and returns
`TinyAgentsError::Compile` naming every populated field it does not honour.

**Rejected until full lowering lands (Phase 5):**

- graph-level: `input`, `output`, `checkpoint`, `interrupt`, `joins`
- per node: `sends`, `join_sources`, `command.update`, `options`, `timeout`,
  `retry`, `metadata`

**Deliberately still accepted (not rejected), with a documented gap:**

- `channels` (state-channel reducers) and `defaults` (the `defaults { … }`
  block, e.g. `recursion_limit`/`backoff`/`checkpoint`). `build_graph` always
  builds the executable graph with `GraphBuilder::overwrite()` regardless of
  what a `channel … <reducer>` declares, so a non-`overwrite` reducer is still
  silently not applied to the runtime state merge. These two are excluded
  from the reject list because they are already read by
  `crate::export::blueprint_to_topology` for introspection (so they are not
  *entirely* inert) and, more importantly, because rejecting them would break
  existing fixtures (`crates/tinyagents-integration-tests/tests/language_pipeline.rs`,
  `e2e_rag_pipeline.rs`, and their `.rag` source) that this change's file
  boundary did not permit editing. A future pass that either lowers channel
  reducers into real per-channel state merge or extends the reject list to
  `channels`/`defaults` will need to touch those fixtures too.

**Conditional route tables are not enforced against a handler's `Command::goto`
at compile time.** `GraphBuilder::with_command_destinations` — which
`build_graph` now calls for every `Routing::Conditional` node — is advisory
only (used by `crate::export` to draw/validate the declared destinations in a
topology view); the runtime always resolves the real successor from the
`Command` a node handler emits, so a handler that `goto`s a label the source
never declared is not rejected at graph-build time. Making that a real
compile-time check would require `GraphBuilder`/`CompiledGraph` to validate
emitted commands against the declared table at run time (or a stricter
builder API), which is out of scope for Phase 1c.

See `crates/tinyagents-graph/src/language.rs` for the exact field list
(`ignored_populated_fields`) and its tests
(`build_graph_rejects_a_populated_ignored_field`,
`build_graph_accepts_a_blueprint_with_no_ignored_fields`).
