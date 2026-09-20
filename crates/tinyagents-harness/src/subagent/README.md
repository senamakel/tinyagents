# harness::subagent

First-class sub-agents with recursion-depth tracking — the harness's flagship
"agents calling agents" surface. It lets one agent run another agent as a
child of itself, the in-harness analogue of `crate::graph::subgraph`'s
recursion on the graph side.

## Design

Three cooperating types:

- **`SubAgent<State, Ctx>`** wraps an `AgentHarness<State, Ctx>` and runs it
  as a child run one recursion level deeper than its caller. `invoke` /
  `invoke_with_events` are the standalone entry points; `invoke_in_parent`
  and `invoke_hosted_in_parent` thread a live `RunContext` so the child
  inherits the parent's cancellation, event sink, stores, and (for the
  hosted variant) host delegation authority.
- **`SubAgentTool<State, Ctx>`** adapts a `SubAgent` into a typed
  `crate::tool::ToolDispatch`, so a parent agent can call another agent as an
  ordinary tool call. Registered via
  `crate::tool::ToolRegistry::register_dispatch` (not the plain
  `tinytools::Tool` path — `SubAgentToolDeclaration::execute` refuses direct
  calls and points at the typed dispatcher).
- **`SubAgentSession<State, Ctx>`** keeps a single `SubAgent` alive across
  multiple turns, reusing the same harness while accumulating the transcript
  — the post-completion, human-in-the-loop *reuse* primitive, distinct from
  `crate::steering`'s mid-run *steering*.

### Depth tracking

Every run carries a `depth` in its `RunConfig` (top-level = `0`). Invoking a
sub-agent at `parent_depth` creates the child at `parent_depth + 1`, capped
by the child harness's `RunLimits::max_depth` (default
`RunLimits::DEFAULT_MAX_DEPTH` = `8`). Exceeding the cap fails fast with
`TinyAgentsError::SubAgentDepth` *before* any model call — `child_config`
computes and checks this on every invocation path.

### Observability

Every invocation brackets the child run with
`AgentEvent::SubAgentStarted`/`SubAgentCompleted`. Invoking through
`invoke_with_events`/`invoke_in_parent`/a `SubAgentSession` routes the
child's own events onto the shared parent sink, so a parent observer sees
the whole nested run tree; `SubAgentSession` additionally emits
`AgentEvent::SubAgentReused` on every send after the first.

## Public surface

- `SubAgent::new` / `with_system_prompt` / `name` / `description` /
  `harness` — construction and accessors.
- `SubAgent::invoke` / `invoke_with_events` / `invoke_in_parent` /
  `invoke_hosted_in_parent` — the four invocation entry points, differing in
  how much of the live parent context they thread through (see the doc
  comment on each for exactly what's inherited).
- `SubAgentTool::new` / `with_tool_name` / `with_parameters` /
  `invoke_in_parent_context` — the typed-parent dispatch adapter; the last
  method is also what its `ToolDispatch::execute` impl calls.
- `SubAgentSession::new` / `from_subagent` / `with_events` /
  `with_parent_depth` / `subagent` / `transcript` / `turns` / `reset` /
  `send` — reusable multi-turn session over one `SubAgent`.
- `ChildDataPolicy<Ctx>` — explicit parent→child application-data transform
  (`new`, `child_data`); required by `SubAgentTool::new` so data inheritance
  is never a silent `Default`.
- `SUBAGENT_INPUT_FIELD` — the argument key (`"input"`) a `SubAgentTool`
  reads the child's user prompt from.

## Files

| File | Role |
| --- | --- |
| `mod.rs` | All impls: constructors, the four invoke paths, `SubAgentTool` dispatch, `SubAgentSession::send`, the private `SubAgentToolDeclaration`. |
| `types.rs` | Public struct/const definitions, re-exported via `pub use types::*`. |
| `test.rs` | `ChildDataPolicy`, `SubAgentTool` dispatch (data inheritance, depth-cap-as-recoverable-result, cancellation), `invoke_in_parent` propagation, depth-cap-before-model-work, `SubAgentSession` reuse. |

## Key invariants

- **Depth is checked before any model call.** `child_config` computes and
  validates the child depth up front, so a run that would exceed
  `max_depth` never reaches the network.
- **`invoke_in_parent` rejects a hosted parent.** A `RunContext` carrying
  `host_authority` must go through `invoke_hosted_in_parent`, which resolves
  the parent's delegate allowlist; falling through the generic path would
  silently discard that authority (`TinyAgentsError::Validation`).
- **A hosted child cannot select an alternate host authority.** It always
  re-enters through the exact capability bundle installed on the parent
  (`crate::runtime::host_invocation_binding`); the child harness supplies
  durable mechanics only.
- **A depth-cap or run-limit failure inside `SubAgentTool` is a recoverable
  tool result, not a hard error** — `invoke_in_parent_context` turns
  `SubAgentDepth`/`LimitExceeded`/`Timeout` into a `tinytools::ToolResult`
  error the parent orchestrator can read as "delegated agent hit a limit,"
  not a thrown error that aborts the parent run.
- **`SubAgentSession` reuses the same harness and `Arc<SubAgent>` across
  every send** — nothing is reconstructed; only the transcript and turn
  counter mutate. `reset()` clears the transcript without touching the
  underlying harness.
- Run/thread ids minted here always suffix a process-unique sequence
  (`crate::ids::next_seq`) so two invocations of the same sub-agent — or two
  `SubAgentSession`s reusing it — never collide on run id.

## Relation to neighbouring modules

Built on `crate::runtime::AgentHarness` (the child agent loop),
`crate::context::RunContext` (the live run being extended into a child), and
`crate::tool::ToolDispatch` (the typed dispatch seam `SubAgentTool`
implements). See `crate::steering` for the complementary mid-run interruption
mechanism, and `crate::graph::subgraph` for the equivalent recursion primitive
on the graph side of the crate.
