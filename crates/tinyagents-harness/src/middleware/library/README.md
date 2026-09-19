# harness::middleware::library

The built-in middleware catalog that ships with the harness. Every type here
is re-exported through `crate::middleware`, so callers write
`tinyagents_harness::middleware::RetryMiddleware`, not
`middleware::library::RetryMiddleware`.

See the parent [`middleware/README.md`](../README.md) for the two extension
shapes (lifecycle `Middleware` vs. wrap `ModelMiddleware`/`ToolMiddleware`) and
onion-ordering rules these implementations build on. A handful of other
built-in middleware (`LoggingMiddleware`, `MessageTrimMiddleware`,
`ContextCompressionMiddleware`, `MicrocompactMiddleware`,
`PromptCacheGuardMiddleware`, `UsageAccountingMiddleware`) live directly in
`middleware/types.rs` instead of here — see that module's README for why.

## Public surface

Grouped by extension shape:

### Resilience (`ModelMiddleware`, wrap around the real model call)

- `RetryMiddleware` — retries the wrapped model call on a
  [retryable][crate::retry::is_retryable] error while the configured
  `RetryPolicy` still permits another attempt. Computes backoff but does not
  sleep by default (see `RetryMiddleware::backoff_for_attempt`), keeping tests
  deterministic.
- `TimeoutMiddleware` — races the wrapped call against `tokio::time::timeout`;
  returns `TinyAgentsError::Timeout` on elapse.
- `ModelFallbackMiddleware` — on a retryable failure, retries against an
  ordered chain of fallback model names, emitting `AgentEvent::FallbackSelected`
  before each attempt.
- `RateLimitMiddleware` — gates calls through a shared `RateLimiter`
  (token bucket); either fails fast (`RateLimitBehavior::Error`) or polls until
  capacity frees up (`RateLimitBehavior::Wait`), with an injectable clock for
  deterministic tests.

### Tool policy / selection (`Middleware` lifecycle hooks)

- `ToolAllowlistMiddleware` — rejects `before_tool` calls whose name is not on
  a fixed allowlist.
- `ToolPolicyMiddleware` — enforces the structured `ToolPolicy` metadata each
  tool advertises (classification, side-effect denylist, background-safety,
  sandbox requirement, approval requirement, result-byte cap) at both
  model-visible exposure time (`before_model`) and execution time
  (`before_tool`). `ToolPolicyMiddleware::strict` gives a fail-closed baseline.
- `DynamicToolSelectionMiddleware` — filters model-visible tools via a plain
  `ToolPredicate(&ToolSchema) -> bool`.
- `ContextualToolSelectionMiddleware` — filters model-visible tools via a
  `ContextualToolPredicate` that also sees run context (depth, tags, requested
  model). `from_lists`/`inheriting` build one from allow/deny name lists, with
  `inheriting` composing a child policy on top of a parent's so a delegated
  sub-agent can only narrow, never widen, tool exposure.
- `HumanApprovalMiddleware` — raises `TinyAgentsError::Interrupted` from
  `before_tool` for flagged tools unless an `ApprovalFn` admits the call; the
  harness-native half of a human-in-the-loop gate.

### Budget (`Middleware` lifecycle hooks)

- `BudgetLimits` / `BudgetTracker` / `BudgetSpend` — declarative token/cost
  limits, a shared accumulating spend tracker (clone to roll up spend across a
  parent and its sub-agents), and a point-in-time snapshot.
- `BudgetMiddleware` — preflights each call in `before_model` (reserving
  estimated input tokens under a single lock so concurrent runs on a shared
  tracker cannot collectively overshoot), reconciles actual usage/cost in
  `after_model`, and releases the reservation in `on_error` so a failed call
  never leaks it. Emits `AgentEvent::BudgetReserved`, `BudgetReconciled`,
  `UsageRecorded`, `CostRecorded`, `BudgetWarning`, and `BudgetExceeded`.

### Observation (`Middleware` lifecycle hooks)

- `StructuredOutputValidatorMiddleware` — validates an `after_model` response
  against an expected `ResponseFormat` (JSON object or provider-schema
  extraction), failing with `TinyAgentsError::StructuredOutput`.
- `DynamicPromptMiddleware<State, Ctx>` — derives an optional system prompt
  from application state and `RunConfig` on each call via a `PromptFn`,
  inserting it at the front of `request.messages`.
- `RedactionMiddleware` — replaces configured literal-substring patterns with
  a mask string across model response text/JSON, tool-call arguments, raw
  provider payloads, tool results, and tool errors. Single-pass, idempotent
  (never matches inside its own mask output).
- `TracingMiddleware` — implements every lifecycle hook, recording a bounded
  ring buffer of `PhaseTrace` begin/end records plus per-phase `TraceCounts`,
  independent of the event stream.

## Files

| File | Role |
| --- | --- |
| `types.rs` | Every public type: middleware structs, their config/outcome types (`RateLimitBehavior`, `BudgetLimits`, `ToolSelectionContext`, `PhaseTrace`, ...), and type aliases (`NowFn`, `ToolPredicate`, `ContextualToolPredicate`, `ApprovalFn`, `PromptFn`). |
| `mod.rs` | Re-exports `types`, declares the `resilience`/`budget`/`tool_policy`/`context`/`observe` submodules, and documents cross-cutting testability guarantees. |
| `resilience.rs` | Constructors and `ModelMiddleware` impls for `RetryMiddleware`, `TimeoutMiddleware`, `ModelFallbackMiddleware`, `RateLimitMiddleware`. |
| `budget.rs` | Constructors and `Middleware` impl for `BudgetMiddleware`, plus `BudgetTracker`/`BudgetLimits` helper methods and the shared input-token estimator. |
| `tool_policy.rs` | Constructors and `Middleware` impls for `ToolAllowlistMiddleware`, `ToolPolicyMiddleware`, `DynamicToolSelectionMiddleware`, `ContextualToolSelectionMiddleware`, `HumanApprovalMiddleware`. |
| `context.rs` | Re-exported from `middleware/types.rs`'s sibling `mod.rs`, **not** this directory — kept here only as a historical note; see `middleware/README.md`. |
| `observe.rs` | Constructors and `Middleware` impls for `StructuredOutputValidatorMiddleware`, `DynamicPromptMiddleware`, `RedactionMiddleware`, `TracingMiddleware`. Also re-implements `LoggingMiddleware`/`UsageAccountingMiddleware` impls declared in `middleware/types.rs`. |
| `test.rs` | Unit tests for every middleware in this directory (construction, hook behavior, event emission, edge cases like poisoned mutexes and concurrent budget reservations). |

## Operational constraints

- **No middleware here sleeps uncontrollably on the wall clock in tests.**
  `RetryMiddleware` only sleeps when its `RetryPolicy` opts in via
  `with_backoff_sleep` (off by default); `TimeoutMiddleware` is exercised under
  `tokio::time` paused-time tests; `RateLimitMiddleware` takes an injectable
  clock and configurable poll interval. Preserve this when adding new
  middleware: prefer computing a delay over unconditionally awaiting one.
- **Budget reservations are keyed by `RunContext::instance_id`, not `run_id`.**
  Concurrent runs sharing a `BudgetTracker` may share a caller-supplied
  `run_id`; only the process-unique instance id keeps each run releasing
  exactly what it reserved.
- **Tool-policy and tool-selection middleware only change what the model
  *sees* in `before_model`.** They do not by themselves guard execution — pair
  them with `ToolAllowlistMiddleware`/`ToolPolicyMiddleware`'s `before_tool`
  enforcement (or your own) if a model can call a tool it was never shown.
- **Wrap middleware in `resilience.rs` are responsible for their own retry
  budget.** `MiddlewareStack` does not cap how many times a `ModelMiddleware`
  calls `next.run(..)`.
