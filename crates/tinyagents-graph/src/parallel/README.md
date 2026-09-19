# graph::parallel

Ordered, bounded-concurrency parallel map/reduce with a configurable failure
policy, plus (in `claims/`) the shared-workspace claim arbitration that
decides when fanned-out workers are safe to run concurrently at all.

Graph `Send` fanout is the low-level primitive for parallel supersteps, but
callers frequently want a *reusable* "run these N items concurrently and
reduce the results" helper independent of the graph executor: deterministic
input-order results, a concurrency cap, per-item success/failure isolation,
and a policy for what to do when some items fail (fail-fast, collect-all,
quorum, best-effort). That is what [`map_reduce`] provides.

## Public surface

- `map_reduce(items, options, f) -> Result<ParallelOutcome<T>>` — runs `f`
  over `items` with bounded concurrency (`ParallelOptions::max_concurrency`,
  `0` = unbounded), collecting per-item outcomes in **input order** regardless
  of completion order, and applying `options.failure_policy`.
- `ParallelOptions` — builder-style options: `with_max_concurrency`,
  `with_failure_policy`, `with_item_timeout`, `with_total_timeout`,
  `with_cancellation`.
- `FailurePolicy` — `FailFast` (return the first input-order error, cancel the
  rest), `CollectAll` (default; always `Ok`, per-item outcomes), `Quorum(n)`
  (error unless at least `n` items succeed), `BestEffort` (always `Ok`, keep
  only successes).
- `ItemOutcome<T>` / `ParallelOutcome<T>` — per-item and aggregate results,
  with `success_count`/`failure_count`/`successes`/`into_successes` helpers.
- `claims` (re-exported): `WorkspaceClaim`, `DispatchPlan`, `DispatchMode`,
  `ClaimConflict`, `ClaimPathError`, `parse_relative_claim_paths`,
  `paths_overlap`, `writes_shared_workspace`,
  `plan_shared_workspace_dispatch` — see `claims/README.md`.

## Files

| File | Role |
| --- | --- |
| `types.rs` | `FailurePolicy`, `ParallelOptions`, `ItemOutcome<T>`, `ParallelOutcome<T>`. |
| `mod.rs` | `map_reduce`: the bounded-concurrency, input-order-preserving driver. |
| `test.rs` | Unit tests (ordering under out-of-order completion, each failure policy, timeouts, cancellation). |
| `claims/` | Shared-workspace claim arbitration (own `README.md`). |

## Operational constraints

- `map_reduce` re-orders by input index after collection; it does **not**
  stream partial results — callers needing incremental output should not use
  this helper.
- `FailFast` cancels remaining in-flight work by dropping the underlying
  stream once every item with a smaller input index has resolved, not merely
  on the first error observed — see the `fail_fast_error` bookkeeping in
  `mod.rs` for why the first-completed error is not necessarily the one
  returned.
- `item_timeout` and `total_timeout` are independent: an item exceeding
  `item_timeout` becomes a per-item failure handled by `FailurePolicy`; the
  whole call exceeding `total_timeout` aborts unconditionally with
  `TinyAgentsError::Timeout`, even under `CollectAll`/`BestEffort`.

## graph::parallel::claims

Shared-workspace claim arbitration for parallel agent fan-out — a decision
module with **no I/O and no scheduling**: given a set of `WorkspaceClaim`s it
returns a `DispatchPlan` saying which workers may run concurrently, which must
be serialized, and which cannot be scheduled at all.

### The rule

A worker is safe to run in parallel when it either has its own root
(`isolated`) or never writes (`!writes`). A worker that writes a *shared* root
must declare the paths it owns; it is then serialized, and its claim is
checked against every claim already granted. Claims are granted in **input
order, first-writer-wins**, which makes the plan a pure function of the input
regardless of which worker happens to finish first.

### Public surface

- `WorkspaceClaim` — one worker's declared relationship to the shared
  workspace (`isolated`, `read_only`, `writing` constructors).
- `plan_shared_workspace_dispatch(claims) -> DispatchPlan` — the planner.
- `DispatchPlan` — `modes` (index-aligned `Option<DispatchMode>`, `None` =
  rejected) plus `conflicts` (index-keyed rejections); `has_serial_work`,
  `parallel_indices`, `serial_indices`.
- `DispatchMode` — `Parallel` / `Serial`.
- `ClaimConflict` — `UnboundedWrite` (writer with no declared paths) /
  `Overlap` (claim collides with an earlier one); `worker_id()` accessor.
- `ClaimPathError` — `Absolute` / `Escaping`, returned by
  `parse_relative_claim_paths`.
- `parse_relative_claim_paths(spec) -> Result<Vec<PathBuf>, ClaimPathError>` —
  parses a comma/newline-separated, optionally bulleted claim list into safe,
  sorted, deduplicated relative paths.
- `paths_overlap(left, right) -> bool` — component-wise (not textual) overlap
  check.
- `writes_shared_workspace(effects: &ToolSideEffects) -> bool` — derives a
  "does this tool write?" claim input from tool side-effect metadata.

### Files

| File | Role |
| --- | --- |
| `types.rs` | `WorkspaceClaim`, `ClaimPathError`, `ClaimConflict`, `DispatchMode`, `DispatchPlan`. |
| `mod.rs` | `parse_relative_claim_paths`, `paths_overlap`, `writes_shared_workspace`, `plan_shared_workspace_dispatch`. |
| `test.rs` | Unit tests (path parsing/safety, overlap semantics, planner ordering and conflict reporting). |

### Operational constraints

- What a host *does* with a `ClaimConflict` (hard rejection vs. warning, and
  what a user reads) is a product decision this crate deliberately does not
  make — it only reports the conflict as data.
- `parse_relative_claim_paths` rejects absolute paths and anything containing
  `..` or a platform path prefix, because either would let a claim reach
  outside the shared root it is meant to partition. This is a security
  boundary, not a convenience check — do not bypass it by constructing
  `WorkspaceClaim::writing` paths from unvalidated input.
- Claim order matters: `plan_shared_workspace_dispatch` grants first-writer-
  wins in the order `claims` is given, so callers that need deterministic,
  reproducible plans must pass claims in a stable order.
