//! Running one superstep's active node set, and folding the results.
//!
//! This is the *execution* half of a superstep, split from the *boundary*
//! half (reducer apply, routing, checkpoint persist — see `boundary.rs`).
//! [`StepRunner`] drives the active node set's handlers, sequentially or
//! concurrently, and hands back a [`StepOutcome`] carrying every
//! `(Activation, Result<NodeResult<Update>>)` pair it actually produced.
//! [`StepRunner::fold_step`] then folds that outcome into a [`StepRun`].
//!
//! Running and folding are deliberately kept as separate steps (rather than
//! folding inline as each branch completes, as the pre-split code did) so a
//! change to the fold policy touches only `fold_step`. Per the C1/C2
//! findings in `docs/runtime-comparison/code-review-graph.md`, `fold_step`
//! now folds **every** `Ok` result regardless of its position in the active
//! set: a parallel step always drives every branch to completion
//! ([`StepRunner::run_parallel`]), so a higher-index branch that completed
//! before a lower-index one interrupted or failed must not be discarded and
//! re-run on resume. `fold_step` partitions the step's results into
//! `completed` (every branch that produced an `Update`/`Command`, in
//! original active-set-index order) and `stalled` (the branches that
//! errored or interrupted, which become the boundary's `pending` set) —
//! see [`StepRun`].

use super::*;

use crate::cache::TaskCacheKey;
use crate::compiled::run_ctx::RunCtx;

/// Counts how many activations of this step's active set target each node
/// (I1): more than one is a `Send` fan-out of the same node, which
/// [`RunCtx::node_context`] surfaces on [`NodeContext::siblings`] so a
/// subgraph node handler can namespace its child checkpoint by task id
/// instead of sharing one namespace across every fan-out branch.
fn sibling_counts(active: &[Activation]) -> HashMap<NodeId, usize> {
    let mut counts: HashMap<NodeId, usize> = HashMap::new();
    for activation in active {
        *counts.entry(activation.node.clone()).or_insert(0) += 1;
    }
    counts
}

/// The raw, unfolded result of running a superstep's active node set: one
/// `(Activation, Result<NodeResult>)` pair per branch that was actually
/// invoked, in active-set index order.
///
/// [`StepRunner::run_sequential`] stops invoking further branches at the
/// first error or interrupt (so `results` may be a strict prefix of the
/// active set); [`StepRunner::run_parallel`] always drives every branch to
/// completion first (so `results` always covers the whole active set). Ready
/// for [`StepRunner::fold_step`].
pub(super) struct StepOutcome<Update> {
    pub(super) results: Vec<(Activation, Result<NodeResult<Update>>)>,
}

/// The folded result of running a superstep's active node set, ready to
/// apply at the step boundary.
pub(super) struct StepRun<Update> {
    /// Branch updates in deterministic active-set index order, from *every*
    /// branch that produced one (an `Update` or a `Command` carrying one),
    /// regardless of whether a lower-index sibling errored or interrupted.
    pub(super) updates: Vec<Update>,
    /// Explicit routing (plain `goto` nodes and/or [`Send`] packets) keyed by
    /// the producing branch's active-set index.
    ///
    /// Keyed by index rather than node id so repeated [`Send`] activations of
    /// the *same* node within a step (map-reduce fanout) each keep their own
    /// [`Command::goto`] — a node-keyed map would let a later activation's
    /// command clobber an earlier one's routing.
    pub(super) goto_map: HashMap<usize, Vec<RouteTarget>>,
    /// Every branch that completed (produced an `Update`/`Command`, not an
    /// error or interrupt), paired with its original active-set index —
    /// needed so a later `route_completed` call can look its `goto_map`
    /// entry back up by that same index. Superset of what the pre-C1/C2 fold
    /// kept (the index-ascending prefix): a higher-index branch that
    /// completed despite a lower-index sibling erroring/interrupting is
    /// included here rather than dropped.
    pub(super) completed: Vec<(usize, Activation)>,
    /// Every branch that errored or interrupted this step, in ascending
    /// original-index order — the boundary's `pending` set (re-run from
    /// scratch on resume/retry). The first entry is always the branch named
    /// by `interrupt`/`failure` below, when either is set.
    pub(super) stalled: Vec<(usize, Activation)>,
    /// Every branch that interrupted this step, active-set-index-paired, in
    /// ascending index order (I1). Empty when nothing interrupted. Unlike
    /// the pre-I1 fold (which surfaced only the lowest-index interrupt),
    /// every interrupted branch is carried through to the boundary — a
    /// `Send` fan-out of one node interrupting on every concurrent
    /// activation surfaces all of them on
    /// [`GraphExecution::interrupts`](super::GraphExecution), each stamped
    /// with its own branch's task id.
    pub(super) interrupted: Vec<(usize, Interrupt)>,
    /// A node-handler failure that survived the node-retry policy, if any —
    /// always the lowest-index error this step. When set, `updates` still
    /// carries the updates of every branch that completed (not just those
    /// with a lower index), so the executor can fold that partial progress
    /// into committed state and persist a resumable failure boundary.
    pub(super) failure: Option<StepFailure>,
}

/// The two accumulators [`StepRunner::fold_result`] fills in as it walks a
/// step's results: branch updates and explicit routing. Bundled so
/// `fold_result` takes one accumulator instead of two separate `&mut`
/// parameters.
struct FoldAccum<Update> {
    updates: Vec<Update>,
    goto_map: HashMap<usize, Vec<RouteTarget>>,
}

/// Runs one superstep's active node set against a [`CompiledGraph`].
///
/// A thin wrapper around a `&CompiledGraph` borrow — it exists to give the
/// step-running/folding methods a home distinct from the boundary and
/// entry-point methods on `CompiledGraph` itself.
pub(super) struct StepRunner<'g, State, Update> {
    pub(super) graph: &'g CompiledGraph<State, Update>,
}

impl<'g, State, Update> StepRunner<'g, State, Update>
where
    State: Clone + Send + Sync + 'static,
    Update: Send + 'static,
{
    /// Wraps a node future in panic safety and the node's effective
    /// timeouts (if any), mapping an elapsed deadline onto
    /// [`TinyAgentsError::Timeout`].
    ///
    /// Two independent ceilings race the handler: the flat `timeout` (max
    /// wall time for the attempt, regardless of heartbeats) and the
    /// `idle_timeout` (max gap between two [`NodeContext::heartbeat`]
    /// calls, re-armed by each one via [`IdleClock::idle_elapsed`]). Either
    /// firing first fails the attempt; a node that never heartbeats sees
    /// its idle timeout fire exactly `idle_timeout` after start, i.e. as a
    /// flat timeout.
    ///
    /// A node handler that panics unwinds through `join_all`/`fut.await`
    /// unless caught here (I4 part 1): [`futures::FutureExt::catch_unwind`]
    /// converts an unwind into an ordinary `Err`, so the panic flows through
    /// the same failure boundary (checkpoint write, `RunFailed` event, status
    /// `Failed`) as any other node error, instead of poisoning the whole run
    /// future and leaving the status store stuck at `Running`.
    async fn run_node_future(
        &self,
        node_id: &NodeId,
        fut: NodeFuture<Update>,
        policy: &NodePolicy<State, Update>,
        idle_clock: &IdleClock,
    ) -> Result<NodeResult<Update>> {
        let node_id_owned = node_id.clone();
        let guarded = async move {
            match futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(fut)).await {
                Ok(result) => result,
                Err(payload) => Err(Self::panic_error(&node_id_owned, payload)),
            }
        };
        let flat = async {
            match policy.timeout {
                Some(timeout) => tokio::time::sleep(timeout).await,
                None => std::future::pending::<()>().await,
            }
        };
        let idle = async {
            match policy.idle_timeout {
                Some(idle) => idle_clock.idle_elapsed(idle).await,
                None => std::future::pending::<()>().await,
            }
        };
        tokio::pin!(guarded);
        tokio::select! {
            result = &mut guarded => result,
            _ = flat => Err(TinyAgentsError::Timeout(format!(
                "node `{node_id}` exceeded its {:?} timeout",
                policy.timeout.unwrap_or_default()
            ))),
            _ = idle => Err(TinyAgentsError::Timeout(format!(
                "node `{node_id}` exceeded its {:?} idle timeout without a heartbeat",
                policy.idle_timeout.unwrap_or_default()
            ))),
        }
    }

    /// Extracts a printable message from a caught panic payload, preferring a
    /// `&str` then a `String` downcast, and produces the
    /// [`TinyAgentsError::Graph`] that stands in for the panic at the normal
    /// failure boundary.
    fn panic_error(node_id: &NodeId, payload: Box<dyn std::any::Any + Send>) -> TinyAgentsError {
        let message = if let Some(s) = payload.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "non-string panic payload".to_string()
        };
        TinyAgentsError::Graph(format!("node `{node_id}` panicked: {message}"))
    }

    /// Runs one node handler under the node's effective retry policy.
    ///
    /// Builds a fresh handler future (and re-clones the context) for each
    /// attempt, so a retried node re-runs from its start — matching the
    /// durable execution model, where a node is never suspended mid-flight.
    /// On a [retryable][tinyagents_harness::retry::is_retryable] error, when
    /// a [`RetryPolicy`](tinyagents_harness::retry::RetryPolicy) is
    /// configured and permits another attempt, it emits
    /// [`GraphEvent::NodeRetryScheduled`], sleeps the (opt-in) backoff, and
    /// retries. Non-retryable errors, absence of a policy, or an exhausted
    /// attempt budget return the error unchanged. The per-node timeout still
    /// bounds every individual attempt via [`Self::run_node_future`].
    async fn run_node_with_retry(
        &self,
        node_id: &NodeId,
        handler: &Arc<NodeHandler<State, Update>>,
        state: &State,
        ctx: NodeContext,
        step: usize,
        policy: &NodePolicy<State, Update>,
    ) -> Result<NodeResult<Update>> {
        let mut attempt = 0usize;
        loop {
            let fut = handler(state.clone(), ctx.clone());
            match self
                .run_node_future(node_id, fut, policy, &ctx.idle_clock)
                .await
            {
                Ok(result) => return Ok(result),
                Err(error) => {
                    let retry = policy
                        .retry
                        .as_ref()
                        .filter(|policy| policy.should_retry(attempt) && is_retryable(&error));
                    let Some(policy) = retry else {
                        return Err(error);
                    };
                    attempt += 1;
                    self.graph.emit(GraphEvent::NodeRetryScheduled {
                        node: node_id.clone(),
                        step,
                        attempt,
                    });
                    policy.sleep_backoff(attempt).await;
                }
            }
        }
    }

    /// Looks up a live cache entry for `node_id`, when it has a
    /// [`crate::NodeCachePolicy`] installed (via
    /// [`crate::CompiledGraph::with_cached_node`]) and a
    /// [`crate::cache::TaskCache`] backend is attached (via
    /// [`crate::CompiledGraph::with_task_cache`]).
    ///
    /// A cache error, a missing entry, or a value that fails to decode into
    /// `Update` are all treated as a miss (`None`) — caching is an
    /// optimization, never a correctness requirement (see the module docs on
    /// [`crate::cache::TaskCache`]).
    async fn try_cache_get(
        &self,
        node_id: &NodeId,
        state: &State,
        send_arg: Option<&serde_json::Value>,
    ) -> Option<Update> {
        let cached = self.graph.cached_nodes.get(node_id)?;
        let cache = self.graph.task_cache.as_ref()?;
        let hash = (cached.key)(state, send_arg);
        let key = TaskCacheKey::new(self.graph.graph_id.clone(), node_id.clone(), hash);
        let value = cache.get(&key).await.ok().flatten()?;
        (cached.decode)(value).ok()
    }

    /// Stores a cache-miss result for `node_id`, when it has a cache policy
    /// and backend configured, and emits the miss's
    /// [`GraphEvent::TaskCompleted`] (`cached: false`). A no-op for a node
    /// with no cache policy, an error result, an interrupt, or a `Command`
    /// with no update to store.
    /// Synchronously encodes a cache-miss result for storage, without ever
    /// awaiting — so nothing derived from `Update` (which is not necessarily
    /// `Sync`) is ever live across a suspension point. Returns `None` for a
    /// node with no cache policy, an error result, an interrupt, or a
    /// `Command` with no update to store.
    fn prepare_cache_put(
        &self,
        node_id: &NodeId,
        state: &State,
        send_arg: Option<&serde_json::Value>,
        result: &Result<NodeResult<Update>>,
    ) -> Option<(TaskCacheKey, serde_json::Value, Option<Duration>)> {
        let cached = self.graph.cached_nodes.get(node_id)?;
        let result = result.as_ref().ok()?;
        let update = match result {
            NodeResult::Update(update) => Some(update),
            NodeResult::Command(command) => command.update.as_ref(),
            NodeResult::Interrupt(_) => None,
        }?;
        let value = (cached.encode)(update).ok()?;
        let hash = (cached.key)(state, send_arg);
        let key = TaskCacheKey::new(self.graph.graph_id.clone(), node_id.clone(), hash);
        Some((key, value, cached.ttl))
    }

    /// Writes a prepared cache-miss entry (see [`Self::prepare_cache_put`])
    /// and emits its [`GraphEvent::TaskCompleted`] (`cached: false`). Takes
    /// only owned, unconditionally `Send + Sync` values, so this is safe to
    /// await from a context that must itself stay `Send` regardless of
    /// `Update`'s auto-trait bounds.
    async fn store_cache_entry(
        &self,
        key: TaskCacheKey,
        value: serde_json::Value,
        ttl: Option<Duration>,
        node_id: &NodeId,
        step: usize,
    ) {
        let Some(cache) = self.graph.task_cache.as_ref() else {
            return;
        };
        let _ = cache.put(&key, value, ttl).await;
        self.graph.emit(GraphEvent::TaskCompleted {
            node: node_id.clone(),
            step,
            cached: false,
        });
    }


    /// Runs one superstep's active node set — concurrently when the graph
    /// opts into it (`with_parallel`) and more than one node is active, else
    /// sequentially — and folds the result. This is the single entry point
    /// `execute_run` calls per step.
    pub(super) async fn run_step(
        &self,
        ctx: &mut RunCtx<'_, State, Update>,
        active: &[Activation],
        state: &State,
        step: usize,
    ) -> Result<StepRun<Update>> {
        let outcome = if self.graph.parallel && active.len() > 1 {
            self.run_parallel(ctx, active, state, step).await?
        } else {
            self.run_sequential(ctx, active, state, step).await?
        };
        Ok(self.fold_step(outcome, step, &mut ctx.visited))
    }

    /// Runs the active node set one node at a time (default behavior).
    ///
    /// Stops invoking further branches at the first error (the run aborts)
    /// or interrupt (later nodes in the step are not started), exactly
    /// preserving milestone-1 semantics: `outcome.results` ends at that
    /// branch.
    async fn run_sequential(
        &self,
        ctx: &mut RunCtx<'_, State, Update>,
        active: &[Activation],
        state: &State,
        step: usize,
    ) -> Result<StepOutcome<Update>> {
        let siblings = sibling_counts(active);
        let mut results = Vec::with_capacity(active.len());
        for activation in active {
            let node_id = &activation.node;
            let node = self
                .graph
                .nodes
                .get(node_id)
                .ok_or_else(|| TinyAgentsError::MissingNode(node_id.to_string()))?;

            self.graph.emit(GraphEvent::TaskScheduled {
                node: node_id.clone(),
                step,
            });

            let send_arg = activation.send_arg.clone();
            let result = if let Some(update) = self
                .try_cache_get(node_id, state, send_arg.as_ref())
                .await
            {
                self.graph.emit(GraphEvent::TaskCompleted {
                    node: node_id.clone(),
                    step,
                    cached: true,
                });
                Ok(NodeResult::Update(update))
            } else {
                self.graph.emit(GraphEvent::NodeStarted {
                    node: node_id.clone(),
                    step,
                });

                let node_ctx = ctx.node_context(
                    activation,
                    step,
                    None,
                    siblings.get(node_id).copied().unwrap_or(1),
                    state,
                );
                let policy = self.graph.effective_policy(node_id);
                let result = self
                    .run_node_with_retry(node_id, &node.handler, state, node_ctx, step, &policy)
                    .await;
                if let Some((key, value, ttl)) =
                    self.prepare_cache_put(node_id, state, send_arg.as_ref(), &result)
                {
                    self.store_cache_entry(key, value, ttl, node_id, step).await;
                }
                result
            };
            let stop = matches!(result, Err(_) | Ok(NodeResult::Interrupt(_)));
            results.push((activation.clone(), result));
            if stop {
                break;
            }
        }
        Ok(StepOutcome { results })
    }

    /// Runs the active node set concurrently (opt-in via `with_parallel`).
    ///
    /// Each branch executes on its own cloned `State` snapshot and a
    /// distinct [`ForkId`], optionally with the [`Send`] argument that
    /// scheduled it. With no `max_concurrency` bound every branch starts
    /// before any is awaited and all are driven via
    /// [`futures::future::join_all`]; with a bound the active set is run in
    /// chunks of at most that many futures, so at most that many node
    /// handlers are in flight at once. Every branch is driven to completion
    /// before this returns, regardless of whether an earlier branch errored
    /// or interrupted — `outcome.results` always covers the whole active
    /// set; [`Self::fold_step`] is what stops at the lowest-index
    /// error/interrupt.
    async fn run_parallel(
        &self,
        ctx: &mut RunCtx<'_, State, Update>,
        active: &[Activation],
        state: &State,
        step: usize,
    ) -> Result<StepOutcome<Update>> {
        // Build one forked context + future per branch. Node lookup and
        // resume consumption happen up front so the futures borrow nothing
        // mutable; each branch drives its handler through the node-retry
        // policy (which also applies the per-node timeout), so a transient
        // failure in one branch is retried without disturbing its siblings.
        let siblings = sibling_counts(active);
        let mut futures = Vec::with_capacity(active.len());
        // Parallel with a cache hit: `true` at `index` means that branch's
        // slot in `futures` is an already-resolved replay, not a handler
        // invocation — the post-loop pass below must not re-cache it (that
        // would spuriously refresh its TTL on every hit).
        let mut cache_hits = vec![false; active.len()];
        for (index, activation) in active.iter().enumerate() {
            let node_id = &activation.node;
            let node = self
                .graph
                .nodes
                .get(node_id)
                .ok_or_else(|| TinyAgentsError::MissingNode(node_id.to_string()))?;

            self.graph.emit(GraphEvent::TaskScheduled {
                node: node_id.clone(),
                step,
            });

            let send_arg = activation.send_arg.clone();
            if let Some(update) = self
                .try_cache_get(node_id, state, send_arg.as_ref())
                .await
            {
                self.graph.emit(GraphEvent::TaskCompleted {
                    node: node_id.clone(),
                    step,
                    cached: true,
                });
                cache_hits[index] = true;
                let fut: std::pin::Pin<
                    Box<dyn std::future::Future<Output = Result<NodeResult<Update>>> + Send + '_>,
                > = Box::pin(async move { Ok(NodeResult::Update(update)) });
                futures.push(fut);
                continue;
            }

            self.graph.emit(GraphEvent::NodeStarted {
                node: node_id.clone(),
                step,
            });
            self.graph.emit(GraphEvent::ContextForked {
                node: node_id.clone(),
                fork: index,
                step,
            });

            let fork = Some(ForkId::new(index, node_id.clone()));
            let node_ctx = ctx.node_context(
                activation,
                step,
                fork,
                siblings.get(node_id).copied().unwrap_or(1),
                state,
            );
            let handler = node.handler.clone();
            let owned_node = node_id.clone();
            let policy = self.graph.effective_policy(node_id);
            // Box each branch future behind a concrete `Send` bound. This
            // keeps the `select_all` rolling window below (used for a
            // `max_concurrency` bound) from requiring a higher-ranked `Send`
            // proof over the borrowed recursion frames, which the compiler
            // cannot discharge for the bare `async` blocks.
            let fut: std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<NodeResult<Update>>> + Send + '_>,
            > = Box::pin(async move {
                self.run_node_with_retry(&owned_node, &handler, state, node_ctx, step, &policy)
                    .await
            });
            futures.push(fut);
        }

        // Drive branches to completion, bounding in-flight count when
        // configured. With a bound, keep a rolling window of `limit`
        // branches in flight instead of fixed `join_all` chunks. A chunked
        // join runs each chunk to completion before starting the next, so a
        // single slow branch head-of-line blocks the whole chunk; the
        // rolling window starts a new branch as soon as *any* in-flight one
        // finishes. `select_all` reports which pending future completed; a
        // parallel index Vec maps it back to the branch's active-set
        // position, so results are re-ordered into deterministic order for
        // the fold below.
        let results = match self.graph.max_concurrency {
            Some(limit) if limit < futures.len() => {
                let total = futures.len();
                let mut slots: Vec<Option<Result<NodeResult<Update>>>> =
                    (0..total).map(|_| None).collect();
                let mut source = futures.into_iter().enumerate();
                let mut running = Vec::with_capacity(limit);
                let mut running_index = Vec::with_capacity(limit);
                for (index, fut) in source.by_ref().take(limit) {
                    running.push(fut);
                    running_index.push(index);
                }
                while !running.is_empty() {
                    let (result, completed, rest) = futures::future::select_all(running).await;
                    let index = running_index.remove(completed);
                    slots[index] = Some(result);
                    running = rest;
                    if let Some((index, fut)) = source.next() {
                        running.push(fut);
                        running_index.push(index);
                    }
                }
                slots
                    .into_iter()
                    .map(|slot| slot.expect("every branch produced a result"))
                    .collect::<Vec<_>>()
            }
            _ => futures::future::join_all(futures).await,
        };

        // Cache-miss branches (not the already-replayed hits above) store
        // their result now that every branch has settled.
        for (index, activation) in active.iter().enumerate() {
            if cache_hits[index] {
                continue;
            }
            self.try_cache_put(
                &activation.node,
                state,
                activation.send_arg.as_ref(),
                &results[index],
                step,
            )
            .await;
        }

        let results = active.iter().cloned().zip(results).collect::<Vec<_>>();
        Ok(StepOutcome { results })
    }

    /// Folds a single successful branch result into the step accumulators.
    ///
    /// Pushes the node to `visited`, records updates/goto, emits the
    /// matching events, and returns the interrupt (with its branch index)
    /// when the branch paused. Returning `Some` means the branch did *not*
    /// complete (it is a `stalled` branch, not a `completed` one) even
    /// though it is not an `Err`.
    fn fold_result(
        &self,
        index: usize,
        node_id: &NodeId,
        step: usize,
        result: NodeResult<Update>,
        accum: &mut FoldAccum<Update>,
        visited: &mut Vec<NodeId>,
    ) -> Option<(usize, Interrupt)> {
        visited.push(node_id.clone());
        match result {
            NodeResult::Update(update) => {
                accum.updates.push(update);
                self.graph.emit(GraphEvent::StateUpdated {
                    node: node_id.clone(),
                    step,
                });
            }
            NodeResult::Command(command) => {
                if let Some(update) = command.update {
                    accum.updates.push(update);
                    self.graph.emit(GraphEvent::StateUpdated {
                        node: node_id.clone(),
                        step,
                    });
                }
                if !command.goto.is_empty() {
                    accum.goto_map.insert(index, command.goto);
                }
            }
            NodeResult::Interrupt(emitted) => {
                self.graph.emit(GraphEvent::InterruptEmitted {
                    interrupt: emitted.clone(),
                });
                return Some((index, emitted));
            }
        }
        self.graph.emit(GraphEvent::NodeCompleted {
            node: node_id.clone(),
            step,
        });
        None
    }

    /// Folds a [`StepOutcome`] into a [`StepRun`].
    ///
    /// Per the module doc (C1/C2), this walks *every* result in
    /// `outcome.results` — never stopping early — and partitions each
    /// branch into `completed` (an `Update`/`Command` result) or `stalled`
    /// (an error or an interrupt). The first error and the first interrupt
    /// encountered (in ascending original-index order) are recorded as this
    /// step's `failure`/`interrupt`; every stalled branch, including any
    /// later error/interrupt beyond the first, still lands in `stalled` so
    /// the boundary can schedule it for resume rather than silently
    /// dropping it or mistaking it for completed. For a sequential run
    /// (which already stops invoking further branches at the first
    /// stop condition — see [`Self::run_sequential`]), `outcome.results` is
    /// simply a strict prefix, so this fold is behaviorally identical to the
    /// old stop-early fold in that mode; the behavior change is scoped to
    /// parallel steps, where `outcome.results` always covers the whole
    /// active set.
    fn fold_step(
        &self,
        outcome: StepOutcome<Update>,
        step: usize,
        visited: &mut Vec<NodeId>,
    ) -> StepRun<Update> {
        let mut accum = FoldAccum {
            updates: Vec::new(),
            goto_map: HashMap::new(),
        };
        let mut completed: Vec<(usize, Activation)> = Vec::new();
        let mut stalled: Vec<(usize, Activation)> = Vec::new();
        let mut interrupted: Vec<(usize, Interrupt)> = Vec::new();
        let mut failure: Option<StepFailure> = None;

        for (index, (activation, result)) in outcome.results.into_iter().enumerate() {
            let node_id = activation.node.clone();
            match result {
                Err(error) => {
                    self.graph.emit(GraphEvent::NodeFailed {
                        node: node_id,
                        step,
                        error: error.to_string(),
                    });
                    if failure.is_none() {
                        failure = Some(StepFailure {
                            failed_index: index,
                            error,
                        });
                    }
                    stalled.push((index, activation));
                }
                Ok(result) => {
                    match self.fold_result(index, &node_id, step, result, &mut accum, visited) {
                        Some(found) => {
                            interrupted.push(found);
                            stalled.push((index, activation));
                        }
                        None => completed.push((index, activation)),
                    }
                }
            }
        }

        StepRun {
            updates: accum.updates,
            goto_map: accum.goto_map,
            completed,
            stalled,
            interrupted,
            failure,
        }
    }
}
