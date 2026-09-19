//! Running one superstep's active node set, and folding the results.
//!
//! This is the *execution* half of a superstep, split from the *boundary*
//! half (reducer apply, routing, checkpoint persist — see `boundary.rs`).
//! [`StepRunner`] drives the active node set's handlers, sequentially or
//! concurrently, and hands back a [`StepOutcome`] carrying every
//! `(Activation, Result<NodeResult<Update>>)` pair it actually produced.
//! [`StepRunner::fold_step`] then folds that outcome into a [`StepRun`] —
//! the same fold this executor has always done: applied in active-set index
//! order, stopping at the first error or interrupt.
//!
//! Running and folding are deliberately kept as separate steps (rather than
//! folding inline as each branch completes, as the pre-split code did) so a
//! future change to the fold policy — running every branch of a parallel
//! step to completion and keeping *all* their results instead of discarding
//! completed higher-index siblings on an interrupt/failure (see the C1/C2
//! findings in `docs/runtime-comparison/code-review-graph.md`) touches only
//! `fold_step`. This PR does not change that policy: `fold_step` still stops
//! at the first error/interrupt in `outcome.results`, exactly like the
//! former inline folds did.

use super::*;

use crate::compiled::run_ctx::RunCtx;

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
    /// Branch updates in deterministic active-set index order.
    pub(super) updates: Vec<Update>,
    /// Explicit routing (plain `goto` nodes and/or [`Send`] packets) keyed by
    /// the producing branch's active-set index.
    ///
    /// Keyed by index rather than node id so repeated [`Send`] activations of
    /// the *same* node within a step (map-reduce fanout) each keep their own
    /// [`Command::goto`] — a node-keyed map would let a later activation's
    /// command clobber an earlier one's routing.
    pub(super) goto_map: HashMap<usize, Vec<RouteTarget>>,
    /// The lowest-index branch interrupt, if any (its active-set index +
    /// value).
    pub(super) interrupt: Option<(usize, Interrupt)>,
    /// A node-handler failure that survived the node-retry policy, if any.
    /// When set, `updates` still carries the updates of the branches that
    /// completed *before* the failing branch, so the executor can fold that
    /// partial progress into committed state and persist a resumable
    /// failure boundary.
    pub(super) failure: Option<StepFailure>,
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
    /// Wraps a node future in the configured per-node timeout (if any),
    /// mapping an elapsed deadline onto [`TinyAgentsError::Timeout`].
    async fn run_node_future(
        &self,
        node_id: &NodeId,
        fut: NodeFuture<Update>,
    ) -> Result<NodeResult<Update>> {
        match self.graph.node_timeout {
            Some(timeout) => match tokio::time::timeout(timeout, fut).await {
                Ok(result) => result,
                Err(_) => Err(TinyAgentsError::Timeout(format!(
                    "node `{node_id}` exceeded its {timeout:?} timeout"
                ))),
            },
            None => fut.await,
        }
    }

    /// Runs one node handler under the graph's node-retry policy.
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
    ) -> Result<NodeResult<Update>> {
        let mut attempt = 0usize;
        loop {
            let fut = handler(state.clone(), ctx.clone());
            match self.run_node_future(node_id, fut).await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    let retry = self
                        .graph
                        .node_retry
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
            self.graph.emit(GraphEvent::NodeStarted {
                node: node_id.clone(),
                step,
            });

            let node_ctx = ctx.node_context(node_id, step, None, activation.send_arg.clone());
            let result = self
                .run_node_with_retry(node_id, &node.handler, state, node_ctx, step)
                .await;
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
        let mut futures = Vec::with_capacity(active.len());
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
            let node_ctx = ctx.node_context(node_id, step, fork, activation.send_arg.clone());
            let handler = node.handler.clone();
            let owned_node = node_id.clone();
            // Box each branch future behind a concrete `Send` bound. This
            // keeps the `select_all` rolling window below (used for a
            // `max_concurrency` bound) from requiring a higher-ranked `Send`
            // proof over the borrowed recursion frames, which the compiler
            // cannot discharge for the bare `async` blocks.
            let fut: std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<NodeResult<Update>>> + Send + '_>,
            > = Box::pin(async move {
                self.run_node_with_retry(&owned_node, &handler, state, node_ctx, step)
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

        let results = active.iter().cloned().zip(results).collect::<Vec<_>>();
        Ok(StepOutcome { results })
    }

    /// Folds a single successful branch result into the step accumulators.
    ///
    /// Pushes the node to `visited`, records updates/goto, emits the
    /// matching events, and returns the interrupt (with its branch index)
    /// when the branch paused.
    fn fold_result(
        &self,
        index: usize,
        node_id: &NodeId,
        step: usize,
        result: NodeResult<Update>,
        updates: &mut Vec<Update>,
        goto_map: &mut HashMap<usize, Vec<RouteTarget>>,
        visited: &mut Vec<NodeId>,
    ) -> Option<(usize, Interrupt)> {
        visited.push(node_id.clone());
        match result {
            NodeResult::Update(update) => {
                updates.push(update);
                self.graph.emit(GraphEvent::StateUpdated {
                    node: node_id.clone(),
                    step,
                });
            }
            NodeResult::Command(command) => {
                if let Some(update) = command.update {
                    updates.push(update);
                    self.graph.emit(GraphEvent::StateUpdated {
                        node: node_id.clone(),
                        step,
                    });
                }
                if !command.goto.is_empty() {
                    goto_map.insert(index, command.goto);
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

    /// Folds a [`StepOutcome`] into a [`StepRun`], in active-set index
    /// order, stopping at the first error or interrupt — exactly the fold
    /// the pre-split sequential/parallel loops did inline. Kept as one
    /// function (rather than re-inlined at each call site) so a future
    /// change to this policy (see the module doc) has one place to change.
    pub(super) fn fold_step(
        &self,
        outcome: StepOutcome<Update>,
        step: usize,
        visited: &mut Vec<NodeId>,
    ) -> StepRun<Update> {
        let mut updates: Vec<Update> = Vec::new();
        let mut goto_map: HashMap<usize, Vec<RouteTarget>> = HashMap::new();
        let mut interrupt: Option<(usize, Interrupt)> = None;
        let mut failure: Option<StepFailure> = None;

        for (index, (activation, result)) in outcome.results.into_iter().enumerate() {
            let node_id = &activation.node;
            let result = match result {
                Ok(result) => result,
                Err(error) => {
                    self.graph.emit(GraphEvent::NodeFailed {
                        node: node_id.clone(),
                        step,
                        error: error.to_string(),
                    });
                    failure = Some(StepFailure {
                        failed_index: index,
                        error,
                    });
                    break;
                }
            };

            if let Some(found) = self.fold_result(
                index,
                node_id,
                step,
                result,
                &mut updates,
                &mut goto_map,
                visited,
            ) {
                interrupt = Some(found);
                break;
            }
        }

        StepRun {
            updates,
            goto_map,
            interrupt,
            failure,
        }
    }
}
