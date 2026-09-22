//! Per-thread **goal + todo list** working together on one thread.
//!
//! This offline example wires both `graph::goals` and `graph::todos` on a single
//! thread and lets the goal *drive* the list:
//!
//! - A durable [`ThreadGoal`] ("ship the v2 release") is the completion
//!   contract, with a token budget.
//! - A [`TodoList`] holds the concrete steps (three items).
//! - A `goal_gate_node` forms a self-driving loop: each iteration the `work`
//!   node rewrites the list one transition further along (Pending →
//!   InProgress → Completed), and once every item is Completed it marks the
//!   goal `Complete`. The gate keeps looping while the goal is Active and under
//!   budget, accounting the iteration's token usage, and routes to `END` when
//!   the goal completes.
//!
//! Both primitives persist on one shared [`InMemoryStore`], addressed by the
//! run's thread id.
//!
//! Run with:
//!
//! ```text
//! cargo run --example goals_and_todos
//! ```

use std::sync::Arc;

use tinyagents_graph::END;
use tinyagents_graph::command::NodeResult;
use tinyagents_graph::*;
use tinyagents_graph::{NodeContext, NodeFuture};
use tinyagents_harness::store::{InMemoryStore, Store};
use tinyagents_harness::*;
use tinyagents_language::*;
use tinyagents_registry::*;

/// The thread both primitives are scoped to.
const THREAD: &str = "release-thread";

/// Roughly the tokens each work iteration "spends", accounted against the goal.
const TOKENS_PER_ITERATION: u64 = 500;

/// State overwritten by each work iteration — just a step counter for display.
#[derive(Clone, Debug, Default)]
struct ReleaseState {
    iteration: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    // One store backs both the goal and the list for this thread.
    let store: Arc<dyn Store> = Arc::new(InMemoryStore::default());

    // 1. Set the durable objective with a generous token budget.
    goal_store::set(
        &store,
        THREAD,
        "Ship the v2 release",
        Some(100_000), // token budget
    )
    .await?;

    // 2. Seed the list with the concrete steps.
    todo_store::replace(
        &store,
        THREAD,
        [
            "Write the changelog",
            "Tag the release",
            "Publish the crate",
        ]
        .into_iter()
        .map(TodoItem::new)
        .collect(),
    )
    .await?;

    println!(
        "Initial list:\n{}\n",
        todo_store::list(&store, THREAD).await?.markdown
    );

    // 3a. The work node: rewrite the list ONE transition further along per
    // iteration, then complete the goal once every item is Completed.
    let work_store = store.clone();
    let work_node = move |mut state: ReleaseState, _ctx: NodeContext| {
        let store = work_store.clone();
        Box::pin(async move {
            state.iteration += 1;
            let mut items = todo_store::list(&store, THREAD).await?.items;

            if let Some(active) = items
                .iter_mut()
                .find(|item| item.status == TodoStatus::InProgress)
            {
                // Finish the step currently in progress.
                active.status = TodoStatus::Completed;
                println!("  ✓ completed: {}", active.content);
                todo_store::replace(&store, THREAD, items).await?;
            } else if let Some(next) = items
                .iter_mut()
                .find(|item| item.status == TodoStatus::Pending)
            {
                // Pull the next step into progress (single-in-progress invariant).
                next.status = TodoStatus::InProgress;
                println!("  → started: {}", next.content);
                todo_store::replace(&store, THREAD, items).await?;
            } else {
                // Every step is Completed — the objective is satisfied.
                goal_store::complete(&store, THREAD).await?;
                println!("  ★ all steps completed → goal complete");
            }

            Ok(NodeResult::Update(state))
        }) as NodeFuture<ReleaseState>
    };

    // 3b. The gate: account each iteration's usage and loop while the goal is
    // Active and under budget, else route to END.
    let gate = goal_gate_node::<ReleaseState, ReleaseState>(
        store.clone(),
        "work",
        |_state: &ReleaseState| GoalProgress {
            tokens_used: TOKENS_PER_ITERATION,
            elapsed_secs: 1,
            made_progress: true,
        },
    );

    // 4. Wire the self-driving loop: START → work → gate → (work | END).
    let graph = GraphBuilder::<ReleaseState, ReleaseState>::overwrite()
        .with_recursion_limit(64)
        .add_node("work", work_node)
        .add_node("gate", gate)
        .set_entry("work")
        .add_edge("work", "gate")
        .with_command_destinations("gate", ["work", END])
        .compile()?;

    println!("Running the goal-driven loop:");
    let exec = graph
        .run_with_thread(THREAD, ReleaseState::default())
        .await?;

    // 5. Report the final state of both primitives.
    let goal = goal_store::get(&store, THREAD).await?.expect("goal exists");
    let todos = todo_store::list(&store, THREAD).await?;

    println!(
        "\nFinished after {} work iterations.\n",
        exec.state.iteration
    );
    println!(
        "Goal: {} — status={}, tokens_used={}/{}",
        goal.objective,
        goal.status.as_str(),
        goal.tokens_used,
        goal.token_budget
            .map(|b| b.to_string())
            .unwrap_or_else(|| "∞".into()),
    );
    println!("\nFinal list:\n{}", todos.markdown);

    Ok(())
}
