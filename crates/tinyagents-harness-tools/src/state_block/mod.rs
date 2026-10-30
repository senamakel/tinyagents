//! The **run-state block**: a short, current view of the thread's goal, the
//! run's plan and the facts its tool calls established, added to every model
//! request.
//!
//! Between `todo` calls a model sees its plan only in an old tool result,
//! further back with every round, and what its failed calls taught it is
//! scattered across raw output. [`StateBlockMiddleware`] rebuilds one block
//! before each model call from sources that cannot be wrong:
//!
//! - **the goal**: the thread's goal ([`crate::goals`]) with its status and any
//!   budget left, so a goal set or completed mid-run shows on the next call;
//! - **the plan**: the thread's todo list from the store the `todo` tool
//!   writes ([`crate::todos`]), with the step in progress and how many tool
//!   rounds it has run;
//! - **the notes**: the model's own facts, blockers and current approach,
//!   and those of its last finished task ([`crate::scratchpad`]), written
//!   through the `todo` tool's note arguments;
//! - **the ledger**: facts derived from the run's own tool calls with no model
//!   involved (`ledger`): the last few failed calls with their exit code and
//!   first stderr line, executables a shell reported missing, and files
//!   written by write/edit tools. Facts correct themselves: a later success of
//!   the same command clears its failure, and a later successful command that
//!   runs a missing executable clears it.
//!
//! ```text
//! [run state] Rebuilt by the harness for this call from your goal, plan and tool results; …
//! Goal (active): Add org_id to the auth schema without downtime
//! Plan: step 2 of 3 in progress (4 tool rounds on it).
//!   [x] 1. Dump the schema
//!   [>] 2. Write the migration
//!   [ ] 3. Run the tests
//! Current approach (your note): add the column nullable, backfill, then constrain
//! Verified facts (your notes, oldest first):
//!   - the table is auth_users
//! Recent failures (oldest first; errors quoted from tool output):
//!   - 2 rounds ago, shell `python3 migrate.py`: "exit code 127; stderr: python3: not found"
//! Reported not found: python3
//! Files written this run: /app/migrate.py
//! ```
//!
//! **Request-only.** The block goes in through
//! [`push_ephemeral_instruction`](tinyagents_harness::middleware::push_ephemeral_instruction)
//! at `before_model_dispatch` (re-applied after request rewrites), so it is
//! never written to the transcript: the provider's cached prefix is untouched,
//! the history holds no stale copies, and the model sees exactly one block,
//! always the current one, always at the tail. Nothing is added while there is
//! no goal, plan or fact.
//!
//! **One read path.** The saved part (goal, plan, notes) is a [`RunState`],
//! read with [`RunState::load`]; a host shows or debugs the state through the
//! same type, and [`RunState::render`] gives the block a turn would start with.
//! A part that cannot be read is left out; the others still show.
//! The ledger lives only in the run.
//!
//! **Model-only.** The middleware emits no events, so a host UI sees none of
//! this; what a UI shows of the goal comes from the goal tools as before.

mod ledger;
mod middleware;
mod render;
mod snapshot;
mod types;

pub use render::STATE_BLOCK_HEADER;
pub use types::{RunState, StateBlockMiddleware};
