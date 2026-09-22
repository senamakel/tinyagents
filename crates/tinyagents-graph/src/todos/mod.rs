//! Per-thread **todo list**: an ordered checklist of steps per thread.
//!
//! Where [`graph::goals`](crate::goals) holds a single durable objective
//! per thread, the todo list holds the concrete steps the run is working
//! through: an ordered list of [`TodoItem`]s, each `pending`, `in_progress`
//! or `completed`. This module owns the data model and markdown rendering
//! ([`types`]), harness-[`Store`](tinyagents_harness::store::Store)-backed
//! persistence with the single-`InProgress` invariant ([`store`]), and the
//! model-facing tool ([`tool`]).
//!
//! The list is the shape Claude Code and Codex keep: the model rewrites the
//! whole list as it works, and nothing else hangs off an item — no ids, no
//! approvals, no assignment, no run log. A list is always `(Store, thread_id)`.

pub mod store;
mod tool;
mod types;

pub use tool::{TodoTool, register_todo_tools, todo_tools};
pub use types::{
    TodoItem, TodoList, TodoStatus, TodosSnapshot, normalise_list, parse_status, render_markdown,
};

#[cfg(test)]
mod test;
