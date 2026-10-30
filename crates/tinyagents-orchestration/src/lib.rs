//! Host-neutral subagent invocation and lifecycle orchestration.
//!
//! This crate owns TinyAgents' host-neutral async subagent lifecycle driver,
//! detached jobs, and completion handling. Hosts supply persistence and
//! concrete execution; policy, credentials, progress, and RPC remain host
//! concerns.
//!
//! `OrchestrationTaskStatus` is the canonical status; the job, detached, outcome,
//! run-ledger and completion statuses convert to and from it with `From` /
//! `TryFrom` (see the `tinyagents-tasks` status mapping docs).
//!
//! Dependency direction is deliberately one way:
//! `orchestration -> {harness, session, tasks, storage ports}`. Lower-level
//! crates never depend on this composition layer.

pub mod run_progress_ledger;
pub mod steering;
pub mod subagent;

#[cfg(test)]
#[path = "lib_boundary_tests.rs"]
mod boundary_tests;
