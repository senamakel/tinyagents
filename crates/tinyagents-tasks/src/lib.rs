//! Graph-independent detached-task machinery for TinyAgents.
//!
//! This crate is the managed child-work surface: task model, `TaskStore`
//! implementations, the detached task registry, restart reconciliation and the
//! orchestration control tools. It does not depend on the graph engine, so
//! `tinyagents-graph` and `tinyagents-orchestration` both build on it.
//!
//! It gives
//! language-model orchestrators stable task ids and typed controls (`spawn`,
//! `await`, `cancel`, `kill`, `status`, `list`, `timeout`, `race`, `yield`, and
//! `steer`) without exposing raw executor handles such as `tokio::JoinHandle`.
//!
//! The controls are ordinary harness tools. Use [`OrchestrationTool`] directly,
//! call [`orchestration_tools`] to build the full set, or call
//! [`register_orchestration_tools`] to insert them into a
//! [`tinyagents_harness::tool::ToolRegistry`] alongside any other tools.

mod completions;
mod driver_store;
mod reconcile;
mod recovery;
mod runtime;
mod status;
mod storage;
mod store;
mod store_registry;
mod tool;
mod types;

pub use completions::{
    CompletionArtifact, CompletionFormatter, CompletionRecord, CompletionResult, CompletionRouter,
    CompletionState, CompletionStatus, CompletionStore, CompletionStoreReplaceOutcome,
    DEFAULT_MAX_ATTEMPTS, DriverCompletionStore, InMemoryCompletionStore,
    NeutralCompletionFormatter, NotifyMode, RecordOutcome, TombstoneOutcome,
};
pub use completions::{
    CompletionDelivery, CompletionDeliveryConfig, CompletionRouterPool, CompletionRouterPoolError,
    CompletionRouterReplaceOutcome, DeliveryFence, DeliveryOutcome, DeliveryTarget, ParentBusyGate,
    ParentBusyLease, ThreadDeliveryLeases, open_driver_completion_store,
};
pub use driver_store::DriverTaskStore;
pub use reconcile::{
    ReconcileOutcome, ReconcileReport, ReconciledTask, reconcile_orphaned_tasks, task_status_label,
};
pub use recovery::{
    MAX_RECOVERY_CHILDREN, MAX_RECOVERY_LABEL_CHARS, RESTART_RECOVERY_INSTRUCTION, RecoveryChild,
    build_restart_recovery_note, recovery_children,
};
pub use runtime::{DetachedAbortEvent, DetachedTaskRegistry};
pub use storage::{collections, open_storage, storage_schema};
pub use store::{InMemoryTaskStore, TaskStore};
pub use store_registry::{TaskStoreRegistry, TaskStoreRegistryError, open_driver_task_store};
pub use tool::{
    OrchestrationStatusOverlay, OrchestrationTool, SteeringRegistry, orchestration_tool_schema,
    orchestration_tool_schemas, orchestration_tools, orchestration_tools_with_status_overlay,
    orchestration_tools_with_steering, register_orchestration_tools,
    register_orchestration_tools_with_status_overlay,
};
pub use types::*;

#[cfg(test)]
#[path = "lib_tests.rs"]
mod test;
