//! TinyAgents graph runtime.
//!
//! The graph module is TinyAgents' durable workflow runtime (LangGraph-style)
//! and one of the load-bearing surfaces of the crate's recursive language-model
//! architecture: because a node can embed another compiled graph
//! ([`subgraph`]) or invoke a sub-agent, **graphs run graphs** and orchestration
//! recurses while every step stays typed, checkpointed, and observable. A
//! workflow driven from a host orchestrator lowers into exactly these same
//! types.
//!
//! The pieces: partial updates and reducers ([`reducer`]), commands and
//! interrupts ([`command`]), a builder/compile contract ([`builder`]), a
//! superstep executor ([`compiled`]), checkpointing ([`checkpoint`]),
//! streaming/events ([`stream`]), run-status snapshots ([`status`]), graph
//! export/visualization ([`export`]), dependency-DAG validation ([`dag`]),
//! subgraph embedding ([`subgraph`]), and
//! per-thread productivity primitives — a durable goal ([`goals`]) and a todo
//! checklist ([`todos`]) — exposed as harness tools.
//!
//! Each concern lives in its own submodule with `types.rs` (definitions),
//! `mod.rs` (implementations), and `test.rs` (unit tests).

pub mod agent_loop;
pub mod builder;
pub mod cache;
pub mod channel;
pub mod checkpoint;
pub mod command;
pub mod compiled;
pub mod dag;
pub mod delegation;
pub mod export;
pub mod goals;
pub mod observability;
pub mod orchestration;
pub mod parallel;
pub mod recursion;
pub mod reducer;
pub mod status;
pub mod storage;
pub mod stream;
pub mod subagent_node;
pub mod subgraph;
pub mod testkit;
pub(crate) mod thread_locks;
pub mod todos;

pub use tinyagents_harness::error::{Result, TinyAgentsError};

#[cfg(test)]
mod test_support;

// --- Durable execution model ---
pub use builder::{
    END, ForkId, GraphBuilder, GraphDefaults, IdleClock, NodeCachePolicy, NodeContext, NodeFuture,
    NodeHandler, NodePolicy, Route, RouterFn, START,
};
pub use cache::{DriverTaskCache, InMemoryTaskCache, TaskCache, TaskCacheKey};
pub use channel::{
    Barrier, BinaryAggregate, Channel, ChannelSet, ChannelState, ChannelUpdate, ChannelWrite,
    Delta, Ephemeral, LastValue, Messages, NamedBarrier, ReducerRegistry, Topic, Untracked,
};
pub use checkpoint::DriverCheckpointer;
pub use checkpoint::{
    BarrierArrivals, CHECKPOINT_FORMAT_VERSION, Checkpoint, CheckpointConfig, CheckpointMetadata,
    CheckpointSource, CheckpointTuple, Checkpointer, CompletedTask, DurabilityMode,
    InMemoryCheckpointer, PendingActivation, PendingWrite,
};
pub use command::{Command, Interrupt, NodeResult, RouteTarget, Send};
pub use compiled::{
    CompiledGraph, DrainHandle, DrainSignal, GraphExecution, GraphInput, ResumeTarget, RunOptions,
    StateSnapshot,
};
pub use dag::{DagIssue, DagNode};
pub use delegation::{
    CURRENT_SCHEMA_VERSION as DELEGATION_SCHEMA_VERSION, DelegationConfig, DelegationOutcome,
    DelegationStage, DelegationStageOutput, DelegationState, PendingApproval, StepRecord,
    delegation_graph_topology, deny_decision, resume_delegation, run_delegation,
    run_delegation_durable, run_or_resume_delegation,
};
pub use export::{
    ChannelInfo, ConditionalEdgeInfo, EdgeInfo, GraphPolicySummary, GraphTopology,
    GraphTopologyReport, NodeInfo, NodePolicySummary, RouteInfo, ValidationReport, WaitingEdgeInfo,
    from_json, to_json, to_mermaid,
};
pub use goals::store as goal_store;
pub use goals::{
    BudgetVerdict, GoalBudgetGuard, GoalProgress, GoalTool, GoalToolKind, GoalUpdateHook,
    ThreadGoal, ThreadGoalStatus, TurnOutcome, account_turn, accrues_usage,
    active_goal_context_block, goal_gate_node, goal_tools, note_user_turn, register_goal_tools,
    run_continuation_tick, turn_tokens,
};
pub use observability::{
    GraphEventJournal, GraphHealthSummary, GraphLangfuseExporter, GraphLatencyMetrics,
    GraphNodeHealth, GraphNodeLatency, GraphObservation, GraphStatusStore, GraphStepLatency,
    GraphTracingSink, InMemoryGraphEventJournal, InMemoryGraphStatusStore, JournalGraphSink,
    SpanMetadataFn, StoreGraphEventJournal, build_graph_langfuse_batch,
};
pub use orchestration::{
    CancelledDetachedTask, DetachedTaskRegistry, DetachedTaskRegistryError, DetachedTaskSnapshot,
    DetachedTaskWaitOutcome, DriverTaskStore, InMemoryTaskStore, OrchestrationControlOutcome,
    OrchestrationTaskFilter, OrchestrationTaskKind, OrchestrationTaskRecord,
    OrchestrationTaskResult, OrchestrationTaskSpec, OrchestrationTaskStatus, OrchestrationTool,
    OrchestrationToolKind, SteeringRegistry, TaskStore, orchestration_tool_schema,
    orchestration_tool_schemas, orchestration_tools, orchestration_tools_with_steering,
    register_orchestration_tools,
};
pub use parallel::{FailurePolicy, ItemOutcome, ParallelOptions, ParallelOutcome, map_reduce};
pub use recursion::{
    ChildRun, ChildRunSink, RecursionFrame, RecursionPolicy, RecursionStack, RunTree,
};
pub use reducer::{
    AppendReducer, ClosureReducer, ClosureStateReducer, MaxReducer, MinReducer, OverwriteReducer,
    OverwriteStateReducer, Reducer, SetUnionReducer, StateReducer,
};
pub use status::GraphRunStatus;
pub use storage::{
    collections, import_legacy_leases, import_legacy_task_cache, open_storage, storage_schema,
};
pub use stream::{
    CollectingSink, GraphEvent, GraphEventEnvelope, GraphEventSink, NoopSink, StreamMode,
    StreamProjection, project_graph_event,
};
#[allow(deprecated)]
pub use subagent_node::{
    AgentInvocation, AgentInvocationBinding, AgentInvoker, InputMapper, OutputMapper,
    SubAgentBudget, SubAgentInput, SubAgentNode, SubAgentOutput, SubAgentPolicy, subagent_node,
};
pub use subgraph::{adapter_subgraph_node, shared_subgraph_node};
pub use testkit::{
    GraphAssertions, GraphEventRecorder, GraphRun, RetryCountingNode, StreamCollector,
    assert_graph, failing_node, fanout_node, interrupting_node, noop_node, run_recorded,
    scripted_route_node, scripted_update_node, subagent_fake_node, subgraph_test_node,
};
pub use todos::store as todo_store;
pub use todos::{
    TodoItem, TodoList, TodoSequenceMiddleware, TodoStatus, TodoTool, TodosSnapshot,
    normalise_list, parse_status, register_todo_tools, render_markdown, todo_tools,
};
