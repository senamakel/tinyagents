//! Workflow definition types: phases, definitions, and validation errors.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One phase of a declarative workflow.
///
/// A phase specifies a set of agents to work on it concurrently and its
/// dependencies (other phases that must complete first). Agents run in
/// parallel within a phase; the phase itself is the unit of scheduling.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkflowPhase {
    /// The human-readable name of this phase.
    pub name: String,
    /// A description of what this phase does.
    pub description: String,
    /// Agent ids that should work on this phase in parallel.
    pub agent_ids: Vec<String>,
    /// Phase names this phase depends on (must complete first).
    pub depends_on: Vec<String>,
}

/// A host-neutral declarative phase DAG.
///
/// Defines a workflow as a directed acyclic graph of phases, each with
/// associated agents, concurrency limits, and dependencies. The orchestration
/// engine schedules phases topologically and manages bounded parallelism via
/// `default_concurrency` and `max_children`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkflowDefinition {
    /// Unique identifier for this workflow.
    pub id: String,
    /// Human-readable workflow name.
    pub name: String,
    /// Description of what the workflow does.
    pub description: String,
    /// All phases in the workflow.
    pub phases: Vec<WorkflowPhase>,
    /// Default concurrency limit for agents in phases that don't specify one.
    pub default_concurrency: u32,
    /// Maximum number of child tasks that can be spawned concurrently.
    pub max_children: u32,
    /// Host-defined wire metadata which the orchestration engine never reads.
    #[serde(flatten, default)]
    pub extensions: BTreeMap<String, Value>,
}

/// List response used by hosts that expose a workflow catalog.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkflowDefinitionListResponse {
    /// The workflow definitions in this response.
    pub definitions: Vec<WorkflowDefinition>,
    /// Total count of workflows in the host's catalog.
    pub count: usize,
}

/// A structural or host-supplied lookup problem in a workflow definition.
///
/// These errors detect issues that prevent a workflow from running:
/// unknown agent references, missing dependencies, cycles, or invalid
/// concurrency settings. They are host-independent validation issues.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DefinitionError {
    /// A phase references an agent the host does not know about.
    UnknownAgent { phase: String, agent_id: String },
    /// A phase depends on another phase that is not defined.
    UnknownDependency { phase: String, depends_on: String },
    /// Multiple phases have the same name.
    DuplicatePhase { name: String },
    /// A phase has no agents assigned to it.
    EmptyPhase { phase: String },
    /// The phase dependencies form a cycle.
    CyclicDependency,
    /// The workflow has no phases.
    NoPhases,
    /// The concurrency settings are invalid (e.g. default > max).
    InvalidConcurrency {
        default_concurrency: u32,
        max_children: u32,
    },
}
