use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One phase of a declarative workflow.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkflowPhase {
    pub name: String,
    pub description: String,
    pub agent_ids: Vec<String>,
    pub depends_on: Vec<String>,
}

/// A host-neutral declarative phase DAG.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkflowDefinition {
    pub id: String,
    pub name: String,
    pub description: String,
    pub phases: Vec<WorkflowPhase>,
    pub default_concurrency: u32,
    pub max_children: u32,
    /// Host-defined wire metadata which the orchestration engine never reads.
    #[serde(flatten, default)]
    pub extensions: BTreeMap<String, Value>,
}

/// List response used by hosts that expose a workflow catalog.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkflowDefinitionListResponse {
    pub definitions: Vec<WorkflowDefinition>,
    pub count: usize,
}

/// A structural or host-supplied lookup problem in a workflow definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DefinitionError {
    UnknownAgent {
        phase: String,
        agent_id: String,
    },
    UnknownDependency {
        phase: String,
        depends_on: String,
    },
    DuplicatePhase {
        name: String,
    },
    EmptyPhase {
        phase: String,
    },
    CyclicDependency,
    NoPhases,
    InvalidConcurrency {
        default_concurrency: u32,
        max_children: u32,
    },
}
