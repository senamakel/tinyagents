//! Workflow definition validation: structural checks and error detection.
//!
//! Validates a workflow definition for structural issues (missing phases,
//! duplicate names, cycles, invalid concurrency) and host-specific issues
//! (unknown agents). Errors are deterministic and host-independent.

use tinyagents_graph::dag::{DagIssue, DagNode, validate_dag};

use super::{DefinitionError, WorkflowDefinition};

/// Validate properties that do not require a host agent registry.
pub fn validate_structure(definition: &WorkflowDefinition) -> Vec<DefinitionError> {
    if definition.phases.is_empty() {
        return vec![DefinitionError::NoPhases];
    }

    let mut errors = definition
        .phases
        .iter()
        .filter(|phase| phase.agent_ids.is_empty())
        .map(|phase| DefinitionError::EmptyPhase {
            phase: phase.name.clone(),
        })
        .collect::<Vec<_>>();
    let nodes = definition
        .phases
        .iter()
        .map(|phase| {
            DagNode::new(
                phase.name.as_str(),
                phase.depends_on.iter().map(String::as_str),
            )
        })
        .collect::<Vec<_>>();
    errors.extend(validate_dag(&nodes).into_iter().map(|issue| match issue {
        DagIssue::DuplicateNode { id } => DefinitionError::DuplicatePhase { name: id },
        DagIssue::UnknownDependency { node, depends_on } => DefinitionError::UnknownDependency {
            phase: node,
            depends_on,
        },
        DagIssue::Cycle => DefinitionError::CyclicDependency,
    }));
    if definition.default_concurrency == 0 || definition.max_children == 0 {
        errors.push(DefinitionError::InvalidConcurrency {
            default_concurrency: definition.default_concurrency,
            max_children: definition.max_children,
        });
    }
    errors
}

/// Validate agent identifiers using a host-owned registry lookup.
pub fn validate_agents<F>(definition: &WorkflowDefinition, is_known: F) -> Vec<DefinitionError>
where
    F: Fn(&str) -> bool,
{
    definition
        .phases
        .iter()
        .flat_map(|phase| {
            phase
                .agent_ids
                .iter()
                .filter(|agent_id| !is_known(agent_id))
                .map(|agent_id| DefinitionError::UnknownAgent {
                    phase: phase.name.clone(),
                    agent_id: agent_id.clone(),
                })
        })
        .collect()
}
