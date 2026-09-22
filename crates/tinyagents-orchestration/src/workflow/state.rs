//! Phase state projection and advancement for workflow runs.
//!
//! This module owns the JSON phase-state document (a BTreeMap of phase names
//! to status + outputs) that the workflow engine persists. It provides queries
//! (what phases are runnable?) and mutations (mark complete, reset after
//! interruption) over this state without touching the underlying persistence layer.

use serde_json::{Value, json};

use super::{WorkflowDefinition, WorkflowPhase};

/// Durable status of one phase in a workflow run.
///
/// Phases transition: Pending → Running → (Completed | Failed). A phase
/// interrupted while running can be reset to Pending for retry; completed
/// phases are immutable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseStatus {
    /// Phase has not yet started.
    Pending,
    /// Phase is currently executing (likely in a child).
    Running,
    /// Phase completed successfully; immutable.
    Completed,
    /// Phase failed; may be retried by resetting to Pending.
    Failed,
}

impl PhaseStatus {
    /// Returns the string representation of this status (used in JSON).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }
}

/// Initializes a phase-state document from a workflow definition.
///
/// All phases start in the Pending status with empty outputs. The returned
/// JSON structure maps phase names to `{ status, outputs, ... }` objects.
pub fn init_phase_states(definition: &WorkflowDefinition) -> Value {
    Value::Object(
        definition
            .phases
            .iter()
            .map(|phase| {
                (
                    phase.name.clone(),
                    json!({ "status": "pending", "outputs": [] }),
                )
            })
            .collect(),
    )
}

/// Queries the current status string of a phase, or None if the phase is unknown.
pub fn phase_status<'a>(phase_states: &'a Value, name: &str) -> Option<&'a str> {
    phase_states.get(name)?.get("status")?.as_str()
}

pub(crate) fn set_phase_status(
    phase_states: &mut Value,
    name: &str,
    status: PhaseStatus,
    outputs: Option<Value>,
) {
    let Some(entries) = phase_states.as_object_mut() else {
        return;
    };
    let entry = entries
        .entry(name.to_owned())
        .or_insert_with(|| json!({ "status": "pending", "outputs": [] }));
    if let Some(object) = entry.as_object_mut() {
        object.insert("status".to_owned(), json!(status.as_str()));
        if let Some(outputs) = outputs {
            object.insert("outputs".to_owned(), outputs);
        }
    }
}

pub(crate) fn set_phase_reason(phase_states: &mut Value, name: &str, reason: &str) {
    if let Some(object) = phase_states.get_mut(name).and_then(Value::as_object_mut) {
        object.insert("reason".to_owned(), json!(reason));
    }
}

/// Make an interrupted phase runnable again. A stopped phase may have spawned
/// children whose results were never durably collected; retrying the whole
/// phase is the only safe, deterministic recovery. Completed phases remain
/// immutable and are never retried.
pub fn reset_running_phases(phase_states: &mut Value, reason: &str) {
    let Some(phases) = phase_states.as_object_mut() else {
        return;
    };
    for entry in phases.values_mut() {
        let Some(state) = entry.as_object_mut() else {
            continue;
        };
        if state.get("status").and_then(Value::as_str) == Some("running") {
            state.insert("status".to_owned(), json!(PhaseStatus::Pending.as_str()));
            state.insert("outputs".to_owned(), json!([]));
            state.insert("reason".to_owned(), json!(reason));
        }
    }
}

/// Finds the next phase that should run: a phase that is not already
/// completed, running, or failed, and all of whose dependencies are completed.
///
/// Returns the first such phase in definition order, or `None` if no phase is
/// ready (either all are done, or some have unmet dependencies).
pub fn next_runnable_phase<'a>(
    definition: &'a WorkflowDefinition,
    phase_states: &Value,
) -> Option<&'a WorkflowPhase> {
    definition.phases.iter().find(|phase| {
        !matches!(
            phase_status(phase_states, &phase.name),
            Some("completed" | "running" | "failed")
        ) && phase
            .depends_on
            .iter()
            .all(|dependency| phase_status(phase_states, dependency) == Some("completed"))
    })
}

/// Checks whether all phases in the workflow have completed.
pub fn all_phases_completed(definition: &WorkflowDefinition, phase_states: &Value) -> bool {
    definition
        .phases
        .iter()
        .all(|phase| phase_status(phase_states, &phase.name) == Some("completed"))
}

/// Collects outputs from all upstream (dependency) phases for a given phase.
/// Filters out empty and null outputs to yield only meaningful results.
pub fn upstream_outputs(phase: &WorkflowPhase, phase_states: &Value) -> Vec<Value> {
    phase
        .depends_on
        .iter()
        .flat_map(|dependency| {
            phase_states
                .get(dependency)
                .and_then(|entry| entry.get("outputs"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(move |item| {
                    durable_output(item)
                        .filter(|output| match output {
                            Value::Null => false,
                            Value::String(text) => !text.trim().is_empty(),
                            _ => true,
                        })
                        .map(|output| json!({ "phase": dependency, "output": output }))
                })
        })
        .collect()
}

/// Composes the prompt for a worker in a phase.
///
/// Includes the phase name and description, the input question, the worker's
/// index (if multiple workers), and relevant upstream outputs from dependencies.
pub fn phase_prompt(
    input: &Value,
    phase: &WorkflowPhase,
    index: usize,
    upstream: &[Value],
) -> String {
    let question = input
        .get("question")
        .or_else(|| input.get("input"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| input.to_string());
    let mut prompt = format!(
        "Workflow phase: {}\n{}\n\nInput:\n{}\n",
        phase.name, phase.description, question
    );
    if phase.agent_ids.len() > 1 {
        prompt.push_str(&format!(
            "\n(You are worker #{} in this phase.)\n",
            index + 1
        ));
    }
    if !upstream.is_empty() {
        prompt.push_str("\nContext from prior phases:\n");
        for item in upstream {
            if let (Some(source), Some(output)) = (
                item.get("phase").and_then(Value::as_str),
                item.get("output"),
            ) {
                prompt.push_str(&format!("- [{source}] {}\n", render_output(output)));
            }
        }
    }
    prompt
}

/// Composes a workflow summary from all final phase outputs.
///
/// Returns `None` if all phases are complete and there are no outputs;
/// otherwise returns a formatted summary of all non-empty outputs in phase order.
pub fn synthesize_summary(definition: &WorkflowDefinition, phase_states: &Value) -> Option<String> {
    let outputs_for = |name: &str| {
        phase_states
            .get(name)
            .and_then(|entry| entry.get("outputs"))
            .and_then(Value::as_array)
            .map(|outputs| {
                outputs
                    .iter()
                    .filter_map(|output| durable_output(output).map(|value| render_output(&value)))
                    .filter(|output| !output.trim().is_empty() && output != "null")
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .filter(|summary| !summary.trim().is_empty())
    };
    outputs_for("synthesize").or_else(|| {
        definition
            .phases
            .iter()
            .rev()
            .find_map(|phase| outputs_for(&phase.name))
    })
}

/// The public/RPC projection remains `{ agentId, output: String }` for
/// compatibility. New rows carry the exact result in `metadata.rawOutput` so
/// future phases retain arbitrary JSON without changing the old wire shape.
fn durable_output(item: &Value) -> Option<Value> {
    item.get("metadata")
        .and_then(|metadata| metadata.get("version"))
        .and_then(Value::as_u64)
        .filter(|version| *version >= 2)
        .and_then(|_| {
            item.get("metadata")
                .and_then(|metadata| metadata.get("rawOutput"))
        })
        .cloned()
        .or_else(|| item.get("output").cloned())
}

/// Preserve every JSON output in prompt context and summaries.  JSON object's
/// map ordering is canonical under serde_json's default map implementation,
/// so repeated resume/synthesis renders the same bytes rather than silently
/// discarding structured child results.
fn render_output(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        _ => serde_json::to_string(value).unwrap_or_else(|_| "<unserializable output>".to_owned()),
    }
}
