use serde_json::{Value, json};

use super::{WorkflowDefinition, WorkflowPhase};

/// Durable status of one phase in a workflow run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

impl PhaseStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }
}

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

pub fn next_runnable_phase<'a>(
    definition: &'a WorkflowDefinition,
    phase_states: &Value,
) -> Option<&'a WorkflowPhase> {
    definition.phases.iter().find(|phase| {
        !matches!(
            phase_status(phase_states, &phase.name),
            Some("completed" | "running")
        ) && phase
            .depends_on
            .iter()
            .all(|dependency| phase_status(phase_states, dependency) == Some("completed"))
    })
}

pub fn all_phases_completed(definition: &WorkflowDefinition, phase_states: &Value) -> bool {
    definition
        .phases
        .iter()
        .all(|phase| phase_status(phase_states, &phase.name) == Some("completed"))
}

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
                    item.get("output")
                        .and_then(Value::as_str)
                        .filter(|output| !output.trim().is_empty())
                        .map(|output| json!({ "phase": dependency, "output": output }))
                })
        })
        .collect()
}

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
                item.get("output").and_then(Value::as_str),
            ) {
                prompt.push_str(&format!("- [{source}] {output}\n"));
            }
        }
    }
    prompt
}

pub fn synthesize_summary(definition: &WorkflowDefinition, phase_states: &Value) -> Option<String> {
    let outputs_for = |name: &str| {
        phase_states
            .get(name)
            .and_then(|entry| entry.get("outputs"))
            .and_then(Value::as_array)
            .map(|outputs| {
                outputs
                    .iter()
                    .filter_map(|output| output.get("output").and_then(Value::as_str))
                    .filter(|output| !output.trim().is_empty())
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
