//! Print the authoritative Rust JSON Schema for persisted session UI records.

use schemars::JsonSchema;
use tinyagents_session::{
    run_ledger::types::{AgentRun, RunEvent, RunTelemetry},
    turn_state::types::TurnState,
};

#[derive(JsonSchema)]
#[schemars(rename_all = "camelCase")]
#[allow(dead_code)]
struct SessionUiWireTypes {
    turn_state: TurnState,
    agent_run: AgentRun,
    run_event: RunEvent,
    run_telemetry: RunTelemetry,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = schemars::generate::SchemaSettings::draft2020_12()
        .for_serialize()
        .into_generator()
        .into_root_schema_for::<SessionUiWireTypes>();
    println!("{}", serde_json::to_string_pretty(&schema)?);
    Ok(())
}
