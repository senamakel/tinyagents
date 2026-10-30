use super::*;

#[test]
fn run_ledger_schemas_use_the_persisted_camel_case_fields_and_enum_names() {
    let run_schema = serde_json::to_value(schemars::schema_for!(AgentRun)).unwrap();
    let run_properties = run_schema["properties"].as_object().unwrap();
    for field in [
        "parentThreadId",
        "workerThreadId",
        "startedAt",
        "completedAt",
    ] {
        assert!(
            run_properties.contains_key(field),
            "AgentRun schema is missing {field}"
        );
    }

    let event_schema = serde_json::to_value(schemars::schema_for!(RunEvent)).unwrap();
    let event_properties = event_schema["properties"].as_object().unwrap();
    for field in ["runId", "eventType", "timestamp"] {
        assert!(
            event_properties.contains_key(field),
            "RunEvent schema is missing {field}"
        );
    }

    assert_eq!(
        serde_json::to_value(AgentRunKind::WorkerThread).unwrap(),
        "worker_thread"
    );
    assert_eq!(
        serde_json::to_value(AgentRunStatus::AwaitingUser).unwrap(),
        "awaiting_user"
    );
}
