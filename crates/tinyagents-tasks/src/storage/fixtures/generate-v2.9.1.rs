use std::path::PathBuf;

use tinyagents_harness::ids::TaskId;
use tinyagents_tasks::{
    DriverTaskStore, OrchestrationTaskKind, OrchestrationTaskResult, OrchestrationTaskSpec,
    TaskStore,
};
use tinystoragedrivers_core::{MemoryStorage, Query, Scope, StorageBackend};

fn main() {
    let output = PathBuf::from(std::env::args().nth(1).expect("output path"));
    let storage = MemoryStorage::new();
    let scoped = storage.for_scope(&Scope::new("local").unwrap()).unwrap();
    let docs = scoped.documents();
    let store = DriverTaskStore::new(std::sync::Arc::clone(docs)).unwrap();
    store
        .insert(
            OrchestrationTaskSpec::new(
                "legacy-task-v291",
                OrchestrationTaskKind::Graph {
                    graph_id: "fixture-graph".into(),
                },
            )
            .with_lineage("run-parent", "run-root")
            .with_thread("thread-fixture")
            .with_node("node-fixture")
            .with_timeout_ms(30_000)
            .with_input(serde_json::json!({"prompt":"fixture input","attempt":2}))
            .with_metadata("source", "v2.9.1-public-writer"),
        )
        .unwrap();
    let id = TaskId::new("legacy-task-v291");
    store.mark_running(&id).unwrap();
    store
        .complete(&id, OrchestrationTaskResult::text("fixture completed"))
        .unwrap();

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut collections = serde_json::Map::new();
    for collection in ["orchestration_tasks", "orchestration_task_history"] {
        let rows = runtime
            .block_on(docs.query(collection, &Query::all()))
            .unwrap()
            .items;
        collections.insert(collection.into(), serde_json::to_value(rows).unwrap());
    }
    std::fs::write(output, serde_json::to_vec_pretty(&collections).unwrap()).unwrap();
}
