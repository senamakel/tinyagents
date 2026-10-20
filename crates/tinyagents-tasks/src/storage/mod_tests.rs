use super::*;
use serde_json::{Value, json};
use tinyagents_storage_kit::{
    DocumentStoreExt, MigrateOptions, Precondition, Registry, Scope, ScopeClass, ScopedStorage,
    StorageBackend,
};
use tinystoragedrivers_testkit::MemoryBackend;

fn scoped() -> ScopedStorage {
    MemoryBackend::new()
        .for_scope(&Scope::new("local").unwrap())
        .unwrap()
}

async fn put_v1(storage: &ScopedStorage, collection: &str, id: &str, body: Value) {
    storage
        .documents()
        .put(collection, id, body, Precondition::None)
        .await
        .unwrap();
}

async fn migrate(storage: &ScopedStorage) {
    let registry = Registry::compose([storage_schema()]).unwrap();
    for item in storage_schema().items {
        if let tinyagents_storage_kit::Item::Collection { spec, .. } = item {
            storage.documents().ensure_collection(&spec).await.unwrap();
        }
    }
    registry
        .migrate(storage, &MigrateOptions::new())
        .await
        .unwrap();
}

fn secs(s: u64) -> Value {
    json!({"secs_since_epoch": s, "nanos_since_epoch": 5_000_000})
}

fn v1_record(task: &str, status: &str) -> Value {
    json!({
        "spec": {"task_id": task, "kind": {"type": "sub_agent", "agent": "worker"},
                 "parent_run_id": "run-1", "root_run_id": "root-1", "thread_id": "t-1",
                 "metadata": {"k": "v"}},
        "status": status,
        "result": {"text": "done"},
        "created_at": secs(1_700_000_000),
        "updated_at": secs(1_700_000_050),
        "started_at": secs(1_700_000_010),
        "ended_at": secs(1_700_000_050),
    })
}

#[test]
fn the_schema_composes_and_names_everything_the_tasks_crate_owns() {
    let schema = storage_schema();
    assert_eq!(schema.owner.as_str(), "tasks");
    let names: Vec<&str> = schema.items.iter().map(|item| item.name()).collect();
    assert_eq!(
        names,
        ["tasks.records", "tasks.history", "tasks.completions"]
    );
    assert!(
        schema
            .items
            .iter()
            .all(|item| item.scope() == ScopeClass::Agent)
    );
    Registry::compose([schema]).unwrap();
}

#[tokio::test]
async fn v1_task_records_become_bare_snake_case_documents_under_key_ids() {
    let storage = scoped();
    put_v1(
        &storage,
        "orchestration_tasks",
        "hashed",
        json!({"taskId": "task/a", "status": "completed", "kind": "sub_agent", "rev": 3,
               "record": v1_record("task/a", "completed")}),
    )
    .await;
    put_v1(
        &storage,
        "orchestration_tasks",
        "bad",
        json!({"taskId": "x", "record": {"nope": 1}}),
    )
    .await;
    migrate(&storage).await;

    let row = storage
        .documents()
        .get("tasks.records", "task%2Fa")
        .await
        .unwrap()
        .expect("id is Key::of([task_id])");
    assert_eq!(row.doc["task_id"], "task/a");
    assert_eq!(row.doc["status"], "completed");
    assert_eq!(row.doc["kind_label"], "sub_agent");
    assert_eq!(row.doc["rev"], 3);
    assert_eq!(row.doc["root_run_id"], "root-1");
    assert_eq!(row.doc["created_ms"], 1_700_000_000_005_i64);
    assert_eq!(row.doc["started_ms"], 1_700_000_010_005_i64);
    assert_eq!(row.doc["ended_ms"], 1_700_000_050_005_i64);
    assert_eq!(row.doc["result"]["text"], "done");
    assert!(row.doc.get("record").is_none(), "no wrapper");
    assert!(row.doc.get("updated_at").is_none(), "updated is metadata");
    assert_eq!(row.meta.schema, 2);
    assert_ne!(
        row.meta.created_ms, 1_700_000_000_005_i64,
        "a cross-collection migration lets the host assign new metadata time"
    );
    let kept = storage
        .documents()
        .query_all("tasks.records", &tinyagents_storage_kit::Query::all())
        .await
        .unwrap();
    assert_eq!(kept.len(), 1, "an undecodable record is dropped");
}

#[tokio::test]
async fn opening_tasks_storage_runs_its_owner_migrations_before_returning() {
    let backend = MemoryBackend::new();
    let scope = Scope::new("local").unwrap();
    let legacy = backend.for_scope(&scope).unwrap();
    put_v1(
        &legacy,
        "orchestration_tasks",
        "old-task",
        json!({"taskId": "old-task", "rev": 1, "record": v1_record("old-task", "completed")}),
    )
    .await;

    let scoped = backend.for_scope(&scope).unwrap();
    let opened = super::open_storage(scoped).await.unwrap();
    let migrated = opened
        .documents()
        .get("tasks.records", "old-task")
        .await
        .unwrap()
        .expect("the owner migration ran as part of open");
    assert_eq!(migrated.doc["task_id"], "old-task");
    assert_eq!(migrated.doc["status"], "completed");
}

#[tokio::test]
async fn opening_tasks_storage_migrates_a_v2_9_1_public_store_fixture() {
    use crate::{DriverTaskStore, TaskStore};
    use tinyagents_harness::ids::TaskId;

    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/legacy-v2.9.1-tasks.json")).unwrap();
    let storage = MemoryBackend::new()
        .for_scope(&Scope::new("local").unwrap())
        .unwrap();
    for collection in ["orchestration_tasks", "orchestration_task_history"] {
        for row in fixture[collection].as_array().unwrap() {
            storage
                .documents()
                .put(
                    collection,
                    row["id"].as_str().unwrap(),
                    row["doc"].clone(),
                    Precondition::None,
                )
                .await
                .unwrap();
        }
    }

    let opened = open_storage(storage).await.unwrap();
    let store = DriverTaskStore::new(std::sync::Arc::clone(opened.documents())).unwrap();
    let id = TaskId::new("legacy-task-v291");
    let record = store.get(&id).unwrap();
    assert_eq!(record.status, crate::OrchestrationTaskStatus::Completed);
    assert_eq!(
        record.spec.thread_id.as_ref().map(|id| id.as_str()),
        Some("thread-fixture")
    );
    assert_eq!(
        record.spec.metadata.get("source").map(String::as_str),
        Some("v2.9.1-public-writer")
    );
    assert_eq!(
        record.result.unwrap().text.as_deref(),
        Some("fixture completed")
    );
    assert_eq!(
        store
            .history(&id)
            .iter()
            .map(|row| row.status)
            .collect::<Vec<_>>(),
        [
            crate::OrchestrationTaskStatus::Pending,
            crate::OrchestrationTaskStatus::Running,
            crate::OrchestrationTaskStatus::Completed,
        ]
    );
}

#[tokio::test]
async fn public_store_reads_documents_written_by_pre_migration_31cc_release() {
    use crate::{DriverTaskStore, TaskStore};
    use tinyagents_harness::ids::TaskId;

    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/legacy-31cc-tasks.json")).unwrap();
    let backend = MemoryBackend::new();
    let storage = backend.for_scope(&Scope::new("local").unwrap()).unwrap();

    for spec in collections() {
        storage.documents().ensure_collection(&spec).await.unwrap();
    }
    for collection in ["tasks.records", "tasks.history"] {
        for row in fixture[collection].as_array().unwrap() {
            storage
                .documents()
                .put(
                    collection,
                    row["id"].as_str().unwrap(),
                    row["doc"].clone(),
                    Precondition::None,
                )
                .await
                .unwrap();
        }
    }

    let opened = open_storage(storage).await.unwrap();
    let store = DriverTaskStore::new(std::sync::Arc::clone(opened.documents())).unwrap();
    let id = TaskId::new("legacy-task-31cc");
    let record = store.get(&id).unwrap();
    assert_eq!(record.status, crate::OrchestrationTaskStatus::Completed);
    assert_eq!(
        record.spec.thread_id.as_ref().map(|id| id.as_str()),
        Some("thread-fixture")
    );
    assert_eq!(
        record.spec.metadata.get("source").map(String::as_str),
        Some("31cc-public-writer")
    );
    assert_eq!(
        record.result.unwrap().text.as_deref(),
        Some("fixture completed")
    );
    assert_eq!(
        store
            .history(&id)
            .iter()
            .map(|row| row.status)
            .collect::<Vec<_>>(),
        [
            crate::OrchestrationTaskStatus::Pending,
            crate::OrchestrationTaskStatus::Running,
            crate::OrchestrationTaskStatus::Completed,
        ]
    );
}

#[tokio::test]
async fn v1_history_snapshots_move_under_task_and_revision() {
    let storage = scoped();
    for rev in 1..=2_u64 {
        put_v1(
            &storage,
            "orchestration_task_history",
            &format!("h{rev}"),
            json!({"taskId": "t", "rev": rev, "record": v1_record("t", "running")}),
        )
        .await;
    }
    migrate(&storage).await;
    for rev in 1..=2 {
        let row = storage
            .documents()
            .get("tasks.history", &format!("t/{rev}"))
            .await
            .unwrap()
            .expect("id is Key::of([task_id, rev])");
        assert_eq!(row.doc["task_id"], "t");
        assert_eq!(row.doc["rev"], rev);
        assert_eq!(row.doc["status"], "running");
    }
}

#[tokio::test]
async fn v1_completions_move_with_unwrapped_bodies_and_ms_times() {
    let storage = scoped();
    put_v1(
        &storage,
        "orchestration_completions",
        "hashed",
        json!({"taskId": "c/1", "parentKey": "p", "state": "pending", "updatedAtMs": 1,
               "record": {
                   "task_id": "c/1", "parent_key": "p", "agent_id": "worker",
                   "status": "success", "result": {"text": "hi"},
                   "finished_at": secs(1_700_000_000), "updated_at": secs(1_700_000_001),
                   "attempts": 2, "state": "pending", "notify_mode": "collect"}}),
    )
    .await;
    put_v1(
        &storage,
        "orchestration_completions",
        "bad",
        json!({"taskId": "x", "record": 5}),
    )
    .await;
    migrate(&storage).await;
    let row = storage
        .documents()
        .get("tasks.completions", "c%2F1")
        .await
        .unwrap()
        .expect("id is Key::of([task_id])");
    assert_eq!(row.doc["parent_key"], "p");
    assert_eq!(row.doc["agent_id"], "worker");
    assert_eq!(row.doc["state"], "pending");
    assert_eq!(row.doc["notify_mode"], "collect");
    assert_eq!(row.doc["attempts"], 2);
    assert_eq!(row.doc["result"]["text"], "hi");
    assert_eq!(row.doc["finished_ms"], 1_700_000_000_005_i64);
    assert!(row.doc.get("updated_at").is_none() && row.doc.get("updatedAtMs").is_none());
    assert_ne!(
        row.meta.created_ms, 1_700_000_000_005_i64,
        "a cross-collection migration lets the host assign new metadata time"
    );
    let kept = storage
        .documents()
        .query_all("tasks.completions", &tinyagents_storage_kit::Query::all())
        .await
        .unwrap();
    assert_eq!(kept.len(), 1);
}

#[tokio::test]
async fn migrating_twice_changes_nothing() {
    let storage = scoped();
    put_v1(
        &storage,
        "orchestration_tasks",
        "h",
        json!({"taskId": "a", "rev": 1, "record": v1_record("a", "pending")}),
    )
    .await;
    migrate(&storage).await;
    let before = storage.documents().get("tasks.records", "a").await.unwrap();
    migrate(&storage).await;
    let after = storage.documents().get("tasks.records", "a").await.unwrap();
    assert_eq!(before.unwrap().doc, after.unwrap().doc);
}
