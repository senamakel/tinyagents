use std::sync::Arc;

use super::*;
use crate::store::TaskStore;
use crate::types::{OrchestrationTaskKind, OrchestrationTaskStatus};
use tinyagents_storage_kit::{Scope, StorageBackend};
use tinystoragedrivers_testkit::MemoryBackend;

fn docs_of(storage: &dyn StorageBackend, scope: &str) -> Arc<dyn DocumentStore> {
    Arc::clone(
        storage
            .for_scope(&Scope::new(scope).unwrap())
            .unwrap()
            .documents(),
    )
}

fn spec(id: &str) -> OrchestrationTaskSpec {
    OrchestrationTaskSpec::new(
        id,
        OrchestrationTaskKind::Graph {
            graph_id: "child".into(),
        },
    )
}

fn store() -> DriverTaskStore {
    DriverTaskStore::new(docs_of(&MemoryBackend::new(), "local")).unwrap()
}

#[test]
fn lifecycle_matches_the_in_memory_state_machine() {
    let driver = store();
    let memory = crate::InMemoryTaskStore::new();
    let id = TaskId::new("task-a");
    for store in [&driver as &dyn TaskStore, &memory] {
        let pending = store.insert(spec("task-a")).unwrap();
        assert_eq!(pending.status, OrchestrationTaskStatus::Pending);
        // Illegal transitions are refused identically.
        assert!(
            store
                .complete(&id, OrchestrationTaskResult::text("x"))
                .is_ok()
        );
        assert!(store.mark_running(&id).is_err(), "completed is terminal");
        assert!(store.insert(spec("task-a")).is_err(), "duplicate id");
        assert!(store.mark_running(&TaskId::new("missing")).is_err());
    }
}

#[test]
fn every_transition_lands_in_the_history_oldest_first() {
    let store = store();
    let id = TaskId::new("task-a");
    store.insert(spec("task-a")).unwrap();
    store.mark_running(&id).unwrap();
    store
        .mark_awaiting_with_question(&id, "which branch?".into())
        .unwrap();
    store.request_cancel(&id).unwrap();
    store.mark_cancelled(&id).unwrap();
    let history = store.history(&id);
    let statuses: Vec<_> = history.iter().map(|r| r.status).collect();
    assert_eq!(
        statuses,
        [
            OrchestrationTaskStatus::Pending,
            OrchestrationTaskStatus::Running,
            OrchestrationTaskStatus::Awaiting,
            OrchestrationTaskStatus::CancelRequested,
            OrchestrationTaskStatus::Cancelled,
        ]
    );
    assert_eq!(history[2].error.as_deref(), Some("which branch?"));
    assert_eq!(
        store.get(&id).unwrap().status,
        OrchestrationTaskStatus::Cancelled
    );
    assert!(store.history(&TaskId::new("missing")).is_empty());
}

#[test]
fn an_already_requested_cancellation_is_idempotent() {
    let store = store();
    let id = TaskId::new("task-a");
    store.insert(spec("task-a")).unwrap();
    assert_eq!(
        store.request_cancel(&id).unwrap().message,
        "cancellation requested"
    );
    assert_eq!(
        store.request_cancel(&id).unwrap().message,
        "cancellation was already requested"
    );
    let killed = store.kill(&id).unwrap();
    assert_eq!(killed.status, OrchestrationTaskStatus::Abandoned);
    assert!(store.kill(&id).is_err(), "abandoned is terminal");
}

#[test]
fn state_and_history_survive_a_driver_reopen_on_the_same_backend() {
    let storage = MemoryBackend::new();
    let id = TaskId::new("task-a");
    {
        let store = DriverTaskStore::new(docs_of(&storage, "local")).unwrap();
        store.insert(spec("task-a")).unwrap();
        store.mark_running(&id).unwrap();
        store
            .complete(&id, OrchestrationTaskResult::text("done"))
            .unwrap();
        assert_eq!(store.history(&id).len(), 3);
    }
    let reopened = DriverTaskStore::new(docs_of(&storage, "local")).unwrap();
    let record = reopened
        .get(&id)
        .expect("task survives reopening a driver on the same backend");
    assert_eq!(record.status, OrchestrationTaskStatus::Completed);
    assert_eq!(record.result.unwrap().text.as_deref(), Some("done"));
    assert_eq!(reopened.history(&id).len(), 3);
}

#[test]
fn list_filters_by_lineage_status_kind_and_time() {
    let store = store();
    store
        .insert(spec("g1").with_lineage("run-1", "root-1").with_thread("t1"))
        .unwrap();
    store
        .insert(
            OrchestrationTaskSpec::new(
                "s1",
                OrchestrationTaskKind::SubAgent {
                    agent: "worker".into(),
                },
            )
            .with_lineage("run-2", "root-1")
            .with_node("spawner"),
        )
        .unwrap();
    store
        .insert(spec("g2").with_lineage("run-3", "root-2"))
        .unwrap();
    store.mark_running(&TaskId::new("g2")).unwrap();

    let ids = |filter: OrchestrationTaskFilter| -> Vec<String> {
        store
            .list(filter)
            .into_iter()
            .map(|r| r.spec.task_id.as_str().to_owned())
            .collect()
    };
    assert_eq!(ids(OrchestrationTaskFilter::default()), ["g1", "g2", "s1"]);
    assert_eq!(
        ids(OrchestrationTaskFilter {
            root_run_id: Some("root-1".into()),
            ..Default::default()
        }),
        ["g1", "s1"]
    );
    assert_eq!(
        ids(OrchestrationTaskFilter {
            parent_run_id: Some("run-2".into()),
            ..Default::default()
        }),
        ["s1"]
    );
    assert_eq!(
        ids(OrchestrationTaskFilter {
            thread_id: Some("t1".into()),
            ..Default::default()
        }),
        ["g1"]
    );
    assert_eq!(
        ids(OrchestrationTaskFilter {
            node_id: Some("spawner".into()),
            ..Default::default()
        }),
        ["s1"]
    );
    assert_eq!(
        ids(OrchestrationTaskFilter {
            status: Some(OrchestrationTaskStatus::Running),
            ..Default::default()
        }),
        ["g2"]
    );
    assert_eq!(
        ids(OrchestrationTaskFilter::default().with_kind("sub_agent")),
        ["s1"]
    );
    assert!(
        ids(OrchestrationTaskFilter::default().created_between(None, Some(std::time::UNIX_EPOCH)))
            .is_empty()
    );
    assert_eq!(
        ids(OrchestrationTaskFilter::default().created_between(Some(std::time::UNIX_EPOCH), None))
            .len(),
        3
    );
}

/// Two executors racing to settle one task: exactly one wins and the loser
/// sees a terminal task rather than overwriting the winner.
#[test]
fn racing_settlements_have_one_winner() {
    let storage = MemoryBackend::new();
    let docs = docs_of(&storage, "local");
    let first = DriverTaskStore::new(docs.clone()).unwrap();
    first.insert(spec("race")).unwrap();
    let id = TaskId::new("race");
    let handles: Vec<_> = (0..8)
        .map(|n| {
            let store = DriverTaskStore::new(docs.clone()).unwrap();
            let id = id.clone();
            std::thread::spawn(move || {
                store
                    .complete(&id, OrchestrationTaskResult::text(format!("writer-{n}")))
                    .is_ok()
            })
        })
        .collect();
    let wins = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .filter(|won| *won)
        .count();
    assert_eq!(wins, 1);
    let history = first.history(&id);
    assert_eq!(history.len(), 2, "insert + exactly one completion");
}

#[test]
fn tenants_never_share_tasks() {
    let storage = MemoryBackend::new();
    let alice = DriverTaskStore::new(docs_of(&storage, "alice")).unwrap();
    let bob = DriverTaskStore::new(docs_of(&storage, "bob")).unwrap();
    alice.insert(spec("shared")).unwrap();
    assert!(bob.get(&TaskId::new("shared")).is_none());
    // The same id is free in the other tenant.
    bob.insert(spec("shared")).unwrap();
    assert!(bob.list(OrchestrationTaskFilter::default()).len() == 1);
}

#[test]
fn any_task_id_is_a_valid_key() {
    let store = store();
    for id in ["with space", "a/b", "ünï", "..", &"x".repeat(5_000)] {
        store.insert(spec(id)).unwrap();
        assert!(store.get(&TaskId::new(id)).is_some(), "id {id:.12}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn works_inside_a_multi_thread_runtime() {
    let store = store();
    let id = TaskId::new("task-mt");
    store.insert(spec("task-mt")).unwrap();
    store.mark_running(&id).unwrap();
    assert_eq!(store.history(&id).len(), 2);
}

#[tokio::test]
async fn works_inside_a_current_thread_runtime() {
    let store = store();
    let id = TaskId::new("task-ct");
    store.insert(spec("task-ct")).unwrap();
    store.mark_running(&id).unwrap();
    assert_eq!(store.history(&id).len(), 2);
}

#[tokio::test]
async fn the_opener_migrates_and_returns_a_durable_store() {
    let backend = MemoryBackend::new();
    let scope = Scope::new("local").unwrap();
    let durable = crate::open_driver_task_store(backend.for_scope(&scope).unwrap())
        .await
        .unwrap();
    durable.insert(spec("t1")).unwrap();
    let reopened = crate::open_driver_task_store(backend.for_scope(&scope).unwrap())
        .await
        .unwrap();
    assert_eq!(reopened.list(OrchestrationTaskFilter::default()).len(), 1);
}

#[tokio::test]
async fn records_and_history_are_v2_documents_with_key_ids_and_ms_times() {
    let storage = MemoryBackend::new();
    let docs = docs_of(&storage, "local");
    let store = DriverTaskStore::new(docs.clone()).unwrap();
    let id = TaskId::new("a/b");
    store
        .insert(spec("a/b").with_lineage("run-1", "root-1"))
        .unwrap();
    store.mark_running(&id).unwrap();

    let row = docs.get("tasks.records", "a%2Fb").await.unwrap().unwrap();
    assert_eq!(row.doc["task_id"], "a/b");
    assert_eq!(row.doc["status"], "running");
    assert_eq!(row.doc["kind_label"], "graph");
    assert_eq!(row.doc["rev"], 2);
    assert_eq!(row.doc["root_run_id"], "root-1");
    assert!(row.doc["created_ms"].as_i64().unwrap() > 1_600_000_000_000);
    assert!(row.doc.get("record").is_none() && row.doc.get("updated_at").is_none());
    assert_eq!(row.meta.schema, 2);
    assert!(row.meta.updated_ms > 0);

    let first = docs.get("tasks.history", "a%2Fb/1").await.unwrap().unwrap();
    assert_eq!(first.doc["status"], "pending");
    assert_eq!(first.meta.schema, 2);
    assert!(
        docs.get("tasks.history", "a%2Fb/2")
            .await
            .unwrap()
            .is_some()
    );
    assert_eq!(
        store.get(&id).unwrap().updated_at,
        std::time::UNIX_EPOCH + std::time::Duration::from_millis(row.meta.updated_ms as u64),
        "the update time is the document's metadata"
    );
}
