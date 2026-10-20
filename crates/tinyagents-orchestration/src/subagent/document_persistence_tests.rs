use super::*;
use std::sync::Arc;
use tinyagents_storage_kit::{Scope, StorageBackend};
use tinystoragedrivers_testkit::MemoryBackend;

fn key(parent: &str) -> SubagentTaskKey {
    SubagentTaskKey {
        root_run_id: "root".into(),
        parent_run_id: parent.into(),
        thread_id: Some("thread".into()),
        task_id: "task".into(),
    }
}

fn pause(key: SubagentTaskKey, checkpoint: &str) -> PersistedSubagentPause {
    let task = key.task_id.clone();
    PersistedSubagentPause {
        key,
        replaces: None,
        outcome: SubagentOutcome {
            status: SubagentOutcomeKind::AwaitingInput(super::super::SubagentPause {
                reason: "answer?".into(),
                resume: SubagentResume {
                    checkpoint: Some(checkpoint.into()),
                    ..Default::default()
                },
            }),
            output: checkpoint.into(),
            ..SubagentOutcome::cancelled(task)
        },
    }
}

fn open(backend: &MemoryBackend, scope: &Scope) -> DocumentSubagentPersistence {
    let scoped = backend.for_scope(scope).unwrap();
    DocumentSubagentPersistence::new(Arc::clone(scoped.documents())).unwrap()
}

#[tokio::test]
async fn reopened_store_preserves_pause_cas_terminal_and_scoped_task_index() {
    let backend = MemoryBackend::new();
    let store = open(&backend, &Scope::new("local").unwrap());
    let first = key("first");
    let second = key("second");
    assert_eq!(
        store.save_pause(pause(first.clone(), "a")).await.unwrap(),
        SubagentPausePersistenceDisposition::Inserted
    );
    assert_eq!(
        store.save_pause(pause(second.clone(), "b")).await.unwrap(),
        SubagentPausePersistenceDisposition::Inserted
    );
    let reopened = open(&backend, &Scope::new("local").unwrap());
    let resume = reopened.load(&first).await.unwrap().unwrap();
    assert_eq!(
        reopened.keys_for_task("task").unwrap(),
        vec![first.clone(), second.clone()]
    );
    assert_eq!(
        reopened
            .save_pause(pause(first.clone(), "discard"))
            .await
            .unwrap(),
        SubagentPausePersistenceDisposition::Existing
    );
    assert_eq!(
        reopened.load_pause(&first).await.unwrap().unwrap().output,
        "a"
    );
    assert_eq!(
        reopened
            .record_terminal(
                &first,
                &SubagentOutcome::completed("task", "premature"),
                None
            )
            .await
            .unwrap(),
        SubagentTerminalPersistenceDisposition::PauseExisting
    );
    assert_eq!(
        reopened
            .record_terminal(
                &first,
                &SubagentOutcome::completed("task", "winner"),
                Some(&resume)
            )
            .await
            .unwrap(),
        SubagentTerminalPersistenceDisposition::Inserted
    );
    assert_eq!(store.load(&first).await.unwrap(), None);
    assert_eq!(
        store.terminal_outcome(&first).unwrap().unwrap().output,
        "winner"
    );
    assert_eq!(store.keys_for_task("task").unwrap(), vec![second]);
    assert_eq!(
        store
            .save_pause(pause(first.clone(), "resurrect"))
            .await
            .unwrap(),
        SubagentPausePersistenceDisposition::TerminalExisting
    );
    assert_eq!(
        store
            .record_terminal(&first, &SubagentOutcome::completed("task", "loser"), None)
            .await
            .unwrap(),
        SubagentTerminalPersistenceDisposition::Existing
    );
}

#[tokio::test]
async fn tenant_documents_do_not_leak_between_identical_task_keys() {
    let backend = MemoryBackend::new();
    let alice = open(&backend, &Scope::new("alice").unwrap());
    let bob = open(&backend, &Scope::new("bob").unwrap());
    let key = key("same");
    alice
        .save_pause(pause(key.clone(), "secret"))
        .await
        .unwrap();
    assert_eq!(bob.load(&key).await.unwrap(), None);
    assert!(bob.keys_for_task("task").unwrap().is_empty());
    bob.record_terminal(&key, &SubagentOutcome::completed("task", "bob"), None)
        .await
        .unwrap();
    assert!(alice.terminal_outcome(&key).unwrap().is_none());
    assert_eq!(
        alice.load_pause(&key).await.unwrap().unwrap().output,
        "secret"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn independent_stores_choose_one_pause_winner_and_return_its_actual_outcome() {
    let backend = MemoryBackend::new();
    let first = open(&backend, &Scope::new("local").unwrap());
    let second = open(&backend, &Scope::new("local").unwrap());
    let key = key("race");
    let left = pause(key.clone(), "left");
    let right = pause(key.clone(), "right");
    let barrier = Arc::new(std::sync::Barrier::new(3));
    let first_worker = first.clone();
    let left_barrier = Arc::clone(&barrier);
    let left = std::thread::spawn(move || {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        left_barrier.wait();
        runtime.block_on(first_worker.save_pause(left)).unwrap()
    });
    let second_worker = second.clone();
    let right_barrier = Arc::clone(&barrier);
    let right = std::thread::spawn(move || {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        right_barrier.wait();
        runtime.block_on(second_worker.save_pause(right)).unwrap()
    });
    barrier.wait();
    let (left, right) = (left.join().unwrap(), right.join().unwrap());
    let dispositions = [left, right];
    assert_eq!(
        dispositions
            .iter()
            .filter(|x| **x == SubagentPausePersistenceDisposition::Inserted)
            .count(),
        1
    );
    assert_eq!(
        dispositions
            .iter()
            .filter(|x| **x == SubagentPausePersistenceDisposition::Existing)
            .count(),
        1
    );
    let winner = first.load_pause(&key).await.unwrap().unwrap().output;
    assert!(winner == "left" || winner == "right");
    assert_eq!(
        second.load_pause(&key).await.unwrap().unwrap().output,
        winner
    );
}

#[tokio::test]
async fn import_fills_missing_parts_without_replacing_committed_state() {
    let backend = MemoryBackend::new();
    let store = open(&backend, &Scope::new("local").unwrap());
    let key = key("legacy");
    store
        .import_record(&key, Some(pause(key.clone(), "legacy")), None)
        .unwrap();
    assert_eq!(
        store.load_pause(&key).await.unwrap().unwrap().output,
        "legacy"
    );
    store
        .import_record(&key, Some(pause(key.clone(), "old")), None)
        .unwrap();
    assert_eq!(
        store.load_pause(&key).await.unwrap().unwrap().output,
        "legacy"
    );
    let resume = store.load(&key).await.unwrap().unwrap();
    store
        .record_terminal(
            &key,
            &SubagentOutcome::completed("task", "new"),
            Some(&resume),
        )
        .await
        .unwrap();
    store
        .import_record(
            &key,
            Some(pause(key.clone(), "resurrect")),
            Some(SubagentOutcome::completed("task", "old")),
        )
        .unwrap();
    assert_eq!(store.load(&key).await.unwrap(), None);
    assert_eq!(store.terminal_outcome(&key).unwrap().unwrap().output, "new");
    assert!(store.keys_for_task("task").unwrap().is_empty());
}

#[tokio::test]
async fn raw_kv_records_reopen_without_changing_host_legacy_namespace() {
    use tinyagents_storage_kit::{Kv, Precondition};
    let backend = MemoryBackend::new();
    let scoped = backend.for_scope(&Scope::new("local").unwrap()).unwrap();
    let docs = Arc::clone(scoped.documents());
    let store = DocumentSubagentPersistence::new(docs.clone()).unwrap();
    let key = key("raw");
    let persisted = pause(key.clone(), "raw-checkpoint");
    store.declare().await.unwrap();
    let name = DocumentSubagentPersistence::name(&key);
    let legacy = serde_json::json!({"pause": {"host_checkpoint": "untouched"}});
    Kv::ns(&docs, OWNER, "lifecycle")
        .unwrap()
        .put(&name, &legacy)
        .await
        .unwrap();
    let id = Key::of([OWNER, NS, &name]).unwrap();
    docs.put(
        tinyagents_storage_kit::primitives::KV,
        id.as_str(),
        serde_json::json!({"owner": OWNER, "ns": NS, "key": name, "value": {"pause": persisted}}),
        Precondition::Absent,
    )
    .await
    .unwrap();
    assert_eq!(
        store.load_pause(&key).await.unwrap().unwrap().output,
        "raw-checkpoint"
    );
    let untouched = Kv::ns(&docs, OWNER, "lifecycle")
        .unwrap()
        .get::<serde_json::Value>(&name)
        .await
        .unwrap();
    assert_eq!(untouched, Some(legacy));
}

#[tokio::test]
async fn reopened_driver_keeps_terminal_and_stale_continuations_cannot_reopen_it() {
    let backend = MemoryBackend::new();
    let key = key("restart");
    let resume = {
        let docs = backend.for_scope(&Scope::new("local").unwrap()).unwrap();
        let store = DocumentSubagentPersistence::new(Arc::clone(docs.documents())).unwrap();
        store
            .save_pause(pause(key.clone(), "checkpoint"))
            .await
            .unwrap();
        store.load(&key).await.unwrap().unwrap()
    };
    {
        let docs = backend.for_scope(&Scope::new("local").unwrap()).unwrap();
        let store = DocumentSubagentPersistence::new(Arc::clone(docs.documents())).unwrap();
        assert_eq!(store.load(&key).await.unwrap(), Some(resume.clone()));
        assert_eq!(
            store
                .record_terminal(
                    &key,
                    &SubagentOutcome::completed("task", "done"),
                    Some(&resume)
                )
                .await
                .unwrap(),
            SubagentTerminalPersistenceDisposition::Inserted
        );
    }
    let docs = backend.for_scope(&Scope::new("local").unwrap()).unwrap();
    let store = DocumentSubagentPersistence::new(Arc::clone(docs.documents())).unwrap();
    assert_eq!(
        store.terminal_outcome(&key).unwrap().unwrap().output,
        "done"
    );
    let mut stale = pause(key.clone(), "stale");
    stale.replaces = Some(resume);
    assert_eq!(
        store.save_pause(stale).await.unwrap(),
        SubagentPausePersistenceDisposition::TerminalExisting
    );
    assert!(store.keys_for_task("task").unwrap().is_empty());
}

#[tokio::test]
async fn pause_replacement_rejects_stale_continuations_and_invalid_imports() {
    let backend = MemoryBackend::new();
    let store = open(&backend, &Scope::new("local").unwrap());
    let task_key = key("replacement");
    let original = pause(task_key.clone(), "first");
    store.save_pause(original).await.unwrap();
    let first = store.load(&task_key).await.unwrap().unwrap();
    let mut next = pause(task_key.clone(), "second");
    next.replaces = Some(first.clone());
    assert_eq!(
        store.save_pause(next).await.unwrap(),
        SubagentPausePersistenceDisposition::Replaced
    );
    let mut stale = pause(task_key.clone(), "stale");
    stale.replaces = Some(first.clone());
    assert_eq!(
        store.save_pause(stale).await.unwrap(),
        SubagentPausePersistenceDisposition::Existing
    );
    assert_eq!(
        store
            .record_terminal(
                &task_key,
                &SubagentOutcome::completed("task", "stale"),
                Some(&first)
            )
            .await
            .unwrap(),
        SubagentTerminalPersistenceDisposition::PauseExisting
    );
    assert_eq!(
        store.pause_outcome(&task_key).unwrap().unwrap().output,
        "second"
    );
    assert!(
        store
            .import_record(&key("other"), Some(pause(task_key.clone(), "bad")), None)
            .is_err()
    );
    assert!(
        store
            .import_record_at(
                "wrong-component",
                Some(pause(task_key.clone(), "bad")),
                None
            )
            .is_err()
    );
    let invalid = PersistedSubagentPause {
        key: task_key.clone(),
        outcome: SubagentOutcome::completed("task", "not a pause"),
        replaces: None,
    };
    assert!(store.save_pause(invalid.clone()).await.is_err());
    assert!(store.import_record(&task_key, Some(invalid), None).is_err());
    assert_eq!(
        store.pause_outcome(&task_key).unwrap().unwrap().output,
        "second"
    );
}
