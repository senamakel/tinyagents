//! Regression tests for the driver-backed checkpointer and append store: the
//! identity questions the old file backends once got wrong (thread ids that
//! differ only by case, the empty thread id, offsets handed out by several
//! handles on one store) must stay right on every backend.

mod common;

use std::sync::Arc;

use common::storage;
use tinyagents_graph::checkpoint::{Checkpoint, Checkpointer, PendingActivation};
use tinyagents_harness::ids::NodeId;
use tinyagents_harness::store::{AppendStore, DriverAppendStore};

fn checkpoint(thread: &str, id: &str) -> Checkpoint<i32> {
    Checkpoint::new(
        1,
        vec![PendingActivation {
            node: NodeId::from("n"),
            send_arg: None,
            task_id: tinyagents_harness::ids::TaskId::from(String::new()),
        }],
    )
    .with_thread_id(thread.to_string())
    .with_checkpoint_id(id.to_string())
    .with_metadata(serde_json::json!({ "source": "loop", "step": 1 }))
}

/// Thread ids that differ only by case must not share state: on a
/// case-insensitive filesystem `Alice` and `alice` were the same file, which
/// silently merged two unrelated lineages.
#[tokio::test]
async fn thread_ids_differing_only_by_case_do_not_share_state() {
    let dir = tempfile::tempdir().unwrap();
    let cp = storage::checkpointer::<i32>(dir.path());
    cp.put(checkpoint("Alice", "upper")).await.unwrap();
    cp.put(checkpoint("alice", "lower")).await.unwrap();

    let upper = cp.list("Alice").await.unwrap();
    let lower = cp.list("alice").await.unwrap();
    assert_eq!(upper.len(), 1, "`Alice` holds only its own checkpoint");
    assert_eq!(lower.len(), 1, "`alice` holds only its own checkpoint");
    assert_eq!(upper[0].checkpoint_id, "upper");
    assert_eq!(lower[0].checkpoint_id, "lower");
}

/// The empty thread id is a valid thread and must be visible to listing.
#[tokio::test]
async fn the_empty_thread_id_is_visible_to_list_threads() {
    let dir = tempfile::tempdir().unwrap();
    let cp = storage::checkpointer::<i32>(dir.path());
    cp.put(checkpoint("", "c1")).await.unwrap();
    let threads = cp.list_threads().await.unwrap();
    assert!(
        threads.iter().any(|t| t.is_empty()),
        "list_threads reports the empty thread id: {threads:?}"
    );
}

/// Separators, dots and non-ASCII are ordinary characters in a thread id.
#[tokio::test]
async fn thread_ids_with_separators_and_unicode_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let cp = storage::checkpointer::<i32>(dir.path());
    let ids = ["a/b", "a\\b", "..", "con", "naïve ☃", "with space", "x.y.z"];
    for (n, id) in ids.iter().enumerate() {
        cp.put(checkpoint(id, &format!("c{n}"))).await.unwrap();
    }
    let mut listed = cp.list_threads().await.unwrap();
    listed.sort();
    let mut expected: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
    expected.sort();
    assert_eq!(listed, expected);
    for (n, id) in ids.iter().enumerate() {
        let checkpoints = cp.list(id).await.unwrap();
        assert_eq!(checkpoints.len(), 1, "{id}");
        assert_eq!(checkpoints[0].checkpoint_id, format!("c{n}"));
    }
}

/// One undecodable record must not break listing for every other thread.
#[tokio::test]
async fn one_undecodable_record_does_not_break_other_threads() {
    use tinystoragedrivers_api::DocumentStore;
    use tinystoragedrivers_types::Precondition;
    let dir = tempfile::tempdir().unwrap();
    let scoped = storage::sqlite(dir.path(), "checkpoints.sqlite");
    let docs = storage::docs(&scoped);
    let cp = tinyagents_graph::checkpoint::DriverCheckpointer::<i32>::new(docs.clone());
    cp.put(checkpoint("good", "c1")).await.unwrap();
    docs.put(
        "graph_checkpoints",
        "poison",
        serde_json::json!({"thread": "bad", "namespace": "", "seq": 0, "checkpoint_id": "x", "record": {"nope": true}}),
        Precondition::None,
    )
    .await
    .unwrap();
    let threads = cp.list_threads().await.expect("listing survives");
    assert!(threads.iter().any(|t| t == "good"));
    assert_eq!(cp.list("good").await.unwrap().len(), 1);
}

/// Two store handles over one database must keep handing out fresh offsets:
/// a per-instance counter would re-issue offsets the first had already used.
#[tokio::test]
async fn offsets_stay_unique_across_store_instances() {
    let dir = tempfile::tempdir().unwrap();
    let streams = || Arc::clone(storage::sqlite(dir.path(), "journal.sqlite").streams());
    let first = DriverAppendStore::new(streams());
    let second = DriverAppendStore::new(streams());

    let mut offsets = Vec::new();
    offsets.push(first.append("s", serde_json::json!(0)).await.unwrap());
    offsets.push(second.append("s", serde_json::json!(1)).await.unwrap());
    offsets.push(first.append("s", serde_json::json!(2)).await.unwrap());
    offsets.push(second.append("s", serde_json::json!(3)).await.unwrap());

    assert_eq!(
        offsets,
        vec![0, 1, 2, 3],
        "a second instance continues the stream instead of restarting it"
    );
    assert_eq!(second.len("s").await.unwrap(), 4);
}

/// `read_from` resolves by offset, not by position.
#[tokio::test]
async fn read_from_resolves_by_offset_not_position() {
    let store = DriverAppendStore::new(Arc::clone(storage::memory().streams()));
    for n in 0..5 {
        store.append("s", serde_json::json!(n)).await.unwrap();
    }
    let tail = store.read_from("s", 3).await.unwrap();
    assert_eq!(
        tail,
        vec![(3, serde_json::json!(3)), (4, serde_json::json!(4))]
    );
    assert!(store.read_from("s", 5).await.unwrap().is_empty());
}
