use std::time::Duration;

use super::*;
use crate::storage::subgraph_key;
use crate::testkit::conformance::{
    checkpointer_concurrent_contract, checkpointer_contract, checkpointer_lineage_contract,
    checkpointer_writes_contract,
};
use serde::Serialize;
use serde_json::json;
use tinyagents_storage_kit::{Precondition, Scope, StorageBackend};
use tinystoragedrivers_testkit::MemoryBackend;

fn docs(storage: &MemoryBackend, scope: &str) -> Arc<dyn DocumentStore> {
    Arc::clone(
        storage
            .for_scope(&Scope::new(scope).unwrap())
            .unwrap()
            .documents(),
    )
}

fn sample(thread: &str, id: &str, _parent: Option<&str>, step: i32) -> Checkpoint<i32> {
    Checkpoint::new(step, Vec::new())
        .with_thread_id(thread.to_string())
        .with_checkpoint_id(id.to_string())
}

fn checkpointer() -> DriverCheckpointer<i32> {
    DriverCheckpointer::new(docs(&MemoryBackend::new(), "local"))
}

#[tokio::test]
async fn passes_the_checkpointer_contract() {
    checkpointer_contract(checkpointer()).await;
}

#[tokio::test]
async fn passes_the_writes_contract() {
    checkpointer_writes_contract(checkpointer()).await;
}

#[tokio::test]
async fn passes_the_lineage_contract() {
    checkpointer_lineage_contract(checkpointer()).await;
}

#[tokio::test]
async fn passes_the_concurrent_contract() {
    checkpointer_concurrent_contract(Arc::new(checkpointer())).await;
}

#[tokio::test]
async fn scopes_and_namespaces_keep_threads_apart() {
    let storage = MemoryBackend::new();
    let alice = DriverCheckpointer::<i32>::new(docs(&storage, "alice"));
    let bob = DriverCheckpointer::<i32>::new(docs(&storage, "bob"));
    let other = DriverCheckpointer::<i32>::with_namespace(docs(&storage, "alice"), "other");
    let checkpoint = sample("t", "c1", None, 1);
    alice.put(checkpoint).await.unwrap();
    assert!(bob.get("t", None).await.unwrap().is_none());
    assert!(other.get("t", None).await.unwrap().is_none());
    assert_eq!(alice.list_threads().await.unwrap(), vec!["t".to_owned()]);
    assert!(bob.list_threads().await.unwrap().is_empty());
    assert!(format!("{alice:?}").contains("graph"));
    let _ = alice.clone();
}

#[tokio::test]
async fn leases_follow_the_claim_protocol() {
    let cp = checkpointer();
    let minute = Duration::from_secs(60);
    assert!(cp.try_claim("t", "a", minute).await.unwrap());
    assert!(
        !cp.try_claim("t", "b", minute).await.unwrap(),
        "live lease is refused"
    );
    assert!(
        cp.try_claim("t", "a", minute).await.unwrap(),
        "same owner re-claims"
    );
    assert!(cp.renew("t", "a", minute).await.unwrap());
    assert!(
        !cp.renew("t", "b", minute).await.unwrap(),
        "only the owner renews"
    );
    assert!(!cp.renew("missing", "a", minute).await.unwrap());

    cp.release("t", "b").await.unwrap();
    assert!(
        !cp.try_claim("t", "b", minute).await.unwrap(),
        "foreign release is a no-op"
    );
    cp.release("t", "a").await.unwrap();
    cp.release("t", "a").await.unwrap();
    assert!(
        cp.try_claim("t", "b", minute).await.unwrap(),
        "released lease is free"
    );

    assert!(cp.try_claim("z", "dead", Duration::ZERO).await.unwrap());
    tokio::time::sleep(Duration::from_millis(5)).await;
    assert!(
        !cp.renew("z", "dead", minute).await.unwrap(),
        "expired lease cannot renew"
    );
    assert!(
        cp.try_claim("z", "new", minute).await.unwrap(),
        "expired lease is reclaimable"
    );
}

#[test]
fn namespaces_encode_injectively() {
    let ns = |parts: &[&str]| parts.iter().map(|p| (*p).to_string()).collect::<Vec<_>>();
    assert_ne!(
        subgraph_key(&ns(&["a", "b"])),
        subgraph_key(&ns(&["a\u{1f}b"]))
    );
    assert_ne!(subgraph_key(&ns(&["a", "b"])), subgraph_key(&ns(&["a/b"])));
    assert_ne!(subgraph_key(&ns(&["ab"])), subgraph_key(&ns(&["a", "b"])));
    assert_eq!(subgraph_key(&[]), "");
    assert_ne!(
        subgraph_key(&ns(&[""])),
        subgraph_key(&[]),
        "the empty segment is not the root"
    );
}

#[tokio::test]
async fn scoped_reads_stay_in_their_namespace_when_ids_repeat() {
    let saver = checkpointer();
    let child = vec!["sub".to_string()];
    saver.put(sample("t", "same", None, 1)).await.unwrap();
    saver
        .put(sample("t", "same", None, 2).with_namespace(child.clone()))
        .await
        .unwrap();
    let root = saver
        .get_scoped("t", Some("same"), &[])
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        root.state, 1,
        "the root never loads the subgraph's checkpoint"
    );
    let nested = saver
        .get_scoped("t", Some("same"), &child)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(nested.state, 2);
    assert_eq!(
        saver
            .get_scoped("t", None, &[])
            .await
            .unwrap()
            .unwrap()
            .state,
        1
    );
    assert!(
        saver
            .get_scoped("t", None, &["other".to_string()])
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn pending_writes_never_merge_across_lookalike_namespaces() {
    let saver = checkpointer();
    let config = |namespace: Vec<String>| CheckpointConfig {
        thread_id: "t".to_string(),
        checkpoint_id: Some("c".to_string()),
        namespace,
    };
    let split = config(vec!["a".to_string(), "b".to_string()]);
    let joined = config(vec!["a\u{1f}b".to_string()]);
    saver
        .put_writes(
            &split,
            &[PendingWrite::data("n", "task", 0, "out", json!("split"))],
        )
        .await
        .unwrap();
    assert!(saver.get_writes(&joined).await.unwrap().is_empty());
    assert_eq!(saver.get_writes(&split).await.unwrap().len(), 1);
}

#[test]
fn driver_failures_are_checkpoint_errors() {
    let error = map_error(StorageError::unavailable("busy"));
    assert!(matches!(error, TinyAgentsError::Checkpoint(ref m) if m.contains("busy")));
}

#[tokio::test]
async fn a_corrupt_record_is_a_checkpoint_error() {
    let storage = MemoryBackend::new();
    let docs = docs(&storage, "local");
    let cp = DriverCheckpointer::<i32>::new(Arc::clone(&docs));
    cp.put(sample("t", "c1", None, 1)).await.unwrap();
    let stored = cp.thread_docs("t").await.unwrap().remove(0);
    docs.put(
        "graph.checkpoints",
        &stored.id,
        json!({"ns": "graph", "thread": "t", "subgraph": "", "seq": 0, "checkpoint_id": "c1", "checkpoint": "nope"}),
        Precondition::None,
    )
    .await
    .unwrap();
    let error = cp.get("t", None).await.unwrap_err();
    assert!(matches!(error, TinyAgentsError::Checkpoint(_)), "{error:?}");
}

/// A `State` whose `Deserialize` counts every call, so a test can assert how
/// many full states a read decoded rather than only what it returned.
#[derive(Debug, Clone, Serialize)]
struct CountedState(i32);

static STATE_DECODES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl<'de> serde::Deserialize<'de> for CountedState {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        STATE_DECODES.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        i32::deserialize(deserializer).map(CountedState)
    }
}

#[tokio::test]
async fn list_and_limited_history_decode_only_what_they_return() {
    use std::sync::atomic::Ordering;
    let cp = DriverCheckpointer::<CountedState>::new(docs(&MemoryBackend::new(), "local"));
    let mut parent: Option<String> = None;
    for step in 0..40 {
        let id = format!("c{step}");
        let checkpoint = Checkpoint::new(CountedState(step), Vec::new())
            .with_thread_id("t".to_string())
            .with_checkpoint_id(id.clone())
            .with_parent_checkpoint_id(parent.clone())
            .with_metadata(json!({ "source": "loop", "step": step }));
        cp.put(checkpoint).await.unwrap();
        parent = Some(id);
    }

    STATE_DECODES.store(0, Ordering::SeqCst);
    let listed = cp.list("t").await.unwrap();
    assert_eq!(listed.len(), 40);
    assert_eq!(listed[0].checkpoint_id, "c0");
    assert_eq!(listed[39].checkpoint_id, "c39");
    assert_eq!(
        STATE_DECODES.load(Ordering::SeqCst),
        0,
        "listing must not deserialize any record's full State"
    );

    let recent = cp.state_history("t", &[], Some(1)).await.unwrap();
    assert_eq!(recent.len(), 1);
    assert_eq!(recent[0].checkpoint.checkpoint_id, "c39");
    assert_eq!(
        STATE_DECODES.load(Ordering::SeqCst),
        1,
        "a one-record history decodes one record, not the whole chain"
    );
}

#[tokio::test]
async fn put_with_writes_persists_the_checkpoint_and_its_writes() {
    let cp = checkpointer();
    let writes = vec![PendingWrite {
        node: tinyagents_harness::ids::NodeId::from("n"),
        task_id: tinyagents_harness::ids::TaskId::from("task-1"),
        idx: 0,
        channel: "out".to_string(),
        payload: json!("hi"),
    }];
    let id = cp
        .put_with_writes(sample("t", "c1", None, 1), &writes)
        .await
        .unwrap();
    assert_eq!(id.as_str(), "c1");
    assert!(cp.get("t", Some("c1")).await.unwrap().is_some());
    let config = CheckpointConfig {
        thread_id: "t".to_string(),
        checkpoint_id: Some("c1".to_string()),
        namespace: vec![],
    };
    let stored = cp.get_writes(&config).await.unwrap();
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].channel, "out");
}

#[tokio::test]
async fn documents_are_v2_bodies_with_the_namespace_in_ids_and_fields() {
    let storage = MemoryBackend::new();
    let docs = docs(&storage, "local");
    let delegation = DriverCheckpointer::<i32>::with_namespace(Arc::clone(&docs), "delegation");
    let flows = DriverCheckpointer::<i32>::with_namespace(Arc::clone(&docs), "flows");
    delegation.put(sample("a/b", "c1", None, 1)).await.unwrap();
    flows.put(sample("a/b", "c1", None, 2)).await.unwrap();

    let row = docs
        .get("graph.checkpoints", &format!("delegation/a%2Fb/{:020}", 0))
        .await
        .unwrap()
        .expect("id is Key::of([ns, thread, seq])");
    assert_eq!(row.doc["ns"], "delegation");
    assert_eq!(row.doc["thread"], "a/b");
    assert_eq!(row.doc["subgraph"], "");
    assert_eq!(row.doc["seq"], 0);
    assert_eq!(row.doc["checkpoint_id"], "c1");
    assert!(
        row.doc["checkpoint"].is_object(),
        "typed, not a JSON string"
    );
    assert_eq!(row.meta.schema, 4);
    let counter = docs
        .get("graph.threads", "flows/a%2Fb")
        .await
        .unwrap()
        .expect("id is Key::of([ns, thread])");
    assert_eq!(
        (counter.doc["ns"].as_str(), counter.doc["next"].as_u64()),
        (Some("flows"), Some(1))
    );
    assert_eq!(delegation.get("a/b", None).await.unwrap().unwrap().state, 1);
    assert_eq!(flows.get("a/b", None).await.unwrap().unwrap().state, 2);
    assert_eq!(delegation.list_threads().await.unwrap(), ["a/b"]);
}

#[tokio::test]
async fn the_lease_lives_in_tsd_leases_and_its_epoch_moves_on_every_new_tenure() {
    let storage = MemoryBackend::new();
    let docs = docs(&storage, "local");
    let cp = DriverCheckpointer::<i32>::with_namespace(Arc::clone(&docs), "flows");
    let minute = Duration::from_secs(60);
    let (owner, key) = cp.lease_key("t");
    let leases = tinyagents_storage_kit::Leases::new(&docs);

    assert!(cp.try_claim("t", "a", minute).await.unwrap());
    let first = leases.current(owner, &key).await.unwrap().unwrap();
    assert_eq!((first.holder.as_str(), first.epoch), ("a", 1));
    assert!(cp.try_claim("t", "a", minute).await.unwrap());
    assert_eq!(
        leases.current(owner, &key).await.unwrap().unwrap().epoch,
        1,
        "re-claim keeps the epoch"
    );

    cp.release("t", "a").await.unwrap();
    assert!(leases.current(owner, &key).await.unwrap().is_none());
    assert!(cp.try_claim("t", "a", minute).await.unwrap());
    let second = leases.current(owner, &key).await.unwrap().unwrap();
    assert_eq!(
        second.epoch, 2,
        "a new tenure of the same holder is a new epoch"
    );
    assert!(
        !cp.try_claim("t", "b", minute).await.unwrap(),
        "namespaces share no lease, but a thread's claim is exclusive"
    );
    let other = DriverCheckpointer::<i32>::with_namespace(Arc::clone(&docs), "delegation");
    assert!(
        other.try_claim("t", "b", minute).await.unwrap(),
        "another namespace's thread is free"
    );
}
