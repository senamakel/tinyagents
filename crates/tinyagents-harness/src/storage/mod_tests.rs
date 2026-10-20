use super::*;
use serde_json::{Value, json};
use tinyagents_storage_kit::{
    Filter, MigrateOptions, Precondition, Registry, Scope, ScopedStorage, StorageBackend,
    WriteOptions,
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
        .put_with(
            collection,
            id,
            body,
            Precondition::None,
            WriteOptions::new().created_ms(1_700_000_000_000),
        )
        .await
        .unwrap();
}

async fn migrate(storage: &ScopedStorage) -> tinyagents_storage_kit::MigrationReport {
    let registry = Registry::compose([storage_schema()]).unwrap();
    // `prepare` needs a backend; the scoped handle's documents are enough here.
    storage
        .documents()
        .ensure_collection(&tinyagents_storage_kit::primitives::kv_collection_spec())
        .await
        .unwrap();
    for item in storage_schema().items {
        if let tinyagents_storage_kit::Item::Collection { spec, .. } = item {
            storage.documents().ensure_collection(&spec).await.unwrap();
        }
    }
    registry
        .migrate(storage, &MigrateOptions::new())
        .await
        .unwrap()
}

#[test]
fn the_schema_composes_and_names_everything_the_harness_owns() {
    let schema = storage_schema();
    assert_eq!(schema.owner.as_str(), "harness");
    let names: Vec<&str> = schema.items.iter().map(|item| item.name()).collect();
    assert_eq!(
        names,
        [
            "harness.run_status",
            "harness.embedding_cache",
            "harness.attachment_stash",
            "harness.journal",
            "harness/attachments/",
            "harness/tool_results/",
            "harness/outputs/",
            "harness/workspace/",
            "harness/generated_media/",
        ]
    );
    assert!(
        schema
            .items
            .iter()
            .all(|item| item.scope() == tinyagents_storage_kit::ScopeClass::Agent)
    );
    Registry::compose([schema]).unwrap();
}

#[tokio::test]
async fn run_status_moves_to_a_bare_snake_case_document_under_its_run_id() {
    let storage = scoped();
    put_v1(
        &storage,
        "harness_run_status",
        "8f3a-hash",
        json!({
            "runId": "run.abc", "rootRunId": "run.root", "threadId": "t-1", "active": true,
            "record": {
                "run_id": "run.abc", "parent_run_id": "run.root", "root_run_id": "run.root",
                "thread_id": "t-1", "component": "model", "status": "running",
                "current_phase": "model", "model_calls": 2, "tool_calls": 1,
                "active_tool_calls": [],
                "usage": {}, "cost": {},
                "started_at": 1_700_000_000_u64, "updated_at": 1_700_000_010_u64,
                "metadata": {"k": "v"}
            }
        }),
    )
    .await;
    // A record from an incompatible build is dropped, not fatal.
    put_v1(
        &storage,
        "harness_run_status",
        "bad",
        json!({"runId": "x", "record": {"nope": 1}}),
    )
    .await;

    let report = migrate(&storage).await;
    assert!(report.ran.iter().any(|step| step.step == "run_status_v2"));

    let row = storage
        .documents()
        .get("harness.run_status", "run.abc")
        .await
        .unwrap()
        .expect("the id is the run id");
    assert_eq!(row.doc["run_id"], "run.abc");
    assert_eq!(row.doc["root_run_id"], "run.root");
    assert_eq!(row.doc["parent_run_id"], "run.root");
    assert_eq!(row.doc["thread_id"], "t-1");
    assert_eq!(row.doc["active"], true);
    assert_eq!(row.doc["started_ms"], 1_700_000_000_000_u64);
    assert_eq!(row.doc["updated_ms"], 1_700_000_010_000_u64);
    assert!(row.doc.get("record").is_none());
    assert!(row.doc.get("runId").is_none());
    assert!(
        row.meta.created_ms > 0,
        "the host assigns creation time to a new cross-collection record"
    );
    assert_eq!(row.meta.schema, 2);
    let count = storage
        .documents()
        .count("harness.run_status", &Filter::default())
        .await
        .unwrap();
    assert_eq!(count, 1, "the undecodable v1 record is dropped");
    let left = storage
        .documents()
        .count("harness_run_status", &Filter::default())
        .await
        .unwrap();
    assert_eq!(left, 0, "the v1 rows are removed");
}

#[tokio::test]
async fn embedding_vectors_move_to_a_key_of_signature_and_hex() {
    let storage = scoped();
    put_v1(
        &storage,
        "harness_embedding_cache",
        "sha256ofparts",
        json!({"sig": "provider=bag;dims=3", "key": "00000000000000ab", "vector": [1.0, 2.0, 3.0]}),
    )
    .await;
    migrate(&storage).await;
    let row = storage
        .documents()
        .get(
            "harness.embedding_cache",
            "provider=bag;dims=3/00000000000000ab",
        )
        .await
        .unwrap()
        .expect("Key::of([sig, key])");
    assert_eq!(row.doc["vector"], json!([1.0, 2.0, 3.0]));
    assert_eq!(row.meta.schema, 2);
}

#[tokio::test]
async fn attachment_index_rows_gain_snake_case_fields_and_the_new_blob_key() {
    let storage = scoped();
    put_v1(
        &storage,
        "attachment_stash_index",
        "abc123",
        json!({"id": "abc123", "key": "attachments/abc123", "mime": "image/png", "len": 68,
               "touchedAtMs": 1_700_000_500_000_u64}),
    )
    .await;
    migrate(&storage).await;
    let row = storage
        .documents()
        .get("harness.attachment_stash", "abc123")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        row.doc,
        json!({"id": "abc123", "key": "harness/attachments/abc123", "mime": "image/png",
               "len": 68, "touched_ms": 1_700_000_500_000_u64})
    );
}

#[tokio::test]
async fn blobs_move_under_harness_with_their_bytes_and_content_type() {
    let storage = scoped();
    let blobs = storage.blobs();
    for (key, bytes) in [
        ("attachments/abc", &b"png"[..]),
        ("tool-results/s/shell/c.txt", b"out"),
        ("outputs/report.md", b"# r"),
        ("workspace/notes.md", b"n"),
        ("generated-media/clip.png", b"img"),
        ("elsewhere/keep.txt", b"keep"),
    ] {
        blobs
            .put(key, bytes.to_vec(), Some("application/test"))
            .await
            .unwrap();
    }
    migrate(&storage).await;
    for (key, bytes) in [
        ("harness/attachments/abc", &b"png"[..]),
        ("harness/tool_results/s/shell/c.txt", b"out"),
        ("harness/outputs/report.md", b"# r"),
        ("harness/workspace/notes.md", b"n"),
        ("harness/generated_media/clip.png", b"img"),
    ] {
        let blob = blobs
            .get(key)
            .await
            .unwrap()
            .unwrap_or_else(|| panic!("{key}"));
        assert_eq!(blob.bytes, bytes, "{key}");
        assert_eq!(blob.meta.content_type.as_deref(), Some("application/test"));
    }
    for old in [
        "attachments/abc",
        "tool-results/s/shell/c.txt",
        "outputs/report.md",
        "workspace/notes.md",
        "generated-media/clip.png",
    ] {
        assert!(blobs.head(old).await.unwrap().is_none(), "{old} is moved");
    }
    assert!(blobs.head("elsewhere/keep.txt").await.unwrap().is_some());
}

#[tokio::test]
async fn migrating_twice_changes_nothing() {
    let storage = scoped();
    put_v1(
        &storage,
        "attachment_stash_index",
        "a",
        json!({"id": "a", "key": "attachments/a", "mime": "image/png", "len": 1, "touchedAtMs": 5}),
    )
    .await;
    migrate(&storage).await;
    let again = migrate(&storage).await;
    assert!(again.ran.is_empty());
    assert_eq!(again.already_done, storage_schema().steps.len());
}

#[test]
fn the_journal_stream_seals_on_finish_and_is_kept_for_thirty_days() {
    use tinyagents_storage_kit::{Item, SealPolicy, SealWhen};
    let item = storage_schema()
        .items
        .into_iter()
        .find(|item| item.name() == "harness.journal")
        .expect("the journal family is declared");
    let Item::Stream { seal, scope, .. } = item else {
        panic!("harness.journal is a stream");
    };
    assert_eq!(scope, tinyagents_storage_kit::ScopeClass::Agent);
    assert_eq!(
        seal,
        SealPolicy::new(SealWhen::OnFinish).retaining_ms(30 * 24 * 60 * 60 * 1000)
    );
}

mod journal_step {
    use super::*;
    use crate::events::{AgentEvent, EventSink};
    use crate::ids::RunId;
    use crate::observability::{AgentObservation, HarnessEventJournal, StoreEventJournal};
    use crate::store::{AppendStore, DriverAppendStore};
    use std::sync::Arc;

    const V1: &str = "16:session_journal/";

    /// The observations of `run` as the v1 journal appended them: the whole
    /// wire type, in a length-prefixed stream.
    async fn seed_v1(
        storage: &ScopedStorage,
        run: &str,
        count: u64,
        root: &str,
    ) -> Vec<AgentObservation> {
        let sink = EventSink::with_stream_id(if root == run { run } else { root });
        let mut written = Vec::new();
        for i in 0..count {
            let record = sink.emit(AgentEvent::ToolStarted {
                parent_call_id: None,
                call_id: format!("c{i}").into(),
                tool_name: format!("tool-{i}"),
                input: None,
            });
            let obs = AgentObservation::from_record(
                &record,
                RunId::new(run),
                (root != run).then(|| RunId::new(root)),
                RunId::new(root),
            );
            storage
                .streams()
                .append(&format!("{V1}{run}"), serde_json::to_value(&obs).unwrap())
                .await
                .unwrap();
            written.push(obs);
        }
        written
    }

    fn journal(storage: &ScopedStorage) -> StoreEventJournal<Arc<dyn AppendStore>> {
        StoreEventJournal::new(
            Arc::new(DriverAppendStore::journal(Arc::clone(storage.streams())))
                as Arc<dyn AppendStore>,
        )
    }

    #[tokio::test]
    async fn a_v1_journal_moves_with_identical_offsets_and_event_ids() {
        let storage = scoped();
        let top = seed_v1(&storage, "run.top", 5, "run.top").await;
        let child = seed_v1(&storage, "run.child", 3, "run.top").await;
        // The sink counter can run ahead of the stream position.
        storage
            .streams()
            .append(
                &format!("{V1}run.gap"),
                serde_json::to_value(AgentObservation {
                    event_id: "run.gap-evt-40".into(),
                    run_id: RunId::new("run.gap"),
                    parent_run_id: None,
                    root_run_id: RunId::new("run.gap"),
                    offset: 40,
                    ts_ms: 9,
                    event: AgentEvent::StreamClosed,
                })
                .unwrap(),
            )
            .await
            .unwrap();

        let report = migrate(&storage).await;
        assert!(report.ran.iter().any(|step| step.step == "run_journal_v2"));

        let journal = journal(&storage);
        assert_eq!(journal.read_from("run.top", 0).await.unwrap(), top);
        assert_eq!(journal.read_from("run.child", 0).await.unwrap(), child);
        let gap = journal.read_from("run.gap", 0).await.unwrap();
        assert_eq!(
            (gap[0].offset, gap[0].event_id.as_str()),
            (40, "run.gap-evt-40")
        );
        for (position, obs) in journal
            .read_from("run.top", 0)
            .await
            .unwrap()
            .iter()
            .enumerate()
        {
            assert_eq!(obs.offset, position as u64, "offsets are unchanged");
            assert_eq!(obs.event_id.as_str(), format!("run.top-evt-{position}"));
        }

        let raw = storage
            .streams()
            .read_window("harness.journal/run.top", 0, 10)
            .await
            .unwrap();
        for entry in raw {
            for repeated in ["run_id", "root_run_id", "event_id"] {
                assert!(entry.value.get(repeated).is_none(), "{repeated} repeated");
            }
        }
        assert!(
            storage.streams().streams(V1).await.unwrap().is_empty(),
            "v1 streams are gone"
        );
    }

    #[tokio::test]
    async fn a_sealed_v1_journal_stays_sealed_and_a_second_run_changes_nothing() {
        let storage = scoped();
        let written = seed_v1(&storage, "run.done", 4, "run.done").await;
        storage
            .streams()
            .seal(&format!("{V1}run.done"))
            .await
            .unwrap();

        migrate(&storage).await;
        let info = storage
            .streams()
            .info("harness.journal/run.done")
            .await
            .unwrap()
            .unwrap();
        assert!(info.sealed_ms.is_some());
        assert_eq!(
            journal(&storage).read_from("run.done", 0).await.unwrap(),
            written
        );

        let again = migrate(&storage).await;
        assert!(again.ran.iter().all(|step| step.step != "run_journal_v2"));
        assert_eq!(
            journal(&storage).read_from("run.done", 0).await.unwrap(),
            written
        );
    }

    #[tokio::test]
    async fn an_entry_that_is_not_an_observation_keeps_its_place() {
        let storage = scoped();
        let good = seed_v1(&storage, "run.odd", 1, "run.odd").await;
        storage
            .streams()
            .append(
                &format!("{V1}run.odd"),
                serde_json::json!({"note": "from another build"}),
            )
            .await
            .unwrap();
        migrate(&storage).await;
        let raw = storage
            .streams()
            .read_window("harness.journal/run.odd", 0, 10)
            .await
            .unwrap();
        assert_eq!(raw.len(), 2, "no entry is dropped, so no offset moves");
        assert_eq!(
            raw[1].value,
            serde_json::json!({"note": "from another build"})
        );
        assert_eq!(
            journal(&storage)
                .read_window("run.odd", 0, 1)
                .await
                .unwrap(),
            good
        );
    }
}
