//! Tests for the store-backed run journal: its stored shape, sealing and the
//! replay pages a sealed run serves.

use std::sync::Arc;

use super::*;
use crate::events::{AgentEvent, EventSink};
use crate::ids::RunId;
use crate::observability::replay::read_run_events_page;
use crate::store::{AppendStore, DriverAppendStore};
use serde_json::json;
use tinyagents_storage_kit::{Scope, ScopedStorage, StorageBackend};
use tinystoragedrivers_testkit::MemoryBackend;

fn storage() -> ScopedStorage {
    MemoryBackend::new()
        .for_scope(&Scope::new("local").unwrap())
        .unwrap()
}

fn journal_over(storage: &ScopedStorage) -> StoreEventJournal<Arc<dyn AppendStore>> {
    StoreEventJournal::new(
        Arc::new(DriverAppendStore::journal(Arc::clone(storage.streams()))) as Arc<dyn AppendStore>,
    )
}

/// Appends `count` observations numbered by a sink streamed under `run`.
async fn seed(journal: &dyn HarnessEventJournal, run: &str, count: usize) {
    let sink = EventSink::with_stream_id(run);
    for i in 0..count {
        let record = sink.emit(AgentEvent::ToolStarted {
            parent_call_id: None,
            call_id: format!("c{i}").into(),
            tool_name: format!("tool-{i}"),
            input: None,
        });
        let obs = AgentObservation::from_record(&record, RunId::new(run), None, RunId::new(run));
        journal.append(obs).await.unwrap();
    }
}

#[tokio::test]
async fn the_stream_is_the_run_and_its_entries_do_not_repeat_the_run_ids() {
    let storage = storage();
    let journal = journal_over(&storage);
    seed(&journal, "run.abc", 2).await;

    let raw = storage
        .streams()
        .read_window("harness.journal/run.abc", 0, 10)
        .await
        .unwrap();
    assert_eq!(raw.len(), 2, "the stream is named by the run");
    for entry in &raw {
        for repeated in ["run_id", "root_run_id", "event_id"] {
            assert!(entry.value.get(repeated).is_none(), "{repeated} repeated");
        }
        assert!(entry.value["ts_ms"].is_u64());
        assert_eq!(entry.value["event"]["kind"], "tool_started");
    }

    let read = journal.read_from("run.abc", 0).await.unwrap();
    assert_eq!(read[1].run_id.as_str(), "run.abc");
    assert_eq!(read[1].root_run_id.as_str(), "run.abc");
    assert_eq!(read[1].event_id.as_str(), "run.abc-evt-1");
    assert_eq!(read[1].offset, 1);
}

#[tokio::test]
async fn sealing_a_run_leaves_every_replay_page_unchanged() {
    let storage = storage();
    let journal = journal_over(&storage);
    seed(&journal, "run.sealed", 7).await;

    let mut before = Vec::new();
    let mut cursor = 0;
    loop {
        let page = read_run_events_page(&journal, "run.sealed", cursor, 3)
            .await
            .unwrap();
        before.push((page.events, page.next_offset));
        match before.last().unwrap().1 {
            Some(next) => cursor = next,
            None => break,
        }
    }
    assert_eq!(before.len(), 3);

    journal.seal_run("run.sealed").await.unwrap();
    journal.seal_run("run.sealed").await.unwrap();

    let info = storage
        .streams()
        .info("harness.journal/run.sealed")
        .await
        .unwrap()
        .unwrap();
    assert!(info.sealed_ms.is_some());
    let mut after = Vec::new();
    let mut cursor = 0;
    loop {
        let page = read_run_events_page(&journal, "run.sealed", cursor, 3)
            .await
            .unwrap();
        after.push((page.events, page.next_offset));
        match after.last().unwrap().1 {
            Some(next) => cursor = next,
            None => break,
        }
    }
    assert_eq!(after, before);
    assert_eq!(
        journal
            .read_filtered("run.sealed", 0, &["tool.started"])
            .await
            .unwrap()
            .len(),
        7
    );
}

#[tokio::test]
async fn a_sealed_run_refuses_further_observations() {
    let storage = storage();
    let journal = journal_over(&storage);
    seed(&journal, "run.done", 1).await;
    journal.seal_run("run.done").await.unwrap();
    let late = AgentObservation {
        event_id: "run.done-evt-9".into(),
        run_id: RunId::new("run.done"),
        parent_run_id: None,
        root_run_id: RunId::new("run.done"),
        offset: 9,
        ts_ms: 1,
        event: AgentEvent::StreamClosed,
    };
    assert!(journal.append(late).await.is_err());
}

#[tokio::test]
async fn sealing_is_a_no_op_for_journals_that_do_not_seal() {
    let memory = InMemoryEventJournal::new();
    seed(&memory, "run.m", 1).await;
    memory.seal_run("run.m").await.unwrap();
    memory.seal_run("never-written").await.unwrap();
    assert_eq!(memory.read_from("run.m", 0).await.unwrap().len(), 1);

    let in_memory_store = StoreEventJournal::new(crate::store::InMemoryAppendStore::new());
    seed(&in_memory_store, "run.s", 1).await;
    in_memory_store.seal_run("run.s").await.unwrap();
    assert_eq!(
        in_memory_store.read_from("run.s", 0).await.unwrap().len(),
        1
    );
}

#[tokio::test]
async fn a_v1_wire_entry_in_the_stream_still_reads() {
    let storage = storage();
    let journal = journal_over(&storage);
    storage
        .streams()
        .append(
            "harness.journal/run.old",
            json!({
                "event_id": "run.old-evt-0", "run_id": "run.old", "root_run_id": "run.old",
                "offset": 0, "ts_ms": 5, "event": {"kind": "stream_closed"}
            }),
        )
        .await
        .unwrap();
    let read = journal.read_from("run.old", 0).await.unwrap();
    assert_eq!(read[0].event_id.as_str(), "run.old-evt-0");
    assert_eq!(read[0].run_id.as_str(), "run.old");
}
