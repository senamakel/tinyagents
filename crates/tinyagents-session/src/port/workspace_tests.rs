use super::*;
use crate::legacy::OnceOutcome;
use crate::transcript::{
    SessionRef, TranscriptMessage, TranscriptMeta, TranscriptTurn, session_stem, write_transcript,
};

fn meta() -> TranscriptMeta {
    TranscriptMeta {
        session_id: None,
        parent_session_id: None,
        agent_name: "agent".into(),
        agent_id: Some("agent".into()),
        agent_type: Some("root".into()),
        dispatcher: "native".into(),
        provider: None,
        model: None,
        created: "2026-01-01T00:00:00Z".into(),
        updated: "2026-01-01T00:00:00Z".into(),
        turn_count: 1,
        prefix_message_count: None,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: Some("thread-ws".into()),
        task_id: None,
    }
}

#[test]
fn legacy_generations_resume_through_the_port_and_a_new_generation_chains() {
    let dir = tempfile::tempdir().unwrap();
    let raw = dir.path().join("session_raw");
    std::fs::create_dir_all(&raw).unwrap();
    let session = SessionRef::root("thread-ws");
    let g1 = session.next_generation();
    let sealed = raw.join(format!("{}.jsonl", session_stem(&session)));
    let head = raw.join(format!("{}.jsonl", session_stem(&g1)));
    let rows = |t: &str| {
        vec![
            TranscriptMessage::user(t),
            TranscriptMessage::assistant("ok"),
        ]
    };
    write_transcript(&sealed, &rows("first"), &meta(), None).unwrap();
    write_transcript(&head, &rows("second"), &meta(), None).unwrap();
    let (sealed_bytes, head_bytes) = (
        std::fs::read(&sealed).unwrap(),
        std::fs::read(&head).unwrap(),
    );

    let opened = WorkspaceStores::open(dir.path(), "agent").unwrap();
    let OnceOutcome::Imported { report, marked } = opened.import_outcome() else {
        panic!("first open imports");
    };
    assert_eq!(report.imported, 2);
    assert!(marked);

    let locator = opened.transcripts();
    assert_eq!(
        locator.session_chain(&session),
        vec![session.clone(), g1.clone()]
    );
    assert_eq!(locator.head_generation(&session), g1);
    let resumed = locator
        .read_session_transcript(&g1)
        .unwrap()
        .read_session()
        .unwrap()
        .unwrap();
    assert_eq!(resumed.messages[0].content, "second");

    // Compaction never erases: a new generation chains, both predecessors stay.
    let (g2, handle) = locator.begin_generation(&g1, meta()).unwrap();
    let next = rows("summary");
    handle
        .append_turn(TranscriptTurn {
            prev: &[],
            next: &next,
            meta: &meta(),
            turn_usage: None,
            request_id: None,
            tools: None,
        })
        .unwrap();
    assert_eq!(locator.session_chain(&session).len(), 3);
    assert_eq!(locator.head_generation(&session), g2);
    assert!(locator.session_exists(&session));

    // Legacy files were only read.
    assert_eq!(std::fs::read(&sealed).unwrap(), sealed_bytes);
    assert_eq!(std::fs::read(&head).unwrap(), head_bytes);
    assert!(dir.path().join(STORE_DIR).is_dir());

    // The handle is cached per workspace and agent.
    let again = WorkspaceStores::open(dir.path(), "agent").unwrap();
    assert!(Arc::ptr_eq(again.provider(), opened.provider()));
}

#[test]
fn a_workspace_with_no_legacy_files_opens_empty() {
    let dir = tempfile::tempdir().unwrap();
    #[allow(deprecated)]
    let opened = WorkspaceStores::open(dir.path(), "agent").unwrap();
    assert!(opened.transcripts().latest_for_agent("agent").is_none());
}

#[test]
fn supplied_backend_is_shared_across_handles_and_scoped_by_agent() {
    use tinystoragedrivers_core::{MemoryStorage, StorageBackend};

    let backend: Arc<dyn StorageBackend> = Arc::new(MemoryStorage::new());
    let first = WorkspaceStores::from_backend(Arc::clone(&backend), "agent").unwrap();
    let session = SessionRef::root("shared-backend-thread");
    let handle = first.transcripts().open_session(&session, meta()).unwrap();
    let messages = vec![
        TranscriptMessage::user("persisted on the host backend"),
        TranscriptMessage::assistant("resumed by another handle"),
    ];
    handle
        .append_turn(TranscriptTurn {
            prev: &[],
            next: &messages,
            meta: &meta(),
            turn_usage: None,
            request_id: None,
            tools: None,
        })
        .unwrap();

    let reopened = WorkspaceStores::from_backend(Arc::clone(&backend), "agent").unwrap();
    assert_eq!(
        reopened
            .transcripts()
            .read_session_transcript(&session)
            .unwrap()
            .read_session()
            .unwrap()
            .unwrap()
            .messages[0]
            .content,
        "persisted on the host backend"
    );
    let other_agent = WorkspaceStores::from_backend(backend, "other-agent").unwrap();
    assert!(!other_agent.transcripts().session_exists(&session));
}

#[test]
fn supplied_sqlite_backend_does_not_create_a_private_workspace_store() {
    use tinystoragedrivers_core::StorageBackend;

    let dir = tempfile::tempdir().unwrap();
    let storage_dir = dir.path().join(STORAGE_DIR);
    let backend: Arc<dyn StorageBackend> =
        Arc::new(SqliteStorage::open(&storage_dir).unwrap());
    let stores = WorkspaceStores::from_backend(backend, "agent").unwrap();
    let session = SessionRef::root("supplied-backend-thread");
    stores
        .transcripts()
        .open_session(&session, meta())
        .unwrap();

    assert!(storage_dir.join("storage.db").is_file());
    assert!(!dir.path().join(STORE_DIR).exists());
}

#[allow(deprecated)]
#[tokio::test]
async fn path_convenience_imports_existing_private_store_into_unified_storage() {
    use tinystoragedrivers_core::{Scope, StorageBackend};

    let dir = tempfile::tempdir().unwrap();
    let old_backend: Arc<dyn StorageBackend> =
        Arc::new(SqliteStorage::open(dir.path().join(STORE_DIR)).unwrap());
    let old_provider = DriverSessionStores::new(old_backend, Scope::local()).unwrap();
    let session = SessionRef::root("private-store-thread");
    let old_stores = old_provider.try_for_agent("agent").unwrap();
    let handle = old_stores
        .transcripts
        .open_session(&session, meta())
        .unwrap();
    let messages = vec![
        TranscriptMessage::user("old private database"),
        TranscriptMessage::assistant("copied into shared storage"),
    ];
    handle
        .append_turn(TranscriptTurn {
            prev: &[],
            next: &messages,
            meta: &meta(),
            turn_usage: None,
            request_id: None,
            tools: None,
        })
        .unwrap();
    old_stores
        .kv
        .put("session", "old-key", serde_json::json!({"value": 7}))
        .await
        .unwrap();
    old_stores
        .journal
        .append(
            "harness.journal/old-run",
            serde_json::json!({"event": "kept"}),
        )
        .await
        .unwrap();
    drop(old_stores);
    drop(old_provider);
    let source_database = dir.path().join(STORE_DIR).join("storage.db");
    let source_bytes = std::fs::read(&source_database).unwrap();

    let moved = WorkspaceStores::open(dir.path(), "agent").unwrap();
    let imported = moved
        .transcripts()
        .read_session_transcript(&session)
        .unwrap()
        .read_session()
        .unwrap()
        .unwrap();
    assert_eq!(imported.messages[0].content, "old private database");
    assert_eq!(imported.messages[1].content, "copied into shared storage");
    assert_eq!(
        moved.stores().kv.get("session", "old-key").await.unwrap(),
        Some(serde_json::json!({"value": 7}))
    );
    assert_eq!(
        moved
            .stores()
            .journal
            .read_from("harness.journal/old-run", 0)
            .await
            .unwrap()[0]
            .1,
        serde_json::json!({"event": "kept"})
    );
    assert!(dir.path().join(STORAGE_DIR).join("storage.db").is_file());
    assert!(dir.path().join(STORE_DIR).join("storage.db").is_file());
    assert_eq!(std::fs::read(&source_database).unwrap(), source_bytes);

    // A completed import is skipped on later launches, so evolved destination
    // state does not conflict with the unchanged legacy source.
    moved
        .stores()
        .kv
        .put("session", "old-key", serde_json::json!({"value": 8}))
        .await
        .unwrap();
    drop(moved);
    std::fs::write(&source_database, b"legacy database is no longer readable").unwrap();
    let reopened = WorkspaceStores::open(dir.path(), "agent").unwrap();
    assert_eq!(
        reopened
            .stores()
            .kv
            .get("session", "old-key")
            .await
            .unwrap(),
        Some(serde_json::json!({"value": 8}))
    );
}

#[allow(deprecated)]
#[tokio::test]
async fn path_import_refuses_conflicting_destination_documents_without_overwriting() {
    use tinystoragedrivers_core::{Scope, StorageBackend};

    let dir = tempfile::tempdir().unwrap();
    let destination_backend: Arc<dyn StorageBackend> =
        Arc::new(SqliteStorage::open(dir.path().join(STORAGE_DIR)).unwrap());
    let destination = DriverSessionStores::new(destination_backend, Scope::local()).unwrap();
    let destination_stores = destination.try_for_agent("agent").unwrap();
    destination_stores
        .kv
        .put("session", "same-key", serde_json::json!({"value": "new"}))
        .await
        .unwrap();

    let old_backend: Arc<dyn StorageBackend> =
        Arc::new(SqliteStorage::open(dir.path().join(STORE_DIR)).unwrap());
    let old = DriverSessionStores::new(old_backend, Scope::local()).unwrap();
    old.try_for_agent("agent")
        .unwrap()
        .kv
        .put("session", "same-key", serde_json::json!({"value": "old"}))
        .await
        .unwrap();

    assert!(WorkspaceStores::open(dir.path(), "agent").is_err());
    assert_eq!(
        destination_stores
            .kv
            .get("session", "same-key")
            .await
            .unwrap(),
        Some(serde_json::json!({"value": "new"}))
    );
}

#[test]
fn a_dropped_handle_releases_its_backend_and_reopens_fresh() {
    let dir = tempfile::tempdir().unwrap();
    let first = WorkspaceStores::open(dir.path(), "agent").unwrap();
    let weak = Arc::downgrade(first.provider());
    drop(first);
    assert!(
        weak.upgrade().is_none(),
        "cache must not keep the backend alive"
    );
    WorkspaceStores::open(dir.path(), "agent").unwrap();
}

#[test]
fn an_incomplete_import_is_retried_on_the_next_open() {
    let dir = tempfile::tempdir().unwrap();
    let raw = dir.path().join("session_raw");
    std::fs::create_dir_all(&raw).unwrap();
    std::fs::write(raw.join("broken.jsonl"), "not json at all\n").unwrap();
    let first = WorkspaceStores::open(dir.path(), "agent").unwrap();
    assert!(matches!(
        first.import_outcome(),
        OnceOutcome::Imported { marked: false, .. }
    ));
    let second = WorkspaceStores::open(dir.path(), "agent").unwrap();
    assert!(!Arc::ptr_eq(first.provider(), second.provider()));
}
