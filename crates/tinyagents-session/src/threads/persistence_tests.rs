use super::*;
use serde_json::json;
use tinyagents_storage_kit::{Scope, StorageBackend};
use tinystoragedrivers_types::ErrorKind;

async fn memory_store() -> DriverConversationStore {
    let backend = tinystoragedrivers_testkit::memory_backend();
    let scoped = backend
        .for_scope(&Scope::new("session-fixture-planner").unwrap())
        .unwrap();
    DriverConversationStore::prepare_append_schema(scoped.documents().as_ref())
        .await
        .unwrap();
    DriverConversationStore::new(scoped)
}

fn thread(id: &str) -> CreateConversationThread {
    CreateConversationThread {
        id: id.into(),
        title: "First".into(),
        created_at: "2026-01-01T00:00:00Z".into(),
        parent_thread_id: None,
        labels: None,
        personality_id: None,
        working_dir: None,
    }
}
fn message(id: &str, content: &str) -> ThreadMessage {
    ThreadMessage {
        id: id.into(),
        content: content.into(),
        message_type: "text".into(),
        extra_metadata: json!({}),
        sender: "assistant".into(),
        created_at: "2026-01-02T00:00:00Z".into(),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn memory_store_preserves_deterministic_append_and_missing_thread_behavior() {
    let store: Box<dyn ConversationPersistence> = Box::new(memory_store().await);
    assert!(store.get_thread("missing").await.unwrap().is_none());
    assert!(store.get_messages("missing").await.unwrap().is_empty());
    assert!(matches!(
        store
            .append_message("missing", message("agent:x", "first"))
            .await,
        Err(ConversationError::Driver(error)) if error.kind() == ErrorKind::NotFound
    ));
    store.ensure_thread(thread("one")).await.unwrap();
    let first = store
        .append_message("one", message("agent:x", "first"))
        .await
        .unwrap();
    assert_eq!(
        store
            .append_message("one", message("agent:x", "second"))
            .await
            .unwrap(),
        first
    );
    assert_eq!(
        store
            .get_thread("one")
            .await
            .unwrap()
            .unwrap()
            .message_count,
        1
    );
}

#[tokio::test]
async fn memory_store_preserves_title_labels_and_metadata_patch() {
    let store = memory_store().await;
    store.ensure_thread(thread("one")).await.unwrap();
    store
        .update_thread_labels("one", vec!["work".into(), "general".into()], "2026-01-03")
        .await
        .unwrap();
    let updated = store
        .update_thread_title("one", "New", "2026-01-04")
        .await
        .unwrap();
    assert_eq!(updated.title, "New");
    assert_eq!(updated.labels, vec!["general"]);
    store
        .append_message("one", message("agent:x", "unchanged"))
        .await
        .unwrap();
    let patched = store
        .update_message(
            "one",
            "agent:x",
            ThreadMessagePatch {
                extra_metadata: Some(json!({"success": true})),
            },
        )
        .await
        .unwrap();
    assert_eq!(patched.content, "unchanged");
    assert_eq!(patched.extra_metadata, json!({"success": true}));
    store
        .update_thread_working_dir("one", Some("/approved".into()), "2026-01-05")
        .await
        .unwrap();
    assert_eq!(
        store
            .get_thread("one")
            .await
            .unwrap()
            .unwrap()
            .working_dir
            .as_deref(),
        Some("/approved")
    );
    store
        .update_thread_working_dir("one", None, "2026-01-06")
        .await
        .unwrap();
    assert!(
        store
            .get_thread("one")
            .await
            .unwrap()
            .unwrap()
            .working_dir
            .is_none()
    );
}

#[tokio::test]
async fn memory_store_tail_delete_recreate_and_purge() {
    let store = memory_store().await;
    store.ensure_thread(thread("one")).await.unwrap();
    for id in ["a", "b", "c"] {
        store.append_message("one", message(id, id)).await.unwrap();
    }
    assert_eq!(
        store.delete_messages_from("one", "unknown").await.unwrap(),
        None
    );
    assert_eq!(
        store.delete_messages_from("one", "b").await.unwrap(),
        Some(2)
    );
    assert_eq!(
        store.get_messages("one").await.unwrap(),
        vec![message("a", "a")]
    );
    assert!(store.delete_thread("one", "2026-01-03").await.unwrap());
    assert!(!store.delete_thread("one", "2026-01-03").await.unwrap());
    store.ensure_thread(thread("one")).await.unwrap();
    assert!(store.get_messages("one").await.unwrap().is_empty());
    store
        .append_message("one", message("new", "new"))
        .await
        .unwrap();
    let counts = store.purge_threads().await.unwrap();
    assert_eq!((counts.thread_count, counts.message_count), (1, 1));
    assert!(store.list_threads().await.unwrap().is_empty());
}

#[tokio::test]
async fn memory_store_substring_and_cjk_search_find_expected_threads() {
    let store = memory_store().await;
    for (id, content) in [
        ("a", "Café substring alphabet"),
        ("b", "東京で日本語"),
        ("c", "unrelated"),
    ] {
        store.ensure_thread(thread(id)).await.unwrap();
        store
            .append_message(id, message(id, content))
            .await
            .unwrap();
    }
    let hit = |query: &'static str| {
        let store = store.clone();
        async move {
            store
                .search_cross_thread_messages(query, 20, None)
                .await
                .unwrap()
                .into_iter()
                .map(|hit| hit.thread_id)
                .collect::<Vec<_>>()
        }
    };
    assert_eq!(hit("cafe").await, vec!["a"]);
    assert_eq!(hit("pha").await, vec!["a"]);
    assert_eq!(hit("東京").await, vec!["b"]);
    assert_eq!(hit("日").await, vec!["b"]);
    assert!(hit("missing").await.is_empty());
    assert!(hit("").await.is_empty());
    assert!(
        store
            .search_cross_thread_messages("東京", 20, Some("b"))
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn memory_store_refuses_authoritative_turn_claim_() {
    let store = memory_store().await;
    let now = chrono::DateTime::parse_from_rfc3339("2026-01-01T00:00:00Z")
        .unwrap()
        .with_timezone(&chrono::Utc);
    let request = ConversationTurnClaimRequest {
        thread_id: "one".into(),
        request_id: "turn".into(),
        owner: "pod".into(),
        now,
        lease_until: now + chrono::Duration::seconds(30),
    };
    assert!(matches!(
        store.claim_turn(request).await,
        Err(ConversationError::Unsupported {
            operation: "claim_turn"
        })
    ));
    let claim = ConversationTurnClaim {
        thread_id: "one".into(),
        request_id: "turn".into(),
        owner: "pod".into(),
        epoch: 1,
        lease_until: now,
    };
    assert!(matches!(
        store.renew_turn(&claim, now, now).await,
        Err(ConversationError::Unsupported {
            operation: "renew_turn"
        })
    ));
    assert!(matches!(
        store.complete_turn(&claim, now).await,
        Err(ConversationError::Unsupported {
            operation: "complete_turn"
        })
    ));
    assert!(matches!(
        store.release_turn(&claim, now).await,
        Err(ConversationError::Unsupported {
            operation: "release_turn"
        })
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn memory_store_preserves_channel_dedup_and_optional_upsert_fields() {
    let store = memory_store().await;
    let mut first = thread("one");
    first.labels = Some(vec!["worker".into()]);
    first.personality_id = Some("persona".into());
    first.working_dir = Some("/approved".into());
    store.ensure_thread(first).await.unwrap();
    let mut updated = thread("one");
    updated.created_at = "2026-01-03T00:00:00Z".into();
    let row = store.ensure_thread(updated).await.unwrap();
    assert_eq!(row.created_at, "2026-01-01T00:00:00Z");
    assert_eq!(row.labels, vec!["tasks"]);
    assert_eq!(row.personality_id.as_deref(), Some("persona"));
    assert_eq!(row.working_dir.as_deref(), Some("/approved"));
    let mut channel = message("platform-id", "first");
    channel.extra_metadata = json!({"scope": "channel"});
    store.append_message("one", channel.clone()).await.unwrap();
    let mut retry = channel.clone();
    retry.content = "different".into();
    assert_eq!(store.append_message("one", retry).await.unwrap(), channel);
    assert_eq!(store.get_messages("one").await.unwrap().len(), 1);
}

#[tokio::test]
async fn update_message_rewrites_only_the_patched_message_and_keeps_appends_working() {
    let store = memory_store().await;
    store.ensure_thread(thread("one")).await.unwrap();
    for id in ["a", "b"] {
        store.append_message("one", message(id, id)).await.unwrap();
    }
    let patched = store
        .update_message(
            "one",
            "b",
            ThreadMessagePatch {
                extra_metadata: Some(json!({"k": 1})),
            },
        )
        .await
        .unwrap();
    assert_eq!(patched.extra_metadata, json!({"k": 1}));
    let rows = store.get_messages("one").await.unwrap();
    assert_eq!(rows[0], message("a", "a"));
    assert_eq!(rows[1].extra_metadata, json!({"k": 1}));
    store
        .append_message("one", message("c", "c"))
        .await
        .unwrap();
    assert_eq!(store.get_messages("one").await.unwrap().len(), 3);
    assert_eq!(
        store
            .get_thread("one")
            .await
            .unwrap()
            .unwrap()
            .message_count,
        3
    );
    assert!(matches!(
        store
            .update_message("one", "missing", ThreadMessagePatch::default())
            .await,
        Err(ConversationError::Driver(error)) if error.kind() == ErrorKind::NotFound
    ));
    assert!(matches!(
        store
            .update_message("ghost", "a", ThreadMessagePatch::default())
            .await,
        Err(ConversationError::Driver(error)) if error.kind() == ErrorKind::NotFound
    ));
}

#[tokio::test]
async fn truncation_resets_counts_and_timestamps_and_the_search_index() {
    let store = memory_store().await;
    store.ensure_thread(thread("one")).await.unwrap();
    let mut first = message("a", "keepsake");
    first.created_at = "2026-01-02T00:00:00Z".into();
    let mut second = message("b", "discarded phrase");
    second.created_at = "2026-01-03T00:00:00Z".into();
    store.append_message("one", first).await.unwrap();
    store.append_message("one", second).await.unwrap();
    assert_eq!(
        store.delete_messages_from("one", "b").await.unwrap(),
        Some(1)
    );
    let row = store.get_thread("one").await.unwrap().unwrap();
    assert_eq!(row.message_count, 1);
    assert_eq!(row.last_message_at, "2026-01-02T00:00:00Z");
    assert!(
        store
            .search_cross_thread_messages("discarded", 5, None)
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        store.delete_messages_from("one", "a").await.unwrap(),
        Some(1)
    );
    let row = store.get_thread("one").await.unwrap().unwrap();
    assert_eq!(row.message_count, 0);
    assert_eq!(row.last_message_at, row.created_at);
    assert_eq!(
        store.delete_messages_from("ghost", "a").await.unwrap(),
        None
    );
}

#[tokio::test]
async fn purge_removes_live_and_deleted_threads_and_counts_only_live_ones() {
    let store = memory_store().await;
    for id in ["one", "two", "gone"] {
        store.ensure_thread(thread(id)).await.unwrap();
        store.append_message(id, message("m", id)).await.unwrap();
    }
    store.delete_thread("gone", "2026-01-03").await.unwrap();
    let stats = store.purge_threads().await.unwrap();
    assert_eq!((stats.thread_count, stats.message_count), (2, 2));
    assert!(store.list_threads().await.unwrap().is_empty());
    assert_eq!(
        store.purge_threads().await.unwrap(),
        ConversationPurgeStats::default()
    );
    store.ensure_thread(thread("one")).await.unwrap();
    assert!(store.get_messages("one").await.unwrap().is_empty());
}

#[tokio::test]
async fn a_rewrite_retires_a_stale_append_intent_instead_of_resurrecting_it() {
    let store = memory_store().await;
    store.ensure_thread(thread("one")).await.unwrap();
    store
        .append_message("one", message("a", "a"))
        .await
        .unwrap();
    store.delete_messages_from("one", "a").await.unwrap();
    store
        .append_message("one", message("b", "b"))
        .await
        .unwrap();
    assert_eq!(
        store.get_messages("one").await.unwrap(),
        vec![message("b", "b")]
    );
}
