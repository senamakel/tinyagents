use super::*;
use crate::testkit::MemoryStorage;

async fn refuses_corruption(change: impl FnOnce(&mut Value)) {
    let store = actor(&MemoryStorage::new()).await;
    let fault = Fault::new(Phase::Reserve, Action::Pause);
    let interrupted = faulted(&store, fault.clone());
    let mut writer =
        Box::pin(interrupted.append_message("same", message("agent:corrupt", "reserved")));
    tokio::select! {
        _ = fault.reached.notified() => {},
        result = &mut writer => panic!("reservation never paused: {result:?}"),
    }
    drop(writer);
    let current = store.load("same").await.unwrap().unwrap();
    let key = pending::AppendIntent::key("same", &current.doc.incarnation).unwrap();
    let row = store
        .docs
        .get(PENDING_COLLECTION, &key)
        .await
        .unwrap()
        .unwrap();
    let mut body = row.doc.clone();
    change(&mut body);
    let version = store
        .docs
        .put(PENDING_COLLECTION, &key, body.clone(), row.unchanged())
        .await
        .unwrap();
    let error = store
        .append_message("same", message("agent:next", "next"))
        .await
        .unwrap_err();
    assert!(
        matches!(error, ConversationError::Driver(ref error) if error.kind() == ErrorKind::Serialization)
    );
    assert!(store.get_messages("same").await.unwrap().is_empty());
    assert_eq!(
        store
            .get_thread("same")
            .await
            .unwrap()
            .unwrap()
            .message_count,
        0
    );
    let stream = current.doc.stream().unwrap();
    assert_eq!(store.streams.len(&stream).await.unwrap(), 0);
    let retained = store
        .docs
        .get(PENDING_COLLECTION, &key)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(retained.version, version);
    assert_eq!(retained.doc, body);
}

#[tokio::test]
async fn foreign_or_malformed_pending_identity_cannot_append_or_publish() {
    for (field, value) in [
        ("thread_id", json!("foreign-thread")),
        ("incarnation", json!("foreign-incarnation")),
        ("token", json!("not-an-attempt-uuid")),
        ("schema", json!("1")),
        ("offset", json!("2")),
        ("offset", json!(9)),
    ] {
        refuses_corruption(|body| body[field] = value.clone()).await;
    }
}

#[tokio::test]
async fn pending_record_cannot_silently_normalize_missing_message_metadata() {
    refuses_corruption(|body| {
        body["record"]
            .as_object_mut()
            .unwrap()
            .remove("extra_metadata");
    })
    .await;
}
