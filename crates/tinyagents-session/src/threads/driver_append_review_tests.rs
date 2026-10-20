use super::*;
use crate::testkit::MemoryStorage;
use std::error::Error;

#[tokio::test]
async fn lost_finalization_ack_keeps_success_after_concurrent_title_edit() {
    let backend = MemoryStorage::new();
    let store = actor(&backend).await;
    // The first exact-row read precedes finalization. Pause the second, after
    // committed() has captured already-published metadata and a real stream row.
    let read = Fault::new_after(Phase::ReadWindow, Action::Pause, 1);
    let observed = faulted(&store, read.clone());
    let writer = faulted(&observed, Fault::new(Phase::Finalize, Action::LoseAck));
    let original = message("ordinary", "accepted once");
    let mut call = Box::pin(writer.append_message("same", original.clone()));
    tokio::select! {
        result = &mut call => panic!("append completed before committed read barrier: {result:?}"),
        () = read.reached.notified() => {}
    }
    store
        .update_thread_title("same", "Concurrent title", "later")
        .await
        .unwrap();
    read.release.notify_one();
    assert_eq!(call.await.unwrap(), original);
    assert_eq!(store.get_messages("same").await.unwrap(), vec![original]);
    assert_eq!(
        store.get_thread("same").await.unwrap().unwrap().title,
        "Concurrent title"
    );
}

#[tokio::test]
async fn missing_published_row_during_finalize_reconcile_preserves_corruption() {
    let backend = MemoryStorage::new();
    let store = actor(&backend).await;
    let finalized = Fault::new(Phase::Finalize, Action::PauseLoseAck);
    let writer = faulted(&store, finalized.clone());
    let mut call = Box::pin(writer.append_message("same", message("ordinary", "accepted")));
    tokio::select! {
        result = &mut call => panic!("append completed before actual finalization: {result:?}"),
        () = finalized.reached.notified() => {}
    }
    let row = store.load("same").await.unwrap().unwrap();
    assert_eq!(row.doc.visible_end, 1);
    assert_eq!(row.doc.thread.message_count, 1);
    let stream = row.doc.stream().unwrap();
    assert!(store.streams.delete_stream(&stream).await.unwrap());
    finalized.release.notify_one();
    let error = call.await.unwrap_err();
    let ConversationError::Driver(native) = error else {
        panic!("definitive committed-row corruption was masked: {error:?}");
    };
    assert_eq!(native.kind(), ErrorKind::Serialization);
    assert!(
        native
            .to_string()
            .contains("missing committed conversation append")
    );
    assert_eq!(
        store.load("same").await.unwrap().unwrap().doc.visible_end,
        1
    );
    assert_eq!(store.streams.len(&stream).await.unwrap(), 0);
}

#[tokio::test]
async fn helping_foreign_uncertain_intent_keeps_callers_unreserved_outcome_retryable() {
    let backend = MemoryStorage::new();
    let store = actor(&backend).await;
    let reserved = Fault::new(Phase::Reserve, Action::Pause);
    let first = faulted(&store, reserved.clone());
    let foreign = message("foreign-ordinary", "foreign intent");
    let mut call = Box::pin(first.append_message("same", foreign.clone()));
    tokio::select! {
        result = &mut call => panic!("foreign append completed before reserve barrier: {result:?}"),
        () = reserved.reached.notified() => {}
    }
    drop(call);
    let failing_read = faulted(&store, Fault::new(Phase::ReadWindow, Action::FailBefore));
    let helper = faulted(&failing_read, Fault::new(Phase::Append, Action::LoseAck));
    let own = message("own-ordinary", "not accepted yet");
    let error = helper
        .append_message("same", own.clone())
        .await
        .unwrap_err();
    let ConversationError::Driver(native) = error else {
        panic!("foreign uncertainty must not describe the unreserved caller: {error:?}");
    };
    assert_eq!(native.kind(), ErrorKind::Unavailable);
    assert!(
        native
            .source()
            .unwrap()
            .downcast_ref::<std::io::Error>()
            .is_some()
    );
    assert!(store.get_messages("same").await.unwrap().is_empty());
    let row = store.load("same").await.unwrap().unwrap();
    let stream = row.doc.stream().unwrap();
    assert_eq!(store.streams.len(&stream).await.unwrap(), 1);
    let pending = store
        .docs
        .query(PENDING_COLLECTION, &Query::all())
        .await
        .unwrap();
    assert_eq!(pending.items.len(), 1);
    assert_eq!(
        store.append_message("same", own.clone()).await.unwrap(),
        own
    );
    assert_eq!(
        store.get_messages("same").await.unwrap(),
        vec![foreign, own]
    );
}
