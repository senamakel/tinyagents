use super::*;
use crate::{Migrations, Scope, StorageBackend};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use tinystoragedrivers_testkit::MemoryBackend;

fn scoped(backend: &MemoryBackend) -> ScopedStorage {
    backend.for_scope(&Scope::new("local").unwrap()).unwrap()
}

#[tokio::test(flavor = "current_thread")]
async fn same_owner_open_sequences_are_serialized() {
    let backend = MemoryBackend::new();
    let storage = scoped(&backend);
    let active = Arc::new(AtomicUsize::new(0));
    let max_active = Arc::new(AtomicUsize::new(0));
    let release_first = Arc::new(tokio::sync::Notify::new());
    let first_active = active.clone();
    let first_max = max_active.clone();
    let first_release = release_first.clone();
    let first = tokio::spawn(async move {
        with_migration_lease(&storage, "harness", move |_| async move {
            let now = first_active.fetch_add(1, Ordering::SeqCst) + 1;
            first_max.fetch_max(now, Ordering::SeqCst);
            first_release.notified().await;
            first_active.fetch_sub(1, Ordering::SeqCst);
            Ok(())
        })
        .await
    });

    let storage = scoped(&backend);
    let second_started = Arc::new(tokio::sync::Notify::new());
    let second_signal = second_started.clone();
    let second_active = active.clone();
    let second_max = max_active.clone();
    let second = tokio::spawn(async move {
        second_signal.notify_one();
        with_migration_lease(&storage, "harness", move |_| async move {
            let now = second_active.fetch_add(1, Ordering::SeqCst) + 1;
            second_max.fetch_max(now, Ordering::SeqCst);
            second_active.fetch_sub(1, Ordering::SeqCst);
            Ok(())
        })
        .await
    });
    second_started.notified().await;
    tokio::task::yield_now().await;
    assert_eq!(active.load(Ordering::SeqCst), 1);
    release_first.notify_one();
    first.await.unwrap().unwrap();
    second.await.unwrap().unwrap();
    assert_eq!(max_active.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_failed_open_releases_its_lease_for_retry() {
    let storage = scoped(&MemoryBackend::new());
    let failed = with_migration_lease(&storage, "graph", |_| async {
        Err::<(), _>(StorageError::backend("injected migration failure"))
    })
    .await;
    assert!(failed.is_err());
    with_migration_lease(&storage, "graph", |_| async { Ok(()) })
        .await
        .unwrap();
}

#[tokio::test]
async fn a_stale_lease_cannot_mark_owner_migration_complete() {
    let storage = scoped(&MemoryBackend::new());
    let now = Arc::new(AtomicU64::new(1_000));
    let clock_now = now.clone();
    let leases = Leases::new(storage.documents())
        .with_clock(Arc::new(move || clock_now.load(Ordering::SeqCst)));
    let stale = MigrationLease::acquire(&storage, "tasks", 5, leases.clone())
        .await
        .unwrap();

    // Model the takeover in the gap after a successful renew and before the
    // holder reads the renewed document version for its checked marker write.
    leases
        .renew("tasks", MIGRATION_KEY, &stale.holder, stale.lease.epoch, 5)
        .await
        .unwrap()
        .expect("first tenure renews");
    now.store(1_010, Ordering::SeqCst);
    let replacement = leases
        .acquire("tasks", MIGRATION_KEY, "replacement", 100)
        .await
        .unwrap()
        .expect("expired tenure can be taken over");
    let lease_doc = storage
        .documents()
        .get(primitives::LEASES, lease_id("tasks").unwrap().as_str())
        .await
        .unwrap()
        .unwrap();
    assert!(ensure_tenure(&lease_doc.doc, &stale.holder, stale.lease.epoch).is_err());
    assert!(stale.mark_done("tasks", "records_v2").await.is_err());
    assert!(
        !Migrations::new(&storage)
            .done("tasks", "records_v2")
            .await
            .unwrap()
    );
    assert!(
        leases
            .release(
                "tasks",
                MIGRATION_KEY,
                &replacement.holder,
                replacement.epoch,
            )
            .await
            .unwrap()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn lease_loss_cancels_an_in_progress_owner_open() {
    struct DropSignal(Arc<AtomicUsize>);
    impl Drop for DropSignal {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let backend = MemoryBackend::new();
    let storage = scoped(&backend);
    let now = Arc::new(AtomicU64::new(1_000));
    let clock_now = now.clone();
    let leases = Leases::new(storage.documents())
        .with_clock(Arc::new(move || clock_now.load(Ordering::SeqCst)));
    let dropped = Arc::new(AtomicUsize::new(0));
    let dropped_work = dropped.clone();
    let takeover_now = now.clone();
    let takeover_leases = leases.clone();
    let takeover_storage = storage.clone();
    let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();

    let opening = tokio::spawn(async move {
        with_migration_lease_using(
            &takeover_storage,
            "tasks",
            30,
            takeover_leases.clone(),
            move |_| async move {
                takeover_now.store(2_000, Ordering::SeqCst);
                takeover_leases
                    .acquire("tasks", MIGRATION_KEY, "replacement", 30)
                    .await?
                    .expect("expired lease is taken over");
                let _drop_signal = DropSignal(dropped_work);
                let _ = entered_tx.send(());
                std::future::pending::<Result<()>>().await
            },
        )
        .await
    });

    entered_rx.await.expect("work entered after takeover");
    let result = tokio::time::timeout(Duration::from_secs(1), opening)
        .await
        .expect("lease loss aborts in-progress work")
        .unwrap();
    assert_eq!(result.unwrap_err().kind(), ErrorKind::Conflict);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    let replacement = leases
        .current("tasks", MIGRATION_KEY)
        .await
        .unwrap()
        .expect("stale cleanup does not release replacement tenure");
    assert_eq!(replacement.holder, "replacement");
}
