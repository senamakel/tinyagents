use super::*;

use std::sync::Arc;

use crate::{Owner, Registry, Schema, ScopeClass, Step, TargetRecord};
use serde_json::json;
use tinystoragedrivers_api::{Storage, StorageBackend as PortStorageBackend};
use tinystoragedrivers_busproxy::{Admission, BusStorage, Loopback, StorageServer};
use tinystoragedrivers_testkit::{Failing, FaultPlan, memory_backend};
use tinystoragedrivers_types::{ErrorKind, Scope, WriteOptions};

#[tokio::test]
async fn retry_accepts_host_created_target_metadata_after_progress_write_failure() {
    let scope = Scope::new("schema-retry-created-ms").unwrap();
    let backend = memory_backend();
    let host_storage = PortStorageBackend::for_scope(&backend, &scope).unwrap();
    let schema = Schema::new(Owner::new("retry").unwrap())
        .collection("retry.current", 2, ScopeClass::Tenant)
        .step(
            Step::new("copy", "retry.current", 1, 2, |records| {
                Ok(records
                    .into_iter()
                    .map(|record| {
                        TargetRecord::new("retry.current", record.id, record.body)
                            .created_ms(record.meta.created_ms)
                    })
                    .collect())
            })
            .source("retry.legacy"),
        );
    let registry = Registry::compose([schema]).unwrap();
    registry.prepare(&backend).await.unwrap();

    // Seed a legacy row with a creation time that is intentionally older than
    // the host's current time. The module's storage-bus call cannot assign
    // host-owned `created_ms` on a new target row.
    host_storage
        .documents()
        .put_with(
            "retry.legacy",
            "row-1",
            json!({"value": "preserve"}),
            tinystoragedrivers_types::Precondition::Absent,
            WriteOptions::new().schema(1).created_ms(1),
        )
        .await
        .unwrap();

    // The first put_with is the migrated target. Fail the following progress
    // put so the retry sees a target written by an interrupted attempt.
    let faults = FaultPlan::new();
    faults.fail_operation_nth("documents.put_with", 2, ErrorKind::Unavailable);
    let documents = Arc::new(Failing::with_plan(
        Arc::clone(host_storage.documents()),
        faults,
    ));
    let served_storage = Storage::new(
        documents,
        Arc::clone(host_storage.streams()),
        Arc::clone(host_storage.kv()),
        Arc::clone(host_storage.blobs()),
        Arc::clone(host_storage.queues()),
        Arc::clone(host_storage.secrets()),
    );
    let server = Arc::new(StorageServer::new(served_storage));
    let module_storage =
        BusStorage::connect(Arc::new(Loopback::new(server, Admission::Confidential)))
            .await
            .unwrap();
    let scoped = crate::ScopedStorage::new(module_storage, scope);

    let first_attempt = registry
        .migrate(&scoped, &MigrateOptions::new())
        .await
        .unwrap_err();
    assert_eq!(
        first_attempt.kind(),
        ErrorKind::Unavailable,
        "the target write must survive while the progress write fails: {first_attempt:?}"
    );
    let migrated = host_storage
        .documents()
        .get("retry.current", "row-1")
        .await
        .unwrap()
        .expect("the first attempt wrote the target before its progress marker failed");
    assert_ne!(migrated.meta.created_ms, 1);

    registry
        .migrate(&scoped, &MigrateOptions::new())
        .await
        .unwrap();
    assert!(
        host_storage
            .documents()
            .get("retry.legacy", "row-1")
            .await
            .unwrap()
            .is_none()
    );
    let migrated_again = host_storage
        .documents()
        .get("retry.current", "row-1")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(migrated_again.doc, json!({"value": "preserve"}));
    assert_eq!(migrated_again.meta.schema, 2);
    assert_ne!(migrated_again.meta.created_ms, 1);
}
