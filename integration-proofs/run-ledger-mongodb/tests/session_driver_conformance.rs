//! Opt-in full session-store and run-ledger conformance against a real local
//! MongoDB replica set. This excluded workspace keeps BSON's serde_json feature
//! unification out of the ordinary TinyAgents workspace graph.
#![cfg(feature = "mongodb-storage-tests")]

use std::sync::Arc;

use tinyagents_session::{
    port::DriverSessionStores,
    run_ledger::{AgentRunListRequest, DriverRunLedger, RunLedgerStore},
    testkit::conformance::{
        run_ledger_conformance, session_store_conformance, session_store_isolation_conformance,
    },
};
use tinystoragedrivers_core::{Scope, StorageBackend};
use tinystoragedrivers_mongodb::MongoStorage;

/// Require an explicit, credential-free loopback URL for the local fixture.
fn local_mongo_url() -> String {
    let value = std::env::var("TSD_MONGO_URL")
        .expect("set TSD_MONGO_URL to run ignored MongoDB conformance tests");
    let url = reqwest::Url::parse(&value).expect("TSD_MONGO_URL must be a valid URI");
    assert_eq!(url.scheme(), "mongodb", "MongoDB SRV URLs are not accepted");
    assert!(
        matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "::1")) && url.port().is_some(),
        "TSD_MONGO_URL must point to a local MongoDB server with an explicit port"
    );
    assert!(
        url.username().is_empty() && url.password().is_none(),
        "credentials are not accepted for local conformance tests"
    );
    value
}

async fn local_backend(label: &str) -> MongoStorage {
    let url = local_mongo_url();
    let database = format!("ta_{label}_{}", uuid::Uuid::new_v4().simple());
    let backend = MongoStorage::connect(&url, &database)
        .await
        .expect("local MongoDB replica set is reachable");
    backend
        .prepare_schema()
        .await
        .expect("the isolated test database schema prepares");
    backend
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires local MongoDB; run explicitly with TSD_MONGO_URL and --ignored"]
async fn mongodb_session_store_round_trip_list_delete_and_agent_isolation() {
    let backend: Arc<dyn StorageBackend> = Arc::new(local_backend("session_store").await);
    let stores =
        DriverSessionStores::new(backend, Scope::tenant("session_store_conformance").unwrap())
            .unwrap();
    session_store_conformance(&stores).await;
    session_store_isolation_conformance(&stores).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires local MongoDB; run explicitly with TSD_MONGO_URL and --ignored"]
async fn mongodb_run_ledger_contract_and_scope_isolation() {
    let backend = local_backend("run_ledger").await;
    let scoped = |scope: &str| {
        Arc::new(DriverRunLedger::new(
            backend.for_scope(&Scope::tenant(scope).unwrap()).unwrap(),
        )) as Arc<dyn RunLedgerStore>
    };
    let ledger = scoped("run_ledger_conformance");
    run_ledger_conformance(ledger.clone()).await;
    assert!(
        scoped("run_ledger_peer")
            .list_agent_runs(&AgentRunListRequest::default())
            .await
            .unwrap()
            .runs
            .is_empty()
    );
}
