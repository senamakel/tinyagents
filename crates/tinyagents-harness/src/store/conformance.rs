//! Reusable storage **conformance** (contract) suites for the harness
//! [`Store`] and [`NamespacedStore`] traits.
//!
//! Mirrors the pattern the graph module established in
//! `tinyagents_graph::testkit::conformance` and the session crate's
//! `tinyagents_session::testkit::conformance`: the same assertions are run
//! against every backend so a defect in one implementation cannot hide behind
//! another, and a downstream author implementing either trait can certify
//! their backend by calling the matching function from a `#[tokio::test]`.
//!
//! # Example
//!
//! ```rust
//! use tinyagents_harness::store::InMemoryStore;
//! use tinyagents_harness::store::conformance::run_store_conformance;
//!
//! # tokio_test::block_on(async {
//! run_store_conformance(&InMemoryStore::new()).await;
//! # });
//! ```
//!
//! Each function panics with a descriptive message on the first violation.

use serde_json::json;

use super::Store;
use super::namespaced::{
    FilterOp, ListNamespacesQuery, Namespace, NamespacedStore, SearchQuery, StoreOp,
};

/// Runs the flat [`Store`] contract against `store`.
///
/// Covers put/get (including overwrite), delete (including deleting an
/// absent key, which must not error), list, and namespace isolation. Any
/// backend that passes this behaves interchangeably as a flat harness
/// [`Store`] — this is what [`crate::store::InMemoryStore`] and
/// [`crate::store::FileStore`] are both certified against.
pub async fn run_store_conformance<S: Store>(store: &S) {
    // put + get round-trips the exact value.
    store
        .put("ns-a", "k1", json!({"v": 1}))
        .await
        .expect("put k1");
    let got = store.get("ns-a", "k1").await.expect("get k1");
    assert_eq!(got, Some(json!({"v": 1})), "get returns what was put");

    // A key that was never written is a `None`, not an error.
    assert!(
        store
            .get("ns-a", "missing")
            .await
            .expect("get of a missing key")
            .is_none(),
        "get of a never-written key returns None"
    );

    // A namespace that was never written behaves the same way for both get
    // and list — no error, just absence.
    assert!(
        store
            .get("ns-never-written", "k1")
            .await
            .expect("get from a never-written namespace")
            .is_none(),
        "get from a never-written namespace returns None"
    );
    assert!(
        store
            .list("ns-never-written")
            .await
            .expect("list of a never-written namespace")
            .is_empty(),
        "list of a never-written namespace returns an empty Vec"
    );

    // put is an upsert: writing the same key again replaces the value.
    store
        .put("ns-a", "k1", json!({"v": 2}))
        .await
        .expect("overwrite k1");
    let got = store
        .get("ns-a", "k1")
        .await
        .expect("get k1 after overwrite");
    assert_eq!(got, Some(json!({"v": 2})), "put overwrites the prior value");

    // list enumerates every key written to the namespace.
    store.put("ns-a", "k2", json!("second")).await.expect("put k2");
    let mut keys = store.list("ns-a").await.expect("list ns-a");
    keys.sort();
    assert_eq!(
        keys,
        vec!["k1".to_string(), "k2".to_string()],
        "list returns every key written to the namespace"
    );

    // Namespaces are independent: a write to one is invisible from another,
    // and listing one namespace never surfaces another's keys.
    store
        .put("ns-b", "k1", json!("other namespace"))
        .await
        .expect("put ns-b k1");
    assert_eq!(
        store.get("ns-b", "k1").await.expect("get ns-b k1"),
        Some(json!("other namespace")),
        "a namespace's own write is visible"
    );
    assert_eq!(
        store.get("ns-a", "k1").await.expect("get ns-a k1 unaffected"),
        Some(json!({"v": 2})),
        "writing to ns-b does not affect ns-a's value for the same key"
    );
    assert_eq!(
        store.list("ns-b").await.expect("list ns-b"),
        vec!["k1".to_string()],
        "listing ns-b never surfaces ns-a's keys"
    );

    // delete removes exactly the deleted key.
    store.delete("ns-a", "k1").await.expect("delete k1");
    assert!(
        store
            .get("ns-a", "k1")
            .await
            .expect("get after delete")
            .is_none(),
        "a deleted key reads back as None"
    );
    assert_eq!(
        store.list("ns-a").await.expect("list after delete"),
        vec!["k2".to_string()],
        "delete removes exactly the deleted key, leaving the rest of the namespace intact"
    );

    // Deleting an already-absent key, or a key in a namespace that was never
    // written, is a no-op — not an error.
    store
        .delete("ns-a", "k1")
        .await
        .expect("delete of an already-absent key must not error");
    store
        .delete("ns-never-written", "nope")
        .await
        .expect("delete from a never-written namespace must not error");
}

/// Runs the [`NamespacedStore`] contract against `store`.
///
/// Covers put/get (including overwrite preserving `created_at_ms`), delete,
/// search (namespace-prefix scoping and field filtering), namespace listing,
/// TTL expiry, and `batch`'s positional-alignment guarantee. Any backend that
/// passes this behaves interchangeably as a [`NamespacedStore`] — this is
/// what [`crate::store::namespaced::InMemoryNamespacedStore`] is certified
/// against.
pub async fn run_namespaced_store_conformance<S: NamespacedStore>(store: &S) {
    let ns_alice = Namespace::new(["users", "alice"]).expect("valid namespace");
    let ns_bob = Namespace::new(["users", "bob"]).expect("valid namespace");

    // put + get round-trips the exact value, addressed by namespace and key.
    store
        .put(&ns_alice, "profile", json!({"name": "Alice"}))
        .await
        .expect("put alice profile");
    let item = store
        .get(&ns_alice, "profile")
        .await
        .expect("get alice profile")
        .expect("item is present");
    assert_eq!(item.value, json!({"name": "Alice"}), "get returns what was put");
    assert_eq!(item.namespace, ns_alice, "item records its own namespace");
    assert_eq!(item.key, "profile", "item records its own key");

    // A key that was never written is a `None`, not an error.
    assert!(
        store
            .get(&ns_alice, "missing")
            .await
            .expect("get of a missing key")
            .is_none(),
        "get of a never-written key returns None"
    );

    // Overwriting preserves the original creation time and advances the
    // update time — an item's identity survives a write, only its content and
    // freshness change.
    let before = store
        .get(&ns_alice, "profile")
        .await
        .expect("get before overwrite")
        .expect("present");
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    store
        .put(&ns_alice, "profile", json!({"name": "Alice", "v": 2}))
        .await
        .expect("overwrite alice profile");
    let after = store
        .get(&ns_alice, "profile")
        .await
        .expect("get after overwrite")
        .expect("present");
    assert_eq!(
        after.created_at_ms, before.created_at_ms,
        "overwrite preserves the item's original creation time"
    );
    assert!(
        after.updated_at_ms >= before.updated_at_ms,
        "overwrite advances the item's update time"
    );
    assert_eq!(after.value["v"], json!(2), "overwrite replaces the value");

    // Namespaces are independent.
    store
        .put(&ns_bob, "profile", json!({"name": "Bob"}))
        .await
        .expect("put bob profile");
    let bob = store
        .get(&ns_bob, "profile")
        .await
        .expect("get bob profile")
        .expect("present");
    assert_eq!(bob.value["name"], json!("Bob"));
    let alice_again = store
        .get(&ns_alice, "profile")
        .await
        .expect("get alice profile again")
        .expect("present");
    assert_eq!(
        alice_again.value["name"],
        json!("Alice"),
        "writing bob's namespace does not affect alice's"
    );

    // search is namespace-prefix scoped: a broader prefix sees both items, a
    // narrower one (a specific user) sees only its own.
    let under_users = store
        .search(SearchQuery {
            namespace_prefix: vec!["users".to_string()],
            ..Default::default()
        })
        .await
        .expect("search under the users prefix");
    assert_eq!(
        under_users.len(),
        2,
        "search finds every item at or beneath the namespace prefix"
    );
    let alice_only = store
        .search(SearchQuery {
            namespace_prefix: vec!["users".to_string(), "alice".to_string()],
            ..Default::default()
        })
        .await
        .expect("search scoped to alice");
    assert_eq!(alice_only.len(), 1, "a narrower prefix excludes bob's item");
    assert_eq!(alice_only[0].key, "profile");

    // search applies the field filter as a conjunction.
    let filtered = store
        .search(SearchQuery {
            namespace_prefix: vec!["users".to_string()],
            filter: [("name".to_string(), FilterOp::Eq(json!("Bob")))]
                .into_iter()
                .collect(),
            ..Default::default()
        })
        .await
        .expect("search with a field filter");
    assert_eq!(filtered.len(), 1, "the filter selects only the matching item");
    assert_eq!(filtered[0].value["name"], json!("Bob"));

    // list_namespaces enumerates namespaces matching a prefix query.
    let namespaces = store
        .list_namespaces(ListNamespacesQuery {
            prefix: Some(vec!["users".to_string()]),
            ..Default::default()
        })
        .await
        .expect("list_namespaces under users");
    assert!(
        namespaces.contains(&ns_alice),
        "list_namespaces includes alice's namespace"
    );
    assert!(
        namespaces.contains(&ns_bob),
        "list_namespaces includes bob's namespace"
    );

    // delete removes exactly the deleted item, and deleting an already-absent
    // item is a no-op rather than an error.
    store
        .delete(&ns_alice, "profile")
        .await
        .expect("delete alice profile");
    assert!(
        store
            .get(&ns_alice, "profile")
            .await
            .expect("get after delete")
            .is_none(),
        "a deleted item reads back as None"
    );
    store
        .delete(&ns_alice, "profile")
        .await
        .expect("delete of an already-absent item must not error");

    // TTL: an item written with a very short explicit lifetime is invisible
    // to both get and search once it has expired, even though nothing ever
    // called `sweep_expired` — expiry is enforced on read.
    store
        .put_with_ttl(&ns_bob, "ephemeral", json!("soon gone"), Some(0.0001))
        .await
        .expect("put with a short ttl");
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert!(
        store
            .get(&ns_bob, "ephemeral")
            .await
            .expect("get an expired item")
            .is_none(),
        "an expired item is invisible to get before it is ever swept"
    );
    let after_expiry = store
        .search(SearchQuery {
            namespace_prefix: vec!["users".to_string(), "bob".to_string()],
            ..Default::default()
        })
        .await
        .expect("search after expiry");
    assert!(
        after_expiry.iter().all(|found| found.key != "ephemeral"),
        "search does not return an expired item"
    );

    // batch: several operations in one call return one result per operation,
    // positionally aligned with the request — the contract every convenience
    // method on the trait is built from.
    let ns_batch = Namespace::new(["batch"]).expect("valid namespace");
    let ops = vec![
        StoreOp::Put {
            namespace: ns_batch.clone(),
            key: "one".to_string(),
            value: Some(json!(1)),
            ttl_minutes: None,
        },
        StoreOp::Put {
            namespace: ns_batch.clone(),
            key: "two".to_string(),
            value: Some(json!(2)),
            ttl_minutes: None,
        },
        StoreOp::Get {
            namespace: ns_batch.clone(),
            key: "one".to_string(),
            refresh_ttl: None,
        },
        StoreOp::Get {
            namespace: ns_batch.clone(),
            key: "two".to_string(),
            refresh_ttl: None,
        },
    ];
    let mut results = store.batch(&ops).await.expect("batch");
    assert_eq!(results.len(), 4, "batch returns one result per submitted operation");
    let two = results
        .remove(3)
        .into_item()
        .expect("op 3 is a Get")
        .expect("present");
    let one = results
        .remove(2)
        .into_item()
        .expect("op 2 is a Get")
        .expect("present");
    assert_eq!(
        (one.value, two.value),
        (json!(1), json!(2)),
        "batch results are positionally aligned with the request, not just in the same order"
    );
}
