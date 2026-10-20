use super::*;
use crate::tool::discover::EmbeddingToolRanker;
use std::sync::atomic::{AtomicUsize, Ordering};
use tinyagents_storage_kit::{Scope, StorageBackend};
use tinyinference_embeddings::{EmbeddingModel, Result as EmbedResult};
use tinystoragedrivers_testkit::MemoryBackend;
use tinytools::{RankCandidate, RankContext, ToolRanker};

fn docs(storage: &dyn StorageBackend, scope: &str) -> Arc<dyn DocumentStore> {
    Arc::clone(
        storage
            .for_scope(&Scope::new(scope).unwrap())
            .unwrap()
            .documents(),
    )
}

struct Bag {
    calls: AtomicUsize,
}

#[async_trait::async_trait]
impl EmbeddingModel for Bag {
    fn name(&self) -> &str {
        "bag"
    }
    fn model_id(&self) -> &str {
        "bag-v1"
    }
    fn dimensions(&self) -> usize {
        3
    }
    async fn embed(&self, texts: &[String]) -> EmbedResult<Vec<Vec<f32>>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(texts
            .iter()
            .map(|t| {
                let t = t.to_ascii_lowercase();
                vec![
                    f32::from(u8::from(t.contains("message") || t.contains("ping"))),
                    f32::from(u8::from(t.contains("email") || t.contains("mail"))),
                    f32::from(u8::from(t.contains("file"))),
                ]
            })
            .collect())
    }
}

fn bag() -> Arc<Bag> {
    Arc::new(Bag {
        calls: AtomicUsize::new(0),
    })
}

const SIGNATURE: &str = "provider=bag;model=bag-v1;dims=3";

fn candidates() -> Vec<RankCandidate> {
    vec![
        RankCandidate::new("SLACK_SEND_MESSAGE", "send a message to a channel")
            .with_family("slack"),
        RankCandidate::new("GMAIL_SEND_EMAIL", "send an email").with_family("gmail"),
        RankCandidate::new("file_read", "read a file"),
    ]
}

#[tokio::test]
async fn vectors_round_trip_exactly_including_full_width_keys() {
    let cache = DriverEmbeddingCache::new(docs(&MemoryBackend::new(), "local"));
    let values = vec![
        (u64::MAX, vec![0.1_f32, -2.5, 3.0e-8]),
        (0, vec![1.0, 0.0, 0.0]),
        (1 << 63, vec![0.25, 0.5, 0.75]),
    ];
    cache.store(SIGNATURE, &values).await.unwrap();
    let keys: Vec<u64> = values.iter().map(|(key, _)| *key).chain([42]).collect();
    let found = cache.load(SIGNATURE, &keys).await.unwrap();
    assert_eq!(found.len(), 3, "a never-stored key is an ordinary miss");
    for (key, vector) in &values {
        assert_eq!(&found[key], vector, "f32 components survive JSON exactly");
    }
    // Repeating a store is idempotent and a disjoint delta merges.
    cache.store(SIGNATURE, &values).await.unwrap();
    cache
        .store(SIGNATURE, &[(7, vec![9.0, 9.0, 9.0])])
        .await
        .unwrap();
    assert_eq!(cache.load(SIGNATURE, &[7, 0]).await.unwrap().len(), 2);
}

#[tokio::test]
async fn signatures_and_scopes_never_share_vectors() {
    let storage = MemoryBackend::new();
    let alice = DriverEmbeddingCache::new(docs(&storage, "alice"));
    let bob = DriverEmbeddingCache::new(docs(&storage, "bob"));
    alice.store("space-a", &[(1, vec![1.0])]).await.unwrap();
    assert!(alice.load("space-b", &[1]).await.unwrap().is_empty());
    assert!(bob.load("space-a", &[1]).await.unwrap().is_empty());
    assert_eq!(alice.load("space-a", &[1]).await.unwrap().len(), 1);
}

#[tokio::test]
async fn non_finite_vectors_are_not_persisted() {
    let cache = DriverEmbeddingCache::new(docs(&MemoryBackend::new(), "local"));
    cache
        .store(
            SIGNATURE,
            &[
                (1, vec![f32::NAN, 0.0]),
                (2, vec![f32::INFINITY]),
                (3, vec![1.0]),
            ],
        )
        .await
        .unwrap();
    let found = cache.load(SIGNATURE, &[1, 2, 3]).await.unwrap();
    assert_eq!(found.keys().copied().collect::<Vec<_>>(), vec![3]);
}

#[tokio::test]
async fn a_ranker_warms_after_a_driver_reopen() {
    let storage = MemoryBackend::new();
    let first = bag();
    {
        let cache = Arc::new(DriverEmbeddingCache::new(docs(&storage, "local")));
        EmbeddingToolRanker::new(first.clone(), SIGNATURE)
            .with_cache(cache)
            .rank("ping", &RankContext::empty(), &candidates(), 1)
            .await
            .unwrap();
    }
    assert_eq!(first.calls.load(Ordering::SeqCst), 2, "catalogue + intent");

    let cache = Arc::new(DriverEmbeddingCache::new(docs(&storage, "local")));
    let warm = bag();
    let hits = EmbeddingToolRanker::new(warm.clone(), SIGNATURE)
        .with_cache(cache.clone())
        .rank("ping", &RankContext::empty(), &candidates(), 1)
        .await
        .unwrap();
    assert_eq!(hits[0].key, "SLACK_SEND_MESSAGE");
    assert_eq!(
        warm.calls.load(Ordering::SeqCst),
        1,
        "a warm cache embeds only the intent"
    );

    // A different embedding space re-embeds everything.
    let other = bag();
    EmbeddingToolRanker::new(other.clone(), "provider=bag;model=bag-v2;dims=3")
        .with_cache(cache)
        .rank("ping", &RankContext::empty(), &candidates(), 1)
        .await
        .unwrap();
    assert_eq!(other.calls.load(Ordering::SeqCst), 2);
}

/// The v2 shape: a `harness.embedding_cache` document under
/// `Key::of([signature, key as 16 hex digits])`, stamped schema 2.
#[tokio::test]
async fn vectors_are_harness_embedding_cache_documents_keyed_by_signature_and_key() {
    let documents = docs(&MemoryBackend::new(), "local");
    let cache = DriverEmbeddingCache::new(Arc::clone(&documents));
    cache
        .store("sig/with/slashes", &[(0xAB, vec![1.0, 2.0])])
        .await
        .unwrap();
    let row = documents
        .get(
            "harness.embedding_cache",
            "sig%2Fwith%2Fslashes/00000000000000ab",
        )
        .await
        .unwrap()
        .expect("the id is Key::of([signature, hex])");
    assert_eq!(
        row.doc,
        serde_json::json!({"sig": "sig/with/slashes", "key": "00000000000000ab", "vector": [1.0, 2.0]})
    );
    assert_eq!(row.meta.schema, 2);
    let found = cache.load("sig/with/slashes", &[0xAB]).await.unwrap();
    assert_eq!(found[&0xAB], vec![1.0, 2.0]);
}
