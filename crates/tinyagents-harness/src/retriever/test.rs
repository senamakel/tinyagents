use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;

use super::*;

#[derive(Default)]
struct RecordingRetriever {
    requests: Arc<Mutex<Vec<RetrievalRequest>>>,
}

#[async_trait]
impl Retriever for RecordingRetriever {
    async fn retrieve(&self, request: RetrievalRequest) -> crate::Result<Vec<RetrievedDocument>> {
        if request.cancellation.is_cancelled() {
            return Err(crate::TinyAgentsError::Cancelled);
        }
        self.requests.lock().unwrap().push(request.clone());
        Ok(vec![
            RetrievedDocument::new("first", "first result", 0.9)
                .with_metadata(json!({"source": "memory"})),
            RetrievedDocument::new("second", "second result", 0.2),
        ]
        .into_iter()
        .take(request.limit)
        .collect())
    }
}

#[tokio::test]
async fn retrieval_contract_propagates_cancellation() {
    let retriever = RecordingRetriever::default();
    let cancellation = crate::CancellationToken::new();
    cancellation.cancel();
    let result = retriever
        .retrieve(RetrievalRequest::new("query", 2).with_cancellation(cancellation))
        .await;
    assert!(matches!(result, Err(crate::TinyAgentsError::Cancelled)));
}

#[tokio::test]
async fn retrieval_contract_preserves_limit_score_metadata_and_request_shape() {
    let retriever = RecordingRetriever::default();
    let request =
        RetrievalRequest::new("find the answer", 1).with_metadata(json!({"collection": "scoped"}));

    let docs = retriever.retrieve(request).await.unwrap();

    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].id, "first");
    assert_eq!(docs[0].score, 0.9);
    assert_eq!(docs[0].metadata, json!({"source": "memory"}));
    let recorded = retriever.requests.lock().unwrap();
    assert_eq!(recorded[0].query, "find the answer");
    assert_eq!(recorded[0].metadata, json!({"collection": "scoped"}));
}
