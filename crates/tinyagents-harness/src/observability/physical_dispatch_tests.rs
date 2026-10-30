use super::*;

use futures::StreamExt;
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};
use tinyinference_llm::model::ModelResponse;
use tinyinference_llm::usage::Usage;

#[derive(Default)]
struct Recorder(Mutex<Vec<PhysicalDispatchOutcome>>);

struct Terminal(Arc<Recorder>);

impl PhysicalDispatchTerminal for Terminal {
    fn on_terminal(self: Box<Self>, outcome: PhysicalDispatchOutcome) {
        self.0.0.lock().unwrap().push(outcome);
    }
}

struct Observer(Arc<Recorder>);

impl PhysicalDispatchObserver for Observer {
    fn on_start(&self, _: &PhysicalDispatchMetadata) -> Box<dyn PhysicalDispatchTerminal> {
        Box::new(Terminal(Arc::clone(&self.0)))
    }
}

struct PendingModel;

#[async_trait::async_trait]
impl ChatModel<()> for PendingModel {
    async fn invoke(&self, _: &(), _: ModelRequest) -> tinyinference_llm::Result<ModelResponse> {
        std::future::pending().await
    }
}

#[tokio::test]
async fn dropping_an_inflight_provider_future_reports_one_cancelled_terminal() {
    let recorder = Arc::new(Recorder::default());
    let observer: Arc<dyn PhysicalDispatchObserver> = Arc::new(Observer(recorder.clone()));
    let model = Arc::new(ObservedChatModel::new(Arc::new(PendingModel), observer));
    let call = tokio::spawn(async move { model.invoke(&(), ModelRequest::new(Vec::new())).await });
    tokio::task::yield_now().await;
    call.abort();
    let _ = call.await;

    let outcomes = recorder.0.lock().unwrap();
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, PhysicalDispatchStatus::Cancelled);
}

struct FirstSuccessThenPending(AtomicUsize);

#[async_trait::async_trait]
impl ChatModel<()> for FirstSuccessThenPending {
    async fn invoke(&self, _: &(), _: ModelRequest) -> tinyinference_llm::Result<ModelResponse> {
        if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
            Ok(ModelResponse {
                usage: Some(Usage::new(2, 3)),
                ..ModelResponse::assistant("ok")
            })
        } else {
            std::future::pending().await
        }
    }
}

#[tokio::test]
async fn cancelling_a_later_attempt_keeps_the_first_attempt_usage() {
    let recorder = Arc::new(Recorder::default());
    let observer: Arc<dyn PhysicalDispatchObserver> = Arc::new(Observer(recorder.clone()));
    let model = Arc::new(ObservedChatModel::new(
        Arc::new(FirstSuccessThenPending(AtomicUsize::new(0))),
        observer,
    ));
    let first = model
        .invoke(&(), ModelRequest::new(Vec::new()))
        .await
        .unwrap();
    assert_eq!(first.usage.as_ref().unwrap().input_tokens, 2);

    let second_model = Arc::clone(&model);
    let second = tokio::spawn(async move {
        second_model
            .invoke(&(), ModelRequest::new(Vec::new()))
            .await
    });
    tokio::task::yield_now().await;
    second.abort();
    let _ = second.await;

    let outcomes = recorder.0.lock().unwrap();
    assert_eq!(outcomes.len(), 2);
    assert_eq!(outcomes[0].status, PhysicalDispatchStatus::Succeeded);
    assert_eq!(outcomes[0].usage.as_ref().unwrap().input_tokens, 2);
    assert_eq!(outcomes[1].status, PhysicalDispatchStatus::Cancelled);
}

struct StreamingModel;

struct EventSequenceModel(Mutex<Option<Vec<ModelStreamItem>>>);

#[async_trait::async_trait]
impl ChatModel<()> for EventSequenceModel {
    async fn invoke(&self, _: &(), _: ModelRequest) -> tinyinference_llm::Result<ModelResponse> {
        Ok(ModelResponse::assistant("unused"))
    }

    async fn stream(&self, _: &(), _: ModelRequest) -> tinyinference_llm::Result<ModelStream> {
        let items = self.0.lock().unwrap().take().expect("one stream call");
        Ok(ModelStream::new(Box::pin(futures::stream::iter(items))))
    }
}

fn event_sequence_model(items: Vec<ModelStreamItem>) -> Arc<dyn ChatModel<()>> {
    Arc::new(EventSequenceModel(Mutex::new(Some(items))))
}

struct PanicsAfterEofModel;

struct PanicsAfterEofStream(usize);

impl futures::Stream for PanicsAfterEofStream {
    type Item = ModelStreamItem;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let item = match this.0 {
            0 => Some(ModelStreamItem::Started),
            1 => Some(ModelStreamItem::UsageDelta(Usage::new(9, 4))),
            2 => None,
            _ => panic!("underlying provider stream polled after EOF"),
        };
        this.0 += 1;
        std::task::Poll::Ready(item)
    }
}

#[async_trait::async_trait]
impl ChatModel<()> for PanicsAfterEofModel {
    async fn invoke(&self, _: &(), _: ModelRequest) -> tinyinference_llm::Result<ModelResponse> {
        Ok(ModelResponse::assistant("unused"))
    }

    async fn stream(&self, _: &(), _: ModelRequest) -> tinyinference_llm::Result<ModelStream> {
        Ok(ModelStream::new(Box::pin(PanicsAfterEofStream(0))))
    }
}

#[async_trait::async_trait]
impl ChatModel<()> for StreamingModel {
    async fn invoke(&self, _: &(), _: ModelRequest) -> tinyinference_llm::Result<ModelResponse> {
        Ok(ModelResponse::assistant("invoke fallback"))
    }

    async fn stream(&self, _: &(), _: ModelRequest) -> tinyinference_llm::Result<ModelStream> {
        let items =
            futures::stream::iter([ModelStreamItem::Started]).chain(futures::stream::pending());
        Ok(ModelStream::new(Box::pin(items)))
    }
}

#[tokio::test]
async fn dropping_a_stream_after_a_chunk_reports_cancellation() {
    let recorder = Arc::new(Recorder::default());
    let observer: Arc<dyn PhysicalDispatchObserver> = Arc::new(Observer(recorder.clone()));
    let model = ObservedChatModel::new(Arc::new(StreamingModel), observer);
    let mut stream = model
        .stream(&(), ModelRequest::new(Vec::new()))
        .await
        .unwrap();
    assert!(matches!(
        stream.next().await,
        Some(ModelStreamItem::Started)
    ));
    drop(stream);

    let outcomes = recorder.0.lock().unwrap();
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, PhysicalDispatchStatus::Cancelled);
}

#[tokio::test]
async fn dispatch_stream_reports_success_only_after_completed_item() {
    let recorder = Arc::new(Recorder::default());
    let observer: Arc<dyn PhysicalDispatchObserver> = Arc::new(Observer(recorder.clone()));
    let response = ModelResponse {
        usage: Some(Usage::new(7, 4)),
        ..ModelResponse::assistant("done")
    };
    let model = ObservedChatModel::new(
        event_sequence_model(vec![
            ModelStreamItem::Started,
            ModelStreamItem::MessageDelta(Default::default()),
            ModelStreamItem::Completed(response),
        ]),
        observer,
    );
    let mut stream = model
        .stream(&(), ModelRequest::new(Vec::new()))
        .await
        .unwrap();

    assert!(matches!(
        stream.next().await,
        Some(ModelStreamItem::Started)
    ));
    assert!(recorder.0.lock().unwrap().is_empty());
    assert!(matches!(
        stream.next().await,
        Some(ModelStreamItem::MessageDelta(_))
    ));
    assert!(recorder.0.lock().unwrap().is_empty());
    assert!(matches!(
        stream.next().await,
        Some(ModelStreamItem::Completed(_))
    ));
    let outcomes = recorder.0.lock().unwrap();
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, PhysicalDispatchStatus::Succeeded);
    assert_eq!(outcomes[0].usage.as_ref().unwrap().input_tokens, 7);
}

#[tokio::test]
async fn dispatch_stream_failed_item_is_the_single_terminal() {
    let recorder = Arc::new(Recorder::default());
    let observer: Arc<dyn PhysicalDispatchObserver> = Arc::new(Observer(recorder.clone()));
    let model = ObservedChatModel::new(
        event_sequence_model(vec![
            ModelStreamItem::Started,
            ModelStreamItem::MessageDelta(Default::default()),
            ModelStreamItem::Failed("private failure".into()),
        ]),
        observer,
    );
    let mut stream = model
        .stream(&(), ModelRequest::new(Vec::new()))
        .await
        .unwrap();

    assert!(matches!(
        stream.next().await,
        Some(ModelStreamItem::Started)
    ));
    assert!(recorder.0.lock().unwrap().is_empty());
    assert!(matches!(
        stream.next().await,
        Some(ModelStreamItem::MessageDelta(_))
    ));
    assert!(recorder.0.lock().unwrap().is_empty());
    assert!(matches!(
        stream.next().await,
        Some(ModelStreamItem::Failed(_))
    ));
    let outcomes = recorder.0.lock().unwrap();
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, PhysicalDispatchStatus::Failed);
}

#[tokio::test]
async fn stream_failure_keeps_latest_cumulative_usage_delta() {
    let recorder = Arc::new(Recorder::default());
    let observer: Arc<dyn PhysicalDispatchObserver> = Arc::new(Observer(recorder.clone()));
    let model = ObservedChatModel::new(
        event_sequence_model(vec![
            ModelStreamItem::UsageDelta(Usage::new(5, 2)),
            ModelStreamItem::UsageDelta(Usage::new(9, 4)),
            ModelStreamItem::Failed("private failure".into()),
        ]),
        observer,
    );
    let mut stream = model
        .stream(&(), ModelRequest::new(Vec::new()))
        .await
        .unwrap();
    while stream.next().await.is_some() {}

    let outcomes = recorder.0.lock().unwrap();
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, PhysicalDispatchStatus::Failed);
    assert_eq!(outcomes[0].usage.as_ref().unwrap().input_tokens, 9);
    assert_eq!(outcomes[0].usage.as_ref().unwrap().output_tokens, 4);
}

#[tokio::test]
async fn cancelled_stream_keeps_latest_cumulative_usage_delta() {
    let recorder = Arc::new(Recorder::default());
    let observer: Arc<dyn PhysicalDispatchObserver> = Arc::new(Observer(recorder.clone()));
    let model = ObservedChatModel::new(
        event_sequence_model(vec![
            ModelStreamItem::UsageDelta(Usage::new(5, 2)),
            ModelStreamItem::UsageDelta(Usage::new(9, 4)),
            ModelStreamItem::MessageDelta(Default::default()),
        ]),
        observer,
    );
    let mut stream = model
        .stream(&(), ModelRequest::new(Vec::new()))
        .await
        .unwrap();
    assert!(matches!(
        stream.next().await,
        Some(ModelStreamItem::UsageDelta(_))
    ));
    assert!(matches!(
        stream.next().await,
        Some(ModelStreamItem::UsageDelta(_))
    ));
    drop(stream);

    let outcomes = recorder.0.lock().unwrap();
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, PhysicalDispatchStatus::Cancelled);
    assert_eq!(outcomes[0].usage.as_ref().unwrap().input_tokens, 9);
    assert_eq!(outcomes[0].usage.as_ref().unwrap().output_tokens, 4);
}

#[tokio::test]
async fn stream_eof_reports_failure_with_latest_cumulative_usage() {
    let recorder = Arc::new(Recorder::default());
    let observer: Arc<dyn PhysicalDispatchObserver> = Arc::new(Observer(recorder.clone()));
    let model = ObservedChatModel::new(Arc::new(PanicsAfterEofModel), observer);
    let mut stream = model
        .stream(&(), ModelRequest::new(Vec::new()))
        .await
        .unwrap();

    assert!(matches!(
        stream.next().await,
        Some(ModelStreamItem::Started)
    ));
    assert!(recorder.0.lock().unwrap().is_empty());
    assert!(matches!(
        stream.next().await,
        Some(ModelStreamItem::UsageDelta(usage)) if usage.input_tokens == 9
    ));
    assert!(recorder.0.lock().unwrap().is_empty());
    assert!(stream.next().await.is_none());
    let outcomes = recorder.0.lock().unwrap().clone();
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, PhysicalDispatchStatus::Failed);
    assert_eq!(outcomes[0].usage.as_ref().unwrap().input_tokens, 9);
    assert_eq!(outcomes[0].usage.as_ref().unwrap().output_tokens, 4);
    assert!(stream.next().await.is_none());
    drop(stream);

    let outcomes = recorder.0.lock().unwrap().clone();
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, PhysicalDispatchStatus::Failed);
    assert_eq!(outcomes[0].usage.as_ref().unwrap().input_tokens, 9);
    assert_eq!(outcomes[0].usage.as_ref().unwrap().output_tokens, 4);
}

#[test]
fn refusal_metadata_is_reported_without_including_provider_text() {
    let response = ModelResponse {
        raw: Some(serde_json::json!({
            "choices": [{"message": {"refusal": "private refusal text"}}]
        })),
        ..ModelResponse::assistant("private refusal text")
    };
    let outcome = dispatch_outcome(Some(&response), true);

    assert_eq!(outcome.status, PhysicalDispatchStatus::Refused);
    assert!(outcome.refused);
    assert!(!format!("{outcome:?}").contains("private refusal text"));
}

#[test]
fn malformed_managed_charge_blocks_provider_cost_fallback() {
    let response = ModelResponse {
        usage: Some(Usage::new(2, 3)),
        raw: Some(serde_json::json!({
            "openhuman_usage_meta": {"charged_amount_usd": "invalid"},
            "usage": {"cost": 0.000999}
        })),
        ..ModelResponse::assistant("ok")
    };

    let outcome = dispatch_outcome(Some(&response), true);

    assert_eq!(outcome.status, PhysicalDispatchStatus::Succeeded);
    let usage = outcome.usage.expect("response token usage is retained");
    assert_eq!((usage.input_tokens, usage.output_tokens), (2, 3));
    assert_eq!(usage.cost_usd, None);
}

#[test]
fn invalid_provider_cost_does_not_survive_charge_normalization() {
    for invalid_cost in [-0.001, 1.0e20] {
        let response = ModelResponse {
            usage: Some(Usage::new(2, 3)),
            raw: Some(serde_json::json!({"usage": {"cost": invalid_cost}})),
            ..ModelResponse::assistant("ok")
        };

        let outcome = dispatch_outcome(Some(&response), true);

        let usage = outcome.usage.expect("response token usage is retained");
        assert_eq!((usage.input_tokens, usage.output_tokens), (2, 3));
        assert_eq!(usage.cost_usd, None, "invalid provider cost {invalid_cost}");
    }
}

#[test]
fn finishing_a_dispatch_does_not_emit_a_second_cancelled_terminal_on_drop() {
    let recorder = Arc::new(Recorder::default());
    let observer: Arc<dyn PhysicalDispatchObserver> = Arc::new(Observer(recorder.clone()));
    let guard = PhysicalDispatchGuard::new(observer.on_start(&PhysicalDispatchMetadata::default()));
    guard.finish(PhysicalDispatchOutcome {
        status: PhysicalDispatchStatus::Succeeded,
        refused: false,
        duration: Duration::ZERO,
        usage: None,
        answered_model: None,
        finish_reason: None,
        served_by: None,
        response_id: None,
        first_output: None,
    });

    let outcomes = recorder.0.lock().unwrap();
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, PhysicalDispatchStatus::Succeeded);
}

#[tokio::test]
async fn dispatch_stream_reports_who_served_it_and_when_output_began() {
    let recorder = Arc::new(Recorder::default());
    let observer: Arc<dyn PhysicalDispatchObserver> = Arc::new(Observer(recorder.clone()));
    let mut response = ModelResponse::assistant("done");
    response.message.id = Some("gen-42".into());
    response.raw = Some(serde_json::json!({"provider": "DeepInfra"}));
    let model = ObservedChatModel::new(
        event_sequence_model(vec![
            ModelStreamItem::Started,
            ModelStreamItem::MessageDelta(tinyinference_llm::MessageDelta {
                reasoning: "thinking".into(),
                ..Default::default()
            }),
            ModelStreamItem::Completed(response),
        ]),
        observer,
    );
    let stream = model
        .stream(&(), ModelRequest::new(Vec::new()))
        .await
        .unwrap();
    let _: Vec<_> = stream.collect().await;

    let outcomes = recorder.0.lock().unwrap();
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].served_by.as_deref(), Some("DeepInfra"));
    assert_eq!(outcomes[0].response_id.as_deref(), Some("gen-42"));
    let first_output = outcomes[0].first_output.expect("reasoning is output");
    assert!(first_output <= outcomes[0].duration);
}

#[tokio::test]
async fn a_stream_without_output_reports_no_first_output() {
    let recorder = Arc::new(Recorder::default());
    let observer: Arc<dyn PhysicalDispatchObserver> = Arc::new(Observer(recorder.clone()));
    let model = ObservedChatModel::new(
        event_sequence_model(vec![
            ModelStreamItem::Started,
            ModelStreamItem::MessageDelta(Default::default()),
            ModelStreamItem::UsageDelta(Usage::new(1, 0)),
            ModelStreamItem::Completed(ModelResponse::assistant("")),
        ]),
        observer,
    );
    let stream = model
        .stream(&(), ModelRequest::new(Vec::new()))
        .await
        .unwrap();
    let _: Vec<_> = stream.collect().await;

    let outcomes = recorder.0.lock().unwrap();
    assert_eq!(outcomes[0].first_output, None);
    assert_eq!(outcomes[0].served_by, None);
    assert_eq!(outcomes[0].response_id, None);
}

#[test]
fn a_cancelled_dispatch_keeps_its_first_output_time() {
    let recorder = Arc::new(Recorder::default());
    let observer: Arc<dyn PhysicalDispatchObserver> = Arc::new(Observer(recorder.clone()));
    let mut guard =
        PhysicalDispatchGuard::new(observer.on_start(&PhysicalDispatchMetadata::default()));
    guard.note_output();
    let first = guard.first_output;
    guard.note_output();
    assert_eq!(guard.first_output, first, "the first time is kept");
    drop(guard);

    let outcomes = recorder.0.lock().unwrap();
    assert_eq!(outcomes[0].status, PhysicalDispatchStatus::Cancelled);
    assert_eq!(outcomes[0].first_output, first);
    assert!(first.is_some());
}
