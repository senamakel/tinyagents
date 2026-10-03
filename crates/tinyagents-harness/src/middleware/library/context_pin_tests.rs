//! Tests for [`ContextCompressionMiddleware`] with
//! [`SummarizationPolicy::pin_turn_user_message`] (issue
//! tinyhumansai/openhuman#6960).
//!
//! The pinned user message sits out of the middle of the folded range: the
//! messages before and after it are summarized, it stays verbatim. These check
//! that the fold re-applies it, that later compactions build on it, and that
//! the persisted [`CompactionRecord`] says where it is.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::context::{RunConfig, RunContext};
use crate::error::Result;
use crate::middleware::{ContextCompressionMiddleware, Middleware, MiddlewareStack};
use crate::summarization::{
    CompactionDecision, CompactionReason, CompactionRecord, CompactionSink, CompressionProvenance,
    SummarizationPolicy, Summarizer, SummaryPlacement, SummaryRecord, SummaryRequest,
    checkpoint_message,
};
use tinyinference_llm::message::Message;
use tinyinference_llm::model::{ModelRequest, ModelResponse};

/// ~30 estimated tokens of assistant work tagged with `tag`.
fn step(tag: &str) -> Message {
    Message::assistant(format!("{tag}:{}", "x".repeat(116)))
}

fn placed_summary(n: u32) -> Message {
    checkpoint_message(SummaryPlacement::User, &format!("summary #{n}"))
}

fn task() -> Message {
    Message::user("write the report")
}

/// Answers with `summary #<n>` and records each request.
#[derive(Default)]
struct ShortSummarizer {
    seen: Arc<Mutex<Vec<SummaryRequest>>>,
}

#[async_trait]
impl Summarizer for ShortSummarizer {
    async fn summarize(&self, messages: &[Message]) -> Result<SummaryRecord> {
        self.summarize_request(&SummaryRequest::new(messages.to_vec()))
            .await
    }

    async fn summarize_request(&self, request: &SummaryRequest) -> Result<SummaryRecord> {
        let mut seen = self.seen.lock().unwrap();
        seen.push(request.clone());
        Ok(SummaryRecord {
            summary: Message::system(format!("summary #{}", seen.len())),
            provenance: CompressionProvenance {
                source_ids: Vec::new(),
                original_token_estimate: 0,
                summary_token_estimate: 0,
                reason: "test".into(),
            },
            usage: None,
        })
    }
}

#[derive(Default)]
struct RecordingSink {
    records: Mutex<Vec<CompactionRecord>>,
}

impl CompactionSink for RecordingSink {
    fn persist(&self, record: &CompactionRecord) -> Result<()> {
        self.records.lock().unwrap().push(record.clone());
        Ok(())
    }
}

struct Fixture {
    stack: MiddlewareStack<()>,
    seen: Arc<Mutex<Vec<SummaryRequest>>>,
    sink: Arc<RecordingSink>,
    c: RunContext,
}

/// A 50-token trigger keeping the newest message, with pinning on.
/// `decline_threshold_after` declines threshold compactions once that many
/// have been offered, so a later call reaches the overflow path.
fn fixture(decline_threshold_after: Option<usize>) -> Fixture {
    let policy = SummarizationPolicy {
        keep_last: 1,
        pin_turn_user_message: true,
        ..SummarizationPolicy::default()
    }
    .with_context_window(170)
    .with_threshold_fraction(0.5);
    let summarizer = ShortSummarizer::default();
    let seen = summarizer.seen.clone();
    let offered = Mutex::new(0usize);
    let mw = Arc::new(
        ContextCompressionMiddleware::with_summarizer(policy, Box::new(summarizer))
            .with_before_compaction(move |c| {
                if c.reason != CompactionReason::Threshold {
                    return CompactionDecision::Proceed;
                }
                let mut offered = offered.lock().unwrap();
                *offered += 1;
                match decline_threshold_after {
                    Some(limit) if *offered > limit => CompactionDecision::Decline,
                    _ => CompactionDecision::Proceed,
                }
            }),
    );
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone() as Arc<dyn Middleware<()>>);
    stack.push_model_middleware(mw);
    let sink = Arc::new(RecordingSink::default());
    let c = RunContext::new(RunConfig::new("pin-run"), ()).with_compaction_sink(sink.clone());
    Fixture {
        stack,
        seen,
        sink,
        c,
    }
}

async fn send(
    stack: &MiddlewareStack<()>,
    c: &mut RunContext,
    transcript: &[Message],
) -> Vec<Message> {
    let mut request = ModelRequest {
        messages: transcript.to_vec(),
        ..Default::default()
    };
    stack.run_before_model(c, &(), &mut request).await.unwrap();
    request.messages
}

fn pinned_indexes(sink: &RecordingSink) -> Vec<Option<u64>> {
    sink.records
        .lock()
        .unwrap()
        .iter()
        .map(|r| r.details.get("pinned_user_index").and_then(|v| v.as_u64()))
        .collect()
}

fn first_kept(sink: &RecordingSink) -> Vec<usize> {
    sink.records
        .lock()
        .unwrap()
        .iter()
        .map(|r| r.first_kept_index)
        .collect()
}

#[tokio::test]
async fn a_threshold_compaction_keeps_the_turn_user_message_verbatim() {
    let Fixture {
        stack,
        seen,
        sink,
        mut c,
    } = fixture(None);
    let transcript = vec![task(), step("a1"), step("a2"), step("a3")];

    let sent = send(&stack, &mut c, &transcript).await;

    assert_eq!(
        sent,
        vec![placed_summary(1), task(), step("a3")],
        "the assignment must follow the summary verbatim"
    );
    assert_eq!(
        seen.lock().unwrap()[0].messages,
        vec![step("a1"), step("a2")]
    );
    // The tail starts at a3 (live index 3); the pinned task is live index 0.
    assert_eq!(first_kept(&sink), vec![3]);
    assert_eq!(pinned_indexes(&sink), vec![Some(0)]);
}

#[tokio::test]
async fn finished_run_carries_the_pinned_message_into_compacted_history() {
    let Fixture { stack, mut c, .. } = fixture(None);
    let transcript = vec![task(), step("a1"), step("a2"), step("a3")];
    send(&stack, &mut c, &transcript).await;

    let mut run = crate::middleware::AgentRun::new();
    run.messages = transcript;
    run.messages.push(Message::assistant("done"));
    stack.run_after_agent(&mut c, &(), &mut run).await.unwrap();

    assert_eq!(
        run.compacted_history.expect("compacted history"),
        vec![
            placed_summary(1),
            task(),
            step("a3"),
            Message::assistant("done")
        ]
    );
}

#[tokio::test]
async fn the_fold_reapplies_the_pinned_message() {
    let Fixture {
        stack,
        seen,
        sink: _sink,
        mut c,
    } = fixture(None);
    let mut transcript = vec![task(), step("a1"), step("a2"), step("a3")];
    send(&stack, &mut c, &transcript).await;

    transcript.push(Message::assistant("ok"));
    let sent = send(&stack, &mut c, &transcript).await;

    assert_eq!(seen.lock().unwrap().len(), 1, "no second summarizer call");
    assert_eq!(
        sent,
        vec![
            placed_summary(1),
            task(),
            step("a3"),
            Message::assistant("ok")
        ]
    );
}

#[tokio::test]
async fn a_second_compaction_keeps_the_pin_and_summarizes_only_new_history() {
    let Fixture {
        stack,
        seen,
        sink,
        mut c,
    } = fixture(None);
    let mut transcript = vec![task(), step("a1"), step("a2"), step("a3")];
    send(&stack, &mut c, &transcript).await;

    transcript.extend([step("a4"), step("a5")]);
    let sent = send(&stack, &mut c, &transcript).await;

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[1].messages, vec![step("a3"), step("a4")]);
    assert_eq!(seen[1].previous_summary.as_deref(), Some("summary #1"));
    assert_eq!(sent, vec![placed_summary(2), task(), step("a5")]);
    assert_eq!(first_kept(&sink), vec![3, 5]);
    assert_eq!(pinned_indexes(&sink), vec![Some(0), Some(0)]);
}

#[tokio::test]
async fn a_later_user_message_takes_over_the_pin() {
    let Fixture {
        stack,
        seen,
        sink,
        mut c,
    } = fixture(None);
    let mut transcript = vec![task(), step("a1"), step("a2"), step("a3")];
    send(&stack, &mut c, &transcript).await;

    let change = Message::user("use the 2025 numbers instead");
    transcript.extend([change.clone(), step("a4"), step("a5")]);
    let sent = send(&stack, &mut c, &transcript).await;

    // The original task is folded now; the newer instruction is pinned.
    assert_eq!(
        seen.lock().unwrap()[1].messages,
        vec![task(), step("a3"), step("a4")]
    );
    assert_eq!(sent, vec![placed_summary(2), change, step("a5")]);
    // live: task 0, a1 1, a2 2, a3 3, change 4, a4 5, a5 6
    assert_eq!(first_kept(&sink), vec![3, 6]);
    assert_eq!(pinned_indexes(&sink), vec![Some(0), Some(4)]);
}

/// Fails its first call with a classified context overflow, then answers and
/// records the retried request.
#[derive(Default)]
struct OverflowOnce {
    calls: Mutex<Vec<ModelRequest>>,
}

impl crate::middleware::ModelBaseCall<(), ()> for OverflowOnce {
    fn call<'a>(
        &'a self,
        _ctx: &'a mut RunContext,
        _state: &'a (),
        request: ModelRequest,
    ) -> crate::middleware::BoxModelFuture<'a> {
        Box::pin(async move {
            let first = {
                let mut calls = self.calls.lock().unwrap();
                calls.push(request);
                calls.len() == 1
            };
            if first {
                return Err(crate::error::TinyAgentsError::Model(
                    "This model's maximum context length is 100 tokens. However, your \
                     messages resulted in 900 tokens."
                        .to_string(),
                ));
            }
            Ok(ModelResponse::assistant("recovered"))
        })
    }
}

#[tokio::test]
async fn an_overflow_after_a_pinned_fold_extends_it_and_keeps_the_pin() {
    let Fixture {
        stack,
        seen,
        sink,
        mut c,
    } = fixture(Some(1));
    let mut transcript = vec![task(), step("a1"), step("a2"), step("a3")];
    send(&stack, &mut c, &transcript).await;

    // The second threshold compaction is declined, so the overflow path does
    // the work, over a request that carries the pinned message after the fold.
    transcript.extend([step("a4"), step("a5")]);
    let request = ModelRequest {
        messages: send(&stack, &mut c, &transcript).await,
        ..Default::default()
    };
    assert_eq!(request.messages[1], task());
    let base = OverflowOnce::default();
    stack
        .run_wrapped_model(&mut c, &(), request, &base)
        .await
        .unwrap();

    let retried = base.calls.lock().unwrap()[1].messages.clone();
    assert_eq!(retried, vec![placed_summary(2), task(), step("a5")]);
    assert_eq!(
        seen.lock().unwrap()[1].messages,
        vec![step("a3"), step("a4")]
    );
    // The pinned message no longer misaligns the request with the live
    // transcript, so the overflow boundary is persisted too.
    assert_eq!(first_kept(&sink), vec![3, 5]);
    assert_eq!(pinned_indexes(&sink), vec![Some(0), Some(0)]);
}

struct FailingSummarizer;

#[async_trait]
impl Summarizer for FailingSummarizer {
    async fn summarize(&self, _messages: &[Message]) -> Result<SummaryRecord> {
        Err(crate::error::TinyAgentsError::Model(
            "summarizer down".into(),
        ))
    }
}

#[tokio::test]
async fn the_fallback_trim_does_not_front_drop_the_pinned_message() {
    let policy = SummarizationPolicy {
        keep_last: 1,
        pin_turn_user_message: true,
        ..SummarizationPolicy::default()
    }
    .with_context_window(170)
    .with_threshold_fraction(0.5);
    let mw = Arc::new(ContextCompressionMiddleware::with_summarizer(
        policy,
        Box::new(FailingSummarizer),
    ));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw as Arc<dyn Middleware<()>>);
    let mut c = RunContext::new(RunConfig::new("pin-run"), ());

    let transcript = vec![
        Message::system("sys"),
        task(),
        step("a1"),
        step("a2"),
        step("a3"),
    ];
    let sent = send(&stack, &mut c, &transcript).await;

    assert_eq!(sent.first(), Some(&Message::system("sys")));
    assert_eq!(
        sent.get(1),
        Some(&task()),
        "the front-drop must keep the turn's assignment: {sent:?}"
    );
    assert_eq!(sent.last(), Some(&step("a3")));
}
