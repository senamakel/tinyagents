//! Tests for how [`ContextCompressionMiddleware`] carries a compaction across
//! calls.
//!
//! The agent loop rebuilds every request from its own working transcript and
//! never sees the summary `before_model` splices in. These tests drive the
//! middleware the same way: each call gets the full, growing transcript.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::context::{RunConfig, RunContext};
use crate::error::Result;
use crate::middleware::ContextCompressionMiddleware;
use crate::middleware::{Middleware, MiddlewareStack};
use crate::summarization::{
    CompactionRecord, CompactionSink, CompressionProvenance, SummarizationPolicy, Summarizer,
    SummaryRecord, SummaryRequest,
};
use tinyinference_llm::message::{ContentBlock, Message, UserMessage};
use tinyinference_llm::model::ModelRequest;

fn ctx() -> RunContext {
    RunContext::new(RunConfig::new("test-run"), ())
}

fn user(text: &str) -> Message {
    Message::User(UserMessage {
        content: vec![ContentBlock::Text(text.to_string())],
    })
}

/// ~60 estimated tokens (chars / 4) tagged with `tag`.
fn chunk(tag: &str) -> Message {
    user(&format!("{tag}:{}", "x".repeat(236)))
}

/// The checkpoint the middleware writes for `summary` (default user placement).
fn cp(summary: &str) -> Message {
    crate::summarization::checkpoint_message(crate::summarization::SummaryPlacement::User, summary)
}

/// Answers every request with a short summary naming how many requests it has
/// seen, and records each request.
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

/// The middleware under test, what its summarizer and sink saw, and its context.
struct Fixture {
    stack: MiddlewareStack<()>,
    seen: Arc<Mutex<Vec<SummaryRequest>>>,
    sink: Arc<RecordingSink>,
    c: RunContext,
}

/// A 300-token window at 0.5 → a 150-token trigger, keeping the newest message.
/// Roomy enough that the checkpoint marker plus one kept chunk stays under it.
fn fixture() -> Fixture {
    let policy = SummarizationPolicy {
        keep_last: 1,
        ..SummarizationPolicy::default()
    }
    .with_context_window(300)
    .with_threshold_fraction(0.5);
    let summarizer = ShortSummarizer::default();
    let seen = summarizer.seen.clone();
    let mw: Arc<dyn Middleware<()>> = Arc::new(ContextCompressionMiddleware::with_summarizer(
        policy,
        Box::new(summarizer),
    ));
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw);
    let sink = Arc::new(RecordingSink::default());
    let c = ctx().with_compaction_sink(sink.clone());
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

#[tokio::test]
async fn reapplies_the_fold_instead_of_recompacting_every_call() {
    let Fixture {
        stack,
        seen,
        sink,
        mut c,
    } = fixture();
    let mut transcript = vec![chunk("m1"), chunk("m2"), chunk("m3")];

    // ~180 tokens: over the 150-token trigger, so the first call compacts m1, m2.
    let sent = send(&stack, &mut c, &transcript).await;
    assert_eq!(seen.lock().unwrap().len(), 1);
    assert_eq!(sent, vec![cp("summary #1"), chunk("m3")]);

    // The loop's transcript still holds m1 and m2 (it never saw the summary)
    // and grows by a small message. The fold is re-applied, so the request
    // stays small and nothing is summarized again.
    transcript.push(user("ok"));
    let sent = send(&stack, &mut c, &transcript).await;
    assert_eq!(seen.lock().unwrap().len(), 1, "no second summarizer call");
    assert_eq!(sent, vec![cp("summary #1"), chunk("m3"), user("ok")]);
    assert_eq!(sink.records.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn compacts_only_history_newer_than_the_fold() {
    let Fixture {
        stack,
        seen,
        sink,
        mut c,
    } = fixture();
    let mut transcript = vec![chunk("m1"), chunk("m2"), chunk("m3")];
    send(&stack, &mut c, &transcript).await;

    // Grow past the trigger again: the next compaction must summarize only
    // what came after m2, building on the first summary rather than re-reading
    // m1 and m2.
    transcript.extend([chunk("m4"), chunk("m5")]);
    let sent = send(&stack, &mut c, &transcript).await;

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[1].messages, vec![chunk("m3"), chunk("m4")]);
    assert_eq!(seen[1].previous_summary.as_deref(), Some("summary #1"));
    // The new summary replaces the one it was built on.
    assert_eq!(sent, vec![cp("summary #2"), chunk("m5")]);

    // Persisted boundaries are positions in the live transcript, which is
    // what a session-backed sink maps to entry ids: m3, then m5.
    let firsts: Vec<usize> = sink
        .records
        .lock()
        .unwrap()
        .iter()
        .map(|r| r.first_kept_index)
        .collect();
    assert_eq!(firsts, vec![2, 4]);
}

#[tokio::test]
async fn drops_the_fold_when_the_transcript_no_longer_matches() {
    let Fixture {
        stack,
        seen,
        sink: _sink,
        mut c,
    } = fixture();
    send(&stack, &mut c, &[chunk("m1"), chunk("m2"), chunk("m3")]).await;

    // A different history (rewritten or replaced): splicing the old summary
    // over it would be wrong, so it is compacted from scratch.
    let other = vec![chunk("n1"), chunk("n2"), chunk("n3")];
    let sent = send(&stack, &mut c, &other).await;

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[1].messages, vec![chunk("n1"), chunk("n2")]);
    // The old summary describes some other history: it is not handed on.
    assert_eq!(seen[1].previous_summary, None);
    assert_eq!(sent, vec![cp("summary #2"), chunk("n3")]);
}

#[tokio::test]
async fn replaces_a_summary_the_host_spliced_in_itself() {
    let Fixture {
        stack,
        seen,
        sink,
        mut c,
    } = fixture();
    let first = send(&stack, &mut c, &[chunk("m1"), chunk("m2"), chunk("m3")]).await;

    // A host that persists the compacted request as its transcript: the fold
    // no longer matches, but its summary is right there. It must be built on
    // and replaced, not kept beside the new one.
    let mut transcript = first.clone();
    transcript.extend([chunk("m4"), chunk("m5")]);
    let sent = send(&stack, &mut c, &transcript).await;

    {
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[1].previous_summary.as_deref(), Some("summary #1"));
    }
    assert_eq!(sent, vec![cp("summary #2"), chunk("m5")]);
    // The host's transcript has its own coordinates now: no boundary in them
    // is persisted.
    assert_eq!(sink.records.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn keeps_recognizing_a_host_spliced_summary_until_it_is_replaced() {
    let Fixture {
        stack,
        seen,
        sink: _sink,
        mut c,
    } = fixture();
    let first = send(&stack, &mut c, &[chunk("m1"), chunk("m2"), chunk("m3")]).await;

    // The host feeds the compacted request back, first below the threshold...
    let mut transcript = first.clone();
    transcript.push(user("ok"));
    let sent = send(&stack, &mut c, &transcript).await;
    assert_eq!(
        sent, transcript,
        "below the threshold the request is left alone"
    );

    // ...then over it. The old summary is still recognized and replaced.
    transcript.extend([chunk("m4"), chunk("m5")]);
    let sent = send(&stack, &mut c, &transcript).await;
    {
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[1].previous_summary.as_deref(), Some("summary #1"));
    }
    assert_eq!(sent, vec![cp("summary #2"), chunk("m5")]);

    // Until the host persists summary #2, it still sends summary #1. The
    // replacement fold must remove that obsolete host summary on reapply.
    let reapplied = send(&stack, &mut c, &transcript).await;
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2, "the replacement fold is reused");
    assert_eq!(reapplied, vec![cp("summary #2"), chunk("m5")]);
}

/// [`keeps_recognizing_a_host_spliced_summary_until_it_is_replaced`] with the
/// opt-in system placement: the host's summary sits in `system`, is lifted out
/// and remembered as the one the next fold replaces.
#[tokio::test]
async fn keeps_recognizing_a_host_spliced_system_summary_until_it_is_replaced() {
    let policy = SummarizationPolicy {
        keep_last: 1,
        ..SummarizationPolicy::default()
    }
    .with_context_window(300)
    .with_threshold_fraction(0.5);
    let summarizer = ShortSummarizer::default();
    let seen = summarizer.seen.clone();
    let mw: Arc<dyn Middleware<()>> = Arc::new(
        ContextCompressionMiddleware::with_summarizer(policy, Box::new(summarizer))
            .with_summary_placement(crate::summarization::SummaryPlacement::System),
    );
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw);
    let sink = Arc::new(RecordingSink::default());
    let mut c = ctx().with_compaction_sink(sink.clone());
    let sys_cp = |summary: &str| {
        crate::summarization::checkpoint_message(
            crate::summarization::SummaryPlacement::System,
            summary,
        )
    };

    let first = send(&stack, &mut c, &[chunk("m1"), chunk("m2"), chunk("m3")]).await;
    assert_eq!(first, vec![sys_cp("summary #1"), chunk("m3")]);

    let mut transcript = first.clone();
    transcript.push(user("ok"));
    let sent = send(&stack, &mut c, &transcript).await;
    assert_eq!(
        sent, transcript,
        "below the threshold the request is left alone"
    );

    transcript.extend([chunk("m4"), chunk("m5")]);
    let sent = send(&stack, &mut c, &transcript).await;
    {
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[1].previous_summary.as_deref(), Some("summary #1"));
    }
    assert_eq!(sent, vec![sys_cp("summary #2"), chunk("m5")]);

    let reapplied = send(&stack, &mut c, &transcript).await;
    assert_eq!(
        seen.lock().unwrap().len(),
        2,
        "the replacement fold is reused"
    );
    assert_eq!(reapplied, vec![sys_cp("summary #2"), chunk("m5")]);
    assert_eq!(sink.records.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn keeps_each_runs_fold_separate() {
    let Fixture {
        stack,
        seen,
        sink: _sink,
        c: mut run_a,
    } = fixture();
    let sink = Arc::new(RecordingSink::default());
    // Independent contexts may share a caller-supplied RunId; their folds
    // must still remain isolated by context instance.
    let mut run_b = RunContext::new(RunConfig::new("test-run"), ()).with_compaction_sink(sink);

    let mut a = vec![chunk("a1"), chunk("a2"), chunk("a3")];
    send(&stack, &mut run_a, &a).await;
    // Run B, on the same middleware, has its own history and its own fold.
    let sent_b = send(&stack, &mut run_b, &[chunk("b1"), chunk("b2"), chunk("b3")]).await;
    assert_eq!(sent_b, vec![cp("summary #2"), chunk("b3")]);
    assert_eq!(seen.lock().unwrap()[1].previous_summary, None);

    // Run A's fold survived B and still applies, with no new summarizer call.
    a.push(user("ok"));
    let sent_a = send(&stack, &mut run_a, &a).await;
    assert_eq!(sent_a, vec![cp("summary #1"), chunk("a3"), user("ok")]);
    assert_eq!(seen.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn keeps_concurrent_contexts_with_the_same_run_id_separate() {
    // A `RunId` is a caller label: two live invocations may share one. Their
    // folds must still not mix.
    let Fixture {
        stack,
        seen,
        sink: _sink,
        c: mut first,
    } = fixture();
    let mut second = ctx();
    assert_eq!(first.run_id(), second.run_id());

    let mut a = vec![chunk("a1"), chunk("a2"), chunk("a3")];
    send(&stack, &mut first, &a).await;
    let sent_b = send(
        &stack,
        &mut second,
        &[chunk("b1"), chunk("b2"), chunk("b3")],
    )
    .await;
    assert_eq!(sent_b, vec![cp("summary #2"), chunk("b3")]);
    assert_eq!(seen.lock().unwrap()[1].previous_summary, None);

    // The second context finishing must not erase the first one's fold.
    stack
        .run_after_agent(&mut second, &(), &mut crate::middleware::AgentRun::new())
        .await
        .unwrap();
    a.push(user("ok"));
    let sent_a = send(&stack, &mut first, &a).await;
    assert_eq!(sent_a, vec![cp("summary #1"), chunk("a3"), user("ok")]);
    assert_eq!(seen.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn forgets_a_runs_fold_when_the_run_ends() {
    let Fixture {
        stack,
        seen,
        sink: _sink,
        mut c,
    } = fixture();
    let transcript = vec![chunk("m1"), chunk("m2"), chunk("m3")];
    send(&stack, &mut c, &transcript).await;
    stack
        .run_after_agent(&mut c, &(), &mut crate::middleware::AgentRun::new())
        .await
        .unwrap();

    // Nothing carried over: the same transcript is compacted afresh.
    send(&stack, &mut c, &transcript).await;
    assert_eq!(seen.lock().unwrap().len(), 2);
}

// ── Overflow path ────────────────────────────────────────────────────────────

/// Fails its first call with a classified context overflow, then answers.
struct OverflowOnce {
    calls: Mutex<usize>,
}

impl crate::middleware::ModelBaseCall<(), ()> for OverflowOnce {
    fn call<'a>(
        &'a self,
        _ctx: &'a mut RunContext,
        _state: &'a (),
        _request: ModelRequest,
    ) -> crate::middleware::BoxModelFuture<'a> {
        Box::pin(async move {
            let first = {
                let mut calls = self.calls.lock().unwrap();
                *calls += 1;
                *calls == 1
            };
            if first {
                return Err(crate::error::TinyAgentsError::Model(
                    "This model's maximum context length is 100 tokens. However, your \
                     messages resulted in 900 tokens."
                        .to_string(),
                ));
            }
            Ok(tinyinference_llm::model::ModelResponse::assistant(
                "recovered",
            ))
        })
    }
}

/// A later `before_model` step that alters the oldest non-system message.
#[derive(Clone, Copy)]
enum LaterStep {
    /// Drops it, as a trim step would.
    DropOldest,
    /// Rewrites it in place, as microcompact blanking a tool body would.
    RewriteOldest,
}

#[async_trait]
impl Middleware<()> for LaterStep {
    fn name(&self) -> &str {
        "later_step"
    }

    async fn before_model(
        &self,
        _ctx: &mut RunContext,
        _state: &(),
        request: &mut ModelRequest,
    ) -> Result<()> {
        if let Some(at) = request
            .messages
            .iter()
            .position(|m| !matches!(m, Message::System(_)))
        {
            match self {
                LaterStep::DropOldest => {
                    request.messages.remove(at);
                }
                LaterStep::RewriteOldest => request.messages[at] = user("[cleared]"),
            }
        }
        Ok(())
    }
}

/// Runs one call through `before_model` and the overflow-recovering model
/// wrap, with threshold compaction declined so only the overflow path fires.
async fn overflow_call(later: Option<LaterStep>) -> Vec<CompactionRecord> {
    let mw = Arc::new(
        ContextCompressionMiddleware::new(
            SummarizationPolicy::default()
                .with_context_window(100)
                .with_threshold_fraction(0.5),
        )
        .with_before_compaction(|c| match c.reason {
            crate::summarization::CompactionReason::Threshold => {
                crate::summarization::CompactionDecision::Decline
            }
            _ => crate::summarization::CompactionDecision::Proceed,
        }),
    );
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());
    if let Some(step) = later {
        stack.push(Arc::new(step));
    }
    stack.push_model_middleware(mw);
    let sink = Arc::new(RecordingSink::default());
    let mut c = ctx().with_compaction_sink(sink.clone());

    let big = "word ".repeat(60);
    let mut request = ModelRequest {
        messages: (1..=5).map(|i| user(&format!("{i} {big}"))).collect(),
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();
    let base = OverflowOnce {
        calls: Mutex::new(0),
    };
    let response = stack
        .run_wrapped_model(&mut c, &(), request, &base)
        .await
        .unwrap()
        .into_response();
    assert_eq!(response.text(), "recovered");
    sink.records.lock().unwrap().clone()
}

#[tokio::test]
async fn overflow_persists_a_boundary_when_the_request_is_aligned() {
    let persisted = overflow_call(None).await;
    assert_eq!(persisted.len(), 1);
    assert_eq!(
        persisted[0].reason,
        crate::summarization::CompactionReason::Overflow
    );
    assert!(persisted[0].first_kept_index > 0);
}

#[tokio::test]
async fn overflow_skips_persistence_when_a_later_step_dropped_messages() {
    // The boundary would be shifted by the dropped message; a resumed session
    // would restore or duplicate the wrong history from it.
    assert!(overflow_call(Some(LaterStep::DropOldest)).await.is_empty());
}

#[tokio::test]
async fn overflow_skips_persistence_when_a_later_step_rewrote_a_message() {
    // Same count, different content: the summary would describe the rewritten
    // placeholder, not what the persisted boundary would fold away.
    assert!(
        overflow_call(Some(LaterStep::RewriteOldest))
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn keeps_system_prompts_ahead_of_the_reapplied_summary() {
    let Fixture {
        stack,
        seen,
        sink: _sink,
        mut c,
    } = fixture();
    let system = Message::system("You are a coding agent.");
    let mut transcript = vec![system.clone(), chunk("m1"), chunk("m2"), chunk("m3")];
    send(&stack, &mut c, &transcript).await;

    transcript.push(user("ok"));
    let sent = send(&stack, &mut c, &transcript).await;
    assert_eq!(seen.lock().unwrap().len(), 1);
    assert_eq!(
        sent,
        vec![system, cp("summary #1"), chunk("m3"), user("ok"),]
    );
}

#[tokio::test]
async fn concat_summarizer_carries_the_previous_summary_forward() {
    let record = crate::summarization::ConcatSummarizer
        .summarize_request(&SummaryRequest {
            messages: vec![user("new")],
            previous_summary: Some("earlier".into()),
        })
        .await
        .unwrap();
    let text = record.summary.text();
    assert!(text.starts_with("earlier\n"), "{text}");
    assert!(text.contains("new"), "{text}");
}

#[tokio::test]
async fn an_unaligned_overflow_summary_is_not_built_on_later() {
    // Threshold compaction is declined until the overflow has happened, so the
    // overflow path fires first, on a request a later step rewrote.
    let allow_threshold = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let gate = allow_threshold.clone();
    let summarizer = ShortSummarizer::default();
    let seen = summarizer.seen.clone();
    let mw = Arc::new(
        ContextCompressionMiddleware::with_summarizer(
            SummarizationPolicy {
                keep_last: 1,
                ..SummarizationPolicy::default()
            }
            .with_context_window(100)
            .with_threshold_fraction(0.5),
            Box::new(summarizer),
        )
        .with_before_compaction(move |c| match c.reason {
            crate::summarization::CompactionReason::Threshold
                if !gate.load(std::sync::atomic::Ordering::SeqCst) =>
            {
                crate::summarization::CompactionDecision::Decline
            }
            _ => crate::summarization::CompactionDecision::Proceed,
        }),
    );
    let mut stack: MiddlewareStack<()> = MiddlewareStack::new();
    stack.push(mw.clone());
    stack.push(Arc::new(LaterStep::RewriteOldest));
    stack.push_model_middleware(mw);
    let mut c = ctx();

    let big = "word ".repeat(60);
    let transcript: Vec<Message> = (1..=5).map(|i| user(&format!("{i} {big}"))).collect();
    let mut request = ModelRequest {
        messages: transcript.clone(),
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();
    let base = OverflowOnce {
        calls: Mutex::new(0),
    };
    stack
        .run_wrapped_model(&mut c, &(), request, &base)
        .await
        .unwrap();
    assert_eq!(seen.lock().unwrap().len(), 1, "the overflow compaction ran");

    // The next threshold compaction summarizes the original transcript from
    // scratch; the overflow's summary of the rewritten request is not its base.
    allow_threshold.store(true, std::sync::atomic::Ordering::SeqCst);
    let mut request = ModelRequest {
        messages: transcript,
        ..Default::default()
    };
    stack
        .run_before_model(&mut c, &(), &mut request)
        .await
        .unwrap();
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[1].previous_summary, None);
}
