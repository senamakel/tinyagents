//! [`ContextCompressionMiddleware`] driven through the real agent loop with a
//! scripted model: compaction happens once per crossing, later compactions
//! are incremental, the summary is a user-role checkpoint after the system
//! prompt, the run hands back its compacted history for the next turn, the
//! trigger follows provider usage, overflow recovers once, and the
//! anti-thrash guard stops paying for summaries that do not help.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;

use crate::context::{RunConfig, RunContext};
use crate::error::Result;
use crate::events::AgentEvent;
use crate::middleware::{AgentRun, ContextCompressionMiddleware};
use crate::runtime::AgentHarness;
use crate::summarization::{
    CompressionProvenance, SummarizationPolicy, Summarizer, SummaryPlacement, SummaryRecord,
    SummaryRequest, checkpoint_message, is_checkpoint,
};
use crate::testkit::{EventRecorder, FakeTool, ScriptedModel};
use tinyinference_llm::message::{AssistantMessage, Message};
use tinyinference_llm::model::{ChatModel, ModelRequest, ModelResponse};
use tinyinference_llm::tool::ToolCall;
use tinyinference_llm::usage::Usage;

/// Tool output of roughly 200 estimated tokens.
fn big_output() -> String {
    "r".repeat(800)
}

/// Records every request and answers with `summary #n`.
#[derive(Default)]
struct RecordingSummarizer {
    seen: Arc<Mutex<Vec<SummaryRequest>>>,
}

#[async_trait]
impl Summarizer for RecordingSummarizer {
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
            usage: Some(Usage::new(100, 10)),
        })
    }
}

fn tool_turn(id: &str) -> ModelResponse {
    let mut response = ModelResponse::assistant("");
    response.message = AssistantMessage {
        id: None,
        content: Vec::new(),
        tool_calls: vec![ToolCall::new(id, "read", json!({}))],
        usage: None,
        origin: None,
    };
    response
}

fn with_input_tokens(response: ModelResponse, input_tokens: u64) -> ModelResponse {
    response.with_usage(Usage::new(input_tokens, 5))
}

struct Fixture {
    harness: AgentHarness<()>,
    model: Arc<ScriptedModel>,
    mw: Arc<ContextCompressionMiddleware>,
    seen: Arc<Mutex<Vec<SummaryRequest>>>,
    recorder: EventRecorder,
}

fn fixture_with(
    responses: Vec<ModelResponse>,
    trigger: u64,
    configure: impl FnOnce(ContextCompressionMiddleware) -> ContextCompressionMiddleware,
) -> Fixture {
    let model = Arc::new(ScriptedModel::new(responses));
    let policy = SummarizationPolicy {
        keep_last: 2,
        ..SummarizationPolicy::default()
    }
    .with_trigger_override(trigger);
    let summarizer = RecordingSummarizer::default();
    let seen = summarizer.seen.clone();
    let mw = Arc::new(configure(ContextCompressionMiddleware::with_summarizer(
        policy,
        Box::new(summarizer),
    )));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::clone(&model) as _);
    harness.register_tool(Arc::new(FakeTool::returning("read", big_output())));
    harness.push_middleware(mw.clone());
    harness.push_model_middleware(mw.clone());
    Fixture {
        harness,
        model,
        mw,
        seen,
        recorder: EventRecorder::new(),
    }
}

fn fixture(responses: Vec<ModelResponse>, trigger: u64) -> Fixture {
    fixture_with(responses, trigger, |mw| mw)
}

impl Fixture {
    async fn run(&self, input: Vec<Message>) -> AgentRun {
        let ctx = RunContext::new(RunConfig::new("loop"), ()).with_events(self.recorder.sink());
        self.harness
            .invoke_in_context(&(), ctx, input)
            .await
            .expect("run succeeds")
    }

    fn summaries(&self) -> Vec<SummaryRequest> {
        self.seen.lock().unwrap().clone()
    }
}

fn task() -> Vec<Message> {
    vec![
        Message::system("You are a coding agent."),
        Message::user("fix the bug"),
    ]
}

fn cp(summary: &str) -> Message {
    checkpoint_message(SummaryPlacement::User, summary)
}

fn six_tool_turns_then_answer() -> Vec<ModelResponse> {
    let mut responses: Vec<ModelResponse> = (1..=6).map(|i| tool_turn(&format!("c{i}"))).collect();
    responses.push(ModelResponse::assistant("done"));
    responses
}

#[tokio::test]
async fn compacts_once_per_crossing_and_incrementally_through_the_loop() {
    let fx = fixture(six_tool_turns_then_answer(), 600);
    let run = fx.run(task()).await;

    let requests = fx.model.requests();
    assert_eq!(requests.len(), 7);

    // Two crossings over seven calls: no compaction on every call after the
    // first one.
    let summaries = fx.summaries();
    assert_eq!(summaries.len(), 2, "one summary per crossing");

    // The first crossing (call 4) folds the task and the first two rounds and
    // keeps the newest round; the summary is a user-role checkpoint right
    // after the system prompt.
    let first = &requests[3].messages;
    assert!(matches!(first[0], Message::System(_)));
    assert_eq!(first[1], cp("summary #1"));
    assert!(matches!(first[1], Message::User(_)));
    assert_eq!(first.len(), 4, "system, checkpoint, kept call and result");

    // Call 5 reuses the fold: same checkpoint, transcript grown by one round,
    // no summarizer call in between (still exactly one summary at that point
    // is implied by the second summary's input below).
    assert_eq!(requests[4].messages[1], cp("summary #1"));
    assert_eq!(requests[4].messages.len(), 6);

    // The second crossing summarizes only what came after the first fold,
    // building on the first summary instead of re-reading it as history.
    assert_eq!(summaries[0].previous_summary, None);
    assert_eq!(summaries[1].previous_summary.as_deref(), Some("summary #1"));
    assert_eq!(summaries[1].messages.len(), 4);
    assert!(summaries[1].messages.iter().all(|m| !is_checkpoint(m)));
    assert!(
        summaries[1]
            .messages
            .iter()
            .all(|m| m.text() != "fix the bug"),
        "the first fold's messages are not summarized again"
    );
    assert_eq!(requests[6].messages[1], cp("summary #2"));

    // The run keeps its full record and hands back the compacted transcript.
    assert_eq!(run.messages.len(), 2 + 12 + 1);
    let compacted = run.compacted_history.expect("compacted history");
    assert!(compacted.len() < run.messages.len());
    assert_eq!(compacted[0], Message::system("You are a coding agent."));
    assert_eq!(compacted[1], cp("summary #2"));
    assert_eq!(compacted.last().map(Message::text).as_deref(), Some("done"));
    assert_eq!(
        compacted.len(),
        2 + 4 + 1,
        "system, checkpoint, two rounds, answer"
    );
}

#[tokio::test]
async fn compaction_events_carry_summarizer_usage_and_latency() {
    let fx = fixture(six_tool_turns_then_answer(), 600);
    fx.run(task()).await;
    let compacted: Vec<_> = fx
        .recorder
        .events()
        .into_iter()
        .filter_map(|event| match event {
            AgentEvent::Compacted {
                usage, latency_ms, ..
            } => Some((usage, latency_ms)),
            _ => None,
        })
        .collect();
    assert_eq!(compacted.len(), 2);
    for (usage, latency_ms) in compacted {
        assert_eq!(usage.map(|u| u.input_tokens), Some(100));
        assert!(latency_ms.is_some());
    }
}

#[tokio::test]
async fn a_later_turn_refines_the_persisted_checkpoint_instead_of_resummarizing_it() {
    let first = fixture(six_tool_turns_then_answer(), 600);
    let run = first.run(task()).await;
    let mut history = run.compacted_history.expect("compacted history");
    history.push(Message::user("now add a test"));

    // A fresh middleware instance, as a host builds per turn.
    let second = fixture(six_tool_turns_then_answer(), 600);
    let run = second.run(history).await;

    let summaries = second.summaries();
    assert!(!summaries.is_empty(), "the second turn crosses again");
    assert_eq!(
        summaries[0].previous_summary.as_deref(),
        Some("summary #2"),
        "the persisted checkpoint is the previous summary"
    );
    assert!(summaries[0].messages.iter().all(|m| !is_checkpoint(m)));
    // Exactly one checkpoint is ever sent.
    for request in second.model.requests() {
        assert!(request.messages.iter().filter(|m| is_checkpoint(m)).count() <= 1);
    }
    let compacted = run.compacted_history.expect("compacted again");
    assert_eq!(
        compacted.iter().filter(|m| is_checkpoint(m)).count(),
        1,
        "the new checkpoint replaces the persisted one"
    );
}

#[tokio::test]
async fn a_turn_that_does_not_compact_reports_no_compacted_history() {
    let mut history = vec![
        Message::system("You are a coding agent."),
        cp("earlier work"),
        Message::user("hello"),
    ];
    let fx = fixture(vec![ModelResponse::assistant("hi")], 600);
    let run = fx.run(std::mem::take(&mut history)).await;
    assert!(fx.summaries().is_empty());
    assert!(run.compacted_history.is_none());
    assert_eq!(fx.model.requests()[0].messages[1], cp("earlier work"));
}

#[tokio::test]
async fn system_placement_stays_available() {
    let fx = fixture_with(six_tool_turns_then_answer(), 600, |mw| {
        mw.with_summary_placement(SummaryPlacement::System)
    });
    fx.run(task()).await;
    let request = &fx.model.requests()[3].messages;
    assert!(matches!(request[1], Message::System(_)));
    assert!(is_checkpoint(&request[1]));
    assert_eq!(fx.mw.summary_placement(), SummaryPlacement::System);
}

#[tokio::test]
async fn the_trigger_follows_reported_usage() {
    // The estimate of the second request is far under the trigger; the
    // provider says the first one was already over it.
    let responses = vec![
        with_input_tokens(tool_turn("c1"), 700),
        ModelResponse::assistant("done"),
    ];
    let fx = fixture(responses, 600);
    fx.run(task()).await;
    assert_eq!(fx.summaries().len(), 1, "usage over the trigger compacts");

    // Without usage the same transcript stays under the estimate trigger.
    let fx = fixture(vec![tool_turn("c1"), ModelResponse::assistant("done")], 600);
    fx.run(task()).await;
    assert!(fx.summaries().is_empty());
}

#[tokio::test]
async fn the_thrash_guard_suppresses_ineffective_compactions() {
    // Every call reports a prompt far over the trigger, so no compaction ever
    // helps.
    let responses = || {
        let mut responses: Vec<ModelResponse> = (1..=5)
            .map(|i| with_input_tokens(tool_turn(&format!("c{i}")), 5_000))
            .collect();
        responses.push(ModelResponse::assistant("done"));
        responses
    };

    let guarded = fixture_with(responses(), 600, |mw| mw.with_thrash_guard(2, 3));
    guarded.run(task()).await;
    assert_eq!(
        guarded.summaries().len(),
        2,
        "two strikes, then the cooldown trims instead of summarizing"
    );
    let trims = guarded
        .recorder
        .events()
        .into_iter()
        .filter(|event| matches!(event, AgentEvent::Compressed { .. }))
        .count();
    assert!(trims > 2, "the suppressed calls trim deterministically");

    let unguarded = fixture_with(responses(), 600, |mw| mw.with_thrash_guard(0, 0));
    unguarded.run(task()).await;
    assert!(unguarded.summaries().len() > 2);
}

/// Fails its first call with a provider context overflow, then answers from
/// the scripted model.
struct OverflowOnce {
    inner: Arc<ScriptedModel>,
    calls: Mutex<usize>,
}

#[async_trait]
impl ChatModel<()> for OverflowOnce {
    async fn invoke(
        &self,
        state: &(),
        request: ModelRequest,
    ) -> tinyinference_llm::Result<ModelResponse> {
        let first = {
            let mut calls = self.calls.lock().unwrap();
            *calls += 1;
            *calls == 1
        };
        if first {
            // What a provider adapter returns for a 400 overflow: a
            // structured, non-retryable failure.
            return Err(tinyinference_llm::Error::Provider(Box::new(
                tinyinference_llm::model::ProviderError {
                    provider: "mock".into(),
                    status: Some(400),
                    message: "This model's maximum context length is 1000 tokens. However, \
                              your messages resulted in 2000 tokens."
                        .into(),
                    retryable: false,
                    ..Default::default()
                },
            )));
        }
        self.inner.invoke(state, request).await
    }
}

#[tokio::test]
async fn a_provider_overflow_compacts_once_and_retries() {
    let inner = Arc::new(ScriptedModel::new(vec![
        tool_turn("c9"),
        ModelResponse::assistant("done"),
    ]));
    let model = Arc::new(OverflowOnce {
        inner: inner.clone(),
        calls: Mutex::new(0),
    });
    let summarizer = RecordingSummarizer::default();
    let seen = summarizer.seen.clone();
    // A trigger far above the transcript: only the provider's error can
    // start this compaction.
    let policy = SummarizationPolicy {
        keep_last: 2,
        ..SummarizationPolicy::default()
    }
    .with_trigger_override(100_000);
    let mw = Arc::new(ContextCompressionMiddleware::with_summarizer(
        policy,
        Box::new(summarizer),
    ));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model.clone() as _);
    harness.register_tool(Arc::new(FakeTool::returning("read", big_output())));
    harness.push_middleware(mw.clone());
    harness.push_model_middleware(mw.clone());

    let mut input = vec![Message::system("You are a coding agent.")];
    for i in 0..4 {
        input.push(Message::user(format!("question {i}: {}", "q".repeat(400))));
        input.push(Message::assistant(format!(
            "answer {i}: {}",
            "a".repeat(400)
        )));
    }
    input.push(Message::user("continue"));
    let run = harness
        .invoke_default(&(), input)
        .await
        .expect("the retry succeeds");

    assert_eq!(
        *model.calls.lock().unwrap(),
        3,
        "one failure, one retry, one more call"
    );
    assert_eq!(seen.lock().unwrap().len(), 1, "compacted exactly once");
    let requests = inner.requests();
    assert!(
        is_checkpoint(&requests[0].messages[1]),
        "the retry is compacted"
    );
    assert!(
        is_checkpoint(&requests[1].messages[1]),
        "the next call reuses the overflow compaction"
    );
    assert!(run.compacted_history.is_some());
}
