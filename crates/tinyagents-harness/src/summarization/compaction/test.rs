//! Tests for cut-point discovery, split-turn summarization, and overflow
//! classification.

use std::sync::Mutex;

use async_trait::async_trait;
use serde_json::json;
use tinyinference_llm::message::{AssistantMessage, Message};
use tinyinference_llm::tool::ToolCall;

use super::*;
use crate::error::TinyAgentsError;
use crate::summarization::{ConcatSummarizer, SummaryRecord, tool_pairing_is_intact};
use crate::token_estimation::estimate_message_tokens;

fn assistant_calling(ids: &[&str]) -> Message {
    Message::Assistant(AssistantMessage {
        id: None,
        content: Vec::new(),
        tool_calls: ids
            .iter()
            .map(|id| ToolCall::new(*id, "lookup", json!({"q": "rust"})))
            .collect(),
        usage: None,
        origin: None,
    })
}

// ---------------------------------------------------------------------------
// find_cut_point
// ---------------------------------------------------------------------------

#[test]
fn find_cut_point_never_splits_a_tool_pair() {
    // `[user, assistant(tool_calls=[c1]), tool(c1), assistant("done")]`. The
    // budget below is chosen (from the messages' own estimated weights) so
    // the *naive* newest-first walk lands the cut exactly on `tool(c1)` —
    // precisely the split the repair exists to prevent.
    let user = Message::user("weather?");
    let call = assistant_calling(&["c1"]);
    let result = Message::tool("c1", "sunny and warm today, 21 degrees");
    let done = Message::assistant("It's sunny and warm.");
    let non_system = vec![user, call, result.clone(), done.clone()];

    let budget = estimate_message_tokens(&result) + estimate_message_tokens(&done);
    let cut = find_cut_point(&non_system, budget, estimate_message_tokens)
        .expect("some cut point should be found");

    // The naive (unrepaired) boundary would be index 2 (`tool(c1)` itself);
    // confirm the repair actually moved it, not that it happened to already
    // be safe.
    assert_ne!(
        cut.index, 2,
        "test setup did not land the naive cut on the tool result"
    );
    assert!(!matches!(non_system[cut.index], Message::Tool(_)));

    let kept = &non_system[cut.index..];
    assert!(
        tool_pairing_is_intact(kept),
        "cut point {} orphans a tool pair: {kept:?}",
        cut.index
    );
}

#[test]
fn find_cut_point_respects_keep_recent_tokens() {
    let messages = vec![
        Message::user("one"),
        Message::user("two"),
        Message::user("three"),
        Message::user("four"),
    ];
    // A generous budget should keep everything (nothing old enough to cut).
    let generous = find_cut_point(&messages, 10_000, estimate_message_tokens);
    assert!(generous.is_none());

    // A tiny budget keeps only the most recent message(s), and whatever it
    // keeps meets or exceeds the requested budget (a cut point is a floor,
    // not an exact count).
    let tiny = find_cut_point(&messages, 1, estimate_message_tokens)
        .expect("a tiny budget should still find a cut point");
    assert!(tiny.index > 0);
    assert!(tiny.tokens_after >= 1);
    assert_eq!(
        tiny.tokens_before + tiny.tokens_after,
        messages.iter().map(estimate_message_tokens).sum::<u64>()
    );
}

#[test]
fn find_cut_point_none_when_nothing_to_summarize() {
    let messages = vec![Message::system("sys"), Message::user("hi")];
    assert!(find_cut_point(&messages, 10_000, estimate_message_tokens).is_none());
}

#[test]
fn find_cut_point_none_for_only_system_messages() {
    let messages = vec![Message::system("sys")];
    assert!(find_cut_point(&messages, 0, estimate_message_tokens).is_none());
}

// ---------------------------------------------------------------------------
// summarize_with_split
// ---------------------------------------------------------------------------

/// Records every call `summarize_request`/`merge` received, so tests can
/// assert both the split happened and what was threaded through it.
#[derive(Default)]
struct RecordingSummarizer {
    requests: Mutex<Vec<SummaryRequest>>,
    merges: Mutex<usize>,
}

#[async_trait]
impl Summarizer for RecordingSummarizer {
    async fn summarize(&self, messages: &[Message]) -> Result<SummaryRecord> {
        self.summarize_request(&SummaryRequest::new(messages.to_vec()))
            .await
    }

    async fn summarize_request(&self, request: &SummaryRequest) -> Result<SummaryRecord> {
        self.requests.lock().unwrap().push(request.clone());
        ConcatSummarizer.summarize(&request.messages).await
    }

    async fn merge(&self, summaries: &[SummaryRecord]) -> Result<SummaryRecord> {
        *self.merges.lock().unwrap() += 1;
        // Delegate to the default concatenation merge via ConcatSummarizer's
        // inherited default (ConcatSummarizer never overrides `merge`).
        ConcatSummarizer.merge(summaries).await
    }
}

#[tokio::test]
async fn split_turn_summarizes_whole_slice_when_under_budget() {
    let summarizer = RecordingSummarizer::default();
    let messages = vec![Message::user("a"), Message::user("b")];
    let total: u64 = messages.iter().map(estimate_message_tokens).sum();

    summarize_with_split(
        &summarizer,
        &messages,
        total + 100,
        None,
        estimate_message_tokens,
    )
    .await
    .unwrap();

    assert_eq!(summarizer.requests.lock().unwrap().len(), 1);
    assert_eq!(*summarizer.merges.lock().unwrap(), 0);
}

#[tokio::test]
async fn split_turn_splits_and_merges_when_over_budget() {
    let summarizer = RecordingSummarizer::default();
    let big = "word ".repeat(50);
    let messages = vec![
        Message::user(format!("first half {big}")),
        Message::user(format!("second half {big}")),
    ];
    let total: u64 = messages.iter().map(estimate_message_tokens).sum();

    let merged = summarize_with_split(
        &summarizer,
        &messages,
        total / 2,
        Some("previous run's summary".to_string()),
        estimate_message_tokens,
    )
    .await
    .unwrap();

    assert_eq!(summarizer.requests.lock().unwrap().len(), 2);
    assert_eq!(*summarizer.merges.lock().unwrap(), 1);
    // The merged text carries content from both halves.
    let text = merged.summary.text();
    assert!(text.contains("first half"));
    assert!(text.contains("second half"));
}

#[tokio::test]
async fn split_turn_threads_previous_summary_to_first_half_only() {
    let summarizer = RecordingSummarizer::default();
    let big = "word ".repeat(50);
    let messages = vec![
        Message::user(format!("alpha {big}")),
        Message::user(format!("beta {big}")),
    ];
    let total: u64 = messages.iter().map(estimate_message_tokens).sum();

    summarize_with_split(
        &summarizer,
        &messages,
        total / 2,
        Some("prior".to_string()),
        estimate_message_tokens,
    )
    .await
    .unwrap();

    let requests = summarizer.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].previous_summary.as_deref(), Some("prior"));
    assert_eq!(requests[1].previous_summary, None);
}

// ---------------------------------------------------------------------------
// Iterative summaries (SummaryRequest.previous_summary)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn iterative_summary_receives_the_previous_summary() {
    let summarizer = RecordingSummarizer::default();
    let messages = vec![Message::user("new turn")];

    summarizer
        .summarize_request(&SummaryRequest::new(messages).with_previous_summary("earlier context"))
        .await
        .unwrap();

    let requests = summarizer.requests.lock().unwrap();
    assert_eq!(
        requests[0].previous_summary.as_deref(),
        Some("earlier context")
    );
}

#[tokio::test]
async fn default_summarize_request_ignores_previous_summary() {
    // `ConcatSummarizer` never overrides `summarize_request`, so the default
    // trait method's back-compat delegation to `summarize` applies: the
    // previous summary is accepted but not threaded into the (no-LLM) output.
    let request = SummaryRequest::new(vec![Message::user("hi")])
        .with_previous_summary("ignored by ConcatSummarizer");
    let record = ConcatSummarizer.summarize_request(&request).await.unwrap();
    assert!(record.summary.text().contains("hi"));
}

// ---------------------------------------------------------------------------
// OverflowClassifier
// ---------------------------------------------------------------------------

#[test]
fn classifies_typed_context_overflow_directly() {
    let classifier = OverflowClassifier::default();
    let err = TinyAgentsError::ContextOverflow {
        provider: "openai".to_string(),
        model: Some("gpt-test".to_string()),
        message: "too long".to_string(),
    };
    assert!(classifier.classify(&err).is_some());
}

#[test]
fn classifies_openai_context_length_exceeded_by_code() {
    let classifier = OverflowClassifier::default();
    let err = TinyAgentsError::Provider(Box::new(tinyinference_llm::model::ProviderError {
        provider: "openai".to_string(),
        model: None,
        status: Some(400),
        code: Some("context_length_exceeded".to_string()),
        message: "This model's maximum context length is 8192 tokens. However, your messages \
                  resulted in 9000 tokens."
            .to_string(),
        retryable: false,
        retry_after_ms: None,
        raw: None,
    }));
    let info = classifier.classify(&err).expect("classified as overflow");
    assert_eq!(info.limit, Some(8192));
    assert_eq!(info.requested, Some(9000));
}

#[test]
fn classifies_anthropic_prompt_is_too_long() {
    let classifier = OverflowClassifier::default();
    let err =
        TinyAgentsError::Model("prompt is too long: 210000 tokens > 200000 maximum".to_string());
    let info = classifier.classify(&err).expect("classified as overflow");
    assert_eq!(info.requested, Some(210_000));
    assert_eq!(info.limit, Some(200_000));
}

#[test]
fn classifies_generic_maximum_context_length_phrasing() {
    let classifier = OverflowClassifier::default();
    let err = TinyAgentsError::Model(
        "request exceeds the maximum context length of 4096 tokens (sent 5000)".to_string(),
    );
    let info = classifier.classify(&err).expect("classified as overflow");
    assert_eq!(info.limit, Some(4096));
    assert_eq!(info.requested, Some(5000));
}

#[test]
fn classifies_local_llama_cpp_n_ctx_messages() {
    let classifier = OverflowClassifier::default();
    let err =
        TinyAgentsError::Model("context size exceeded (n_ctx = 4096, tokens = 4300)".to_string());
    let info = classifier.classify(&err).expect("classified as overflow");
    assert_eq!(info.limit, Some(4096));
    assert_eq!(info.requested, Some(4300));
}

#[test]
fn classifies_http_400_and_413_bodies_mentioning_the_context_window() {
    let classifier = OverflowClassifier::default();
    let err_400 = TinyAgentsError::Provider(Box::new(tinyinference_llm::model::ProviderError {
        provider: "custom".to_string(),
        model: None,
        status: Some(400),
        code: None,
        message: "request too long for the context window".to_string(),
        retryable: false,
        retry_after_ms: None,
        raw: None,
    }));
    assert!(classifier.classify(&err_400).is_some());

    let err_413 = TinyAgentsError::Provider(Box::new(tinyinference_llm::model::ProviderError {
        provider: "custom".to_string(),
        model: None,
        status: Some(413),
        code: None,
        message: "payload too large: context exceeded".to_string(),
        retryable: false,
        retry_after_ms: None,
        raw: None,
    }));
    assert!(classifier.classify(&err_413).is_some());
}

#[test]
fn does_not_classify_unrelated_provider_errors() {
    let classifier = OverflowClassifier::default();
    let err = TinyAgentsError::Provider(Box::new(tinyinference_llm::model::ProviderError {
        provider: "openai".to_string(),
        model: None,
        status: Some(429),
        code: Some("rate_limit_exceeded".to_string()),
        message: "rate limit exceeded, please retry later".to_string(),
        retryable: true,
        retry_after_ms: Some(1000),
        raw: None,
    }));
    assert!(classifier.classify(&err).is_none());
    assert!(
        classifier
            .classify(&TinyAgentsError::Tool("boom".into()))
            .is_none()
    );
}

#[test]
fn with_pattern_extends_the_classifier() {
    let classifier = OverflowClassifier::empty().with_pattern("acme", |probe| {
        probe
            .message
            .contains("ACME_CONTEXT_FULL")
            .then_some(OverflowInfo::default())
    });
    assert_eq!(classifier.pattern_labels(), vec!["acme"]);
    let err = TinyAgentsError::Model("ACME_CONTEXT_FULL: cannot proceed".to_string());
    assert!(classifier.classify(&err).is_some());
    assert!(
        classifier
            .classify(&TinyAgentsError::Model("something else".to_string()))
            .is_none()
    );
}

#[test]
fn default_classifier_has_the_documented_built_in_patterns() {
    let labels = OverflowClassifier::default().pattern_labels();
    assert_eq!(
        labels,
        vec!["openai", "anthropic", "llama_cpp", "generic", "http_body"]
    );
}
