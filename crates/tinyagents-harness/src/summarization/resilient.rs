//! Fault-tolerant, per-turn-caching wrapper around any [`Summarizer`].

use std::hash::{Hash, Hasher};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use tinyinference_llm::message::Message;

use super::{
    CompressionProvenance, SummarizationPolicy, Summarizer, SummaryRecord, SummaryRequest,
    estimate_tokens, render_message_for_summary,
};
use crate::error::Result;
use crate::token_estimation::estimate_slice_tokens;

/// Token budget for the deterministic-trim fallback summary, as a fraction of
/// the policy's summarization trigger budget. The fallback must actually *free*
/// tokens (so the turn shrinks below the window), so it targets a small slice of
/// the trigger point rather than echoing the whole compacted head back.
const FALLBACK_TRIM_TRIGGER_FRACTION: f64 = 0.25;
/// Hard floor / ceiling (tokens) for the deterministic-trim fallback budget, so
/// tiny windows still keep *something* and huge windows don't defeat the point.
const FALLBACK_TRIM_MIN_TOKENS: u64 = 1_024;
const FALLBACK_TRIM_MAX_TOKENS: u64 = 8_192;

/// A single cached summary keyed by the shape of its input slice.
///
/// `key` is a content hash of the exact `to_summarize` slice the crate handed us
/// (message count folded in). Repeat calls within a turn that present the same
/// slice (retries, re-planning, or a stalled tool loop that re-issues an
/// identical model request) reuse the cached [`SummaryRecord`] instead of
/// re-dispatching the summarizer LLM.
struct CachedSummary {
    key: u64,
    record: SummaryRecord,
}

/// Fault-tolerant, per-turn-caching [`Summarizer`] adapter (issue #4461).
///
/// Wraps the real (LLM-backed) [`super::ModelSummarizer`] the turn hands the
/// crate [`ContextCompressionMiddleware`][crate::middleware::ContextCompressionMiddleware]
/// and hardens two regressions the crate introduced versus the legacy engine:
///
/// 1. **Failure no longer aborts the turn.** The crate's `before_model` does
///    `self.summarizer.summarize(..).await?`, so any provider hiccup maps to
///    [`crate::TinyAgentsError::Model`] and fails the whole run — on exactly the
///    longest, most valuable threads. This adapter instead catches the error,
///    logs a `warn`, trips a **per-turn circuit breaker**, and returns a
///    deterministic (LLM-free) trim of the input. The turn continues, matching
///    the legacy `warn! + circuit-breaker + deterministic-trim` fallback. Once
///    the breaker is tripped, every later compaction in the turn skips the
///    known-bad LLM and trims directly.
///
/// 2. **No re-summarizing identical input.** The crate rebuilds the request from
///    `messages.clone()` each loop iteration and rewrites only that per-call
///    clone, so the working transcript never shrinks. Any call that presents the
///    same `to_summarize` slice (retries, re-planning, an identical re-issued
///    request) would otherwise spend a fresh full-transcript summarizer LLM call.
///    A single-slot content-hash cache makes those repeat calls free until the
///    transcript actually grows past the threshold again.
///
/// Constructed fresh per turn inside the turn assembly,
/// so the breaker flag and cache are naturally per-turn state — no task-locals.
pub struct FaultTolerantCachingSummarizer {
    /// The real LLM-backed summarizer we guard.
    inner: Box<dyn Summarizer>,
    /// Per-turn circuit breaker: set once `inner` fails, thereafter every
    /// compaction trims deterministically without touching the LLM.
    breaker_tripped: AtomicBool,
    /// Single-slot cache of the last produced summary, keyed by input-slice hash.
    cache: Mutex<Option<CachedSummary>>,
    /// Token budget for the deterministic-trim fallback (derived from the
    /// policy's context window at construction).
    fallback_trim_budget: u64,
}

impl FaultTolerantCachingSummarizer {
    /// Wrap `inner` with per-turn fault tolerance + caching, sizing the
    /// deterministic-trim fallback budget from `policy`'s trigger budget.
    pub fn new(inner: Box<dyn Summarizer>, policy: &SummarizationPolicy) -> Self {
        let fallback_trim_budget = ((policy.trigger_budget() as f64
            * FALLBACK_TRIM_TRIGGER_FRACTION) as u64)
            .clamp(FALLBACK_TRIM_MIN_TOKENS, FALLBACK_TRIM_MAX_TOKENS);
        tracing::debug!(
            fallback_trim_budget,
            trigger_budget = policy.trigger_budget(),
            "[tinyagents::summarize] installing fault-tolerant caching summarizer adapter"
        );
        Self {
            inner,
            breaker_tripped: AtomicBool::new(false),
            cache: Mutex::new(None),
            fallback_trim_budget,
        }
    }

    /// Content hash of the complete request, including structured message data
    /// and any prior checkpoint, so either kind of change busts the cache.
    fn request_key(request: &SummaryRequest) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        request.messages.len().hash(&mut hasher);
        for message in &request.messages {
            match serde_json::to_vec(message) {
                Ok(encoded) => encoded.hash(&mut hasher),
                Err(_) => format!("{message:?}").hash(&mut hasher),
            }
        }
        request.previous_summary.hash(&mut hasher);
        hasher.finish()
    }

    /// Deterministic, LLM-free fallback: front-drop the oldest messages until the
    /// remaining slice fits [`fallback_trim_budget`][Self::fallback_trim_budget]
    /// tokens (the same front-drop semantics as
    /// [`MessageTrimMiddleware`][crate::middleware::MessageTrimMiddleware]
    /// with [`TrimStrategy::MaxTokens`][crate::summarization::TrimStrategy]),
    /// then render the survivors into a single system checkpoint message. Never
    /// fails, spends no tokens, and produces the same [`SummaryRecord`] shape the
    /// LLM path does so provenance still surfaces downstream.
    fn deterministic_trim(
        &self,
        messages: &[Message],
        previous_summary: Option<&str>,
        cause: &str,
    ) -> SummaryRecord {
        let original_token_estimate =
            estimate_slice_tokens(messages) + previous_summary.map_or(0, estimate_tokens);
        let source_ids: Vec<String> = (0..messages.len()).map(|i| format!("msg-{i}")).collect();

        // Front-drop oldest messages until the tail fits the budget. Keep at
        // least the single most-recent message so the summary is never empty.
        let mut start = 0usize;
        let previous_token_estimate = previous_summary.map_or(0, estimate_tokens);
        loop {
            let remaining = previous_token_estimate + estimate_slice_tokens(&messages[start..]);
            if remaining <= self.fallback_trim_budget || start + 1 >= messages.len() {
                break;
            }
            start += 1;
        }
        let dropped = start;

        let mut body = String::from(
            "=== Conversation Summary (deterministic trim — summarizer unavailable) ===\n",
        );
        if dropped > 0 {
            body.push_str(&format!(
                "[{dropped} older message(s) dropped to fit the context budget]\n",
            ));
        }
        if let Some(previous) = previous_summary {
            body.push_str(&format!("Previous summary (older context):\n{previous}\n"));
        }
        for msg in &messages[start..] {
            body.push_str(&format!("{}\n", render_message_for_summary(msg)));
        }
        let summary_token_estimate = estimate_tokens(&body);

        tracing::warn!(
            cause,
            head_messages = messages.len(),
            dropped,
            from_tokens = original_token_estimate,
            to_tokens = summary_token_estimate,
            "[tinyagents::summarize] deterministic-trim fallback (no LLM); turn continues"
        );

        SummaryRecord {
            summary: Message::system(body),
            provenance: CompressionProvenance {
                source_ids,
                original_token_estimate,
                summary_token_estimate,
                reason: format!(
                    "deterministic-trim fallback (summarizer LLM unavailable: {cause}); \
                     front-dropped {dropped} message(s) to a {}-token budget",
                    self.fallback_trim_budget
                ),
            },
            usage: None,
        }
    }
}

#[async_trait]
impl Summarizer for FaultTolerantCachingSummarizer {
    async fn summarize(&self, messages: &[Message]) -> Result<SummaryRecord> {
        self.summarize_request(&SummaryRequest::new(messages.to_vec()))
            .await
    }

    async fn summarize_request(&self, request: &SummaryRequest) -> Result<SummaryRecord> {
        let key = Self::request_key(request);

        // Cache hit: an identical slice was already summarized this turn.
        if let Ok(guard) = self.cache.lock()
            && let Some(cached) = guard.as_ref()
            && cached.key == key
        {
            tracing::debug!(
                key,
                head_messages = request.messages.len(),
                "[tinyagents::summarize] reusing cached summary (identical input slice; \
                 no summarizer LLM call)"
            );
            return Ok(cached.record.clone());
        }

        // Circuit open from an earlier failure this turn: skip the known-bad LLM
        // and trim deterministically without even attempting a call.
        let record = if self.breaker_tripped.load(Ordering::Relaxed) {
            tracing::debug!(
                key,
                head_messages = request.messages.len(),
                "[tinyagents::summarize] circuit breaker open; trimming deterministically \
                 (skipping summarizer LLM)"
            );
            self.deterministic_trim(
                &request.messages,
                request.previous_summary.as_deref(),
                "circuit breaker open (earlier summarizer failure)",
            )
        } else {
            match self.inner.summarize_request(request).await {
                Ok(record) => record,
                Err(err) => {
                    // Trip the per-turn breaker and fall back — never propagate,
                    // so compaction failure can no longer abort the turn.
                    self.breaker_tripped.store(true, Ordering::Relaxed);
                    tracing::warn!(
                        error = %err,
                        key,
                        head_messages = request.messages.len(),
                        "[tinyagents::summarize] summarizer failed; tripping per-turn circuit \
                         breaker and falling back to deterministic trim"
                    );
                    self.deterministic_trim(
                        &request.messages,
                        request.previous_summary.as_deref(),
                        &err.to_string(),
                    )
                }
            }
        };

        // Cache the result (LLM or fallback) so a repeat identical slice is free.
        if let Ok(mut guard) = self.cache.lock() {
            *guard = Some(CachedSummary {
                key,
                record: record.clone(),
            });
        }
        Ok(record)
    }
}
