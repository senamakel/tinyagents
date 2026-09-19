//! Durable, rule-driven compaction: token-budget cut points, split-turn
//! summarization, and overflow classification.
//!
//! This is the harness's port of pi's `compaction.ts` /
//! `overflow.ts` (`docs/runtime-comparison/pi.md` §4.5): where
//! [`super::types::SummarizationPolicy`] decides *when* to compact and splits
//! by a fixed `keep_last` message count, this module adds a *token-budget*
//! cut point ([`find_cut_point`]), the "a single turn is itself too big to
//! summarize in one call" case ([`summarize_with_split`]), and a table-driven
//! classifier for turning a provider's context-overflow error into a typed
//! [`OverflowInfo`] ([`OverflowClassifier`]) so a caller can drive an
//! overflow → compact → retry loop
//! ([`crate::middleware::ContextCompressionMiddleware`]).

use std::sync::Arc;

use tinyinference_llm::message::Message;

use crate::error::{Result, TinyAgentsError};

use super::pairing::find_safe_cutoff_point;
use super::trim::partition_system;
use super::types::{CompactionReason, SummaryRecord, SummaryRequest, Summarizer};

// ---------------------------------------------------------------------------
// Cut points
// ---------------------------------------------------------------------------

/// A validated cut point into a message slice: everything before
/// [`Self::index`] (in the non-system message slice the cut was computed
/// over) is old enough to fold into a summary; everything from `index` on is
/// kept verbatim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CutPoint {
    /// Index, into the non-system message slice, of the first message that
    /// stays verbatim. Always a safe boundary: never inside an assistant
    /// tool-call turn and its answering tool results (see
    /// [`super::pairing::find_safe_cutoff_point`]).
    pub index: usize,
    /// Estimated tokens of the messages that would be folded into a summary
    /// (`non_system[..index]`).
    pub tokens_before: u64,
    /// Estimated tokens of the messages kept verbatim (`non_system[index..]`).
    pub tokens_after: u64,
}

/// Finds a cut point that keeps at least `keep_recent_tokens` worth of the
/// most recent messages verbatim, walking `messages` newest-first.
///
/// Port of pi's `findCutPoint` (`compaction.ts:370`). System messages are
/// excluded from consideration (partitioned out first, as every other
/// pairing-aware operation in this module does) and are always implicitly
/// kept by the caller. Returns `None` when there is nothing to cut: either
/// `messages` has no non-system content, or the whole non-system slice
/// already fits within `keep_recent_tokens` (nothing old enough to
/// summarize).
///
/// # Safety
///
/// The raw token-budget candidate is repaired with
/// [`find_safe_cutoff_point`] before being returned, so the result never
/// splits an assistant tool-call turn from the tool results answering it —
/// the same invariant [`super::types::SummarizationPolicy::plan`] enforces
/// for its count-based split. `keep_recent_tokens` is therefore a *minimum*
/// retained budget, not an exact one: the repaired boundary may keep
/// (never drop) a few extra tokens to preserve pairing.
///
/// # Example
///
/// ```
/// use tinyagents_harness::summarization::{estimate_tokens, find_cut_point};
/// use tinyinference_llm::message::Message;
///
/// let messages = vec![
///     Message::user("hello"),
///     Message::assistant("hi there"),
///     Message::user("what's the weather?"),
/// ];
/// let cut = find_cut_point(&messages, 4, |m| estimate_tokens(&m.text())).unwrap();
/// assert!(cut.index > 0);
/// ```
pub fn find_cut_point(
    messages: &[Message],
    keep_recent_tokens: u64,
    estimator: impl Fn(&Message) -> u64,
) -> Option<CutPoint> {
    let (_system, non_system) = partition_system(messages);
    if non_system.is_empty() {
        return None;
    }

    // Walk newest-first, accumulating tokens until the budget would be
    // exceeded; `idx` lands on the oldest message still inside the budget.
    let mut acc = 0u64;
    let mut idx = 0usize;
    for i in (0..non_system.len()).rev() {
        let tokens = estimator(&non_system[i]);
        if acc + tokens > keep_recent_tokens {
            idx = i + 1;
            break;
        }
        acc += tokens;
        idx = i;
    }

    let safe_idx = find_safe_cutoff_point(&non_system, idx);
    if safe_idx == 0 {
        // Everything fits (or pairing repair pulled the cut all the way back
        // to the start) — nothing old enough to compact.
        return None;
    }

    let tokens_before: u64 = non_system[..safe_idx].iter().map(&estimator).sum();
    let tokens_after: u64 = non_system[safe_idx..].iter().map(&estimator).sum();

    Some(CutPoint {
        index: safe_idx,
        tokens_before,
        tokens_after,
    })
}

// ---------------------------------------------------------------------------
// Split-turn summarization
// ---------------------------------------------------------------------------

/// Summarizes `messages` with `summarizer`, splitting into two halves and
/// merging their summaries when `messages` alone estimates above
/// `max_turn_tokens` — the "a single turn is too big for one summarization
/// call" case a fixed-size compaction batch can otherwise hit (a turn with a
/// huge tool result, for instance).
///
/// `previous_summary`, when set, is threaded to the *first* half's
/// [`SummaryRequest::previous_summary`] only — the second half has no
/// predecessor of its own within this split, and
/// [`Summarizer::merge`] is what reconciles the two halves into one summary
/// that itself becomes the next call's `previous_summary`.
///
/// When `messages` fits under `max_turn_tokens`, or no interior message
/// index is a safe split boundary (see
/// [`super::pairing::find_safe_cutoff_point`] — this can happen when the
/// whole turn is a single indivisible tool-call/tool-result pair), the whole
/// slice is summarized in one call instead of forcing an unsafe split.
pub async fn summarize_with_split(
    summarizer: &dyn Summarizer,
    messages: &[Message],
    max_turn_tokens: u64,
    previous_summary: Option<String>,
    estimator: impl Fn(&Message) -> u64,
) -> Result<SummaryRecord> {
    if messages.is_empty() {
        return Err(TinyAgentsError::Validation(
            "cannot summarize an empty turn".into(),
        ));
    }

    let total: u64 = messages.iter().map(&estimator).sum();
    if total <= max_turn_tokens || messages.len() < 2 {
        return summarizer
            .summarize_request(&SummaryRequest {
                messages: messages.to_vec(),
                previous_summary,
            })
            .await;
    }

    let midpoint = messages.len() / 2;
    let split = find_safe_cutoff_point(messages, midpoint);
    if split == 0 || split >= messages.len() {
        // No safe interior boundary — fall back to one call rather than
        // breaking tool-call pairing to force a split.
        return summarizer
            .summarize_request(&SummaryRequest {
                messages: messages.to_vec(),
                previous_summary,
            })
            .await;
    }

    let (first_half, second_half) = messages.split_at(split);

    let first_summary = summarizer
        .summarize_request(&SummaryRequest {
            messages: first_half.to_vec(),
            previous_summary,
        })
        .await?;
    let second_summary = summarizer
        .summarize_request(&SummaryRequest {
            messages: second_half.to_vec(),
            previous_summary: None,
        })
        .await?;

    summarizer.merge(&[first_summary, second_summary]).await
}

// ---------------------------------------------------------------------------
// before_compaction hook
// ---------------------------------------------------------------------------

/// What [`super::types::CompactionRecord`]-producing code hands to a
/// `before_compaction` hook so it can decide whether to proceed, decline, or
/// substitute its own summary — pi's `AgentHarness` compaction operation
/// (`docs/runtime-comparison/pi.md` §4.5).
#[derive(Clone, Debug)]
pub struct CompactionContext {
    /// Why this compaction is about to run.
    pub reason: CompactionReason,
    /// Estimated tokens of the transcript immediately before compaction.
    pub tokens_before: u64,
    /// Number of messages that would be folded into the summary.
    pub to_summarize_count: usize,
    /// Number of messages that would be kept verbatim.
    pub to_keep_count: usize,
}

/// A `before_compaction` hook's decision.
#[derive(Clone, Debug)]
pub enum CompactionDecision {
    /// Run the compaction as planned.
    Proceed,
    /// Skip this compaction; leave the transcript untouched. The caller that
    /// triggered compaction is responsible for deciding what happens next
    /// (for the overflow → compact → retry path, a decline means the
    /// original provider error propagates instead of a retry).
    Decline,
    /// Run the compaction, but install this text as the summary instead of
    /// calling the configured [`Summarizer`].
    UseSummary(String),
}

// ---------------------------------------------------------------------------
// Overflow classification
// ---------------------------------------------------------------------------

/// Best-effort structured detail extracted from a classified overflow error:
/// the token count the request attempted to send and the provider's context
/// limit, when either is recoverable from the error text.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OverflowInfo {
    /// Tokens the failed request attempted to send, when the provider's
    /// message reports it.
    pub requested: Option<u64>,
    /// The provider's context-window limit, when the provider's message
    /// reports it.
    pub limit: Option<u64>,
}

/// A cheap, structured view of a model-call failure handed to an
/// [`OverflowClassifier`] pattern: the provider's raw message plus whatever
/// structured detail is available.
#[derive(Clone, Copy, Debug)]
pub struct OverflowProbe<'a> {
    /// The provider's human-readable error message, verbatim.
    pub message: &'a str,
    /// The provider's error code or type, when reported
    /// (e.g. `"context_length_exceeded"`).
    pub code: Option<&'a str>,
    /// The transport HTTP status, when the failure came from an HTTP
    /// response.
    pub status: Option<u16>,
}

type OverflowMatchFn = Arc<dyn Fn(&OverflowProbe<'_>) -> Option<OverflowInfo> + Send + Sync>;

#[derive(Clone)]
struct OverflowPattern {
    label: &'static str,
    matches: OverflowMatchFn,
}

impl std::fmt::Debug for OverflowPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OverflowPattern")
            .field("label", &self.label)
            .finish_non_exhaustive()
    }
}

/// A table-driven matcher that classifies a [`TinyAgentsError`] as a
/// provider context-window overflow, or `None` for every other failure.
///
/// Port of pi's `isContextOverflow` (`overflow.ts:135`), generalized to a
/// pattern table so a host can register a pattern for a provider or local
/// server this crate does not ship a built-in for (see [`Self::with_pattern`]).
/// [`Self::default`] ships patterns for OpenAI's `context_length_exceeded`,
/// Anthropic's `"prompt is too long"`, a generic `"maximum context length"`
/// phrasing several providers share, local llama.cpp's `n_ctx` messages, and
/// a fallback for an HTTP 400/413 whose body mentions the context window.
///
/// # Example
///
/// ```
/// use tinyagents_harness::error::TinyAgentsError;
/// use tinyagents_harness::summarization::OverflowClassifier;
///
/// let classifier = OverflowClassifier::default();
/// let err = TinyAgentsError::Model(
///     "This model's maximum context length is 8192 tokens. \
///      However, your messages resulted in 9000 tokens.".to_string(),
/// );
/// let info = classifier.classify(&err).expect("classified as overflow");
/// assert_eq!(info.limit, Some(8192));
/// assert_eq!(info.requested, Some(9000));
/// ```
#[derive(Clone, Debug)]
pub struct OverflowClassifier {
    patterns: Vec<OverflowPattern>,
}

impl Default for OverflowClassifier {
    fn default() -> Self {
        Self::with_builtins()
    }
}

impl OverflowClassifier {
    /// An empty classifier with no patterns — every error classifies as
    /// `None`. Prefer [`OverflowClassifier::default()`] for the built-in
    /// provider patterns; use this only to build a classifier with entirely
    /// custom patterns.
    pub fn empty() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    /// Registers an additional pattern, checked after every pattern already
    /// registered (built-ins first when starting from [`Self::default`]).
    ///
    /// `matcher` returns `Some(info)` when the probe indicates an overflow
    /// (`info`'s fields may both be `None` when no numeric detail is
    /// recoverable — the match itself is still meaningful) and `None`
    /// otherwise.
    pub fn with_pattern(
        mut self,
        label: &'static str,
        matcher: impl Fn(&OverflowProbe<'_>) -> Option<OverflowInfo> + Send + Sync + 'static,
    ) -> Self {
        self.patterns.push(OverflowPattern {
            label,
            matches: Arc::new(matcher),
        });
        self
    }

    /// The labels of every registered pattern, in check order. Exposed for
    /// tests and diagnostics.
    pub fn pattern_labels(&self) -> Vec<&'static str> {
        self.patterns.iter().map(|p| p.label).collect()
    }

    /// Classifies `error`, returning `Some(OverflowInfo)` when a registered
    /// pattern (or [`TinyAgentsError::ContextOverflow`] directly) matches.
    pub fn classify(&self, error: &TinyAgentsError) -> Option<OverflowInfo> {
        match error {
            // Already a typed overflow — trust it directly, and try the
            // pattern table over its message only to fill in numeric detail.
            TinyAgentsError::ContextOverflow { message, .. } => {
                let probe = OverflowProbe {
                    message,
                    code: None,
                    status: None,
                };
                Some(self.match_probe(&probe).unwrap_or_default())
            }
            TinyAgentsError::Provider(provider_error) => {
                let probe = OverflowProbe {
                    message: &provider_error.message,
                    code: provider_error.code.as_deref(),
                    status: provider_error.status,
                };
                self.match_probe(&probe)
            }
            TinyAgentsError::Model(message) => {
                let probe = OverflowProbe {
                    message,
                    code: None,
                    status: None,
                };
                self.match_probe(&probe)
            }
            _ => None,
        }
    }

    fn match_probe(&self, probe: &OverflowProbe<'_>) -> Option<OverflowInfo> {
        self.patterns.iter().find_map(|pattern| (pattern.matches)(probe))
    }
}

/// The built-in pattern table: OpenAI, Anthropic, a generic phrasing, local
/// llama.cpp, and an HTTP-status fallback, checked in that order.
impl OverflowClassifier {
    /// Builds the classifier [`Default`] returns. A free function so
    /// `Default::default()` and any caller rebuilding the built-in set (for
    /// example after calling [`Self::empty`]) share one definition.
    fn with_builtins() -> Self {
        Self::empty()
            .with_pattern("openai", |probe| {
                let code_hit = probe.code == Some("context_length_exceeded");
                let message_hit = probe.message.contains("context_length_exceeded");
                if !(code_hit || message_hit) {
                    return None;
                }
                let numbers = extract_numbers(probe.message);
                Some(OverflowInfo {
                    limit: numbers.first().copied(),
                    requested: numbers.get(1).copied(),
                })
            })
            .with_pattern("anthropic", |probe| {
                if !probe.message.to_lowercase().contains("prompt is too long") {
                    return None;
                }
                let numbers = extract_numbers(probe.message);
                Some(OverflowInfo {
                    requested: numbers.first().copied(),
                    limit: numbers.get(1).copied(),
                })
            })
            .with_pattern("llama_cpp", |probe| {
                if !probe.message.contains("n_ctx") {
                    return None;
                }
                let numbers = extract_numbers(probe.message);
                Some(OverflowInfo {
                    limit: numbers.first().copied(),
                    requested: numbers.get(1).copied(),
                })
            })
            .with_pattern("generic", |probe| {
                if !probe
                    .message
                    .to_lowercase()
                    .contains("maximum context length")
                {
                    return None;
                }
                let numbers = extract_numbers(probe.message);
                Some(OverflowInfo {
                    limit: numbers.first().copied(),
                    requested: numbers.get(1).copied(),
                })
            })
            .with_pattern("http_body", |probe| {
                let status_hit = matches!(probe.status, Some(400) | Some(413));
                if !status_hit {
                    return None;
                }
                let lower = probe.message.to_lowercase();
                let body_hit = lower.contains("context")
                    || lower.contains("too long")
                    || lower.contains("token limit");
                if !body_hit {
                    return None;
                }
                Some(OverflowInfo::default())
            })
    }
}

/// Extracts every run of ASCII digits from `text` as a `u64`, in order of
/// appearance, tolerating `,` thousands separators inside a run (`"9,000"` →
/// `9000`). Best-effort: used only to fill optional
/// [`OverflowInfo`] detail, never to decide whether an error is an overflow.
fn extract_numbers(text: &str) -> Vec<u64> {
    let mut numbers = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_digit() {
            current.push(ch);
        } else if ch == ',' && !current.is_empty() {
            continue;
        } else if !current.is_empty() {
            if let Ok(n) = current.parse::<u64>() {
                numbers.push(n);
            }
            current.clear();
        }
    }
    if !current.is_empty()
        && let Ok(n) = current.parse::<u64>()
    {
        numbers.push(n);
    }
    numbers
}

#[cfg(test)]
mod test;
