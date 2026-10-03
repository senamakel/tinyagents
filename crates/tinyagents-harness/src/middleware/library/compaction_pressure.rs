//! [`CompactionPressure`]: the usage-based trigger and the anti-thrash guard
//! behind [`crate::middleware::ContextCompressionMiddleware`].
//!
//! Kept apart from the middleware so the arithmetic is testable without a
//! model, a stack, or a summarizer.

use tinyinference_llm::message::Message;
use tinyinference_llm::usage::Usage;

use crate::middleware::types::{CompactionPressure, MeasuredPrompt};
use crate::summarization::SummarizationPolicy;

/// How the prompt size the trigger compares was obtained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PromptSource {
    /// The previous call's provider-reported prompt tokens plus an estimate
    /// of what was appended since.
    Measured,
    /// The chars-based estimate of the whole request (no usage available, or
    /// the request no longer extends the measured one).
    Estimated,
}

impl PromptSource {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            PromptSource::Measured => "usage",
            PromptSource::Estimated => "estimate",
        }
    }
}

impl CompactionPressure {
    /// Starts a model call: spends one call of an active suppression window.
    /// Returns whether summarization is suppressed for this call.
    pub(crate) fn begin_call(&mut self) -> bool {
        self.pending = None;
        if self.suppressed_for > 0 {
            self.suppressed_for -= 1;
            return true;
        }
        false
    }

    /// The best available prompt size for a request of `messages` carrying
    /// `schema_tokens` of tool declarations.
    ///
    /// With a measurement whose request this one extends, that is the
    /// provider's own count plus an estimate of the appended messages and of
    /// any schema growth; otherwise the whole-request estimate.
    pub(crate) fn prompt_tokens(
        &self,
        messages: &[Message],
        schema_tokens: u64,
    ) -> (u64, PromptSource) {
        if let Some(measured) = self.measured
            && measured.messages <= messages.len()
        {
            let appended =
                crate::token_estimation::estimate_slice_tokens(&messages[measured.messages..]);
            let schema_growth = schema_tokens.saturating_sub(measured.schema_tokens);
            return (
                measured.prompt_tokens + appended + schema_growth,
                PromptSource::Measured,
            );
        }
        (
            crate::token_estimation::estimate_slice_tokens(messages) + schema_tokens,
            PromptSource::Estimated,
        )
    }

    /// Records the shape of the request this middleware let through, so the
    /// usage reported for it can be attributed in [`Self::observe`].
    pub(crate) fn note_request(&mut self, messages: usize, schema_tokens: u64) {
        self.pending = Some((messages, schema_tokens));
    }

    /// Marks that a compaction just ran, so the next reported usage judges
    /// whether it brought the prompt under the trigger.
    pub(crate) fn note_compaction(&mut self) {
        self.awaiting_verdict = true;
    }

    /// Takes the usage the provider reported for the call this middleware
    /// last let through. Records the measurement and, when that call followed
    /// a compaction, scores it: a prompt still at or above the trigger is a
    /// strike, and `strike_limit` strikes in a row suppress summarization for
    /// `cooldown_calls` calls. Returns `true` when this observation engaged
    /// the guard.
    pub(crate) fn observe(
        &mut self,
        usage: Option<&Usage>,
        policy: &SummarizationPolicy,
        strike_limit: u32,
        cooldown_calls: u32,
    ) -> bool {
        let Some((messages, schema_tokens)) = self.pending.take() else {
            return false;
        };
        let Some(usage) = usage.filter(|usage| usage.input_tokens > 0) else {
            return false;
        };
        let prompt_tokens = usage.input_tokens;
        self.measured = Some(MeasuredPrompt {
            prompt_tokens,
            messages,
            schema_tokens,
        });
        if !std::mem::take(&mut self.awaiting_verdict) {
            return false;
        }
        if !policy.exceeds_trigger(prompt_tokens) {
            if self.strikes > 0 {
                tracing::debug!(
                    prompt_tokens,
                    trigger = policy.trigger_budget(),
                    "[context_compression] compaction effective; strikes reset"
                );
            }
            self.strikes = 0;
            return false;
        }
        self.strikes += 1;
        tracing::warn!(
            prompt_tokens,
            trigger = policy.trigger_budget(),
            strikes = self.strikes,
            strike_limit,
            "[context_compression] strike: prompt still over the trigger after compaction"
        );
        if strike_limit == 0 || self.strikes < strike_limit {
            return false;
        }
        self.strikes = 0;
        self.suppressed_for = cooldown_calls;
        tracing::warn!(
            cooldown_calls,
            "[context_compression] suppressing summarization after repeated ineffective \
             compactions; deterministic trim runs instead"
        );
        true
    }
}

#[cfg(test)]
#[path = "compaction_pressure_tests.rs"]
mod tests;
