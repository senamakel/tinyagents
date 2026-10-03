//! [`TurnClockMiddleware`]: tell the model how much of the run's wall-clock
//! budget is spent once more than half of it is gone (openhuman#6953).
//!
//! # Why
//!
//! A run with a wall-clock deadline is killed when the deadline passes, even
//! mid-tool. Nothing in the model's context said how long the run had been
//! going, so a model three minutes from the deadline happily started a
//! ten-minute command and the run died with no answer. [`TurnClock`] is the
//! shared reading of that budget; the middleware writes it into the
//! transcript where the model reads it.
//!
//! # Where the note goes, and how often
//!
//! The note is appended to a **tool result** in `after_tool`, as a trailing
//! text block. A tool result is the tail of the transcript at that moment and
//! becomes durable history once folded, so:
//!
//! - no system message appears mid-conversation (some providers hoist those,
//!   which reorders the cached prefix), and
//! - the note is never rewritten afterwards, so it cannot churn the provider's
//!   prompt-prefix cache the way a per-request injection would.
//!
//! It fires at most once per tenth of the budget past the half-way mark
//! (50%, 60%, …, 100%), so a long run carries at most six notes. Below half
//! of the budget, and on a run with no deadline at all, results are untouched.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;

use crate::context::RunContext;
use crate::error::Result as TaResult;
use crate::middleware::{AgentRun, Middleware, ToolInvocationIdentity};
use tinytools::{ToolContent, ToolResult};

/// Bands per budget: the note refreshes every tenth of it.
const BANDS: u128 = 10;
/// The first band that gets a note: half of the budget.
const FIRST_NOTED_BAND: u32 = 5;

/// Render a duration for the note: seconds under a minute, minutes and
/// seconds under ten minutes (where seconds still matter), whole minutes
/// under an hour, then hours and minutes.
pub(crate) fn format_clock(duration: Duration) -> String {
    let secs = duration.as_secs();
    match secs {
        0..60 => format!("{secs}s"),
        60..600 => format!("{}m{:02}s", secs / 60, secs % 60),
        600..3600 => format!("{}m", secs / 60),
        _ => format!("{}h{:02}m", secs / 3600, (secs % 3600) / 60),
    }
}

/// One reading of a run's wall-clock budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TurnClock {
    /// Time since the run's limit tracker started.
    pub elapsed: Duration,
    /// The run's whole wall-clock budget.
    pub budget: Duration,
}

impl TurnClock {
    /// Read the clock for `ctx`.
    ///
    /// The budget is the tighter of the run's own deadline (`RunConfig`'s
    /// timeout) and `host_budget` — normally the harness's
    /// `RunPolicy::limits.max_wall_clock_ms`, which the loop enforces but does
    /// not copy onto the context. `None` when neither is set: an unbounded run
    /// has no clock to read.
    pub fn of<C>(ctx: &RunContext<C>, host_budget: Option<Duration>) -> Option<Self> {
        let run_budget = ctx
            .limits
            .limits()
            .max_wall_clock_ms
            .map(Duration::from_millis);
        let budget = match (run_budget, host_budget) {
            (Some(run), Some(host)) => run.min(host),
            (run, host) => run.or(host)?,
        };
        Some(Self {
            elapsed: ctx.limits.elapsed(),
            budget,
        })
    }

    /// Budget left, saturating at zero once the deadline has passed.
    pub fn remaining(&self) -> Duration {
        self.budget.saturating_sub(self.elapsed)
    }

    /// Which tenth of the budget the run is in, once at least half is spent
    /// (`5` through `10`); `None` before that, or for a zero budget.
    pub fn band(&self) -> Option<u32> {
        let budget = self.budget.as_millis();
        if budget == 0 {
            return None;
        }
        let band = (self.elapsed.as_millis() * BANDS / budget).min(BANDS) as u32;
        (band >= FIRST_NOTED_BAND).then_some(band)
    }

    /// The line the model reads, e.g.
    /// `[turn budget: 32m elapsed / 28m remaining]`.
    pub fn note(&self) -> String {
        format!(
            "[turn budget: {} elapsed / {} remaining]",
            format_clock(self.elapsed),
            format_clock(self.remaining())
        )
    }
}

/// Appends a [`TurnClock::note`] to a tool result once per tenth of the run's
/// wall-clock budget, starting at half of it. See the module docs for why the
/// note rides a tool result.
///
/// Generic over state and run-context payload; nothing here reads them.
pub struct TurnClockMiddleware {
    /// The host's budget for the run (see [`TurnClock::of`]).
    budget: Option<Duration>,
    /// Last band noted, per run instance, so a band is noted once.
    noted: Mutex<HashMap<u64, u32>>,
}

impl TurnClockMiddleware {
    /// Build the middleware over the host's wall-clock budget for the run,
    /// normally the same value set as `RunPolicy::limits.max_wall_clock_ms`.
    pub fn new(budget: Option<Duration>) -> Self {
        Self {
            budget,
            noted: Mutex::default(),
        }
    }

    /// Record `band` for the run and return whether it is new.
    fn claim_band(&self, instance_id: u64, band: u32) -> bool {
        let mut noted = self
            .noted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match noted.get(&instance_id) {
            Some(last) if *last >= band => false,
            _ => {
                noted.insert(instance_id, band);
                true
            }
        }
    }
}

/// Whether `text` is one JSON document (an object or array), ignoring
/// surrounding whitespace.
///
/// Such a tool result is read by machines as well as the model — hosts parse
/// workflow proposals and sub-agent payloads out of it — so a trailing
/// annotation would make it unparseable. The duration line and the turn-budget
/// note both skip it.
pub(crate) fn is_json_document(text: &str) -> bool {
    let trimmed = text.trim();
    (trimmed.starts_with('{') || trimmed.starts_with('['))
        && serde_json::from_str::<serde::de::IgnoredAny>(trimmed).is_ok()
}

/// Whether a tool result carries structured output that must stay parseable:
/// a JSON block, or text that is one JSON document.
fn is_structured(result: &ToolResult) -> bool {
    result
        .content
        .iter()
        .any(|block| matches!(block, ToolContent::Json { .. }))
        || is_json_document(&result.output())
}

/// Append `note` after the result's own content, in the plain blocks and in
/// the markdown rendering (which replaces the blocks when a caller prefers
/// markdown).
fn append_note(result: &mut ToolResult, note: &str) {
    result.content.push(ToolContent::Text {
        text: format!("\n{note}"),
    });
    if let Some(markdown) = result.markdown_formatted.as_mut() {
        markdown.push('\n');
        markdown.push_str(note);
    }
}

#[async_trait]
impl<S: Send + Sync, C: Send + Sync> Middleware<S, C> for TurnClockMiddleware {
    fn name(&self) -> &str {
        "turn_clock"
    }

    async fn after_tool(
        &self,
        ctx: &mut RunContext<C>,
        _state: &S,
        invocation: &ToolInvocationIdentity,
        result: &mut ToolResult,
    ) -> TaResult<()> {
        let Some(clock) = TurnClock::of(ctx, self.budget) else {
            return Ok(());
        };
        let Some(band) = clock.band() else {
            return Ok(());
        };
        if is_structured(result) {
            tracing::trace!(
                target: "tinyagents::middleware",
                call_id = %invocation.call_id(),
                "[turn_clock] structured tool result; leaving the note for a later one"
            );
            return Ok(());
        }
        if !self.claim_band(ctx.instance_id(), band) {
            return Ok(());
        }
        let note = clock.note();
        tracing::debug!(
            target: "tinyagents::middleware",
            run_id = %ctx.run_id(),
            call_id = %invocation.call_id(),
            tool = invocation.tool_name(),
            band,
            elapsed_ms = clock.elapsed.as_millis() as u64,
            budget_ms = clock.budget.as_millis() as u64,
            "[turn_clock] appending turn budget note to tool result"
        );
        append_note(result, &note);
        Ok(())
    }

    async fn after_agent(
        &self,
        ctx: &mut RunContext<C>,
        _state: &S,
        _run: &mut AgentRun,
    ) -> TaResult<()> {
        self.noted
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&ctx.instance_id());
        Ok(())
    }
}

#[cfg(test)]
#[path = "turn_clock_tests.rs"]
mod tests;
