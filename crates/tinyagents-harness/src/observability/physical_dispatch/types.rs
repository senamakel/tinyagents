//! Public metadata and terminal types for physical provider dispatches.
use std::time::{Duration, Instant};

/// Metadata available when one provider invocation is admitted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PhysicalDispatchMetadata {
    /// Requested model identifier.
    pub requested_model: Option<String>,
    /// Harness call identifier, when supplied.
    pub call_id: Option<String>,
}

/// Outcome of an admitted provider invocation. It contains no request or response content.
#[derive(Clone, Debug, PartialEq)]
pub struct PhysicalDispatchOutcome {
    /// Whether the provider call returned or was interrupted.
    pub status: PhysicalDispatchStatus,
    /// Whether the provider reported a refusal response.
    pub refused: bool,
    /// Time spent in the provider invocation.
    pub duration: Duration,
    /// Provider-reported usage, when returned.
    pub usage: Option<PhysicalDispatchUsage>,
    /// Provider-reported answering model.
    pub answered_model: Option<String>,
    /// Provider finish reason.
    pub finish_reason: Option<String>,
    /// Upstream provider that served the call, when a routing gateway reports
    /// it (OpenRouter's top-level `provider`).
    pub served_by: Option<String>,
    /// Provider's id for the response (`chatcmpl-…`, `gen-…`, `msg_…`).
    pub response_id: Option<String>,
    /// Time from the invocation's start to its first streamed output (text,
    /// reasoning or a tool call). `None` for a buffered call or a stream that
    /// produced no output.
    pub first_output: Option<Duration>,
}

/// Terminal state of a physical provider invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhysicalDispatchStatus {
    /// The provider returned a response, including refusal responses.
    Succeeded,
    /// The provider returned a refusal response.
    Refused,
    /// The provider returned an error or ended a stream without a terminal item.
    Failed,
    /// The future was dropped before returning.
    Cancelled,
}

/// Provider-reported usage, without raw metadata.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PhysicalDispatchUsage {
    /// Input tokens.
    pub input_tokens: u64,
    /// Output tokens.
    pub output_tokens: u64,
    /// Cached input tokens.
    pub cached_tokens: u64,
    /// Reasoning tokens.
    pub reasoning_tokens: u64,
    /// Provider charge in USD, if reported.
    pub cost_usd: Option<f64>,
}

/// Terminal callback paired with one start callback.
pub trait PhysicalDispatchTerminal: Send {
    /// Report the invocation's sole terminal outcome.
    fn on_terminal(self: Box<Self>, outcome: PhysicalDispatchOutcome);
}

/// Synchronous observer called only after provider-level admission succeeds.
pub trait PhysicalDispatchObserver: Send + Sync {
    /// Record a start and return its terminal callback.
    fn on_start(&self, metadata: &PhysicalDispatchMetadata) -> Box<dyn PhysicalDispatchTerminal>;
}

/// Terminal guard that records cancellation when the invocation future drops.
pub struct PhysicalDispatchGuard {
    terminal: Option<Box<dyn PhysicalDispatchTerminal>>,
    started: Instant,
    usage: Option<PhysicalDispatchUsage>,
    pub(super) first_output: Option<Duration>,
}

impl PhysicalDispatchGuard {
    /// Begin a dispatch and retain its paired terminal callback.
    pub fn new(terminal: Box<dyn PhysicalDispatchTerminal>) -> Self {
        Self {
            terminal: Some(terminal),
            started: Instant::now(),
            usage: None,
            first_output: None,
        }
    }

    /// Record that the stream produced its first output now. Later calls keep
    /// the first time.
    pub fn note_output(&mut self) {
        if self.first_output.is_none() {
            self.first_output = Some(self.started.elapsed());
        }
    }

    /// Retain the most recent cumulative usage reported by a stream.
    pub fn update_usage(&mut self, usage: PhysicalDispatchUsage) {
        self.usage = Some(usage);
    }

    /// Finish with a provider result. Dropping an unfinished guard reports cancellation.
    pub fn finish(mut self, mut outcome: PhysicalDispatchOutcome) {
        outcome.duration = self.started.elapsed();
        if outcome.first_output.is_none() {
            outcome.first_output = self.first_output;
        }
        if outcome.usage.is_none() {
            outcome.usage = self.usage.take();
        }
        if let Some(terminal) = self.terminal.take() {
            terminal.on_terminal(outcome);
        }
    }
}

impl Drop for PhysicalDispatchGuard {
    fn drop(&mut self) {
        if let Some(terminal) = self.terminal.take() {
            terminal.on_terminal(PhysicalDispatchOutcome {
                status: PhysicalDispatchStatus::Cancelled,
                refused: false,
                duration: self.started.elapsed(),
                usage: self.usage.take(),
                answered_model: None,
                finish_reason: None,
                served_by: None,
                response_id: None,
                first_output: self.first_output,
            });
        }
    }
}
