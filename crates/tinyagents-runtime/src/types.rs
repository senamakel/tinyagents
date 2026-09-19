use tinyagents_harness::{
    CancellationToken,
    context::{RunConfig, RunContext},
};
use tinyinference_llm::message::Message;

/// Selects the durable transcript a turn should load before execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResumeMode {
    /// Keep this session's current in-memory history.
    #[default]
    Never,
    /// Load the most recent transcript for the configured agent/stem key.
    LatestForAgent,
    /// Load the most recent root transcript matching `TurnOptions::thread_id`.
    Thread,
}

/// Explicit runtime controls for one session turn.
pub struct TurnOptions<C = ()> {
    /// Opaque correlation identifier persisted with transcript rows.
    pub request_id: Option<String>,
    /// Optional conversation thread identifier used for resume and metadata.
    pub thread_id: Option<String>,
    /// Whether the driver should use its streaming invocation path.
    pub stream: bool,
    /// Transcript resume behavior requested for this turn.
    pub resume: ResumeMode,
    /// Cooperative cancellation shared with the caller.
    pub cancellation: CancellationToken,
    /// Explicit live execution context consumed by the driver.
    pub run_context: RunContext<C>,
}

/// The codec-visible, durable subset of one turn's explicit options.
///
/// `RunContext` itself is live and consumed by the driver. A clone of its host
/// context is captured before that handoff so transcript reconciliation can
/// stamp host-owned data after the driver returns without relying on task-local
/// state or a lossy default context.
#[derive(Clone, Debug)]
pub struct TranscriptTurnOptions<C = ()> {
    /// Opaque correlation identifier for the current turn.
    pub request_id: Option<String>,
    /// Conversation thread selected for this turn.
    pub thread_id: Option<String>,
    /// Whether this turn used the streaming driver path.
    pub stream: bool,
    /// Resume mode selected before execution.
    pub resume: ResumeMode,
    /// Host-owned context cloned from `TurnOptions::run_context.data`.
    pub context: C,
}

impl<C: Clone> TurnOptions<C> {
    pub(crate) fn transcript_options(&self) -> TranscriptTurnOptions<C> {
        TranscriptTurnOptions {
            request_id: self.request_id.clone(),
            thread_id: self.thread_id.clone(),
            stream: self.stream,
            resume: self.resume,
            context: self.run_context.data.clone(),
        }
    }
}

impl Default for TurnOptions<()> {
    fn default() -> Self {
        let cancellation = CancellationToken::new();
        Self {
            request_id: None,
            thread_id: None,
            stream: false,
            resume: ResumeMode::Never,
            run_context: RunContext::new(RunConfig::new("session"), ())
                .with_cancellation(cancellation.clone()),
            cancellation,
        }
    }
}

/// The input a host asks a session to execute.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionTurnRequest {
    /// The next user/application message. Hooks may replace it before the
    /// runtime performs trailing-input deduplication.
    pub input: Message,
}

impl SessionTurnRequest {
    /// Creates a request with one next input message.
    pub fn new(input: Message) -> Self {
        Self { input }
    }
}

/// A committed turn result.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionTurnOutcome {
    /// The full logical history after this turn.
    pub history: Vec<Message>,
    /// The driver's final visible output, when it produced one.
    pub output: Option<String>,
    /// `true` when the driver intentionally ended at an interruptible point.
    pub interrupted: bool,
}

/// The result of loading a transcript into a session.
#[derive(Clone, Debug, PartialEq)]
pub struct SessionResume {
    /// Whether a transcript was found and decoded.
    pub loaded: bool,
    /// The loaded model history, or the existing history when none was found.
    pub history: Vec<Message>,
}

/// The one terminal observation emitted for each call to [`crate::Session::turn`].
#[derive(Clone, Debug, PartialEq)]
pub enum SessionTerminal {
    /// The turn committed. The outcome supplies durable finalization data.
    Completed(SessionTurnOutcome),
    /// The turn was cooperatively cancelled.
    Cancelled,
    /// The turn ended with an error after any recoverable partial persistence.
    Failed(String),
}
