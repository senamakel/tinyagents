use std::collections::BTreeMap;

use tinyagents_harness::{CancellationToken, context::RunContext};
use tinyagents_runtime::ToolSnapshot;
use tinyinference_llm::{message::Message, usage::UsageTotals};

/// A host-provided request to run a task through a subagent.
///
/// `parent_run` is live execution data, deliberately not a serializable host
/// DTO. The planner must derive the child's fully resolved context explicitly
/// instead of consulting task-local state.
pub struct SubagentRequest<C = ()> {
    /// Host-local task id. Its durable lifecycle identity is scoped by
    /// [`SubagentTaskKey`], derived from this request's parent run.
    pub task_id: String,
    /// The explicit live parent context from which a child context is derived.
    pub parent_run: RunContext<C>,
    /// Host-visible task input that the planner converts into model messages.
    pub input: String,
    /// Optional host-owned conversation thread correlation id.
    pub thread_id: Option<String>,
    /// A caller-supplied checkpoint. When absent the driver asks persistence.
    pub resume: Option<SubagentResume>,
}

/// Durable, host-neutral identity for one subagent lifecycle.
///
/// A task id is only unique inside the recursive run that created it. The
/// parent run and root run therefore scope durable persistence, in-memory
/// coalescing, and terminal cache entries. The optional thread adds the host's
/// durable conversation partition when it is available.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SubagentTaskKey {
    /// Top-level recursive run that owns this task tree.
    pub root_run_id: String,
    /// Immediate run that requested this subagent lifecycle.
    pub parent_run_id: String,
    /// Host conversation partition, if either request or parent supplies one.
    pub thread_id: Option<String>,
    /// Host-local task id within the scoped recursive run.
    pub task_id: String,
}

impl<C> SubagentRequest<C> {
    /// Derives the durable lifecycle key without exposing host context data.
    pub fn task_key(&self) -> SubagentTaskKey {
        SubagentTaskKey {
            root_run_id: self.parent_run.lineage().root_run_id.as_str().to_owned(),
            parent_run_id: self.parent_run.run_id().as_str().to_owned(),
            thread_id: self
                .thread_id
                .clone()
                .or_else(|| self.parent_run.thread_id().map(|id| id.as_str().to_owned())),
            task_id: self.task_id.clone(),
        }
    }
}

/// A neutral checkpoint offered to a planner for resumption.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SubagentResume {
    /// Lossless model history available to the host planner.
    pub history: Vec<Message>,
    /// Opaque checkpoint token; its interpretation remains host-owned.
    pub checkpoint: Option<String>,
    /// Small neutral metadata. This deliberately excludes paths and credentials.
    pub metadata: BTreeMap<String, String>,
}

/// Fully resolved, immutable execution input produced by a host planner.
///
/// The planner, not this orchestration layer, resolves agent identity, prompt
/// messages, limits, workspace policy and the tool declaration allowlist.
pub struct PreparedSubagent<C = ()> {
    /// Host-stable task identity.
    pub task_id: String,
    /// Host-resolved agent identity.
    pub agent_key: String,
    /// Complete model input, including any resume history and prompt prefix.
    pub input: Vec<Message>,
    /// Frozen model-visible tool declarations for this one execution.
    pub tools: ToolSnapshot,
    /// Explicit child run context, including lineage and host context data.
    pub run_context: RunContext<C>,
}

/// The one execution handed to a [`crate::subagent::SubagentExecutor`].
pub struct SubagentExecution<C = ()> {
    /// The planner's complete, host-resolved execution description.
    pub prepared: PreparedSubagent<C>,
    /// Cooperative cancellation shared with the lifecycle owner.
    pub cancellation: CancellationToken,
}

/// A neutral reference to a host-owned artifact.
///
/// The reference intentionally contains no filesystem path or URL. Hosts own
/// artifact authorization and resolution.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ArtifactReference {
    /// Stable host artifact identifier.
    pub id: String,
    /// Optional neutral media-type hint.
    pub media_type: Option<String>,
    /// Opaque, non-location metadata for a host to interpret.
    pub metadata: BTreeMap<String, String>,
}

/// A neutral suspension point that can later be supplied to a planner.
#[derive(Clone, Debug, PartialEq)]
pub struct SubagentPause {
    /// Why execution needs input or an external host action.
    pub reason: String,
    /// Resume state captured at the suspension point.
    pub resume: SubagentResume,
}

/// A neutral non-successful but terminal completion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubagentIncomplete {
    /// A host-safe explanation for incomplete work.
    pub reason: String,
}

/// The visible status of one subagent run.
#[derive(Clone, Debug, PartialEq)]
pub enum SubagentStatus {
    /// The subagent completed normally.
    Completed,
    /// The subagent stopped at a resumable input boundary.
    AwaitingInput(SubagentPause),
    /// The subagent terminated without a complete result.
    Incomplete(SubagentIncomplete),
    /// Cooperative cancellation won the lifecycle race.
    Cancelled,
}

/// Complete neutral result of one subagent execution.
#[derive(Clone, Debug, PartialEq)]
pub struct SubagentOutcome {
    /// Host-stable task identity.
    pub task_id: String,
    /// Final visible text, retained even when cancellation arrives after work.
    pub output: String,
    /// Complete model history without lossy transcript conversion.
    pub history: Vec<Message>,
    /// Terminal or pause state.
    pub status: SubagentStatus,
    /// Model usage reported by the nested execution exactly once.
    pub usage: UsageTotals,
    /// Host-owned artifacts represented by neutral references.
    pub artifacts: Vec<ArtifactReference>,
}

impl SubagentOutcome {
    /// Creates the truthful empty result used when cancellation prevents a
    /// planner or executor from starting.
    pub fn cancelled(task_id: impl Into<String>) -> Self {
        Self {
            task_id: task_id.into(),
            output: String::new(),
            history: Vec::new(),
            status: SubagentStatus::Cancelled,
            usage: UsageTotals::default(),
            artifacts: Vec::new(),
        }
    }

    pub(crate) fn cancelled_preserving(mut self) -> Self {
        self.status = SubagentStatus::Cancelled;
        self
    }
}

/// The durable input to [`crate::subagent::SubagentPersistence::save_pause`].
#[derive(Clone, Debug, PartialEq)]
pub struct PersistedSubagentPause {
    /// Durable scoped lifecycle identity.
    pub key: SubagentTaskKey,
    /// The resumable pause state.
    pub pause: SubagentPause,
}

/// Typed lifecycle failures. Adapters classify their errors at the seam that
/// owns them; the driver never flattens them into an untyped host error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubagentError {
    /// The planner rejected or could not resolve a request.
    Planning(String),
    /// The executor could not finish a prepared run.
    Execution(String),
    /// Resume loading or outcome persistence failed.
    Persistence(String),
    /// A required planner, executor, or persistence seam was not provided.
    MissingCapability(&'static str),
    /// A host seam returned an outcome for a task other than the one reserved
    /// by this lifecycle. The driver rejects it before any persistence or
    /// terminal cache write can corrupt another task's record.
    TaskIdMismatch {
        /// Task id the caller reserved.
        expected: String,
        /// Task id returned by the planner or executor.
        actual: String,
    },
    /// Cooperative cancellation interrupted the lifecycle.
    Cancelled,
}

impl std::fmt::Display for SubagentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Planning(message) => write!(f, "subagent planning failed: {message}"),
            Self::Execution(message) => write!(f, "subagent execution failed: {message}"),
            Self::Persistence(message) => write!(f, "subagent persistence failed: {message}"),
            Self::MissingCapability(capability) => {
                write!(f, "subagent host capability is unavailable: {capability}")
            }
            Self::TaskIdMismatch { expected, actual } => write!(
                f,
                "subagent host seam returned task id {actual:?}, expected {expected:?}"
            ),
            Self::Cancelled => write!(f, "subagent execution was cancelled"),
        }
    }
}

impl std::error::Error for SubagentError {}
