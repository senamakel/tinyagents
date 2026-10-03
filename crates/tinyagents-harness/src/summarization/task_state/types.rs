//! Types of the typed task-state checkpoint.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tinyinference_llm::model::{ChatModel, ResponseFormat};

/// The model-written half of a checkpoint: the facts only a reader of the
/// conversation can state. Every field defaults, so a partial reply from a
/// weak model still yields a usable state.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TaskState {
    /// The task in one or two sentences.
    pub goal: String,
    /// Every exact identifier, name, value, message or flag the task
    /// requires, verbatim.
    pub requirements: Vec<String>,
    /// Constraints stated by the user or discovered.
    pub constraints: Vec<String>,
    /// `decision — reason`.
    pub decisions: Vec<String>,
    /// `exact error line -> fix`, or `-> unresolved`.
    pub errors_and_fixes: Vec<String>,
    /// Work already done.
    pub todos_done: Vec<String>,
    /// Work still open.
    pub todos_open: Vec<String>,
    /// What the agent currently believes about the problem.
    pub current_hypothesis: String,
    /// The exact command used to run the tests, or empty.
    pub test_command: String,
    /// The very next concrete action (tool and argument).
    pub next_step: String,
}

/// The deterministic half of a checkpoint, read from tool calls and results
/// (see `ledger.rs`) and carried verbatim from one checkpoint to the next.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TaskLedger {
    /// The first user message (the task statement), verbatim up to a cap.
    pub original_task: Option<String>,
    /// Files created or modified, first-seen order.
    pub files_modified: Vec<String>,
    /// Files read, first-seen order.
    pub files_read: Vec<String>,
    /// The most recent shell commands and how they ended.
    pub commands: Vec<CommandRecord>,
}

/// One shell command and its outcome.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CommandRecord {
    /// The command's first line.
    pub command: String,
    /// Whether the result read as a failure.
    pub failed: bool,
    /// The most informative error line, when it failed.
    pub error: Option<String>,
}

/// A [`crate::summarization::Summarizer`] that compacts history into a typed
/// task state: a [`TaskLedger`] copied from the transcript plus a
/// [`TaskState`] written by one structured model call that updates the
/// previous checkpoint's state.
///
/// Chosen by measurement (openhuman-benchmarks `compaction/`): against
/// free-form summaries it kept the most of what a full-context agent knows,
/// both for a single compaction and across three successive ones, on
/// DeepSeek V4 Flash and on Qwen3-8B.
pub struct TaskStateSummarizer {
    pub(crate) model: Arc<dyn ChatModel<()>>,
    pub(crate) model_id: String,
    /// Largest slice of history (estimated tokens) sent in one call; longer
    /// histories are folded in sequential chunks, each updating the state.
    pub(crate) max_chunk_tokens: u64,
    /// Structured-output mode requested from the provider, if any. Off by
    /// default: some endpoints mis-handle JSON mode with reasoning on (Qwen3
    /// on DashScope answers `"display_json"`), and the reply is parsed
    /// leniently either way.
    pub(crate) response_format: Option<ResponseFormat>,
}
