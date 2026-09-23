//! Domain types for the per-thread todo list.
//!
//! A **todo list** is a per-thread ordered checklist of [`TodoItem`]s — the
//! concrete steps a run is working through, distinct from the single
//! per-thread [`ThreadGoal`](crate::goals::ThreadGoal). It is the shape Claude
//! Code and Codex keep: the model rewrites the whole list as it works, each
//! item is a line of text plus one of three states, and nothing else.

use serde::{Deserialize, Serialize};

/// Lifecycle state of a todo item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    /// Not started.
    Pending,
    /// Currently being worked. At most one item may be `InProgress` at a time.
    InProgress,
    /// Finished.
    Completed,
}

impl TodoStatus {
    /// The stable lower-snake-case status label.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
        }
    }
}

/// One item on a thread's todo list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TodoItem {
    /// The step, as the model wrote it.
    pub content: String,
    /// Lifecycle state.
    #[serde(default = "default_status")]
    pub status: TodoStatus,
}

fn default_status() -> TodoStatus {
    TodoStatus::Pending
}

impl TodoItem {
    /// Creates a `Pending` item with `content`.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            status: TodoStatus::Pending,
        }
    }

    /// Creates an item with `content` in `status`.
    pub fn with_status(content: impl Into<String>, status: TodoStatus) -> Self {
        Self {
            content: content.into(),
            status,
        }
    }
}

/// A per-thread list: the items in order plus a last-mutation stamp.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TodoList {
    /// The thread this list belongs to.
    pub thread_id: String,
    /// The items, in list order.
    pub items: Vec<TodoItem>,
    /// Last-mutation timestamp (unix-epoch milliseconds, as a string).
    pub updated_at: String,
}

impl TodoList {
    /// An empty list for `thread_id`.
    pub fn empty(thread_id: impl Into<String>) -> Self {
        Self {
            thread_id: thread_id.into(),
            items: Vec::new(),
            updated_at: now_stamp(),
        }
    }
}

/// A single store outcome: the post-mutation items plus a markdown rendering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TodosSnapshot {
    /// The thread the list belongs to.
    pub thread_id: String,
    /// The items after the mutation.
    pub items: Vec<TodoItem>,
    /// GitHub-flavored markdown rendering of the items.
    pub markdown: String,
}

/// Parses a status label (plus the aliases models commonly write) into a
/// [`TodoStatus`].
pub fn parse_status(raw: &str) -> Result<TodoStatus, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "pending" | "todo" | "open" | "not_started" => Ok(TodoStatus::Pending),
        "in_progress" | "in-progress" | "inprogress" | "started" | "active" => {
            Ok(TodoStatus::InProgress)
        }
        "completed" | "complete" | "done" | "finished" => Ok(TodoStatus::Completed),
        other => Err(format!(
            "invalid status '{other}' (expected pending|in_progress|completed)"
        )),
    }
}

/// Renders a list as GitHub-flavored markdown: one `- [marker] content` line
/// per item (`[ ]` pending, `[~]` in progress, `[x]` completed).
pub fn render_markdown(items: &[TodoItem]) -> String {
    if items.is_empty() {
        return "_No todos yet._".to_string();
    }
    let mut out = String::new();
    for item in items {
        let marker = match item.status {
            TodoStatus::Pending => "[ ]",
            TodoStatus::InProgress => "[~]",
            TodoStatus::Completed => "[x]",
        };
        out.push_str("- ");
        out.push_str(marker);
        out.push(' ');
        out.push_str(&item.content);
        out.push('\n');
    }
    out.trim_end().to_string()
}

/// Normalises a list in place: trims the thread id and every item's content,
/// drops empty items, and stamps `updated_at`.
pub fn normalise_list(list: &mut TodoList) {
    list.thread_id = list.thread_id.trim().to_string();
    list.updated_at = now_stamp();
    for item in list.items.iter_mut() {
        item.content = item.content.trim().to_string();
    }
    list.items.retain(|item| !item.content.is_empty());
}

/// Current unix time in milliseconds. Dependency-free (no `chrono`).
pub(crate) fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Current unix time in milliseconds, as a string — the timestamp format the
/// list uses.
pub(crate) fn now_stamp() -> String {
    now_millis().to_string()
}
