//! `todo` as a **session todo list** — the shape Claude Code and Codex use.
//!
//! One call writes the whole list, `{"todos": [{"content", "status"}]}`, and
//! a call with no arguments reads it back. There is no per-card CRUD, no
//! approval gate, no evidence or plan: the list is the progress checklist a
//! model rewrites as it works. It shares the board [`store`] (so the
//! single-`in_progress` invariant and ordering are the same code) but exposes
//! only `content`, a three-state `status`, and an id-free checklist rendering.
//!
//! The list is keyed by the caller's [`ToolRunContext::thread_id`]; a host
//! that scopes lists differently (per agent session, say) hands the key it
//! wants through [`write`] / [`read`] instead of the [`Tool`] entry point.
//! Every argument problem is a [`ToolResult::error`] the model can correct,
//! never an `Err`: an `Err` out of a tool dispatch is fatal to the run.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use super::store;
use super::types::{TaskBoardCard, TaskCardStatus, TodosSnapshot};
use tinyagents_harness::error::Result;
use tinyagents_harness::store::Store;
use tinyagents_harness::tool::ToolRegistry;
use tinytools::{Tool, ToolPolicy, ToolResult, ToolRunContext, ToolSideEffects};

const TOOL_NAME: &str = "todo";
const SESSION_TODO_MARKER: &str = "session_todo";

const DESCRIPTION: &str = "Your todo list for this conversation. Pass the complete list every \
    time; it replaces what was there. Use it for work with 3+ steps: write the steps up front, \
    keep exactly one `in_progress`, mark each `completed` only after its work has actually run \
    and its result is in this conversation. Writing the list is bookkeeping, not work: the same \
    response that updates it must also carry the tool call that does the next step, and one \
    update per response is enough. Omit `todos` to read the current list.";

/// One item as the model writes it.
#[derive(Debug, Deserialize)]
pub struct TodoItem {
    pub content: String,
    #[serde(default)]
    pub status: Option<String>,
}

/// The three states the model is told about. Board states the list cannot
/// produce (`ready`, `awaiting_approval`, `rejected`, `blocked`) fold into the
/// nearest one so a list written by the board tool still reads sensibly.
pub fn wire_status(status: TaskCardStatus) -> &'static str {
    match status {
        TaskCardStatus::InProgress => "in_progress",
        TaskCardStatus::Done | TaskCardStatus::Rejected => "completed",
        TaskCardStatus::Todo
        | TaskCardStatus::Ready
        | TaskCardStatus::AwaitingApproval
        | TaskCardStatus::Blocked => "pending",
    }
}

/// The checklist as the model should read it back: one line per item, no
/// ids. The board renderer appends `` `(task-n)` `` to every line, and a model
/// that had just written the list read those ids as "a list pre-filled from a
/// previous session" and wrote the same list again until the repeat guard
/// stopped the run.
pub fn render_checklist(cards: &[TaskBoardCard]) -> String {
    if cards.is_empty() {
        return "_No todos._".to_string();
    }
    cards
        .iter()
        .map(|card| {
            let marker = match wire_status(card.status) {
                "in_progress" => "[~]",
                "completed" => "[x]",
                _ => "[ ]",
            };
            format!("- {marker} {}", card.title)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The JSON a call answers with: the list as the model sees it plus the
/// checklist rendering for transcripts.
pub fn payload(snapshot: &TodosSnapshot) -> Value {
    let todos: Vec<Value> = snapshot
        .cards
        .iter()
        .map(|card| json!({ "content": card.title, "status": wire_status(card.status) }))
        .collect();
    json!({ "todos": todos, "markdown": render_checklist(&snapshot.cards) })
}

/// Turns the model's `todos` array into board cards. Every problem is a
/// message for the model.
pub fn parse_items(raw: &Value) -> std::result::Result<Vec<TaskBoardCard>, String> {
    let items: Vec<TodoItem> =
        serde_json::from_value(raw.clone()).map_err(|e| format!("invalid `todos`: {e}"))?;
    let mut cards = Vec::with_capacity(items.len());
    for item in items {
        let content = item.content.trim();
        if content.is_empty() {
            return Err("every todo needs non-empty `content`".to_string());
        }
        let mut card = TaskBoardCard::new(content);
        card.status = match item.status.as_deref().map(str::trim) {
            None | Some("pending") => TaskCardStatus::Todo,
            Some("in_progress") => TaskCardStatus::InProgress,
            Some("completed") => TaskCardStatus::Done,
            Some(_) => return Err("status must be `pending`, `in_progress`, or `completed`".into()),
        };
        card.source_metadata = Some(json!({ SESSION_TODO_MARKER: true }));
        cards.push(card);
    }
    Ok(cards)
}

/// Whether `card` is a session checklist item rather than dispatchable work.
pub(crate) fn is_session_todo_card(card: &TaskBoardCard) -> bool {
    card.source_metadata
        .as_ref()
        .and_then(|metadata| metadata.get(SESSION_TODO_MARKER))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// Replaces the list under `key` with `cards` (an empty list clears it).
pub async fn write(
    store: &Arc<dyn Store>,
    key: &str,
    cards: Vec<TaskBoardCard>,
) -> Result<TodosSnapshot> {
    store::replace(store, key, cards).await
}

/// Reads the list under `key`.
pub async fn read(store: &Arc<dyn Store>, key: &str) -> Result<TodosSnapshot> {
    store::list(store, key).await
}

/// Resolves one call against `key`: a write when `todos` is an array, a read
/// when the call carries no arguments or `todos: null`, and a shape error for
/// anything else (a write that used some other key such as the retired
/// `cards`).
pub async fn call(store: &Arc<dyn Store>, key: &str, args: &Value) -> Result<ToolResult> {
    let Some(args) = args.as_object() else {
        return Ok(ToolResult::error("arguments must be an object"));
    };
    let outcome = match args.get("todos") {
        None => {
            if !args.is_empty() {
                Err(format!(
                    "unknown arguments {:?}: pass `todos` (the full list of {{content, status}}), \
                 or no arguments to read the list",
                    args.keys().collect::<Vec<_>>()
                ))
            } else {
                read(store, key).await.map_err(|e| e.to_string())
            }
        }
        Some(Value::Null) => match args.len() {
            1 => read(store, key).await.map_err(|e| e.to_string()),
            _ => Err(format!(
                "unknown arguments {:?}: pass only `todos`, or no arguments to read the list",
                args.keys().collect::<Vec<_>>()
            )),
        },
        Some(raw) => match parse_items(raw) {
            Ok(cards) => write(store, key, cards).await.map_err(|e| e.to_string()),
            Err(error) => Err(error),
        },
    };
    Ok(match outcome {
        Ok(snapshot) => ToolResult::json(payload(&snapshot)),
        Err(message) => ToolResult::error(message),
    })
}

/// The `todo` [`Tool`], keyed by the caller's thread.
pub struct SessionTodoTool {
    store: Arc<dyn Store>,
}

impl SessionTodoTool {
    /// Creates the tool backed by `store`.
    pub fn new(store: Arc<dyn Store>) -> Self {
        Self { store }
    }
}

/// Registers the session-list `todo` tool into a tool registry.
pub fn register_session_todo_tool<State: Send + Sync, Ctx: Send + Sync>(
    registry: &mut ToolRegistry<State, Ctx>,
    store: Arc<dyn Store>,
) -> &mut ToolRegistry<State, Ctx> {
    registry.register(Arc::new(SessionTodoTool::new(store)));
    registry
}

#[async_trait]
impl Tool for SessionTodoTool {
    fn name(&self) -> &str {
        TOOL_NAME
    }

    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "todos": {
                    "type": ["array", "null"],
                    "description": "The full list, in order. Pass null to read the current list.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": { "type": "string" },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"]
                            }
                        },
                        "required": ["content", "status"]
                    }
                }
            }
        })
    }

    fn policy(&self) -> ToolPolicy {
        ToolPolicy {
            classified: true,
            side_effects: ToolSideEffects {
                read_only: false,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    async fn execute(&self, _args: Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::error(
            "todo tool requires an active thread (no thread_id in tool context)",
        ))
    }

    async fn execute_with_context(
        &self,
        args: Value,
        _options: tinytools::ToolCallOptions,
        context: Option<&dyn ToolRunContext>,
    ) -> anyhow::Result<ToolResult> {
        let Some(key) = context.and_then(ToolRunContext::thread_id) else {
            return self.execute(args).await;
        };
        Ok(call(&self.store, key, &args).await?)
    }
}
