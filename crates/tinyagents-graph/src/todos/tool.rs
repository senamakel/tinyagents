//! `todo` — the harness [`Tool`] over the per-thread todo list.
//!
//! One call writes the whole list: `{"todos": [{"content", "status"}]}`. There
//! is no per-item CRUD; the list is a progress checklist the model rewrites as
//! it works. Omitting `todos` reads the current list. The list is bound to the
//! caller's [`ToolExecutionContext::thread_id`] (never a tool argument), so a
//! model can't address another thread's list; the bare [`Tool::call`] entry
//! point (no context) errors. Returns the updated items plus a markdown
//! rendering.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use super::store;
use super::types::{TodoItem, TodoStatus, parse_status};
use tinyagents_harness::error::Result;
use tinyagents_harness::store::Store;
use tinyagents_harness::tool::ToolRegistry;
use tinytools::{Tool, ToolPolicy, ToolResult, ToolRunContext, ToolSideEffects};

const TODO_TOOL_NAME: &str = "todo";

const TODO_DESCRIPTION: &str = "Your todo list for this thread. Pass the complete list every \
    time; it replaces what was there. Use it for work with 3+ steps: write the steps up front, \
    keep exactly one `in_progress`, mark each `completed` the moment it is done. Omit `todos` \
    to read the current list. The list is bound automatically to the current thread — do not \
    pass a thread id.";

/// The `todo` harness [`Tool`], backed by a [`Store`](tinyagents_harness::store::Store).
pub struct TodoTool {
    store: Arc<dyn Store>,
}

/// One item as the model writes it. `status` accepts the Claude-style
/// `pending` / `in_progress` / `completed` plus the aliases [`parse_status`]
/// knows (`todo`, `done`, ...).
#[derive(Deserialize)]
struct TodoArg {
    content: String,
    #[serde(default)]
    status: Option<String>,
}

impl TodoTool {
    /// Creates the `todo` tool backed by `store`.
    pub fn new(store: Arc<dyn Store>) -> Self {
        Self { store }
    }

    async fn dispatch(&self, thread_id: &str, args: &Value) -> Result<TodoOutcome> {
        let snap = match args.get("todos") {
            None | Some(Value::Null) => store::list(&self.store, thread_id).await,
            Some(raw) => {
                let items = match parse_items(raw) {
                    Ok(items) => items,
                    Err(message) => return Ok(TodoOutcome::Error(message)),
                };
                store::replace(&self.store, thread_id, items).await
            }
        };
        // A domain error (invariant violation) is surfaced to the model rather
        // than failing the whole run.
        match snap {
            Ok(snap) => Ok(TodoOutcome::Ok(json!({
                "threadId": snap.thread_id,
                "todos": snap.items,
                "markdown": snap.markdown,
            }))),
            Err(e) => Ok(TodoOutcome::Error(e.to_string())),
        }
    }
}

/// Internal dispatch outcome: a structured payload or a model-facing error.
enum TodoOutcome {
    Ok(Value),
    Error(String),
}

/// Decodes the model's `todos` array into items, rejecting blank content and
/// unknown statuses with a message the model can act on.
fn parse_items(raw: &Value) -> std::result::Result<Vec<TodoItem>, String> {
    let args: Vec<TodoArg> =
        serde_json::from_value(raw.clone()).map_err(|e| format!("invalid `todos`: {e}"))?;
    let mut items = Vec::with_capacity(args.len());
    for arg in args {
        let content = arg.content.trim();
        if content.is_empty() {
            return Err("every todo needs non-empty `content`".to_string());
        }
        let status = match arg.status.as_deref() {
            None => TodoStatus::Pending,
            Some(raw) => parse_status(raw)?,
        };
        items.push(TodoItem::with_status(content, status));
    }
    Ok(items)
}

fn parameters_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "The full list, in order.",
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

fn error_result(message: impl Into<String>) -> ToolResult {
    ToolResult::error(message)
}

/// Builds the `todo` tool backed by `store`.
pub fn todo_tools(store: Arc<dyn Store>) -> Vec<Arc<TodoTool>> {
    vec![Arc::new(TodoTool::new(store))]
}

/// Registers the `todo` tool into a tool registry.
pub fn register_todo_tools<State: Send + Sync, Ctx: Send + Sync>(
    registry: &mut ToolRegistry<State, Ctx>,
    store: Arc<dyn Store>,
) -> &mut ToolRegistry<State, Ctx> {
    registry.register(Arc::new(TodoTool::new(store)));
    registry
}

#[async_trait]
impl Tool for TodoTool {
    fn name(&self) -> &str {
        TODO_TOOL_NAME
    }

    fn description(&self) -> &str {
        TODO_DESCRIPTION
    }

    fn parameters_schema(&self) -> Value {
        parameters_schema()
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
        Ok(error_result(
            "todo tool requires an active thread (no thread_id in tool context)",
        ))
    }

    async fn execute_with_context(
        &self,
        args: Value,
        _options: tinytools::ToolCallOptions,
        context: Option<&dyn ToolRunContext>,
    ) -> anyhow::Result<ToolResult> {
        let Some(thread_id) = context.and_then(ToolRunContext::thread_id) else {
            return Ok(error_result(
                "todo tool requires an active thread (no thread_id in tool context)",
            ));
        };
        match self.dispatch(thread_id, &args).await? {
            TodoOutcome::Ok(payload) => Ok(ToolResult::json(payload)),
            TodoOutcome::Error(message) => Ok(error_result(message)),
        }
    }
}
