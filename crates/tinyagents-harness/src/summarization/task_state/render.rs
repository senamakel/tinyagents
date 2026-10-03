//! Rendering a task state as the checkpoint text, and reading the carried
//! facts back out of a previous checkpoint.
//!
//! The deterministic facts are written inside XML-ish tags so the next
//! compaction can carry them forward exactly, without asking a model to
//! remember them: `<original-task>`, `<modified-files>` and `<read-files>`.
//! The model-written fields are read back from their `## ` sections, so the
//! state is written once (a JSON copy beside the sections doubled every
//! checkpoint).

use super::types::{CommandRecord, TaskLedger, TaskState};

/// First line of every task-state checkpoint body.
pub const TASK_STATE_HEADER: &str = "# Task state (compacted)";

/// Renders the checkpoint body: readable sections for the agent, plus tagged
/// blocks the next compaction parses back.
#[must_use]
pub fn render_task_state(state: &TaskState, ledger: &TaskLedger) -> String {
    // One line per item and per scalar: a newline inside a value would read
    // back as a new item or section.
    let list = |items: &[String]| -> String {
        if items.is_empty() {
            NONE_ITEM.to_string()
        } else {
            items
                .iter()
                .map(|i| format!("- {}", one_line(i)))
                .collect::<Vec<_>>()
                .join("\n")
        }
    };
    let or_none = |s: &str| {
        if s.trim().is_empty() {
            NONE.to_string()
        } else {
            one_line(s)
        }
    };
    let commands = if ledger.commands.is_empty() {
        "- none".to_string()
    } else {
        ledger
            .commands
            .iter()
            .map(render_command)
            .collect::<Vec<_>>()
            .join("\n")
    };
    let task = ledger.original_task.as_deref().unwrap_or("");
    format!(
        "{TASK_STATE_HEADER}\n\n\
         <original-task>\n{task}\n</original-task>\n\n\
         ## Goal\n{goal}\n\n\
         ## Requirements (verbatim)\n{requirements}\n\n\
         ## Constraints\n{constraints}\n\n\
         ## Decisions\n{decisions}\n\n\
         ## Errors and fixes\n{errors}\n\n\
         ## Done\n{done}\n\n\
         ## Open\n{open}\n\n\
         ## Current hypothesis\n{hypothesis}\n\n\
         ## Test command\n{test}\n\n\
         ## Next step\n{next}\n\n\
         ## Recent commands\n{commands}\n\n\
         <modified-files>\n{modified}\n</modified-files>\n\
         <read-files>\n{read}\n</read-files>",
        goal = or_none(&state.goal),
        requirements = list(&state.requirements),
        constraints = list(&state.constraints),
        decisions = list(&state.decisions),
        errors = list(&state.errors_and_fixes),
        done = list(&state.todos_done),
        open = list(&state.todos_open),
        hypothesis = or_none(&state.current_hypothesis),
        test = or_none(&state.test_command),
        next = or_none(&state.next_step),
        modified = ledger.files_modified.join("\n"),
        read = ledger.read_only_files().join("\n"),
    )
}

const NONE: &str = "none";
const NONE_ITEM: &str = "- none";

/// Section headings of the model-written fields, in render order.
const GOAL: &str = "## Goal";
const REQUIREMENTS: &str = "## Requirements (verbatim)";
const CONSTRAINTS: &str = "## Constraints";
const DECISIONS: &str = "## Decisions";
const ERRORS: &str = "## Errors and fixes";
const DONE: &str = "## Done";
const OPEN: &str = "## Open";
const HYPOTHESIS: &str = "## Current hypothesis";
const TEST: &str = "## Test command";
const NEXT: &str = "## Next step";

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The body of the `heading` section: the lines up to the next `## `
/// heading or tagged block.
fn section<'a>(text: &'a str, heading: &str) -> Option<&'a str> {
    let at = text.find(&format!("\n{heading}\n"))? + heading.len() + 2;
    let rest = &text[at..];
    let end = rest
        .find("\n## ")
        .into_iter()
        .chain(rest.find("\n<"))
        .min()
        .unwrap_or(rest.len());
    Some(rest[..end].trim())
}

fn section_items(text: &str, heading: &str) -> Vec<String> {
    section(text, heading)
        .map(|body| {
            body.lines()
                .filter_map(|l| l.trim().strip_prefix("- "))
                .map(str::trim)
                .filter(|l| !l.is_empty() && *l != NONE)
                .map(String::from)
                .collect()
        })
        .unwrap_or_default()
}

fn section_text(text: &str, heading: &str) -> String {
    section(text, heading)
        .filter(|body| *body != NONE)
        .map(String::from)
        .unwrap_or_default()
}

fn render_command(c: &CommandRecord) -> String {
    match (&c.failed, &c.error) {
        (false, _) => format!("- `{}` → ok", c.command),
        (true, Some(error)) => format!("- `{}` → FAILED: {error}", c.command),
        (true, None) => format!("- `{}` → FAILED", c.command),
    }
}

/// The content of the first `<tag>…</tag>` block in `text`, trimmed.
fn tagged<'a>(text: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = text.find(&open)? + open.len();
    let end = start + text[start..].find(&close)?;
    Some(text[start..end].trim())
}

/// Reads the facts a previous checkpoint carries: its ledger (task and file
/// lists; commands are not carried, the recent ones are re-read) and its
/// model-written state. A previous summary that is not a task-state
/// checkpoint (a free-form summary from another summarizer) yields an empty
/// ledger and `None`; the caller then hands its text to the model instead.
#[must_use]
pub fn parse_carried(previous: &str) -> (TaskLedger, Option<TaskState>) {
    let lines = |tag: &str| -> Vec<String> {
        tagged(previous, tag)
            .map(|block| {
                block
                    .lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty())
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default()
    };
    let ledger = TaskLedger {
        original_task: tagged(previous, "original-task")
            .filter(|t| !t.is_empty())
            .map(String::from),
        files_modified: lines("modified-files"),
        files_read: lines("read-files"),
        commands: Vec::new(),
    };
    let state = previous
        .trim_start()
        .starts_with(TASK_STATE_HEADER)
        .then(|| TaskState {
            goal: section_text(previous, GOAL),
            requirements: section_items(previous, REQUIREMENTS),
            constraints: section_items(previous, CONSTRAINTS),
            decisions: section_items(previous, DECISIONS),
            errors_and_fixes: section_items(previous, ERRORS),
            todos_done: section_items(previous, DONE),
            todos_open: section_items(previous, OPEN),
            current_hypothesis: section_text(previous, HYPOTHESIS),
            test_command: section_text(previous, TEST),
            next_step: section_text(previous, NEXT),
        });
    (ledger, state)
}

/// The JSON object in a model reply, tolerating prose or code fences around
/// it. `None` when no object parses.
#[must_use]
pub fn parse_state_reply(reply: &str) -> Option<TaskState> {
    let start = reply.find('{')?;
    let end = reply.rfind('}')?;
    if end <= start {
        return None;
    }
    serde_json::from_str(&reply[start..=end]).ok()
}

#[cfg(test)]
#[path = "render_tests.rs"]
mod tests;
