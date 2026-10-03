//! The deterministic half of a task-state checkpoint: facts read straight out
//! of the transcript's tool calls and results, with no model involved.
//!
//! These are the facts free-form summaries lose first (which files were
//! touched, which command failed and with what error), and the ones a model of
//! any size gets right for free when they are copied instead of recalled.

use std::collections::HashMap;

use serde_json::Value;
use tinyinference_llm::message::Message;

use super::types::{CommandRecord, TaskLedger};
use crate::summarization::is_checkpoint;

/// Longest original task statement carried verbatim.
pub(crate) const MAX_TASK_CHARS: usize = 6_000;
/// Commands kept in the ledger (the most recent ones).
pub(crate) const MAX_COMMANDS: usize = 15;
/// Longest single command line kept.
const MAX_COMMAND_CHARS: usize = 200;

impl TaskLedger {
    /// Folds the facts in `messages` into this ledger (which may carry facts
    /// from earlier checkpoints). Files keep first-seen order; commands keep
    /// the most recent [`MAX_COMMANDS`].
    pub fn absorb(&mut self, messages: &[Message]) {
        if self.original_task.is_none() {
            self.original_task = messages
                .iter()
                .find(|m| matches!(m, Message::User(_)) && !is_checkpoint(m))
                .map(|m| truncate_chars(m.text().trim(), MAX_TASK_CHARS))
                .filter(|t| !t.is_empty());
        }

        let results: HashMap<&str, String> = messages
            .iter()
            .filter_map(|m| match m {
                Message::Tool(t) => Some((t.tool_call_id.as_str(), m.text())),
                _ => None,
            })
            .collect();

        for message in messages {
            let Message::Assistant(assistant) = message else {
                continue;
            };
            for call in &assistant.tool_calls {
                let result = results.get(call.id.as_str()).map(String::as_str);
                self.absorb_call(&call.name, &call.arguments, result);
            }
        }
        if self.commands.len() > MAX_COMMANDS {
            self.commands.drain(..self.commands.len() - MAX_COMMANDS);
        }
    }

    fn absorb_call(&mut self, name: &str, arguments: &Value, result: Option<&str>) {
        let name = name.to_ascii_lowercase();
        match ToolKind::of(&name) {
            ToolKind::Shell => {
                let Some(command) = shell_command(arguments) else {
                    return;
                };
                let (reads, writes) = shell_files(&command);
                for path in writes {
                    push_unique(&mut self.files_modified, path);
                }
                for path in reads {
                    push_unique(&mut self.files_read, path);
                }
                let failure = result.and_then(failure_of);
                let first_line = command.trim().lines().next().unwrap_or_default();
                self.commands.push(CommandRecord {
                    command: truncate_chars(first_line, MAX_COMMAND_CHARS),
                    failed: failure.is_some(),
                    error: failure.and_then(error_signature),
                });
            }
            ToolKind::Edit => {
                for path in argument_paths(arguments)
                    .into_iter()
                    .chain(patch_paths(arguments))
                {
                    push_unique(&mut self.files_modified, path);
                }
            }
            ToolKind::Read => {
                for path in argument_paths(arguments) {
                    push_unique(&mut self.files_read, path);
                }
            }
            ToolKind::Other => {}
        }
    }

    /// Files read that were not also modified (a modified file is listed once).
    #[must_use]
    pub fn read_only_files(&self) -> Vec<&str> {
        self.files_read
            .iter()
            .filter(|f| !self.files_modified.contains(f))
            .map(String::as_str)
            .collect()
    }
}

/// What a tool call does to the workspace, judged from its name.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ToolKind {
    Shell,
    Edit,
    Read,
    Other,
}

impl ToolKind {
    pub(crate) fn of(lower_name: &str) -> Self {
        const SHELL: &[&str] = &[
            "shell",
            "bash",
            "exec",
            "terminal",
            "run_command",
            "run_shell_command",
            "execute_command",
            "command",
            "sh",
            "local_shell",
        ];
        if SHELL.contains(&lower_name) {
            return Self::Shell;
        }
        if ["edit", "patch", "replace", "write", "create_file"]
            .iter()
            .any(|k| lower_name.contains(k))
        {
            return Self::Edit;
        }
        if ["read", "view", "open_file", "cat"]
            .iter()
            .any(|k| lower_name.contains(k))
        {
            return Self::Read;
        }
        Self::Other
    }
}

fn shell_command(arguments: &Value) -> Option<String> {
    let value = ["command", "cmd", "script"]
        .iter()
        .find_map(|k| arguments.get(*k))?;
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Array(parts) => Some(
            parts
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(" "),
        ),
        _ => None,
    }
}

fn argument_paths(arguments: &Value) -> Vec<String> {
    [
        "path",
        "file_path",
        "filePath",
        "filename",
        "target_file",
        "file",
    ]
    .iter()
    .filter_map(|k| arguments.get(*k).and_then(Value::as_str))
    .filter_map(clean_path)
    .collect()
}

/// Paths named by a patch carried in any string argument (`*** Update File:`
/// apply_patch envelopes and unified-diff `+++ b/…` headers).
fn patch_paths(arguments: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let Value::Object(map) = arguments else {
        return out;
    };
    for value in map.values() {
        let Some(text) = value.as_str() else { continue };
        for line in text.lines() {
            let path = [
                "*** Update File:",
                "*** Add File:",
                "*** Delete File:",
                "+++ b/",
                "+++ ",
            ]
            .iter()
            .find_map(|prefix| line.strip_prefix(prefix));
            if let Some(path) = path.and_then(|p| clean_path(p.trim()))
                && path != "/dev/null"
            {
                push_unique(&mut out, path);
            }
        }
    }
    out
}

/// Files a shell command reads and writes, from the common idioms: output
/// redirection, `tee`, `sed -i`, `cp`/`mv` targets, `touch`, and
/// `cat`/`head`/`tail`/`nl`/`sed -n` reads. Heredoc bodies are skipped.
pub(crate) fn shell_files(command: &str) -> (Vec<String>, Vec<String>) {
    let mut reads = Vec::new();
    let mut writes = Vec::new();
    for segment in segments(&strip_heredoc_bodies(command)) {
        let tokens = tokenize(&segment);
        let mut words: Vec<&str> = Vec::new();
        let mut iter = tokens.iter().map(String::as_str).peekable();
        while let Some(token) = iter.next() {
            if let Some(rest) = token
                .strip_prefix(">>")
                .or_else(|| token.strip_prefix('>'))
                .or_else(|| token.strip_prefix("1>"))
            {
                let target = if rest.is_empty() {
                    iter.next().unwrap_or("")
                } else {
                    rest
                };
                if let Some(path) = clean_path(target) {
                    push_unique(&mut writes, path);
                }
                continue;
            }
            if token.starts_with("2>") || token.starts_with('<') {
                if token == "2>" || token == "<" {
                    iter.next();
                }
                continue;
            }
            words.push(token);
        }
        let Some((&program, args)) = words.split_first() else {
            continue;
        };
        let program = program.rsplit('/').next().unwrap_or(program);
        let operands = || args.iter().filter(|a| !a.starts_with('-')).copied();
        match program {
            "tee" => operands()
                .filter_map(clean_path)
                .for_each(|p| push_unique(&mut writes, p)),
            "touch" => operands()
                .filter_map(clean_path)
                .for_each(|p| push_unique(&mut writes, p)),
            "cp" | "mv" => {
                if let Some(p) = operands().last().and_then(clean_path) {
                    push_unique(&mut writes, p);
                }
            }
            "sed" => {
                let in_place = args.iter().any(|a| a.starts_with("-i"));
                // The script is the first operand; files follow it.
                let files: Vec<String> = operands().skip(1).filter_map(clean_path).collect();
                let target = if in_place { &mut writes } else { &mut reads };
                files.into_iter().for_each(|p| push_unique(target, p));
            }
            "cat" | "head" | "tail" | "nl" | "less" | "more" => {
                operands()
                    .filter_map(clean_path)
                    .for_each(|p| push_unique(&mut reads, p));
            }
            _ => {}
        }
    }
    (reads, writes)
}

/// Drops the bodies of heredocs (`<<EOF … EOF`), which are data, not commands.
fn strip_heredoc_bodies(command: &str) -> String {
    let mut out = Vec::new();
    let mut terminator: Option<String> = None;
    for line in command.lines() {
        if let Some(end) = &terminator {
            if line.trim() == end {
                terminator = None;
            }
            continue;
        }
        if let Some(at) = line.find("<<") {
            // `cat <<'EOF' > path`: the tag, then the rest of the command line.
            let after = line[at + 2..].trim_start_matches('-').trim_start();
            let after = after.trim_start_matches(['\'', '"']);
            let tag: String = after
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            let rest = after[tag.len()..].trim_start_matches(['\'', '"']);
            if !tag.is_empty() {
                terminator = Some(tag);
            }
            out.push(format!("{} {rest}", &line[..at]));
            continue;
        }
        out.push(line.to_string());
    }
    out.join("\n")
}

/// Splits a command line on `&&`, `||`, `;`, `|` and newlines (outside quotes).
fn segments(command: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut chars = command.chars().peekable();
    while let Some(c) = chars.next() {
        match (quote, c) {
            (Some(q), c) if c == q => {
                quote = None;
                current.push(c);
            }
            (Some(_), c) => current.push(c),
            (None, '\'' | '"') => {
                quote = Some(c);
                current.push(c);
            }
            (None, ';' | '\n' | '|') => {
                if c == '|' && chars.peek() == Some(&'|') {
                    chars.next();
                }
                out.push(std::mem::take(&mut current));
            }
            (None, '&') if chars.peek() == Some(&'&') => {
                chars.next();
                out.push(std::mem::take(&mut current));
            }
            (None, c) => current.push(c),
        }
    }
    out.push(current);
    out.into_iter().filter(|s| !s.trim().is_empty()).collect()
}

/// Whitespace tokenizer that honours single and double quotes.
fn tokenize(segment: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut started = false;
    for c in segment.chars() {
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), c) => current.push(c),
            (None, '\'' | '"') => {
                quote = Some(c);
                started = true;
            }
            (None, c) if c.is_whitespace() => {
                if started || !current.is_empty() {
                    out.push(std::mem::take(&mut current));
                }
                started = false;
            }
            (None, c) => current.push(c),
        }
    }
    if started || !current.is_empty() {
        out.push(current);
    }
    out
}

/// A token that plausibly names a file: no shell metacharacters, not a flag,
/// not a device, and with a path separator or an extension.
fn clean_path(raw: &str) -> Option<String> {
    let path = raw.trim().trim_start_matches("./");
    let looks_like_file = !path.is_empty()
        && !path.starts_with('-')
        && !path.starts_with("/dev/")
        && !path.starts_with('&')
        && !path.contains([
            '$', '*', '?', '`', '(', ')', '{', '}', '=', '\'', '"', '<', '>',
        ])
        && (path.contains('/') || path.contains('.'))
        && path.len() <= 300;
    looks_like_file.then(|| path.to_string())
}

fn push_unique(list: &mut Vec<String>, value: String) {
    if !list.contains(&value) {
        list.push(value);
    }
}

pub(crate) fn truncate_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let kept: String = text.chars().take(max).collect();
    format!("{kept}…")
}

/// The output of a failed command, or `None` when the result reads as a
/// success. Recognises exit-code lines (`Command failed (exit code 1)`,
/// `exit code: 2`, `"exit_code": 1`) and common failure markers. Exit 141
/// (SIGPIPE from `| head`) is a success.
pub(crate) fn failure_of(result: &str) -> Option<&str> {
    let lower = result.to_ascii_lowercase();
    for anchor in [
        "exit code",
        "exit_code\":",
        "exited with code",
        "exit status",
    ] {
        if let Some(at) = lower.find(anchor) {
            let code: String = lower[at + anchor.len()..]
                .chars()
                .skip_while(|c| !c.is_ascii_digit() && *c != '-')
                .take_while(|c| c.is_ascii_digit() || *c == '-')
                .collect();
            if let Ok(code) = code.parse::<i64>() {
                return (code != 0 && code != 141).then_some(result);
            }
        }
    }
    let failed = result.contains("Traceback (most recent call last)")
        || result.lines().any(|l| {
            let l = l.trim_start();
            l.starts_with("panic: ")
                || l.starts_with("FAIL")
                || l.starts_with("--- FAIL")
                || l.starts_with("error[E")
                || l.starts_with("error:")
                || l.starts_with("Error:")
                || l.starts_with("fatal:")
        });
    failed.then_some(result)
}

/// The most informative single line of a failure's output.
pub(crate) fn error_signature(failure: &str) -> Option<String> {
    let rank = |line: &str| -> u8 {
        let s = line.trim();
        if s.is_empty() || s.len() > 400 || s.to_ascii_lowercase().contains("exit code") {
            return 0;
        }
        let first = s.split(':').next().unwrap_or_default();
        if (first.ends_with("Error") || first.ends_with("Exception")) && !first.contains(' ') {
            return 9;
        }
        if s.starts_with("panic: ") {
            return 9;
        }
        if s.starts_with("error[E") || s.contains("error TS") {
            return 8;
        }
        // file:line: message (compilers, go test).
        let mut parts = s.splitn(3, ':');
        if let (Some(file), Some(line), Some(_)) = (parts.next(), parts.next(), parts.next())
            && file.contains('.')
            && !file.contains(' ')
            && line.trim().parse::<u32>().is_ok()
        {
            return 7;
        }
        if s.starts_with("FAILED ") || s.starts_with("--- FAIL") {
            return 6;
        }
        if s.contains("AssertionError") || s.starts_with("assert ") {
            return 5;
        }
        if s.starts_with("error:") || s.starts_with("Error:") || s.starts_with("fatal:") {
            return 4;
        }
        if s.contains("Error") || s.contains("FAIL") || s.contains("failed") {
            return 2;
        }
        0
    };
    let mut best: Option<(&str, u8)> = None;
    for line in failure.lines() {
        let r = rank(line);
        if r > best.map_or(0, |(_, b)| b) {
            best = Some((line.trim(), r));
        }
    }
    best.map(|(line, _)| truncate_chars(line, 300))
}

#[cfg(test)]
#[path = "ledger_tests.rs"]
mod tests;
