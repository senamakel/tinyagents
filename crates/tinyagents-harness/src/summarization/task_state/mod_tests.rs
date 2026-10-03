use super::*;

use serde_json::json;
use tinyinference_llm::message::AssistantMessage;
use tinyinference_llm::tool::ToolCall;

use crate::testkit::ScriptedModel;

fn shell(id: &str, command: &str, output: &str) -> Vec<Message> {
    vec![
        Message::Assistant(AssistantMessage {
            id: None,
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(id, "shell", json!({ "command": command }))],
            usage: None,
            origin: None,
        }),
        Message::tool(id, output),
    ]
}

fn history() -> Vec<Message> {
    let mut m = vec![Message::user("Implement default arguments in anko.")];
    m.extend(shell("c1", "cat vm/vm.go", "package vm"));
    m.extend(shell(
        "c2",
        "go test ./vm/...",
        "FAIL\nvm_test.go:9: boom\nCommand failed (exit code 1)",
    ));
    m
}

const STATE_REPLY: &str = r#"{"goal": "default args", "requirements": ["invalid default argument declaration"], "todos_open": ["fix error text"], "test_command": "go test ./vm/...", "next_step": "edit parser"}"#;

#[tokio::test]
async fn checkpoint_holds_model_state_and_ledger() {
    let model = Arc::new(ScriptedModel::replies(vec![STATE_REPLY]));
    let summarizer = TaskStateSummarizer::new(model.clone(), "m");
    let record = summarizer.summarize(&history()).await.unwrap();
    let body = record.summary.text();

    assert!(body.starts_with(TASK_STATE_HEADER));
    assert!(body.contains("Implement default arguments in anko."));
    assert!(body.contains("- invalid default argument declaration"));
    assert!(body.contains("<read-files>\nvm/vm.go\n</read-files>"));
    assert!(body.contains("→ FAILED: vm_test.go:9: boom"));
    assert_eq!(model.requests().len(), 1);
    // The transcript is fenced and the instruction comes last.
    let prompt = model.requests()[0].messages[1].text();
    assert!(prompt.contains("<transcript>"));
    assert!(prompt.trim_end().ends_with("error lines exactly."));
}

#[tokio::test]
async fn second_compaction_updates_the_previous_state_and_carries_files() {
    let model = Arc::new(ScriptedModel::replies(vec![
        STATE_REPLY,
        r#"{"goal": "default args", "todos_done": ["fix error text"]}"#,
    ]));
    let summarizer = TaskStateSummarizer::new(model.clone(), "m");
    let first = summarizer.summarize(&history()).await.unwrap();

    let later = shell("c3", "sed -i 's/x/y/' parser/parser.go.y", "");
    let request = SummaryRequest::new(later).with_previous_summary(first.summary.text());
    let second = summarizer.summarize_request(&request).await.unwrap();
    let body = second.summary.text();

    // The previous model state reached the model as JSON to update.
    let prompt = model.requests()[1].messages[1].text();
    assert!(prompt.contains("<previous_state>"));
    assert!(prompt.contains("\"todos_open\":[\"fix error text\"]"));
    // Carried exactly: original task and files from the first window.
    assert!(body.contains("Implement default arguments in anko."));
    assert!(body.contains("<modified-files>\nparser/parser.go.y\n</modified-files>"));
    assert!(body.contains("<read-files>\nvm/vm.go\n</read-files>"));
    assert!(body.contains("## Done\n- fix error text"));
}

#[tokio::test]
async fn markup_then_bad_json_degrades_to_the_ledger_instead_of_failing() {
    let model = Arc::new(ScriptedModel::replies(vec![
        "<｜DSML｜invoke name=\"shell\">",
        "I think the state is fine.",
    ]));
    let summarizer = TaskStateSummarizer::new(model.clone(), "m");
    let record = summarizer.summarize(&history()).await.unwrap();
    assert_eq!(model.requests().len(), 2, "one retry, then give up");
    assert!(record.provenance.reason.contains("ledger only"));
    let body = record.summary.text();
    assert!(body.contains("Implement default arguments in anko."));
    assert!(body.contains("→ FAILED: vm_test.go:9: boom"));
}

#[tokio::test]
async fn a_model_outage_with_nothing_to_fall_back_on_is_an_error() {
    // No replies scripted: every call fails. Plain chat with no tool calls and
    // no user task gives the ledger nothing to carry.
    let model = Arc::new(ScriptedModel::replies(Vec::<&str>::new()));
    let summarizer = TaskStateSummarizer::new(model, "m");
    let err = summarizer
        .summarize(&[Message::assistant("thinking")])
        .await;
    assert!(err.is_err());
}

#[tokio::test]
async fn long_histories_are_folded_in_sequential_chunks() {
    let mut messages = vec![Message::user("task")];
    for i in 0..6 {
        messages.extend(shell(
            &format!("c{i}"),
            &format!("echo {i}"),
            &"x".repeat(400),
        ));
    }
    let replies: Vec<String> = (0..10)
        .map(|i| format!("{{\"goal\": \"step {i}\"}}"))
        .collect();
    let model = Arc::new(ScriptedModel::replies(replies));
    let summarizer = TaskStateSummarizer::new(model.clone(), "m").with_max_chunk_tokens(250);
    let record = summarizer.summarize(&messages).await.unwrap();

    let requests = model.requests();
    assert!(
        requests.len() > 1,
        "expected several chunks, got {}",
        requests.len()
    );
    // Every chunk after the first updates the state the previous one wrote,
    // and no chunk opens on an orphaned tool result.
    for (i, request) in requests.iter().enumerate().skip(1) {
        let prompt = request.messages[1].text();
        assert!(
            prompt.contains(&format!("\"goal\":\"step {}\"", i - 1)),
            "chunk {i}"
        );
        assert!(
            !prompt.contains("<transcript>\ntool:"),
            "chunk {i} starts on a tool result"
        );
    }
    assert!(
        record
            .summary
            .text()
            .contains(&format!("step {}", requests.len() - 1))
    );
}

#[tokio::test]
async fn merge_unions_files_and_keeps_the_later_state() {
    let model = Arc::new(ScriptedModel::replies(vec![
        STATE_REPLY,
        r#"{"goal": "later"}"#,
    ]));
    let summarizer = TaskStateSummarizer::new(model, "m");
    let a = summarizer.summarize(&history()).await.unwrap();
    let b = summarizer
        .summarize(&shell("c9", "touch new/file.rs", ""))
        .await
        .unwrap();
    let merged = summarizer.merge(&[a, b]).await.unwrap();
    let body = merged.summary.text();
    assert!(body.contains("## Goal\nlater"));
    assert!(body.contains("<modified-files>\nnew/file.rs\n</modified-files>"));
    assert!(body.contains("Implement default arguments in anko."));
}

#[test]
fn bounded_caps_lists_and_items() {
    let many = |n: usize| (0..n).map(|i| format!("item {i}")).collect::<Vec<_>>();
    let state = TaskState {
        requirements: many(60),
        decisions: many(40),
        todos_done: many(40),
        todos_open: many(40),
        errors_and_fixes: vec!["e".repeat(5_000)],
        current_hypothesis: "h".repeat(5_000),
        ..TaskState::default()
    }
    .bounded();
    assert_eq!(state.requirements.len(), 40);
    assert_eq!(
        state.requirements[0], "item 0",
        "requirements keep the first"
    );
    assert_eq!(state.decisions.len(), 12);
    assert_eq!(
        state.decisions[11], "item 39",
        "history keeps the most recent"
    );
    assert_eq!(state.todos_done[0], "item 28");
    assert_eq!(state.todos_open[0], "item 0", "open work keeps the oldest");
    assert!(state.errors_and_fixes[0].chars().count() <= 401);
    assert!(state.current_hypothesis.chars().count() <= 801);
}

#[tokio::test]
async fn checkpoint_size_stays_bounded_across_many_compactions() {
    // A model that only ever adds to its lists, as the live DeepSWE run did.
    let replies: Vec<String> = (0..30)
        .map(|i| {
            let grow = |p: &str| (0..(i + 1) * 3).map(|j| format!("\"{p} {j}: {}\"", "x".repeat(200))).collect::<Vec<_>>().join(",");
            format!(
                "{{\"goal\": \"g\", \"decisions\": [{}], \"todos_done\": [{}], \"errors_and_fixes\": [{}]}}",
                grow("decision"),
                grow("done"),
                grow("error")
            )
        })
        .collect();
    let model = Arc::new(ScriptedModel::replies(replies));
    let summarizer = TaskStateSummarizer::new(model, "m");
    let mut previous: Option<String> = None;
    let mut sizes = Vec::new();
    for i in 0..30 {
        let mut request =
            SummaryRequest::new(shell(&format!("c{i}"), &format!("cat src/f{i}.rs"), "ok"));
        if let Some(p) = &previous {
            request = request.with_previous_summary(p.clone());
        }
        let body = summarizer
            .summarize_request(&request)
            .await
            .unwrap()
            .summary
            .text();
        sizes.push(body.len());
        previous = Some(body);
    }
    let last = *sizes.last().unwrap();
    // 3 lists x 12 items x ~215 chars, plus the ledger: well under 12k chars.
    assert!(last < 12_000, "checkpoint grew to {last} chars: {sizes:?}");
}
