//! [`ContextCompressionMiddleware`] with the typed task-state summarizer and a
//! token-budgeted tail, driven through the real agent loop: the checkpoint is
//! a user-role task state after the system prompt, the original task and file
//! lists survive repeated compactions, and later compactions update the
//! previous state instead of re-summarizing it.

use std::sync::Arc;

use serde_json::json;

use crate::context::{RunConfig, RunContext};
use crate::middleware::ContextCompressionMiddleware;
use crate::runtime::AgentHarness;
use crate::summarization::{
    SummarizationPolicy, TaskStateSummarizer, is_checkpoint, task_state::TASK_STATE_HEADER,
};
use crate::testkit::{FakeTool, ScriptedModel};
use tinyinference_llm::message::{AssistantMessage, Message};
use tinyinference_llm::model::ModelResponse;
use tinyinference_llm::tool::ToolCall;

fn shell_turn(i: usize) -> ModelResponse {
    let mut response = ModelResponse::assistant("");
    response.message = AssistantMessage {
        id: None,
        content: Vec::new(),
        tool_calls: vec![ToolCall::new(
            format!("c{i}"),
            "shell",
            json!({ "command": format!("cat src/file{i}.rs") }),
        )],
        usage: None,
        origin: None,
    };
    response
}

#[tokio::test]
async fn task_state_checkpoints_carry_the_task_and_files_across_compactions() {
    let mut responses: Vec<ModelResponse> = (0..10).map(shell_turn).collect();
    responses.push(ModelResponse::assistant("done"));
    let model = Arc::new(ScriptedModel::new(responses));
    let state_model = Arc::new(ScriptedModel::replies(
        (0..10)
            .map(|i| format!("{{\"goal\": \"fix the bug\", \"todos_open\": [\"step {i}\"]}}"))
            .collect::<Vec<_>>(),
    ));

    let policy = SummarizationPolicy::default().with_trigger_override(1_200);
    let mw = Arc::new(
        ContextCompressionMiddleware::with_summarizer(
            policy,
            Box::new(TaskStateSummarizer::new(state_model.clone(), "state")),
        )
        .with_keep_recent_tokens(400),
    );
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::clone(&model) as _);
    harness.register_tool(Arc::new(FakeTool::returning("shell", "r".repeat(800))));
    harness.push_middleware(mw.clone());
    harness.push_model_middleware(mw.clone());

    let ctx = RunContext::new(RunConfig::new("task-state"), ());
    harness
        .invoke_in_context(
            &(),
            ctx,
            vec![
                Message::system("You are a coding agent."),
                Message::user("fix the bug in src"),
            ],
        )
        .await
        .expect("run succeeds");

    let state_calls = state_model.requests();
    assert!(
        state_calls.len() >= 2,
        "expected repeated compactions, got {}",
        state_calls.len()
    );
    // Later compactions update the previous state rather than re-reading it as history.
    let later = state_calls.last().unwrap().messages[1].text();
    assert!(later.contains("<previous_state>"), "{later}");
    assert!(
        !later.contains(TASK_STATE_HEADER),
        "the old checkpoint must not be re-summarized as history"
    );

    // The last main-model request: system prompt, then the task-state checkpoint as a user message.
    let requests = model.requests();
    let last = requests.last().unwrap();
    assert!(matches!(last.messages[0], Message::System(_)));
    assert!(matches!(last.messages[1], Message::User(_)) && is_checkpoint(&last.messages[1]));
    let checkpoint = last.messages[1].text();
    assert!(checkpoint.contains(TASK_STATE_HEADER));
    assert!(checkpoint.contains("<original-task>\nfix the bug in src\n</original-task>"));
    // Files read before the first compaction are still listed after the last one.
    assert!(checkpoint.contains("src/file0.rs"), "{checkpoint}");
    assert_eq!(
        last.messages.iter().filter(|m| is_checkpoint(m)).count(),
        1,
        "exactly one checkpoint in the request"
    );
}
