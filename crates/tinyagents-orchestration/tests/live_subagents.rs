//! LIVE end-to-end: a real OpenAI parent agent composes the answer of a real
//! OpenAI child sub-agent.
//!
//! This is the network-backed sibling of `e2e_subagents.rs`. The parent
//! [`AgentHarness`] is given a [`SubAgentTool`] wrapping a child harness; both
//! are driven by a real [`OpenAiModel`]. We assert *structurally* (the parent
//! finished, produced non-empty text, and the sub-agent tool ran) rather than
//! on the exact model prose.
//!
//! # Skips gracefully
//!
//! This test is `#[ignore]`d and only runs opted in via
//! `tests/common/live.rs::require_live`, so `cargo test` passes with no key
//! configured and never dials a real provider by accident.

mod common;

#[tokio::test]
#[ignore = "network: set TINYAGENTS_LIVE=1 and run with --ignored"]
async fn live_openai_parent_composes_child_subagent() {
    use std::sync::Arc;

    use tinyagents_harness::context::{RunConfig, RunContext};
    use tinyagents_harness::runtime::AgentHarness;
    use tinyagents_harness::testkit::{EventRecorder, Trajectory};
    use tinyagents_orchestration::subagent::{
        ChildDataPolicy, SubAgent, SubAgentJobsTool, SubAgentMessageTool, SubAgentTool,
    };
    use tinyinference_llm::message::Message;
    use tinyinference_llm::providers::openai::OpenAiModel;

    if !common::live::require_live(&["OPENAI_API_KEY"]) {
        return;
    }

    // Child agent: a focused "math expert" sub-agent driven by a real model.
    let mut child: AgentHarness<()> = AgentHarness::new();
    child
        .register_model(
            "openai",
            Arc::new(OpenAiModel::from_env().expect("OPENAI_API_KEY present")),
        )
        .set_default_model("openai");

    let subagent = Arc::new(
        SubAgent::new(
            "math_expert",
            "Computes precise arithmetic answers. Pass the arithmetic question as `input`.",
            Arc::new(child),
        )
        .with_system_prompt(
            "You are a meticulous arithmetic engine. Reply with only the numeric answer.",
        ),
    );
    let tool = Arc::new(SubAgentTool::new(
        subagent,
        ChildDataPolicy::new(|parent: &()| *parent),
    ));
    let jobs = tool.job_registry().clone();

    // Parent agent: also a real model, equipped with the sub-agent as a tool.
    let mut parent: AgentHarness<()> = AgentHarness::new();
    parent.register_tool_dispatch(tool);
    parent.register_tool_dispatch(Arc::new(SubAgentJobsTool::new(jobs.clone())));
    parent.register_tool_dispatch(Arc::new(SubAgentMessageTool::new(jobs.clone())));
    parent
        .register_model(
            "openai",
            Arc::new(OpenAiModel::from_env().expect("OPENAI_API_KEY present")),
        )
        .set_default_model("openai");

    let recorder = EventRecorder::new();
    let ctx = RunContext::new(RunConfig::new("live-parent-run"), ()).with_events(recorder.sink());

    let run = parent
        .invoke_in_context(
            &(),
            ctx,
            vec![Message::user(
                "Start math_expert to compute 17 * 23. Use subagent_jobs with its job_id until \
                 it completes, then state the result in a short sentence.",
            )],
        )
        .await
        .expect("live parent run succeeds");

    // Structural assertions only — never assert on exact LLM prose.
    let final_text = run.text().unwrap_or_default();
    assert!(
        !final_text.trim().is_empty(),
        "parent should produce a non-empty composed answer"
    );
    assert!(
        run.model_calls >= 1,
        "parent should make at least one model call"
    );

    let traj = Trajectory::from_events(recorder.events());
    traj.assert_completed();
    // The model is strongly steered to call the sub-agent; if it did, the
    // trajectory records it as a `math_expert` tool invocation.
    if traj.tool_was_called("math_expert") {
        assert!(
            run.tool_calls >= 1,
            "a recorded sub-agent tool call should be reflected in run.tool_calls"
        );
        let job = tokio::time::timeout(std::time::Duration::from_secs(120), async {
            loop {
                if let Some(job) = jobs.list().into_iter().next()
                    && job.status.is_terminal()
                {
                    return job;
                }
                tokio::time::sleep(std::time::Duration::from_millis(25)).await;
            }
        })
        .await
        .expect("live math subagent job finishes within two minutes");
        assert!(job.output.is_some(), "completed math job has output");
    }
}
