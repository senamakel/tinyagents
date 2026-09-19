//! Equivalence coverage for A5: the compiled-graph rendition of the agent
//! loop (`tinyagents_graph::agent_loop`) against the harness's built-in
//! direct loop.
//!
//! Each scenario below runs the identical scripted model/harness setup
//! twice — once with `RunPolicy::execution = LoopExecution::Direct` (the
//! default), once with `LoopExecution::Graph` plus
//! `AgentHarness::with_loop_driver(Arc::new(GraphLoopDriver::new()))` — and
//! asserts the two runs agree on transcript, structured output, and usage,
//! and that the direct run's `AgentEvent` kind sequence appears (in order,
//! extra graph events allowed) within the graph run's sequence. See
//! `docs/modules/harness/state-graph.md` for the documented scope of the
//! graph rendition (it is a subset of the direct loop's behavior, not a
//! byte-for-byte reimplementation).

use std::sync::Arc;

use serde_json::json;

use tinyagents_graph::agent_loop::{
    compile_loop, node, AgentLoopGraphExt, GraphLoopDriver, LoopRuntime, LoopState,
};
use tinyagents_graph::{FileCheckpointer, InMemoryCheckpointer};
use tinyagents_harness::TinyAgentsError;
use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::runtime::{AgentHarness, LoopExecution, RunPolicy};
use tinyagents_harness::steering::{SteeringCommand, SteeringHandle};
use tinyagents_harness::testkit::{EventRecorder, FakeTool};
use tinyinference_llm::message::{AssistantMessage, Message};
use tinyinference_llm::model::{ModelResponse, ResponseFormat};
use tinyinference_llm::providers::MockModel;
use tinyinference_llm::tool::ToolCall;
use tinyinference_llm::usage::Usage;

fn tool_call_response(id: &str, name: &str, arguments: serde_json::Value) -> ModelResponse {
    ModelResponse {
        message: AssistantMessage {
            id: Some(format!("msg-{id}")),
            content: Vec::new(),
            tool_calls: vec![ToolCall::new(id, name, arguments)],
            usage: Some(Usage::new(7, 3)),
            origin: None,
        },
        usage: Some(Usage::new(7, 3)),
        finish_reason: Some("tool_calls".to_string()),
        ..ModelResponse::assistant("")
    }
}

/// Builds a harness for `execution`, registering `model` as the default.
fn harness_for(
    execution: LoopExecution,
    model: Arc<MockModel>,
) -> AgentHarness<()> {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", model).set_default_model("mock");
    if matches!(execution, LoopExecution::Graph) {
        harness.with_loop_driver(Arc::new(GraphLoopDriver::new()));
    }
    let mut policy = RunPolicy::default();
    policy.execution = execution;
    harness.with_policy(policy);
    harness
}

/// Asserts `expected` appears, in order, as a (not necessarily contiguous)
/// subsequence of `actual` — the "same kind sequence, extra graph events
/// allowed" contract.
fn assert_kinds_subsequence(expected: &[String], actual: &[String]) {
    let mut cursor = 0;
    for kind in expected {
        let Some(offset) = actual[cursor..].iter().position(|k| k == kind) else {
            panic!(
                "expected event kind `{kind}` not found (in order) in graph run's kinds: \
                 {actual:?}; direct run's kinds were: {expected:?}"
            );
        };
        cursor += offset + 1;
    }
}

// ── Scenario 1: tool call ───────────────────────────────────────────────────

#[tokio::test]
async fn tool_call_scenario_matches_direct_and_graph() {
    let mut direct_run = None;
    let mut direct_kinds = None;
    let mut graph_run = None;
    let mut graph_kinds = None;

    for execution in [LoopExecution::Direct, LoopExecution::Graph] {
        let model = Arc::new(MockModel::with_responses(vec![
            tool_call_response("call-1", "lookup", json!({ "q": "x" })),
            ModelResponse::assistant("done"),
        ]));
        let mut harness = harness_for(execution, model);
        harness.register_tool(Arc::new(FakeTool::returning("lookup", "tool-output")));

        let recorder = EventRecorder::new();
        let ctx = RunContext::new(RunConfig::new("tool-call"), ()).with_events(recorder.sink());
        let run = harness
            .invoke_in_context(&(), ctx, vec![Message::user("look something up")])
            .await
            .expect("run completes");

        match execution {
            LoopExecution::Direct => {
                direct_run = Some(run);
                direct_kinds = Some(recorder.kinds());
            }
            LoopExecution::Graph => {
                graph_run = Some(run);
                graph_kinds = Some(recorder.kinds());
            }
        }
    }

    let (direct_run, graph_run) = (direct_run.unwrap(), graph_run.unwrap());
    assert_eq!(direct_run.model_calls, graph_run.model_calls);
    assert_eq!(direct_run.tool_calls, graph_run.tool_calls);
    assert_eq!(direct_run.executed_tools, graph_run.executed_tools);
    assert_eq!(direct_run.text(), graph_run.text());
    assert_eq!(direct_run.usage, graph_run.usage);
    assert_kinds_subsequence(&direct_kinds.unwrap(), &graph_kinds.unwrap());
}

// ── Scenario 2: structured output ───────────────────────────────────────────

#[tokio::test]
async fn structured_output_scenario_matches_direct_and_graph() {
    let schema = json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
    });

    let mut direct_run = None;
    let mut graph_run = None;

    for execution in [LoopExecution::Direct, LoopExecution::Graph] {
        let model = Arc::new(MockModel::with_responses(vec![ModelResponse::assistant(
            r#"{"answer":"42"}"#,
        )]));
        let mut harness = harness_for(execution, model);
        let mut policy = harness.policy().clone();
        policy.default_response_format = Some(ResponseFormat::auto("answer", schema.clone()));
        harness.with_policy(policy);

        let run = harness
            .invoke_default(&(), vec![Message::user("what is the answer")])
            .await
            .expect("run completes");

        match execution {
            LoopExecution::Direct => direct_run = Some(run),
            LoopExecution::Graph => graph_run = Some(run),
        }
    }

    let (direct_run, graph_run) = (direct_run.unwrap(), graph_run.unwrap());
    assert_eq!(direct_run.structured, graph_run.structured);
    assert_eq!(direct_run.structured, Some(json!({ "answer": "42" })));
}

// ── Scenario 3: limit stop ──────────────────────────────────────────────────

#[tokio::test]
async fn limit_stop_scenario_matches_direct_and_graph() {
    let mut direct_err = None;
    let mut graph_err = None;

    for execution in [LoopExecution::Direct, LoopExecution::Graph] {
        // Always requests a tool call, so the loop never finishes on its own
        // and must hit the model-call cap.
        let model = Arc::new(MockModel::with_tool_call("lookup", json!({})));
        let mut harness = harness_for(execution, model);
        harness.register_tool(Arc::new(FakeTool::returning("lookup", "tool-output")));

        let ctx = RunContext::new(RunConfig::new("limit-stop").with_max_model_calls(2), ());
        let err = harness
            .invoke_in_context(&(), ctx, vec![Message::user("loop forever")])
            .await
            .expect_err("run hits the model-call cap");

        match execution {
            LoopExecution::Direct => direct_err = Some(err),
            LoopExecution::Graph => graph_err = Some(err),
        }
    }

    assert!(matches!(
        direct_err.unwrap(),
        TinyAgentsError::LimitExceeded(_)
    ));
    assert!(matches!(
        graph_err.unwrap(),
        TinyAgentsError::LimitExceeded(_)
    ));
}

// ── Scenario 4: approval interrupt ──────────────────────────────────────────

mod interrupt_middleware {
    use async_trait::async_trait;
    use tinyagents_harness::context::{MiddlewareControl, RunContext};
    use tinyagents_harness::middleware::Middleware;
    use tinyinference_llm::model::ModelResponse;

    pub struct RequireApproval;

    #[async_trait]
    impl Middleware<(), ()> for RequireApproval {
        fn name(&self) -> &str {
            "require_approval"
        }

        async fn after_model(
            &self,
            ctx: &mut RunContext<()>,
            _state: &(),
            _response: &mut ModelResponse,
        ) -> tinyagents_harness::Result<()> {
            ctx.request_control(MiddlewareControl::Interrupt {
                node: "review".into(),
                message: "needs approval".into(),
            });
            Ok(())
        }
    }
}

#[tokio::test]
async fn approval_interrupt_scenario_matches_direct_and_graph() {
    for execution in [LoopExecution::Direct, LoopExecution::Graph] {
        let model = Arc::new(MockModel::constant("hi"));
        let mut harness = harness_for(execution, model);
        harness.push_middleware(Arc::new(interrupt_middleware::RequireApproval));

        let err = harness
            .invoke_default(&(), vec![Message::user("do the risky thing")])
            .await
            .expect_err("both engines surface MiddlewareControl::Interrupt as an error");

        match err {
            TinyAgentsError::Interrupted { node, message } => {
                assert_eq!(node, "review");
                assert_eq!(message, "needs approval");
            }
            other => panic!("expected Interrupted, got {other:?}"),
        }
    }
}

// ── Scenario 5: steering inject ─────────────────────────────────────────────

#[tokio::test]
async fn steering_inject_scenario_matches_direct_and_graph() {
    let mut direct_messages = None;
    let mut graph_messages = None;

    for execution in [LoopExecution::Direct, LoopExecution::Graph] {
        let model = Arc::new(MockModel::with_responses(vec![ModelResponse::assistant(
            "done",
        )]));
        let harness = harness_for(execution, model.clone());

        let steering = SteeringHandle::allow_all();
        steering.send(SteeringCommand::InjectMessage(Message::user(
            "ORCHESTRATOR: answer in French.",
        )));
        let ctx = RunContext::new(RunConfig::new("steer-inject"), ()).with_steering(steering);

        let run = harness
            .invoke_in_context(&(), ctx, vec![Message::user("hello")])
            .await
            .expect("run completes");
        assert_eq!(run.model_calls, 1);

        let injected = run
            .messages
            .iter()
            .any(|message| message.text().contains("ORCHESTRATOR"));
        assert!(injected, "the injected steering message must reach the transcript");

        match execution {
            LoopExecution::Direct => direct_messages = Some(run.messages),
            LoopExecution::Graph => graph_messages = Some(run.messages),
        }
    }

    assert_eq!(direct_messages, graph_messages);
}

// ── Scenario 6: output retry ────────────────────────────────────────────────

#[tokio::test]
async fn output_retry_scenario_matches_direct_and_graph() {
    let schema = json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
    });

    let mut direct_run = None;
    let mut graph_run = None;

    for execution in [LoopExecution::Direct, LoopExecution::Graph] {
        // First reply is not valid JSON for the schema; second reply repairs it.
        let model = Arc::new(MockModel::with_responses(vec![
            ModelResponse::assistant("not json"),
            ModelResponse::assistant(r#"{"answer":"fixed"}"#),
        ]));
        let mut harness = harness_for(execution, model);
        let mut policy = harness.policy().clone();
        policy.default_response_format = Some(ResponseFormat::auto("answer", schema.clone()));
        harness.with_policy(policy);

        let run = harness
            .invoke_default(&(), vec![Message::user("what is the answer")])
            .await
            .expect("run completes after one repair turn");

        match execution {
            LoopExecution::Direct => direct_run = Some(run),
            LoopExecution::Graph => graph_run = Some(run),
        }
    }

    let (direct_run, graph_run) = (direct_run.unwrap(), graph_run.unwrap());
    assert_eq!(direct_run.structured, graph_run.structured);
    assert_eq!(direct_run.structured, Some(json!({ "answer": "fixed" })));
    assert_eq!(direct_run.model_calls, graph_run.model_calls);
    assert_eq!(direct_run.model_calls, 2);
}

// ── Checkpoint + resume across an interrupt ─────────────────────────────────

mod pause_middleware {
    use async_trait::async_trait;
    use tinyagents_harness::context::{MiddlewareControl, RunContext};
    use tinyagents_harness::middleware::Middleware;
    use tinyinference_llm::model::ModelResponse;

    /// Requests a graph-level interrupt (an approval gate) on the first
    /// `after_model` hook only.
    pub struct PauseOnce(pub std::sync::atomic::AtomicBool);

    impl PauseOnce {
        pub fn new() -> Self {
            Self(std::sync::atomic::AtomicBool::new(true))
        }
    }

    #[async_trait]
    impl Middleware<(), ()> for PauseOnce {
        fn name(&self) -> &str {
            "pause_once"
        }

        async fn after_model(
            &self,
            ctx: &mut RunContext<()>,
            _state: &(),
            _response: &mut ModelResponse,
        ) -> tinyagents_harness::Result<()> {
            if self
                .0
                .swap(false, std::sync::atomic::Ordering::SeqCst)
            {
                ctx.request_control(MiddlewareControl::Interrupt {
                    node: "review".into(),
                    message: "needs approval".into(),
                });
            }
            Ok(())
        }
    }
}

/// Drives `compile_loop`'s real `CompiledGraph` directly (not through
/// `AgentHarness::invoke`/`GraphLoopDriver`), which is what actually gets
/// checkpoint/resume: A5 item 2's "approvals surfacing as graph interrupts".
async fn checkpoint_resume_with<C>(checkpointer: Arc<C>)
where
    C: tinyagents_graph::checkpoint::Checkpointer<
            tinyagents_graph::agent_loop::LoopState,
        > + 'static,
{
    let model = Arc::new(MockModel::with_responses(vec![
        tool_call_response("call-1", "lookup", json!({})),
        ModelResponse::assistant("done"),
    ]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model)
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("lookup", "tool-output")))
        .push_middleware(Arc::new(pause_middleware::PauseOnce::new()));
    let harness = Arc::new(harness);

    let rt = Arc::new(LoopRuntime::for_run(
        harness.clone(),
        Arc::new(()),
        RunContext::new(RunConfig::new("checkpoint-resume"), ()),
    ));
    let graph = compile_loop(rt).expect("graph compiles").with_checkpointer(checkpointer);

    let execution = graph
        .run_with_thread(
            "checkpoint-resume-thread",
            LoopState::seed(vec![Message::user("look something up")]),
        )
        .await
        .expect("first leg completes to the interrupt");
    assert_eq!(execution.interrupts.len(), 1, "run paused at the approval gate");
    assert!(!execution.state.finished);

    let resumed = graph
        .resume(
            "checkpoint-resume-thread",
            tinyagents_graph::Command {
                update: None,
                goto: Vec::new(),
                resume: Some(json!({ "approved": true })),
                resume_by_task: Default::default(),
            },
        )
        .await
        .expect("resume drives the run to completion");
    assert!(resumed.state.finished);
    assert_eq!(resumed.state.final_text.as_deref(), Some("done"));
}

#[tokio::test]
async fn checkpoint_resume_across_interrupt_in_memory() {
    checkpoint_resume_with(Arc::new(InMemoryCheckpointer::new())).await;
}

#[tokio::test]
async fn checkpoint_resume_across_interrupt_file() {
    let dir = tempdir();
    checkpoint_resume_with(Arc::new(FileCheckpointer::new(dir.path()))).await;
}

/// Minimal temp-dir helper (avoids pulling in the `tempfile` crate just for
/// this one test).
fn tempdir() -> TempDir {
    let path = std::env::temp_dir().join(format!(
        "loop_as_graph-{}-{}",
        std::process::id(),
        tinyagents_harness::ids::now_ms()
    ));
    std::fs::create_dir_all(&path).expect("create temp dir");
    TempDir(path)
}

struct TempDir(std::path::PathBuf);

impl TempDir {
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ── `iter()` stepping with `override_next` ──────────────────────────────────

#[tokio::test]
async fn iter_steps_node_by_node_and_honors_override_next() {
    let model = Arc::new(MockModel::with_responses(vec![
        tool_call_response("call-1", "lookup", json!({})),
        ModelResponse::assistant("done"),
    ]));
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness
        .register_model("mock", model)
        .set_default_model("mock")
        .register_tool(Arc::new(FakeTool::returning("lookup", "tool-output")));
    let harness = Arc::new(harness);

    let ctx = RunContext::new(RunConfig::new("iter-stepping"), ());
    let mut iter = harness
        .iter(Arc::new(()), ctx, vec![Message::user("look something up")])
        .expect("iter starts");

    let step = iter.next().await.expect("plan step").expect("not finished");
    assert_eq!(step.node, node::PLAN);
    assert_eq!(step.next.as_deref(), Some(node::MODEL));

    let step = iter.next().await.expect("model step").expect("not finished");
    assert_eq!(step.node, node::MODEL);
    assert_eq!(step.next.as_deref(), Some(node::TOOLS));

    // Redirect the very next activation back to `plan` instead of the
    // naturally-routed `tools` — exercising `override_next` — then let the
    // (now unoverridden) routing carry the run to completion.
    iter.override_next(node::PLAN);
    let step = iter.next().await.expect("overridden step").expect("not finished");
    assert_eq!(step.node, node::PLAN);
    assert_eq!(step.next.as_deref(), Some(node::MODEL));

    let state = iter.run_to_end().await.expect("run finishes");
    assert!(state.finished);
}
