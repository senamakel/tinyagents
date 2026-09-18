//! Tests for host-driven graph-to-agent delegation.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::*;
use crate::builder::GraphBuilder;
use tinyagents_harness::cancel::CancellationToken;
use tinyagents_harness::events::EventSink;

#[derive(Clone, Default)]
struct RecordingInvoker {
    requests: Arc<Mutex<Vec<AgentInvocation>>>,
}

impl RecordingInvoker {
    fn requests(&self) -> Vec<AgentInvocation> {
        self.requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl AgentInvoker for RecordingInvoker {
    async fn invoke(&self, request: AgentInvocation) -> crate::Result<SubAgentOutput> {
        self.requests.lock().unwrap().push(request.clone());
        Ok(SubAgentOutput {
            text: format!("done:{}", request.input.prompt),
            model_calls: 1,
            ..SubAgentOutput::default()
        })
    }
}

fn graph(invoker: Arc<dyn AgentInvoker>) -> crate::CompiledGraph<String, String> {
    GraphBuilder::<String, String>::overwrite()
        .add_node(
            "delegate",
            subagent_node(SubAgentNode::from_fns(
                "researcher",
                |state: &String| SubAgentInput::prompt(state.clone()),
                |output: SubAgentOutput| output.text,
            )),
        )
        .set_entry("delegate")
        .set_finish("delegate")
        .compile()
        .unwrap()
        .with_agent_invoker(invoker, EventSink::new(), CancellationToken::new())
}

#[tokio::test]
async fn delegation_uses_carried_invoker_and_preserves_graph_lineage() {
    let invoker = Arc::new(RecordingInvoker::default());
    let graph = graph(invoker.clone());

    let run = graph.run("question".to_string()).await.unwrap();

    assert_eq!(run.state, "done:question");
    let requests = invoker.requests();
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert_eq!(request.agent_id, "researcher");
    assert_eq!(request.input.prompt, "question");
    assert_eq!(request.parent_run_id, run.run_id);
    assert_eq!(request.root_run_id, run.root_run_id);
    assert_eq!(request.node_id.as_str(), "delegate");
    assert_eq!(request.graph_id, run.graph_id);
    assert!(request.cancellation.is_some());
}

#[tokio::test]
async fn concurrent_sibling_graph_runs_have_isolated_parent_identity() {
    let invoker = Arc::new(RecordingInvoker::default());
    let graph = graph(invoker.clone());

    let (left, right) = tokio::join!(
        graph.run("left".to_string()),
        graph.run("right".to_string())
    );
    let left = left.unwrap();
    let right = right.unwrap();
    let requests = invoker.requests();
    assert_eq!(requests.len(), 2);
    assert_ne!(left.run_id, right.run_id);
    for request in requests {
        if request.input.prompt == "left" {
            assert_eq!(request.parent_run_id, left.run_id);
            assert_eq!(request.root_run_id, left.root_run_id);
        } else {
            assert_eq!(request.input.prompt, "right");
            assert_eq!(request.parent_run_id, right.run_id);
            assert_eq!(request.root_run_id, right.root_run_id);
        }
    }
}

#[tokio::test]
async fn missing_host_invoker_is_an_explicit_capability_error() {
    let graph = GraphBuilder::<String, String>::overwrite()
        .add_node(
            "delegate",
            subagent_node(SubAgentNode::from_fns(
                "researcher",
                |state: &String| SubAgentInput::prompt(state.clone()),
                |output: SubAgentOutput| output.text,
            )),
        )
        .set_entry("delegate")
        .set_finish("delegate")
        .compile()
        .unwrap();

    let error = graph.run("question".to_string()).await.unwrap_err();
    assert!(matches!(error, crate::TinyAgentsError::Capability(_)));
}
