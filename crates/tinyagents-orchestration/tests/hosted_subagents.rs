//! Hosted subagent authorization and capability-bundle propagation.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use tinyagents_definition::{AgentDefinition, InMemoryDefinitionRegistry};
use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::host::{
    AllowAllSecurityGate, HostCapabilities, ModelResolveRequest, ModelResolver,
    StaticContextComposer,
};
use tinyagents_harness::runtime::{
    AgentHarness, AgentInvocation, AgentTurnRequest, InvocationRuntime,
};
use tinyagents_harness::testkit::ScriptedModel;
use tinyagents_orchestration::subagent::{
    ChildDataPolicy, SubAgent, SubAgentJobRegistry, SubAgentJobStatus, SubAgentTool,
};
use tinyinference_llm::model::{ChatModel, ModelResponse};

fn delegation_call() -> ModelResponse {
    let mut response = ModelResponse::assistant("");
    response
        .message
        .tool_calls
        .push(tinyinference_llm::tool::ToolCall::new(
            "delegate",
            "worker",
            json!({"input": "child task"}),
        ));
    response
}

struct AgentResolver {
    parent: Arc<dyn ChatModel<()>>,
    worker: Arc<dyn ChatModel<()>>,
}

#[async_trait]
impl ModelResolver<()> for AgentResolver {
    async fn resolve(
        &self,
        request: &ModelResolveRequest,
    ) -> tinyagents_harness::error::Result<Arc<dyn ChatModel<()>>> {
        Ok(if request.agent_id == "worker" {
            self.worker.clone()
        } else {
            self.parent.clone()
        })
    }
}

fn host(
    parent: AgentDefinition,
    parent_model: Arc<dyn ChatModel<()>>,
    worker_model: Arc<dyn ChatModel<()>>,
) -> HostCapabilities<()> {
    HostCapabilities::new(
        Arc::new(StaticContextComposer::empty()),
        Arc::new(InMemoryDefinitionRegistry::new(vec![
            parent,
            AgentDefinition::new("worker", "Worker", "child"),
        ])),
        Arc::new(AllowAllSecurityGate),
        Arc::new(AgentResolver {
            parent: parent_model,
            worker: worker_model,
        }),
    )
}

fn runtime_with_worker(
    child_harness: AgentHarness<()>,
) -> (InvocationRuntime<(), ()>, SubAgentJobRegistry) {
    let child = Arc::new(SubAgent::new("worker", "child", Arc::new(child_harness)));
    let jobs = SubAgentJobRegistry::new();
    let mut overlay = AgentHarness::new();
    overlay.register_tool_dispatch(Arc::new(
        SubAgentTool::new(child, ChildDataPolicy::new(|_: &()| ())).with_job_registry(jobs.clone()),
    ));
    (InvocationRuntime::new(overlay), jobs)
}

#[tokio::test]
async fn authorized_child_reuses_the_parent_host_bundle() {
    let parent_model = Arc::new(ScriptedModel::new(vec![
        delegation_call(),
        ModelResponse::assistant("parent answer"),
    ]));
    let worker_model = Arc::new(ScriptedModel::replies(vec!["child answer"]));
    let parent = AgentDefinition::new("parent", "Parent", "delegates").with_subagents(["worker"]);
    let entry = AgentHarness::new();
    let (runtime, jobs) = runtime_with_worker(AgentHarness::new());
    let run = entry
        .invoke_agent(
            AgentInvocation::new(
                host(parent, parent_model, worker_model),
                AgentTurnRequest::new(
                    "parent",
                    vec![tinyinference_llm::message::Message::user("delegate")],
                ),
                RunContext::new(RunConfig::new("authorized-child"), ()),
            )
            .with_runtime(runtime),
            &(),
        )
        .await
        .expect("authorized child runs through the parent's host bundle");

    assert_eq!(run.text().as_deref(), Some("parent answer"));
    let job = tokio::time::timeout(std::time::Duration::from_secs(1), async {
        loop {
            if let Some(job) = jobs.list().into_iter().next()
                && job.status.is_terminal()
            {
                return job;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("hosted child job reaches a terminal state");
    assert_eq!(job.status, SubAgentJobStatus::Completed);
    assert_eq!(job.output.as_deref(), Some("child answer"));
}

#[tokio::test]
async fn parent_denial_cannot_fall_back_to_the_child_harness() {
    let parent_model = Arc::new(ScriptedModel::new(vec![delegation_call()]));
    let local_child_model = Arc::new(ScriptedModel::replies(vec!["local bypass"]));
    let mut child_harness = AgentHarness::new();
    child_harness.register_model("local", local_child_model.clone());

    let (runtime, _jobs) = runtime_with_worker(child_harness);
    let error = AgentHarness::new()
        .invoke_agent(
            AgentInvocation::new(
                host(
                    AgentDefinition::new("parent", "Parent", "does not delegate"),
                    parent_model,
                    Arc::new(ScriptedModel::replies(vec!["host worker"])),
                ),
                AgentTurnRequest::new(
                    "parent",
                    vec![tinyinference_llm::message::Message::user("delegate")],
                ),
                RunContext::new(RunConfig::new("denied-child"), ()),
            )
            .with_runtime(runtime),
            &(),
        )
        .await
        .expect_err("the parent's delegate allowlist denies the child");

    assert_eq!(
        error.to_string(),
        "model error: hosted agent invocation failed"
    );
    assert!(
        local_child_model.requests().is_empty(),
        "the child cannot substitute its own model authority"
    );
}
