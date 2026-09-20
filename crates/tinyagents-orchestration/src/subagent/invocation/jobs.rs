//! Asynchronous subagent job registry and host-facing control tools.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use tinyagents_harness::context::RunContext;
use tinyagents_harness::error::TinyAgentsError;
use tinyagents_harness::ids::next_seq;
use tinyagents_harness::steering::{
    SteeringCommand, SteeringCommandKind, SteeringHandle, SteeringPolicy,
};
use tinyagents_harness::tool::{ToolDispatch, ToolRegistry};
use tinyinference_llm::message::Message;
use tinytools::{Tool, ToolResult};

use super::{
    SubAgentJob, SubAgentJobEntry, SubAgentJobError, SubAgentJobId, SubAgentJobRegistry,
    SubAgentJobStatus,
};

impl SubAgentJobRegistry {
    /// Creates an empty asynchronous job registry.
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn create(&self, agent: &str, owner: u64) -> (SubAgentJobId, SteeringHandle) {
        let id = SubAgentJobId(format!("subagent-job-{}", next_seq()));
        let steering =
            SteeringHandle::new(SteeringPolicy::new().allow(SteeringCommandKind::InjectMessage));
        let entry = SubAgentJobEntry {
            job: SubAgentJob {
                id: id.clone(),
                agent: agent.to_owned(),
                status: SubAgentJobStatus::Queued,
                output: None,
                error: None,
            },
            owner,
            steering: steering.clone(),
        };
        self.write().insert(id.clone(), entry);
        (id, steering)
    }

    pub(crate) fn mark_running(&self, id: &SubAgentJobId) {
        if let Some(entry) = self.write().get_mut(id) {
            entry.job.status = SubAgentJobStatus::Running;
        }
    }

    pub(crate) fn mark_result(
        &self,
        id: &SubAgentJobId,
        result: Result<tinyagents_harness::middleware::AgentRun, TinyAgentsError>,
    ) {
        let mut entries = self.write();
        let Some(entry) = entries.get_mut(id) else {
            return;
        };
        match result {
            Ok(run) => {
                entry.job.status = SubAgentJobStatus::Completed;
                entry.job.output = run.text();
            }
            Err(TinyAgentsError::Cancelled) => {
                entry.job.status = SubAgentJobStatus::Cancelled;
                entry.job.error = Some(TinyAgentsError::Cancelled.to_string());
            }
            Err(error) => {
                entry.job.status = SubAgentJobStatus::Failed;
                entry.job.error = Some(error.to_string());
            }
        }
    }

    /// Returns a snapshot for `job_id` when it belongs to `owner`.
    pub(crate) fn get_owned(&self, job_id: &str, owner: u64) -> Option<SubAgentJob> {
        self.read()
            .get(&SubAgentJobId(job_id.to_owned()))
            .filter(|entry| entry.owner == owner)
            .map(|entry| entry.job.clone())
    }

    /// Returns a job snapshot for trusted host-side supervision.
    ///
    /// Model-visible tools must use the run-scoped dispatch path instead.
    pub fn get(&self, job_id: &str) -> Option<SubAgentJob> {
        self.read()
            .get(&SubAgentJobId(job_id.to_owned()))
            .map(|entry| entry.job.clone())
    }

    /// Returns this run's jobs in stable id order.
    fn list_owned(&self, owner: u64) -> Vec<SubAgentJob> {
        let mut jobs = self
            .read()
            .values()
            .filter(|entry| entry.owner == owner)
            .map(|entry| entry.job.clone())
            .collect::<Vec<_>>();
        jobs.sort_by(|left, right| left.id.0.cmp(&right.id.0));
        jobs
    }

    /// Returns every job for trusted host-side supervision.
    ///
    /// Model-visible tools must use the run-scoped dispatch path instead.
    pub fn list(&self) -> Vec<SubAgentJob> {
        let mut jobs = self
            .read()
            .values()
            .map(|entry| entry.job.clone())
            .collect::<Vec<_>>();
        jobs.sort_by(|left, right| left.id.0.cmp(&right.id.0));
        jobs
    }

    /// Queues a user message for delivery at the running child's next safe
    /// steering checkpoint.
    pub(crate) fn send_message_owned(
        &self,
        job_id: &str,
        owner: u64,
        message: impl Into<String>,
    ) -> Result<(), SubAgentJobError> {
        let id = SubAgentJobId(job_id.to_owned());
        let entries = self.read();
        let entry = entries
            .get(&id)
            .filter(|entry| entry.owner == owner)
            .ok_or_else(|| SubAgentJobError::NotFound(job_id.to_owned()))?;
        if entry.job.status.is_terminal() {
            return Err(SubAgentJobError::Terminal {
                job_id: job_id.to_owned(),
                status: entry.job.status,
            });
        }
        entry
            .steering
            .send(SteeringCommand::InjectMessage(Message::user(
                message.into(),
            )));
        Ok(())
    }

    fn read(
        &self,
    ) -> std::sync::RwLockReadGuard<'_, std::collections::HashMap<SubAgentJobId, SubAgentJobEntry>>
    {
        self.inner
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn write(
        &self,
    ) -> std::sync::RwLockWriteGuard<'_, std::collections::HashMap<SubAgentJobId, SubAgentJobEntry>>
    {
        self.inner
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Typed host tool that queries a job or lists jobs owned by the requesting run.
#[derive(Clone)]
pub struct SubAgentJobsTool {
    jobs: SubAgentJobRegistry,
}

impl SubAgentJobsTool {
    /// Creates the query tool over `jobs`.
    pub fn new(jobs: SubAgentJobRegistry) -> Self {
        Self { jobs }
    }
}

#[async_trait]
impl Tool for SubAgentJobsTool {
    fn name(&self) -> &str {
        "subagent_jobs"
    }

    fn description(&self) -> &str {
        "Query an asynchronous subagent job by id, or list all subagent jobs."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "job_id": { "type": "string" }
            }
        })
    }

    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        let _ = args;
        anyhow::bail!("subagent_jobs requires typed-parent dispatch")
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ToolDispatch<State, Ctx> for SubAgentJobsTool {
    fn tool(&self) -> Arc<dyn Tool> {
        Arc::new(self.clone())
    }

    async fn execute(
        &self,
        _state: &State,
        args: Value,
        _options: tinytools::ToolCallOptions,
        parent: &RunContext<Ctx>,
    ) -> anyhow::Result<ToolResult> {
        let object = args
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("arguments must be an object"))?;
        if let Some(value) = object.get("job_id") {
            let job_id = value
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("job_id must be a string when provided"))?;
            let job = self
                .jobs
                .get_owned(job_id, parent.instance_id())
                .ok_or_else(|| SubAgentJobError::NotFound(job_id.to_owned()))?;
            Ok(ToolResult::json(serde_json::to_value(job)?))
        } else {
            Ok(ToolResult::json(serde_json::to_value(
                self.jobs.list_owned(parent.instance_id()),
            )?))
        }
    }
}

/// Host tool that sends a message to a queued or running subagent job.
#[derive(Clone)]
pub struct SubAgentMessageTool {
    jobs: SubAgentJobRegistry,
}

impl SubAgentMessageTool {
    /// Creates the message tool over `jobs`.
    pub fn new(jobs: SubAgentJobRegistry) -> Self {
        Self { jobs }
    }
}

#[async_trait]
impl Tool for SubAgentMessageTool {
    fn name(&self) -> &str {
        "subagent_message"
    }

    fn description(&self) -> &str {
        "Send a message to a queued or running asynchronous subagent job."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "job_id": { "type": "string" },
                "message": { "type": "string" }
            },
            "required": ["job_id", "message"]
        })
    }

    async fn execute(&self, args: Value) -> anyhow::Result<ToolResult> {
        let _ = args;
        anyhow::bail!("subagent_message requires typed-parent dispatch")
    }
}

#[async_trait]
impl<State: Send + Sync, Ctx: Send + Sync> ToolDispatch<State, Ctx> for SubAgentMessageTool {
    fn tool(&self) -> Arc<dyn Tool> {
        Arc::new(self.clone())
    }

    async fn execute(
        &self,
        _state: &State,
        args: Value,
        _options: tinytools::ToolCallOptions,
        parent: &RunContext<Ctx>,
    ) -> anyhow::Result<ToolResult> {
        let job_id = args
            .get("job_id")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("job_id must be a string"))?;
        let message = args
            .get("message")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("message must be a string"))?;
        self.jobs
            .send_message_owned(job_id, parent.instance_id(), message)?;
        Ok(ToolResult::json(json!({
            "job_id": job_id,
            "status": "message_queued"
        })))
    }
}

/// Registers the standard run-scoped query and message tools in a harness registry.
pub fn register_subagent_job_tools<State: Send + Sync, Ctx: Send + Sync>(
    registry: &mut ToolRegistry<State, Ctx>,
    jobs: SubAgentJobRegistry,
) -> &mut ToolRegistry<State, Ctx> {
    registry
        .register_dispatch(Arc::new(SubAgentJobsTool::new(jobs.clone())))
        .register_dispatch(Arc::new(SubAgentMessageTool::new(jobs)))
}
