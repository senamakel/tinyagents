use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use chrono::Utc;
use serde_json::json;
use tinyagents_graph::dag::{DagNode, has_cycle};
use tinyagents_session::run_ledger::{
    self, AgentTeam, AgentTeamListRequest, AgentTeamListResponse, AgentTeamMember,
    AgentTeamMemberStatus, AgentTeamMemberUpsert, AgentTeamStatus, AgentTeamTask,
    AgentTeamTaskStatus, AgentTeamTaskUpsert, AgentTeamUpsert, ClaimOutcome, CompletionOutcome,
    RunEvent, RunEventAppend, RunEventListRequest,
};
use uuid::Uuid;

use super::{LEAD_SENDER, MemberShutdown, NewMember, TEAM_MESSAGE_EVENT, TeamError, TeamView};

/// Durable team state required by [`TeamService`].
///
/// The trait intentionally mirrors the session run-ledger operations rather
/// than introducing another task store. Hosts can substitute a fake ledger in
/// tests or select their own durable implementation.
pub trait TeamLedger: Send + Sync {
    /// Upserts a team row: creates if absent, updates if present.
    fn upsert_team(&self, upsert: AgentTeamUpsert) -> Result<AgentTeam>;
    /// Retrieves a team by id.
    fn get_team(&self, id: &str) -> Result<Option<AgentTeam>>;
    /// Lists teams according to the request filters.
    fn list_teams(&self, request: &AgentTeamListRequest) -> Result<AgentTeamListResponse>;
    /// Upserts a team member: creates if absent, updates if present.
    fn upsert_member(&self, upsert: AgentTeamMemberUpsert) -> Result<AgentTeamMember>;
    /// Lists members of a team in creation order.
    fn list_members(&self, team_id: &str) -> Result<Vec<AgentTeamMember>>;
    /// Lists tasks assigned to a team in creation order.
    fn list_tasks(&self, team_id: &str) -> Result<Vec<AgentTeamTask>>;
    /// Upserts a task: creates if absent, updates if present.
    fn upsert_task(&self, upsert: AgentTeamTaskUpsert) -> Result<AgentTeamTask>;
    /// Attempts to claim a task for a member using an atomic CAS operation.
    /// Returns the claim outcome (success, already claimed, not found).
    fn claim_task(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        claim_token: &str,
    ) -> Result<ClaimOutcome>;
    /// Records a task completion with optional evidence, atomically advancing
    /// the task status and optionally validating evidence before transition.
    fn complete_task(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        evidence: &[String],
        require_evidence: bool,
    ) -> Result<CompletionOutcome>;
    /// Stops a member and releases its claimed tasks, returning the stopped
    /// member and the released task ids.
    fn shutdown_member(
        &self,
        team_id: &str,
        member_id: &str,
    ) -> Result<Option<(AgentTeamMember, Vec<String>)>>;
    /// Appends an event to the durable run event log.
    fn append_event(&self, event: RunEventAppend) -> Result<RunEvent>;
    /// Lists events from the run event log according to the request filters.
    fn list_events(&self, request: &RunEventListRequest) -> Result<Vec<RunEvent>>;
}

/// [`TeamLedger`] backed by `tinyagents-session`'s run ledger at a caller
/// supplied workspace root. It makes no workspace or host policy decision.
#[derive(Debug, Clone)]
pub struct SessionTeamLedger {
    workspace_dir: PathBuf,
}

impl SessionTeamLedger {
    pub fn new(workspace_dir: impl Into<PathBuf>) -> Self {
        Self {
            workspace_dir: workspace_dir.into(),
        }
    }

    pub fn workspace_dir(&self) -> &Path {
        &self.workspace_dir
    }
}

impl TeamLedger for SessionTeamLedger {
    fn upsert_team(&self, upsert: AgentTeamUpsert) -> Result<AgentTeam> {
        Ok(run_ledger::upsert_agent_team(&self.workspace_dir, upsert)?)
    }
    fn get_team(&self, id: &str) -> Result<Option<AgentTeam>> {
        Ok(run_ledger::get_agent_team(&self.workspace_dir, id)?)
    }
    fn list_teams(&self, request: &AgentTeamListRequest) -> Result<AgentTeamListResponse> {
        Ok(run_ledger::list_agent_teams(&self.workspace_dir, request)?)
    }
    fn upsert_member(&self, upsert: AgentTeamMemberUpsert) -> Result<AgentTeamMember> {
        Ok(run_ledger::upsert_agent_team_member(
            &self.workspace_dir,
            upsert,
        )?)
    }
    fn list_members(&self, team_id: &str) -> Result<Vec<AgentTeamMember>> {
        Ok(run_ledger::list_agent_team_members(
            &self.workspace_dir,
            team_id,
        )?)
    }
    fn list_tasks(&self, team_id: &str) -> Result<Vec<AgentTeamTask>> {
        Ok(run_ledger::list_agent_team_tasks(
            &self.workspace_dir,
            team_id,
        )?)
    }
    fn upsert_task(&self, upsert: AgentTeamTaskUpsert) -> Result<AgentTeamTask> {
        Ok(run_ledger::upsert_agent_team_task(
            &self.workspace_dir,
            upsert,
        )?)
    }
    fn claim_task(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        claim_token: &str,
    ) -> Result<ClaimOutcome> {
        Ok(run_ledger::claim_agent_team_task(
            &self.workspace_dir,
            team_id,
            task_id,
            member_id,
            claim_token,
        )?)
    }
    fn complete_task(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        evidence: &[String],
        require_evidence: bool,
    ) -> Result<CompletionOutcome> {
        Ok(run_ledger::complete_agent_team_task(
            &self.workspace_dir,
            team_id,
            task_id,
            member_id,
            evidence,
            require_evidence,
        )?)
    }
    fn shutdown_member(
        &self,
        team_id: &str,
        member_id: &str,
    ) -> Result<Option<(AgentTeamMember, Vec<String>)>> {
        Ok(run_ledger::shutdown_agent_team_member(
            &self.workspace_dir,
            team_id,
            member_id,
        )?)
    }
    fn append_event(&self, event: RunEventAppend) -> Result<RunEvent> {
        Ok(run_ledger::append_run_event(&self.workspace_dir, event)?)
    }
    fn list_events(&self, request: &RunEventListRequest) -> Result<Vec<RunEvent>> {
        Ok(run_ledger::list_recent_run_events(&self.workspace_dir, request)?.events)
    }
}

/// Host-neutral service for durable, dependency-aware agent teams.
///
/// Provides a high-level API for team management: creating teams, adding
/// members, creating and claiming tasks, composing prompts, and shutting down
/// members. All mutations are durably persisted via a caller-supplied
/// [`TeamLedger`]; the service enforces coordination invariants (no duplicate
/// member names, valid task dependencies, no cycles).
///
/// Generic over the ledger to allow hosts to inject their own storage
/// implementation or a test double.
#[derive(Debug, Clone)]
pub struct TeamService<L> {
    ledger: L,
}

impl<L> TeamService<L> {
    /// Creates a service wrapping the provided [`TeamLedger`].
    pub fn new(ledger: L) -> Self {
        Self { ledger }
    }

    /// Returns a reference to the underlying ledger.
    pub fn ledger(&self) -> &L {
        &self.ledger
    }
}

impl<L: TeamLedger> TeamService<L> {
    pub fn create_team(
        &self,
        lead_agent_id: &str,
        parent_thread_id: Option<&str>,
        summary: Option<&str>,
        members: &[NewMember],
    ) -> Result<TeamView> {
        let mut seen = HashSet::new();
        for member in members {
            if !seen.insert(member.name.as_str()) {
                return Err(anyhow!(TeamError::DuplicateMemberName {
                    name: member.name.clone()
                }));
            }
        }
        let team_id = format!("team-{}", Uuid::new_v4().simple());
        self.ledger.upsert_team(AgentTeamUpsert {
            id: team_id.clone(),
            parent_thread_id: parent_thread_id.map(str::to_string),
            lead_agent_id: lead_agent_id.to_string(),
            status: AgentTeamStatus::Active,
            summary: summary.map(str::to_string),
            created_at: None,
            closed_at: None,
        })?;
        for member in members {
            self.ledger.upsert_member(AgentTeamMemberUpsert {
                id: format!("member-{}", Uuid::new_v4().simple()),
                team_id: team_id.clone(),
                name: member.name.clone(),
                agent_id: member.agent_id.clone(),
                member_status: AgentTeamMemberStatus::Pending,
                current_task_id: None,
                worker_thread_id: None,
                run_id: None,
                created_at: None,
            })?;
        }
        self.team_view(&team_id)
    }

    pub fn list_teams(&self, request: &AgentTeamListRequest) -> Result<AgentTeamListResponse> {
        self.ledger.list_teams(request)
    }

    pub fn get_team(&self, team_id: &str) -> Result<Option<TeamView>> {
        if self.ledger.get_team(team_id)?.is_some() {
            self.team_view(team_id).map(Some)
        } else {
            Ok(None)
        }
    }

    pub fn assign_task(
        &self,
        team_id: &str,
        title: &str,
        objective: Option<&str>,
        owner_member_id: Option<&str>,
        depends_on: &[String],
    ) -> Result<AgentTeamTask> {
        self.ledger
            .get_team(team_id)?
            .ok_or_else(|| anyhow!("unknown team: {team_id}"))?;
        let existing = self.ledger.list_tasks(team_id)?;
        if let Some(owner) = owner_member_id
            && !self
                .ledger
                .list_members(team_id)?
                .iter()
                .any(|member| member.id == owner)
        {
            return Err(anyhow!(TeamError::UnknownMember {
                member_id: owner.to_string()
            }));
        }
        let task_id = format!("task-{}", Uuid::new_v4().simple());
        validate_dependencies(&task_id, depends_on, &existing)?;
        self.ledger.upsert_task(AgentTeamTaskUpsert {
            id: task_id,
            team_id: team_id.to_string(),
            title: title.to_string(),
            objective: objective.map(str::to_string),
            status: AgentTeamTaskStatus::Todo,
            owner_member_id: owner_member_id.map(str::to_string),
            depends_on: depends_on.to_vec(),
            gate_status: None,
            gate_reason: None,
            evidence: vec![],
            source_run_id: None,
            order_index: existing.len() as i64,
            created_at: None,
        })
    }

    pub fn claim_task(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        claim_token: &str,
    ) -> Result<ClaimOutcome> {
        self.ensure_member(team_id, member_id)?;
        self.ledger
            .claim_task(team_id, task_id, member_id, claim_token)
    }

    pub fn message_member(
        &self,
        team_id: &str,
        from_member_id: Option<&str>,
        to_member_id: Option<&str>,
        content: &str,
        visibility: Option<&str>,
    ) -> Result<RunEvent> {
        self.ledger
            .get_team(team_id)?
            .ok_or_else(|| anyhow!("unknown team: {team_id}"))?;
        if let Some(from) = from_member_id {
            self.ensure_member(team_id, from)?;
        }
        if let Some(to) = to_member_id {
            self.ensure_member(team_id, to)?;
        }
        self.ledger.append_event(RunEventAppend {
            run_id: team_id.to_string(),
            event_type: TEAM_MESSAGE_EVENT.to_string(),
            payload: json!({"from": from_member_id.unwrap_or(LEAD_SENDER), "to": to_member_id,
                "content": content, "visibility": visibility.unwrap_or("team")}),
        })
    }

    pub fn list_messages(&self, team_id: &str, limit: Option<u32>) -> Result<Vec<RunEvent>> {
        Ok(self
            .ledger
            .list_events(&RunEventListRequest {
                run_id: team_id.to_string(),
                after_sequence: None,
                limit,
            })?
            .into_iter()
            .filter(|event| event.event_type == TEAM_MESSAGE_EVENT)
            .collect())
    }

    pub fn complete_task(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        evidence: &[String],
        require_evidence: bool,
    ) -> Result<CompletionOutcome> {
        self.ensure_member(team_id, member_id)?;
        self.ledger
            .complete_task(team_id, task_id, member_id, evidence, require_evidence)
    }

    pub fn shutdown_member(&self, team_id: &str, member_id: &str) -> Result<MemberShutdown> {
        self.ledger
            .shutdown_member(team_id, member_id)?
            .map(|(member, released_task_ids)| MemberShutdown {
                member,
                released_task_ids,
            })
            .ok_or_else(|| {
                anyhow!(TeamError::UnknownMember {
                    member_id: member_id.to_string()
                })
            })
    }

    pub fn close_team(&self, team_id: &str, summary: Option<&str>) -> Result<AgentTeam> {
        let existing = self
            .ledger
            .get_team(team_id)?
            .ok_or_else(|| anyhow!("unknown team: {team_id}"))?;
        self.ledger.upsert_team(AgentTeamUpsert {
            id: team_id.to_string(),
            parent_thread_id: existing.parent_thread_id,
            lead_agent_id: existing.lead_agent_id,
            status: AgentTeamStatus::Closed,
            summary: summary.map(str::to_string),
            created_at: Some(existing.created_at),
            closed_at: Some(Utc::now()),
        })
    }

    fn team_view(&self, team_id: &str) -> Result<TeamView> {
        let team = self
            .ledger
            .get_team(team_id)?
            .ok_or_else(|| anyhow!("team missing after creation: {team_id}"))?;
        Ok(TeamView {
            team,
            members: self.ledger.list_members(team_id)?,
            tasks: self.ledger.list_tasks(team_id)?,
        })
    }

    fn ensure_member(&self, team_id: &str, member_id: &str) -> Result<()> {
        if self
            .ledger
            .list_members(team_id)?
            .iter()
            .any(|member| member.id == member_id)
        {
            Ok(())
        } else {
            Err(anyhow!(TeamError::UnknownMember {
                member_id: member_id.to_string()
            }))
        }
    }
}

fn validate_dependencies(
    new_task_id: &str,
    depends_on: &[String],
    existing: &[AgentTeamTask],
) -> Result<()> {
    let known: HashSet<&str> = existing.iter().map(|task| task.id.as_str()).collect();
    for dependency in depends_on {
        if dependency == new_task_id {
            return Err(anyhow!(TeamError::SelfDependency {
                task_id: new_task_id.to_string()
            }));
        }
        if !known.contains(dependency.as_str()) {
            return Err(anyhow!(TeamError::UnknownDependency {
                depends_on: dependency.clone()
            }));
        }
    }
    if has_task_cycle(new_task_id, depends_on, existing) {
        return Err(anyhow!(TeamError::CyclicDependency));
    }
    Ok(())
}

fn has_task_cycle(new_task_id: &str, depends_on: &[String], existing: &[AgentTeamTask]) -> bool {
    let mut nodes: Vec<DagNode<'_>> = existing
        .iter()
        .map(|task| DagNode::new(task.id.as_str(), task.depends_on.iter().map(String::as_str)))
        .collect();
    nodes.push(DagNode::new(
        new_task_id,
        depends_on.iter().map(String::as_str),
    ));
    has_cycle(&nodes)
}

/// Select the next task a member may claim without making any policy decision.
pub fn claimable_task<'a>(
    tasks: &'a [AgentTeamTask],
    member_id: &str,
) -> Option<&'a AgentTeamTask> {
    let done: HashSet<&str> = tasks
        .iter()
        .filter(|task| task.status == AgentTeamTaskStatus::Done)
        .map(|task| task.id.as_str())
        .collect();
    tasks.iter().find(|task| {
        matches!(
            task.status,
            AgentTeamTaskStatus::Todo | AgentTeamTaskStatus::Ready
        ) && task.claimed_by_member_id.is_none()
            && task
                .owner_member_id
                .as_deref()
                .map(|owner| owner == member_id)
                .unwrap_or(true)
            && task
                .depends_on
                .iter()
                .all(|dependency| done.contains(dependency.as_str()))
    })
}

#[cfg(test)]
mod dependency_tests {
    use chrono::Utc;
    use tinyagents_session::run_ledger::AgentTeamTaskStatus;

    use super::*;

    fn task(id: &str, depends_on: &[&str]) -> AgentTeamTask {
        let now = Utc::now();
        AgentTeamTask {
            id: id.to_string(),
            team_id: "team".to_string(),
            title: id.to_string(),
            objective: None,
            status: AgentTeamTaskStatus::Todo,
            owner_member_id: None,
            claimed_by_member_id: None,
            claim_token: None,
            depends_on: depends_on
                .iter()
                .map(|dependency| (*dependency).to_string())
                .collect(),
            gate_status: "pending".to_string(),
            gate_reason: None,
            evidence: Vec::new(),
            source_run_id: None,
            order_index: 0,
            created_at: now,
            updated_at: now,
        }
    }

    #[test]
    fn rejects_self_dependency() {
        let error =
            validate_dependencies("task-self", &["task-self".to_string()], &[]).unwrap_err();
        assert_eq!(
            error.downcast::<TeamError>().unwrap(),
            TeamError::SelfDependency {
                task_id: "task-self".to_string()
            }
        );
    }

    #[test]
    fn rejects_dependency_cycle() {
        let existing = vec![task("task-a", &["task-new"])];
        let error =
            validate_dependencies("task-new", &["task-a".to_string()], &existing).unwrap_err();
        assert_eq!(
            error.downcast::<TeamError>().unwrap(),
            TeamError::CyclicDependency
        );
    }
}
