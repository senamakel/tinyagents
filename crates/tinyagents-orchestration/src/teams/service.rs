//! Team service and durable ledger: the public API for team management.
//!
//! [`TeamService`] validates team structure, manages member and task
//! persistence (via [`TeamLedger`]), and enforces coordination invariants.
//! [`SessionTeamLedger`] provides a built-in `tinyagents-session` backend;
//! hosts can supply their own ledger implementation for testing or custom
//! storage.

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use chrono::Utc;
use serde_json::json;
use tinyagents_graph::dag::{DagNode, has_cycle};
use tinyagents_session::run_ledger::{
    AgentTeam, AgentTeamListRequest, AgentTeamListResponse, AgentTeamMember,
    AgentTeamMemberRunAdmission, AgentTeamMemberStatus, AgentTeamMemberUpsert, AgentTeamStatus,
    AgentTeamTask, AgentTeamTaskStatus, AgentTeamTaskUpsert, AgentTeamUpsert, ClaimOutcome,
    CompletionOutcome, RunEvent, RunEventAppend, RunEventListRequest, RunLedgerStore,
};
use tinyagents_storage_kit::Blocking;
use uuid::Uuid;

use super::runtime::drain_run_events;
use super::{LEAD_SENDER, MemberShutdown, NewMember, TEAM_MESSAGE_EVENT, TeamError, TeamView};

/// Host-provided execution for a claimed team member task.
#[async_trait]
pub trait TeamMemberExecutor: Send + Sync {
    /// Run `task` as `member` and return the evidence to submit at completion.
    async fn execute(&self, member: &AgentTeamMember, task: &AgentTeamTask) -> Result<Vec<String>>;
}

/// Typed claim failures from [`TeamService::start_member_with_started`].
#[derive(Debug)]
pub enum TeamMemberStartError {
    /// The task was already claimed.
    AlreadyClaimed { task_id: String },
    /// One or more task dependencies are incomplete.
    Blocked { task_id: String, unmet: Vec<String> },
    /// The task does not exist in the requested team.
    UnknownTask { task_id: String },
    /// The member already owns another active task.
    AlreadyActive {
        /// Member whose active run prevented admission.
        member_id: String,
        /// Task currently assigned to that member, if recorded.
        current_task_id: Option<String>,
    },
    /// Team/member validation or a ledger operation failed.
    Other(anyhow::Error),
}

impl std::fmt::Display for TeamMemberStartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyClaimed { task_id } => write!(f, "task is already claimed: {task_id}"),
            Self::Blocked { unmet, .. } => {
                write!(f, "task dependencies are incomplete: {}", unmet.join(", "))
            }
            Self::UnknownTask { task_id } => write!(f, "unknown task: {task_id}"),
            Self::AlreadyActive { member_id, .. } => {
                write!(f, "team member is already active: {member_id}")
            }
            Self::Other(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for TeamMemberStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Other(error) => Some(error.as_ref()),
            _ => None,
        }
    }
}

/// Durable identity delivered after a team member claim becomes active.
#[derive(Clone, Debug)]
pub struct TeamMemberStarted {
    /// Run id recorded on the active member row.
    pub run_id: String,
    /// Active member row, including its worker thread and run ids.
    pub member: AgentTeamMember,
    /// Task returned by the successful durable claim.
    pub task: AgentTeamTask,
}

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
    /// Completes a task only while `member_id` still owns `claim_token`.
    fn complete_task_if_claimed(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        claim_token: &str,
        evidence: &[String],
        require_evidence: bool,
    ) -> Result<CompletionOutcome>;
    /// Releases a claimed task back to the queue after a worker failure.
    /// Implementations must only release the claim held by `member_id` with
    /// `claim_token`, so cleanup from an old worker cannot release a successor.
    fn release_task(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        claim_token: &str,
    ) -> Result<bool>;
    /// Legacy unconditional member-running setter. It does not reserve against
    /// another active run; new task admission must use
    /// [`TeamLedger::try_mark_member_running`].
    fn mark_member_running(
        &self,
        team_id: &str,
        member_id: &str,
        task_id: &str,
        worker_thread_id: &str,
        run_id: &str,
    ) -> Result<Option<AgentTeamMember>>;
    /// Atomically reserves the member for a task. The default fails closed so
    /// legacy ledgers cannot silently permit overlapping member runs.
    fn try_mark_member_running(
        &self,
        team_id: &str,
        member_id: &str,
        task_id: &str,
        worker_thread_id: &str,
        run_id: &str,
    ) -> Result<AgentTeamMemberRunAdmission> {
        let _ = (team_id, member_id, task_id, worker_thread_id, run_id);
        Err(anyhow!(
            "atomic team member admission is unsupported by this ledger"
        ))
    }
    /// Marks a running member idle after its worker finishes.
    fn mark_member_idle(
        &self,
        team_id: &str,
        member_id: &str,
        task_id: &str,
        worker_thread_id: &str,
        run_id: &str,
    ) -> Result<bool>;
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

/// [`TeamLedger`] over a bound `tinyagents-session` run-ledger handle.
///
/// The handle is whatever backend the host opened (SQLite, MongoDB, memory);
/// this adapter opens no file, picks no workspace and has no fallback. Every
/// call reaches the ledger's async operations through a [`Blocking`] bridge
/// thread, so it works from inside or outside a tokio runtime. The claim,
/// completion and shutdown operations are the ledger's atomic ones: a contested
/// task goes to exactly one member.
#[derive(Clone)]
pub struct SessionTeamLedger {
    ledger: Arc<dyn RunLedgerStore>,
    bridge: Blocking,
}

impl std::fmt::Debug for SessionTeamLedger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionTeamLedger").finish_non_exhaustive()
    }
}

impl SessionTeamLedger {
    /// Adapt `ledger`, starting a bridge thread for the synchronous calls.
    ///
    /// # Errors
    ///
    /// The bridge thread cannot start.
    pub fn new(ledger: Arc<dyn RunLedgerStore>) -> Result<Self> {
        Ok(Self::with_bridge(ledger, Blocking::new()?))
    }

    /// Adapt `ledger`, sharing an existing bridge.
    pub fn with_bridge(ledger: Arc<dyn RunLedgerStore>, bridge: Blocking) -> Self {
        Self { ledger, bridge }
    }

    fn run<T: Send + 'static>(
        &self,
        work: impl std::future::Future<
            Output = std::result::Result<T, tinyagents_session::run_ledger::LedgerError>,
        > + Send
        + 'static,
    ) -> Result<T> {
        Ok(self.bridge.run(work)??)
    }
}

impl TeamLedger for SessionTeamLedger {
    fn upsert_team(&self, upsert: AgentTeamUpsert) -> Result<AgentTeam> {
        let ledger = Arc::clone(&self.ledger);
        self.run(async move { ledger.upsert_agent_team(upsert).await })
    }
    fn get_team(&self, id: &str) -> Result<Option<AgentTeam>> {
        let (ledger, id) = (Arc::clone(&self.ledger), id.to_owned());
        self.run(async move { ledger.get_agent_team(&id).await })
    }
    fn list_teams(&self, request: &AgentTeamListRequest) -> Result<AgentTeamListResponse> {
        let (ledger, request) = (Arc::clone(&self.ledger), request.clone());
        self.run(async move { ledger.list_agent_teams(&request).await })
    }
    fn upsert_member(&self, upsert: AgentTeamMemberUpsert) -> Result<AgentTeamMember> {
        let ledger = Arc::clone(&self.ledger);
        self.run(async move { ledger.upsert_agent_team_member(upsert).await })
    }
    fn list_members(&self, team_id: &str) -> Result<Vec<AgentTeamMember>> {
        let (ledger, team_id) = (Arc::clone(&self.ledger), team_id.to_owned());
        self.run(async move { ledger.list_agent_team_members(&team_id).await })
    }
    fn list_tasks(&self, team_id: &str) -> Result<Vec<AgentTeamTask>> {
        let (ledger, team_id) = (Arc::clone(&self.ledger), team_id.to_owned());
        self.run(async move { ledger.list_agent_team_tasks(&team_id).await })
    }
    fn upsert_task(&self, upsert: AgentTeamTaskUpsert) -> Result<AgentTeamTask> {
        let ledger = Arc::clone(&self.ledger);
        self.run(async move { ledger.upsert_agent_team_task(upsert).await })
    }
    fn claim_task(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        claim_token: &str,
    ) -> Result<ClaimOutcome> {
        let ledger = Arc::clone(&self.ledger);
        let (team, task, member, token) = (
            team_id.to_owned(),
            task_id.to_owned(),
            member_id.to_owned(),
            claim_token.to_owned(),
        );
        self.run(async move {
            ledger
                .claim_agent_team_task(&team, &task, &member, &token)
                .await
        })
    }
    fn complete_task(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        evidence: &[String],
        require_evidence: bool,
    ) -> Result<CompletionOutcome> {
        let ledger = Arc::clone(&self.ledger);
        let (team, task, member, evidence) = (
            team_id.to_owned(),
            task_id.to_owned(),
            member_id.to_owned(),
            evidence.to_vec(),
        );
        self.run(async move {
            ledger
                .complete_agent_team_task(&team, &task, &member, &evidence, require_evidence)
                .await
        })
    }
    fn complete_task_if_claimed(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        claim_token: &str,
        evidence: &[String],
        require_evidence: bool,
    ) -> Result<CompletionOutcome> {
        let (ledger, team, task, member, token, evidence) = (
            Arc::clone(&self.ledger),
            team_id.to_owned(),
            task_id.to_owned(),
            member_id.to_owned(),
            claim_token.to_owned(),
            evidence.to_vec(),
        );
        self.run(async move {
            ledger
                .complete_agent_team_task_if_claimed(
                    &team,
                    &task,
                    &member,
                    &token,
                    &evidence,
                    require_evidence,
                )
                .await
        })
    }
    fn release_task(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        claim_token: &str,
    ) -> Result<bool> {
        let (ledger, team, task, member, token) = (
            Arc::clone(&self.ledger),
            team_id.to_owned(),
            task_id.to_owned(),
            member_id.to_owned(),
            claim_token.to_owned(),
        );
        self.run(async move {
            ledger
                .release_agent_team_task_if_claimed(&team, &task, &member, &token)
                .await
        })
    }
    fn mark_member_running(
        &self,
        team_id: &str,
        member_id: &str,
        task_id: &str,
        worker_thread_id: &str,
        run_id: &str,
    ) -> Result<Option<AgentTeamMember>> {
        let (ledger, team, member, task, thread, run) = (
            Arc::clone(&self.ledger),
            team_id.to_owned(),
            member_id.to_owned(),
            task_id.to_owned(),
            worker_thread_id.to_owned(),
            run_id.to_owned(),
        );
        self.run(async move {
            ledger
                .mark_agent_team_member_running(&team, &member, &task, &thread, &run)
                .await
        })
    }
    fn try_mark_member_running(
        &self,
        team_id: &str,
        member_id: &str,
        task_id: &str,
        worker_thread_id: &str,
        run_id: &str,
    ) -> Result<AgentTeamMemberRunAdmission> {
        let (ledger, team, member, task, thread, run) = (
            Arc::clone(&self.ledger),
            team_id.to_owned(),
            member_id.to_owned(),
            task_id.to_owned(),
            worker_thread_id.to_owned(),
            run_id.to_owned(),
        );
        self.run(async move {
            ledger
                .try_start_agent_team_member(&team, &member, &task, &thread, &run)
                .await
        })
    }
    fn mark_member_idle(
        &self,
        team_id: &str,
        member_id: &str,
        task_id: &str,
        worker_thread_id: &str,
        run_id: &str,
    ) -> Result<bool> {
        let (ledger, team, member, task, thread, run) = (
            Arc::clone(&self.ledger),
            team_id.to_owned(),
            member_id.to_owned(),
            task_id.to_owned(),
            worker_thread_id.to_owned(),
            run_id.to_owned(),
        );
        self.run(async move {
            Ok(ledger
                .mark_agent_team_member_idle_if_current(&team, &member, &task, &thread, &run)
                .await?
                .is_some())
        })
    }
    fn shutdown_member(
        &self,
        team_id: &str,
        member_id: &str,
    ) -> Result<Option<(AgentTeamMember, Vec<String>)>> {
        let ledger = Arc::clone(&self.ledger);
        let (team, member) = (team_id.to_owned(), member_id.to_owned());
        self.run(async move { ledger.shutdown_agent_team_member(&team, &member).await })
    }
    fn append_event(&self, event: RunEventAppend) -> Result<RunEvent> {
        let ledger = Arc::clone(&self.ledger);
        self.run(async move { ledger.append_run_event(event).await })
    }
    fn list_events(&self, request: &RunEventListRequest) -> Result<Vec<RunEvent>> {
        let (ledger, request) = (Arc::clone(&self.ledger), request.clone());
        self.run(async move { Ok(ledger.list_recent_run_events(&request).await?.events) })
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

struct MemberExecutionCleanup<'a, L: TeamLedger> {
    service: &'a TeamService<L>,
    team_id: &'a str,
    task_id: &'a str,
    member_id: &'a str,
    claim_token: &'a str,
    execution: Option<(String, String)>,
    armed: bool,
}

impl<L: TeamLedger> MemberExecutionCleanup<'_, L> {
    fn cleanup(&mut self) -> Result<()> {
        self.service.release_and_idle(
            self.team_id,
            self.task_id,
            self.member_id,
            self.claim_token,
            self.execution
                .as_ref()
                .map(|(thread, run)| (thread.as_str(), run.as_str())),
        )?;
        self.armed = false;
        Ok(())
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl<L: TeamLedger> Drop for MemberExecutionCleanup<'_, L> {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.service.release_and_idle(
                self.team_id,
                self.task_id,
                self.member_id,
                self.claim_token,
                self.execution
                    .as_ref()
                    .map(|(thread, run)| (thread.as_str(), run.as_str())),
            );
        }
    }
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
        let team = self
            .ledger
            .get_team(team_id)?
            .ok_or_else(|| anyhow!("unknown team: {team_id}"))?;
        if team.status == AgentTeamStatus::Closed {
            return Err(anyhow!("team is closed: {team_id}"));
        }
        let existing = self.ledger.list_tasks(team_id)?;
        if let Some(owner) = owner_member_id
            && !self.ledger.list_members(team_id)?.iter().any(|member| {
                member.id == owner && member.member_status != AgentTeamMemberStatus::Stopped
            })
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
        self.ensure_team_active(team_id)?;
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
        Ok(drain_run_events(&self.ledger, team_id)?
            .into_iter()
            .filter(|event| event.event_type == TEAM_MESSAGE_EVENT)
            .take(limit.unwrap_or(u32::MAX) as usize)
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

    /// Claim and execute one team task, then durably complete it.
    ///
    /// Worker failures release the task and return the member to idle before
    /// the original error is propagated. Completion gate failures are also
    /// released so another attempt can claim the task.
    pub async fn start_member<E: TeamMemberExecutor + ?Sized>(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        executor: &E,
        require_evidence: bool,
    ) -> Result<CompletionOutcome> {
        self.start_member_with_started(
            team_id,
            task_id,
            member_id,
            executor,
            require_evidence,
            |_| {},
        )
        .await
    }

    /// Claims and activates a member task, then calls `on_started` before the
    /// executor is polled. A host can return an immediate response from the
    /// callback while this same future continues execution and retains the
    /// normal owner-controlled evidence and cleanup path.
    pub async fn start_member_with_started<E, Started>(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        executor: &E,
        require_evidence: bool,
        on_started: Started,
    ) -> Result<CompletionOutcome>
    where
        E: TeamMemberExecutor + ?Sized,
        Started: FnOnce(TeamMemberStarted),
    {
        self.ensure_member(team_id, member_id)?;
        let claim_token = Uuid::new_v4().to_string();
        let claimed = match self.claim_task(team_id, task_id, member_id, &claim_token)? {
            ClaimOutcome::Claimed(task) => *task,
            ClaimOutcome::AlreadyClaimed => {
                return Err(anyhow!(TeamMemberStartError::AlreadyClaimed {
                    task_id: task_id.to_string(),
                }));
            }
            ClaimOutcome::Blocked { unmet } => {
                return Err(anyhow!(TeamMemberStartError::Blocked {
                    task_id: task_id.to_string(),
                    unmet,
                }));
            }
            ClaimOutcome::UnknownTask => {
                return Err(anyhow!(TeamMemberStartError::UnknownTask {
                    task_id: task_id.to_string(),
                }));
            }
        };
        let mut cleanup = MemberExecutionCleanup {
            service: self,
            team_id,
            task_id,
            member_id,
            claim_token: &claim_token,
            execution: None,
            armed: true,
        };
        let worker_thread_id = format!("team-worker-{}", Uuid::new_v4().simple());
        let run_id = format!("team-run-{}", Uuid::new_v4().simple());
        let member = match self.ledger.try_mark_member_running(
            team_id,
            member_id,
            task_id,
            &worker_thread_id,
            &run_id,
        ) {
            Ok(AgentTeamMemberRunAdmission::Started(member)) => member,
            Ok(AgentTeamMemberRunAdmission::AlreadyActive(member)) => {
                return Err(anyhow!(TeamMemberStartError::AlreadyActive {
                    member_id: member_id.to_owned(),
                    current_task_id: member.current_task_id,
                }));
            }
            Ok(
                AgentTeamMemberRunAdmission::NotAdmissible(_)
                | AgentTeamMemberRunAdmission::Missing,
            ) => {
                return Err(anyhow!(TeamError::UnknownMember {
                    member_id: member_id.to_owned(),
                }));
            }
            Err(error) => {
                return Err(error.context("admit team member run"));
            }
        };
        cleanup.execution = Some((
            member
                .worker_thread_id
                .clone()
                .unwrap_or_else(|| worker_thread_id.clone()),
            member.run_id.clone().unwrap_or_else(|| run_id.clone()),
        ));

        on_started(TeamMemberStarted {
            run_id: member.run_id.clone().unwrap_or_else(|| run_id.clone()),
            member: member.clone(),
            task: claimed.clone(),
        });

        let execution = executor.execute(&member, &claimed).await;
        let completion = match execution {
            Ok(evidence) => self.ledger.complete_task_if_claimed(
                team_id,
                task_id,
                member_id,
                &claim_token,
                &evidence,
                require_evidence,
            ),
            Err(error) => {
                return match cleanup.cleanup() {
                    Ok(()) => Err(error.context("team member execution failed")),
                    Err(cleanup_error) => Err(anyhow!(
                        "team member execution failed: {error}; cleanup failed: {cleanup_error}"
                    )),
                };
            }
        };
        match completion {
            Ok(outcome @ CompletionOutcome::Completed(_)) => {
                self.ledger.mark_member_idle(
                    team_id,
                    member_id,
                    task_id,
                    member
                        .worker_thread_id
                        .as_deref()
                        .unwrap_or(&worker_thread_id),
                    member.run_id.as_deref().unwrap_or(&run_id),
                )?;
                cleanup.disarm();
                Ok(outcome)
            }
            Ok(outcome) => {
                cleanup.cleanup()?;
                Ok(outcome)
            }
            Err(error) => match cleanup.cleanup() {
                Ok(()) => Err(error.context("complete team task")),
                Err(cleanup_error) => Err(anyhow!(
                    "complete team task failed: {error}; cleanup failed: {cleanup_error}"
                )),
            },
        }
    }

    fn release_and_idle(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        claim_token: &str,
        execution: Option<(&str, &str)>,
    ) -> Result<()> {
        let release = self
            .ledger
            .release_task(team_id, task_id, member_id, claim_token);
        let idle = execution.map_or(Ok(false), |(thread, run)| {
            self.ledger
                .mark_member_idle(team_id, member_id, task_id, thread, run)
        });
        match (release, idle) {
            (Ok(_), Ok(_)) => Ok(()),
            (Err(error), Ok(_)) | (Ok(_), Err(error)) => Err(error),
            (Err(release), Err(idle)) => Err(anyhow!(
                "failed to release team task: {release}; failed to idle team member: {idle}"
            )),
        }
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
        if self.ledger.list_members(team_id)?.iter().any(|member| {
            member.id == member_id && member.member_status != AgentTeamMemberStatus::Stopped
        }) {
            Ok(())
        } else {
            Err(anyhow!(TeamError::UnknownMember {
                member_id: member_id.to_string()
            }))
        }
    }

    fn ensure_team_active(&self, team_id: &str) -> Result<()> {
        let team = self
            .ledger
            .get_team(team_id)?
            .ok_or_else(|| anyhow!("unknown team: {team_id}"))?;
        if team.status == AgentTeamStatus::Active {
            Ok(())
        } else {
            Err(anyhow!("team is closed: {team_id}"))
        }
    }
}

#[cfg(test)]
#[path = "service_tests.rs"]
mod tests;

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
#[path = "service_dependency_tests.rs"]
mod dependency_tests;
