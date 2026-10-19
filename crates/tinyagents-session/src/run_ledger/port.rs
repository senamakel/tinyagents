//! Object-safe durable run-ledger storage contract.
//!
//! Every method is bound to one tenant/agent scope by the handle that carries
//! it. Return values, partial-update rules, event sequences (starting at one),
//! and claim/lifecycle fencing are part of the contract.
//! Implementations must preserve transaction boundaries: a status CAS and its
//! optional event commit together, task dependency checks and claims are atomic,
//! and workflow writes validate revision, live owner and lease expiry together.
//!
//! [`DriverRunLedger`](super::DriverRunLedger) implements the port over the
//! storage ports, so any backend a host opened holds the ledger. Retention
//! includes the classic session, message and tool-call history
//! ([`SessionHistoryStore`](super::SessionHistoryStore)) and must not silently
//! omit it. An implementation that cannot meet a method's contract must return
//! an error instead of using another backend.

use super::error::LedgerResult as Result;
use chrono::{DateTime, Utc};

use super::command_center::{AgentWorkRow, CommandCenterView, ControlVerb};
use super::tool_effects::{ToolEffectRow, ToolEffectSettle, ToolEffectStart};
use super::types::*;
use crate::retention::RetentionReport;

/// Bound storage for runs, workflows, coordination and effects.
///
/// Handles are selected by the host, never by a run's payload or tool arguments.
/// Shared deployments must bind their tenant/agent namespace before returning a
/// handle. The classic orphan sweep assumes exclusive process ownership; shared
/// backends must not interrupt another live owner's work at startup.
///
/// Storage operations are async so driver implementations can await their
/// ports without blocking a caller runtime. Pure recovery classifiers remain
/// synchronous. Callback parameters are shared safely across awaited operations.
#[async_trait::async_trait]
pub trait RunLedgerStore: Send + Sync {
    /// Inserts a new [`AgentRun`] or merges fields into an existing one with the
    /// same id, returning the row as stored.
    ///
    /// Most fields are `COALESCE`d against the existing row on conflict, so
    /// passing `None` leaves them unchanged rather than clearing them — except
    /// `status` and `updated_at`, which are always overwritten, and `metadata`,
    /// which only replaces the stored value when the incoming JSON is non-empty
    /// (`{}` is treated as "no metadata supplied"). `kind` additionally refuses
    /// to downgrade a `worker_thread` run back to `subagent`, since a run that
    /// has already been promoted to a worker thread should not silently revert.
    /// Use `transition_agent_run_status` instead when a caller needs to
    /// *clear* `error` or `completed_at`.
    async fn upsert_agent_run(&self, upsert: AgentRunUpsert) -> Result<AgentRun>;

    /// Inserts a new [`WorkflowRun`] or merges fields into an existing one with
    /// the same id, returning the row as stored.
    ///
    /// Bumps `revision` on every call, including the first insert, so
    /// `compare_and_swap_workflow_run` callers always have a fresh fencing
    /// token. When `status` transitions to a terminal value the driver lease
    /// (`lease_owner` / `lease_expires_at`) is cleared, since a finished run has
    /// nothing left to drive. Prefer `compare_and_swap_workflow_run` or
    /// `compare_and_swap_workflow_run_lifecycle` for a driver actively holding
    /// a lease — this plain upsert has no revision fencing of its own and can
    /// clobber a concurrent driver's write.
    async fn upsert_workflow_run(&self, upsert: WorkflowRunUpsert) -> Result<WorkflowRun>;

    /// Atomically lease a workflow run to one driver.  A second live driver gets
    /// the authoritative row as `Busy`; it must not schedule a duplicate phase.
    async fn try_claim_workflow_run(
        &self,
        id: &str,
        owner: &str,
        lease_for: chrono::Duration,
    ) -> Result<WorkflowLeaseClaim>;

    /// Compare-and-swap a workflow transition while renewing its driver's lease.
    /// `None` means another driver or an out-of-band lifecycle operation changed
    /// the row; callers must reload rather than overwrite that state.
    async fn compare_and_swap_workflow_run(
        &self,
        upsert: WorkflowRunUpsert,
        expected_revision: u64,
        owner: &str,
        lease_for: chrono::Duration,
    ) -> Result<Option<WorkflowRun>>;

    /// Renew a live workflow driver's lease without changing its durable revision.
    ///
    /// A phase can legitimately run longer than the lease interval.  Renewing is
    /// deliberately not a state transition: child registration and phase commits
    /// retain their revision fencing, while this small heartbeat only proves that
    /// the same owner is still alive. `false` means ownership was lost and the
    /// caller must cancel its children and stop driving immediately.
    async fn renew_workflow_run_lease(
        &self,
        id: &str,
        owner: &str,
        lease_for: chrono::Duration,
    ) -> Result<bool>;

    /// Compare-and-swap a host lifecycle transition (stop or resume).
    ///
    /// Lifecycle commands are allowed to fence an in-flight driver, but only from
    /// the exact revision their caller observed.  The write always clears the
    /// driver lease.  That makes a stop→resume hand-off safe: the old driver can no
    /// longer commit, and a resumer gets a fresh owner through `try_claim_*`.
    async fn compare_and_swap_workflow_run_lifecycle(
        &self,
        upsert: WorkflowRunUpsert,
        expected_revision: u64,
    ) -> Result<Option<WorkflowRun>>;

    /// Appends a new [`RunEvent`] to a run's event log, allocating its sequence
    /// number atomically.
    ///
    /// `sequence` is `MAX(sequence) + 1` for the run, computed by the same
    /// `INSERT ... SELECT` statement that writes the row (see the inline comment
    /// at the call site) so two connections appending concurrently cannot
    /// compute the same next value and lose one event to a primary-key conflict.
    async fn append_run_event(&self, event: RunEventAppend) -> Result<RunEvent>;

    /// Inserts a new [`RunTelemetry`] row or merges partial fields into an
    /// existing one, returning the row as stored.
    ///
    /// Every counter field is `Option`, and `None` means "leave this field
    /// unchanged" rather than "reset to zero" — see the inline comment at the
    /// call site for why the insert and update sides need different `COALESCE`
    /// targets to make that true on first insert as well as on later updates.
    async fn upsert_run_telemetry(&self, upsert: RunTelemetryUpsert) -> Result<RunTelemetry>;

    /// Fetches a single [`AgentRun`] by id, or `None` if no row matches.
    async fn get_agent_run(&self, id: &str) -> Result<Option<AgentRun>>;

    /// Apply a durable status transition to a single agent run.
    ///
    /// Unlike `upsert_agent_run` — whose `ON CONFLICT` clause `COALESCE`s the
    /// `error` and `completed_at` columns and can therefore only ever *set* them —
    /// this is a direct `UPDATE` that can both set and *clear* both columns. That
    /// is required by control verbs such as "retry", which moves a failed run back
    /// to `pending` and must drop the stale failure reason and completion time.
    ///
    /// `status` is always written. `error` and `completed_at` are written verbatim,
    /// so passing `None` clears the column. `updated_at` is bumped to now. Returns
    /// the freshly-read run, or `None` when no row matched `id` (e.g. it was
    /// deleted between a prior read and this write).
    async fn transition_agent_run_status(
        &self,
        id: &str,
        status: AgentRunStatus,
        error: Option<&str>,
        completed_at: Option<DateTime<Utc>>,
    ) -> Result<Option<AgentRun>>;

    /// Compare-and-set form of `transition_agent_run_status`: move run `id` to
    /// `status` only if it is still in `expected`, and record `event` in the same
    /// transaction, so a status change never commits without its timeline entry
    /// and never lands on top of a status another writer recorded meanwhile.
    ///
    /// Status change and optional event are one atomic commit; a rejected CAS writes neither.
    async fn transition_agent_run_status_from(
        &self,
        id: &str,
        expected: AgentRunStatus,
        status: AgentRunStatus,
        error: Option<&str>,
        completed_at: Option<DateTime<Utc>>,
        event: Option<RunEventAppend>,
    ) -> Result<RunTransition>;

    /// Settle non-terminal `agent_runs` rows left behind by a previous process.
    ///
    /// A freshly-booted core has no in-flight subagents — any detached run task
    /// from a prior process is gone with that process. So a row still marked
    /// `running` (or `pending`) at startup is, by definition, orphaned: its driver
    /// died without firing the host's terminal completion notification, so the
    /// host's run-ledger finalizer never settled it. Without
    /// this sweep those rows render as perpetual "running" timeline entries on every
    /// thread reopen.
    ///
    /// We stamp them `interrupted` (outcome unknown — mirrors the turn-state
    /// `mark_all_interrupted` recovery) and set `completed_at`. `awaiting_user` /
    /// `paused` are intentionally left untouched: those are resumable states a user
    /// may still continue.
    ///
    ///
    /// Only valid under exclusive process ownership; never sweep live shared owners.
    async fn interrupt_orphaned_agent_runs(&self) -> Result<usize>;

    /// Lists agent runs, most-recently-updated first, with optional filters
    /// (status, kind, parent run, parent thread) and pagination.
    ///
    /// `limit` is capped at 500 regardless of the requested value.
    async fn list_agent_runs(&self, request: &AgentRunListRequest) -> Result<AgentRunListResponse>;

    /// Lists a run's events in `sequence` order, optionally starting after a
    /// given cursor (`after_sequence`), for polling "what's new" incrementally.
    ///
    /// `limit` is capped at 1000 regardless of the requested value.
    async fn list_recent_run_events(
        &self,
        request: &RunEventListRequest,
    ) -> Result<RunEventListResponse>;

    /// Fetches a single [`WorkflowRun`] by id, or `None` if no row matches.
    async fn get_workflow_run(&self, id: &str) -> Result<Option<WorkflowRun>>;

    /// List durable workflow runs, most-recently-updated first, with optional
    /// filters (definition id, status, parent thread) and pagination. Mirrors
    /// `list_agent_runs` for the workflow_runs table.
    async fn list_workflow_runs(
        &self,
        request: &WorkflowRunListRequest,
    ) -> Result<WorkflowRunListResponse>;

    /// Insert or update a team row.
    async fn upsert_agent_team(&self, upsert: AgentTeamUpsert) -> Result<AgentTeam>;

    /// Fetch a single team by id.
    async fn get_agent_team(&self, id: &str) -> Result<Option<AgentTeam>>;

    /// List teams, most-recently-updated first, with optional thread/status filters.
    async fn list_agent_teams(
        &self,
        request: &AgentTeamListRequest,
    ) -> Result<AgentTeamListResponse>;

    /// Insert or update a team member. `UNIQUE(team_id, name)` enforces unique names.
    async fn upsert_agent_team_member(
        &self,
        upsert: AgentTeamMemberUpsert,
    ) -> Result<AgentTeamMember>;

    /// Fetch a single member by id.
    async fn get_agent_team_member(&self, id: &str) -> Result<Option<AgentTeamMember>>;

    /// List all members of a team, by creation order.
    async fn list_agent_team_members(&self, team_id: &str) -> Result<Vec<AgentTeamMember>>;

    /// Insert or update a team task.
    async fn upsert_agent_team_task(&self, upsert: AgentTeamTaskUpsert) -> Result<AgentTeamTask>;

    /// Fetch a single task by id.
    async fn get_agent_team_task(&self, id: &str) -> Result<Option<AgentTeamTask>>;

    /// List all tasks of a team, by `order_index` then creation order.
    async fn list_agent_team_tasks(&self, team_id: &str) -> Result<Vec<AgentTeamTask>>;

    /// Atomically claim a task for a member.
    ///
    /// All steps run inside a single `with_connection` transaction so that the
    /// dependency check and the compare-and-swap observe a consistent snapshot:
    /// 1. Resolve the task by `(id, team_id)`; absent → [`ClaimOutcome::UnknownTask`].
    /// 2. For every dependency id, look up its status; collect those not `done`
    ///    into `unmet`. Non-empty → [`ClaimOutcome::Blocked`].
    /// 3. WHERE-guarded `UPDATE ... WHERE claimed_by_member_id IS NULL`: SQLite
    ///    serializes writers, so exactly one concurrent claimer flips the row from
    ///    unclaimed to claimed. `rows_affected == 0` → already taken
    ///    ([`ClaimOutcome::AlreadyClaimed`]); otherwise re-fetch and return
    ///    [`ClaimOutcome::Claimed`].
    async fn claim_agent_team_task(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        claim_token: &str,
    ) -> Result<ClaimOutcome>;

    /// Quality-gate a task's completion and, on pass, transition it to `done`.
    ///
    /// Runs inside a single transaction so the gate evaluation and the status flip
    /// observe one consistent snapshot:
    /// 1. Resolve the task by `(id, team_id)`; absent → [`CompletionOutcome::UnknownTask`].
    /// 2. The completer must be the current claimant and the task must be
    ///    `in_progress`; otherwise [`CompletionOutcome::NotClaimed`].
    /// 3. Evaluate the quality gate (every dependency `done`, claimant matches any
    ///    pre-assigned owner, evidence present when `require_evidence`). Any unmet
    ///    invariant records `gate_status = "failed"` + the joined reasons and leaves
    ///    the task `in_progress` → [`CompletionOutcome::GateFailed`].
    /// 4. On pass, merge `evidence`, set `status = "done"`, `gate_status = "passed"`,
    ///    clear `gate_reason`, re-fetch → [`CompletionOutcome::Completed`].
    async fn complete_agent_team_task(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        evidence: &[String],
        require_evidence: bool,
    ) -> Result<CompletionOutcome>;

    /// Complete a task only while `member_id` still owns `claim_token`.
    /// The claim check, gate evaluation, and status update share the same
    /// transaction so an old worker cannot complete a successor's claim.
    async fn complete_agent_team_task_if_claimed(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        claim_token: &str,
        evidence: &[String],
        require_evidence: bool,
    ) -> Result<CompletionOutcome>;

    /// Stop a team member and release any task it is actively working on.
    ///
    /// In one transaction: unclaim the member's `in_progress` tasks back to `todo`
    /// (clearing claimant + token so another teammate can pick them up), then mark
    /// the member `stopped` and clear its `current_task_id`. Returns the updated
    /// member plus the ids of the tasks that were released, or `None` if the member
    /// is not part of the team.
    async fn shutdown_agent_team_member(
        &self,
        team_id: &str,
        member_id: &str,
    ) -> Result<Option<(AgentTeamMember, Vec<String>)>>;

    /// Legacy unconditional setter for a member's active task and execution ids.
    /// It does not verify that the member is pending or idle and can overwrite a
    /// concurrent active run. New task admission must use
    /// [`RunLedgerStore::try_start_agent_team_member`]. Returns the updated member,
    /// or `None` if the member is not in the team.
    async fn mark_agent_team_member_running(
        &self,
        team_id: &str,
        member_id: &str,
        task_id: &str,
        worker_thread_id: &str,
        run_id: &str,
    ) -> Result<Option<AgentTeamMember>>;

    /// Atomically admit a member to one task execution. Pending and idle
    /// members become active; active members are returned unchanged, and
    /// stopped or missing members are not admitted.
    async fn try_start_agent_team_member(
        &self,
        team_id: &str,
        member_id: &str,
        task_id: &str,
        worker_thread_id: &str,
        run_id: &str,
    ) -> Result<AgentTeamMemberRunAdmission>;

    /// Mark a member idle: status → `idle`, clearing `current_task_id`. The
    /// `worker_thread_id` / `run_id` are intentionally retained as a pointer to the
    /// member's last run for history. Returns the updated member, or `None` if the
    /// member is not in the team. Used when a worker run finishes (completed,
    /// gate-failed, or failed) so the member is free to pick up new work.
    async fn mark_agent_team_member_idle(
        &self,
        team_id: &str,
        member_id: &str,
    ) -> Result<Option<AgentTeamMember>>;

    /// Mark a member idle only if it still owns this exact task execution.
    /// This compare-and-set prevents cleanup from an older worker from idling
    /// a member that has since been stopped and assigned a newer run.
    async fn mark_agent_team_member_idle_if_current(
        &self,
        team_id: &str,
        member_id: &str,
        task_id: &str,
        worker_thread_id: &str,
        run_id: &str,
    ) -> Result<Option<AgentTeamMember>>;

    /// Release a single `in_progress` task back to `todo`, clearing its claim and
    /// resetting the quality gate. Returns `true` if a row was actually released
    /// (the task existed, belonged to the team, and was `in_progress`). Used by the
    /// live runtime when a worker run fails or is aborted, so the task is free for
    /// another teammate — the per-task analogue of the bulk release in
    /// `shutdown_agent_team_member`.
    async fn release_agent_team_task(&self, team_id: &str, task_id: &str) -> Result<bool>;

    /// Release a task only while the specified member still owns the exact
    /// claim token. The comparison and release are one atomic operation.
    async fn release_agent_team_task_if_claimed(
        &self,
        team_id: &str,
        task_id: &str,
        member_id: &str,
        claim_token: &str,
    ) -> Result<bool>;

    /// Records that a tool call has been admitted and is about to execute.
    ///
    /// Idempotent on `(run_id, call_id)`: a second write re-stamps `started_at` and
    /// overwrites the tool, idempotency and summary fields, so a caller that retries
    /// after an ambiguous failure need not special-case "already started".
    async fn record_tool_started(&self, start: ToolEffectStart) -> Result<ToolEffectRow>;

    /// Settles a recorded call to a terminal status (`completed`, `failed`,
    /// `interrupted` or `deferred`) and stamps `settled_at`. A settle for a call
    /// that was never recorded creates its row.
    async fn settle_tool_effect(&self, settle: ToolEffectSettle) -> Result<Option<ToolEffectRow>>;

    /// Lists a run's calls still `started`, oldest first: the calls a resumed run
    /// must classify before it re-executes anything.
    async fn list_unresolved_tool_effects(&self, run_id: &str) -> Result<Vec<ToolEffectRow>>;

    /// Settles one call as `interrupted`: it was in flight across a resume and is
    /// not safe to re-execute blindly.
    async fn mark_tool_effect_interrupted(
        &self,
        run_id: &str,
        call_id: &str,
    ) -> Result<Option<ToolEffectRow>>;

    /// Deletes every **finished** session that ended before `older_than`, together
    /// with its messages, tool calls and search-index entries.
    ///
    /// Running sessions are never touched, whatever their start time: a long-lived
    /// session is not stale data. `session_messages` and `session_tool_calls`
    /// cascade through their foreign keys (the connection sets
    /// `PRAGMA foreign_keys = ON`), but the FTS entries do not — the virtual table
    /// has no foreign keys — so they are removed explicitly.
    async fn prune_sessions_before(&self, older_than: DateTime<Utc>) -> Result<usize>;

    /// Trims a session's transcript to its most recent `keep_last` messages,
    /// returning how many were removed.
    ///
    /// The oldest messages go first. Their FTS entries are content-addressed by
    /// session rather than by message id, so trimming a session's messages
    /// necessarily leaves its index entries stale; call `reindex_fts` afterwards
    /// if search accuracy for trimmed sessions matters.
    async fn trim_session_messages(&self, session_id: &str, keep_last: usize) -> Result<usize>;

    /// Deletes tool-call rows created before `older_than`, returning how many.
    async fn prune_tool_calls_before(&self, older_than: DateTime<Utc>) -> Result<usize>;

    /// Deletes run-ledger events stamped before `older_than`, returning how many.
    ///
    /// Event sequences are per run and allocated as `MAX(sequence) + 1`, so pruning
    /// the head of a run's log does not renumber or collide with later appends.
    async fn prune_run_events_before(&self, older_than: DateTime<Utc>) -> Result<usize>;

    /// Deletes run-telemetry rows last updated before `older_than`, returning how
    /// many.
    async fn prune_run_telemetry_before(&self, older_than: DateTime<Utc>) -> Result<usize>;

    /// Applies one age-based retention pass across every table, returning what it
    /// removed.
    ///
    /// Sessions are pruned first so their messages and tool calls cascade rather
    /// than being deleted twice; the remaining passes then catch orphaned rows that
    /// outlived their session or belong to the run ledger.
    ///
    /// Includes classic sessions/messages/tool calls as well as events and telemetry.
    async fn apply_retention(&self, older_than: DateTime<Utc>) -> Result<RetentionReport>;

    /// Rebuilds `sessions_fts` from the authoritative tables, returning the number
    /// of index rows written.
    ///
    /// The repair path the database never had. Any row whose FTS entry was lost —
    /// to a failure between the two old autocommit statements, to a retention pass
    /// that trimmed messages, or to a change of the
    /// [snippet cap](crate::fts::set_fts_snippet_bytes) — becomes searchable again.
    /// The whole rebuild runs in one transaction, so search is never left with a
    /// half-built index.
    async fn reindex_fts(&self) -> Result<usize>;

    /// Lists recent background agent work grouped by command-center bucket.
    /// Reads at most `limit` (default 200, capped 500) most-recently-updated runs.
    /// Read-only.
    async fn list_agent_work(
        &self,
        limit: Option<u32>,
        display_name: &(dyn for<'a> Fn(&'a str) -> Option<String> + Sync),
    ) -> Result<CommandCenterView>;

    /// Applies a control verb (stop, retry, continue, follow-up) to one run as a
    /// durable transition plus a timeline event in one atomic commit, and returns
    /// the re-projected row.
    async fn apply_control(
        &self,
        run_id: &str,
        verb: ControlVerb,
        message: Option<&str>,
        reason: Option<&str>,
        display_name: &(dyn for<'a> Fn(&'a str) -> Option<String> + Sync),
    ) -> Result<AgentWorkRow>;

    /// Pure recovery classification, preserving the existing missing-effect policy.
    fn classify_recovery_with(
        &self,
        tail: &[super::recovery::DanglingToolCall],
        effects: &[ToolEffectRow],
        missing: super::recovery::MissingEffectRow,
    ) -> Vec<super::recovery::CallRecovery> {
        super::recovery::classify_recovery_with(tail, effects, missing)
    }

    /// Classify recovery under the strict policy that every failed start write aborts execution.
    fn classify_recovery(
        &self,
        tail: &[super::recovery::DanglingToolCall],
        effects: &[ToolEffectRow],
    ) -> Vec<super::recovery::CallRecovery> {
        super::recovery::classify_recovery(tail, effects)
    }

    /// Return the most cautious verdict from an already classified recovery tail.
    fn overall_recovery(
        &self,
        classified: &[super::recovery::CallRecovery],
    ) -> super::recovery::RecoveryClass {
        super::recovery::overall_recovery(classified)
    }
}
