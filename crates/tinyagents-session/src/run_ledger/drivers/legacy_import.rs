//! Import of a classic `sessions.db` into the ledger's document collections.
//!
//! Read-only, id-preserving and only-if-absent: every row keeps the id it had
//! (session ids, dense message and tool-call ids and event sequences), a
//! record already present in the target is never overwritten
//! (so an edit made after an earlier import survives a re-run), and the source
//! file is never written. The import is not one transaction; it is idempotent,
//! so a pass that stops partway is finished by running it again. Once every
//! family is in, the full-text index is rebuilt from the imported records with
//! [`reindex_fts`](crate::run_ledger::RunLedgerStore::reindex_fts), because the
//! classic index lived in a SQLite virtual table that cannot be copied.
use super::events::EventHead;
use super::history::{MESSAGE_COUNTER, TOOL_CALL_COUNTER, message_id, tool_call_id};
use super::storage::{
    AGENTS, EFFECTS, EVENT_HEADS, EVENTS, HIST_MESSAGES, HIST_SESSIONS, HIST_TOOL_CALLS,
    HISTORY_OWNER, MEMBERS, TASKS, TEAMS, TELEMETRY, WORKFLOWS, encode, key,
};
use super::*;
use crate::legacy::{QuiescentSource, rows, source};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tinystoragedrivers_core::{Counters, ErrorKind, Precondition, primitives::LEASES};

const LOG_PREFIX: &str = "[session_db:import]";

/// Per-family tally of one import pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportCounts {
    /// Rows the source held.
    pub read: usize,
    /// Rows written by this pass.
    pub imported: usize,
    /// Rows the target already had, left untouched.
    pub already_present: usize,
}

impl ImportCounts {
    fn tally(read: usize, imported: usize) -> Self {
        Self {
            read,
            imported,
            already_present: read - imported,
        }
    }
}

/// What [`DriverRunLedger::import_legacy_database`] did, per family.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyDatabaseReport {
    /// Classic `sessions` rows.
    pub sessions: ImportCounts,
    /// Classic `session_messages` rows, with their original ids.
    pub messages: ImportCounts,
    /// Classic `session_tool_calls` rows, with their original ids.
    pub tool_calls: ImportCounts,
    /// `agent_runs` rows.
    pub agent_runs: ImportCounts,
    /// `run_telemetry` rows.
    pub telemetry: ImportCounts,
    /// `run_events` rows, with their original sequences.
    pub events: ImportCounts,
    /// `workflow_runs` rows, leases included.
    pub workflow_runs: ImportCounts,
    /// `agent_teams` rows.
    pub teams: ImportCounts,
    /// `agent_team_members` rows.
    pub members: ImportCounts,
    /// `agent_team_tasks` rows.
    pub tasks: ImportCounts,
    /// `tool_effects` rows.
    pub tool_effects: ImportCounts,
    /// Full-text rows written by the rebuild that ends the pass.
    pub fts_rows: usize,
}

impl DriverRunLedger {
    /// Write `doc` at `id` unless something is already there; `true` when written.
    async fn put_if_absent(&self, collection: &str, id: &str, doc: Value) -> Result<bool> {
        match self
            .scoped
            .documents()
            .put(collection, id, doc, Precondition::Absent)
            .await
        {
            Ok(_) => Ok(true),
            Err(error)
                if matches!(error.kind(), ErrorKind::Conflict | ErrorKind::AlreadyExists) =>
            {
                Ok(false)
            }
            Err(error) => Err(error.into()),
        }
    }

    /// Raise a dense-id counter so the next allocation is above `last_used`.
    async fn raise_counter(&self, name: &str, last_used: u64) -> Result<()> {
        Counters::new(self.scoped.documents())
            .set_at_least(HISTORY_OWNER, name, last_used)
            .await?;
        Ok(())
    }

    /// Keep a workflow run's classic driver lease as its `tsd.leases` record
    /// (epoch 1), unless a lease for the run already exists.
    async fn import_workflow_lease(&self, run: &WorkflowRun) -> Result<()> {
        let (Some(holder), Some(expires)) = (&run.lease_owner, run.lease_expires_at) else {
            return Ok(());
        };
        let id = key(&[LEDGER_OWNER, &super::storage::workflow_lease_key(&run.id)?])?;
        let body = serde_json::json!({
            "holder": holder,
            "epoch": 1,
            "expires_ms": expires.timestamp_millis(),
        });
        self.put_if_absent(LEASES, &id, body).await?;
        Ok(())
    }

    /// Raise a run's event head so the next sequence is at least `next`.
    async fn raise_event_head(&self, run_id: &str, next: u64) -> Result<()> {
        let wanted = (next <= i64::MAX as u64).then_some(next);
        self.change(EVENT_HEADS, run_id, |previous: Option<EventHead>| {
            Ok(match previous {
                // A terminal head (`None`) is already past every sequence.
                Some(head)
                    if head
                        .next
                        .is_none_or(|have| wanted.is_some_and(|w| have >= w)) =>
                {
                    (None, ())
                }
                _ => (Some(EventHead { next: wanted }), ()),
            })
        })
        .await
        .map_err(Into::into)
    }

    /// Import every classic table of the quiescent `source` that this ledger
    /// holds, then rebuild the full-text index.
    ///
    /// # Errors
    ///
    /// An owner/scope mismatch, an uncheckpointed (WAL or journal) source, a
    /// schema this importer does not know, a row that does not decode, or a
    /// native write refusal. Rows written before the failure stay; running the
    /// import again completes it.
    pub async fn import_legacy_database(
        &self,
        source: &QuiescentSource,
    ) -> Result<LegacyDatabaseReport> {
        self.declare().await?;
        // Everything is read before the first write, so the source connection
        // is never held across an await and a bad row refuses the whole pass
        // before anything lands.
        let loaded = {
            let snapshot = source::open_source(source, self.scope())?;
            rows::require_classic(&snapshot)?;
            Loaded {
                sessions: rows::sessions(&snapshot)?,
                messages: rows::messages(&snapshot)?,
                tool_calls: rows::tool_calls(&snapshot)?,
                agent_runs: rows::agent_runs(&snapshot)?,
                telemetry: rows::telemetry(&snapshot)?,
                events: rows::run_events(&snapshot)?,
                workflows: rows::workflow_runs(&snapshot)?,
                teams: rows::teams(&snapshot)?,
                members: rows::members(&snapshot)?,
                tasks: rows::team_tasks(&snapshot)?,
                effects: rows::tool_effects(&snapshot)?,
            }
        };
        tracing::debug!(
            "{LOG_PREFIX} loaded sessions={} messages={} tool_calls={} agent_runs={} events={}",
            loaded.sessions.len(),
            loaded.messages.len(),
            loaded.tool_calls.len(),
            loaded.agent_runs.len(),
            loaded.events.len()
        );
        let mut report = LegacyDatabaseReport::default();

        // Sessions before their messages, so a message never outlives a session
        // that failed to land.
        let mut imported = 0;
        for session in &loaded.sessions {
            imported += usize::from(
                self.put_if_absent(HIST_SESSIONS, &key(&[&session.id])?, encode(session)?)
                    .await?,
            );
        }
        report.sessions = ImportCounts::tally(loaded.sessions.len(), imported);

        let mut imported = 0;
        let mut highest = 0u64;
        for message in &loaded.messages {
            highest = highest.max(u64::try_from(message.id).unwrap_or(0));
            imported += usize::from(
                self.put_if_absent(HIST_MESSAGES, &message_id(message.id)?, encode(message)?)
                    .await?,
            );
        }
        self.raise_counter(MESSAGE_COUNTER, highest).await?;
        report.messages = ImportCounts::tally(loaded.messages.len(), imported);

        let mut imported = 0;
        let mut highest = 0u64;
        for call in &loaded.tool_calls {
            highest = highest.max(u64::try_from(call.id).unwrap_or(0));
            imported += usize::from(
                self.put_if_absent(HIST_TOOL_CALLS, &tool_call_id(call.id)?, encode(call)?)
                    .await?,
            );
        }
        self.raise_counter(TOOL_CALL_COUNTER, highest).await?;
        report.tool_calls = ImportCounts::tally(loaded.tool_calls.len(), imported);

        let mut imported = 0;
        for run in &loaded.agent_runs {
            imported += usize::from(
                self.put_if_absent(AGENTS, &key(&[&run.id])?, encode(run)?)
                    .await?,
            );
        }
        report.agent_runs = ImportCounts::tally(loaded.agent_runs.len(), imported);

        let mut imported = 0;
        for row in &loaded.telemetry {
            imported += usize::from(
                self.put_if_absent(TELEMETRY, &key(&[&row.run_id])?, encode(row)?)
                    .await?,
            );
        }
        report.telemetry = ImportCounts::tally(loaded.telemetry.len(), imported);

        let mut imported = 0;
        let mut heads: std::collections::BTreeMap<&str, u64> = Default::default();
        for event in &loaded.events {
            let head = heads.entry(event.run_id.as_str()).or_insert(0);
            *head = (*head).max(event.sequence);
            imported += usize::from(
                self.put_if_absent(
                    EVENTS,
                    &key(&[&event.run_id, &event.sequence.to_string()])?,
                    encode(event)?,
                )
                .await?,
            );
        }
        for (run_id, highest) in heads {
            self.raise_event_head(run_id, highest + 1).await?;
        }
        report.events = ImportCounts::tally(loaded.events.len(), imported);

        let mut imported = 0;
        for run in &loaded.workflows {
            imported += usize::from(
                self.put_if_absent(WORKFLOWS, &key(&[&run.id])?, encode(run)?)
                    .await?,
            );
            self.import_workflow_lease(run).await?;
        }
        report.workflow_runs = ImportCounts::tally(loaded.workflows.len(), imported);

        let mut imported = 0;
        for team in &loaded.teams {
            imported += usize::from(
                self.put_if_absent(TEAMS, &key(&[&team.id])?, encode(team)?)
                    .await?,
            );
        }
        report.teams = ImportCounts::tally(loaded.teams.len(), imported);

        let mut imported = 0;
        for member in &loaded.members {
            imported += usize::from(
                self.put_if_absent(MEMBERS, &key(&[&member.id])?, encode(member)?)
                    .await?,
            );
        }
        report.members = ImportCounts::tally(loaded.members.len(), imported);

        let mut imported = 0;
        for task in &loaded.tasks {
            imported += usize::from(
                self.put_if_absent(TASKS, &key(&[&task.id])?, encode(task)?)
                    .await?,
            );
        }
        report.tasks = ImportCounts::tally(loaded.tasks.len(), imported);

        let mut imported = 0;
        for row in &loaded.effects {
            imported += usize::from(
                self.put_if_absent(EFFECTS, &key(&[&row.run_id, &row.call_id])?, encode(row)?)
                    .await?,
            );
        }
        report.tool_effects = ImportCounts::tally(loaded.effects.len(), imported);

        report.fts_rows = self.history_reindex().await?;
        tracing::info!(
            "{LOG_PREFIX} imported sessions={}/{} messages={}/{} tool_calls={}/{} agent_runs={}/{} \
             events={}/{} fts_rows={}",
            report.sessions.imported,
            report.sessions.read,
            report.messages.imported,
            report.messages.read,
            report.tool_calls.imported,
            report.tool_calls.read,
            report.agent_runs.imported,
            report.agent_runs.read,
            report.events.imported,
            report.events.read,
            report.fts_rows
        );
        Ok(report)
    }
}

struct Loaded {
    sessions: Vec<crate::types::SessionRecord>,
    messages: Vec<crate::types::SessionMessage>,
    tool_calls: Vec<crate::types::SessionToolCall>,
    agent_runs: Vec<AgentRun>,
    telemetry: Vec<RunTelemetry>,
    events: Vec<RunEvent>,
    workflows: Vec<WorkflowRun>,
    teams: Vec<AgentTeam>,
    members: Vec<AgentTeamMember>,
    tasks: Vec<AgentTeamTask>,
    effects: Vec<ToolEffectRow>,
}
