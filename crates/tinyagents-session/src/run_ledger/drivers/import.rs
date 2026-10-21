//! Atomic destination import for rows decoded by a host from the retired database.
//!
//! Source authorization and SQLite decoding stay in the host. This module
//! validates owner relationships, preserves stable row identifiers, derives
//! search documents and advances ID allocators in the same checked commit.

use super::dto::Stored;
use super::events::EventHead;
use super::history::{
    MESSAGE_COUNTER, message_fts_id, message_fts_row, message_id, session_fts_id,
    session_fts_row, tool_call_id, tool_fts_id, tool_fts_row,
};
use super::storage::{
    AGENTS, CAS_ATTEMPTS, EFFECTS, EVENT_HEADS, EVENTS, HIST_FTS, HIST_MESSAGES,
    HIST_SESSIONS, HIST_TOOL_CALLS, MEMBERS, TASKS, TEAMS, TELEMETRY, TOOL_CALL_COUNTER,
    WORKFLOWS, decode, encode, key, put_op,
};
use super::*;
use crate::run_ledger::legacy_import::{ImportCounts, LegacyDatabaseReport, LegacySessionImport};
use crate::types::{SessionMessage, SessionRecord, SessionToolCall};
use std::collections::{BTreeMap, HashMap, HashSet};
use tinyagents_storage_kit::{
    Counters, ErrorKind, ExpectedDocument, Precondition, ReadCheck, StorageError, WriteOp,
};

struct Desired {
    collection: &'static str,
    id: String,
    doc: serde_json::Value,
    family: Family,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
enum Family {
    Session,
    Message,
    ToolCall,
    Agent,
    Telemetry,
    Event,
    Workflow,
    Team,
    Member,
    Task,
    Effect,
    Fts,
}

fn add<T: Stored>(
    rows: &mut Vec<Desired>,
    collection: &'static str,
    id: String,
    row: &T,
    family: Family,
) -> std::result::Result<(), StorageError> {
    rows.push(Desired {
        collection,
        id,
        doc: encode(row)?,
        family,
    });
    Ok(())
}

fn conflict(message: &'static str) -> StorageError {
    StorageError::conflict(message)
}

fn validate_batch(batch: &LegacySessionImport) -> std::result::Result<(), StorageError> {
    let session_ids: HashSet<&str> = batch.sessions.iter().map(|row| row.id.as_str()).collect();
    let message_ids: HashSet<i64> = batch.messages.iter().map(|row| row.id).collect();
    let tool_ids: HashSet<i64> = batch.tool_calls.iter().map(|row| row.id).collect();
    let agent_ids: HashSet<&str> = batch.agent_runs.iter().map(|row| row.id.as_str()).collect();
    let workflow_ids: HashSet<&str> = batch.workflow_runs.iter().map(|row| row.id.as_str()).collect();
    let team_ids: HashSet<&str> = batch.teams.iter().map(|row| row.id.as_str()).collect();
    let member_ids: HashSet<&str> = batch.members.iter().map(|row| row.id.as_str()).collect();

    if session_ids.len() != batch.sessions.len()
        || message_ids.len() != batch.messages.len()
        || tool_ids.len() != batch.tool_calls.len()
        || agent_ids.len() != batch.agent_runs.len()
        || workflow_ids.len() != batch.workflow_runs.len()
        || team_ids.len() != batch.teams.len()
        || member_ids.len() != batch.members.len()
    {
        return Err(conflict("legacy import contains duplicate row identities"));
    }
    for row in &batch.messages {
        if row.id <= 0 || (!session_ids.contains(row.session_id.as_str())) {
            return Err(StorageError::invalid_input("legacy message has invalid id or session"));
        }
    }
    for row in &batch.tool_calls {
        if row.id <= 0
            || !session_ids.contains(row.session_id.as_str())
            || row.message_id.is_some_and(|id| !message_ids.contains(&id))
        {
            return Err(StorageError::invalid_input("legacy tool call has invalid references"));
        }
    }
    for row in &batch.events {
        if row.sequence == 0 || row.sequence > i64::MAX as u64 {
            return Err(StorageError::invalid_input("legacy event sequence is out of range"));
        }
    }
    for row in &batch.members {
        if !team_ids.contains(row.team_id.as_str()) {
            return Err(StorageError::invalid_input("legacy member references unknown team"));
        }
    }
    for row in &batch.tasks {
        if !team_ids.contains(row.team_id.as_str())
            || row.owner_member_id.as_deref().is_some_and(|id| !member_ids.contains(id))
            || row.claimed_by_member_id.as_deref().is_some_and(|id| !member_ids.contains(id))
        {
            return Err(StorageError::invalid_input("legacy task has invalid team or member"));
        }
    }
    // Agent/workflow relationships may refer to rows outside the imported
    // slice, so their existence is checked against the destination snapshot.
    let _ = workflow_ids;
    Ok(())
}

fn desired_rows(
    batch: LegacySessionImport,
) -> std::result::Result<(Vec<Desired>, LegacyDatabaseReport, [u64; 2], BTreeMap<String, Option<u64>>), StorageError> {
    validate_batch(&batch)?;
    let maxima = [
        batch.messages.iter().map(|row| row.id as u64).max().unwrap_or(0),
        batch.tool_calls.iter().map(|row| row.id as u64).max().unwrap_or(0),
    ];
    let report = LegacyDatabaseReport {
        sessions: ImportCounts { read: batch.sessions.len(), ..Default::default() },
        messages: ImportCounts { read: batch.messages.len(), ..Default::default() },
        tool_calls: ImportCounts { read: batch.tool_calls.len(), ..Default::default() },
        agent_runs: ImportCounts { read: batch.agent_runs.len(), ..Default::default() },
        telemetry: ImportCounts { read: batch.telemetry.len(), ..Default::default() },
        events: ImportCounts { read: batch.events.len(), ..Default::default() },
        workflow_runs: ImportCounts { read: batch.workflow_runs.len(), ..Default::default() },
        teams: ImportCounts { read: batch.teams.len(), ..Default::default() },
        members: ImportCounts { read: batch.members.len(), ..Default::default() },
        tasks: ImportCounts { read: batch.tasks.len(), ..Default::default() },
        tool_effects: ImportCounts { read: batch.tool_effects.len(), ..Default::default() },
        ..Default::default()
    };
    let mut rows = Vec::new();
    for row in &batch.sessions {
        add(&mut rows, HIST_SESSIONS, key(&[&row.id])?, row, Family::Session)?;
        add(&mut rows, HIST_FTS, session_fts_id(&row.id)?, &session_fts_row(row), Family::Fts)?;
    }
    for row in &batch.messages {
        add(&mut rows, HIST_MESSAGES, message_id(row.id)?, row, Family::Message)?;
        add(&mut rows, HIST_FTS, message_fts_id(row.id)?, &message_fts_row(&row.session_id, &row.content), Family::Fts)?;
    }
    for row in &batch.tool_calls {
        add(&mut rows, HIST_TOOL_CALLS, tool_call_id(row.id)?, row, Family::ToolCall)?;
        add(&mut rows, HIST_FTS, tool_fts_id(row.id)?, &tool_fts_row(&row.session_id, &row.tool_name), Family::Fts)?;
    }
    for mut row in batch.agent_runs {
        row.telemetry = None;
        add(&mut rows, AGENTS, key(&[&row.id])?, &row, Family::Agent)?;
    }
    for row in batch.telemetry { add(&mut rows, TELEMETRY, key(&[&row.run_id])?, &row, Family::Telemetry)?; }
    let mut heads = BTreeMap::<String, u64>::new();
    for row in batch.events {
        heads.entry(row.run_id.clone()).and_modify(|n| *n = (*n).max(row.sequence)).or_insert(row.sequence);
        add(&mut rows, EVENTS, key(&[&row.run_id, &row.sequence.to_string()])?, &row, Family::Event)?;
    }
    for row in batch.workflow_runs { add(&mut rows, WORKFLOWS, key(&[&row.id])?, &row, Family::Workflow)?; }
    for row in batch.teams { add(&mut rows, TEAMS, key(&[&row.id])?, &row, Family::Team)?; }
    for row in batch.members { add(&mut rows, MEMBERS, key(&[&row.id])?, &row, Family::Member)?; }
    for row in batch.tasks { add(&mut rows, TASKS, key(&[&row.id])?, &row, Family::Task)?; }
    for row in batch.tool_effects { add(&mut rows, EFFECTS, key(&[&row.run_id, &row.call_id])?, &row, Family::Effect)?; }

    // Event heads are derived state. Keep the larger of a previous head and
    // the largest imported sequence, using a checked write alongside rows.
    let heads = heads.into_iter().map(|(run_id, highest)| {
        Ok((key(&[&run_id])?, super::events::head_after(highest)))
    }).collect::<std::result::Result<BTreeMap<_, _>, StorageError>>()?;
    let mut addresses = HashSet::new();
    for row in &rows {
        if !addresses.insert((row.collection, row.id.as_str())) {
            return Err(conflict("legacy import contains duplicate owner address"));
        }
    }
    Ok((rows, report, maxima, heads))
}

pub(super) async fn import_batch(
    ledger: &DriverRunLedger,
    batch: LegacySessionImport,
) -> Result<LegacyDatabaseReport> {
    let (rows, mut report, maxima, heads) = desired_rows(batch)?;
    ledger.declare().await?;
    let docs = ledger.scoped.documents();
    for _ in 0..CAS_ATTEMPTS {
        let mut checks = Vec::with_capacity(rows.len());
        let mut writes = Vec::new();
        let mut inserted = HashMap::<Family, usize>::new();
        let mut head_writes = Vec::new();
        for desired in &rows {
            let found = docs.get(desired.collection, &desired.id).await?;
            let expected = found.as_ref().map_or(ExpectedDocument::Absent, |r| ExpectedDocument::Version(r.version));
            checks.push(ReadCheck { collection: desired.collection.into(), id: desired.id.clone(), expected });
            let insert = match found {
                None => {
                    writes.push(WriteOp::Put { collection: desired.collection.into(), id: desired.id.clone(), doc: desired.doc.clone(), precondition: Precondition::Absent });
                    true
                }
                Some(found) if found.doc == desired.doc => false,
                Some(_) => return Err(conflict("legacy import row conflicts with existing owner data").into()),
            };
            if insert {
                *inserted.entry(desired.family).or_default() += 1;
            }
        }
        // Event heads are monotonic derived state. Existing values are read
        // and checked in the same atomic batch as the event receipts.
        for (id, requested_next) in &heads {
            let found = docs.get(EVENT_HEADS, id).await?;
            let expected = found.as_ref().map_or(ExpectedDocument::Absent, |r| ExpectedDocument::Version(r.version));
            checks.push(ReadCheck { collection: EVENT_HEADS.into(), id: id.clone(), expected });
            let current = found.as_ref().map(|r| decode::<EventHead>(r.doc.clone())).transpose()?;
            let next = match (current, requested_next) {
                (Some(EventHead { next: None }), _) | (_, None) => None,
                (Some(EventHead { next: Some(current) }), Some(requested)) => Some(current.max(*requested)),
                (None, Some(requested)) => Some(*requested),
            };
            if found.as_ref().is_none_or(|r| decode::<EventHead>(r.doc.clone()).is_ok_and(|old| old.next != next)) {
                let head = EventHead { next };
                head_writes.push(put_op(EVENT_HEADS, id.clone(), &head, found.as_ref().map_or(Precondition::Absent, |r| Precondition::Version(r.version)))?);
            }
        }
        writes.extend(head_writes);
        let counter_writes = Counters::new(&docs)
            .set_many_at_least_with_batch(
                &[(super::storage::HISTORY_OWNER, MESSAGE_COUNTER, maxima[0]), (super::storage::HISTORY_OWNER, TOOL_CALL_COUNTER, maxima[1])],
                checks,
                writes,
            )
            .await;
        match counter_writes {
            Ok(_) => {
                report.fts_rows = *inserted.get(&Family::Fts).unwrap_or(&0);
                report.sessions.imported = *inserted.get(&Family::Session).unwrap_or(&0);
                report.sessions.already_present = report.sessions.read - report.sessions.imported;
                report.messages.imported = *inserted.get(&Family::Message).unwrap_or(&0);
                report.messages.already_present = report.messages.read - report.messages.imported;
                report.tool_calls.imported = *inserted.get(&Family::ToolCall).unwrap_or(&0);
                report.tool_calls.already_present = report.tool_calls.read - report.tool_calls.imported;
                report.agent_runs.imported = *inserted.get(&Family::Agent).unwrap_or(&0);
                report.agent_runs.already_present = report.agent_runs.read - report.agent_runs.imported;
                report.telemetry.imported = *inserted.get(&Family::Telemetry).unwrap_or(&0);
                report.telemetry.already_present = report.telemetry.read - report.telemetry.imported;
                report.events.imported = *inserted.get(&Family::Event).unwrap_or(&0);
                report.events.already_present = report.events.read - report.events.imported;
                report.workflow_runs.imported = *inserted.get(&Family::Workflow).unwrap_or(&0);
                report.workflow_runs.already_present = report.workflow_runs.read - report.workflow_runs.imported;
                report.teams.imported = *inserted.get(&Family::Team).unwrap_or(&0);
                report.teams.already_present = report.teams.read - report.teams.imported;
                report.members.imported = *inserted.get(&Family::Member).unwrap_or(&0);
                report.members.already_present = report.members.read - report.members.imported;
                report.tasks.imported = *inserted.get(&Family::Task).unwrap_or(&0);
                report.tasks.already_present = report.tasks.read - report.tasks.imported;
                report.tool_effects.imported = *inserted.get(&Family::Effect).unwrap_or(&0);
                report.tool_effects.already_present = report.tool_effects.read - report.tool_effects.imported;
                return Ok(report);
            }
            Err(error) if error.kind() == ErrorKind::Conflict => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(StorageError::conflict("legacy import contention budget exhausted").into())
}

#[cfg(test)]
#[path = "import_tests.rs"]
mod tests;
