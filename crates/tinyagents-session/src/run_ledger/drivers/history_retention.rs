//! Age and count retention plus index repair for the classic session history.
//!
//! Each batch is atomic and chunked, and every pass is idempotent: a pass that
//! stops partway (a crash, a refused batch) is finished by running it again.
//! The classic single-transaction guarantee cannot hold across a backend that
//! limits batch size; instead children are always deleted before the record
//! that owns them, so an interrupted pass never leaves a record whose children
//! vanished while the parent still claims them as live data.
use super::dto_history::FtsRow;
use super::history::{
    message_fts_id, message_fts_row, session_fts_id, session_fts_row, tool_fts_id, tool_fts_row,
};
use super::storage::{
    EVENTS, HIST_FTS, HIST_MESSAGES, HIST_SESSIONS, HIST_TOOL_CALLS, decode, put_op,
};
use super::*;
use crate::types::{SessionMessage, SessionRecord, SessionStatus, SessionToolCall};
use serde_json::Value;
use std::collections::HashSet;
use tinyagents_storage_kit::{DocumentStoreExt, Precondition, Versioned, WriteOp, WriteResult};

const LOG_PREFIX: &str = "[session_db:retention]";

/// Operations per atomic delete or put batch.
const BATCH: usize = 100;

fn delete_op(collection: &str, id: String) -> WriteOp {
    WriteOp::Delete {
        collection: collection.into(),
        id,
        precondition: Precondition::None,
    }
}

fn removed(results: &[WriteResult]) -> usize {
    results
        .iter()
        .filter(|result| matches!(result, WriteResult::Delete { removed: true }))
        .count()
}

impl DriverRunLedger {
    /// Apply `ops` in atomic chunks, returning every result in order.
    async fn apply_chunked(&self, ops: Vec<WriteOp>) -> Result<Vec<WriteResult>> {
        let mut results = Vec::with_capacity(ops.len());
        let mut ops = ops.into_iter().peekable();
        while ops.peek().is_some() {
            let chunk: Vec<WriteOp> = ops.by_ref().take(BATCH).collect();
            results.extend(
                self.scoped
                    .documents()
                    .atomic_batch_checked(Vec::new(), chunk)
                    .await?,
            );
        }
        Ok(results)
    }

    async fn all_rows(&self, collection: &str, filter: Filter) -> Result<Vec<Versioned<Value>>> {
        self.declare().await?;
        Ok(self
            .scoped
            .documents()
            .query_all(collection, &Query::filter(filter))
            .await?)
    }

    /// Ids of the index rows owned by `session_id`.
    async fn fts_ids_of(&self, session_id: &str) -> Result<Vec<String>> {
        Ok(self
            .all_rows(HIST_FTS, Filter::eq("session_id", session_id))
            .await?
            .into_iter()
            .map(|row| row.id)
            .collect())
    }

    /// Delete `session_id`'s messages, tool calls and index rows.
    async fn delete_session_children(&self, session_id: &str) -> Result<()> {
        let filter = || Filter::eq("session_id", session_id);
        let mut ops: Vec<WriteOp> = Vec::new();
        for row in self.all_rows(HIST_MESSAGES, filter()).await? {
            ops.push(delete_op(HIST_MESSAGES, row.id));
        }
        for row in self.all_rows(HIST_TOOL_CALLS, filter()).await? {
            ops.push(delete_op(HIST_TOOL_CALLS, row.id));
        }
        for id in self.fts_ids_of(session_id).await? {
            ops.push(delete_op(HIST_FTS, id));
        }
        self.apply_chunked(ops).await?;
        Ok(())
    }

    /// Finished sessions whose `ended_at` is before `older_than`.
    async fn expired_sessions(
        &self,
        older_than: DateTime<Utc>,
    ) -> Result<Vec<(Versioned<Value>, SessionRecord)>> {
        let filter = Filter::ne("status", "running")
            .and(Filter::lt("ended_ms", older_than.timestamp_millis() + 1));
        let mut out = Vec::new();
        for row in self.all_rows(HIST_SESSIONS, filter).await? {
            let session: SessionRecord = decode(row.doc.clone())?;
            // Compare instants: the stored time is millisecond-truncated.
            if session.status != SessionStatus::Running
                && session.ended_at.is_some_and(|ended| ended < older_than)
            {
                out.push((row, session));
            }
        }
        Ok(out)
    }

    pub(super) async fn history_prune_sessions(&self, older_than: DateTime<Utc>) -> Result<usize> {
        tracing::debug!("{LOG_PREFIX} prune_sessions_before.entry cutoff={older_than}");
        let mut count = 0;
        for (row, session) in self.expired_sessions(older_than).await? {
            self.delete_session_children(&session.id).await?;
            let deletes = vec![
                delete_op(HIST_FTS, session_fts_id(&session.id)?),
                WriteOp::Delete {
                    collection: HIST_SESSIONS.into(),
                    id: row.id,
                    // A session refreshed since it was read is no longer stale.
                    precondition: Precondition::Version(row.version),
                },
            ];
            match self
                .scoped
                .documents()
                .atomic_batch_checked(Vec::new(), deletes)
                .await
            {
                Ok(results) => count += usize::from(removed(&results[1..]) > 0),
                Err(error) if error.kind() == tinystoragedrivers_types::ErrorKind::Conflict => {}
                Err(error) => return Err(error.into()),
            }
        }
        tracing::debug!("{LOG_PREFIX} prune_sessions_before.exit removed={count}");
        Ok(count)
    }

    pub(super) async fn history_trim_messages(
        &self,
        session_id: &str,
        keep_last: usize,
    ) -> Result<usize> {
        tracing::debug!(
            "{LOG_PREFIX} trim_session_messages.entry session={session_id} keep_last={keep_last}"
        );
        self.declare().await?;
        let rows = self
            .scoped
            .documents()
            .query_all(
                HIST_MESSAGES,
                &Query::filter(Filter::eq("session_id", session_id)).sort(Sort::desc("id")),
            )
            .await?;
        let mut ops = Vec::new();
        for row in rows.into_iter().skip(keep_last) {
            let message: SessionMessage = decode(row.doc)?;
            ops.push(delete_op(HIST_MESSAGES, row.id));
            // The trimmed message's index row goes with it, so search cannot
            // surface a session for text the transcript no longer holds.
            ops.push(delete_op(HIST_FTS, message_fts_id(message.id)?));
        }
        let results = self.apply_chunked(ops).await?;
        let removed = results
            .iter()
            .step_by(2)
            .filter(|result| matches!(result, WriteResult::Delete { removed: true }))
            .count();
        tracing::debug!("{LOG_PREFIX} trim_session_messages.exit removed={removed}");
        Ok(removed)
    }

    pub(super) async fn history_prune_tool_calls(
        &self,
        older_than: DateTime<Utc>,
    ) -> Result<usize> {
        tracing::debug!("{LOG_PREFIX} prune_tool_calls_before.entry cutoff={older_than}");
        let filter = Filter::lt("created_ms", older_than.timestamp_millis() + 1);
        let mut ops = Vec::new();
        for row in self.all_rows(HIST_TOOL_CALLS, filter).await? {
            let call: SessionToolCall = decode(row.doc)?;
            if call.created_at < older_than {
                ops.push(delete_op(HIST_TOOL_CALLS, row.id));
                ops.push(delete_op(HIST_FTS, tool_fts_id(call.id)?));
            }
        }
        let results = self.apply_chunked(ops).await?;
        let count = results
            .iter()
            .step_by(2)
            .filter(|result| matches!(result, WriteResult::Delete { removed: true }))
            .count();
        tracing::debug!("{LOG_PREFIX} prune_tool_calls_before.exit removed={count}");
        Ok(count)
    }

    pub(super) async fn history_prune_events(&self, older_than: DateTime<Utc>) -> Result<usize> {
        tracing::debug!("{LOG_PREFIX} prune_run_events_before.entry cutoff={older_than}");
        let mut ops = Vec::new();
        for row in self.all_rows(EVENTS, Filter::default()).await? {
            let event: RunEvent = decode(row.doc)?;
            // Sequences come from a per-run head, so pruning the front of a log
            // never renumbers or collides with later appends.
            if event.timestamp < older_than {
                ops.push(delete_op(EVENTS, row.id));
            }
        }
        let results = self.apply_chunked(ops).await?;
        let count = removed(&results);
        tracing::debug!("{LOG_PREFIX} prune_run_events_before.exit removed={count}");
        Ok(count)
    }

    pub(super) async fn history_apply_retention(
        &self,
        older_than: DateTime<Utc>,
    ) -> Result<RetentionReport> {
        let sessions = self.history_prune_sessions(older_than).await?;
        // Sessions go first so their calls cascade rather than being counted
        // twice; the rest catches calls that outlived their session.
        let report = RetentionReport {
            sessions,
            messages: 0,
            tool_calls: self.history_prune_tool_calls(older_than).await?,
            run_events: self.history_prune_events(older_than).await?,
            run_telemetry: self.prune_telemetry(older_than).await?,
        };
        tracing::info!(
            "{LOG_PREFIX} apply_retention.exit removed total={} sessions={} tool_calls={} \
             run_events={} run_telemetry={}",
            report.total(),
            report.sessions,
            report.tool_calls,
            report.run_events,
            report.run_telemetry
        );
        Ok(report)
    }

    /// Rebuild the index from the authoritative records; returns rows written.
    ///
    /// Stale rows (for records no longer present, or trimmed under an older
    /// snippet cap) are removed after the fresh rows are written, so search
    /// is never left empty while the rebuild runs.
    pub(super) async fn history_reindex(&self) -> Result<usize> {
        tracing::debug!("{LOG_PREFIX} reindex_fts.entry");
        let mut rows: Vec<(String, FtsRow)> = Vec::new();
        for row in self.all_rows(HIST_SESSIONS, Filter::default()).await? {
            let session: SessionRecord = decode(row.doc)?;
            rows.push((session_fts_id(&session.id)?, session_fts_row(&session)));
        }
        for row in self.all_rows(HIST_MESSAGES, Filter::default()).await? {
            let message: SessionMessage = decode(row.doc)?;
            rows.push((
                message_fts_id(message.id)?,
                message_fts_row(&message.session_id, &message.content),
            ));
        }
        for row in self.all_rows(HIST_TOOL_CALLS, Filter::default()).await? {
            let call: SessionToolCall = decode(row.doc)?;
            rows.push((
                tool_fts_id(call.id)?,
                tool_fts_row(&call.session_id, &call.tool_name),
            ));
        }
        let wanted: HashSet<String> = rows.iter().map(|(id, _)| id.clone()).collect();
        let written = rows.len();
        let mut puts = Vec::with_capacity(written);
        for (id, row) in &rows {
            puts.push(put_op(HIST_FTS, id.clone(), row, Precondition::None)?);
        }
        self.apply_chunked(puts).await?;
        let stale: Vec<WriteOp> = self
            .all_rows(HIST_FTS, Filter::default())
            .await?
            .into_iter()
            .filter(|row| !wanted.contains(&row.id))
            .map(|row| delete_op(HIST_FTS, row.id))
            .collect();
        self.apply_chunked(stale).await?;
        tracing::info!("{LOG_PREFIX} reindex_fts.exit rows={written}");
        Ok(written)
    }
}
