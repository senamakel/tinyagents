//! Classic session history (sessions, messages, tool calls) as documents.
//!
//! Every searchable write lands the record and its full-text row in one
//! validated batch, so a record is never committed without its index entry.
//! Index rows have deterministic ids derived from the record they shadow, which
//! lets retention remove exactly the rows of the records it deletes and lets
//! [`DriverRunLedger::reindex_fts`](RunLedgerStore::reindex_fts) rebuild the
//! index idempotently.
use super::dto::Stored;
use super::dto_history::FtsRow;
use super::storage::{
    CAS_ATTEMPTS, HIST_FTS, HIST_MESSAGES, HIST_SESSIONS, HIST_TOOL_CALLS, HISTORY_OWNER, decode,
    key, now, put_op,
};
use super::*;
use crate::fts::{fts_snippet_bytes, snippet};
use crate::run_ledger::history::{
    MAX_TOOL_OUTPUT_BYTES, NewSessionMessage, NewSessionToolCall, SessionEnd, SessionHistoryStore,
    SessionStart,
};
use crate::types::{
    SessionMessage, SessionRecord, SessionSearchParams, SessionSearchResult, SessionStatus,
    SessionToolCall,
};
use serde_json::{Value, json};
use std::collections::HashSet;
use tinyagents_storage_kit::{
    Counters, DocumentStoreExt, ErrorKind, ExpectedDocument, ID_FIELD, Precondition, ReadCheck,
    Versioned, WriteOp,
};

const LOG_PREFIX: &str = "[session_db]";

type StoreResult<T> = std::result::Result<T, StorageError>;

/// The `tsd.counters` names of the dense message and tool-call id sequences
/// (owner `history`).
pub(super) const MESSAGE_COUNTER: &str = "messages";
pub(super) const TOOL_CALL_COUNTER: &str = "tool_calls";

pub(super) fn session_fts_id(session_id: &str) -> StoreResult<String> {
    key(&["s", session_id])
}
pub(super) fn message_fts_id(id: i64) -> StoreResult<String> {
    key(&["m", &id.to_string()])
}
pub(super) fn tool_fts_id(id: i64) -> StoreResult<String> {
    key(&["t", &id.to_string()])
}
pub(super) fn message_id(id: i64) -> StoreResult<String> {
    key(&[&id.to_string()])
}
pub(super) fn tool_call_id(id: i64) -> StoreResult<String> {
    key(&[&id.to_string()])
}

fn put_fts(id: String, row: &FtsRow) -> StoreResult<WriteOp> {
    put_op(HIST_FTS, id, row, Precondition::None)
}

pub(super) fn session_fts_row(session: &SessionRecord) -> FtsRow {
    FtsRow {
        session_id: session.id.clone(),
        agent_name: session.agent_definition_name.clone(),
        content: String::new(),
        tool_name: String::new(),
    }
}

pub(super) fn message_fts_row(session_id: &str, content: &str) -> FtsRow {
    FtsRow {
        session_id: session_id.into(),
        agent_name: String::new(),
        content: snippet(content, fts_snippet_bytes()).into(),
        tool_name: String::new(),
    }
}

pub(super) fn tool_fts_row(session_id: &str, tool_name: &str) -> FtsRow {
    FtsRow {
        session_id: session_id.into(),
        agent_name: String::new(),
        content: String::new(),
        tool_name: tool_name.into(),
    }
}

/// Truncate `output` to [`MAX_TOOL_OUTPUT_BYTES`] on a character boundary.
fn bounded_output(output: &str) -> String {
    if output.len() <= MAX_TOOL_OUTPUT_BYTES {
        return output.to_owned();
    }
    let mut bounded = snippet(output, MAX_TOOL_OUTPUT_BYTES).to_owned();
    bounded.push_str("\n...[truncated]");
    bounded
}

impl DriverRunLedger {
    pub(super) async fn read_session(&self, id: &str) -> Result<Option<Versioned<SessionRecord>>> {
        Ok(self.read::<SessionRecord>(HIST_SESSIONS, id).await?)
    }

    /// Commit `writes` guarded by `checks`, retrying only version conflicts.
    async fn commit_history(&self, checks: Vec<ReadCheck>, writes: Vec<WriteOp>) -> Result<bool> {
        match self
            .scoped
            .documents()
            .atomic_batch_checked(checks, writes)
            .await
        {
            Ok(_) => Ok(true),
            Err(error) if error.kind() == ErrorKind::Conflict => Ok(false),
            Err(error) => Err(error.into()),
        }
    }

    /// Allocate the next id of `counter` and write `build(id)` guarded by the
    /// session's version, in one atomic commit.
    ///
    /// The id comes from `tsd.counters` ([`Counters::next`]) before the commit,
    /// so unlike the single-document counter it replaces it is not part of the
    /// atomic batch: a commit that fails (or the session ending first) leaves a
    /// gap in the dense id sequence, and a retry after a lost session-version
    /// check keeps the id it drew rather than drawing another.
    async fn append_counted<R: Send>(
        &self,
        session_id: &str,
        counter: &str,
        build: impl Fn(i64) -> Result<(Vec<WriteOp>, R)> + Send + Sync,
    ) -> Result<R> {
        if self.read_session(session_id).await?.is_none() {
            return Err(
                StorageError::invalid_input(format!("session '{session_id}' not found")).into(),
            );
        }
        let id = Counters::new(self.scoped.documents())
            .next(HISTORY_OWNER, counter)
            .await?;
        if id > i64::MAX as u64 {
            return Err(StorageError::serialization("history id exhausted").into());
        }
        let (writes, result) = build(id as i64)?;
        for _ in 0..CAS_ATTEMPTS {
            let Some(session) = self.read_session(session_id).await? else {
                return Err(StorageError::invalid_input(format!(
                    "session '{session_id}' not found"
                ))
                .into());
            };
            let checks = vec![ReadCheck {
                collection: HIST_SESSIONS.into(),
                id: key(&[session_id])?,
                expected: ExpectedDocument::Version(session.version),
            }];
            if self.commit_history(checks, writes.clone()).await? {
                return Ok(result);
            }
        }
        Err(StorageError::conflict("history id allocation contention budget exhausted").into())
    }

    /// Page `filter` over `collection` ordered by `sort`, returning rows and the total.
    async fn page_sorted<T: Stored + Send>(
        &self,
        collection: &str,
        filter: Filter,
        sorts: &[Sort],
        limit: u64,
        offset: u64,
    ) -> Result<(Vec<T>, u64)> {
        self.declare().await?;
        let total = self.scoped.documents().count(collection, &filter).await?;
        let mut rows = Vec::new();
        if limit == 0 {
            return Ok((rows, total));
        }
        let mut skip = offset;
        let mut query = Query::filter(filter).limit(256);
        for sort in sorts {
            query = query.sort(sort.clone());
        }
        loop {
            let page = self.scoped.documents().query(collection, &query).await?;
            for row in page.items {
                if skip > 0 {
                    skip -= 1;
                    continue;
                }
                rows.push(decode(row.doc)?);
                if rows.len() as u64 == limit {
                    return Ok((rows, total));
                }
            }
            match page.next {
                Some(cursor) => query.cursor = Some(cursor),
                None => return Ok((rows, total)),
            }
        }
    }

    /// Session ids that have a full-text row matching every term of `text`.
    async fn fts_session_ids(&self, text: &str) -> Result<HashSet<String>> {
        self.declare().await?;
        let terms: Vec<&str> = text.split_whitespace().collect();
        let mut candidates: Option<HashSet<String>> = None;
        for term in &terms {
            let hits = self
                .scoped
                .documents()
                .search(HIST_FTS, term, FTS_HIT_LIMIT)
                .await?;
            let ids: HashSet<String> = hits.into_iter().map(|hit| hit.id).collect();
            candidates = Some(match candidates {
                None => ids,
                Some(previous) => previous.intersection(&ids).cloned().collect(),
            });
            if candidates.as_ref().is_some_and(HashSet::is_empty) {
                return Ok(HashSet::new());
            }
        }
        let wanted: Vec<Vec<String>> = terms
            .iter()
            .map(|term| tinystoragedrivers_types::value::tokens(&Value::String((*term).into())))
            .collect();
        let mut sessions = HashSet::new();
        for id in candidates.unwrap_or_default() {
            let Some(found) = self.scoped.documents().get(HIST_FTS, &id).await? else {
                continue;
            };
            let row: FtsRow = decode(found.doc)?;
            let have: HashSet<String> = [&row.agent_name, &row.content, &row.tool_name]
                .into_iter()
                .flat_map(|field| tinystoragedrivers_types::value::tokens(&json!(field)))
                .collect();
            // The driver ranks by any-token overlap; a quoted multi-token term
            // must still match on every token within the one row.
            if wanted.iter().flatten().all(|token| have.contains(token)) {
                sessions.insert(row.session_id);
            }
        }
        Ok(sessions)
    }

    async fn tool_session_ids(&self, tool_name: &str) -> Result<HashSet<String>> {
        let rows = self
            .scoped
            .documents()
            .query_all(
                HIST_TOOL_CALLS,
                &Query::filter(Filter::eq("tool_name", tool_name)),
            )
            .await?;
        rows.into_iter()
            .map(|row| Ok(decode::<SessionToolCall>(row.doc)?.session_id))
            .collect::<StoreResult<_>>()
            .map_err(Into::into)
    }
}

/// Upper bound on index rows the driver returns for one search term.
const FTS_HIT_LIMIT: usize = 100_000;

#[async_trait::async_trait]
impl SessionHistoryStore for DriverRunLedger {
    async fn record_session_start(&self, start: SessionStart) -> Result<SessionRecord> {
        tracing::debug!(
            "{LOG_PREFIX} record_session_start id={} agent={}",
            start.id,
            start.agent_definition_id
        );
        self.declare().await?;
        let record = SessionRecord {
            id: start.id.clone(),
            agent_definition_id: start.agent_definition_id,
            agent_definition_name: start.agent_definition_name,
            session_key: start.session_key,
            parent_session_id: start.parent_session_id,
            thread_id: start.thread_id,
            source_channel: start.source_channel,
            status: SessionStatus::Running,
            model: start.model,
            turn_count: 0,
            input_tokens: 0,
            output_tokens: 0,
            cached_input_tokens: 0,
            cost_usd: 0.0,
            transcript_path: start.transcript_path,
            started_at: now(),
            ended_at: None,
        };
        let writes = vec![
            put_op(
                HIST_SESSIONS,
                key(&[&start.id])?,
                &record,
                Precondition::Absent,
            )?,
            put_fts(session_fts_id(&start.id)?, &session_fts_row(&record))?,
        ];
        self.scoped
            .documents()
            .atomic_batch_checked(Vec::new(), writes)
            .await?;
        Ok(record)
    }

    async fn record_session_end(&self, id: &str, end: SessionEnd) -> Result<SessionRecord> {
        tracing::debug!(
            "{LOG_PREFIX} record_session_end id={id} status={}",
            end.status.as_str()
        );
        self.change(HIST_SESSIONS, id, |previous: Option<SessionRecord>| {
            let Some(mut record) = previous else {
                return Err(StorageError::invalid_input(format!(
                    "session '{id}' not found"
                )));
            };
            record.status = end.status;
            record.turn_count = end.turn_count;
            record.input_tokens = end.input_tokens;
            record.output_tokens = end.output_tokens;
            record.cached_input_tokens = end.cached_input_tokens;
            record.cost_usd = end.cost_usd;
            record.ended_at = Some(now());
            Ok((Some(record.clone()), record))
        })
        .await
        .map_err(Into::into)
    }

    async fn record_message(&self, message: NewSessionMessage) -> Result<i64> {
        tracing::trace!(
            "{LOG_PREFIX} record_message session={} role={} len={}",
            message.session_id,
            message.role,
            message.content.len()
        );
        self.declare().await?;
        let session_id = message.session_id.clone();
        self.append_counted(&session_id, MESSAGE_COUNTER, |id| {
            let record = SessionMessage {
                id,
                session_id: message.session_id.clone(),
                role: message.role.clone(),
                content: message.content.clone(),
                reasoning_content: message
                    .reasoning_content
                    .clone()
                    .filter(|value| !value.trim().is_empty()),
                model: message.model.clone(),
                input_tokens: message.input_tokens,
                output_tokens: message.output_tokens,
                cost_usd: message.cost_usd,
                created_at: now(),
            };
            let writes = vec![
                put_op(
                    HIST_MESSAGES,
                    message_id(id)?,
                    &record,
                    Precondition::Absent,
                )?,
                put_fts(
                    message_fts_id(id)?,
                    &message_fts_row(&message.session_id, &message.content),
                )?,
            ];
            Ok((writes, id))
        })
        .await
    }

    async fn record_tool_call(&self, call: NewSessionToolCall) -> Result<i64> {
        tracing::trace!(
            "{LOG_PREFIX} record_tool_call session={} tool={} status={}",
            call.session_id,
            call.tool_name,
            call.status
        );
        self.declare().await?;
        let session_id = call.session_id.clone();
        self.append_counted(&session_id, TOOL_CALL_COUNTER, |id| {
            let record = SessionToolCall {
                id,
                session_id: call.session_id.clone(),
                message_id: call.message_id,
                tool_name: call.tool_name.clone(),
                tool_input: call.tool_input.clone(),
                tool_output: call.tool_output.as_deref().map(bounded_output),
                status: call.status.clone(),
                duration_ms: call.duration_ms,
                created_at: now(),
            };
            let writes = vec![
                put_op(
                    HIST_TOOL_CALLS,
                    tool_call_id(id)?,
                    &record,
                    Precondition::Absent,
                )?,
                put_fts(
                    tool_fts_id(id)?,
                    &tool_fts_row(&call.session_id, &call.tool_name),
                )?,
            ];
            Ok((writes, id))
        })
        .await
    }

    async fn get_session(&self, id: &str) -> Result<Option<SessionRecord>> {
        Ok(self.read_session(id).await?.map(|row| row.doc))
    }

    async fn list_sessions(
        &self,
        limit: Option<u32>,
        offset: Option<u32>,
        status: Option<&str>,
        parent_id: Option<&str>,
    ) -> Result<SessionSearchResult> {
        let mut filter = Filter::default();
        if let Some(status) = status {
            filter = filter.and(Filter::eq("status", status));
        }
        if let Some(parent) = parent_id {
            filter = filter.and(Filter::eq("parent_session_id", parent));
        }
        let (sessions, total) = self
            .page_sorted(
                HIST_SESSIONS,
                filter,
                &[Sort::desc("started_ms"), Sort::desc(ID_FIELD)],
                u64::from(limit.unwrap_or(50).min(500)),
                u64::from(offset.unwrap_or(0)),
            )
            .await?;
        Ok(SessionSearchResult { sessions, total })
    }

    async fn search_sessions(&self, params: &SessionSearchParams) -> Result<SessionSearchResult> {
        tracing::debug!(
            "{LOG_PREFIX} search_sessions query={} agent={} tool={}",
            params.query.as_deref().unwrap_or("-"),
            params.agent_id.as_deref().unwrap_or("-"),
            params.tool_name.as_deref().unwrap_or("-"),
        );
        let mut filter = Filter::default();
        for (field, value) in [
            ("agent_definition_id", &params.agent_id),
            ("source_channel", &params.source_channel),
            ("parent_session_id", &params.parent_session_id),
            ("status", &params.status),
            ("thread_id", &params.thread_id),
        ] {
            if let Some(value) = value {
                filter = filter.and(Filter::eq(field, value.as_str()));
            }
        }
        let mut allowed: Option<HashSet<String>> = None;
        if let Some(text) = params.query.as_ref().filter(|q| !q.trim().is_empty()) {
            allowed = Some(self.fts_session_ids(text).await?);
        }
        if let Some(tool) = &params.tool_name {
            let ids = self.tool_session_ids(tool).await?;
            allowed = Some(match allowed {
                None => ids,
                Some(previous) => previous.intersection(&ids).cloned().collect(),
            });
        }
        let limit = u64::from(params.limit.unwrap_or(50).min(500));
        let offset = u64::from(params.offset.unwrap_or(0));
        let Some(allowed) = allowed else {
            let (sessions, total) = self
                .page_sorted(
                    HIST_SESSIONS,
                    filter,
                    &[Sort::desc("started_ms"), Sort::desc(ID_FIELD)],
                    limit,
                    offset,
                )
                .await?;
            return Ok(SessionSearchResult { sessions, total });
        };
        self.declare().await?;
        let mut matched: Vec<SessionRecord> = Vec::new();
        let mut query = Query::filter(filter)
            .sort(Sort::desc("started_ms"))
            .sort(Sort::desc(ID_FIELD))
            .limit(256);
        loop {
            let page = self.scoped.documents().query(HIST_SESSIONS, &query).await?;
            for row in page.items {
                let session: SessionRecord = decode(row.doc)?;
                if allowed.contains(&session.id) {
                    matched.push(session);
                }
            }
            match page.next {
                Some(cursor) => query.cursor = Some(cursor),
                None => break,
            }
        }
        let total = matched.len() as u64;
        let sessions = matched
            .into_iter()
            .skip(usize::try_from(offset).unwrap_or(usize::MAX))
            .take(usize::try_from(limit).unwrap_or(usize::MAX))
            .collect();
        Ok(SessionSearchResult { sessions, total })
    }

    async fn list_messages(
        &self,
        session_id: &str,
        limit: Option<u32>,
    ) -> Result<Vec<SessionMessage>> {
        let (rows, _) = self
            .page_sorted(
                HIST_MESSAGES,
                Filter::eq("session_id", session_id),
                &[Sort::asc("id")],
                u64::from(limit.unwrap_or(200).min(1000)),
                0,
            )
            .await?;
        Ok(rows)
    }

    async fn list_tool_calls(
        &self,
        session_id: &str,
        limit: Option<u32>,
    ) -> Result<Vec<SessionToolCall>> {
        let (rows, _) = self
            .page_sorted(
                HIST_TOOL_CALLS,
                Filter::eq("session_id", session_id),
                &[Sort::asc("id")],
                u64::from(limit.unwrap_or(200).min(1000)),
                0,
            )
            .await?;
        Ok(rows)
    }

    async fn list_children(&self, session_id: &str) -> Result<Vec<SessionRecord>> {
        let (rows, _) = self
            .page_sorted(
                HIST_SESSIONS,
                Filter::eq("parent_session_id", session_id),
                &[Sort::asc("started_ms"), Sort::asc(ID_FIELD)],
                u64::MAX,
                0,
            )
            .await?;
        Ok(rows)
    }

    async fn mark_interrupted(&self) -> Result<usize> {
        self.declare().await?;
        let running = self
            .scoped
            .documents()
            .query_all(
                HIST_SESSIONS,
                &Query::filter(Filter::eq("status", "running")),
            )
            .await?;
        let now = now();
        let mut changed = 0;
        for row in running {
            let mut session: SessionRecord = decode(row.doc)?;
            session.status = SessionStatus::Interrupted;
            session.ended_at = Some(now);
            let put = self
                .scoped
                .documents()
                .put(
                    HIST_SESSIONS,
                    &row.id,
                    super::storage::encode(&session)?,
                    Precondition::Version(row.version),
                )
                .await;
            match put {
                Ok(_) => changed += 1,
                // A concurrent writer finished or refreshed it; not stale any more.
                Err(error) if error.kind() == ErrorKind::Conflict => {}
                Err(error) => return Err(error.into()),
            }
        }
        if changed > 0 {
            tracing::info!("{LOG_PREFIX} marked {changed} running session(s) as interrupted");
        }
        Ok(changed)
    }
}
