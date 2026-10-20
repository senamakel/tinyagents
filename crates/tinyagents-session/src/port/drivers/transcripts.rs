//! [`TranscriptLocator`] and [`TranscriptHistory`] over a driver
//! [`DocumentStore`].
//!
//! # Layout
//!
//! Each transcript stem is an append-only log of entries in
//! [`ENTRIES`], one document per write, numbered `0, 1, 2, …`:
//!
//! ```text
//! { "stem", "seq", "written", "meta"?, "set"?, "extend"?, "tools"?,
//!   "partial"?, "request_id"?, "clear"?, "seal"? }
//! ```
//!
//! Replaying the entries in order rebuilds the transcript: `set` replaces the
//! logical messages (a first write or a compaction), `extend` appends to them
//! (an ordinary turn only stores its new rows), `partial` records a
//! display-only partial, `clear` empties both, and `seal` closes the
//! generation. A writer claims the next number with an insert-only write, so
//! two writers racing for the same transcript (two processes sharing a
//! database) are serialized: the loser re-reads, sees the winner's entry, and
//! decides again — which is also how a write that races a seal is refused.
//!
//! [`INDEX`] holds one small document per stem (id `Key::of([stem])`) with the
//! lookup fields (`thread_id`, `agent_id`, `agent_name`, `subagent`), so
//! "newest root transcript of this thread" is one indexed query instead of a
//! scan of every log. Its application `created_ms` field preserves the
//! transcript's original creation time across cross-collection migration,
//! independently of host-assigned record metadata. It is refreshed after each
//! write that changes the fields. A successor generation is reserved as a `tsd.leases`
//! record (see `transcripts_reservation.rs`) before its predecessor is sealed.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tinyagents_harness::store::SchemaAdmission;
use tinyagents_storage_kit::primitives::{Leases, Repo};
use tinyagents_storage_kit::{Blocking, DocumentStoreExt};
use tinyagents_storage_kit::{
    CollectionSpec, ErrorKind, Filter, ID_FIELD, IndexSpec, Key, Query, Sort, StorageError,
    WriteOptions,
};
use tinystoragedrivers_api::DocumentStore;
use tokio::sync::{Mutex, OnceCell};

use super::super::helpers::MAX_GENERATIONS;
use crate::transcript::{
    SessionRef, SessionTranscript, TranscriptHistory, TranscriptLocator, TranscriptMessage,
    TranscriptMeta, TranscriptPartial, TranscriptRead, TranscriptTurn, TurnUsage,
    same_transcript_messages, session_stem, stamped_rows,
};
use reservation::Reservation;
pub(super) use reservation::leases_spec;

/// One document per transcript stem: the lookup fields.
pub(super) const INDEX: &str = "session.transcripts";
const CREATED_FIELD: &str = "created_ms";
/// One document per write to a transcript: its append-only log.
pub(super) const ENTRIES: &str = "session.transcript_entries";

/// The body version both collections are written at.
pub(crate) const SCHEMA_VERSION: u32 = 2;

/// Index refresh attempts after a durable write before leaving the repair
/// to the next write or read.
const INDEX_ATTEMPTS: usize = 3;

/// Insert attempts before a contended write gives up.
const CAS_ATTEMPTS: usize = 64;

/// The id of `stem`'s index document.
fn index_id(stem: &str) -> Result<String, StorageError> {
    Key::of([stem]).map(|key| key.as_str().to_owned())
}

/// The id of entry `seq` of `stem`: zero-padded, so ids list in log order.
fn entry_id(stem: &str, seq: u64) -> Result<String, StorageError> {
    Key::of([stem, format!("{seq:010}").as_str()]).map(|key| key.as_str().to_owned())
}

/// Whether `stem` names a sub-agent transcript: `__` separates a parent stem
/// from its child's.
fn is_subagent(stem: &str) -> bool {
    stem.split_once("__")
        .is_some_and(|(parent, child)| !parent.is_empty() && !child.is_empty())
}

fn serialization(error: &serde_json::Error) -> StorageError {
    StorageError::serialization(error.to_string())
}

/// A display-only partial as stored.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredPartial {
    content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    iteration: Option<u32>,
}

impl From<&TranscriptPartial> for StoredPartial {
    fn from(partial: &TranscriptPartial) -> Self {
        Self {
            content: partial.content.clone(),
            reasoning_content: partial.reasoning_content.clone(),
            iteration: partial.iteration,
        }
    }
}

impl From<StoredPartial> for TranscriptPartial {
    fn from(stored: StoredPartial) -> Self {
        Self {
            content: stored.content,
            reasoning_content: stored.reasoning_content,
            iteration: stored.iteration,
        }
    }
}

/// One entry of a transcript's log.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Entry {
    /// Whether this entry makes the transcript exist. Only a bare seal does
    /// not: sealing a generation nobody wrote leaves it unwritten.
    #[serde(default)]
    written: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    meta: Option<TranscriptMeta>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    set: Option<Vec<TranscriptMessage>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    extend: Option<Vec<TranscriptMessage>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tools: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    partial: Option<StoredPartial>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    request_id: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    clear: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    seal: bool,
}

/// A transcript rebuilt from its log, up to `next_seq`.
#[derive(Debug, Default)]
struct Replay {
    next_seq: u64,
    meta: Option<TranscriptMeta>,
    messages: Vec<TranscriptMessage>,
    tools: Option<Value>,
    partials: Vec<(TranscriptPartial, Option<String>)>,
    written: bool,
    sealed: bool,
    /// The lookup fields last written to [`INDEX`] by this handle.
    indexed: Option<Value>,
    /// The generation reservation this handle was opened with, until its
    /// first write claims it (see [`HistoryInner::claim_reservation`]).
    reservation: Option<Reservation>,
}

impl Replay {
    fn apply(&mut self, seq: u64, entry: Entry) {
        self.next_seq = seq + 1;
        if let Some(meta) = entry.meta {
            self.meta = Some(meta);
        }
        if let Some(set) = entry.set {
            self.messages = normalized(set);
        }
        if let Some(extend) = entry.extend {
            self.messages.extend(normalized(extend));
        }
        if let Some(tools) = entry.tools {
            self.tools = Some(tools);
        }
        if let Some(partial) = entry.partial {
            self.partials.push((partial.into(), entry.request_id));
        }
        if entry.clear {
            self.messages.clear();
            self.partials.clear();
        }
        self.sealed |= entry.seal;
        self.written |= entry.written;
    }
}

fn normalized(rows: Vec<TranscriptMessage>) -> Vec<TranscriptMessage> {
    rows.into_iter()
        .map(TranscriptMessage::normalized)
        .collect()
}

/// The declarations shared by bootstrap and lazy locator initialization.
pub(super) fn collection_specs() -> [CollectionSpec; 2] {
    [
        CollectionSpec::new(ENTRIES)
            .index(IndexSpec::new("by_stem", ["stem", "seq"]))
            .index(IndexSpec::new("by_stem_written", ["stem", "written"])),
        CollectionSpec::new(INDEX)
            .index(IndexSpec::new("by_thread", ["thread_id", CREATED_FIELD]))
            .index(IndexSpec::new(
                "by_agent_name",
                ["agent_name", CREATED_FIELD],
            ))
            .index(IndexSpec::new("by_agent_id", ["agent_id", CREATED_FIELD])),
    ]
}

/// `query` ordered newest first by the transcript's original creation time,
/// ties broken by id so the order is stable.
fn newest_first(query: Query) -> Query {
    query
        .sort(Sort::desc(CREATED_FIELD))
        .sort(Sort::desc(ID_FIELD))
}

/// `query` ordered oldest first, with the same stable tie-break.
pub(super) fn oldest_first(query: Query) -> Query {
    query
        .sort(Sort::asc(CREATED_FIELD))
        .sort(Sort::asc(ID_FIELD))
}

/// Admits the collections (and the lease collection reservations use), once
/// per locator.
#[derive(Debug, Default)]
struct Declared {
    admitted: OnceCell<()>,
    schema_admission: SchemaAdmission,
}

impl Declared {
    async fn ensure(&self, docs: &Arc<dyn DocumentStore>) -> Result<(), StorageError> {
        self.admitted
            .get_or_try_init(|| async {
                for spec in collection_specs() {
                    self.schema_admission.admit(docs.as_ref(), &spec).await?;
                }
                self.schema_admission
                    .admit(docs.as_ref(), &leases_spec())
                    .await?;
                Ok(())
            })
            .await
            .map(|_| ())
    }
}

/// Runs `future` on `bridge`, flattening the bridge's own failure into the
/// driver error.
fn run_on<T, Fut>(bridge: &Blocking, future: Fut) -> Result<T, StorageError>
where
    Fut: Future<Output = Result<T, StorageError>> + Send + 'static,
    T: Send + 'static,
{
    bridge.run(future)?
}

// ── History ──────────────────────────────────────────────────────────

/// The typed handles a locator and its histories write through.
struct Tables {
    /// Writes of [`ENTRIES`] at [`SCHEMA_VERSION`].
    entries: Repo<Value>,
    /// Writes of [`INDEX`] at [`SCHEMA_VERSION`].
    index: Repo<Value>,
    /// Generation reservations.
    leases: Leases,
}

impl Tables {
    fn new(docs: &Arc<dyn DocumentStore>) -> Self {
        let options = WriteOptions::new().schema(SCHEMA_VERSION);
        Self {
            entries: Repo::new(docs, ENTRIES)
                .expect("the entries collection name is valid")
                .with_options(options),
            index: Repo::new(docs, INDEX)
                .expect("the index collection name is valid")
                .with_options(options),
            leases: Leases::new(docs),
        }
    }
}

/// A [`TranscriptHistory`] bound to one stem of a driver-backed locator.
///
/// Handles are cheap and independent: two handles on one stem (or two
/// processes) stay consistent because every write claims the next log number
/// with an insert-only write and re-reads on a clash.
pub struct DriverTranscriptHistory {
    inner: Arc<HistoryInner>,
    bridge: Blocking,
    path: PathBuf,
}

struct HistoryInner {
    docs: Arc<dyn DocumentStore>,
    tables: Arc<Tables>,
    declared: Arc<Declared>,
    stem: String,
    seed: TranscriptMeta,
    replay: Mutex<Replay>,
}

impl std::fmt::Debug for DriverTranscriptHistory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DriverTranscriptHistory")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl DriverTranscriptHistory {
    /// The display-only partials recorded so far, oldest first, with their
    /// request ids.
    ///
    /// # Errors
    ///
    /// When the log cannot be read.
    pub fn partials(&self) -> anyhow::Result<Vec<(TranscriptPartial, Option<String>)>> {
        let inner = Arc::clone(&self.inner);
        Ok(run_on(&self.bridge, async move {
            let mut replay = inner.replay.lock().await;
            inner.refresh(&mut replay).await?;
            Ok(replay.partials.clone())
        })?)
    }

    /// Appends `entry` built by `build` from the current replay; returns
    /// whether anything was written (`build` answers `None` to write
    /// nothing).
    fn commit<B>(&self, build: B) -> anyhow::Result<bool>
    where
        B: Fn(&Replay, &TranscriptMeta) -> anyhow::Result<Option<Entry>> + Send + 'static,
    {
        let inner = Arc::clone(&self.inner);
        run_on(&self.bridge, async move { Ok(inner.commit(build).await) })?
    }

    /// Seals this generation; later writes to it are refused.
    ///
    /// With a `baseline`, the seal is conditional on the transcript still
    /// holding exactly those messages, checked against the same replay the
    /// seal is claimed on: a turn another writer committed after the caller
    /// read the baseline makes the seal fail instead of being dropped from
    /// the successor.
    fn seal(&self, baseline: Option<Vec<TranscriptMessage>>) -> anyhow::Result<()> {
        self.commit(move |replay, _| {
            if let Some(baseline) = &baseline {
                let current: &[TranscriptMessage] = if replay.written {
                    &replay.messages
                } else {
                    &[]
                };
                anyhow::ensure!(
                    same_transcript_messages(current, baseline),
                    "transcript baseline is stale; reload the session before creating a generation"
                );
            }
            Ok((!replay.sealed).then(|| Entry {
                seal: true,
                ..Entry::default()
            }))
        })
        .map(|_| ())
    }

    /// Records the display-only `partial` of an interrupted turn, unless the
    /// generation is sealed or the partial is empty.
    fn record_partial(
        &self,
        partial: &TranscriptPartial,
        request_id: Option<&str>,
    ) -> anyhow::Result<bool> {
        if partial.content.is_empty() {
            return Ok(false);
        }
        let partial = StoredPartial::from(partial);
        let request_id = request_id.map(str::to_string);
        self.commit(move |replay, seed| {
            Ok((!replay.sealed).then(|| Entry {
                written: true,
                meta: (!replay.written).then(|| seed.clone()),
                partial: Some(partial.clone()),
                request_id: request_id.clone(),
                ..Entry::default()
            }))
        })
    }
}

impl HistoryInner {
    /// Applies every entry written since `replay` was last brought up to date.
    async fn refresh(&self, replay: &mut Replay) -> Result<(), StorageError> {
        self.declared.ensure(&self.docs).await?;
        let query = Query::filter(
            Filter::eq("stem", self.stem.as_str()).and(Filter::gte("seq", replay.next_seq)),
        )
        .sort(Sort::asc("seq"));
        for stored in self.docs.query_all(ENTRIES, &query).await? {
            let seq = stored
                .doc
                .get("seq")
                .and_then(Value::as_u64)
                .ok_or_else(|| {
                    StorageError::serialization(format!(
                        "transcript entry {} has no seq",
                        stored.id
                    ))
                })?;
            let entry: Entry = serde_json::from_value(stored.doc).map_err(|error| {
                StorageError::serialization(format!(
                    "transcript entry {} is unreadable: {error}",
                    stored.id
                ))
            })?;
            replay.apply(seq, entry);
        }
        Ok(())
    }

    async fn commit<B>(&self, build: B) -> anyhow::Result<bool>
    where
        B: Fn(&Replay, &TranscriptMeta) -> anyhow::Result<Option<Entry>>,
    {
        let mut replay = self.replay.lock().await;
        for _ in 0..CAS_ATTEMPTS {
            self.refresh(&mut replay).await?;
            let Some(entry) = build(&replay, &self.seed)? else {
                return Ok(false);
            };
            self.claim_reservation(&mut replay).await?;
            let seq = replay.next_seq;
            let mut doc = serde_json::to_value(&entry).map_err(|error| serialization(&error))?;
            if let Value::Object(fields) = &mut doc {
                fields.insert("stem".into(), json!(self.stem));
                fields.insert("seq".into(), json!(seq));
            }
            match self
                .tables
                .entries
                .insert(&entry_id(&self.stem, seq)?, &doc)
                .await
            {
                Ok(_) => {
                    replay.apply(seq, entry);
                    // The entry is durable: report the write as done. A
                    // failed index refresh is retried a few times here, then
                    // by this handle's next write and by any read of the
                    // transcript; it never makes the durable write fail.
                    let mut last = None;
                    for _ in 0..INDEX_ATTEMPTS {
                        match self.index(&mut replay).await {
                            Ok(()) => {
                                last = None;
                                break;
                            }
                            Err(error) => last = Some(error),
                        }
                    }
                    if let Some(error) = last {
                        tracing::warn!(
                            target: "tinyagents_session::port::drivers",
                            stem = %self.stem,
                            seq,
                            %error,
                            "[session-store] transcript index refresh failed; retried on the next write or read"
                        );
                    }
                    return Ok(true);
                }
                Err(error) if error.kind() == ErrorKind::Conflict => {}
                Err(error) => return Err(error.into()),
            }
        }
        anyhow::bail!(
            "transcript {} kept changing under {CAS_ATTEMPTS} write attempts",
            self.stem
        )
    }

    /// Claims the generation reservation this handle was opened with, once,
    /// before its first write: a renewal of the lease it reserved. If another
    /// process took the reservation over in the meantime (this one stalled
    /// past the reservation's lifetime), the generation is theirs and this
    /// handle's writes are refused instead of overwriting it.
    async fn claim_reservation(&self, replay: &mut Replay) -> anyhow::Result<()> {
        let Some(held) = replay.reservation.as_ref() else {
            return Ok(());
        };
        reservation::claim(&self.tables.leases, &self.stem, held).await?;
        replay.reservation = None;
        Ok(())
    }

    /// Releases the reservation of a generation this handle never wrote.
    async fn release_reservation(&self, replay: &mut Replay) -> Result<(), StorageError> {
        let Some(held) = replay.reservation.take() else {
            return Ok(());
        };
        reservation::release(&self.tables.leases, &self.stem, &held)
            .await
            .map(drop)
    }

    /// Brings the index up to this replay on a read, best effort. A write
    /// whose index refresh failed is otherwise only repaired by the next
    /// write, and a transcript nobody writes again would stay invisible to
    /// thread and agent lookups. Usually one read of a fresh index document.
    async fn repair_index(&self, replay: &mut Replay) {
        if let Err(error) = self.index(replay).await {
            tracing::debug!(
                target: "tinyagents_session::port::drivers",
                stem = %self.stem,
                %error,
                "[session-store] transcript index repair on read failed"
            );
        }
    }

    /// Refreshes this stem's [`INDEX`] document when its lookup fields
    /// changed.
    ///
    /// Fenced by log position: the document records the entry it was built
    /// from (`indexed_seq`), and a handle whose replay is older than that
    /// leaves it alone, so a delayed writer never puts back stale lookup
    /// fields. The write itself is one [`Repo::update`] compare-and-swap.
    async fn index(&self, replay: &mut Replay) -> Result<(), StorageError> {
        if !replay.written {
            return Ok(());
        }
        let meta = replay.meta.as_ref().unwrap_or(&self.seed);
        let created_ms = chrono::DateTime::parse_from_rfc3339(&meta.created)
            .map(|created| created.timestamp_millis())
            .unwrap_or_default();
        let fields = json!({
            "stem": self.stem,
            "subagent": is_subagent(&self.stem),
            "written": true,
            "thread_id": meta.thread_id,
            "agent_id": meta.agent_id,
            "agent_name": meta.agent_name,
            "created_ms": created_ms,
        });
        if replay.indexed.as_ref() == Some(&fields) {
            return Ok(());
        }
        let seq = replay.next_seq.saturating_sub(1);
        self.tables
            .index
            .update(&index_id(&self.stem)?, |existing| {
                let newer = existing.as_ref().is_some_and(|found| {
                    found.get("written") == Some(&json!(true))
                        && found
                            .get("indexed_seq")
                            .and_then(Value::as_u64)
                            .is_some_and(|indexed| indexed >= seq)
                });
                if newer {
                    // Someone indexed this entry or a later one: the document
                    // is at least as fresh as this replay.
                    return Ok(None);
                }
                let mut doc = fields.clone();
                doc["indexed_seq"] = json!(seq);
                Ok(Some(doc))
            })
            .await?;
        replay.indexed = Some(fields);
        Ok(())
    }
}

impl TranscriptRead for DriverTranscriptHistory {
    fn path(&self) -> &Path {
        &self.path
    }

    fn read_session(&self) -> anyhow::Result<Option<SessionTranscript>> {
        let inner = Arc::clone(&self.inner);
        Ok(run_on(&self.bridge, async move {
            let mut replay = inner.replay.lock().await;
            inner.refresh(&mut replay).await?;
            inner.repair_index(&mut replay).await;
            Ok(replay.written.then(|| SessionTranscript {
                meta: replay.meta.clone().unwrap_or_else(|| inner.seed.clone()),
                messages: replay.messages.clone(),
                tools: replay.tools.clone(),
            }))
        })?)
    }

    fn read_display(&self) -> anyhow::Result<Option<crate::transcript::DisplaySessionTranscript>> {
        self.display()
    }
}

/// The entry recording a turn whose logical messages are now `next`.
///
/// Mirrors the JSONL writer: the turn's `prev` must be what is stored
/// (otherwise the caller's view is stale and the write is refused rather than
/// allowed to replace newer rows), an extension stores only its new rows and
/// anything else stores the whole set, and the written rows carry the turn's
/// usage and request id exactly as a transcript file records them.
fn turn_entry(replay: &Replay, turn: &TurnRecord) -> anyhow::Result<Option<Entry>> {
    anyhow::ensure!(!replay.sealed, "transcript generation is sealed");
    let stored: &[TranscriptMessage] = if replay.written {
        &replay.messages
    } else {
        &[]
    };
    anyhow::ensure!(
        !replay.written || same_transcript_messages(stored, &turn.prev),
        "transcript baseline is stale; reload the session before persisting"
    );
    let common = stored
        .iter()
        .zip(&turn.next)
        .take_while(|(left, right)| left.same_row_as(right))
        .count();
    let usage = turn.turn_usage.as_ref();
    let request_id = turn.request_id.as_deref();
    let (set, extend) = if replay.written && common == stored.len() {
        (
            None,
            Some(stamped_rows(&turn.next[common..], usage, request_id)),
        )
    } else {
        (Some(stamped_rows(&turn.next, usage, request_id)), None)
    };
    Ok(Some(Entry {
        written: true,
        meta: Some(turn.meta.clone()),
        set,
        extend,
        tools: turn.tools.clone(),
        partial: turn.partial.clone(),
        request_id: turn.request_id.clone(),
        ..Entry::default()
    }))
}

/// An owned [`TranscriptTurn`], so it can cross onto the bridge.
struct TurnRecord {
    prev: Vec<TranscriptMessage>,
    next: Vec<TranscriptMessage>,
    meta: TranscriptMeta,
    turn_usage: Option<TurnUsage>,
    tools: Option<Value>,
    request_id: Option<String>,
    partial: Option<StoredPartial>,
}

impl TurnRecord {
    fn new(turn: &TranscriptTurn<'_>, partial: Option<&TranscriptPartial>) -> Self {
        Self {
            prev: normalized(turn.prev.to_vec()),
            next: normalized(turn.next.to_vec()),
            meta: turn.meta.clone(),
            turn_usage: turn.turn_usage.cloned(),
            tools: turn.tools.cloned(),
            request_id: turn.request_id.map(str::to_string),
            partial: partial
                .filter(|partial| !partial.content.is_empty())
                .map(StoredPartial::from),
        }
    }
}

impl TranscriptHistory for DriverTranscriptHistory {
    fn append_turn(&self, turn: TranscriptTurn<'_>) -> anyhow::Result<()> {
        let record = TurnRecord::new(&turn, None);
        self.commit(move |replay, _| turn_entry(replay, &record))
            .map(|_| ())
    }

    fn append_turn_with_partial(
        &self,
        turn: TranscriptTurn<'_>,
        partial: Option<&TranscriptPartial>,
    ) -> anyhow::Result<()> {
        let record = TurnRecord::new(&turn, partial);
        self.commit(move |replay, _| turn_entry(replay, &record))
            .map(|_| ())
    }

    fn messages(&self) -> anyhow::Result<Vec<TranscriptMessage>> {
        let inner = Arc::clone(&self.inner);
        Ok(run_on(&self.bridge, async move {
            let mut replay = inner.replay.lock().await;
            inner.refresh(&mut replay).await?;
            Ok(replay.messages.clone())
        })?)
    }

    fn append(&self, message: TranscriptMessage) -> anyhow::Result<()> {
        let message = message.normalized();
        self.commit(move |replay, seed| {
            anyhow::ensure!(!replay.sealed, "transcript generation is sealed");
            Ok(Some(Entry {
                written: true,
                meta: (!replay.written).then(|| seed.clone()),
                extend: Some(vec![message.clone()]),
                ..Entry::default()
            }))
        })
        .map(|_| ())
    }

    fn replace(&self, messages: &[TranscriptMessage]) -> anyhow::Result<()> {
        let messages = normalized(messages.to_vec());
        self.commit(move |replay, seed| {
            anyhow::ensure!(!replay.sealed, "transcript generation is sealed");
            Ok(Some(Entry {
                written: true,
                meta: (!replay.written).then(|| seed.clone()),
                set: Some(messages.clone()),
                ..Entry::default()
            }))
        })
        .map(|_| ())
    }

    fn clear(&self) -> anyhow::Result<()> {
        // Clearing a reserved generation nobody wrote gives the reservation
        // back, so the compaction can be retried at once instead of after
        // the stale timeout.
        let inner = Arc::clone(&self.inner);
        let released = run_on(&self.bridge, async move {
            let mut replay = inner.replay.lock().await;
            inner.refresh(&mut replay).await?;
            if replay.written || replay.reservation.is_none() {
                return Ok(false);
            }
            inner.release_reservation(&mut replay).await?;
            Ok(true)
        })?;
        if released {
            return Ok(());
        }
        self.commit(|replay, _| {
            anyhow::ensure!(!replay.sealed, "transcript generation is sealed");
            Ok(replay.written.then(|| Entry {
                written: true,
                clear: true,
                ..Entry::default()
            }))
        })
        .map(|_| ())
    }
}

// ── Locator ──────────────────────────────────────────────────────────

/// A [`TranscriptLocator`] keeping transcripts in a driver
/// [`DocumentStore`]; see the module docs for the layout.
#[derive(Clone)]
pub struct DriverTranscriptLocator {
    docs: Arc<dyn DocumentStore>,
    tables: Arc<Tables>,
    bridge: Blocking,
    label: String,
    declared: Arc<Declared>,
}

impl std::fmt::Debug for DriverTranscriptLocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DriverTranscriptLocator")
            .field("label", &self.label)
            .finish_non_exhaustive()
    }
}

impl DriverTranscriptLocator {
    /// Transcripts in `docs`, with synchronous calls run on `bridge`.
    ///
    /// `label` names where `docs` points (driver, database, scope). It is
    /// the locator's [`destination_key`](TranscriptLocator::destination_key)
    /// and the prefix of every handle's path, so two locators should share a
    /// label only when they address the same data.
    pub fn new(docs: Arc<dyn DocumentStore>, bridge: Blocking, label: impl Into<String>) -> Self {
        Self::new_with_schema_admission(docs, bridge, label, SchemaAdmission::Declare)
    }

    /// Transcripts with an immutable schema policy shared by their history handles.
    pub fn new_with_schema_admission(
        docs: Arc<dyn DocumentStore>,
        bridge: Blocking,
        label: impl Into<String>,
        schema_admission: SchemaAdmission,
    ) -> Self {
        Self {
            tables: Arc::new(Tables::new(&docs)),
            docs,
            bridge,
            label: label.into(),
            declared: Arc::new(Declared {
                admitted: OnceCell::new(),
                schema_admission,
            }),
        }
    }

    fn handle(&self, stem: &str, seed: TranscriptMeta) -> DriverTranscriptHistory {
        self.reserved_handle(stem, seed, None)
    }

    /// A handle that holds the generation reservation `reservation`; its
    /// first write claims it (or is refused if it was taken over).
    fn reserved_handle(
        &self,
        stem: &str,
        seed: TranscriptMeta,
        reservation: Option<Reservation>,
    ) -> DriverTranscriptHistory {
        DriverTranscriptHistory {
            inner: Arc::new(HistoryInner {
                docs: Arc::clone(&self.docs),
                tables: Arc::clone(&self.tables),
                declared: Arc::clone(&self.declared),
                stem: stem.to_string(),
                seed,
                replay: Mutex::new(Replay {
                    reservation,
                    ..Replay::default()
                }),
            }),
            bridge: self.bridge.clone(),
            path: PathBuf::from(format!("{}/{stem}", self.label)),
        }
    }

    fn run<T, F, Fut>(&self, op: F) -> Result<T, StorageError>
    where
        F: FnOnce(Arc<dyn DocumentStore>, Arc<Declared>) -> Fut,
        Fut: Future<Output = Result<T, StorageError>> + Send + 'static,
        T: Send + 'static,
    {
        run_on(
            &self.bridge,
            op(Arc::clone(&self.docs), Arc::clone(&self.declared)),
        )
    }

    /// The stem of the newest written root transcript matching `filter`.
    fn newest_root_stem(&self, filter: Filter, what: &str) -> Option<String> {
        let found = self.run(|docs, declared| async move {
            declared.ensure(&docs).await?;
            let query = newest_first(Query::filter(
                filter
                    .and(Filter::eq("subagent", false))
                    .and(Filter::eq("written", true)),
            ))
            .limit(1);
            Ok(docs
                .query(INDEX, &query)
                .await?
                .items
                .into_iter()
                .next()
                .and_then(|found| {
                    found
                        .doc
                        .get("stem")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                }))
        });
        found.unwrap_or_else(|error| {
            tracing::warn!(
                target: "tinyagents_session::port::drivers",
                locator = %self.label,
                lookup = what,
                %error,
                "[session-store] transcript lookup failed"
            );
            None
        })
    }

    /// The newest written root transcript matching `filter`.
    fn newest_root(&self, filter: Filter, what: &str) -> Option<Arc<dyn TranscriptRead>> {
        self.newest_root_stem(filter, what).map(|stem| {
            let seed = discovered_seed(&stem);
            Arc::new(self.handle(&stem, seed)) as Arc<dyn TranscriptRead>
        })
    }
}

/// A seed for a handle on a transcript found by lookup. It is used only if
/// the transcript turns out to be unwritten, which a lookup never returns.
fn discovered_seed(stem: &str) -> TranscriptMeta {
    TranscriptMeta {
        agent_name: stem.to_string(),
        agent_id: None,
        agent_type: None,
        dispatcher: String::new(),
        provider: None,
        model: None,
        created: String::new(),
        updated: String::new(),
        turn_count: 0,
        prefix_message_count: None,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: None,
        task_id: None,
        session_id: None,
        parent_session_id: None,
    }
}

impl TranscriptLocator for DriverTranscriptLocator {
    fn destination_key(&self) -> Option<String> {
        Some(format!("{}/transcripts", self.label))
    }

    fn latest_for_agent(&self, agent_name: &str) -> Option<Arc<dyn TranscriptRead>> {
        let filter = Filter::eq("agent_name", agent_name).or(Filter::eq("agent_id", agent_name));
        self.newest_root(filter, "latest_for_agent")
    }

    fn root_for_thread(&self, thread_id: &str) -> Option<Arc<dyn TranscriptRead>> {
        self.root_for_thread_scoped(thread_id, None)
    }

    fn root_for_thread_scoped(
        &self,
        thread_id: &str,
        agent_id: Option<&str>,
    ) -> Option<Arc<dyn TranscriptRead>> {
        let thread_id = thread_id.trim();
        if thread_id.is_empty() {
            return None;
        }
        let mut filter = Filter::eq("thread_id", thread_id);
        if let Some(agent_id) = agent_id {
            filter = filter.and(Filter::eq("agent_id", agent_id));
        }
        self.newest_root(filter, "root_for_thread")
    }

    fn open_stem(
        &self,
        stem: &str,
        seed: TranscriptMeta,
    ) -> anyhow::Result<Arc<dyn TranscriptHistory>> {
        Ok(Arc::new(self.handle(stem, seed)))
    }

    fn thread_transcripts(
        &self,
        thread_id: &str,
        agent_id: Option<&str>,
    ) -> Option<crate::transcript::ThreadTranscripts> {
        self.list_thread(thread_id, agent_id)
    }

    /// Folds the thread's pre-identity roots into `session` once; see
    /// `transcripts_adoption.rs`.
    fn adopt_legacy(
        &self,
        session: &SessionRef,
        thread_id: &str,
        seed: &TranscriptMeta,
    ) -> anyhow::Result<Option<crate::transcript::SessionAdoption>> {
        self.adopt(session, thread_id, seed)
    }

    fn session_exists(&self, session: &SessionRef) -> bool {
        let stem = session_stem(session);
        let written = self.run(|docs, declared| async move {
            declared.ensure(&docs).await?;
            docs.count(
                ENTRIES,
                &Filter::eq("stem", stem).and(Filter::eq("written", true)),
            )
            .await
        });
        match written {
            Ok(count) => count > 0,
            Err(error) => {
                tracing::warn!(
                    target: "tinyagents_session::port::drivers",
                    locator = %self.label,
                    %error,
                    "[session-store] transcript existence check failed"
                );
                false
            }
        }
    }

    fn read_session_transcript(&self, session: &SessionRef) -> Option<Arc<dyn TranscriptRead>> {
        if !self.session_exists(session) {
            return None;
        }
        let stem = session_stem(session);
        let seed = discovered_seed(&stem);
        Some(Arc::new(self.handle(&stem, seed)))
    }

    fn append_interrupted_partial(
        &self,
        thread_id: &str,
        agent_id: Option<&str>,
        partial: &TranscriptPartial,
        request_id: Option<&str>,
    ) -> anyhow::Result<bool> {
        if partial.content.is_empty() {
            return Ok(false);
        }
        let thread_id = thread_id.trim();
        if thread_id.is_empty() {
            return Ok(false);
        }
        let mut filter = Filter::eq("thread_id", thread_id);
        if let Some(agent_id) = agent_id {
            filter = filter.and(Filter::eq("agent_id", agent_id));
        }
        let Some(stem) = self.newest_root_stem(filter, "append_interrupted_partial") else {
            return Ok(false);
        };
        self.handle(&stem, discovered_seed(&stem))
            .record_partial(partial, request_id)
    }

    /// Reserves the successor generation, then seals `session`. The
    /// reservation is what keeps two compactions from both opening the
    /// successor; it is released again if the seal fails.
    fn begin_generation(
        &self,
        session: &SessionRef,
        seed: TranscriptMeta,
    ) -> anyhow::Result<(SessionRef, Arc<dyn TranscriptHistory>)> {
        self.begin_generation_checked(session, seed, None)
    }

    /// [`Self::begin_generation`], sealing only if the predecessor still
    /// holds exactly `baseline` — checked as part of the seal itself, so a
    /// turn committed in between cannot be lost from the successor.
    fn begin_generation_from_baseline(
        &self,
        session: &SessionRef,
        seed: TranscriptMeta,
        baseline: &[TranscriptMessage],
    ) -> anyhow::Result<(SessionRef, Arc<dyn TranscriptHistory>)> {
        self.begin_generation_checked(session, seed, Some(normalized(baseline.to_vec())))
    }
}

impl DriverTranscriptLocator {
    fn begin_generation_checked(
        &self,
        session: &SessionRef,
        seed: TranscriptMeta,
        baseline: Option<Vec<TranscriptMessage>>,
    ) -> anyhow::Result<(SessionRef, Arc<dyn TranscriptHistory>)> {
        let successor = session.next_generation();
        anyhow::ensure!(
            successor.generation <= MAX_GENERATIONS,
            "session generation limit reached"
        );
        let stem = session_stem(&successor);
        let reservation = {
            let stem = stem.clone();
            let docs = Arc::clone(&self.docs);
            let tables = Arc::clone(&self.tables);
            let declared = Arc::clone(&self.declared);
            run_on(&self.bridge, async move {
                declared.ensure(&docs).await?;
                Ok(reservation::reserve(&docs, &tables.leases, &stem).await)
            })??
        };
        let predecessor_stem = session_stem(session);
        if let Err(error) = self.handle(&predecessor_stem, seed.clone()).seal(baseline) {
            self.release(&stem, &reservation);
            return Err(error);
        }
        let mut meta = seed;
        meta.session_id = Some(successor.session_id());
        meta.parent_session_id = successor.parent_session_id();
        let handle = self.reserved_handle(&stem, meta, Some(reservation));
        Ok((successor, Arc::new(handle)))
    }

    /// Drops the reservation of `stem` this call made, at the epoch it
    /// took: if another process has since taken it over, the reservation is
    /// theirs and stays.
    fn release(&self, stem: &str, held: &Reservation) {
        let tables = Arc::clone(&self.tables);
        let (stem_owned, held_owned) = (stem.to_owned(), held.clone());
        let released = run_on(&self.bridge, async move {
            reservation::release(&tables.leases, &stem_owned, &held_owned).await
        });
        if let Err(error) = released {
            tracing::warn!(
                target: "tinyagents_session::port::drivers",
                stem,
                %error,
                "[session-store] could not release a generation reservation"
            );
        }
    }
}

#[path = "transcripts_adoption.rs"]
mod adoption;
#[path = "transcripts_display.rs"]
mod display;
#[path = "transcripts_reservation.rs"]
mod reservation;

#[cfg(test)]
#[path = "transcripts_tests.rs"]
pub(super) mod tests;
