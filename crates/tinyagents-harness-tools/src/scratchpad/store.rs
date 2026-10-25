//! Persistence for the per-thread scratchpad, on the harness
//! `Store`.
//!
//! One serialized [`Scratchpad`] per thread under [`SCRATCHPAD_NAMESPACE`],
//! keyed by the hex-encoded thread id, written `load → apply → put` under a
//! per-thread async mutex (the same single-process caveat as
//! [`crate::todos::store`]). Lists are capped and de-duplicated so the
//! scratchpad stays a short working view, not a log.

use std::sync::{Arc, OnceLock};

use tokio::sync::Mutex;

use super::types::{ClosedPlan, Scratchpad, ScratchpadUpdate, TaskNotes};
use crate::thread_locks::ThreadLockMap;
use crate::todos::store as todo_store;
use tinyagents_harness::error::{Result, TinyAgentsError};
use tinyagents_harness::store::Store;

/// The [`Store`] namespace holding one [`Scratchpad`] per thread.
pub const SCRATCHPAD_NAMESPACE: &str = "graph.scratchpad";
/// Facts kept; the oldest go first.
pub const MAX_FACTS: usize = 12;
/// Blockers kept; the oldest go first.
pub const MAX_BLOCKERS: usize = 8;
/// Longest note kept, in characters.
pub const MAX_NOTE_CHARS: usize = 300;

fn thread_lock(thread_id: &str) -> Arc<Mutex<()>> {
    static LOCKS: OnceLock<ThreadLockMap> = OnceLock::new();
    LOCKS
        .get_or_init(|| ThreadLockMap::new("scratchpad lock map"))
        .lock_for(thread_id)
}

/// Hex-encodes the thread id into a [`Store`]-safe key.
fn key(thread_id: &str) -> String {
    thread_id
        .as_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn validate_thread_id(thread_id: &str) -> Result<String> {
    let trimmed = thread_id.trim();
    if trimmed.is_empty() {
        return Err(TinyAgentsError::Validation(
            "scratchpad thread_id must not be empty or whitespace".to_string(),
        ));
    }
    Ok(trimmed.to_string())
}

/// The thread's scratchpad, or `None` when nothing was ever noted.
pub async fn get(store: &Arc<dyn Store>, thread_id: &str) -> Result<Option<Scratchpad>> {
    let thread_id = validate_thread_id(thread_id)?;
    match store.get(SCRATCHPAD_NAMESPACE, &key(&thread_id)).await? {
        Some(value) => Ok(Some(serde_json::from_value(value)?)),
        None => Ok(None),
    }
}

/// Applies `update` to the thread's scratchpad and returns the result.
pub async fn apply(
    store: &Arc<dyn Store>,
    thread_id: &str,
    update: &ScratchpadUpdate,
) -> Result<Scratchpad> {
    let thread_id = validate_thread_id(thread_id)?;
    let lock = thread_lock(&thread_id);
    let _guard = lock.lock().await;
    let mut pad = get(store, &thread_id).await?.unwrap_or_default();
    pad.thread_id = thread_id.clone();
    merge(&mut pad, update);
    pad.updated_at = crate::todos::now_stamp();
    store
        .put(
            SCRATCHPAD_NAMESPACE,
            &key(&thread_id),
            serde_json::to_value(&pad)?,
        )
        .await?;
    Ok(pad)
}

/// Ends the turn's bookkeeping for a plan that finished: when the thread's
/// plan has steps and none open, and the model noted facts or blockers, those
/// move to [`Scratchpad::last_task`] (replacing the previous task's), the
/// working notes and approach are cleared, and the plan with its notes comes
/// back for the host to keep. Otherwise nothing changes and `None` comes back:
/// an open plan's notes carry into the next turn.
///
/// Call it once the turn's final reply is committed. A second call finds the
/// working notes empty and returns `None`, so a retried hook does not hand the
/// same notes over twice.
pub async fn close_finished_plan(
    store: &Arc<dyn Store>,
    thread_id: &str,
) -> Result<Option<ClosedPlan>> {
    let thread_id = validate_thread_id(thread_id)?;
    let lock = thread_lock(&thread_id);
    let _guard = lock.lock().await;
    let items = todo_store::get(store, &thread_id)
        .await?
        .map(|list| list.items)
        .unwrap_or_default();
    if items.is_empty() || items.iter().any(|item| item.status.is_open()) {
        return Ok(None);
    }
    let Some(mut pad) = get(store, &thread_id).await? else {
        return Ok(None);
    };
    let notes = TaskNotes {
        facts: std::mem::take(&mut pad.facts),
        blockers: std::mem::take(&mut pad.blockers),
    };
    if notes.is_empty() {
        return Ok(None);
    }
    pad.approach = None;
    pad.last_task = Some(notes.clone());
    pad.updated_at = crate::todos::now_stamp();
    store
        .put(
            SCRATCHPAD_NAMESPACE,
            &key(&thread_id),
            serde_json::to_value(&pad)?,
        )
        .await?;
    tracing::debug!(
        facts = notes.facts.len(),
        blockers = notes.blockers.len(),
        "[tinyagents::scratchpad] finished plan: notes moved to last_task"
    );
    Ok(Some(ClosedPlan {
        thread_id,
        items,
        notes,
    }))
}

/// Deletes the thread's notes, returning whether there were any.
pub async fn clear(store: &Arc<dyn Store>, thread_id: &str) -> Result<bool> {
    let thread_id = validate_thread_id(thread_id)?;
    let lock = thread_lock(&thread_id);
    let _guard = lock.lock().await;
    let pad_key = key(&thread_id);
    let existed = store.get(SCRATCHPAD_NAMESPACE, &pad_key).await?.is_some();
    if existed {
        store.delete(SCRATCHPAD_NAMESPACE, &pad_key).await?;
    }
    Ok(existed)
}

/// The pure merge: retractions first, then additions, then caps.
pub(super) fn merge(pad: &mut Scratchpad, update: &ScratchpadUpdate) {
    let needles: Vec<String> = update
        .retract
        .iter()
        .map(|needle| needle.trim().to_lowercase())
        .filter(|needle| !needle.is_empty())
        .collect();
    let retracted = |text: &str| {
        let text = text.to_lowercase();
        needles.iter().any(|needle| text.contains(needle))
    };
    pad.facts.retain(|fact| !retracted(fact));
    pad.blockers.retain(|blocker| !retracted(blocker));
    if let Some(last) = &mut pad.last_task {
        last.facts.retain(|fact| !retracted(fact));
        last.blockers.retain(|blocker| !retracted(blocker));
        if last.is_empty() {
            pad.last_task = None;
        }
    }
    if pad.approach.as_deref().is_some_and(retracted) {
        pad.approach = None;
    }
    add(&mut pad.facts, &update.facts, MAX_FACTS);
    add(&mut pad.blockers, &update.blockers, MAX_BLOCKERS);
    if let Some(approach) = &update.approach {
        let approach = clip(approach);
        pad.approach = (!approach.is_empty()).then_some(approach);
    }
}

/// Appends the new notes that are not already there, then keeps the newest `max`.
fn add(notes: &mut Vec<String>, new: &[String], max: usize) {
    for note in new
        .iter()
        .map(|note| clip(note))
        .filter(|note| !note.is_empty())
    {
        notes.retain(|known| !known.eq_ignore_ascii_case(&note));
        notes.push(note);
    }
    if notes.len() > max {
        notes.drain(..notes.len() - max);
    }
}

/// One line, at most [`MAX_NOTE_CHARS`] characters.
fn clip(note: &str) -> String {
    let line = note.split_whitespace().collect::<Vec<_>>().join(" ");
    if line.chars().count() <= MAX_NOTE_CHARS {
        return line;
    }
    let mut cut: String = line.chars().take(MAX_NOTE_CHARS - 1).collect();
    cut.push('…');
    cut
}
