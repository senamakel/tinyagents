//! CRUD for the per-thread todo list, on the harness
//! [`Store`](tinyagents_harness::store::Store).
//!
//! Each thread's list is a single serialized [`TodoList`] value under the
//! [`TODOS_NAMESPACE`] namespace, keyed by the hex-encoded thread id. Every
//! mutation runs `load → mutate → normalise → put` under a **per-thread async
//! mutex** ([`thread_lock`]) so the read-modify-write is atomic within the
//! process (the same single-process caveat as
//! [`graph::goals::store`](crate::goals::store)).
//!
//! The list is rewritten wholesale ([`replace`]) — there is no per-item CRUD,
//! because the model that owns it always writes the complete list. Each
//! mutator returns a [`TodosSnapshot`] — the normalised items plus a markdown
//! rendering — so an agent transcript and a UI stay in lock-step.

use std::sync::{Arc, OnceLock};

use tokio::sync::Mutex;

use super::types::{
    TodoItem, TodoList, TodoStatus, TodosSnapshot, normalise_list, now_stamp, render_markdown,
};
use crate::thread_locks::ThreadLockMap;
use tinyagents_harness::error::{Result, TinyAgentsError};
use tinyagents_harness::store::Store;

/// The [`Store`] namespace holding one [`TodoList`] per thread.
pub const TODOS_NAMESPACE: &str = "graph.todos";

/// Serialises `load → mutate → put` per thread so a read-modify-write is atomic
/// within the process. Unused mutexes are reclaimed (see
/// [`ThreadLockMap`](crate::thread_locks::ThreadLockMap)) so the map
/// does not grow with every thread id ever seen.
fn thread_lock(thread_id: &str) -> Arc<Mutex<()>> {
    static LOCKS: OnceLock<ThreadLockMap> = OnceLock::new();
    LOCKS
        .get_or_init(|| ThreadLockMap::new("todo lock map"))
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
            "todo list thread_id must not be empty or whitespace".to_string(),
        ));
    }
    Ok(trimmed.to_string())
}

/// Loads the raw items for `thread_id` (empty when the thread has no list).
async fn load_items(store: &Arc<dyn Store>, thread_id: &str) -> Result<Vec<TodoItem>> {
    Ok(get(store, thread_id)
        .await?
        .map(|list| list.items)
        .unwrap_or_default())
}

/// Load a list without normalising it, preserving the distinction between an
/// absent list and a present empty list.
pub async fn get(store: &Arc<dyn Store>, thread_id: &str) -> Result<Option<TodoList>> {
    let thread_id = validate_thread_id(thread_id)?;
    match store.get(TODOS_NAMESPACE, &key(&thread_id)).await? {
        Some(value) => Ok(Some(serde_json::from_value(value)?)),
        None => Ok(None),
    }
}

/// Delete a list value outright, returning whether one was present.
///
/// This differs from [`clear`], which persists a present, empty list.
pub async fn delete(store: &Arc<dyn Store>, thread_id: &str) -> Result<bool> {
    let thread_id = validate_thread_id(thread_id)?;
    let lock = thread_lock(&thread_id);
    let _guard = lock.lock().await;
    let list_key = key(&thread_id);
    let existed = store.get(TODOS_NAMESPACE, &list_key).await?.is_some();
    if existed {
        store.delete(TODOS_NAMESPACE, &list_key).await?;
    }
    Ok(existed)
}

/// Normalises and persists `items` for `thread_id`, returning the normalised set.
async fn save_items(
    store: &Arc<dyn Store>,
    thread_id: &str,
    items: Vec<TodoItem>,
) -> Result<Vec<TodoItem>> {
    let mut list = TodoList {
        thread_id: thread_id.to_string(),
        items,
        updated_at: now_stamp(),
    };
    normalise_list(&mut list);
    let value = serde_json::to_value(&list)?;
    store.put(TODOS_NAMESPACE, &key(thread_id), value).await?;
    Ok(list.items)
}

fn snapshot(thread_id: &str, items: Vec<TodoItem>) -> TodosSnapshot {
    let markdown = render_markdown(&items);
    TodosSnapshot {
        thread_id: thread_id.to_string(),
        items,
        markdown,
    }
}

/// At most one item may be `InProgress` at a time. Returns a
/// [`Validation`](TinyAgentsError::Validation) error otherwise (never silently
/// fixes it), so the model is told to narrow its focus rather than having the
/// list quietly rewritten under it.
fn enforce_single_in_progress(items: &[TodoItem]) -> Result<()> {
    let in_progress = items
        .iter()
        .filter(|item| matches!(item.status, TodoStatus::InProgress))
        .count();
    if in_progress > 1 {
        return Err(TinyAgentsError::Validation(format!(
            "only one todo may be `in_progress` at a time (got {in_progress})"
        )));
    }
    Ok(())
}

/// Snapshot the current list without mutating.
pub async fn list(store: &Arc<dyn Store>, thread_id: &str) -> Result<TodosSnapshot> {
    let thread_id = validate_thread_id(thread_id)?;
    let lock = thread_lock(&thread_id);
    let _guard = lock.lock().await;
    let items = load_items(store, &thread_id).await?;
    Ok(snapshot(&thread_id, items))
}

/// Wholesale-replace the thread's list. Blank items are dropped on normalise.
pub async fn replace(
    store: &Arc<dyn Store>,
    thread_id: &str,
    items: Vec<TodoItem>,
) -> Result<TodosSnapshot> {
    let thread_id = validate_thread_id(thread_id)?;
    let lock = thread_lock(&thread_id);
    let _guard = lock.lock().await;
    enforce_single_in_progress(&items)?;
    let items = save_items(store, &thread_id, items).await?;
    Ok(snapshot(&thread_id, items))
}

/// Empty the list.
pub async fn clear(store: &Arc<dyn Store>, thread_id: &str) -> Result<TodosSnapshot> {
    let thread_id = validate_thread_id(thread_id)?;
    let lock = thread_lock(&thread_id);
    let _guard = lock.lock().await;
    let items = save_items(store, &thread_id, Vec::new()).await?;
    Ok(snapshot(&thread_id, items))
}
