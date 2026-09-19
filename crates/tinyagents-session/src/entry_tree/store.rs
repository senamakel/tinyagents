//! SQLite plumbing for the entry tree: row (de)serialization, ordinal
//! allocation, ancestor-chain walks, and the `branch_entries` index.

use rusqlite::{Connection, OptionalExtension, params};

use tinyagents_harness::error::Result;

use super::types::{Branch, Entry, EntryId, EntryKind};
use crate::context::StorageContext;

/// Serializes an [`EntryKind`] to the `(kind, payload_json)` columns.
fn encode_kind(kind: &EntryKind) -> Result<(&'static str, String)> {
    let tag = match kind {
        EntryKind::Message(_) => "message",
        EntryKind::Compaction(_) => "compaction",
        EntryKind::BranchSummary(_) => "branch_summary",
        EntryKind::Label(_) => "label",
        EntryKind::Custom(_) => "custom",
    };
    let payload = serde_json::to_string(kind).storage_context("failed to encode entry kind")?;
    Ok((tag, payload))
}

fn decode_kind(payload_json: &str) -> Result<EntryKind> {
    serde_json::from_str(payload_json).storage_context("failed to decode entry kind")
}

fn map_entry_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<(String, Option<String>, u64, String, String)> {
    Ok((
        row.get(0)?,
        row.get(1)?,
        row.get::<_, i64>(2)? as u64,
        row.get(3)?,
        row.get(4)?,
    ))
}

fn row_to_entry(id: String, parent_id: Option<String>, ordinal: u64, payload_json: String, ts: String) -> Result<Entry> {
    Ok(Entry {
        id: EntryId(id),
        parent_id: parent_id.map(EntryId),
        ordinal,
        kind: decode_kind(&payload_json)?,
        ts,
    })
}

/// Allocates the next ordinal for `session_id` (0-based, monotonic).
///
/// Must be called inside a write transaction ([`super::super::store::with_transaction`])
/// to avoid two racing appends allocating the same ordinal.
pub(super) fn next_ordinal(conn: &Connection, session_id: &str) -> Result<u64> {
    let max: Option<i64> = conn
        .query_row(
            "SELECT MAX(ordinal) FROM entry_tree_entries WHERE session_id = ?1",
            params![session_id],
            |row| row.get(0),
        )
        .storage_context("failed to read max entry ordinal")?;
    Ok(max.map(|m| m as u64 + 1).unwrap_or(0))
}

/// Inserts one entry row. Callers choose the id (either a freshly derived
/// `EntryId::derive`, or a legacy-preserved one) and the ordinal.
pub(super) fn insert_entry(
    conn: &Connection,
    session_id: &str,
    entry: &Entry,
) -> Result<()> {
    let (tag, payload) = encode_kind(&entry.kind)?;
    conn.execute(
        "INSERT INTO entry_tree_entries (session_id, id, parent_id, ordinal, kind, payload_json, ts)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            session_id,
            entry.id.as_str(),
            entry.parent_id.as_ref().map(EntryId::as_str),
            entry.ordinal as i64,
            tag,
            payload,
            entry.ts,
        ],
    )
    .storage_context("failed to insert entry")?;
    Ok(())
}

pub(super) fn get_entry(conn: &Connection, session_id: &str, id: &EntryId) -> Result<Option<Entry>> {
    let row = conn
        .query_row(
            "SELECT id, parent_id, ordinal, payload_json, ts
             FROM entry_tree_entries WHERE session_id = ?1 AND id = ?2",
            params![session_id, id.as_str()],
            map_entry_row,
        )
        .optional()
        .storage_context("failed to read entry")?;
    row.map(|(id, parent_id, ordinal, payload_json, ts)| {
        row_to_entry(id, parent_id, ordinal, payload_json, ts)
    })
    .transpose()
}

/// Walks `parent_id` from `tip` to the session root, returning entries in
/// **chronological** (root-first) order.
pub(super) fn ancestor_chain(conn: &Connection, session_id: &str, tip: &EntryId) -> Result<Vec<Entry>> {
    let mut chain = Vec::new();
    let mut current = Some(tip.clone());
    while let Some(id) = current {
        let entry = get_entry(conn, session_id, &id)?
            .storage_context(&format!("entry tree: dangling reference to entry {id}"))?;
        current = entry.parent_id.clone();
        chain.push(entry);
    }
    chain.reverse();
    Ok(chain)
}

/// Entries in a session with no children — the current tips of the tree.
pub(super) fn leaf_entries(conn: &Connection, session_id: &str) -> Result<Vec<EntryId>> {
    let mut stmt = conn
        .prepare(
            "SELECT id FROM entry_tree_entries e
             WHERE e.session_id = ?1
               AND NOT EXISTS (
                 SELECT 1 FROM entry_tree_entries c
                 WHERE c.session_id = e.session_id AND c.parent_id = e.id
               )
             ORDER BY e.ordinal",
        )
        .storage_context("failed to prepare leaf query")?;
    let rows = stmt
        .query_map(params![session_id], |row| row.get::<_, String>(0))
        .storage_context("failed to query leaves")?;
    let mut out = Vec::new();
    for row in rows {
        out.push(EntryId(row.storage_context("failed to read leaf row")?));
    }
    Ok(out)
}

/// The entry with the greatest ordinal in the session — the default parent
/// for a plain (non-fork) append, i.e. "the current tip" for linear use.
pub(super) fn head(conn: &Connection, session_id: &str) -> Result<Option<EntryId>> {
    conn.query_row(
        "SELECT id FROM entry_tree_entries WHERE session_id = ?1 ORDER BY ordinal DESC LIMIT 1",
        params![session_id],
        |row| row.get::<_, String>(0),
    )
    .optional()
    .storage_context("failed to read entry tree head")
    .map(|opt| opt.map(EntryId))
}

pub(super) fn insert_label(conn: &Connection, session_id: &str, name: &str, entry_id: &EntryId) -> Result<()> {
    conn.execute(
        "INSERT INTO entry_tree_labels (session_id, name, entry_id) VALUES (?1, ?2, ?3)
         ON CONFLICT(session_id, name) DO UPDATE SET entry_id = excluded.entry_id",
        params![session_id, name, entry_id.as_str()],
    )
    .storage_context("failed to insert label")?;
    Ok(())
}

pub(super) fn list_branches(conn: &Connection, session_id: &str) -> Result<Vec<Branch>> {
    let mut stmt = conn
        .prepare("SELECT name, entry_id FROM entry_tree_labels WHERE session_id = ?1 ORDER BY name")
        .storage_context("failed to prepare branch query")?;
    let rows = stmt
        .query_map(params![session_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .storage_context("failed to query branches")?;
    let mut out = Vec::new();
    for row in rows {
        let (name, tip_id) = row.storage_context("failed to read branch row")?;
        out.push(Branch {
            name,
            tip_id: EntryId(tip_id),
        });
    }
    Ok(out)
}

/// Replaces the `branch_entries` materialization for `tip` with the current
/// ancestor chain, computed by walking `parent_id`.
pub(super) fn reindex_tip(conn: &Connection, session_id: &str, tip: &EntryId) -> Result<()> {
    let chain = ancestor_chain(conn, session_id, tip)?;
    conn.execute(
        "DELETE FROM branch_entries WHERE session_id = ?1 AND tip_id = ?2",
        params![session_id, tip.as_str()],
    )
    .storage_context("failed to clear branch_entries for tip")?;
    for entry in &chain {
        conn.execute(
            "INSERT INTO branch_entries (session_id, tip_id, entry_id, ordinal)
             VALUES (?1, ?2, ?3, ?4)",
            params![session_id, tip.as_str(), entry.id.as_str(), entry.ordinal as i64],
        )
        .storage_context("failed to insert branch_entries row")?;
    }
    Ok(())
}

/// Rebuilds `branch_entries` for every known tip (every leaf plus every
/// labeled entry) from scratch.
pub(super) fn rebuild_index(conn: &Connection, session_id: &str) -> Result<()> {
    conn.execute(
        "DELETE FROM branch_entries WHERE session_id = ?1",
        params![session_id],
    )
    .storage_context("failed to clear branch_entries")?;
    let mut tips = leaf_entries(conn, session_id)?;
    for branch in list_branches(conn, session_id)? {
        if !tips.contains(&branch.tip_id) {
            tips.push(branch.tip_id);
        }
    }
    for tip in &tips {
        reindex_tip(conn, session_id, tip)?;
    }
    Ok(())
}

/// Reads the materialized ancestor chain for `tip` from `branch_entries`, in
/// chronological order. Returns `None` if the tip has no index rows (not yet
/// built, or stale) so the caller can fall back to [`ancestor_chain`].
pub(super) fn indexed_chain(conn: &Connection, session_id: &str, tip: &EntryId) -> Result<Option<Vec<Entry>>> {
    let mut stmt = conn
        .prepare(
            "SELECT e.id, e.parent_id, e.ordinal, e.payload_json, e.ts
             FROM branch_entries b
             JOIN entry_tree_entries e
               ON e.session_id = b.session_id AND e.id = b.entry_id
             WHERE b.session_id = ?1 AND b.tip_id = ?2
             ORDER BY b.ordinal",
        )
        .storage_context("failed to prepare indexed chain query")?;
    let rows = stmt
        .query_map(params![session_id, tip.as_str()], map_entry_row)
        .storage_context("failed to query indexed chain")?;
    let mut out = Vec::new();
    for row in rows {
        let (id, parent_id, ordinal, payload_json, ts) =
            row.storage_context("failed to read indexed chain row")?;
        out.push(row_to_entry(id, parent_id, ordinal, payload_json, ts)?);
    }
    if out.is_empty() {
        Ok(None)
    } else {
        Ok(Some(out))
    }
}
