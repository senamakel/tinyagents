//! SQLite-backed [`TaskCache`] behind the optional `sqlite` cargo feature.
//!
//! One row per [`TaskCacheKey`] in a `task_cache` table; `put` upserts, and
//! `get` reports a row whose `expires_at` (unix millis) has passed as a miss
//! and deletes it. The connection is opened with the same pragmas as
//! [`crate::SqliteCheckpointer`] (WAL, `synchronous = NORMAL`, busy
//! timeout) via the shared `prepare_connection` helper, and every call runs
//! on `spawn_blocking` so the executor is never blocked on disk I/O.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use rusqlite::{Connection, OptionalExtension, params};

use super::{TaskCache, TaskCacheKey};
use crate::checkpoint::prepare_connection;
use crate::{Result, TinyAgentsError};
use tinyagents_harness::ids::GraphId;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS task_cache (
    graph_id   TEXT NOT NULL,
    node_id    TEXT NOT NULL,
    hash       TEXT NOT NULL,
    value      TEXT NOT NULL,
    expires_at INTEGER,
    PRIMARY KEY (graph_id, node_id, hash)
);
";

/// A [`TaskCache`] persisted in a SQLite database. Cheap to clone; clones
/// share one connection (and so the same data, including for `:memory:`).
#[derive(Clone)]
pub struct SqliteTaskCache {
    conn: Arc<Mutex<Connection>>,
}

fn sqlite_err(context: &str, err: impl std::fmt::Display) -> TinyAgentsError {
    TinyAgentsError::Graph(format!("sqlite task cache: {context}: {err}"))
}

impl SqliteTaskCache {
    /// Opens (creating if needed) a cache database at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path.as_ref()).map_err(|e| sqlite_err("open database", e))?;
        Self::from_connection(conn)
    }

    /// Opens an ephemeral in-memory cache.
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(|e| sqlite_err("open in-memory", e))?;
        Self::from_connection(conn)
    }

    /// Wraps a caller-owned [`Connection`], applying the shared pragmas and
    /// ensuring the (idempotent) schema exists.
    pub fn from_connection(conn: Connection) -> Result<Self> {
        prepare_connection(&conn)?;
        conn.execute_batch(SCHEMA)
            .map_err(|e| sqlite_err("create schema", e))?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Runs `f` against the locked connection on the blocking pool.
    async fn with_conn<T, F>(&self, context: &'static str, f: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&Connection) -> Result<T> + Send + 'static,
    {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|_| sqlite_err(context, "connection lock poisoned"))?;
            f(&conn)
        })
        .await
        .map_err(|e| sqlite_err(context, e))?
    }
}

fn now_ms() -> i64 {
    tinyagents_harness::ids::now_ms() as i64
}

#[async_trait]
impl TaskCache for SqliteTaskCache {
    async fn get(&self, key: &TaskCacheKey) -> Result<Option<serde_json::Value>> {
        let key = key.clone();
        self.with_conn("get", move |conn| {
            let row: Option<(String, Option<i64>)> = conn
                .query_row(
                    "SELECT value, expires_at FROM task_cache
                     WHERE graph_id = ?1 AND node_id = ?2 AND hash = ?3",
                    params![key.graph_id.as_str(), key.node_id.as_str(), key.hash],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()
                .map_err(|e| sqlite_err("read task_cache", e))?;
            let Some((value, expires_at)) = row else {
                return Ok(None);
            };
            if expires_at.is_some_and(|at| at <= now_ms()) {
                conn.execute(
                    "DELETE FROM task_cache WHERE graph_id = ?1 AND node_id = ?2 AND hash = ?3",
                    params![key.graph_id.as_str(), key.node_id.as_str(), key.hash],
                )
                .map_err(|e| sqlite_err("delete expired task_cache row", e))?;
                return Ok(None);
            }
            serde_json::from_str(&value)
                .map(Some)
                .map_err(|e| sqlite_err("decode cached value", e))
        })
        .await
    }

    async fn put(
        &self,
        key: &TaskCacheKey,
        value: serde_json::Value,
        ttl: Option<Duration>,
    ) -> Result<()> {
        let key = key.clone();
        let expires_at = ttl.map(|ttl| now_ms().saturating_add(ttl.as_millis() as i64));
        self.with_conn("put", move |conn| {
            let encoded =
                serde_json::to_string(&value).map_err(|e| sqlite_err("encode cached value", e))?;
            conn.execute(
                "INSERT INTO task_cache (graph_id, node_id, hash, value, expires_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(graph_id, node_id, hash)
                 DO UPDATE SET value = excluded.value, expires_at = excluded.expires_at",
                params![
                    key.graph_id.as_str(),
                    key.node_id.as_str(),
                    key.hash,
                    encoded,
                    expires_at
                ],
            )
            .map_err(|e| sqlite_err("upsert task_cache", e))?;
            Ok(())
        })
        .await
    }

    async fn clear(&self, graph_id: &GraphId) -> Result<()> {
        let graph_id = graph_id.as_str().to_string();
        self.with_conn("clear", move |conn| {
            conn.execute(
                "DELETE FROM task_cache WHERE graph_id = ?1",
                params![graph_id],
            )
            .map_err(|e| sqlite_err("clear task_cache", e))?;
            Ok(())
        })
        .await
    }
}
