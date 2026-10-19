//! Classic-database fixtures for the importer tests.
//!
//! The classic `sessions.db` schema is no longer created by this crate; it is
//! reproduced here, verbatim, so tests can build databases exactly as the
//! classic store did at any schema version and populate them with realistic
//! rows. The list is append-only history: index == schema version.

use tinystoragedrivers_sqlite::rusqlite::{Connection, params};

pub(crate) const CLASSIC_MIGRATIONS: &[&str] = &[
    // ---- 0: base session tables ------------------------------------------
    "CREATE TABLE IF NOT EXISTS sessions (
        id                    TEXT PRIMARY KEY,
        agent_definition_id   TEXT NOT NULL,
        agent_definition_name TEXT NOT NULL,
        session_key           TEXT NOT NULL,
        parent_session_id     TEXT,
        thread_id             TEXT,
        source_channel        TEXT,
        status                TEXT NOT NULL DEFAULT 'running',
        model                 TEXT,
        turn_count            INTEGER NOT NULL DEFAULT 0,
        input_tokens          INTEGER NOT NULL DEFAULT 0,
        output_tokens         INTEGER NOT NULL DEFAULT 0,
        cached_input_tokens   INTEGER NOT NULL DEFAULT 0,
        cost_usd              REAL NOT NULL DEFAULT 0.0,
        transcript_path       TEXT,
        started_at            TEXT NOT NULL,
        ended_at              TEXT,
        FOREIGN KEY (parent_session_id) REFERENCES sessions(id) ON DELETE SET NULL
     );
     CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions(agent_definition_id);
     CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
     CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);
     CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
     CREATE INDEX IF NOT EXISTS idx_sessions_thread ON sessions(thread_id);
     CREATE INDEX IF NOT EXISTS idx_sessions_channel ON sessions(source_channel);
     CREATE INDEX IF NOT EXISTS idx_sessions_key ON sessions(session_key);

     CREATE TABLE IF NOT EXISTS session_messages (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  TEXT NOT NULL,
        role        TEXT NOT NULL,
        content     TEXT NOT NULL,
        model       TEXT,
        input_tokens  INTEGER,
        output_tokens INTEGER,
        cost_usd    REAL,
        created_at  TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
     );
     CREATE INDEX IF NOT EXISTS idx_messages_session ON session_messages(session_id);

     CREATE TABLE IF NOT EXISTS session_tool_calls (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  TEXT NOT NULL,
        message_id  INTEGER,
        tool_name   TEXT NOT NULL,
        tool_input  TEXT,
        tool_output TEXT,
        status      TEXT NOT NULL DEFAULT 'pending',
        duration_ms INTEGER,
        created_at  TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
        FOREIGN KEY (message_id) REFERENCES session_messages(id) ON DELETE SET NULL
     );
     CREATE INDEX IF NOT EXISTS idx_tool_calls_session ON session_tool_calls(session_id);
     CREATE INDEX IF NOT EXISTS idx_tool_calls_name ON session_tool_calls(tool_name);",
    // ---- 1: full-text search index ---------------------------------------
    "CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
        session_id,
        agent_definition_name,
        content,
        tool_name
     );",
    // ---- 2: run ledger ---------------------------------------------------
    // The original DDL also carried `task_board_id` / `task_card_id` columns
    // for the retired per-thread task board. They are gone from this
    // statement rather than dropped by a later migration: a database created
    // before their removal keeps the two nullable columns, unread and
    // unwritten, which is harmless, and a fresh database never grows them.
    "CREATE TABLE IF NOT EXISTS agent_runs (
        id                 TEXT PRIMARY KEY,
        kind               TEXT NOT NULL,
        parent_run_id      TEXT,
        parent_thread_id   TEXT,
        agent_id           TEXT,
        status             TEXT NOT NULL,
        prompt_ref         TEXT,
        worker_thread_id   TEXT,
        checkpoint_path    TEXT,
        checkpoint_json    TEXT,
        summary            TEXT,
        error              TEXT,
        metadata_json      TEXT NOT NULL DEFAULT '{}',
        started_at         TEXT NOT NULL,
        updated_at         TEXT NOT NULL,
        completed_at       TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_agent_runs_status ON agent_runs(status);
    CREATE INDEX IF NOT EXISTS idx_agent_runs_kind ON agent_runs(kind);
    CREATE INDEX IF NOT EXISTS idx_agent_runs_parent ON agent_runs(parent_run_id);
    CREATE INDEX IF NOT EXISTS idx_agent_runs_thread ON agent_runs(parent_thread_id);
    CREATE INDEX IF NOT EXISTS idx_agent_runs_updated ON agent_runs(updated_at);
    CREATE INDEX IF NOT EXISTS idx_agent_runs_worker_thread ON agent_runs(worker_thread_id);

    CREATE TABLE IF NOT EXISTS workflow_runs (
        id                 TEXT PRIMARY KEY,
        definition_id      TEXT NOT NULL,
        parent_thread_id   TEXT,
        input_json         TEXT NOT NULL DEFAULT '{}',
        phase_states_json  TEXT NOT NULL DEFAULT '{}',
        child_run_ids_json TEXT NOT NULL DEFAULT '[]',
        status             TEXT NOT NULL,
        summary            TEXT,
        started_at         TEXT NOT NULL,
        updated_at         TEXT NOT NULL,
        completed_at       TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_workflow_runs_definition ON workflow_runs(definition_id);
    CREATE INDEX IF NOT EXISTS idx_workflow_runs_status ON workflow_runs(status);
    CREATE INDEX IF NOT EXISTS idx_workflow_runs_thread ON workflow_runs(parent_thread_id);

    CREATE TABLE IF NOT EXISTS run_events (
        run_id      TEXT NOT NULL,
        sequence    INTEGER NOT NULL,
        event_type  TEXT NOT NULL,
        payload_json TEXT NOT NULL DEFAULT '{}',
        timestamp   TEXT NOT NULL,
        PRIMARY KEY (run_id, sequence)
    );
    CREATE INDEX IF NOT EXISTS idx_run_events_timestamp ON run_events(timestamp);

    CREATE TABLE IF NOT EXISTS run_telemetry (
        run_id              TEXT PRIMARY KEY,
        input_tokens        INTEGER NOT NULL DEFAULT 0,
        output_tokens       INTEGER NOT NULL DEFAULT 0,
        cached_input_tokens INTEGER NOT NULL DEFAULT 0,
        cost_usd            REAL NOT NULL DEFAULT 0.0,
        elapsed_ms          INTEGER,
        tool_count          INTEGER NOT NULL DEFAULT 0,
        model               TEXT,
        provider            TEXT,
        error               TEXT,
        updated_at          TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS agent_teams (
        id                 TEXT PRIMARY KEY,
        parent_thread_id   TEXT,
        lead_agent_id      TEXT NOT NULL,
        status             TEXT NOT NULL,
        summary            TEXT,
        created_at         TEXT NOT NULL,
        updated_at         TEXT NOT NULL,
        closed_at          TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_agent_teams_thread ON agent_teams(parent_thread_id);
    CREATE INDEX IF NOT EXISTS idx_agent_teams_status ON agent_teams(status);

    CREATE TABLE IF NOT EXISTS agent_team_members (
        id                 TEXT PRIMARY KEY,
        team_id            TEXT NOT NULL,
        name               TEXT NOT NULL,
        agent_id           TEXT,
        member_status      TEXT NOT NULL,
        current_task_id    TEXT,
        worker_thread_id   TEXT,
        run_id             TEXT,
        created_at         TEXT NOT NULL,
        updated_at         TEXT NOT NULL,
        UNIQUE(team_id, name)
    );
    CREATE INDEX IF NOT EXISTS idx_agent_team_members_team ON agent_team_members(team_id);

    CREATE TABLE IF NOT EXISTS agent_team_tasks (
        id                  TEXT PRIMARY KEY,
        team_id             TEXT NOT NULL,
        title               TEXT NOT NULL,
        objective           TEXT,
        status              TEXT NOT NULL,
        owner_member_id     TEXT,
        claimed_by_member_id TEXT,
        claim_token         TEXT,
        depends_on_json     TEXT NOT NULL DEFAULT '[]',
        gate_status         TEXT NOT NULL DEFAULT 'pending',
        gate_reason         TEXT,
        evidence_json       TEXT NOT NULL DEFAULT '[]',
        source_run_id       TEXT,
        order_index         INTEGER NOT NULL DEFAULT 0,
        created_at          TEXT NOT NULL,
        updated_at          TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_agent_team_tasks_team ON agent_team_tasks(team_id);
    CREATE INDEX IF NOT EXISTS idx_agent_team_tasks_status ON agent_team_tasks(status);
    CREATE INDEX IF NOT EXISTS idx_agent_team_tasks_claimed ON agent_team_tasks(claimed_by_member_id);",
    // ---- 3: index coverage for the ORDER BY / sort columns ---------------
    //
    // `list_workflow_runs` and `list_agent_teams` both order by `updated_at
    // DESC` and `list_agent_team_tasks` by `(order_index, created_at)`, none of
    // which had an index — every listing was a full scan plus a sort. The
    // `agent_runs` table already had its `updated_at` index; these bring the
    // rest up to parity.
    "CREATE INDEX IF NOT EXISTS idx_workflow_runs_updated ON workflow_runs(updated_at);
     CREATE INDEX IF NOT EXISTS idx_agent_teams_updated ON agent_teams(updated_at);
     CREATE INDEX IF NOT EXISTS idx_agent_team_tasks_order
        ON agent_team_tasks(team_id, order_index, created_at);",
    // ---- 4: retain hidden assistant reasoning ---------------------------
    "ALTER TABLE session_messages ADD COLUMN reasoning_content TEXT;",
    // ---- 5: workflow-driver compare-and-swap leases ---------------------
    "ALTER TABLE workflow_runs ADD COLUMN revision INTEGER NOT NULL DEFAULT 0;
     ALTER TABLE workflow_runs ADD COLUMN lease_owner TEXT;
     ALTER TABLE workflow_runs ADD COLUMN lease_expires_at TEXT;
     CREATE INDEX IF NOT EXISTS idx_workflow_runs_lease ON workflow_runs(lease_expires_at);",
    // ---- 6: the historical schema revision is intentionally empty ------
    "",
    // ---- 7: tool-effect ledger (B5, see `super::run_ledger::tool_effects`) -
    //
    // One row per admitted tool call, written `started` before the call
    // executes and updated in place to a terminal status
    // (`completed`/`failed`/`interrupted`) once it settles. A row still
    // `started` for a run with no live driver is the signature of a crash
    // between admission and settlement; `list_unresolved_tool_effects` finds
    // exactly those rows so a resumed run can decide, per tool, whether it is
    // safe to re-execute (see `tinytools::ToolReplay`).
    "CREATE TABLE IF NOT EXISTS tool_effects (
        run_id           TEXT NOT NULL,
        call_id          TEXT NOT NULL,
        tool             TEXT NOT NULL,
        status           TEXT NOT NULL,
        idempotency_key  TEXT,
        effect_summary   TEXT,
        started_at       TEXT NOT NULL,
        settled_at       TEXT,
        PRIMARY KEY (run_id, call_id)
     );
     CREATE INDEX IF NOT EXISTS idx_tool_effects_status ON tool_effects(run_id, status);
     CREATE INDEX IF NOT EXISTS idx_tool_effects_idempotency
        ON tool_effects(idempotency_key);",
];

/// Create a classic database at `path`, migrated through `version` (inclusive),
/// and run `populate` on it. The connection is closed (and the database
/// checkpointed) before this returns, so the file can be admitted as a source.
pub(crate) fn classic_database(
    path: &std::path::Path,
    version: usize,
    populate: impl FnOnce(&Connection),
) {
    let conn = Connection::open(path).expect("create classic database");
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_version (
            id      INTEGER PRIMARY KEY CHECK (id = 1),
            version INTEGER NOT NULL
         );",
    )
    .expect("schema_version table");
    for (index, sql) in CLASSIC_MIGRATIONS.iter().enumerate().take(version + 1) {
        conn.execute_batch(sql).expect("classic migration");
        conn.execute(
            "INSERT INTO schema_version (id, version) VALUES (1, ?1)
             ON CONFLICT(id) DO UPDATE SET version = excluded.version",
            params![index as i64],
        )
        .expect("record version");
    }
    populate(&conn);
    drop(conn);
}
