//! Durable agent-team coordination operations.
//!
//! Teams, members, and tasks share the run-ledger database but have distinct
//! lifecycle and claim semantics from agent and workflow runs.

use super::*;

// ---------------------------------------------------------------------------
// Agent-team coordination (issue #3374)
// ---------------------------------------------------------------------------

/// Insert or update a team row.
pub fn upsert_agent_team(workspace_dir: &Path, upsert: AgentTeamUpsert) -> Result<AgentTeam> {
    let now = Utc::now();
    let created_at = upsert.created_at.unwrap_or(now);
    tinyagents_tracing::debug!(
        "{LOG_PREFIX} upsert_agent_team.entry id={} lead={} status={}",
        upsert.id,
        upsert.lead_agent_id,
        upsert.status.as_str()
    );
    // One transaction for the write *and* the read-back. Committing the insert
    // on an autocommit connection, closing it, then re-opening to `get_*` hands
    // the caller whatever a concurrent writer left behind rather than what this
    // call wrote — an upsert that reports someone else's row.
    let team = crate::store::with_transaction(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        conn.execute(
            "INSERT INTO agent_teams (
                id, parent_thread_id, lead_agent_id, status, summary,
                created_at, updated_at, closed_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(id) DO UPDATE SET
                parent_thread_id = COALESCE(excluded.parent_thread_id, agent_teams.parent_thread_id),
                lead_agent_id = excluded.lead_agent_id,
                status = excluded.status,
                summary = COALESCE(excluded.summary, agent_teams.summary),
                updated_at = excluded.updated_at,
                closed_at = COALESCE(excluded.closed_at, agent_teams.closed_at)",
            params![
                upsert.id,
                upsert.parent_thread_id,
                upsert.lead_agent_id,
                upsert.status.as_str(),
                upsert.summary,
                created_at.to_rfc3339(),
                now.to_rfc3339(),
                upsert.closed_at.map(|dt| dt.to_rfc3339()),
            ],
        )
        .storage_context("upsert agent team")?;
        get_agent_team_inner(conn, &upsert.id)?.storage_context("agent team missing after upsert")
    })?;
    tinyagents_tracing::debug!("{LOG_PREFIX} upsert_agent_team.exit id={}", team.id);
    Ok(team)
}

/// Fetch a single team by id.
pub fn get_agent_team(workspace_dir: &Path, id: &str) -> Result<Option<AgentTeam>> {
    tinyagents_tracing::debug!("{LOG_PREFIX} get_agent_team.entry id={id}");
    crate::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let team = get_agent_team_inner(conn, id)?;
        tinyagents_tracing::debug!(
            "{LOG_PREFIX} get_agent_team.exit id={id} found={}",
            team.is_some()
        );
        Ok(team)
    })
}

/// List teams, most-recently-updated first, with optional thread/status filters.
pub fn list_agent_teams(
    workspace_dir: &Path,
    request: &AgentTeamListRequest,
) -> Result<AgentTeamListResponse> {
    tinyagents_tracing::debug!(
        "{LOG_PREFIX} list_agent_teams.entry parent_thread={:?} status={:?} limit={:?} offset={:?}",
        request.parent_thread_id,
        request.status,
        request.limit,
        request.offset
    );
    crate::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let mut where_clauses = Vec::new();
        let mut values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(thread) = request
            .parent_thread_id
            .as_deref()
            .filter(|s| !s.trim().is_empty())
        {
            values.push(Box::new(thread.to_string()));
            where_clauses.push(format!("parent_thread_id = ?{}", values.len()));
        }
        if let Some(status) = request.status.as_deref().filter(|s| !s.trim().is_empty()) {
            values.push(Box::new(status.to_string()));
            where_clauses.push(format!("status = ?{}", values.len()));
        }

        let where_sql = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };
        let count_sql = format!("SELECT COUNT(*) FROM agent_teams {where_sql}");
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            values.iter().map(|v| v.as_ref()).collect();
        let count = conn.query_row(&count_sql, params_ref.as_slice(), |row| {
            row.get::<_, i64>(0)
        })? as usize;

        let limit = request.limit.unwrap_or(50).min(500) as i64;
        // `offset` is `u64`; convert checked so a value > i64::MAX surfaces a
        // clear error instead of wrapping negative and corrupting pagination.
        let offset = i64::try_from(request.offset.unwrap_or(0))
            .storage_context("agent team list offset exceeds i64::MAX")?;
        values.push(Box::new(limit));
        let limit_idx = values.len();
        values.push(Box::new(offset));
        let offset_idx = values.len();

        let query_sql = format!(
            "SELECT id, parent_thread_id, lead_agent_id, status, summary,
                    created_at, updated_at, closed_at
             FROM agent_teams {where_sql}
             ORDER BY updated_at DESC
             LIMIT ?{limit_idx} OFFSET ?{offset_idx}"
        );
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            values.iter().map(|v| v.as_ref()).collect();
        let mut stmt = conn.prepare(&query_sql)?;
        let rows = stmt.query_map(params_ref.as_slice(), map_agent_team_row)?;
        let mut teams = Vec::new();
        for row in rows {
            teams.push(row?);
        }
        tinyagents_tracing::debug!(
            "{LOG_PREFIX} list_agent_teams.exit count={count} returned={}",
            teams.len()
        );
        Ok(AgentTeamListResponse { teams, count })
    })
}

/// Insert or update a team member. `UNIQUE(team_id, name)` enforces unique names.
pub fn upsert_agent_team_member(
    workspace_dir: &Path,
    upsert: AgentTeamMemberUpsert,
) -> Result<AgentTeamMember> {
    let now = Utc::now();
    let created_at = upsert.created_at.unwrap_or(now);
    tinyagents_tracing::debug!(
        "{LOG_PREFIX} upsert_agent_team_member.entry id={} team={} name={} status={}",
        upsert.id,
        upsert.team_id,
        upsert.name,
        upsert.member_status.as_str()
    );
    // One transaction for the write *and* the read-back. Committing the insert
    // on an autocommit connection, closing it, then re-opening to `get_*` hands
    // the caller whatever a concurrent writer left behind rather than what this
    // call wrote — an upsert that reports someone else's row.
    let member = crate::store::with_transaction(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        conn.execute(
            "INSERT INTO agent_team_members (
                id, team_id, name, agent_id, member_status,
                current_task_id, worker_thread_id, run_id, created_at, updated_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                agent_id = COALESCE(excluded.agent_id, agent_team_members.agent_id),
                member_status = excluded.member_status,
                current_task_id = COALESCE(excluded.current_task_id, agent_team_members.current_task_id),
                worker_thread_id = COALESCE(excluded.worker_thread_id, agent_team_members.worker_thread_id),
                run_id = COALESCE(excluded.run_id, agent_team_members.run_id),
                updated_at = excluded.updated_at",
            params![
                upsert.id,
                upsert.team_id,
                upsert.name,
                upsert.agent_id,
                upsert.member_status.as_str(),
                upsert.current_task_id,
                upsert.worker_thread_id,
                upsert.run_id,
                created_at.to_rfc3339(),
                now.to_rfc3339(),
            ],
        )
        .storage_context("upsert agent team member")?;
        get_agent_team_member_inner(conn, &upsert.id)?
            .storage_context("agent team member missing after upsert")
    })?;
    tinyagents_tracing::debug!(
        "{LOG_PREFIX} upsert_agent_team_member.exit id={}",
        member.id
    );
    Ok(member)
}

/// Fetch a single member by id.
pub fn get_agent_team_member(workspace_dir: &Path, id: &str) -> Result<Option<AgentTeamMember>> {
    tinyagents_tracing::debug!("{LOG_PREFIX} get_agent_team_member.entry id={id}");
    crate::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let member = get_agent_team_member_inner(conn, id)?;
        tinyagents_tracing::debug!(
            "{LOG_PREFIX} get_agent_team_member.exit id={id} found={}",
            member.is_some()
        );
        Ok(member)
    })
}

/// List all members of a team, by creation order.
pub fn list_agent_team_members(
    workspace_dir: &Path,
    team_id: &str,
) -> Result<Vec<AgentTeamMember>> {
    tinyagents_tracing::debug!("{LOG_PREFIX} list_agent_team_members.entry team={team_id}");
    crate::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let mut stmt = conn.prepare(
            "SELECT id, team_id, name, agent_id, member_status,
                    current_task_id, worker_thread_id, run_id, created_at, updated_at
             FROM agent_team_members WHERE team_id = ?1
             ORDER BY created_at ASC",
        )?;
        let rows = stmt.query_map(params![team_id], map_agent_team_member_row)?;
        let mut members = Vec::new();
        for row in rows {
            members.push(row?);
        }
        tinyagents_tracing::debug!(
            "{LOG_PREFIX} list_agent_team_members.exit team={team_id} count={}",
            members.len()
        );
        Ok(members)
    })
}

/// Insert or update a team task.
pub fn upsert_agent_team_task(
    workspace_dir: &Path,
    upsert: AgentTeamTaskUpsert,
) -> Result<AgentTeamTask> {
    let now = Utc::now();
    let created_at = upsert.created_at.unwrap_or(now);
    let depends_on_json =
        serde_json::to_string(&upsert.depends_on).storage_context("serialize task depends_on")?;
    let evidence_json =
        serde_json::to_string(&upsert.evidence).storage_context("serialize task evidence")?;
    let gate_status = upsert.gate_status.unwrap_or_else(|| "pending".to_string());
    tinyagents_tracing::debug!(
        "{LOG_PREFIX} upsert_agent_team_task.entry id={} team={} status={} deps={}",
        upsert.id,
        upsert.team_id,
        upsert.status.as_str(),
        upsert.depends_on.len()
    );
    // One transaction for the write *and* the read-back. Committing the insert
    // on an autocommit connection, closing it, then re-opening to `get_*` hands
    // the caller whatever a concurrent writer left behind rather than what this
    // call wrote — an upsert that reports someone else's row.
    let task = crate::store::with_transaction(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        conn.execute(
            "INSERT INTO agent_team_tasks (
                id, team_id, title, objective, status, owner_member_id,
                claimed_by_member_id, claim_token, depends_on_json, gate_status,
                gate_reason, evidence_json, source_run_id, order_index,
                created_at, updated_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL, NULL, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
             ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                objective = COALESCE(excluded.objective, agent_team_tasks.objective),
                status = excluded.status,
                -- A claim only means anything while the task is in_progress.
                -- Editing a claimed task through this upsert (to change
                -- dependencies, or to reset status during recovery) used to
                -- leave claimed_by_member_id/claim_token set on a todo row,
                -- which strands it: a fresh claim returns AlreadyClaimed,
                -- completion returns NotClaimed, and release/shutdown skip it
                -- because they only match in_progress. Drop the claim whenever
                -- the new status is not in_progress; preserve it otherwise so
                -- an unrelated edit does not steal a live claim.
                claimed_by_member_id = CASE
                    WHEN excluded.status = 'in_progress'
                    THEN agent_team_tasks.claimed_by_member_id ELSE NULL END,
                claim_token = CASE
                    WHEN excluded.status = 'in_progress'
                    THEN agent_team_tasks.claim_token ELSE NULL END,
                owner_member_id = COALESCE(excluded.owner_member_id, agent_team_tasks.owner_member_id),
                depends_on_json = excluded.depends_on_json,
                gate_status = excluded.gate_status,
                gate_reason = COALESCE(excluded.gate_reason, agent_team_tasks.gate_reason),
                evidence_json = excluded.evidence_json,
                source_run_id = COALESCE(excluded.source_run_id, agent_team_tasks.source_run_id),
                order_index = excluded.order_index,
                updated_at = excluded.updated_at",
            params![
                upsert.id,
                upsert.team_id,
                upsert.title,
                upsert.objective,
                upsert.status.as_str(),
                upsert.owner_member_id,
                depends_on_json,
                gate_status,
                upsert.gate_reason,
                evidence_json,
                upsert.source_run_id,
                upsert.order_index,
                created_at.to_rfc3339(),
                now.to_rfc3339(),
            ],
        )
        .storage_context("upsert agent team task")?;
        get_agent_team_task_inner(conn, &upsert.id)?
            .storage_context("agent team task missing after upsert")
    })?;
    tinyagents_tracing::debug!("{LOG_PREFIX} upsert_agent_team_task.exit id={}", task.id);
    Ok(task)
}

/// Fetch a single task by id.
pub fn get_agent_team_task(workspace_dir: &Path, id: &str) -> Result<Option<AgentTeamTask>> {
    tinyagents_tracing::debug!("{LOG_PREFIX} get_agent_team_task.entry id={id}");
    crate::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let task = get_agent_team_task_inner(conn, id)?;
        tinyagents_tracing::debug!(
            "{LOG_PREFIX} get_agent_team_task.exit id={id} found={}",
            task.is_some()
        );
        Ok(task)
    })
}

/// List all tasks of a team, by `order_index` then creation order.
pub fn list_agent_team_tasks(workspace_dir: &Path, team_id: &str) -> Result<Vec<AgentTeamTask>> {
    tinyagents_tracing::debug!("{LOG_PREFIX} list_agent_team_tasks.entry team={team_id}");
    crate::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let mut stmt = conn.prepare(
            "SELECT id, team_id, title, objective, status, owner_member_id,
                    claimed_by_member_id, claim_token, depends_on_json, gate_status,
                    gate_reason, evidence_json, source_run_id, order_index,
                    created_at, updated_at
             FROM agent_team_tasks WHERE team_id = ?1
             ORDER BY order_index ASC, created_at ASC",
        )?;
        let rows = stmt.query_map(params![team_id], map_agent_team_task_row)?;
        let mut tasks = Vec::new();
        for row in rows {
            tasks.push(row?);
        }
        tinyagents_tracing::debug!(
            "{LOG_PREFIX} list_agent_team_tasks.exit team={team_id} count={}",
            tasks.len()
        );
        Ok(tasks)
    })
}

/// Atomically claim a task for a member.
///
/// All steps run inside a single `with_connection` transaction so that the
/// dependency check and the compare-and-swap observe a consistent snapshot:
/// 1. Resolve the task by `(id, team_id)`; absent → [`ClaimOutcome::UnknownTask`].
/// 2. For every dependency id, look up its status; collect those not `done`
///    into `unmet`. Non-empty → [`ClaimOutcome::Blocked`].
/// 3. WHERE-guarded `UPDATE ... WHERE claimed_by_member_id IS NULL`: SQLite
///    serializes writers, so exactly one concurrent claimer flips the row from
///    unclaimed to claimed. `rows_affected == 0` → already taken
///    ([`ClaimOutcome::AlreadyClaimed`]); otherwise re-fetch and return
///    [`ClaimOutcome::Claimed`].
pub fn claim_agent_team_task(
    workspace_dir: &Path,
    team_id: &str,
    task_id: &str,
    member_id: &str,
    claim_token: &str,
) -> Result<ClaimOutcome> {
    tinyagents_tracing::debug!(
        "{LOG_PREFIX} claim_agent_team_task.entry team={team_id} task={task_id} member={member_id}"
    );
    let outcome = crate::store::with_transaction(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;

        // 1. Resolve the task within this team.
        let task = match get_agent_team_task_inner(conn, task_id)? {
            Some(task) if task.team_id == team_id => task,
            _ => {
                tinyagents_tracing::debug!(
                    "{LOG_PREFIX} claim_agent_team_task.unknown team={team_id} task={task_id}"
                );
                return Ok(ClaimOutcome::UnknownTask);
            }
        };

        // 2. Dependency gate: every dep must be `done`.
        let mut unmet = Vec::new();
        for dep_id in &task.depends_on {
            let dep_status: Option<String> = conn
                .query_row(
                    "SELECT status FROM agent_team_tasks WHERE id = ?1 AND team_id = ?2",
                    params![dep_id, team_id],
                    |row| row.get(0),
                )
                .optional()?;
            let is_done = dep_status.as_deref() == Some(AgentTeamTaskStatus::Done.as_str());
            if !is_done {
                unmet.push(dep_id.clone());
            }
        }
        if !unmet.is_empty() {
            tinyagents_tracing::debug!(
                "{LOG_PREFIX} claim_agent_team_task.blocked team={team_id} task={task_id} unmet={}",
                unmet.len()
            );
            return Ok(ClaimOutcome::Blocked { unmet });
        }

        // 3. Compare-and-swap on the unclaimed guard **and the status**.
        //
        // `claimed_by_member_id IS NULL` alone is not a guard: `upsert_agent_team_task`
        // deliberately NULLs that column whenever the new status is not
        // `in_progress`, so every `done` task also satisfies it. A stale worker
        // re-claiming a finished task therefore flipped it straight back to
        // `in_progress` — and stranded everything downstream, because the
        // completion gate re-checks that each dependency is still `done` and now
        // reports it unfinished. A terminal task is not claimable, whatever its
        // claim column says.
        let now = Utc::now();
        let rows_affected = conn
            .execute(
                "UPDATE agent_team_tasks
                 SET claimed_by_member_id = ?1, claim_token = ?2, status = 'in_progress', updated_at = ?3
                 WHERE id = ?4 AND team_id = ?5 AND claimed_by_member_id IS NULL
                   AND status IN ('todo', 'ready', 'blocked')",
                params![member_id, claim_token, now.to_rfc3339(), task_id, team_id],
            )
            .storage_context("compare-and-swap claim agent team task")?;
        if rows_affected == 0 {
            tinyagents_tracing::debug!(
                "{LOG_PREFIX} claim_agent_team_task.already_claimed team={team_id} task={task_id} \
                 status={}",
                task.status.as_str()
            );
            return Ok(ClaimOutcome::AlreadyClaimed);
        }

        let claimed = get_agent_team_task_inner(conn, task_id)?
            .storage_context("claimed task missing after compare-and-swap")?;
        Ok(ClaimOutcome::Claimed(Box::new(claimed)))
    })?;
    tinyagents_tracing::debug!(
        "{LOG_PREFIX} claim_agent_team_task.exit team={team_id} task={task_id} outcome={}",
        match &outcome {
            ClaimOutcome::Claimed(_) => "claimed",
            ClaimOutcome::AlreadyClaimed => "already_claimed",
            ClaimOutcome::Blocked { .. } => "blocked",
            ClaimOutcome::UnknownTask => "unknown",
        }
    );
    Ok(outcome)
}

/// Quality-gate a task's completion and, on pass, transition it to `done`.
///
/// Runs inside a single transaction so the gate evaluation and the status flip
/// observe one consistent snapshot:
/// 1. Resolve the task by `(id, team_id)`; absent → [`CompletionOutcome::UnknownTask`].
/// 2. The completer must be the current claimant and the task must be
///    `in_progress`; otherwise [`CompletionOutcome::NotClaimed`].
/// 3. Evaluate the quality gate (every dependency `done`, claimant matches any
///    pre-assigned owner, evidence present when `require_evidence`). Any unmet
///    invariant records `gate_status = "failed"` + the joined reasons and leaves
///    the task `in_progress` → [`CompletionOutcome::GateFailed`].
/// 4. On pass, merge `evidence`, set `status = "done"`, `gate_status = "passed"`,
///    clear `gate_reason`, re-fetch → [`CompletionOutcome::Completed`].
pub fn complete_agent_team_task(
    workspace_dir: &Path,
    team_id: &str,
    task_id: &str,
    member_id: &str,
    evidence: &[String],
    require_evidence: bool,
) -> Result<CompletionOutcome> {
    tinyagents_tracing::debug!(
        "{LOG_PREFIX} complete_agent_team_task.entry team={team_id} task={task_id} member={member_id}"
    );
    let outcome = crate::store::with_transaction(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;

        // 1. Resolve the task within this team.
        let task = match get_agent_team_task_inner(conn, task_id)? {
            Some(task) if task.team_id == team_id => task,
            _ => {
                tinyagents_tracing::debug!(
                    "{LOG_PREFIX} complete_agent_team_task.unknown team={team_id} task={task_id}"
                );
                return Ok(CompletionOutcome::UnknownTask);
            }
        };

        // 2. Only the current claimant may complete, and only while in progress.
        let is_claimant = task.claimed_by_member_id.as_deref() == Some(member_id);
        let in_progress = task.status == AgentTeamTaskStatus::InProgress;
        if !is_claimant || !in_progress {
            tinyagents_tracing::debug!(
                "{LOG_PREFIX} complete_agent_team_task.not_claimed team={team_id} task={task_id} claimant={is_claimant} in_progress={in_progress}"
            );
            return Ok(CompletionOutcome::NotClaimed);
        }

        // Merge prior evidence with the newly-supplied links (de-duplicated,
        // order-preserving) so a retry that adds evidence accumulates it.
        let mut merged_evidence = task.evidence.clone();
        for link in evidence {
            if !merged_evidence.iter().any(|e| e == link) {
                merged_evidence.push(link.clone());
            }
        }

        // 3. Quality gate.
        let reasons =
            evaluate_completion_gate(conn, team_id, &task, &merged_evidence, require_evidence)?;
        let now = Utc::now();
        if !reasons.is_empty() {
            let joined = reasons.join("; ");
            // Persist the merged evidence even though the gate failed. Evidence
            // accumulates across attempts (see `merged_evidence` above), so
            // dropping it here punished a caller for an unrelated gate failure:
            // after a dependency was fixed, a retry that did not resend the same
            // links would fail `require_evidence` on evidence it had already
            // submitted. Only the gate verdict is a failure; the submission is
            // still real.
            let evidence_json = serde_json::to_string(&merged_evidence)
                .storage_context("serialize completion evidence")?;
            conn.execute(
                "UPDATE agent_team_tasks
                 SET gate_status = 'failed', gate_reason = ?1,
                     evidence_json = ?2, updated_at = ?3
                 WHERE id = ?4 AND team_id = ?5",
                params![joined, evidence_json, now.to_rfc3339(), task_id, team_id],
            )
            .storage_context("record failed completion gate")?;
            tinyagents_tracing::debug!(
                "{LOG_PREFIX} complete_agent_team_task.gate_failed team={team_id} task={task_id} reasons={}",
                reasons.len()
            );
            return Ok(CompletionOutcome::GateFailed { reasons });
        }

        // 4. Gate passed — flip to done. The WHERE clause is the real CAS: the
        // `claimed_by_member_id` guard stops a concurrent shutdown/unclaim from
        // completing a task it no longer holds, and the `status = 'in_progress'`
        // guard stops a concurrent double-complete by the same member (the
        // snapshot check above is a read, not part of the swap — only one of two
        // racing UPDATEs flips `in_progress -> done`).
        let evidence_json = serde_json::to_string(&merged_evidence)
            .storage_context("serialize completion evidence")?;
        let rows_affected = conn
            .execute(
                "UPDATE agent_team_tasks
                 SET status = 'done', gate_status = 'passed', gate_reason = NULL,
                     evidence_json = ?1, updated_at = ?2
                 WHERE id = ?3 AND team_id = ?4 AND claimed_by_member_id = ?5
                   AND status = 'in_progress'",
                params![evidence_json, now.to_rfc3339(), task_id, team_id, member_id],
            )
            .storage_context("complete agent team task")?;
        if rows_affected == 0 {
            tinyagents_tracing::debug!(
                "{LOG_PREFIX} complete_agent_team_task.lost_claim team={team_id} task={task_id}"
            );
            return Ok(CompletionOutcome::NotClaimed);
        }

        let done = get_agent_team_task_inner(conn, task_id)?
            .storage_context("completed task missing after update")?;
        Ok(CompletionOutcome::Completed(Box::new(done)))
    })?;
    tinyagents_tracing::debug!(
        "{LOG_PREFIX} complete_agent_team_task.exit team={team_id} task={task_id} outcome={}",
        match &outcome {
            CompletionOutcome::Completed(_) => "completed",
            CompletionOutcome::GateFailed { .. } => "gate_failed",
            CompletionOutcome::NotClaimed => "not_claimed",
            CompletionOutcome::UnknownTask => "unknown",
        }
    );
    Ok(outcome)
}

/// Evaluate the quality-gate invariants for a completing task. Returns one
/// human-readable reason per unmet invariant (empty = gate passes).
fn evaluate_completion_gate(
    conn: &Connection,
    team_id: &str,
    task: &AgentTeamTask,
    merged_evidence: &[String],
    require_evidence: bool,
) -> Result<Vec<String>> {
    let mut reasons = Vec::new();

    // Every dependency must still be `done` (defends against a dependency that
    // regressed after this task was claimed).
    for dep_id in &task.depends_on {
        let dep_status: Option<String> = conn
            .query_row(
                "SELECT status FROM agent_team_tasks WHERE id = ?1 AND team_id = ?2",
                params![dep_id, team_id],
                |row| row.get(0),
            )
            .optional()?;
        if dep_status.as_deref() != Some(AgentTeamTaskStatus::Done.as_str()) {
            reasons.push(format!("dependency {dep_id} is not done"));
        }
    }

    // No overlapping ownership: a pre-assigned owner must be the one completing.
    if let Some(owner) = task
        .owner_member_id
        .as_ref()
        .filter(|owner| Some(owner.as_str()) != task.claimed_by_member_id.as_deref())
    {
        reasons.push(format!(
            "task is owned by {owner} but claimed by {}",
            task.claimed_by_member_id.as_deref().unwrap_or("nobody")
        ));
    }

    // Evidence gate.
    if require_evidence && merged_evidence.is_empty() {
        reasons.push("completion requires at least one evidence link".to_string());
    }

    Ok(reasons)
}

/// Stop a team member and release any task it is actively working on.
///
/// In one transaction: unclaim the member's `in_progress` tasks back to `todo`
/// (clearing claimant + token so another teammate can pick them up), then mark
/// the member `stopped` and clear its `current_task_id`. Returns the updated
/// member plus the ids of the tasks that were released, or `None` if the member
/// is not part of the team.
pub fn shutdown_agent_team_member(
    workspace_dir: &Path,
    team_id: &str,
    member_id: &str,
) -> Result<Option<(AgentTeamMember, Vec<String>)>> {
    tinyagents_tracing::debug!(
        "{LOG_PREFIX} shutdown_agent_team_member.entry team={team_id} member={member_id}"
    );
    let result = crate::store::with_transaction(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;

        // Existence + team-membership check only; the row is intentionally not
        // reused — the caller-facing member is re-read after the UPDATEs below so
        // it reflects the stopped state.
        match get_agent_team_member_inner(conn, member_id)? {
            Some(found) if found.team_id == team_id => {}
            _ => {
                tinyagents_tracing::debug!(
                    "{LOG_PREFIX} shutdown_agent_team_member.unknown team={team_id} member={member_id}"
                );
                return Ok(None);
            }
        }

        // Collect the ids first so the caller can report exactly what was freed.
        let released: Vec<String> = {
            let mut stmt = conn.prepare(
                "SELECT id FROM agent_team_tasks
                 WHERE team_id = ?1 AND claimed_by_member_id = ?2 AND status = 'in_progress'",
            )?;
            let ids = stmt.query_map(params![team_id, member_id], |row| row.get::<_, String>(0))?;
            let mut out = Vec::new();
            for id in ids {
                out.push(id?);
            }
            out
        };

        let now = Utc::now();
        conn.execute(
            "UPDATE agent_team_tasks
             SET claimed_by_member_id = NULL, claim_token = NULL, status = 'todo', updated_at = ?1
             WHERE team_id = ?2 AND claimed_by_member_id = ?3 AND status = 'in_progress'",
            params![now.to_rfc3339(), team_id, member_id],
        )
        .storage_context("release tasks on member shutdown")?;
        conn.execute(
            "UPDATE agent_team_members
             SET member_status = 'stopped', current_task_id = NULL, updated_at = ?1
             WHERE id = ?2 AND team_id = ?3",
            params![now.to_rfc3339(), member_id, team_id],
        )
        .storage_context("stop agent team member")?;

        let member = get_agent_team_member_inner(conn, member_id)?
            .storage_context("member missing after shutdown")?;
        Ok(Some((member, released)))
    })?;
    tinyagents_tracing::debug!(
        "{LOG_PREFIX} shutdown_agent_team_member.exit team={team_id} member={member_id} released={}",
        result.as_ref().map(|(_, r)| r.len()).unwrap_or(0)
    );
    Ok(result)
}

/// Mark a member as actively running a task: status → `active`, with the
/// current task id and the worker/run identifiers of the spawned agent. Used by
/// the live runtime right after it claims a task and dispatches a worker.
/// Returns the updated member, or `None` if the member is not in the team.
pub fn mark_agent_team_member_running(
    workspace_dir: &Path,
    team_id: &str,
    member_id: &str,
    task_id: &str,
    worker_thread_id: &str,
    run_id: &str,
) -> Result<Option<AgentTeamMember>> {
    tinyagents_tracing::debug!(
        "{LOG_PREFIX} mark_agent_team_member_running.entry team={team_id} member={member_id} task={task_id} run={run_id}"
    );
    crate::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let now = Utc::now();
        let changed = conn
            .execute(
                "UPDATE agent_team_members
                 SET member_status = 'active', current_task_id = ?1,
                     worker_thread_id = ?2, run_id = ?3, updated_at = ?4
                 WHERE id = ?5 AND team_id = ?6",
                params![
                    task_id,
                    worker_thread_id,
                    run_id,
                    now.to_rfc3339(),
                    member_id,
                    team_id
                ],
            )
            .storage_context("mark agent team member running")?;
        if changed == 0 {
            return Ok(None);
        }
        get_agent_team_member_inner(conn, member_id)
    })
}

/// Mark a member idle: status → `idle`, clearing `current_task_id`. The
/// `worker_thread_id` / `run_id` are intentionally retained as a pointer to the
/// member's last run for history. Returns the updated member, or `None` if the
/// member is not in the team. Used when a worker run finishes (completed,
/// gate-failed, or failed) so the member is free to pick up new work.
pub fn mark_agent_team_member_idle(
    workspace_dir: &Path,
    team_id: &str,
    member_id: &str,
) -> Result<Option<AgentTeamMember>> {
    tinyagents_tracing::debug!(
        "{LOG_PREFIX} mark_agent_team_member_idle.entry team={team_id} member={member_id}"
    );
    crate::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let now = Utc::now();
        let changed = conn
            .execute(
                "UPDATE agent_team_members
                 SET member_status = 'idle', current_task_id = NULL, updated_at = ?1
                 WHERE id = ?2 AND team_id = ?3",
                params![now.to_rfc3339(), member_id, team_id],
            )
            .storage_context("mark agent team member idle")?;
        if changed == 0 {
            return Ok(None);
        }
        get_agent_team_member_inner(conn, member_id)
    })
}

/// Release a single `in_progress` task back to `todo`, clearing its claim and
/// resetting the quality gate. Returns `true` if a row was actually released
/// (the task existed, belonged to the team, and was `in_progress`). Used by the
/// live runtime when a worker run fails or is aborted, so the task is free for
/// another teammate — the per-task analogue of the bulk release in
/// `shutdown_agent_team_member`.
pub fn release_agent_team_task(workspace_dir: &Path, team_id: &str, task_id: &str) -> Result<bool> {
    tinyagents_tracing::debug!(
        "{LOG_PREFIX} release_agent_team_task.entry team={team_id} task={task_id}"
    );
    crate::store::with_connection(workspace_dir, |conn| {
        init_run_ledger_schema(conn)?;
        let now = Utc::now();
        let changed = conn
            .execute(
                "UPDATE agent_team_tasks
                 SET status = 'todo', claimed_by_member_id = NULL, claim_token = NULL,
                     gate_status = 'pending', gate_reason = NULL, updated_at = ?1
                 WHERE id = ?2 AND team_id = ?3 AND status = 'in_progress'",
                params![now.to_rfc3339(), task_id, team_id],
            )
            .storage_context("release agent team task")?;
        tinyagents_tracing::debug!(
            "{LOG_PREFIX} release_agent_team_task.exit team={team_id} task={task_id} released={}",
            changed > 0
        );
        Ok(changed > 0)
    })
}
