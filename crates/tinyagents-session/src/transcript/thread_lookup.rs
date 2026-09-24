//! Thread-keyed lookups over the canonical `session_raw` store: locating a thread's
//! root transcripts and summing its token / cost usage.

use super::paths::raw_session_dir;
use super::reader::{read_last_assistant_usage, read_transcript, read_transcript_meta_only};
use super::types::{SubagentArchetypeUsage, ThreadUsageSummary};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Find the newest root transcript whose metadata declares `thread_id`.
///
/// Root transcripts live directly under `session_raw/` and do not carry
/// the `__` separator used for sub-agent siblings. This helper is the
/// bridge PR-2 can use to route UI thread reads to the canonical root
/// transcript without accidentally folding delegated worker transcripts
/// into the main chat timeline.
pub fn find_root_transcript_for_thread(workspace_dir: &Path, thread_id: &str) -> Option<PathBuf> {
    find_root_transcripts_for_thread(workspace_dir, thread_id).pop()
}

/// [`find_root_transcript_for_thread`], additionally scoped to `agent_id`
/// when given.
///
/// Plain thread-id matching is ambiguous when several distinct agents can be
/// given the same caller-supplied `thread_id` — agents can share one
/// `session_raw/` store, keyed only by thread id in `_meta.thread_id`. Two
/// agents handed the same id would otherwise resolve to whichever one wrote
/// the newest matching transcript, splicing one agent's history (and its
/// caller's data) into another agent's turn. `agent_id` filters on
/// `_meta.agent_id`, the agent definition id written at turn time, so the
/// resume this seeds only ever pulls from the calling agent's own history.
///
/// `agent_id: None` retains the unscoped behaviour for callers that guarantee
/// one logical agent drives each `thread_id` (#5351).
pub fn find_root_transcript_for_thread_scoped(
    workspace_dir: &Path,
    thread_id: &str,
    agent_id: Option<&str>,
) -> Option<PathBuf> {
    let mut matches = find_root_transcripts_for_thread(workspace_dir, thread_id);
    if let Some(agent_id) = agent_id {
        matches.retain(|path| {
            read_transcript(path)
                .ok()
                .and_then(|transcript| transcript.meta.agent_id)
                .as_deref()
                == Some(agent_id)
        });
    }
    matches.pop()
}

/// Finds every root transcript whose metadata declares `thread_id`, oldest
/// first by `meta.created`.
///
/// Underlies [`find_root_transcript_for_thread`] (which takes the newest) and
/// [`super::history::TranscriptLocator::root_for_thread_scoped`]'s
/// agent-id filtering; exposed directly for callers that need the full
/// ordered history rather than just the latest match.
pub fn find_root_transcripts_for_thread(workspace_dir: &Path, thread_id: &str) -> Vec<PathBuf> {
    root_transcripts_for_thread_in_dir(&raw_session_dir(workspace_dir), thread_id).0
}

/// [`find_root_transcripts_for_thread`], additionally reporting whether the
/// scan hit any root `.jsonl` file it could not read at all — matching or
/// not, since a read failure happens *before* the thread-id comparison, so
/// which thread an unreadable file belonged to can never be determined.
///
/// Exists for [`super::adoption::adopt_legacy_session_transcripts`]: that
/// caller's idempotency marker is the destination file it writes, so folding
/// only the *readable* matches and reporting success would permanently
/// strand an unreadable file's turns — the marker's existence stops every
/// later retry. [`find_root_transcripts_for_thread`] itself is used by
/// callers (thread resume, usage summaries) that already treat "unreadable"
/// as "absent" and are safe to keep doing so; only adoption's
/// once-and-only-once contract needs to know the difference.
pub fn find_root_transcripts_for_thread_reporting_unreadable(
    workspace_dir: &Path,
    thread_id: &str,
) -> (Vec<PathBuf>, bool) {
    root_transcripts_for_thread_in_dir(&raw_session_dir(workspace_dir), thread_id)
}

fn root_transcripts_for_thread_in_dir(raw_dir: &Path, thread_id: &str) -> (Vec<PathBuf>, bool) {
    let thread_id = thread_id.trim();
    if thread_id.is_empty() {
        return (Vec::new(), false);
    }

    let entries = match fs::read_dir(raw_dir) {
        Ok(entries) => entries,
        // A workspace with no session_raw/ yet has genuinely adopted
        // nothing — not an error. Any other failure (permissions, a
        // transient I/O error) means the scan could not actually see
        // whether a matching root exists, which callers relying on
        // `unreadable` (adoption's idempotency contract) must not treat the
        // same as "confirmed nothing here".
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return (Vec::new(), false);
        }
        Err(error) => {
            tracing::warn!(
                "[transcript] could not scan {} for thread {thread_id}: {error}",
                raw_dir.display()
            );
            return (Vec::new(), true);
        }
    };
    let mut any_unreadable = false;
    // Keyed by `meta.created` so the order is chronological rather than
    // lexicographic. Modern stems are `{unix_ts}_{agent_id}` and sort the same
    // either way, but a legacy `{agent}_{index}` root encodes no time at all —
    // and because digits sort before letters, every legacy root sorted *after*
    // every modern one regardless of when it was written. `project_from_files`
    // concatenates these in order, so that reordered the rendered view and
    // could attach a sub-agent trail to the wrong turn.
    //
    // An explicit loop rather than a filter/filter_map chain: both the
    // directory-entry read and the transcript read below can independently
    // fail and need to set the same `any_unreadable` flag, and two closures
    // cannot each hold a mutable borrow of it at once.
    let mut matches: Vec<(String, PathBuf)> = Vec::new();
    for entry in entries {
        let path = match entry {
            Ok(entry) => entry.path(),
            Err(error) => {
                // An entry the directory iterator itself could not read
                // (e.g. a race with concurrent deletion, a transient I/O
                // error) is exactly as invisible to this scan as a file that
                // failed `read_transcript` below — the `.flatten()` this
                // loop replaced would have hidden it from every caller,
                // including adoption's fail-closed contract.
                tracing::warn!(
                    "[transcript] could not read a directory entry in {}: {error}",
                    raw_dir.display()
                );
                any_unreadable = true;
                continue;
            }
        };
        let is_candidate = path.extension().and_then(|s| s.to_str()) == Some("jsonl")
            && path
                .file_stem()
                .and_then(|s| s.to_str())
                .is_some_and(|stem| !stem.contains("__"));
        if !is_candidate {
            continue;
        }
        match read_transcript(&path) {
            Ok(transcript) if transcript.meta.thread_id.as_deref() == Some(thread_id) => {
                matches.push((transcript.meta.created.clone(), path));
            }
            Ok(_) => {}
            Err(err) => {
                tracing::warn!(
                    "[transcript] skipping unreadable root transcript candidate {}: {err}",
                    path.display()
                );
                any_unreadable = true;
            }
        }
    }

    // A compaction generation inherits its predecessor's `meta.created`
    // (`begin_generation` seeds from the live meta), so `created` alone ties a
    // whole chain. The generation number breaks that tie before the path does:
    // by path, `X.g1.jsonl` sorts *before* `X.jsonl` (`g` < `j`) and `.g10`
    // before `.g2`, which made the newest-wins lookups resolve a sealed
    // generation instead of the head. Path stays the final tiebreak so the
    // order is total and deterministic.
    matches.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| path_generation(&left.1).cmp(&path_generation(&right.1)))
            .then_with(|| left.1.cmp(&right.1))
    });
    (
        matches.into_iter().map(|(_, path)| path).collect(),
        any_unreadable,
    )
}

/// The compaction generation a transcript path encodes: `n` for a
/// `{stem}.g{n}.jsonl` successor, `0` otherwise.
///
/// Unambiguous because [`super::session::session_stem`] never lets a `.`
/// through inside a component and appends a digest to every component, so a
/// trailing `.g{digits}` can only be the generation suffix it writes.
pub(crate) fn path_generation(path: &Path) -> u32 {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(|stem| stem.rsplit_once(".g"))
        .filter(|(_, digits)| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|(_, digits)| digits.parse().ok())
        .unwrap_or(0)
}

/// Summed token/cost usage for `thread_id` across its root transcripts, or
/// `None` when the thread has no persisted turns yet.
pub fn read_thread_usage_summary(
    workspace_dir: &Path,
    thread_id: &str,
) -> Option<ThreadUsageSummary> {
    let thread_id = thread_id.trim();
    if thread_id.is_empty() {
        return None;
    }

    // Single scan: split the thread's transcripts into root (orchestrator) and
    // `__` sub-agent files. Root totals stay the parent's; sub-agent files are
    // grouped by archetype for the per-agent breakdown.
    let mut root_matches: Vec<PathBuf> = Vec::new();
    let mut sub_matches: Vec<PathBuf> = Vec::new();
    let raw_dir = raw_session_dir(workspace_dir);
    let Ok(entries) = fs::read_dir(&raw_dir) else {
        return None;
    };
    for path in entries.flatten().map(|entry| entry.path()) {
        if path.extension().and_then(|s| s.to_str()) != Some("jsonl") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let is_subagent = stem.contains("__");
        let matches_thread = read_transcript_meta_only(&path)
            .map(|m| m.thread_id.as_deref() == Some(thread_id))
            .unwrap_or(false);
        if !matches_thread {
            continue;
        }
        if is_subagent {
            sub_matches.push(path);
        } else {
            root_matches.push(path);
        }
    }

    if root_matches.is_empty() && sub_matches.is_empty() {
        return None;
    }
    root_matches.sort_by(|left, right| left.file_name().cmp(&right.file_name()));

    let mut summary = ThreadUsageSummary::default();
    for path in &root_matches {
        if let Some(meta) = read_transcript_meta_only(path) {
            summary.input_tokens = summary.input_tokens.saturating_add(meta.input_tokens);
            summary.output_tokens = summary.output_tokens.saturating_add(meta.output_tokens);
            summary.cached_input_tokens = summary
                .cached_input_tokens
                .saturating_add(meta.cached_input_tokens);
            summary.cost_usd += meta.charged_amount_usd;
            summary.turn_count = summary.turn_count.saturating_add(meta.turn_count);
        }
    }

    // Newest root transcript drives the last-turn gauge + model + updated stamp.
    if let Some(newest) = root_matches.last() {
        if let Some(meta) = read_transcript_meta_only(newest) {
            summary.updated = meta.updated;
        }
        if let Some((usage, model)) = read_last_assistant_usage(newest) {
            summary.last_turn_input_tokens = usage.input;
            summary.last_turn_output_tokens = usage.output;
            summary.model = model;
        }
    }

    // Group sub-agent transcripts by archetype (`agent_name`).
    let mut groups: BTreeMap<String, SubagentArchetypeUsage> = BTreeMap::new();
    for path in &sub_matches {
        let Some(meta) = read_transcript_meta_only(path) else {
            continue;
        };
        let group =
            groups
                .entry(meta.agent_name.clone())
                .or_insert_with(|| SubagentArchetypeUsage {
                    agent_id: meta.agent_name.clone(),
                    ..Default::default()
                });
        group.input_tokens = group.input_tokens.saturating_add(meta.input_tokens);
        group.output_tokens = group.output_tokens.saturating_add(meta.output_tokens);
        group.cached_input_tokens = group
            .cached_input_tokens
            .saturating_add(meta.cached_input_tokens);
        group.runs = group.runs.saturating_add(1);
        if group.model.is_none()
            && let Some((_, model)) = read_last_assistant_usage(path)
        {
            group.model = model;
        }
    }
    summary.subagents = groups.into_values().collect();

    Some(summary)
}
