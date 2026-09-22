//! Adoption of conversations written before session identity existed.
//!
//! A host that minted a transcript stem per process left one conversation
//! spread across several root transcripts, and resume only ever loaded the
//! newest of them — so the opening turns of a thread became unreachable to the
//! model while the UI, which concatenates every matching root, still displayed
//! them.
//!
//! [`adopt_legacy_session_transcripts`] closes that gap once per conversation.
//! The first time a session is resumed and has no transcript of its own, the
//! legacy roots for its thread are read in `_meta.created` order and written
//! into the session's generation 0. The legacy files are never modified,
//! moved, or deleted: adoption only ever *adds* the file the session layer
//! will use from then on.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use super::paths::resolve_keyed_transcript_path;
use super::reader::read_transcript;
use super::session::{SessionRef, session_stem};
use super::thread_lookup::find_root_transcripts_for_thread;
use super::types::{TranscriptMessage, TranscriptMeta};
use super::writer::write_transcript_if_absent;

/// What adoption did for one session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionAdoption {
    /// The transcript now backing the session.
    pub path: PathBuf,
    /// The legacy roots folded into it, oldest first.
    pub adopted: Vec<PathBuf>,
    /// Messages carried over.
    pub messages: usize,
}

/// An adoption lock older than this is assumed to belong to a process that
/// crashed mid-adoption rather than one still working, and is reclaimed
/// rather than left blocking every future resume forever.
const STALE_LOCK_AGE: Duration = Duration::from_secs(60);

/// Fold the legacy root transcripts of `session`'s thread into its own
/// generation 0, if it has none yet and any exist.
///
/// Returns `Ok(None)` when there is nothing to do — the session already has a
/// transcript, the thread has no legacy roots, or a concurrent adopter is
/// already handling this session — which makes repeat calls harmless.
///
/// `thread_id` is passed separately from `session` because the legacy files are
/// keyed by `_meta.thread_id`, which is what the host had before it had a
/// session key, and the two are not required to be the same string.
pub fn adopt_legacy_session_transcripts(
    workspace_dir: &Path,
    session: &SessionRef,
    thread_id: &str,
    seed_meta: &TranscriptMeta,
) -> Result<Option<SessionAdoption>> {
    let stem = session_stem(session);
    let destination = resolve_keyed_transcript_path(workspace_dir, &stem)?;
    if destination.exists() {
        return Ok(None);
    }

    // Two processes can otherwise both pass the check above, both scan the
    // same legacy roots, and race to write `destination`: whichever full
    // rewrite lands last wins and the other's turns (if it observed a
    // different, possibly more complete, set of roots) are gone with no way
    // to notice, because both callers now see `destination.exists()` and
    // return `Ok(None)`. Holding this lock for the whole check-scan-write
    // makes one adoption per session the only one that can ever run at a
    // time; every other concurrent caller backs off and treats the winner's
    // result as its own.
    let lock_path = adoption_lock_path(&destination);
    let Some(_lock) = AdoptionLock::acquire(&lock_path)? else {
        tracing::debug!(
            "[transcript-adoption] session={stem} adoption already in progress elsewhere; \
             skipping"
        );
        return Ok(None);
    };
    // Re-check now that the lock is held: another process may have finished
    // adoption (or written the session's own first turn) between the
    // unlocked check above and acquiring the lock.
    if destination.exists() {
        return Ok(None);
    }

    // Oldest first, by `_meta.created`. Anything already pointing at this
    // session's own file is excluded so a partially-adopted workspace cannot
    // fold a file into itself.
    let legacy: Vec<PathBuf> = find_root_transcripts_for_thread(workspace_dir, thread_id)
        .into_iter()
        .filter(|path| path != &destination)
        .collect();
    if legacy.is_empty() {
        return Ok(None);
    }

    let mut messages: Vec<TranscriptMessage> = Vec::new();
    let mut meta = seed_meta.clone();
    meta.session_id = Some(session.session_id());
    meta.parent_session_id = session.parent_session_id();
    meta.thread_id = Some(thread_id.to_string());
    meta.turn_count = 0;
    meta.input_tokens = 0;
    meta.output_tokens = 0;
    meta.cached_input_tokens = 0;
    meta.charged_amount_usd = 0.0;
    let mut earliest_created: Option<String> = None;
    let mut latest_updated: Option<String> = None;
    let mut adopted = Vec::new();

    for path in legacy {
        // A legacy candidate must be read to be filtered, so an unreadable
        // one fails the whole call rather than being silently skipped. The
        // destination is this call's own idempotency marker: skipping it and
        // writing anyway would create that marker over an *incomplete* fold,
        // and because `destination.exists()` short-circuits every later
        // call, the skipped file's turns would never be retried — even after
        // the file became readable again. Failing instead leaves nothing on
        // disk, so a later resume simply tries the whole fold again.
        let transcript = read_transcript(&path).with_context(|| {
            format!(
                "legacy root {} is unreadable; deferring adoption rather than finalizing a \
                 fold that would permanently drop its turns",
                path.display()
            )
        })?;

        // Session-identified files are not pre-identity legacy transcripts:
        // they are either one of this thread's *other* agents (when
        // `session.agent_id` is set, matching the rule
        // `find_root_transcript_for_thread_scoped` already applies), or a
        // session file — including one already adopted — that happens to
        // share this thread id. Folding either in would mix another agent's
        // history into this one, or duplicate content that adoption already
        // recovered once.
        if transcript.meta.session_id.is_some() {
            continue;
        }
        if let Some(expected_agent) = session.agent_id.as_deref()
            && transcript.meta.agent_id.as_deref() != Some(expected_agent)
        {
            continue;
        }

        messages.extend(transcript.messages);
        meta.turn_count += transcript.meta.turn_count;
        meta.input_tokens += transcript.meta.input_tokens;
        meta.output_tokens += transcript.meta.output_tokens;
        meta.cached_input_tokens += transcript.meta.cached_input_tokens;
        meta.charged_amount_usd += transcript.meta.charged_amount_usd;
        if !transcript.meta.created.is_empty()
            && earliest_created
                .as_ref()
                .is_none_or(|earliest| transcript.meta.created < *earliest)
        {
            earliest_created = Some(transcript.meta.created.clone());
        }
        if !transcript.meta.updated.is_empty()
            && latest_updated
                .as_ref()
                .is_none_or(|latest| transcript.meta.updated > *latest)
        {
            latest_updated = Some(transcript.meta.updated.clone());
        }
        adopted.push(path);
    }

    if messages.is_empty() {
        return Ok(None);
    }
    if let Some(created) = earliest_created {
        meta.created = created;
    }
    if let Some(updated) = latest_updated {
        meta.updated = updated;
    }

    // `write_transcript_if_absent` never overwrites an existing destination:
    // the adoption lock only serializes competing *adopters*, not a normal
    // session turn independently creating this same first transcript while
    // adoption is still scanning. If that happened, `destination` now holds
    // real conversation data that must not be clobbered with an adoption
    // fold that started before it existed — so a lost race here discards
    // this call's fold and reports `Ok(None)`, the same as "nothing to do".
    if !write_transcript_if_absent(&destination, &messages, &meta)? {
        tracing::debug!(
            "[transcript-adoption] session={stem} lost the race to a concurrent write; \
             discarding this fold"
        );
        return Ok(None);
    }
    tracing::info!(
        "[transcript-adoption] session={stem} adopted {} legacy root(s) totalling {} message(s)",
        adopted.len(),
        messages.len()
    );
    Ok(Some(SessionAdoption {
        path: destination,
        messages: messages.len(),
        adopted,
    }))
}

/// The exclusive-create lock path guarding one destination's adoption.
fn adoption_lock_path(destination: &Path) -> PathBuf {
    let mut file_name = destination
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_default();
    file_name.push(".adopting");
    destination.with_file_name(file_name)
}

/// An exclusive-create file lock held for the duration of one adoption.
///
/// Backed by [`std::fs::OpenOptions::create_new`] rather than an in-process
/// mutex because concurrent adopters are typically separate processes (two
/// hosts, or a process restarted mid-turn) with no shared memory to
/// synchronize on.
///
/// The correctness boundary that actually prevents data loss is
/// [`write_transcript_if_absent`]'s atomic publish, not this lock — the lock
/// is an efficiency optimization that lets concurrent *adopters* avoid
/// redundant scanning, not the thing standing between two writers and a
/// corrupted file. That is deliberate: reclaiming a lock purely by file age
/// (below) can never be made fully race-free without OS-level leases this
/// module does not have, so the design accepts an occasional double-scan
/// under reclamation rather than a lock a crashed owner can block forever —
/// knowing that even a full double-scan-and-write race resolves safely
/// through the atomic publish underneath it.
struct AdoptionLock {
    path: PathBuf,
    /// Written into the lock file's content at creation. [`Drop`] reads the
    /// file back and only removes it when the content still matches this
    /// token, so a lock this instance *lost* ownership of (reclaimed by
    /// another process as stale, see [`Self::acquire`]) is never unlinked
    /// out from under its new, legitimate owner.
    token: String,
}

impl AdoptionLock {
    /// Acquires the lock at `path`, or returns `Ok(None)` when another
    /// process already holds a fresh one.
    ///
    /// A lock older than [`STALE_LOCK_AGE`] is reclaimed on the assumption
    /// that its owner crashed before releasing it — otherwise a single crash
    /// mid-adoption would block that session's adoption forever. The
    /// remaining reclaim-under-a-live-owner race this cannot fully close
    /// (two processes both observe the same stale lock; see [`Self::token`]
    /// for how `Drop` avoids compounding it) resolves safely because the
    /// eventual writes still go through [`write_transcript_if_absent`].
    fn acquire(path: &Path) -> Result<Option<Self>> {
        let token = lock_token();
        match Self::create_exclusive(path, &token)? {
            true => Ok(Some(Self {
                path: path.to_path_buf(),
                token,
            })),
            false if lock_is_stale(path) => {
                tracing::warn!(
                    "[transcript-adoption] reclaiming stale adoption lock {}",
                    path.display()
                );
                let _ = std::fs::remove_file(path);
                match Self::create_exclusive(path, &token)? {
                    true => Ok(Some(Self {
                        path: path.to_path_buf(),
                        token,
                    })),
                    // Lost the race to reclaim it — the winner will finish
                    // the adoption (or lose its own race to a normal write,
                    // safely, via `write_transcript_if_absent`).
                    false => Ok(None),
                }
            }
            false => Ok(None),
        }
    }

    /// Attempts to create `path` exclusively with `token` as its content.
    /// Returns `Ok(true)` on success, `Ok(false)` when `path` already
    /// exists.
    fn create_exclusive(path: &Path, token: &str) -> Result<bool> {
        use std::io::Write;
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
        {
            Ok(mut file) => {
                // Best-effort: the exclusive create above is what actually
                // establishes ownership. A failed or partial token write
                // only widens `Drop`'s safety margin (it would then decline
                // to remove a lock it cannot positively confirm as its own),
                // it never narrows it.
                let _ = file.write_all(token.as_bytes());
                Ok(true)
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(false),
            Err(error) => {
                Err(error).with_context(|| format!("create adoption lock {}", path.display()))
            }
        }
    }

    /// Whether this instance's token is still what is on disk at `path`.
    fn still_owns(&self) -> bool {
        std::fs::read_to_string(&self.path).is_ok_and(|contents| contents == self.token)
    }
}

impl Drop for AdoptionLock {
    fn drop(&mut self) {
        if self.still_owns() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

fn lock_is_stale(path: &Path) -> bool {
    std::fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .and_then(|modified| {
            SystemTime::now()
                .duration_since(modified)
                .map_err(|_| std::io::Error::other("clock went backwards"))
        })
        .is_ok_and(|age| age > STALE_LOCK_AGE)
}

/// A per-process-unique token for one lock acquisition: pid plus a
/// monotonically increasing in-process counter. Not a cryptographic nonce —
/// it only has to distinguish this acquisition from acquisitions by other
/// processes and from earlier acquisitions in this one, which pid+counter
/// already guarantees deterministically and without any external
/// dependency.
fn lock_token() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nonce = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{}-{nonce}", std::process::id())
}

#[cfg(test)]
#[path = "adoption_test.rs"]
mod test;
