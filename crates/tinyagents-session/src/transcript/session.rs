//! Durable identity for a conversation.
//!
//! Before this module a session was identified only by the caller-supplied
//! *stem*, which hosts minted per process as `{unix_ts}_{agent}`. Two
//! consequences followed, and both cost users their history:
//!
//! 1. Every cold boot produced a new stem, so one conversation accumulated
//!    several transcript files. Resume picked the newest by `_meta.created`
//!    ([`super::find_root_transcript_for_thread`]), so any file that persisted
//!    less than it was seeded with silently shortened the conversation for
//!    every session after it.
//! 2. Two processes over one workspace each minted their own stem and neither
//!    could see the other's turns.
//!
//! A [`SessionRef`] is that missing identity. It is built from values the host
//! already has — the conversation key (a thread id) and the agent id — and
//! [`session_stem`] maps it to a filename **deterministically, with no
//! timestamp**. The same conversation therefore resolves to the same
//! transcript in every process, on every launch, forever.
//!
//! # Generations
//!
//! A transcript is what the model sees, so a compaction genuinely does shorten
//! it. That must never be done by rewriting history in place: the replaced
//! turns would be gone. Instead a compaction *seals* the current generation and
//! opens the next one ([`SessionRef::next_generation`]), which starts from the
//! compacted set and records the sealed generation as its parent. Generation
//! `n` stays on disk byte-for-byte, so the full conversation remains
//! recoverable by walking the chain even though the model only ever sees the
//! head.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::paths::sanitize_stem;

/// The separator every root-transcript scan uses to recognise a sub-agent
/// stem. Kept in one place because both [`session_stem`] and the scans in
/// [`super::paths`] / [`super::thread_lookup`] must agree on it exactly.
pub(crate) const SUBAGENT_SEPARATOR: &str = "__";

/// Durable identity of one conversation, as seen by the session layer.
///
/// `session_key` is the host's stable name for the conversation — OpenHuman
/// passes its `thread_id`. `agent_id` scopes it, because one caller-supplied
/// key can legitimately be handed to several independently configured agents
/// and each needs its own transcript (see
/// [`super::find_root_transcript_for_thread_scoped`] for the bug that taught us
/// this). `generation` distinguishes the segments a compaction creates.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SessionRef {
    /// The host's stable name for the conversation.
    pub session_key: String,
    /// The agent definition this session belongs to, when the host has one.
    pub agent_id: Option<String>,
    /// Compaction segment. `0` is the original; each compaction adds one.
    pub generation: u32,
    /// Stem of the parent session for a sub-agent, forming the `parent__child`
    /// chain the root scans filter on. `None` for a root session.
    parent_stem: Option<String>,
}

impl SessionRef {
    /// A root session for `session_key`, unscoped by agent.
    pub fn root(session_key: impl Into<String>) -> Self {
        Self {
            session_key: session_key.into(),
            agent_id: None,
            generation: 0,
            parent_stem: None,
        }
    }

    /// A root session for `session_key`, scoped to one agent definition.
    pub fn scoped(session_key: impl Into<String>, agent_id: impl Into<String>) -> Self {
        Self {
            session_key: session_key.into(),
            agent_id: Some(agent_id.into()),
            generation: 0,
            parent_stem: None,
        }
    }

    /// A sub-agent session beneath `parent`.
    ///
    /// The resulting stem is `{parent stem}__{child stem}`, which is what keeps
    /// a delegated worker out of every root-transcript scan while still
    /// recording the delegation path in one flat filename.
    pub fn child_of(parent: &SessionRef, child_key: impl Into<String>) -> Self {
        Self {
            session_key: child_key.into(),
            agent_id: None,
            generation: 0,
            parent_stem: Some(session_stem(parent)),
        }
    }

    /// Generation 0 of this session — the chain's first segment.
    pub fn first_generation(&self) -> Self {
        Self {
            generation: 0,
            ..self.clone()
        }
    }

    /// The successor this session's next compaction writes into.
    pub fn next_generation(&self) -> Self {
        Self {
            generation: self.generation.saturating_add(1),
            ..self.clone()
        }
    }

    /// Whether this session is a delegated sub-agent rather than a root.
    pub fn is_subagent(&self) -> bool {
        self.parent_stem.is_some()
    }

    /// The session id recorded in `_meta.session_id` — the stem, which is the
    /// one name that is unique per generation and stable across processes.
    pub fn session_id(&self) -> String {
        session_stem(self)
    }

    /// The session id of the generation this one succeeded, if any.
    pub fn parent_session_id(&self) -> Option<String> {
        (self.generation > 0).then(|| {
            session_stem(&Self {
                generation: self.generation - 1,
                ..self.clone()
            })
        })
    }
}

/// The transcript stem for `session`: deterministic, filesystem-safe, and
/// **free of any timestamp**. That absence is the point — it is what makes one
/// conversation resolve to one file across restarts and across processes.
///
/// Shape: `{key}` for a root generation 0, `{key}.{agent}` when scoped,
/// `.g{n}` appended from generation 1, and the whole thing prefixed with
/// `{parent}__` for a sub-agent.
pub fn session_stem(session: &SessionRef) -> String {
    let mut stem = sanitize_component(&session.session_key);
    if let Some(agent_id) = session
        .agent_id
        .as_deref()
        .filter(|id| !id.trim().is_empty())
    {
        stem.push('.');
        stem.push_str(&sanitize_component(agent_id));
    }
    if session.generation > 0 {
        stem.push_str(&format!(".g{}", session.generation));
    }
    match session.parent_stem.as_deref() {
        Some(parent) => format!("{parent}{SUBAGENT_SEPARATOR}{stem}"),
        None => stem,
    }
}

/// Longest human-readable prefix kept before the disambiguating digest.
/// Bounds every component (and therefore the filenames built from it) well
/// under common filesystem name limits (255 bytes), even after a `.g{n}`
/// suffix, an agent id, and a chain of `__`-joined sub-agent ancestors.
const MAX_COMPONENT_PREFIX: usize = 80;

/// One component of a stem: path-safe, bounded in length, and encoded so
/// that no two *different* raw values can ever collide on the same
/// filename — including collisions introduced by the sanitization itself.
///
/// Two lossy transforms are needed to keep [`SUBAGENT_SEPARATOR`] and the
/// `.` separators (agent id, `.g{n}` generation suffix) unambiguous:
///   * runs of `_` are collapsed, so a component can never reproduce
///     `__` and be mistaken for the sub-agent separator; and
///   * `.` is replaced with `-`, so a literal `.` in a raw component can
///     never be mistaken for the reserved agent/generation separator.
///
/// Both are lossy: distinct raw values (`a_b` vs `a__b`, `t.a` vs a `t` root
/// scoped to agent `a`, `thread-1.g1` vs `thread-1`'s next generation) could
/// otherwise sanitize to the *same* text and silently share one transcript.
/// A short deterministic digest of the untouched raw value is appended to
/// rule that out: two components produce the same encoded stem only when
/// their raw values are identical. `DefaultHasher::new()` uses fixed keys
/// (not the per-process-random keys `RandomState` uses for hash maps), so
/// the digest — like the rest of this module — carries no randomness and no
/// timestamp: the same raw value always re-derives the same stem.
fn sanitize_component(value: &str) -> String {
    let sanitized = sanitize_stem(value);
    let mut out = String::with_capacity(sanitized.len().min(MAX_COMPONENT_PREFIX));
    let mut kept = 0usize;
    for ch in sanitized.chars() {
        if kept >= MAX_COMPONENT_PREFIX {
            break;
        }
        // `.` is reserved for the agent-id and generation separators.
        let ch = if ch == '.' { '-' } else { ch };
        if ch == '_' && out.ends_with('_') {
            continue;
        }
        out.push(ch);
        kept += 1;
    }

    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    out.push('~');
    out.push_str(&format!("{:016x}", hasher.finish()));
    out
}

#[cfg(test)]
#[path = "session_test.rs"]
mod test;
