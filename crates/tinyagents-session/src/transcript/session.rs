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

/// Longest parent-chain prefix a [`SessionRef::child_of`] call keeps
/// verbatim before collapsing it into a digest.
///
/// [`MAX_COMPONENT_PREFIX`] bounds one component, but `parent_stem` is
/// already the *entire* ancestor chain, and each further `child_of` call
/// concatenates onto it without bound — a delegation several levels deep,
/// each level with a long key, would otherwise grow the final stem past
/// filesystem name limits. Once the chain built so far exceeds this bound,
/// [`bounded_parent_stem`] replaces it with a short digest instead of
/// continuing to grow linearly with depth, so the worst case stays bounded
/// regardless of how deep delegation nests; ordinary shallow delegation (the
/// common case, see `nested_delegation_records_the_whole_path_in_one_flat_stem`)
/// keeps its fully readable, unbounded-until-this-point chain.
const MAX_PARENT_CHAIN_PREFIX: usize = 120;

/// Separator between a component's human-readable prefix and its
/// disambiguating digest. Must be a character [`sanitize_stem`] itself
/// already allows through unchanged (alphanumeric, `_`, `-`, `.`):
/// [`resolve_keyed_transcript_path`](super::paths::resolve_keyed_transcript_path)
/// re-sanitizes the whole stem this function builds before it ever becomes a
/// filename, and a separator outside that set would silently get replaced
/// with `_` at that second pass — reintroducing exactly the alias this
/// function exists to prevent.
const DIGEST_SEPARATOR: char = '-';

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
/// their raw values are identical.
///
/// The digest is [`fnv1a64`], not `std::collections::hash_map::DefaultHasher`:
/// the standard library explicitly documents `DefaultHasher`'s algorithm as
/// unspecified and subject to change between Rust releases. A durable
/// identity that is supposed to "re-derive the same stem forever" cannot be
/// built on a hash the language is free to change out from under it — a
/// toolchain upgrade would silently re-derive different filenames for every
/// existing conversation. FNV-1a's definition is fixed arithmetic with no
/// language- or library-level discretion, so it carries the same forever
/// guarantee the rest of this module's "no timestamp, no randomness" design
/// already relies on.
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

    out.push(DIGEST_SEPARATOR);
    out.push_str(&format!("{:016x}", fnv1a64(value.as_bytes())));
    out
}

/// FNV-1a, 64-bit variant: a small, fully-specified, non-cryptographic hash
/// with no algorithmic discretion left to a library or language version — see
/// [`sanitize_component`] for why that fixedness is the point. Operates on
/// bytes rather than `str::hash`, so it does not depend on
/// [`std::hash::Hash`]'s own algorithm-agnostic contract either.
fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = OFFSET_BASIS;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// `parent`'s stem, bounded to [`MAX_PARENT_CHAIN_PREFIX`]: verbatim when
/// short enough, otherwise collapsed to a fixed-length digest. See
/// [`MAX_PARENT_CHAIN_PREFIX`] for why this exists.
///
/// Never introduces (or removes) a [`SUBAGENT_SEPARATOR`]: the replacement
/// is a whole new component substituted for the whole prior chain, not a
/// truncation of it — truncating the chain string directly could cut
/// through an existing `__` and either fabricate one at a new position or
/// destroy the one recording a real ancestor boundary. Composed of only
/// alphanumerics and `-`, so it is stable under [`sanitize_component`]'s own
/// second pass same as every other component.
fn bounded_parent_stem(parent: &SessionRef) -> String {
    let stem = session_stem(parent);
    if stem.len() <= MAX_PARENT_CHAIN_PREFIX {
        return stem;
    }
    format!("chain{DIGEST_SEPARATOR}{:016x}", fnv1a64(stem.as_bytes()))
}

#[cfg(test)]
#[path = "session_test.rs"]
mod test;
