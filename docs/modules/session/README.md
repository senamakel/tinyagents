# Session: the conversation entry tree

`tinyagents-session` has two shapes over the same session database:

1. **Linear history** — `record_message`/`ops::*` (SQLite `session_messages`)
   and the JSONL transcript (`transcript::*`, `session_raw/{stem}.jsonl`).
   Both are flat, append-in-order logs. This is unchanged by this module and
   remains the default path for a host that never branches a conversation.
2. **The entry tree** (`entry_tree::*`, this document) — an append-only,
   *branchable* tree of the same conversational content, addressed by
   `EntryId` rather than file position. A host opts in by calling
   `entry_tree::EntryTree` directly; nothing in the linear path depends on it.

This corresponds to feature gap **E1** in
`docs/runtime-comparison/feature-gaps.md`, modeled on pi's in-place entry
tree (`docs/runtime-comparison/pi.md` §4.3).

## Why a second shape

The linear log answers "what happened, in order." It cannot answer "what did
we try before we backed up and redid this turn" (there is no second path to
point to) or "resume from three turns ago, on a *new* line, without losing
the abandoned attempt." Both are ordinary desktop-assistant needs: retry a
turn, compare two continuations, or let a user "rewind" without deleting
what they rewound past. The entry tree makes every one of those an
in-database operation instead of a file-copy or a lost branch.

## Data model

```
EntryTree::new(workspace_dir, session_id) -> EntryTree
```

Every node is an `Entry`:

```rust
pub struct Entry {
    pub id: EntryId,             // "{session_id}:{ordinal}", deterministic
    pub parent_id: Option<EntryId>, // None only for the session root
    pub ordinal: u64,            // monotonic per-session sequence
    pub kind: EntryKind,
    pub ts: String,              // RFC-3339
}
```

`EntryKind` covers five node shapes:

| Kind             | Carries                                                        | Purpose |
|------------------|-----------------------------------------------------------------|---------|
| `Message`        | a `transcript::TranscriptMessage`                                | an ordinary conversation turn |
| `Compaction`      | `summary`, `first_kept_entry_id`, `tokens_before`, `usage`, `details` | durable record of a context reduction |
| `BranchSummary`   | `from_id`, `summary`                                             | a note on the abandoned path at a navigation point |
| `Label`           | `name`                                                            | a named bookmark on a tip |
| `Custom`          | `kind`, `payload`, `display`                                     | a host-defined out-of-band record |

A session is not one list but a *tree*: any entry may have more than one
child, each child starting a new branch. A **tip** is any entry with no
children; [`EntryTree::tips`] lists them all. A **branch** is a named tip
(`EntryTree::label`/`EntryTree::branches`) — the tree equivalent of a git
branch pointer, except the pointer never moves once written (labeling
appends a new `Label` entry rather than mutating one in place, so the label
history itself is preserved).

## Appending

Two entry points:

- `append(parent_id: Option<&EntryId>, kind)` — explicit parent. `None` is
  valid only for a session's first entry.
- `append_to_head(kind)` — parent defaults to the session's current head
  (the entry with the greatest ordinal). A caller that only ever calls this
  produces a plain, non-branching chain, identical in shape to the pre-tree
  transcript — this is what "existing linear append still works" means in
  practice: nothing about the tree forces branching, it only makes branching
  possible.

## Context projection: `build_context`

```rust
pub fn build_context(&self, tip: &EntryId) -> Result<Vec<Message>>
```

Projects one tip's ancestor chain (root → tip) to a model-ready message
list. The rule:

1. Walk the chain looking for the **newest** `Compaction` entry on the path
   to `tip` (closest to the tip, since a chain may cross more than one
   compaction over its lifetime).
2. If found: the result starts with **one synthesized `Message::System`**
   whose text is the compaction's `summary`, followed by every entry from
   `first_kept_entry_id` (inclusive) to `tip`, in chronological order.
   Nothing older than the compaction is ever included — this is the "context
   never reads past a compaction" invariant from pi's harness design.
3. If not found: every entry from the root is kept.

`Label` and `BranchSummary` entries are tree bookkeeping, not conversation
content, and are always skipped. `Custom` entries become `Message::Custom`.
`Message` entries convert through a best-effort, documented mapping (below).

Context projection reads the `branch_entries` materialized index when it has
rows for the tip, and falls back to a live `parent_id` walk otherwise — see
[Index](#the-branch_entries-index).

### `TranscriptMessage` → `Message` conversion

`transcript::TranscriptMessage` has no structured content blocks, tool-call
id, or typed role beyond a bare string, so the conversion is intentionally
lossy at the edges: `system`/`user`/`assistant` map to their typed
counterparts as one text content block (an assistant entry's `tool_calls`
are always empty — a durable transcript row does not currently carry them
structurally). Any other role, including `tool` (no `tool_call_id` is
recoverable from a bare row), is carried through as
`Message::Custom{kind: "legacy:{role}", payload: {"content": ...}, display:
Some(content)}` so no content is silently dropped. A host that needs full
tool-call fidelity in a projected context should keep the tool call id on
the entry itself via `EntryKind::Custom` instead of `EntryKind::Message`.

## Fork semantics

```rust
pub fn fork(&self, tip: &EntryId, fork: Fork) -> Result<EntryId>
```

`Fork { scope, position }`:

- `position: At` — the fork point is `tip` itself.
- `position: Before` — the fork point is `tip`'s parent (drops `tip` from
  the new branch). Calling this on the session root is an error: the root
  has no parent to fork before.
- `scope: Branch` — **no entries are copied.** The returned `EntryId` is an
  *existing* node — the fork point. Appending to it (via `append(Some(&id),
  ...)`) grows a new sibling subtree in place; this is the cheap, common
  case (branch/retry-from-here).
- `scope: Tree` — the entire root-to-fork-point ancestor chain is
  **duplicated** as brand-new entries (new ids via the same ordinal
  allocator, same `kind`/payload), and the returned id is the copy of the
  fork point. Nothing reachable from the *original* tip changes. Use this
  when the caller needs a history that is independently addressable — for
  example, before an edit that must not perturb ids another view still
  holds a reference to.

`Tree` fork is `O(depth)` — this mirrors the same tradeoff noted in
`docs/runtime-comparison/pi.md` §5 for pi's own SQLite branch cache (an
admitted `O(history)` copy on first divergence): a path copy is the only way
to get independently addressable history without a copy-on-write entry
representation, and depth is bounded by the length of one conversation, not
by the number of branches.

Labels (`EntryTree::label`) never copy either — a label is a `Label` entry
appended as a child of the target tip, and the label→tip mapping in
`entry_tree_labels` is repointed to the new entry. Older labels sharing an
ancestor are unaffected.

## Legacy import

Pre-tree data has no parent pointer. `entry_tree::legacy` derives one
deterministically for both sources this crate already produces:

- `legacy::from_transcript(session_id, &SessionTranscript)` /
  `legacy::from_messages(session_id, &[TranscriptMessage])` — JSONL message
  array, in file order.
- `legacy::from_session_messages(session_id, &[SessionMessage])` — SQLite
  `session_messages` rows, in `id` (insertion) order. The SQL-only columns
  (`model`, token counts, cost, `reasoning_content`) are folded into
  `TranscriptMessage::extra_metadata` so importing loses nothing.

Both assign `EntryId::derive(session_id, ordinal)` (`"{session_id}:{n}"`,
0-based) and chain entry *n* as the parent of entry *n+1* — the only sound
parent assignment for data that is, by construction, a single unbranched
sequence. Because the id is a pure function of `(session_id, ordinal)`,
re-deriving from the same source always produces the same ids.

`EntryTree::import_legacy(&[Entry])` persists the result, skipping any id
that already exists — so importing the same source twice (e.g. a resumed
session re-reading its own transcript) is a no-op rather than a duplicate or
an error.

## The `branch_entries` index

`build_context` and any tip-scoped read need the full root-to-tip chain.
Walking `parent_id` one row at a time is `O(depth)` per call; `branch_entries`
materializes that chain once per known tip:

```
branch_entries(session_id, tip_id, entry_id, ordinal)
```

`EntryTree::rebuild_index()` recomputes it from scratch for every current
tip (every leaf, plus every named branch) by walking `parent_id`. It is a
performance operation, not a correctness prerequisite: `build_context` and
`ancestor_chain` fall back to a live walk whenever a tip has no (or a stale)
index row, so an un-rebuilt or partially stale index never produces a wrong
answer, only a slower one. Call `rebuild_index` after a burst of
branching/forking, or on a schedule, rather than after every single append.

## Schema

Migration 6 (`crates/tinyagents-session/src/migrations.rs`) adds three
tables, additive to the existing session/message/run-ledger schema:

- `entry_tree_entries(session_id, id, parent_id, ordinal, kind, payload_json, ts)`
  — the tree itself; `PRIMARY KEY (session_id, id)`.
- `entry_tree_labels(session_id, name, entry_id)` — branch name → tip.
- `branch_entries(session_id, tip_id, entry_id, ordinal)` — the rebuildable
  index described above.

As with every migration in this crate, these are additive `CREATE TABLE IF
NOT EXISTS` statements; a workspace database that has never used the entry
tree is unaffected until the first `EntryTree` call.

## What this module does not do

- It does not replace `record_message`/the JSONL writer. A host using only
  those two paths never touches `entry_tree` and pays no cost for it beyond
  the (unused) tables migration 6 creates.
- It does not run compaction, choose cut points, or call a summarizer —
  `CompactionEntry` is a place to *record* the result of a compaction
  decided elsewhere (see gap E5 / `docs/runtime-comparison/feature-gaps.md`).
- Tool-call fidelity in `build_context`'s projection of legacy `Message`
  entries is best-effort (see above) — a host that needs exact round-trip of
  tool calls should model them as `EntryKind::Custom` from the start rather
  than relying on the `TranscriptMessage` conversion.
