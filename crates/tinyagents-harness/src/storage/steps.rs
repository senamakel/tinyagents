//! The v1 -> v2 migration steps of [`super::storage_schema`]. Each is a pure
//! function of its input, so a resumed run produces the same output.

use serde_json::{Value, json};
use tinyagents_storage_kit::{Key, Result, SourceRecord, Step, TargetRecord};

use super::{ATTACHMENT_STASH, EMBEDDING_CACHE, JOURNAL, RUN_STATUS, SCHEMA_VERSION};
use crate::observability::{journal_entry_from_v1, status_record};

/// The blob prefixes that move under `harness/`: `(v1 prefix, v2 prefix)`.
pub(super) const BLOB_PREFIXES: [(&str, &str); 5] = [
    ("attachments/", "harness/attachments/"),
    ("tool-results/", "harness/tool_results/"),
    ("outputs/", "harness/outputs/"),
    ("workspace/", "harness/workspace/"),
    ("generated-media/", "harness/generated_media/"),
];

/// A target record that keeps the source's first-write time when it had one.
fn target(collection: &str, id: String, body: Value, source: &SourceRecord) -> TargetRecord {
    let record = TargetRecord::new(collection, id, body);
    if source.meta.created_ms > 0 {
        record.created_ms(source.meta.created_ms)
    } else {
        record
    }
}

/// `harness_run_status` -> `harness.run_status`: unwrap `record`, snake_case
/// body, seconds to milliseconds, id = run id. An undecodable record is dropped.
pub(super) fn run_status() -> Step {
    Step::new("run_status_v2", RUN_STATUS, 1, SCHEMA_VERSION, |batch| {
        Ok(batch
            .iter()
            .filter_map(|source| {
                let (id, body) = status_record::from_v1(&source.body)?;
                Some(target(RUN_STATUS, id, body, source))
            })
            .collect())
    })
    .source("harness_run_status")
}

/// `harness_embedding_cache` -> `harness.embedding_cache`: the body is already
/// `{sig, key, vector}`; the id becomes `Key::of([sig, key])`.
pub(super) fn embedding_cache() -> Step {
    Step::new(
        "embedding_cache_v2",
        EMBEDDING_CACHE,
        1,
        SCHEMA_VERSION,
        |batch| {
            Ok(batch
                .iter()
                .filter_map(|source| {
                    let sig = source.body.get("sig")?.as_str()?;
                    let key = source.body.get("key")?.as_str()?;
                    let id = Key::of([sig, key]).ok()?.as_str().to_owned();
                    Some(target(EMBEDDING_CACHE, id, source.body.clone(), source))
                })
                .collect())
        },
    )
    .source("harness_embedding_cache")
}

/// `attachment_stash_index` -> `harness.attachment_stash`: `touchedAtMs` becomes
/// `touched_ms` and `key` the blob's new `harness/attachments/<id>` key.
pub(super) fn attachment_stash() -> Step {
    Step::new(
        "attachment_stash_v2",
        ATTACHMENT_STASH,
        1,
        SCHEMA_VERSION,
        |batch| {
            Ok(batch
                .iter()
                .filter_map(|source| {
                    let body = &source.body;
                    let id = body.get("id")?.as_str()?;
                    let record = json!({
                        "id": id,
                        "key": format!("harness/attachments/{id}"),
                        "mime": body.get("mime")?.as_str()?,
                        "len": body.get("len")?.as_u64()?,
                        "touched_ms": body.get("touchedAtMs")?.as_u64()?,
                    });
                    Some(target(ATTACHMENT_STASH, source.id.clone(), record, source))
                })
                .collect::<Vec<_>>())
        },
    )
    .source("attachment_stash_index")
}

/// Move the blobs under the v1 prefix `from` to the v2 prefix `to`, unchanged.
pub(super) fn blobs(id: &'static str, from: &'static str, to: &'static str) -> Step {
    Step::blobs(
        id,
        to,
        1,
        SCHEMA_VERSION,
        move |key, content_type, bytes| {
            let rest = key.strip_prefix(from).unwrap_or(&key).to_owned();
            Result::Ok(Some((format!("{to}{rest}"), content_type, bytes)))
        },
    )
    .source_prefix(from)
}

/// The v1 journal stream family: the session crate's `session_journal/` prefix
/// behind the length prefix `DriverAppendStore` used to write.
const V1_JOURNAL_PREFIX: &str = "16:session_journal/";

/// `16:session_journal/<run>` -> `harness.journal/<run>`: each entry loses the
/// run id, root run id and event id the stream name and counter repeat.
///
/// Every entry keeps its place, so offsets, and with them the event ids
/// `{run}-evt-{offset}`, are unchanged. An entry that is not an observation
/// of this run is copied as it is: dropping it would shift every later offset.
pub(super) fn run_journal() -> Step {
    Step::stream(
        "run_journal_v2",
        JOURNAL,
        1,
        SCHEMA_VERSION,
        |source, entries| {
            let Some(run) = source.strip_prefix(V1_JOURNAL_PREFIX) else {
                return Ok(None);
            };
            let entries = entries
                .into_iter()
                .map(|entry| journal_entry_from_v1(run, entry))
                .collect();
            Ok(Some((format!("{JOURNAL}/{run}"), entries)))
        },
    )
    .source_prefix(V1_JOURNAL_PREFIX)
}
