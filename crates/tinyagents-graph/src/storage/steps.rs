//! The v1 -> v2 migration steps of [`super::storage_schema`]. Each is a pure
//! function of its input, so a resumed run produces the same output.

use serde_json::Value;
use tinyagents_storage_kit::{SourceRecord, Step, TargetRecord};

use super::types::{
    CheckpointBody, ThreadBody, WritesBody, checkpoint_id_key, subgraph_key, thread_id_key,
    writes_id_key,
};
use super::{CHECKPOINTS, THREADS, WRITES};
use crate::checkpoint::PendingWrite;

/// One v1 family: the namespace it becomes and its three legacy collections.
pub(super) struct Legacy {
    pub(super) ns: &'static str,
    pub(super) checkpoints: &'static str,
    pub(super) threads: &'static str,
    pub(super) writes: &'static str,
    pub(super) leases: &'static str,
}

/// Every prefix `DriverCheckpointer` wrote under: its default (`graph`), the
/// delegation graph's (`delegation_graph`) and workflows' (`flows_graph`).
pub(super) const LEGACY: [Legacy; 3] = [
    Legacy {
        ns: "graph",
        checkpoints: "graph_checkpoints",
        threads: "graph_threads",
        writes: "graph_writes",
        leases: "graph_leases",
    },
    Legacy {
        ns: "delegation",
        checkpoints: "delegation_graph_checkpoints",
        threads: "delegation_graph_threads",
        writes: "delegation_graph_writes",
        leases: "delegation_graph_leases",
    },
    Legacy {
        ns: "flows",
        checkpoints: "flows_graph_checkpoints",
        threads: "flows_graph_threads",
        writes: "flows_graph_writes",
        leases: "flows_graph_leases",
    },
];

/// The step ids per legacy family, in [`LEGACY`] order.
const CHECKPOINT_STEPS: [&str; 3] = [
    "checkpoints_graph_v2",
    "checkpoints_delegation_v2",
    "checkpoints_flows_v2",
];
const THREAD_STEPS: [&str; 3] = [
    "threads_graph_v2",
    "threads_delegation_v2",
    "threads_flows_v2",
];
const WRITE_STEPS: [&str; 3] = ["writes_graph_v2", "writes_delegation_v2", "writes_flows_v2"];

/// The steps of the three collections. A collection takes one step per legacy
/// prefix and a step chain cannot fork, so the chain runs
/// `1 -> 2 -> 3 -> 4`: the declared version counts the prefixes, not four
/// reshapes of the body.
pub(super) fn all() -> Vec<Step> {
    let mut steps = Vec::new();
    for (index, legacy) in LEGACY.iter().enumerate() {
        let from = 1 + u32::try_from(index).unwrap_or(0);
        steps.push(checkpoints(CHECKPOINT_STEPS[index], from, legacy));
    }
    for (index, legacy) in LEGACY.iter().enumerate() {
        let from = 1 + u32::try_from(index).unwrap_or(0);
        steps.push(threads(THREAD_STEPS[index], from, legacy));
    }
    for (index, legacy) in LEGACY.iter().enumerate() {
        let from = 1 + u32::try_from(index).unwrap_or(0);
        steps.push(writes(WRITE_STEPS[index], from, legacy));
    }
    steps
}

/// A target that keeps the time the record was first written, when known.
fn target(
    collection: &str,
    id: String,
    body: &impl serde::Serialize,
    created_ms: i64,
) -> Option<TargetRecord> {
    let record = TargetRecord::new(collection, id, serde_json::to_value(body).ok()?);
    Some(if created_ms > 0 {
        record.created_ms(created_ms)
    } else {
        record
    })
}

fn str_field(body: &Value, field: &str) -> Option<String> {
    body.get(field)?.as_str().map(str::to_owned)
}

/// The subgraph namespace of a v1 row: an array of segments.
fn namespace_of(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter_map(|part| part.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

fn checkpoints(id: &'static str, from: u32, legacy: &'static Legacy) -> Step {
    Step::new(id, CHECKPOINTS, from, from + 1, move |batch| {
        Ok(batch
            .iter()
            .filter_map(|source| checkpoint_row(legacy.ns, source))
            .collect())
    })
    .source(legacy.checkpoints)
}

fn checkpoint_row(ns: &str, source: &SourceRecord) -> Option<TargetRecord> {
    let thread = str_field(&source.body, "thread")?;
    let record = source.body.get("record")?.clone();
    if !record.is_object() {
        return None;
    }
    let seq = source.body.get("seq")?.as_u64()?;
    let body = CheckpointBody {
        ns: ns.to_owned(),
        subgraph: subgraph_key(&namespace_of(record.get("namespace"))),
        checkpoint_id: str_field(&source.body, "checkpoint_id")?,
        thread: thread.clone(),
        seq,
        checkpoint: record.clone(),
    };
    let written = record
        .get("created_at")
        .and_then(Value::as_i64)
        .filter(|ms| *ms > 0)
        .unwrap_or(source.meta.created_ms);
    target(
        CHECKPOINTS,
        checkpoint_id_key(ns, &thread, seq),
        &body,
        written,
    )
}

fn threads(id: &'static str, from: u32, legacy: &'static Legacy) -> Step {
    Step::new(id, THREADS, from, from + 1, move |batch| {
        Ok(batch
            .iter()
            .filter_map(|source| {
                let thread = str_field(&source.body, "thread")?;
                let body = ThreadBody {
                    ns: legacy.ns.to_owned(),
                    next: source.body.get("next")?.as_u64()?,
                    thread: thread.clone(),
                };
                target(
                    THREADS,
                    thread_id_key(legacy.ns, &thread),
                    &body,
                    source.meta.created_ms,
                )
            })
            .collect())
    })
    .source(legacy.threads)
}

fn writes(id: &'static str, from: u32, legacy: &'static Legacy) -> Step {
    Step::new(id, WRITES, from, from + 1, move |batch| {
        Ok(batch
            .iter()
            .filter_map(|source| {
                let thread = str_field(&source.body, "thread")?;
                let checkpoint_id = str_field(&source.body, "checkpoint_id")?;
                let subgraph = subgraph_key(&namespace_of(source.body.get("namespace")));
                let writes: Vec<PendingWrite> =
                    serde_json::from_value(source.body.get("writes")?.clone()).ok()?;
                let id = writes_id_key(legacy.ns, &thread, &subgraph, &checkpoint_id);
                let body = WritesBody {
                    ns: legacy.ns.to_owned(),
                    thread,
                    subgraph,
                    checkpoint_id,
                    writes,
                };
                target(WRITES, id, &body, source.meta.created_ms)
            })
            .collect())
    })
    .source(legacy.writes)
}
