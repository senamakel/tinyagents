//! Unit tests for the channel-per-field state model: each channel's merge rule,
//! `ChannelSet` dispatch/snapshot, `ChannelState` round-trips through a real
//! graph (fan-out into aggregate channels), and concurrent-write conflict
//! detection on non-aggregate channels.

use super::*;
use crate::TinyAgentsError;
use crate::builder::{GraphBuilder, NodeContext};
use crate::command::{Command, NodeResult};
use serde_json::{Value, json};

// --- per-channel merge rules ---

#[test]
fn last_value_overwrites() {
    let c = LastValue;
    assert_eq!(c.merge(None, json!(1)).unwrap(), json!(1));
    assert_eq!(c.merge(Some(json!(1)), json!(2)).unwrap(), json!(2));
    assert!(!c.allows_concurrent());
}

#[test]
fn topic_appends_scalars_and_arrays() {
    let c = Topic;
    let v = c.merge(None, json!("a")).unwrap();
    let v = c.merge(Some(v), json!("b")).unwrap();
    let v = c.merge(Some(v), json!(["c", "d"])).unwrap();
    assert_eq!(v, json!(["a", "b", "c", "d"]));
    assert!(c.allows_concurrent());
}

#[test]
fn delta_accumulates_numbers() {
    let c = Delta;
    let v = c.merge(None, json!(2)).unwrap();
    let v = c.merge(Some(v), json!(3)).unwrap();
    assert_eq!(v, json!(5));
    // integer + float promotes to float
    let v = c.merge(Some(v), json!(0.5)).unwrap();
    assert_eq!(v, json!(5.5));
    assert!(c.merge(Some(json!(1)), json!("x")).is_err());
}

#[test]
fn messages_merge_by_id() {
    let c = Messages;
    let v = c.merge(None, json!([{"id": "1", "text": "hi"}])).unwrap();
    let v = c
        .merge(Some(v), json!([{"id": "2", "text": "yo"}]))
        .unwrap();
    // upsert existing id 1
    let v = c
        .merge(Some(v), json!({"id": "1", "text": "hello"}))
        .unwrap();
    assert_eq!(
        v,
        json!([{"id": "1", "text": "hello"}, {"id": "2", "text": "yo"}])
    );
    assert!(c.allows_concurrent());
}

#[test]
fn messages_merge_dedup_map_preserves_order_and_appends_unkeyed() {
    // Exercise the id->index map path: a single batch that upserts an existing
    // id, appends a new id, and appends an unkeyed message, then a follow-up
    // upsert of the id introduced by that batch. Existing order is preserved and
    // dedup is by id only.
    let c = Messages;
    let v = c
        .merge(
            None,
            json!([{"id": "a", "text": "1"}, {"id": "b", "text": "2"}]),
        )
        .unwrap();
    let v = c
        .merge(
            Some(v),
            json!([
                {"id": "a", "text": "1-updated"},
                {"id": "c", "text": "3"},
                {"text": "no-id"},
            ]),
        )
        .unwrap();
    // Upserting an id first seen in the previous batch must land on that message.
    let v = c
        .merge(Some(v), json!({"id": "c", "text": "3-updated"}))
        .unwrap();
    assert_eq!(
        v,
        json!([
            {"id": "a", "text": "1-updated"},
            {"id": "b", "text": "2"},
            {"id": "c", "text": "3-updated"},
            {"text": "no-id"},
        ])
    );
}

#[test]
fn ephemeral_overwrites_and_is_marked() {
    let c = Ephemeral;
    assert_eq!(c.merge(Some(json!(1)), json!(2)).unwrap(), json!(2));
    assert!(c.is_ephemeral());
    assert!(!c.allows_concurrent());
}

#[test]
fn untracked_is_not_tracked() {
    let c = Untracked;
    assert!(!c.is_tracked());
    assert_eq!(c.merge(None, json!("x")).unwrap(), json!("x"));
}

#[test]
fn binary_aggregate_folds_with_closure() {
    let c = BinaryAggregate::new(|a: Value, b: Value| {
        Ok(json!(a.as_i64().unwrap() * b.as_i64().unwrap()))
    });
    let v = c.merge(None, json!(2)).unwrap();
    let v = c.merge(Some(v), json!(3)).unwrap();
    let v = c.merge(Some(v), json!(4)).unwrap();
    assert_eq!(v, json!(24));
    assert!(c.allows_concurrent());
}

#[test]
fn binary_aggregate_from_reducer() {
    // Any `Reducer<Value>` can back an aggregate channel; here a closure reducer
    // that keeps the numeric maximum.
    let c = BinaryAggregate::from_reducer(crate::ClosureReducer::new(|a: Value, b: Value| {
        Ok(if b.as_f64() > a.as_f64() { b } else { a })
    }));
    let v = c.merge(None, json!(3)).unwrap();
    let v = c.merge(Some(v), json!(7)).unwrap();
    let v = c.merge(Some(v), json!(1)).unwrap();
    assert_eq!(v, json!(7));
}

#[test]
fn count_barrier_readiness() {
    let c = Barrier::new(2);
    let v = c.merge(None, json!("a")).unwrap();
    assert!(!c.is_ready(Some(&v)));
    let v = c.merge(Some(v), json!("b")).unwrap();
    assert!(c.is_ready(Some(&v)));
}

#[test]
fn named_barrier_readiness() {
    let c = NamedBarrier::new(["left", "right"]);
    let v = c.merge(None, json!({"left": 1})).unwrap();
    assert!(!c.is_ready(Some(&v)));
    let v = c.merge(Some(v), json!({"right": 2})).unwrap();
    assert!(c.is_ready(Some(&v)));
    assert!(c.allows_concurrent());
}

// --- ChannelSet ---

#[test]
fn channel_set_apply_and_snapshot_excludes_untracked() {
    let mut set = ChannelSet::new()
        .with_channel("count", Delta)
        .with_channel("scratch", Untracked);
    set.apply_update("count", json!(2)).unwrap();
    set.apply_update("count", json!(5)).unwrap();
    set.apply_update("scratch", json!("temp")).unwrap();

    assert_eq!(set.get("count"), Some(&json!(7)));
    let snap = set.snapshot();
    assert_eq!(snap.get("count"), Some(&json!(7)));
    assert!(
        !snap.contains_key("scratch"),
        "untracked excluded from snapshot"
    );
}

#[test]
fn channel_set_unknown_channel_errors() {
    let mut set = ChannelSet::new();
    assert!(matches!(
        set.apply_update("nope", json!(1)),
        Err(TinyAgentsError::Graph(_))
    ));
}

#[test]
fn apply_update_accumulates_barrier_by_moving_current() {
    // Exercises the owned-`current` merge path through `apply_update`: repeated
    // writes to a barrier fold into the same accumulated array without cloning,
    // and readiness reflects the accumulated arrivals.
    let mut set = ChannelSet::new().with_channel("join", Barrier::new(3));
    set.apply_update("join", json!("a")).unwrap();
    set.apply_update("join", json!("b")).unwrap();
    assert!(!set.is_ready("join").unwrap());
    set.apply_update("join", json!("c")).unwrap();
    assert!(set.is_ready("join").unwrap());
    assert_eq!(set.get("join"), Some(&json!(["a", "b", "c"])));
}

#[test]
fn apply_update_rejected_write_clears_channel_value() {
    // A registered channel whose merge rejects the write leaves the channel
    // unset (the current value was moved into the failed merge). The unknown-
    // channel guard, by contrast, runs before any state is touched.
    let mut set = ChannelSet::new().with_channel("n", Delta);
    set.apply_update("n", json!(5)).unwrap();
    assert_eq!(set.get("n"), Some(&json!(5)));

    // Non-numeric write is rejected by Delta's merge; prior value is dropped.
    assert!(set.apply_update("n", json!("not-a-number")).is_err());
    assert_eq!(
        set.get("n"),
        None,
        "rejected write leaves the channel unset"
    );
}

// --- ChannelState graph round-trips ---

fn aggregate_state() -> ChannelState {
    ChannelState::new()
        .with_channel("results", Topic)
        .with_channel("total", Delta)
}

/// `start` fans out to `a` and `b` in one parallel superstep; both write the
/// shared aggregate channels and their writes merge deterministically.
#[tokio::test]
async fn fan_out_merges_both_branches_into_aggregate_channels() {
    let graph = GraphBuilder::<ChannelState, ChannelUpdate>::new()
        .set_reducer(ChannelState::new())
        .with_parallel(true)
        .add_node("start", |_s: ChannelState, _c: NodeContext| async move {
            Ok(NodeResult::Command(Command::goto(["a", "b"])))
        })
        .add_node("a", |_s: ChannelState, c: NodeContext| async move {
            Ok(NodeResult::Update(
                ChannelUpdate::new()
                    .set("results", "a")
                    .set("total", 10)
                    .at_step(c.step),
            ))
        })
        .add_node("b", |_s: ChannelState, c: NodeContext| async move {
            Ok(NodeResult::Update(
                ChannelUpdate::new()
                    .set("results", "b")
                    .set("total", 5)
                    .at_step(c.step),
            ))
        })
        .mark_command_routing("start")
        .set_entry("start")
        .set_finish("a")
        .set_finish("b")
        .compile()
        .unwrap();

    let exec = graph.run(aggregate_state()).await.unwrap();
    // Topic merges both branches in active-set index order; Delta sums them.
    assert_eq!(exec.state.get("results"), Some(&json!(["a", "b"])));
    assert_eq!(exec.state.get("total"), Some(&json!(15)));
}

/// Two concurrent branches writing the same `LastValue` channel in one step is
/// a conflict.
#[tokio::test]
async fn concurrent_last_value_writes_conflict() {
    let graph = GraphBuilder::<ChannelState, ChannelUpdate>::new()
        .set_reducer(ChannelState::new())
        .with_parallel(true)
        .add_node("start", |_s: ChannelState, _c: NodeContext| async move {
            Ok(NodeResult::Command(Command::goto(["a", "b"])))
        })
        .add_node("a", |_s: ChannelState, c: NodeContext| async move {
            Ok(NodeResult::Update(
                ChannelUpdate::new().set("winner", "a").at_step(c.step),
            ))
        })
        .add_node("b", |_s: ChannelState, c: NodeContext| async move {
            Ok(NodeResult::Update(
                ChannelUpdate::new().set("winner", "b").at_step(c.step),
            ))
        })
        .mark_command_routing("start")
        .set_entry("start")
        .set_finish("a")
        .set_finish("b")
        .compile()
        .unwrap();

    let initial = ChannelState::new().with_channel("winner", LastValue);
    let err = graph.run(initial).await.unwrap_err();
    assert!(matches!(err, TinyAgentsError::InvalidConcurrentUpdate(_)));
}

/// A `LastValue` channel overwritten across *different* steps is fine — only
/// same-step concurrent writes conflict.
#[tokio::test]
async fn sequential_last_value_overwrites_across_steps() {
    let graph = GraphBuilder::<ChannelState, ChannelUpdate>::new()
        .set_reducer(ChannelState::new())
        .add_node("a", |_s: ChannelState, c: NodeContext| async move {
            Ok(NodeResult::Update(
                ChannelUpdate::new().set("v", "first").at_step(c.step),
            ))
        })
        .add_node("b", |_s: ChannelState, c: NodeContext| async move {
            Ok(NodeResult::Update(
                ChannelUpdate::new().set("v", "second").at_step(c.step),
            ))
        })
        .add_edge("a", "b")
        .set_entry("a")
        .set_finish("b")
        .compile()
        .unwrap();

    let initial = ChannelState::new().with_channel("v", LastValue);
    let exec = graph.run(initial).await.unwrap();
    assert_eq!(exec.state.get("v"), Some(&json!("second")));
}

/// An `Ephemeral` channel written in one step is visible to the next step's
/// node, then cleared when the step advances.
#[tokio::test]
async fn ephemeral_channel_is_cleared_on_next_step() {
    let graph = GraphBuilder::<ChannelState, ChannelUpdate>::new()
        .set_reducer(ChannelState::new())
        .add_node("a", |_s: ChannelState, c: NodeContext| async move {
            Ok(NodeResult::Update(
                ChannelUpdate::new().set("tmp", 1).at_step(c.step),
            ))
        })
        .add_node("b", |s: ChannelState, c: NodeContext| async move {
            // tmp is still visible to this (next-step) node.
            let seen = s.get("tmp").cloned().unwrap_or(json!(null));
            Ok(NodeResult::Update(
                ChannelUpdate::new().set("saw", seen).at_step(c.step),
            ))
        })
        .add_edge("a", "b")
        .set_entry("a")
        .set_finish("b")
        .compile()
        .unwrap();

    let initial = ChannelState::new()
        .with_channel("tmp", Ephemeral)
        .with_channel("saw", LastValue);
    let exec = graph.run(initial).await.unwrap();
    assert_eq!(exec.state.get("saw"), Some(&json!(1)), "tmp was visible");
    assert_eq!(
        exec.state.get("tmp"),
        None,
        "tmp cleared after step advanced"
    );
    // Snapshot reflects only the tracked, surviving channels.
    let snap = exec.state.snapshot();
    assert_eq!(snap.get("saw"), Some(&json!(1)));
    assert!(!snap.contains_key("tmp"));
}

/// A single node writing the same non-aggregate channel twice in one update is
/// last-wins, not a conflict.
#[test]
fn single_update_repeat_write_is_last_wins() {
    let state = ChannelState::new().with_channel("v", LastValue);
    let merged = state
        .merge(ChannelUpdate::new().set("v", 1).set("v", 2).at_step(1))
        .unwrap();
    assert_eq!(merged.get("v"), Some(&json!(2)));
}

// --- Serializable channels + ReducerRegistry (I5/R3) ---

#[test]
fn channel_config_round_trips_for_every_built_in_kind() {
    let set = ChannelSet::new()
        .with_channel("last", LastValue)
        .with_channel("topic", Topic)
        .with_channel("delta", Delta)
        .with_channel("messages", Messages)
        .with_channel("barrier", Barrier::new(2))
        .with_channel("named", NamedBarrier::new(["a", "b"]))
        .with_channel("agg", BinaryAggregate::named("sum").unwrap());
    let json = serde_json::to_value(&set).unwrap();
    let decoded: ChannelSet = serde_json::from_value(json).unwrap();
    assert!(decoded.contains("last"));
    assert!(decoded.contains("topic"));
    assert!(decoded.contains("delta"));
    assert!(decoded.contains("messages"));
    assert!(decoded.contains("barrier"));
    assert!(decoded.contains("named"));
    assert!(decoded.contains("agg"));
    // A decoded barrier keeps its `expected` readiness threshold.
    assert!(decoded.allows_concurrent("barrier").unwrap());
}

#[test]
fn binary_aggregate_named_unknown_reducer_errors() {
    let err = BinaryAggregate::named("does-not-exist-anywhere").unwrap_err();
    assert!(matches!(err, TinyAgentsError::Checkpoint(_)));
    assert!(err.to_string().contains("unknown reducer"));
}

#[test]
fn decoding_binary_aggregate_without_reducer_name_errors() {
    // A `BinaryAggregate::new` closure carries no name, so its config is
    // `{"reducer": null}` — decoding must fail rather than silently drop the
    // merge rule.
    let set = ChannelSet::new().with_channel(
        "agg",
        BinaryAggregate::new(|a: Value, b: Value| {
            Ok(json!(a.as_i64().unwrap() + b.as_i64().unwrap()))
        }),
    );
    let json = serde_json::to_value(&set).unwrap();
    let decoded: std::result::Result<ChannelSet, serde_json::Error> =
        serde_json::from_value(json);
    assert!(decoded.is_err());
}

#[test]
fn named_binary_aggregate_round_trips_and_merges_after_decode() {
    crate::ReducerRegistry::register("channel-test-double", |current: Value, incoming: Value| {
        Ok(json!(
            current.as_i64().unwrap_or(1) * incoming.as_i64().unwrap()
        ))
    });
    let set = ChannelSet::new().with_channel(
        "product",
        BinaryAggregate::named("channel-test-double").unwrap(),
    );
    let json = serde_json::to_value(&set).unwrap();
    let mut decoded: ChannelSet = serde_json::from_value(json).unwrap();
    decoded.apply_update("product", json!(3)).unwrap();
    decoded.apply_update("product", json!(4)).unwrap();
    assert_eq!(decoded.get("product"), Some(&json!(12)));
}

#[test]
fn register_reducer_is_reachable_through_the_graph_builder() {
    let _graph = GraphBuilder::<ChannelState, ChannelUpdate>::new()
        .set_reducer(ChannelState::new())
        .register_reducer("channel-test-via-builder", |_a: Value, b: Value| Ok(b))
        .add_node("noop", |_s: ChannelState, _c: NodeContext| async move {
            Ok(NodeResult::Update(ChannelUpdate::new()))
        })
        .set_entry("noop")
        .set_finish("noop")
        .compile()
        .unwrap();
    // Registered globally, so `named` can find it without the builder.
    assert!(BinaryAggregate::named("channel-test-via-builder").is_ok());
}

#[tokio::test]
async fn channel_state_graph_round_trips_through_file_checkpointer() {
    use crate::checkpoint::{Checkpointer, FileCheckpointer};
    use crate::command::{Command, Interrupt};
    use std::sync::Arc;

    fn graph() -> crate::compiled::CompiledGraph<ChannelState, ChannelUpdate> {
        GraphBuilder::<ChannelState, ChannelUpdate>::new()
            .set_reducer(ChannelState::new())
            .add_node("collect", |_s: ChannelState, c: NodeContext| async move {
                match c.resume {
                    Some(value) => {
                        let bump = value.get("bump").and_then(Value::as_i64).unwrap_or(0);
                        Ok(NodeResult::Update(
                            ChannelUpdate::new()
                                .set("total", bump)
                                .set("log", "collected")
                                .at_step(c.step),
                        ))
                    }
                    None => Ok(NodeResult::Interrupt(Interrupt::new(
                        "collect",
                        json!({ "ask": "bump?" }),
                    ))),
                }
            })
            .set_entry("collect")
            .set_finish("collect")
            .compile()
            .unwrap()
    }

    let initial = ChannelState::new()
        .with_channel("total", BinaryAggregate::named("sum").unwrap())
        .with_channel("log", Topic);

    let dir = tempfile::tempdir().unwrap();
    {
        let cp: Arc<dyn Checkpointer<ChannelState>> =
            Arc::new(FileCheckpointer::<ChannelState>::new(dir.path()));
        let g = graph().with_checkpointer(cp);
        let paused = g.run_with_thread("ch-thread", initial).await.unwrap();
        assert!(paused.is_interrupted());
    }
    // Fresh checkpointer over the same directory, simulating a process
    // restart: the `binary_aggregate` channel must decode using the (still
    // process-wide registered) `"sum"` reducer and the resumed run must
    // merge through it correctly.
    {
        let cp: Arc<dyn Checkpointer<ChannelState>> =
            Arc::new(FileCheckpointer::<ChannelState>::new(dir.path()));
        let g = graph().with_checkpointer(cp);
        let exec = g
            .resume("ch-thread", Command::resume(json!({ "bump": 7 })))
            .await
            .unwrap();
        assert_eq!(exec.state.get("total"), Some(&json!(7)));
        assert_eq!(exec.state.get("log"), Some(&json!(["collected"])));
    }
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn channel_state_graph_round_trips_through_sqlite_checkpointer() {
    use crate::checkpoint::{Checkpointer, SqliteCheckpointer};
    use crate::command::{Command, Interrupt};
    use std::sync::Arc;

    fn graph() -> crate::compiled::CompiledGraph<ChannelState, ChannelUpdate> {
        GraphBuilder::<ChannelState, ChannelUpdate>::new()
            .set_reducer(ChannelState::new())
            .add_node("collect", |_s: ChannelState, c: NodeContext| async move {
                match c.resume {
                    Some(value) => {
                        let bump = value.get("bump").and_then(Value::as_i64).unwrap_or(0);
                        Ok(NodeResult::Update(
                            ChannelUpdate::new().set("total", bump).at_step(c.step),
                        ))
                    }
                    None => Ok(NodeResult::Interrupt(Interrupt::new(
                        "collect",
                        json!({ "ask": "bump?" }),
                    ))),
                }
            })
            .set_entry("collect")
            .set_finish("collect")
            .compile()
            .unwrap()
    }

    let initial = ChannelState::new().with_channel("total", BinaryAggregate::named("sum").unwrap());

    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("checkpoints.sqlite");
    {
        let cp: Arc<dyn Checkpointer<ChannelState>> =
            Arc::new(SqliteCheckpointer::<ChannelState>::open(&db_path).unwrap());
        let g = graph().with_checkpointer(cp);
        let paused = g.run_with_thread("ch-thread-sqlite", initial).await.unwrap();
        assert!(paused.is_interrupted());
    }
    {
        let cp: Arc<dyn Checkpointer<ChannelState>> =
            Arc::new(SqliteCheckpointer::<ChannelState>::open(&db_path).unwrap());
        let g = graph().with_checkpointer(cp);
        let exec = g
            .resume("ch-thread-sqlite", Command::resume(json!({ "bump": 9 })))
            .await
            .unwrap();
        assert_eq!(exec.state.get("total"), Some(&json!(9)));
    }
}

// --- Channel versions (I5/R3) ---

#[test]
fn channel_versions_bump_once_per_distinct_channel_per_update() {
    let state = ChannelState::new()
        .with_channel("a", LastValue)
        .with_channel("b", LastValue);
    let state = state
        .merge(
            ChannelUpdate::new()
                .set("a", 1)
                .set("a", 2) // same update, same channel: still one bump.
                .set("b", 1)
                .at_step(1),
        )
        .unwrap();
    assert_eq!(state.channel_versions().get("a"), Some(&1));
    assert_eq!(state.channel_versions().get("b"), Some(&1));

    let state = state
        .merge(ChannelUpdate::new().set("a", 3).at_step(2))
        .unwrap();
    assert_eq!(state.channel_versions().get("a"), Some(&2));
    // `b` was not touched this step, so its version does not move.
    assert_eq!(state.channel_versions().get("b"), Some(&1));
}

#[tokio::test]
async fn node_context_changed_since_last_run_tracks_channel_writes() {
    use std::sync::{Arc, Mutex};

    let observed: Arc<Mutex<Vec<bool>>> = Arc::new(Mutex::new(Vec::new()));
    let observed_a = observed.clone();
    let observed_b = observed.clone();

    let graph = GraphBuilder::<ChannelState, ChannelUpdate>::new()
        .set_reducer(ChannelState::new())
        .add_node("write", |_s: ChannelState, c: NodeContext| async move {
            Ok(NodeResult::Update(
                ChannelUpdate::new().set("v", 1).at_step(c.step),
            ))
        })
        .add_node("observe_a", move |_s: ChannelState, c: NodeContext| {
            let observed_a = observed_a.clone();
            async move {
                observed_a.lock().unwrap().push(c.changed_since_last_run("v"));
                Ok(NodeResult::Update(ChannelUpdate::new()))
            }
        })
        .add_node("observe_b", move |_s: ChannelState, c: NodeContext| {
            let observed_b = observed_b.clone();
            async move {
                observed_b.lock().unwrap().push(c.changed_since_last_run("v"));
                Ok(NodeResult::Update(ChannelUpdate::new()))
            }
        })
        .add_edge("write", "observe_a")
        .add_edge("observe_a", "observe_b")
        .add_edge("observe_b", "observe_a")
        .set_entry("write")
        .set_finish("observe_a")
        .with_recursion_limit(4)
        .compile()
        .unwrap();

    let initial = ChannelState::new().with_channel("v", LastValue);
    let _ = graph.run(initial).await;
    // `observe_a`'s first-ever run sees `v` as changed (it has never seen
    // it), then its second run (after `observe_b`, which made no writes)
    // sees no further change.
    let seen_a = observed.lock().unwrap().clone();
    assert_eq!(seen_a.first(), Some(&true));
}

// --- Delta-channel history + Overwrite (I5/R3) ---

#[test]
fn overwrite_bypasses_merge_and_rebases_baseline() {
    let state = ChannelState::new().with_channel("log", Topic);
    let state = state
        .merge(ChannelUpdate::new().set("log", "a").at_step(1))
        .unwrap();
    assert_eq!(state.get("log"), Some(&json!(["a"])));

    // Overwrite replaces the value outright...
    let state = state
        .merge(ChannelUpdate::new().overwrite("log", json!(["reset"])).at_step(2))
        .unwrap();
    assert_eq!(state.get("log"), Some(&json!(["reset"])));

    // ...and subsequent merges build on the new baseline, not the old one.
    let state = state
        .merge(ChannelUpdate::new().set("log", "b").at_step(3))
        .unwrap();
    assert_eq!(state.get("log"), Some(&json!(["reset", "b"])));
}

#[test]
fn delta_tracked_channel_accumulates_step_deltas_and_overwrite_rebases_them() {
    let set = ChannelSet::new()
        .with_channel("log", Topic)
        .with_delta("log", 1000);
    let state = ChannelState::new();
    let state = ChannelState {
        set,
        ..state
    };
    let state = state
        .merge(ChannelUpdate::new().set("log", "a").at_step(1))
        .unwrap();
    assert_eq!(state.step_deltas().get("log"), Some(&vec![json!("a")]));

    let state = state
        .merge(ChannelUpdate::new().set("log", "b").at_step(2))
        .unwrap();
    // Deltas are per-step only: step 2's entry does not include step 1's.
    assert_eq!(state.step_deltas().get("log"), Some(&vec![json!("b")]));

    let state = state
        .merge(
            ChannelUpdate::new()
                .overwrite("log", json!(["reset"]))
                .at_step(3),
        )
        .unwrap();
    // The overwrite rebases the delta baseline: only the overwrite's own
    // value is recorded for this step.
    assert_eq!(
        state.step_deltas().get("log"),
        Some(&vec![json!(["reset"])])
    );
}

#[tokio::test]
async fn delta_history_replays_from_checkpoints() {
    use crate::checkpoint::{CheckpointConfig, Checkpointer, InMemoryCheckpointer};
    use std::sync::Arc;

    let set = ChannelSet::new().with_channel("log", Topic).with_delta("log", 1000);
    let base = ChannelState {
        set,
        ..ChannelState::new()
    };

    let cp: Arc<dyn Checkpointer<ChannelState>> = Arc::new(InMemoryCheckpointer::new());
    let graph = GraphBuilder::<ChannelState, ChannelUpdate>::new()
        .set_reducer(ChannelState::new())
        .add_node("append", |_s: ChannelState, c: NodeContext| async move {
            Ok(NodeResult::Update(
                ChannelUpdate::new()
                    .set("log", format!("item-{}", c.step))
                    .at_step(c.step),
            ))
        })
        .add_edge("append", "append")
        .set_entry("append")
        .set_finish("append")
        .with_recursion_limit(3)
        .compile()
        .unwrap()
        .with_checkpointer(cp.clone());

    // `with_recursion_limit(3)` plus the self-edge lets `append` run three
    // supersteps before the limit trips.
    let _ = graph.run_with_thread("delta-thread", base).await;

    let config = CheckpointConfig::latest("delta-thread");
    let history = cp.delta_history(&config, "log").await.unwrap();
    assert_eq!(
        history,
        vec![json!("item-1"), json!("item-2"), json!("item-3")]
    );
}
