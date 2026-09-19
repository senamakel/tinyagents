//! The process-wide named reducer registry backing [`crate::BinaryAggregate`]
//! channels.
//!
//! A [`crate::BinaryAggregate`] channel's merge rule is a closure, which is
//! not serializable — but a durable [`crate::Checkpoint`] must be able to
//! decode a `ChannelState` with no context beyond the bytes on disk (it
//! implements plain `DeserializeOwned`, not a seeded deserialize). This
//! registry is the bridge: a reducer is registered once under a stable name
//! (the built-ins below, or a caller's own via
//! [`crate::GraphBuilder::register_reducer`]), the channel's
//! [`crate::Channel::config`] persists only that *name*, and decoding a
//! `binary_aggregate` channel looks the closure back up by name — see
//! [`crate::BinaryAggregate::named`] and `channel_from_config` in `mod.rs`.
//!
//! The registry is global (not scoped to one [`crate::GraphBuilder`]) because
//! decoding happens with no builder in scope at all — only the checkpoint
//! bytes. A name registered anywhere in the process is visible to every
//! decode; decoding a name nobody has registered fails with
//! `TinyAgentsError::Checkpoint("unknown reducer ...")` rather than silently
//! losing the reducer's behavior.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use serde_json::{Value, json};

use crate::{Result, TinyAgentsError};

pub(crate) type ReduceFn = Arc<dyn Fn(Value, Value) -> Result<Value> + Send + Sync>;

fn numeric_add(a: &Value, b: &Value) -> Result<Value> {
    let err = || TinyAgentsError::Graph("`sum` reducer requires numeric values".to_string());
    if a.is_i64() && b.is_i64() {
        return Ok(Value::from(a.as_i64().unwrap() + b.as_i64().unwrap()));
    }
    let sum = a.as_f64().ok_or_else(err)? + b.as_f64().ok_or_else(err)?;
    Ok(Value::from(sum))
}

fn storage() -> &'static Mutex<HashMap<String, ReduceFn>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, ReduceFn>>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut map: HashMap<String, ReduceFn> = HashMap::new();
        map.insert(
            "append".to_string(),
            Arc::new(|current: Value, incoming: Value| {
                let mut list = match current {
                    Value::Array(items) => items,
                    other => vec![other],
                };
                match incoming {
                    Value::Array(items) => list.extend(items),
                    other => list.push(other),
                }
                Ok(Value::Array(list))
            }),
        );
        map.insert(
            "last".to_string(),
            Arc::new(|_current: Value, incoming: Value| Ok(incoming)),
        );
        map.insert(
            "sum".to_string(),
            Arc::new(|current: Value, incoming: Value| numeric_add(&current, &incoming)),
        );
        map.insert(
            "max".to_string(),
            Arc::new(|current: Value, incoming: Value| {
                Ok(if incoming.as_f64() > current.as_f64() {
                    incoming
                } else {
                    current
                })
            }),
        );
        map.insert(
            "min".to_string(),
            Arc::new(|current: Value, incoming: Value| {
                Ok(if incoming.as_f64() < current.as_f64() {
                    incoming
                } else {
                    current
                })
            }),
        );
        map.insert(
            "set_union".to_string(),
            Arc::new(|current: Value, incoming: Value| {
                let mut list = match current {
                    Value::Array(items) => items,
                    other => vec![other],
                };
                let incoming = match incoming {
                    Value::Array(items) => items,
                    other => vec![other],
                };
                for item in incoming {
                    if !list.contains(&item) {
                        list.push(item);
                    }
                }
                Ok(Value::Array(list))
            }),
        );
        Mutex::new(map)
    })
}

/// The process-wide named registry of [`crate::BinaryAggregate`] reducer
/// closures. See the module docs for why this exists and why it is global.
///
/// Pre-registered built-ins: `"append"`, `"last"`, `"sum"`, `"max"`,
/// `"min"`, `"set_union"`.
pub struct ReducerRegistry;

impl ReducerRegistry {
    /// Registers `f` under `name`, overwriting any previous registration of
    /// that name (including a built-in). Prefer
    /// [`crate::GraphBuilder::register_reducer`], which delegates here.
    pub fn register(
        name: impl Into<String>,
        f: impl Fn(Value, Value) -> Result<Value> + Send + Sync + 'static,
    ) {
        let mut guard = storage().lock().unwrap_or_else(|poison| poison.into_inner());
        guard.insert(name.into(), Arc::new(f));
    }

    /// Looks up the reducer registered under `name`.
    pub(crate) fn get(name: &str) -> Option<ReduceFn> {
        let guard = storage().lock().unwrap_or_else(|poison| poison.into_inner());
        guard.get(name).cloned()
    }

    /// Looks up `name`, producing the standard
    /// `TinyAgentsError::Checkpoint("unknown reducer ...")` error a
    /// checkpoint decode raises for a name nobody registered.
    pub(crate) fn require(name: &str) -> Result<ReduceFn> {
        Self::get(name).ok_or_else(|| {
            TinyAgentsError::Checkpoint(format!(
                "unknown reducer `{name}`: no closure is registered under this name in this \
                 process (register it with GraphBuilder::register_reducer before decoding this \
                 checkpoint)"
            ))
        })
    }

    /// The `{"reducer": name}` config payload for a named `BinaryAggregate`
    /// channel, or `{"reducer": null}` for an unnamed one (which
    /// [`crate::channel::channel_from_config`] then rejects on decode).
    pub(crate) fn config_for(name: Option<&str>) -> Value {
        json!({ "reducer": name })
    }
}
