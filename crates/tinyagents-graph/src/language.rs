//! Materialization of declarative language blueprints into executable graphs.
//!
//! This is the bridge from `tinyagents-language`'s parsed [`Blueprint`] (a
//! `.rag` program) to this crate's [`GraphBuilder`]/[`CompiledGraph`]: a host
//! supplies a [`NodeFactory`] that turns each [`NodeSpec`] into a
//! [`BoxedNode`] handler, and [`build_graph`] wires those handlers into a
//! whole-state (`GraphBuilder::overwrite`) graph and compiles it. This module
//! knows nothing about what a node handler actually does — that is entirely
//! the factory's responsibility — only how to assemble the compiled topology
//! around it.
//!
//! # What is lowered
//!
//! [`build_graph`] lowers the entry node, each node's Rust-side handler (via
//! [`NodeFactory`]), and its [`Routing`] (a static edge, command-routing
//! marker, or terminal), plus every other populated `Blueprint`/`NodeSpec`
//! field it can faithfully express against the *generic*, host-owned `State`
//! type — see the module-level docs in each helper below for the exact
//! mapping. Two field groups get real runtime behavior (they change what the
//! compiled graph does):
//!
//! - graph-level `joins` and node-level `join_sources` lower onto
//!   [`GraphBuilder::add_waiting_edge`] — the same barrier/fan-in primitive
//!   hand-written graphs use.
//! - node-level `timeout` and `retry` lower onto
//!   [`GraphBuilder::with_node_timeout`] / [`CompiledGraph::with_node_retry`]
//!   — but only *graph-wide*: this builder has no per-node timeout/retry
//!   policy API, so `build_graph` requires every node that declares one to
//!   declare the *same* one, and fails closed (`TinyAgentsError::Compile`)
//!   when two nodes disagree, naming both. See
//!   `docs/modules/expressive-language/implementation-status.md` for why.
//!
//! Everything else that was previously silently dropped (Phase 1c made these
//! hard-reject instead) is now either genuinely structural (validated against
//! the built topology, e.g. `sends`/`join_sources` targets must be declared
//! nodes) or attached as inert, behavior-free export metadata via
//! [`GraphBuilder::with_node_metadata`]/[`GraphBuilder::mark_interrupt`] —
//! visible to `crate::export`, never silently dropped, but not enforced at
//! run time because the generic `State`/`Update` types give `build_graph` no
//! way to apply a declared literal write or input/output projection without
//! the caller committing to a concrete state shape (see
//! `crate::channel::ChannelState` for the opt-in typed alternative).
//! `channels`/`defaults` remain exactly as documented before this change:
//! accepted, read by `crate::export`, but not wired into the whole-state
//! `overwrite()` reducer this function always uses.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use tinyagents_harness::error::{Result, TinyAgentsError};
use tinyagents_harness::retry::RetryPolicy;
use tinyagents_language::{Blueprint, IoFieldSpec, Literal, NodeSpec, Routing};

use crate::{CompiledGraph, GraphBuilder, NodeHandler};

/// A durable node handler materialized from a declarative node specification.
pub type BoxedNode<State> = Arc<NodeHandler<State, State>>;

/// Builds runtime node handlers from declarative node specifications.
pub trait NodeFactory<State> {
    /// Materializes one executable handler.
    ///
    /// # Errors
    ///
    /// Returns an error when the node kind is unsupported or a required
    /// capability binding is unavailable.
    fn make(&self, spec: &NodeSpec) -> Result<BoxedNode<State>>;
}

/// Renders a [`Literal`] for a metadata value or an error message.
fn literal_display(value: &Literal) -> String {
    value.as_display()
}

/// Validates a graph-level `input`/`output` field list: names must be
/// non-empty and unique. There is no runtime input/output projection in this
/// crate's executor (a node handler receives/returns the whole `State`), so
/// this is a validated no-op rather than a silent drop: a blueprint with a
/// duplicate or malformed field name is rejected instead of deploying with an
/// ambiguous shape nobody enforces.
fn validate_io_fields(context: &str, fields: &[IoFieldSpec]) -> Result<()> {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for field in fields {
        if field.name.trim().is_empty() {
            return Err(TinyAgentsError::Compile(format!(
                "graph `{context}` field has an empty name"
            )));
        }
        if !seen.insert(field.name.as_str()) {
            return Err(TinyAgentsError::Compile(format!(
                "graph `{context}` declares duplicate field `{}`",
                field.name
            )));
        }
    }
    Ok(())
}

/// Parses a `timeout <literal>` value into a [`Duration`].
///
/// Accepts a bare number (seconds) or a string/identifier of the form
/// `<number><unit>` where `unit` is one of `ms`, `s`, `m`, `h` (e.g. `"500ms"`,
/// `"30s"`, `"2m"`, `"1h"`). This is a small, self-contained parser — the
/// language crate's lexer does not tokenize a suffixed literal like `30s` as
/// one token (`3`0` lexes as a number, leaving a stray `s` identifier), so
/// only a quoted string (`timeout "30s"`) or a bare number (`timeout 30`,
/// seconds) reaches here as a single literal; see
/// `docs/modules/expressive-language/implementation-status.md` ("Duration
/// literals like `60s`").
fn parse_duration_literal(raw: &str) -> std::result::Result<Duration, String> {
    let trimmed = raw.trim();
    let split_at = trimmed
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(trimmed.len());
    let (num_part, unit) = trimmed.split_at(split_at);
    let value: f64 = num_part
        .parse()
        .map_err(|_| format!("invalid timeout literal `{raw}`"))?;
    let seconds = match unit {
        "" | "s" => value,
        "ms" => value / 1000.0,
        "m" => value * 60.0,
        "h" => value * 3600.0,
        other => {
            return Err(format!(
                "unsupported timeout unit `{other}` in `{raw}` (expected s, ms, m, or h)"
            ));
        }
    };
    if !seconds.is_finite() || seconds < 0.0 {
        return Err(format!(
            "timeout `{raw}` must be a finite, non-negative duration"
        ));
    }
    Ok(Duration::from_secs_f64(seconds))
}

/// Computes the single graph-wide node timeout implied by every node's
/// declared `timeout`, or `None` if no node declares one.
///
/// This builder has no per-node timeout policy
/// (`GraphBuilder::with_node_timeout` applies to every node), so when two or
/// more nodes declare *different* timeouts there is no faithful lowering:
/// this returns `TinyAgentsError::Compile` naming every disagreeing node
/// instead of silently picking one (last-registered, first-registered, …) or
/// silently dropping the rest.
fn uniform_node_timeout(blueprint: &Blueprint) -> Result<Option<Duration>> {
    let mut declared: Vec<(&str, Duration)> = Vec::new();
    for spec in &blueprint.nodes {
        let Some(raw) = &spec.timeout else { continue };
        let duration = parse_duration_literal(raw).map_err(|message| {
            TinyAgentsError::Compile(format!("node `{}` `timeout`: {message}", spec.name))
        })?;
        declared.push((spec.name.as_str(), duration));
    }
    let Some((_, first)) = declared.first().copied() else {
        return Ok(None);
    };
    let disagreeing: Vec<String> = declared
        .iter()
        .filter(|(_, d)| *d != first)
        .map(|(name, d)| format!("`{name}`={d:?}"))
        .collect();
    if !disagreeing.is_empty() {
        return Err(TinyAgentsError::Compile(format!(
            "per-node timeout not supported yet: node `{}`={:?} disagrees with {}",
            declared[0].0, first, disagreeing.join(", ")
        )));
    }
    Ok(Some(first))
}

/// Reads one `retry { key value … }` entry into the matching [`RetryPolicy`]
/// field, or `Err` naming the offending key/value.
fn apply_retry_entry(
    node: &str,
    policy: &mut RetryPolicy,
    key: &str,
    value: &Literal,
) -> std::result::Result<(), TinyAgentsError> {
    let num = |value: &Literal| -> std::result::Result<f64, TinyAgentsError> {
        match value {
            Literal::Num(n) => Ok(*n),
            other => Err(TinyAgentsError::Compile(format!(
                "node `{node}` `retry.{key}` expects a number, got `{}`",
                literal_display(other)
            ))),
        }
    };
    let whole = |value: &Literal| -> std::result::Result<u64, TinyAgentsError> {
        let n = num(value)?;
        if n < 0.0 || n.fract() != 0.0 {
            return Err(TinyAgentsError::Compile(format!(
                "node `{node}` `retry.{key}` expects a non-negative whole number, got `{}`",
                literal_display(value)
            )));
        }
        Ok(n as u64)
    };
    let boolean = |value: &Literal| -> std::result::Result<bool, TinyAgentsError> {
        match value {
            Literal::Bool(b) => Ok(*b),
            other => Err(TinyAgentsError::Compile(format!(
                "node `{node}` `retry.{key}` expects a boolean, got `{}`",
                literal_display(other)
            ))),
        }
    };

    match key {
        "max_attempts" => policy.max_attempts = whole(value)? as usize,
        "initial_backoff_ms" => policy.initial_backoff_ms = whole(value)?,
        "max_backoff_ms" => policy.max_backoff_ms = whole(value)?,
        "multiplier" => policy.multiplier = num(value)?,
        "jitter" => policy.jitter = boolean(value)?,
        "backoff_sleep" => policy.backoff_sleep = boolean(value)?,
        "max_retry_after_ms" => policy.max_retry_after_ms = whole(value)?,
        other => {
            return Err(TinyAgentsError::Compile(format!(
                "node `{node}` `retry` declares unsupported key `{other}` (expected one of \
max_attempts, initial_backoff_ms, max_backoff_ms, multiplier, jitter, backoff_sleep, \
max_retry_after_ms)"
            )));
        }
    }
    Ok(())
}

/// Computes the single graph-wide [`RetryPolicy`] implied by every node's
/// declared `retry { … }`, or `None` if no node declares one.
///
/// Like [`uniform_node_timeout`], this builder has no per-node retry API
/// (`CompiledGraph::with_node_retry` applies to every node), so two nodes
/// declaring different policies is a `TinyAgentsError::Compile` rather than a
/// silent pick.
fn uniform_node_retry(blueprint: &Blueprint) -> Result<Option<RetryPolicy>> {
    let mut declared: Vec<(&str, RetryPolicy)> = Vec::new();
    for spec in &blueprint.nodes {
        if spec.retry.is_empty() {
            continue;
        }
        let mut policy = RetryPolicy::default();
        for (key, value) in &spec.retry {
            apply_retry_entry(spec.name.as_str(), &mut policy, key.as_str(), value)?;
        }
        declared.push((spec.name.as_str(), policy));
    }
    let Some((first_name, first)) = declared.first().cloned() else {
        return Ok(None);
    };
    let disagreeing: Vec<&str> = declared
        .iter()
        .skip(1)
        .filter(|(_, p)| *p != first)
        .map(|(name, _)| *name)
        .collect();
    if !disagreeing.is_empty() {
        return Err(TinyAgentsError::Compile(format!(
            "per-node retry not supported yet: node `{first_name}` declares a different \
`retry` policy than {}",
            disagreeing.join(", ")
        )));
    }
    Ok(Some(first))
}

/// Wires a blueprint into a durable whole-state graph.
///
/// # Errors
///
/// Returns [`TinyAgentsError::Compile`] when a declared field cannot be
/// faithfully lowered — an unknown `retry`/timeout-unit value, two nodes
/// disagreeing on the graph-wide `timeout`/`retry` policy, a duplicate
/// `input`/`output` field name, or a `sends`/`join_sources` target that is
/// not a declared node — before returning a compiled graph with a policy
/// nobody actually enforces. Also propagates factory errors and graph
/// topology validation failures from [`GraphBuilder::compile`].
pub fn build_graph<State, F>(
    blueprint: &Blueprint,
    factory: &F,
) -> Result<CompiledGraph<State, State>>
where
    State: Clone + Send + Sync + 'static,
    F: NodeFactory<State>,
{
    validate_io_fields("input", &blueprint.input)?;
    validate_io_fields("output", &blueprint.output)?;
    // `checkpoint`/`interrupt` (graph-level policy names, e.g. `"inherit"`)
    // are accepted as a validated no-op: this crate's checkpoint/interrupt
    // support (`CompiledGraph::with_checkpointer`) takes a materialized
    // `Arc<dyn Checkpointer<State>>` instance, which a blueprint cannot
    // supply — there is no registry of checkpointer *instances* keyed by
    // policy name to look one up in. A host that wants the declared policy
    // enforced attaches a checkpointer to the `CompiledGraph` this function
    // returns.

    let node_names: BTreeSet<&str> = blueprint.nodes.iter().map(|n| n.name.as_str()).collect();

    let mut builder = GraphBuilder::<State, State>::overwrite().set_entry(blueprint.start.as_str());

    for spec in &blueprint.nodes {
        let handler = factory.make(spec)?;
        builder = builder.add_node(spec.name.as_str(), move |state, ctx| {
            (handler.clone())(state, ctx)
        });
        builder = match &spec.routing {
            Routing::Next(target) => builder.add_edge(spec.name.as_str(), target.as_str()),
            Routing::Conditional(routes) => {
                // Conditional routing is not lowered into
                // `add_conditional_edges` here: the node is marked
                // command-routing instead, so the materialized handler
                // itself must resolve `spec.routing`'s labeled targets and
                // return them via `Command::goto` at runtime. The route
                // table is not enforced against a handler's `Command::goto`
                // at compile time: `with_command_destinations` is advisory
                // only (used by `crate::export` to draw/validate the
                // declared destinations), because the runtime always
                // resolves the real successor from the `Command` a node
                // emits. Record it anyway so export/introspection sees the
                // declared labels instead of nothing.
                let destinations: BTreeSet<&str> =
                    routes.iter().map(|(_, target)| target.as_str()).collect();
                builder.with_command_destinations(spec.name.as_str(), destinations)
            }
            Routing::Terminal => builder.set_finish(spec.name.as_str()),
        };

        // `sends`: the actual dynamic fan-out (`Command::goto` carrying
        // `RouteTarget::Send`) is emitted by the handler itself at run time —
        // this function has no way to force an opaque `NodeFactory`-produced
        // handler to emit anything. What it *can* do: validate every
        // declared target is a real node (defense in depth — `Blueprint` is
        // `Deserialize`, so a stored/tampered blueprint can reference a node
        // that no longer exists even though the language compiler already
        // checked this at compile time) and surface the declared fan-out as
        // export metadata instead of dropping it.
        if !spec.sends.is_empty() {
            for send in &spec.sends {
                if !node_names.contains(send.target.as_str()) {
                    return Err(TinyAgentsError::Compile(format!(
                        "node `{}` `sends` target `{}` is not a declared node",
                        spec.name, send.target
                    )));
                }
            }
            let value = spec
                .sends
                .iter()
                .map(|s| match &s.input {
                    Some(input) => format!("{}:{input}", s.target),
                    None => s.target.clone(),
                })
                .collect::<Vec<_>>()
                .join(",");
            builder = builder.with_node_metadata(spec.name.as_str(), "sends", value);
        }

        // `join_sources`: a node-level barrier is the same primitive as a
        // graph-level `join` (below), just declared inline on the waiting
        // node. Lowers onto the real `add_waiting_edge` fan-in mechanism.
        for source in &spec.join_sources {
            if !node_names.contains(source.as_str()) {
                return Err(TinyAgentsError::Compile(format!(
                    "node `{}` `join_sources` names undeclared node `{source}`",
                    spec.name
                )));
            }
            builder = builder.add_waiting_edge(source.as_str(), spec.name.as_str());
        }

        // `command.update`: a `Command`'s `update` field is a typed `Update`
        // produced by the handler, not a bag of `(String, Literal)` pairs —
        // there is no generic way to turn declared literals into an opaque
        // `State`'s partial update without the caller committing to a
        // concrete shape (see `crate::channel::ChannelState` for that opt-in
        // path). Recorded as metadata instead of dropped.
        if let Some(command) = &spec.command
            && !command.update.is_empty()
        {
            let value = command
                .update
                .iter()
                .map(|(key, value)| format!("{key}={}", literal_display(value)))
                .collect::<Vec<_>>()
                .join(",");
            builder = builder.with_node_metadata(spec.name.as_str(), "command.update", value);
        }

        // `options`: choices presented by an `interrupt`-kind node. Marks the
        // node as an interrupt point for `crate::export` (the same marker
        // `GraphBuilder::mark_interrupt` provides for hand-built graphs) and
        // records the choices themselves as metadata.
        if !spec.options.is_empty() {
            builder = builder
                .mark_interrupt(spec.name.as_str())
                .with_node_metadata(spec.name.as_str(), "options", spec.options.join(","));
        }

        // `metadata`: free-form `key value` annotations. A direct, lossless
        // mapping onto `GraphBuilder::with_node_metadata`.
        for (key, value) in &spec.metadata {
            builder =
                builder.with_node_metadata(spec.name.as_str(), key.as_str(), literal_display(value));
        }
    }

    // Graph-level `joins`: the same barrier/fan-in primitive as node-level
    // `join_sources`, declared at the graph level instead of inline on the
    // waiting node.
    for join in &blueprint.joins {
        for source in &join.sources {
            if !node_names.contains(source.as_str()) {
                return Err(TinyAgentsError::Compile(format!(
                    "graph join names undeclared source node `{source}`"
                )));
            }
        }
        if join.target != tinyagents_language::END && !node_names.contains(join.target.as_str()) {
            return Err(TinyAgentsError::Compile(format!(
                "graph join names undeclared target node `{}`",
                join.target
            )));
        }
        for source in &join.sources {
            builder = builder.add_waiting_edge(source.as_str(), join.target.as_str());
        }
    }

    // Per-node `timeout`, applied graph-wide (or rejected on disagreement —
    // see `uniform_node_timeout`).
    if let Some(timeout) = uniform_node_timeout(blueprint)? {
        builder = builder.with_node_timeout(timeout);
    }

    let graph = builder.compile()?;

    // Per-node `retry`, applied graph-wide (or rejected on disagreement —
    // see `uniform_node_retry`). Applied post-compile since
    // `CompiledGraph::with_node_retry` (unlike the timeout knob) lives on the
    // frozen graph, not the builder.
    let graph = match uniform_node_retry(blueprint)? {
        Some(policy) => graph.with_node_retry(policy),
        None => graph,
    };

    Ok(graph)
}

#[cfg(test)]
mod test;
