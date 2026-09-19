use super::*;
use tinyagents_harness::error::TinyAgentsError;
use tinyagents_language::compiler::compile;
use tinyagents_language::parser::parse_str;

#[derive(Clone, Debug, Default, PartialEq)]
struct S {
    trail: Vec<String>,
}

struct EchoFactory;

impl NodeFactory<S> for EchoFactory {
    fn make(&self, spec: &NodeSpec) -> Result<BoxedNode<S>> {
        let name = spec.name.clone();
        Ok(Arc::new(move |mut state: S, _ctx: crate::NodeContext| {
            let name = name.clone();
            Box::pin(async move {
                state.trail.push(name);
                Ok(crate::NodeResult::Update(state))
            }) as crate::NodeFuture<S>
        }))
    }
}

fn blueprint(src: &str) -> Blueprint {
    compile(&parse_str(src).unwrap()).unwrap().remove(0)
}

fn node_mut<'a>(bp: &'a mut Blueprint, name: &str) -> &'a mut NodeSpec {
    bp.nodes.iter_mut().find(|n| n.name == name).unwrap()
}

#[tokio::test]
async fn build_graph_accepts_a_blueprint_with_no_ignored_fields() {
    let bp = blueprint(
        "graph g { start a node a { kind model next b } node b { kind model next END } }",
    );
    assert_eq!(bp.start, "a");

    let graph = build_graph::<S, _>(&bp, &EchoFactory).expect("no ignored fields, graph builds");
    let run = graph.run(S::default()).await.expect("graph runs to end");
    assert_eq!(run.state.trail, vec!["a".to_string(), "b".to_string()]);
}

#[test]
fn build_graph_lowers_options_to_interrupt_marker_and_metadata() {
    let bp = blueprint(
        "graph g { start a node a { kind model options [\"approve\", \"reject\"] next END } }",
    );

    let graph = build_graph::<S, _>(&bp, &EchoFactory).expect("options is lowered, not rejected");
    let topology = graph.topology();
    let node = topology.nodes.iter().find(|n| n.id == "a").unwrap();
    assert!(
        node.interrupt,
        "options marks the node as an interrupt point"
    );
    assert_eq!(
        node.metadata.get("options").map(String::as_str),
        Some("approve,reject")
    );
}

#[test]
fn build_graph_lowers_node_metadata() {
    let bp = blueprint(
        "graph g { start a node a { kind model metadata { owner \"triage\" priority 3 } next END } }",
    );

    let graph = build_graph::<S, _>(&bp, &EchoFactory).expect("metadata is lowered");
    let topology = graph.topology();
    let node = topology.nodes.iter().find(|n| n.id == "a").unwrap();
    assert_eq!(
        node.metadata.get("owner").map(String::as_str),
        Some("triage")
    );
    assert_eq!(node.metadata.get("priority").map(String::as_str), Some("3"));
}

#[test]
fn build_graph_lowers_sends_to_metadata_and_validates_targets() {
    let bp = blueprint(
        "graph g { start a \
         node a { kind model sends [send b, send c] } \
         node b { kind model next END } \
         node c { kind model next END } }",
    );

    let graph = build_graph::<S, _>(&bp, &EchoFactory).expect("sends is lowered");
    let topology = graph.topology();
    let node = topology.nodes.iter().find(|n| n.id == "a").unwrap();
    assert_eq!(node.metadata.get("sends").map(String::as_str), Some("b,c"));

    // A `sends` target that is not a declared node is rejected even though
    // the language compiler already validated it at compile time — this
    // guards a hand-built or deserialized `Blueprint` that bypassed that
    // check (`Blueprint` is `Deserialize`).
    let mut tampered = bp.clone();
    node_mut(&mut tampered, "a").sends[0].target = "ghost".to_string();
    let err = build_graph::<S, _>(&tampered, &EchoFactory).unwrap_err();
    match err {
        TinyAgentsError::Compile(message) => {
            assert!(message.contains("ghost"), "got: {message}");
        }
        other => panic!("expected Compile, got {other:?}"),
    }
}

#[test]
fn build_graph_lowers_command_update_to_metadata() {
    let bp = blueprint(
        "graph g { start a node a { kind model command { goto END update { status \"done\" } } } }",
    );

    let graph = build_graph::<S, _>(&bp, &EchoFactory).expect("command.update is lowered");
    let topology = graph.topology();
    let node = topology.nodes.iter().find(|n| n.id == "a").unwrap();
    assert_eq!(
        node.metadata.get("command.update").map(String::as_str),
        Some("status=done")
    );
}

#[test]
fn build_graph_lowers_graph_level_joins_to_waiting_edges() {
    let bp = blueprint(
        "graph g { start a \
         node a { kind model routes { toB -> b, toC -> c } } \
         node b { kind model next d } \
         node c { kind model next d } \
         node d { kind model next END } \
         join [b, c] -> d }",
    );

    let graph = build_graph::<S, _>(&bp, &EchoFactory).expect("joins is lowered");
    let topology = graph.topology();
    let waiting = topology
        .waiting_edges
        .iter()
        .find(|w| w.target == "d")
        .expect("d has a waiting/barrier edge");
    assert_eq!(waiting.predecessors, vec!["b".to_string(), "c".to_string()]);
}

#[test]
fn build_graph_lowers_node_join_sources_to_waiting_edges() {
    let bp = blueprint(
        "graph g { start a \
         node a { kind model routes { toB -> b, toC -> c } } \
         node b { kind model next d } \
         node c { kind model next d } \
         node d { kind join sources [b, c] next END } }",
    );

    let graph = build_graph::<S, _>(&bp, &EchoFactory).expect("join_sources is lowered");
    let topology = graph.topology();
    let waiting = topology
        .waiting_edges
        .iter()
        .find(|w| w.target == "d")
        .expect("d has a waiting/barrier edge");
    assert_eq!(waiting.predecessors, vec!["b".to_string(), "c".to_string()]);
}

#[test]
fn build_graph_rejects_undeclared_join_source() {
    let mut bp = blueprint(
        "graph g { start a node a { kind model next b } node b { kind join sources [a] next END } }",
    );
    node_mut(&mut bp, "b").join_sources[0] = "ghost".to_string();

    let err = build_graph::<S, _>(&bp, &EchoFactory).unwrap_err();
    match err {
        TinyAgentsError::Compile(message) => assert!(message.contains("ghost"), "got: {message}"),
        other => panic!("expected Compile, got {other:?}"),
    }
}

#[test]
fn build_graph_accepts_checkpoint_and_interrupt_policy_as_validated_noop() {
    let bp = blueprint(
        "graph g { start a checkpoint inherit interrupt manual node a { kind model next END } }",
    );
    assert_eq!(bp.checkpoint.as_deref(), Some("inherit"));
    assert_eq!(bp.interrupt.as_deref(), Some("manual"));

    // No runtime attach point exists for a bare policy name (see the
    // `build_graph` docs), so this is accepted without error rather than
    // silently dropped or falsely claimed as enforced.
    build_graph::<S, _>(&bp, &EchoFactory).expect("checkpoint/interrupt policy names do not error");
}

#[test]
fn build_graph_accepts_input_and_output_shapes() {
    let bp = blueprint(
        "graph g { start a input { question string } output { answer string } \
         node a { kind model next END } }",
    );

    build_graph::<S, _>(&bp, &EchoFactory).expect("input/output is a validated no-op");
}

#[test]
fn build_graph_rejects_duplicate_io_field_names() {
    let mut bp = blueprint("graph g { start a node a { kind model next END } }");
    bp.input.push(IoFieldSpec {
        name: "question".to_string(),
        ty: "string".to_string(),
    });
    bp.input.push(IoFieldSpec {
        name: "question".to_string(),
        ty: "number".to_string(),
    });

    let err = build_graph::<S, _>(&bp, &EchoFactory).unwrap_err();
    match err {
        TinyAgentsError::Compile(message) => {
            assert!(message.contains("question"), "got: {message}");
        }
        other => panic!("expected Compile, got {other:?}"),
    }
}

#[test]
fn build_graph_lowers_uniform_node_timeout() {
    let bp = blueprint(
        "graph g { start a node a { kind model timeout \"30s\" next b } \
         node b { kind model timeout 30 next END } }",
    );

    let graph = build_graph::<S, _>(&bp, &EchoFactory).expect("uniform timeout is lowered");
    let topology = graph.topology();
    assert_eq!(topology.policy.node_timeout_ms, Some(30_000));
}

#[test]
fn build_graph_rejects_disagreeing_per_node_timeouts() {
    let bp = blueprint(
        "graph g { start a node a { kind model timeout 10 next b } \
         node b { kind model timeout 20 next END } }",
    );

    let err = build_graph::<S, _>(&bp, &EchoFactory).unwrap_err();
    match err {
        TinyAgentsError::Compile(message) => {
            assert!(
                message.contains("per-node timeout not supported yet"),
                "got: {message}"
            );
        }
        other => panic!("expected Compile, got {other:?}"),
    }
}

#[test]
fn build_graph_rejects_an_unsupported_retry_key() {
    let bp = blueprint(
        "graph g { start a node a { kind model retry { backoff \"exponential\" } next END } }",
    );

    let err = build_graph::<S, _>(&bp, &EchoFactory).unwrap_err();
    match err {
        TinyAgentsError::Compile(message) => {
            assert!(message.contains("backoff"), "got: {message}");
        }
        other => panic!("expected Compile, got {other:?}"),
    }
}

#[test]
fn build_graph_rejects_disagreeing_per_node_retry() {
    let bp = blueprint(
        "graph g { start a node a { kind model retry { max_attempts 2 } next b } \
         node b { kind model retry { max_attempts 5 } next END } }",
    );

    let err = build_graph::<S, _>(&bp, &EchoFactory).unwrap_err();
    match err {
        TinyAgentsError::Compile(message) => {
            assert!(
                message.contains("per-node retry not supported yet"),
                "got: {message}"
            );
        }
        other => panic!("expected Compile, got {other:?}"),
    }
}

#[tokio::test]
async fn build_graph_lowers_uniform_node_retry_and_recovers_transient_failure() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let bp = blueprint(
        "graph g { start flaky node flaky { kind model retry { max_attempts 4 } next END } }",
    );

    struct FlakyFactory {
        attempts: Arc<AtomicUsize>,
    }

    impl NodeFactory<S> for FlakyFactory {
        fn make(&self, _spec: &NodeSpec) -> Result<BoxedNode<S>> {
            let attempts = self.attempts.clone();
            Ok(Arc::new(move |mut state: S, _ctx: crate::NodeContext| {
                let attempts = attempts.clone();
                Box::pin(async move {
                    let n = attempts.fetch_add(1, Ordering::SeqCst);
                    if n < 2 {
                        Err(TinyAgentsError::Model(format!("transient blip {n}")))
                    } else {
                        state.trail.push("flaky".to_string());
                        Ok(crate::NodeResult::Update(state))
                    }
                }) as crate::NodeFuture<S>
            }))
        }
    }

    let attempts = Arc::new(AtomicUsize::new(0));
    let factory = FlakyFactory {
        attempts: attempts.clone(),
    };
    let graph =
        build_graph::<S, _>(&bp, &factory).expect("uniform retry is lowered onto with_node_retry");
    let run = graph
        .run(S::default())
        .await
        .expect("the retry policy recovers the transient failure");
    assert_eq!(run.state.trail, vec!["flaky".to_string()]);
    assert_eq!(attempts.load(Ordering::SeqCst), 3);
}
