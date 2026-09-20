//! Offline concurrency stress coverage for the harness.
//!
//! The 100-agent case is ignored in the ordinary suite so CI does not turn a
//! capacity experiment into a timing-sensitive gate. Run it explicitly with:
//!
//! ```text
//! cargo test -p tinyagents-integration-tests --test harness_stress \
//!   --release -- --ignored --nocapture
//! ```

use std::sync::Arc;

use tinyagents_harness::context::RunConfig;
use tinyagents_harness::runtime::AgentHarness;
use tinyinference_llm::message::Message;
use tinyinference_llm::providers::MockModel;

fn harness() -> Arc<AgentHarness<()>> {
    let mut harness = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::constant("done")));
    Arc::new(harness)
}

async fn run_agents(concurrency: usize, runs_each: usize) {
    let harness = harness();
    let barrier = Arc::new(tokio::sync::Barrier::new(concurrency + 1));
    let mut tasks = tokio::task::JoinSet::new();
    for agent in 0..concurrency {
        let harness = Arc::clone(&harness);
        let barrier = Arc::clone(&barrier);
        tasks.spawn(async move {
            barrier.wait().await;
            for iteration in 0..runs_each {
                harness
                    .invoke(
                        &(),
                        (),
                        RunConfig::new(format!("stress-{agent}-{iteration}")),
                        vec![Message::user("stress")],
                    )
                    .await
                    .expect("concurrent harness run succeeds");
            }
        });
    }
    barrier.wait().await;
    while let Some(result) = tasks.join_next().await {
        result.expect("stress task does not panic");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sixteen_agents_can_share_one_harness() {
    run_agents(16, 5).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "capacity test; run explicitly in release mode"]
async fn one_hundred_agents_can_share_one_harness() {
    run_agents(100, 20).await;
}
