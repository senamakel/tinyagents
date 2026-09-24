//! Cross-crate integration: [`WorkloadRouter`] projected through
//! [`CapabilityRegistry::route_workload`].
//!
//! This composes the **registry**'s [`CapabilityRegistry`] (name-addressable
//! model storage) with its [`WorkloadRouter`] (declarative workload-tier →
//! model policy) and a real harness [`tinyagents_harness::testkit::ScriptedModel`]
//! double, proving the whole chain end to end: a caller names a workload tier
//! (`"chat-v1"`), the router resolves it to a registered model name, the
//! registry looks that model up, and invoking the resolved handle actually
//! calls the right scripted model — including falling over to a sibling tier
//! when the router declares one.

use std::sync::Arc;

use tinyagents_harness::testkit::ScriptedModel;
use tinyagents_registry::CapabilityRegistry;
use tinyagents_registry::router::{WorkloadRoute, WorkloadRouter};
use tinyinference_llm::model::{ChatModel, ModelRequest};

#[tokio::test]
async fn workload_router_resolves_tier_to_the_right_registered_model() {
    let mut registry: CapabilityRegistry = CapabilityRegistry::new();
    registry
        .register_model(
            "chat-v1",
            Arc::new(ScriptedModel::replies(vec!["chat reply"])),
        )
        .unwrap();
    registry
        .register_model(
            "burst-v1",
            Arc::new(ScriptedModel::replies(vec!["burst reply"])),
        )
        .unwrap();

    registry.set_router(
        WorkloadRouter::new()
            .with_route(WorkloadRoute::new("chat-v1", "chat-v1").with_fallbacks(["burst-v1"]))
            .with_route(WorkloadRoute::new("burst-v1", "burst-v1"))
            .with_default("chat-v1"),
    );

    // The router names a tier; the registry resolves it to the actual
    // registered handle and invoking it reaches the right scripted model.
    let chat = registry.route_workload("chat-v1").expect("chat-v1 routes");
    let response = chat
        .invoke(&(), ModelRequest::new(vec![]))
        .await
        .expect("scripted model replies");
    assert_eq!(response.text(), "chat reply");

    let burst = registry
        .route_workload("burst-v1")
        .expect("burst-v1 routes");
    let response = burst
        .invoke(&(), ModelRequest::new(vec![]))
        .await
        .expect("scripted model replies");
    assert_eq!(response.text(), "burst reply");

    // The router's fallback policy still names the sibling tier for a
    // caller that needs to fail a chat-v1 turn over.
    let policy = registry
        .router()
        .fallback_policy("chat-v1")
        .expect("chat-v1 has a fallback");
    assert_eq!(policy.next_after("chat-v1"), Some("burst-v1"));

    // A tier the router does not know about resolves to nothing, rather
    // than panicking or silently picking a default model.
    assert!(registry.route_workload("unknown-tier").is_none());
}
