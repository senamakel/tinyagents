//! Host-neutral subagent invocation and lifecycle orchestration.
//!
//! This crate owns TinyAgents' subagent-facing surfaces: direct child-agent
//! invocation, the typed tool adapter, reusable child sessions, and the
//! durable lifecycle driver. Hosts supply persistence and concrete execution;
//! policy, credentials, progress, and RPC remain host concerns.
//!
//! Dependency direction is deliberately one way:
//! `orchestration -> {harness, runtime}`. The lower-level crates never
//! depend on this composition layer.

pub mod subagent;

#[cfg(test)]
mod boundary_tests {
    #[test]
    fn dependency_direction_stays_one_way_and_host_free() {
        let manifest = include_str!("../Cargo.toml");
        assert!(!manifest.contains("openhuman"));

        for lower_layer in [
            include_str!("../../tinyagents-harness/Cargo.toml"),
            include_str!("../../tinyagents-runtime/Cargo.toml"),
        ] {
            assert!(
                !lower_layer.contains("tinyagents-orchestration"),
                "lower TinyAgents layers must not depend on orchestration"
            );
        }
    }

    #[test]
    fn public_subagent_surface_compiles() {
        fn assert_executor<E: crate::subagent::SubagentExecutor>() {}
        let _ = assert_executor::<NeverExecutor>;
    }

    struct NeverExecutor;

    #[async_trait::async_trait]
    impl crate::subagent::SubagentExecutor for NeverExecutor {
        async fn execute(
            &self,
            _execution: crate::subagent::SubagentExecution,
        ) -> Result<crate::subagent::SubagentOutcome, crate::subagent::SubagentError> {
            unreachable!()
        }
    }
}
