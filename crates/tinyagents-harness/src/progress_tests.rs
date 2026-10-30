use crate::progress::AgentProgress;

#[test]
fn context_compacting_is_a_progress_event() {
    let event = AgentProgress::ContextCompacting;

    assert!(matches!(event, AgentProgress::ContextCompacting));
    assert_ne!(
        std::mem::discriminant(&AgentProgress::ContextCompacting),
        std::mem::discriminant(&AgentProgress::TurnStarted),
    );
}

#[test]
fn subagent_usage_exposes_its_known_cost() {
    let usage = super::progress::SubagentUsage {
        charged_amount_usd: 0.25,
        cost_source: crate::cost::CostSource::Provider,
        ..Default::default()
    };

    assert_eq!(usage.cost().usd(), Some(0.25));
}
