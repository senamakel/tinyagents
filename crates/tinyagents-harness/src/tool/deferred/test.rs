use std::collections::BTreeMap;

use serde_json::json;

use super::*;
use tinyinference_llm::tool::ToolCall;
use tinytools::ToolResult;

fn requests() -> DeferredToolRequests {
    DeferredToolRequests {
        calls: vec![ToolCall::new("ext-1", "external", json!({}))],
        approvals: vec![
            ToolCall::new("appr-1", "delete", json!({"path": "a"})),
            ToolCall::new("appr-2", "delete", json!({"path": "b"})),
        ],
        metadata: BTreeMap::new(),
    }
}

#[test]
fn remaining_reports_every_unresolved_id_in_deferral_order() {
    let requests = requests();
    let empty = DeferredToolResults::default();
    assert_eq!(
        requests.remaining(&empty),
        vec![
            CallId::new("appr-1"),
            CallId::new("appr-2"),
            CallId::new("ext-1")
        ]
    );

    let mut partial = DeferredToolResults::default();
    partial
        .approvals
        .insert(CallId::new("appr-2"), ToolApprovalDecision::Approve);
    partial.calls.insert(
        CallId::new("ext-1"),
        DeferredCallResult::Result(ToolResult::success("done")),
    );
    assert_eq!(requests.remaining(&partial), vec![CallId::new("appr-1")]);

    partial.approvals.insert(
        CallId::new("appr-1"),
        ToolApprovalDecision::Deny {
            message: "no".into(),
        },
    );
    assert!(requests.remaining(&partial).is_empty());
}

#[test]
fn remaining_accepts_a_decision_in_either_map() {
    // A host that does not track which list a call came from may answer an
    // approval through `calls` (it ran the tool itself) or an external call
    // through `approvals`; both count as resolved.
    let requests = requests();
    let mut results = DeferredToolResults::default();
    results.calls.insert(
        CallId::new("appr-1"),
        DeferredCallResult::Failed("host refused".into()),
    );
    results
        .approvals
        .insert(CallId::new("ext-1"), ToolApprovalDecision::Approve);
    assert_eq!(requests.remaining(&results), vec![CallId::new("appr-2")]);
}
