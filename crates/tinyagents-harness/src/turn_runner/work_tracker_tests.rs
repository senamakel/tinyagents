use super::*;

fn iteration_started() -> AgentProgress {
    AgentProgress::IterationStarted {
        iteration: 1,
        max_iterations: 50,
    }
}

fn model_call_completed(subagent_task_id: Option<&str>) -> AgentProgress {
    AgentProgress::ModelCallCompleted {
        model: "m".into(),
        provider_id: "managed".into(),
        subagent_task_id: subagent_task_id.map(str::to_string),
        input: None,
        output: None,
        iteration: 1,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        cache_creation_tokens: 0,
        reasoning_tokens: 0,
        cost_usd: None,
    }
}

fn tool_started(call_id: &str) -> AgentProgress {
    AgentProgress::ToolCallStarted {
        call_id: call_id.into(),
        tool_name: "shell".into(),
        arguments: serde_json::json!({}),
        iteration: 1,
        display_label: None,
        display_detail: None,
    }
}

fn tool_completed(call_id: &str) -> AgentProgress {
    AgentProgress::ToolCallCompleted {
        call_id: call_id.into(),
        tool_name: "shell".into(),
        success: true,
        output_chars: 0,
        output: String::new(),
        arguments: None,
        elapsed_ms: 0,
        iteration: 1,
        failure: None,
        display_label: None,
        display_detail: None,
        structured: None,
    }
}

fn signal(tracker: &mut WorkTracker, event: &AgentProgress) -> Option<&'static str> {
    tracker.observe(event).map(|signal| match signal {
        TurnSignal::WorkStarted => "started",
        TurnSignal::WorkFinished => "finished",
        TurnSignal::Progress(_) => "progress",
    })
}

/// A provider retrying a failed call remains work until the parent call ends.
#[test]
fn an_open_model_call_is_work_until_it_completes() {
    let mut tracker = WorkTracker::default();
    assert_eq!(signal(&mut tracker, &iteration_started()), Some("started"));
    assert_eq!(
        signal(
            &mut tracker,
            &AgentProgress::TextDelta {
                delta: "hi".into(),
                iteration: 1,
            }
        ),
        None
    );
    assert_eq!(
        signal(&mut tracker, &model_call_completed(None)),
        Some("finished")
    );
}

#[test]
fn a_subagent_model_call_does_not_close_parent_work() {
    let mut tracker = WorkTracker::default();
    signal(&mut tracker, &iteration_started());
    assert_eq!(
        signal(&mut tracker, &model_call_completed(Some("task-1"))),
        None
    );
    assert_eq!(
        signal(&mut tracker, &model_call_completed(None)),
        Some("finished")
    );
}

/// Parallel tool calls stay open until the last completion arrives.
#[test]
fn open_tool_calls_finish_only_after_the_last_call_completes() {
    let mut tracker = WorkTracker::default();
    assert_eq!(signal(&mut tracker, &tool_started("a")), Some("started"));
    assert_eq!(signal(&mut tracker, &tool_started("b")), None);
    assert_eq!(signal(&mut tracker, &tool_completed("a")), None);
    assert_eq!(signal(&mut tracker, &tool_completed("b")), Some("finished"));
}

/// A tool start hands work over from the model call without an idle gap.
#[test]
fn tool_start_closes_its_model_call_without_finishing_work() {
    let mut tracker = WorkTracker::default();
    signal(&mut tracker, &iteration_started());
    assert_eq!(signal(&mut tracker, &tool_started("a")), None);
    assert_eq!(signal(&mut tracker, &tool_completed("a")), Some("finished"));
}

/// The next iteration closes a tool whose completion was lost under backpressure.
#[test]
fn the_next_iteration_closes_a_tool_with_missing_completion() {
    let mut tracker = WorkTracker::default();
    signal(&mut tracker, &tool_started("a"));
    assert_eq!(signal(&mut tracker, &iteration_started()), None);
    assert_eq!(
        signal(&mut tracker, &model_call_completed(None)),
        Some("finished")
    );
}

#[test]
fn turn_completion_closes_all_open_work() {
    let mut tracker = WorkTracker::default();
    signal(&mut tracker, &iteration_started());
    assert_eq!(
        signal(&mut tracker, &AgentProgress::turn_completed(1)),
        Some("finished")
    );

    signal(&mut tracker, &tool_started("a"));
    assert_eq!(
        signal(&mut tracker, &AgentProgress::turn_completed(1)),
        Some("finished")
    );
}

#[test]
fn duplicate_starts_and_unknown_completions_do_not_emit_spurious_edges() {
    let mut tracker = WorkTracker::default();
    assert_eq!(signal(&mut tracker, &tool_started("a")), Some("started"));
    assert_eq!(signal(&mut tracker, &tool_started("a")), None);
    assert_eq!(signal(&mut tracker, &tool_completed("missing")), None);
    assert_eq!(signal(&mut tracker, &tool_completed("a")), Some("finished"));
}
