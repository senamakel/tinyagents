use super::*;

fn sample() -> (TaskState, TaskLedger) {
    let state = TaskState {
        goal: "Support default function arguments".into(),
        requirements: vec!["invalid default argument declaration".into()],
        todos_open: vec!["fix parser error message".into()],
        test_command: "go test ./vm/...".into(),
        next_step: "apply_patch parser/parser.go.y".into(),
        ..TaskState::default()
    };
    let ledger = TaskLedger {
        original_task: Some("Add default arguments.\nKeep `name = expression` syntax.".into()),
        files_modified: vec!["parser/parser.go.y".into()],
        files_read: vec!["vm/vm.go".into(), "parser/parser.go.y".into()],
        commands: vec![CommandRecord {
            command: "go test ./vm/...".into(),
            failed: true,
            error: Some("default_arguments_test.go:16: expected substring".into()),
        }],
    };
    (state, ledger)
}

#[test]
fn rendering_round_trips_the_carried_facts() {
    let (state, ledger) = sample();
    let body = render_task_state(&state, &ledger);
    assert!(body.starts_with(TASK_STATE_HEADER));
    assert!(body.contains("- `go test ./vm/...` → FAILED: default_arguments_test.go:16"));

    let (carried, carried_state) = parse_carried(&body);
    assert_eq!(carried.original_task, ledger.original_task);
    assert_eq!(carried.files_modified, vec!["parser/parser.go.y"]);
    // A modified file is listed once, under modified.
    assert_eq!(carried.files_read, vec!["vm/vm.go"]);
    assert!(
        carried.commands.is_empty(),
        "commands are re-read, not carried"
    );
    assert_eq!(carried_state, Some(state));
}

#[test]
fn a_free_form_previous_summary_carries_nothing() {
    let (ledger, state) = parse_carried("=== Conversation Summary (compacted) ===\n## Goal\nstuff");
    assert_eq!(ledger, TaskLedger::default());
    assert!(state.is_none());
}

#[test]
fn state_replies_parse_through_fences_and_prose() {
    let reply = "Here is the state:\n```json\n{\"goal\": \"g\", \"todos_open\": [\"x\"]}\n```";
    let state = parse_state_reply(reply).unwrap();
    assert_eq!(state.goal, "g");
    assert_eq!(state.todos_open, vec!["x"]);
    // Missing keys default; unknown keys are ignored.
    assert!(parse_state_reply("{\"goal\": \"g\", \"extra\": 1}").is_some());
    assert!(parse_state_reply("no json here").is_none());
    assert!(parse_state_reply("{\"goal\": [}").is_none());
}

#[test]
fn the_state_is_written_once_and_read_back_from_its_sections() {
    let (mut state, ledger) = sample();
    state.decisions = vec!["use goyacc output as is — no network".into()];
    state.current_hypothesis = "the parser\nreturns syntax error".into();
    let body = render_task_state(&state, &ledger);
    assert!(!body.contains("<task-state>"), "no JSON copy of the state");
    assert_eq!(
        body.matches("invalid default argument declaration").count(),
        1
    );

    let (_, carried) = parse_carried(&body);
    let carried = carried.expect("a task-state checkpoint carries its state");
    assert_eq!(carried.decisions, state.decisions);
    assert_eq!(carried.requirements, state.requirements);
    assert_eq!(carried.todos_open, state.todos_open);
    assert!(
        carried.constraints.is_empty(),
        "`- none` reads back as empty"
    );
    // Multi-line values are written on one line, so they read back whole.
    assert_eq!(
        carried.current_hypothesis,
        "the parser returns syntax error"
    );
    assert_eq!(carried.next_step, state.next_step);
}
