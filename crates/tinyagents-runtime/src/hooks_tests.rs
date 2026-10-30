use super::*;
use crate::ResumeMode;
use std::sync::{Arc, Mutex};
use tinyinference_llm::message::Message;

#[tokio::test]
async fn callback_adapter_implements_runtime_session_hooks() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let before_commit = Arc::clone(&seen);
    let after_commit = Arc::clone(&seen);
    let terminal = Arc::clone(&seen);
    let hooks: FnSessionHooks = FnSessionHooks::new(
        |request, options, state| {
            Box::pin(async move {
                request.input = Message::user("resumed");
                options.thread_id = Some(format!("resume-{}", state.committed_turns));
                Ok(ResumePreparation::default())
            })
        },
        |request, _options, state| {
            Box::pin(async move {
                request.input = Message::user(if state.resumed {
                    "resumed turn"
                } else {
                    "fresh turn"
                });
                Ok(TurnPreparation::default())
            })
        },
        move |outcome, options| {
            let seen = Arc::clone(&before_commit);
            Box::pin(async move {
                seen.lock().unwrap().push(format!(
                    "before:{}:{}",
                    outcome.output.as_deref().unwrap_or("none"),
                    options.thread_id.as_deref().unwrap_or("none")
                ));
                Ok(())
            })
        },
        move |receipt| {
            let seen = Arc::clone(&after_commit);
            Box::pin(async move {
                seen.lock().unwrap().push(format!(
                    "commit:{}",
                    receipt.outcome.output.as_deref().unwrap_or("none")
                ));
                Ok(())
            })
        },
        move |terminal_value| {
            let seen = Arc::clone(&terminal);
            Box::pin(async move {
                seen.lock().unwrap().push(match terminal_value {
                    SessionTerminal::Completed(_) => "terminal:completed".to_owned(),
                    SessionTerminal::Cancelled => "terminal:cancelled".to_owned(),
                    SessionTerminal::Failed(error) => format!("terminal:failed:{error}"),
                });
                Ok(())
            })
        },
    );

    let prefix = crate::PrefixSnapshot::default();
    let raw_history = [];
    let state = SessionStateView {
        history: &[],
        raw_history: &raw_history,
        prefix: &prefix,
        prefix_is_frozen: false,
        transcript_target: None,
        committed_turns: 4,
        resumed: true,
        identity: None,
    };
    let mut request = SessionTurnRequest::new(Message::user("original"));
    let mut options = TurnOptions::default();
    SessionHooks::before_resume(&hooks, &mut request, &mut options, state)
        .await
        .unwrap();
    assert_eq!(options.thread_id.as_deref(), Some("resume-4"));
    SessionHooks::before_turn(&hooks, &mut request, &mut options, state)
        .await
        .unwrap();
    assert_eq!(request.input, Message::user("resumed turn"));

    let outcome = SessionTurnOutcome {
        history: vec![request.input.clone()],
        output: Some("result".into()),
        interrupted: false,
    };
    let transcript_options = TranscriptTurnOptions {
        request_id: None,
        thread_id: options.thread_id.clone(),
        stream: false,
        resume: ResumeMode::Never,
        context: (),
    };
    SessionHooks::before_commit(&hooks, &outcome, &transcript_options)
        .await
        .unwrap();
    SessionHooks::after_commit(
        &hooks,
        CommitReceipt {
            outcome: outcome.clone(),
            options: transcript_options,
            transcript: None,
        },
    )
    .await
    .unwrap();
    SessionHooks::on_terminal(&hooks, SessionTerminal::Completed(outcome))
        .await
        .unwrap();
    assert_eq!(
        *seen.lock().unwrap(),
        [
            "before:result:resume-4",
            "commit:result",
            "terminal:completed"
        ]
    );
}

#[tokio::test]
async fn callback_adapter_propagates_hook_errors_through_runtime_trait() {
    let hooks: FnSessionHooks = FnSessionHooks::new(
        |_, _, _| Box::pin(async { Ok(ResumePreparation::default()) }),
        |_, _, _| Box::pin(async { Err(RuntimeError::Hook("before-turn".into())) }),
        |_, _| Box::pin(async { Err(RuntimeError::Hook("before-commit".into())) }),
        |_| Box::pin(async { Err(RuntimeError::Hook("after-commit".into())) }),
        |_| Box::pin(async { Err(RuntimeError::Hook("terminal".into())) }),
    );
    let prefix = crate::PrefixSnapshot::default();
    let raw_history = [];
    let state = SessionStateView {
        history: &[],
        raw_history: &raw_history,
        prefix: &prefix,
        prefix_is_frozen: false,
        transcript_target: None,
        committed_turns: 0,
        resumed: false,
        identity: None,
    };
    let mut request = SessionTurnRequest::new(Message::user("x"));
    let mut options = TurnOptions::default();
    assert!(matches!(
        SessionHooks::before_turn(&hooks, &mut request, &mut options, state).await,
        Err(RuntimeError::Hook(message)) if message == "before-turn"
    ));
    assert_eq!(
        SessionHooks::before_commit(
            &hooks,
            &SessionTurnOutcome {
                history: Vec::new(),
                output: None,
                interrupted: false
            },
            &TranscriptTurnOptions {
                request_id: None,
                thread_id: None,
                stream: false,
                resume: ResumeMode::Never,
                context: ()
            }
        )
        .await,
        Err(RuntimeError::Hook("before-commit".into()))
    );
    assert_eq!(
        SessionHooks::after_commit(
            &hooks,
            CommitReceipt {
                outcome: SessionTurnOutcome {
                    history: Vec::new(),
                    output: None,
                    interrupted: false
                },
                options: TranscriptTurnOptions {
                    request_id: None,
                    thread_id: None,
                    stream: false,
                    resume: ResumeMode::Never,
                    context: ()
                },
                transcript: None
            }
        )
        .await,
        Err(RuntimeError::Hook("after-commit".into()))
    );
    assert_eq!(
        SessionHooks::on_terminal(&hooks, SessionTerminal::Cancelled).await,
        Err(RuntimeError::Hook("terminal".into()))
    );
}
