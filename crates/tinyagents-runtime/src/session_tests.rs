use super::*;
use crate::test_store::{fingerprint, memory, read, seed};
use crate::{DriverFailure, DriverOutcome, SessionBuilder, TranscriptCodec};
use async_trait::async_trait;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use tinyagents_session::transcript::{
    BackgroundAppend, BackgroundAppendOutcome, SessionRef, SessionTranscript, TranscriptLocator,
    TranscriptMessage, TranscriptMeta, TranscriptTurn, session_stem,
};
use tinyinference_llm::message::Message;
use tokio::sync::Notify;
use tokio::time::{Duration, timeout};

struct ReplayCaptureDriver(Arc<Mutex<Vec<Message>>>);

#[async_trait]
impl SessionDriver for ReplayCaptureDriver {
    async fn execute(&self, request: DriverRequest) -> Result<DriverOutcome, DriverFailure> {
        *self.0.lock().unwrap() = request.history;
        Err(DriverFailure {
            error: RuntimeError::Driver("stop after replay capture".into()),
            partial: None,
            outcome: None,
        })
    }
}

struct AppendReplyDriver;

#[async_trait]
impl SessionDriver for AppendReplyDriver {
    async fn execute(&self, request: DriverRequest) -> Result<DriverOutcome, DriverFailure> {
        let mut history = request.history;
        history.push(Message::assistant("reply after replay"));
        Ok(DriverOutcome {
            history,
            output: Some("reply after replay".into()),
            partial: None,
            interrupted: false,
            outcome: None,
            tool_outcomes: Vec::new(),
        })
    }
}

struct NativeReplayCodec;

impl TranscriptCodec for NativeReplayCodec {
    fn decode_history(&self, transcript: &SessionTranscript) -> Result<Vec<Message>, RuntimeError> {
        crate::transcript::native_transcript_to_messages(&transcript.messages)
    }
}

struct ReplayProbeCodec {
    seen_rows: Arc<Mutex<Vec<TranscriptMessage>>>,
    synthesize_leading_tool: bool,
}

impl TranscriptCodec for ReplayProbeCodec {
    fn decode_history(&self, transcript: &SessionTranscript) -> Result<Vec<Message>, RuntimeError> {
        *self.seen_rows.lock().unwrap() = transcript.messages.clone();
        let mut decoded = transcript
            .messages
            .iter()
            .map(|row| match row.role.as_str() {
                "system" => Message::system(&row.content),
                "user" => Message::user(&row.content),
                "tool" => Message::tool(row.id.as_deref().unwrap_or("missing-id"), &row.content),
                _ => Message::assistant(&row.content),
            })
            .collect::<Vec<_>>();
        if self.synthesize_leading_tool {
            let prefix = decoded
                .iter()
                .take_while(|message| matches!(message, Message::System(_)))
                .count();
            decoded.insert(prefix, Message::tool("codec-orphan", "decoded orphan"));
        }
        Ok(decoded)
    }

    fn reconcile(
        &self,
        prior: &[TranscriptMessage],
        _: &[Message],
        _: &[Message],
        _: &crate::TranscriptTurnOptions,
    ) -> Result<Vec<TranscriptMessage>, RuntimeError> {
        Ok(prior.to_vec())
    }
}

fn seed_replay_transcript(
    locator: &dyn TranscriptLocator,
    session: &SessionRef,
    messages: &[TranscriptMessage],
) {
    let mut transcript_meta = meta();
    transcript_meta.session_id = Some(session.session_id());
    transcript_meta.agent_id = session.agent_id.clone();
    transcript_meta.thread_id = Some(session.session_key.clone());
    let handle = locator
        .open_session(session, transcript_meta.clone())
        .unwrap();
    handle
        .append_turn(TranscriptTurn {
            prev: &[],
            next: messages,
            meta: &transcript_meta,
            turn_usage: None,
            request_id: None,
            tools: None,
        })
        .unwrap();
}

#[tokio::test]
async fn resume_repairs_only_the_leading_orphan_tool_head_for_dispatch() {
    let locator = memory();
    let session_ref = SessionRef::scoped("thread-orphan-runtime", "agent");
    let rows = vec![
        TranscriptMessage::system("frozen prompt"),
        TranscriptMessage::tool("orphan result").with_id("orphan-call"),
        TranscriptMessage::user("keep this user turn"),
        TranscriptMessage::tool("later malformed tool").with_id("later-call"),
        TranscriptMessage::assistant("trailing assistant remains"),
    ];
    seed_replay_transcript(locator.as_ref(), &session_ref, &rows);
    let before = fingerprint(
        locator
            .read_session_transcript(&session_ref)
            .unwrap()
            .as_ref(),
    );
    let seen_rows = Arc::new(Mutex::new(Vec::new()));
    let dispatched = Arc::new(Mutex::new(Vec::new()));
    let mut session = SessionBuilder::new(Arc::new(ReplayCaptureDriver(dispatched.clone())))
        .codec(Arc::new(ReplayProbeCodec {
            seen_rows: seen_rows.clone(),
            synthesize_leading_tool: false,
        }))
        .session(locator.clone(), session_ref.clone(), meta())
        .build()
        .unwrap();

    assert!(
        session
            .turn(
                SessionTurnRequest::new(Message::user("next")),
                TurnOptions {
                    resume: ResumeMode::Session,
                    thread_id: Some(session_ref.session_key.clone()),
                    ..TurnOptions::default()
                },
            )
            .await
            .is_err()
    );

    let replay_rows = seen_rows.lock().unwrap().clone();
    assert_eq!(
        replay_rows
            .iter()
            .map(|row| row.content.as_str())
            .collect::<Vec<_>>(),
        [
            "frozen prompt",
            "keep this user turn",
            "later malformed tool",
            "trailing assistant remains"
        ],
        "the codec should receive the replay-only raw-row view"
    );
    let replay = dispatched.lock().unwrap().clone();
    assert_eq!(
        replay.iter().map(Message::text).collect::<Vec<_>>(),
        [
            "frozen prompt",
            "keep this user turn",
            "later malformed tool",
            "trailing assistant remains",
            "next"
        ],
        "later malformed tool rows and trailing assistant rows remain visible to dispatch"
    );
    let after = fingerprint(
        locator
            .read_session_transcript(&session_ref)
            .unwrap()
            .as_ref(),
    );
    assert_eq!(
        after, before,
        "replay repair must never rewrite durable transcript rows"
    );
}

#[tokio::test]
async fn resume_repairs_a_leading_tool_synthesized_by_the_decoder() {
    let locator = memory();
    let session_ref = SessionRef::scoped("thread-decoded-orphan-runtime", "agent");
    let rows = vec![
        TranscriptMessage::system("frozen prompt"),
        TranscriptMessage::user("durable user row"),
    ];
    seed_replay_transcript(locator.as_ref(), &session_ref, &rows);
    let before = fingerprint(
        locator
            .read_session_transcript(&session_ref)
            .unwrap()
            .as_ref(),
    );
    let seen_rows = Arc::new(Mutex::new(Vec::new()));
    let dispatched = Arc::new(Mutex::new(Vec::new()));
    let mut session = SessionBuilder::new(Arc::new(ReplayCaptureDriver(dispatched.clone())))
        .codec(Arc::new(ReplayProbeCodec {
            seen_rows,
            synthesize_leading_tool: true,
        }))
        .session(locator.clone(), session_ref.clone(), meta())
        .build()
        .unwrap();

    assert!(
        session
            .turn(
                SessionTurnRequest::new(Message::user("next")),
                TurnOptions {
                    resume: ResumeMode::Session,
                    thread_id: Some(session_ref.session_key.clone()),
                    ..TurnOptions::default()
                },
            )
            .await
            .is_err()
    );
    assert_eq!(
        dispatched
            .lock()
            .unwrap()
            .iter()
            .map(Message::text)
            .collect::<Vec<_>>(),
        ["frozen prompt", "durable user row", "next"],
        "decoded history must be repaired before dispatch"
    );
    let after = fingerprint(
        locator
            .read_session_transcript(&session_ref)
            .unwrap()
            .as_ref(),
    );
    assert_eq!(
        after, before,
        "decoded repair must leave durable rows unchanged"
    );
}

#[tokio::test]
async fn successful_turn_after_orphan_replay_preserves_user_row_and_original_generation() {
    let locator = memory();
    let session_ref = SessionRef::scoped("thread-orphan-append", "agent");
    let mut system = TranscriptMessage::system("frozen prompt");
    system.extra_metadata = Some(serde_json::json!({"host": "system metadata"}));
    let mut orphan = TranscriptMessage::tool("orphan result").with_id("orphan-call");
    orphan.extra_metadata = Some(serde_json::json!({"host": "retained only in sealed source"}));
    let mut user = TranscriptMessage::user("persisted user");
    user.extra_metadata = Some(serde_json::json!({"source": "original user row"}));
    let rows = vec![system, orphan, user];
    seed_replay_transcript(locator.as_ref(), &session_ref, &rows);
    let original = locator
        .read_session_transcript(&session_ref)
        .unwrap()
        .read_session()
        .unwrap()
        .unwrap();
    let original_fingerprint = fingerprint(
        locator
            .read_session_transcript(&session_ref)
            .unwrap()
            .as_ref(),
    );
    let mut session = SessionBuilder::new(Arc::new(AppendReplyDriver))
        .codec(Arc::new(NativeReplayCodec))
        .session(locator.clone(), session_ref.clone(), meta())
        .build()
        .unwrap();

    session
        .turn(
            SessionTurnRequest::new(Message::user("new user turn")),
            TurnOptions {
                resume: ResumeMode::Session,
                thread_id: Some(session_ref.session_key.clone()),
                ..TurnOptions::default()
            },
        )
        .await
        .unwrap();

    let old_generation = locator
        .read_session_transcript(&session_ref)
        .unwrap()
        .read_session()
        .unwrap()
        .unwrap();
    assert_eq!(old_generation.messages, original.messages);
    assert_eq!(
        fingerprint(
            locator
                .read_session_transcript(&session_ref)
                .unwrap()
                .as_ref()
        ),
        original_fingerprint,
        "the sealed source generation must remain unchanged"
    );
    let active_ref = locator.head_generation(&session_ref);
    assert_eq!(
        active_ref.generation, 1,
        "repairing history commits a successor generation"
    );
    let active = locator
        .read_session_transcript(&active_ref)
        .unwrap()
        .read_session()
        .unwrap()
        .unwrap();
    assert_eq!(
        active
            .messages
            .iter()
            .map(|row| (row.role.as_str(), row.content.as_str()))
            .collect::<Vec<_>>(),
        [
            ("system", "frozen prompt"),
            ("user", "persisted user"),
            ("user", "new user turn"),
            ("assistant", "reply after replay"),
        ],
        "the active generation retains prior conversation and commits the new turn"
    );
    assert_eq!(
        active.messages[1].extra_metadata,
        Some(serde_json::json!({"source": "original user row"})),
        "the matched user row keeps its opaque durable metadata"
    );
    assert_eq!(
        active.meta.session_id.as_deref(),
        Some(active_ref.session_id().as_str())
    );
    assert_eq!(
        active.meta.thread_id.as_deref(),
        Some("thread-orphan-append")
    );
}

#[tokio::test]
async fn successful_replay_repairs_same_source_for_thread_and_latest_modes() {
    for resume in [ResumeMode::Thread, ResumeMode::LatestForAgent] {
        let locator = memory();
        let session_ref = SessionRef::scoped("thread-orphan-scan", "agent");
        let mut user = TranscriptMessage::user("persisted user");
        user.extra_metadata = Some(serde_json::json!({"source": "kept"}));
        let rows = vec![
            TranscriptMessage::system("frozen prompt"),
            TranscriptMessage::tool("orphan result").with_id("orphan-call"),
            user,
        ];
        seed_replay_transcript(locator.as_ref(), &session_ref, &rows);
        let mut session = SessionBuilder::new(Arc::new(AppendReplyDriver))
            .codec(Arc::new(NativeReplayCodec))
            .session(locator.clone(), session_ref.clone(), meta())
            .resume_agent("agent")
            .build()
            .unwrap();

        session
            .turn(
                SessionTurnRequest::new(Message::user("new user turn")),
                TurnOptions {
                    resume,
                    thread_id: Some(session_ref.session_key.clone()),
                    ..TurnOptions::default()
                },
            )
            .await
            .unwrap();

        let active_ref = locator.head_generation(&session_ref);
        let active = locator
            .read_session_transcript(&active_ref)
            .unwrap()
            .read_session()
            .unwrap()
            .unwrap();
        assert_eq!(active_ref.generation, 1, "{resume:?}");
        assert_eq!(
            active
                .messages
                .iter()
                .map(|row| (row.role.as_str(), row.content.as_str()))
                .collect::<Vec<_>>(),
            [
                ("system", "frozen prompt"),
                ("user", "persisted user"),
                ("user", "new user turn"),
                ("assistant", "reply after replay")
            ],
            "same-source {resume:?} replay keeps repaired raw/message correspondence"
        );
        assert_eq!(
            active.messages[1].extra_metadata,
            Some(serde_json::json!({"source": "kept"}))
        );
    }
}

struct LegacyToolReplayCodec;

impl TranscriptCodec for LegacyToolReplayCodec {
    fn decode_history(&self, transcript: &SessionTranscript) -> Result<Vec<Message>, RuntimeError> {
        Ok(transcript
            .messages
            .iter()
            .map(|row| match row.role.as_str() {
                "system" => Message::system(&row.content),
                "user" => Message::user(&row.content),
                "legacy_tool" => Message::tool(row.id.as_deref().unwrap_or("legacy"), &row.content),
                _ => Message::assistant(&row.content),
            })
            .collect())
    }
}

#[tokio::test]
async fn successful_replay_keeps_raw_rows_aligned_after_decoded_tool_repair() {
    let locator = memory();
    let session_ref = SessionRef::scoped("thread-legacy-tool", "agent");
    let mut legacy_tool = TranscriptMessage::tool("orphan result").with_id("legacy-call");
    legacy_tool.role = "legacy_tool".into();
    legacy_tool.extra_metadata = Some(serde_json::json!({"opaque": "preserve"}));
    let rows = vec![
        TranscriptMessage::system("frozen prompt"),
        legacy_tool,
        TranscriptMessage::user("persisted user"),
    ];
    seed_replay_transcript(locator.as_ref(), &session_ref, &rows);
    let original_fingerprint = fingerprint(
        locator
            .read_session_transcript(&session_ref)
            .unwrap()
            .as_ref(),
    );
    let mut session = SessionBuilder::new(Arc::new(AppendReplyDriver))
        .codec(Arc::new(LegacyToolReplayCodec))
        .session(locator.clone(), session_ref.clone(), meta())
        .build()
        .unwrap();

    session
        .turn(
            SessionTurnRequest::new(Message::user("new user turn")),
            TurnOptions {
                resume: ResumeMode::Session,
                thread_id: Some(session_ref.session_key.clone()),
                ..TurnOptions::default()
            },
        )
        .await
        .unwrap();

    assert_eq!(
        fingerprint(
            locator
                .read_session_transcript(&session_ref)
                .unwrap()
                .as_ref()
        ),
        original_fingerprint,
        "the original legacy raw row remains in the sealed source generation"
    );
    let active_ref = locator.head_generation(&session_ref);
    let active = locator
        .read_session_transcript(&active_ref)
        .unwrap()
        .read_session()
        .unwrap()
        .unwrap();
    assert_eq!(active_ref.generation, 1);
    assert_eq!(
        active
            .messages
            .iter()
            .map(|row| (row.role.as_str(), row.content.as_str()))
            .collect::<Vec<_>>(),
        [
            ("system", "frozen prompt"),
            ("user", "persisted user"),
            ("user", "new user turn"),
            ("assistant", "reply after replay")
        ]
    );
}

struct BlockingReadSink {
    locator: Arc<dyn TranscriptLocator>,
    session: SessionRef,
    started: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    release: Arc<Notify>,
    finished: Arc<Notify>,
}

#[async_trait]
impl crate::PostCommitSink for BlockingReadSink {
    async fn after_commit(
        &self,
        observation: &crate::PostCommitObservation,
    ) -> Result<(), RuntimeError> {
        assert!(observation.receipt.transcript.is_some());
        assert_eq!(observation.input, Message::user("commit this"));
        let head = self.locator.head_generation(&self.session);
        let read = self.locator.read_session_transcript(&head).unwrap();
        let transcript = read.read_session().unwrap().unwrap();
        assert!(
            transcript
                .messages
                .iter()
                .any(|row| row.content == "committed reply")
        );
        if let Some(started) = self.started.lock().unwrap().take() {
            let _ = started.send(());
        }
        self.release.notified().await;
        self.finished.notify_one();
        Ok(())
    }
}

struct ReplyDriver;

#[async_trait]
impl SessionDriver for ReplyDriver {
    async fn execute(&self, request: DriverRequest) -> Result<DriverOutcome, DriverFailure> {
        let mut history = request.history;
        history.push(Message::assistant("committed reply"));
        Ok(DriverOutcome {
            history,
            output: Some("committed reply".into()),
            partial: None,
            interrupted: false,
            outcome: None,
            tool_outcomes: Vec::new(),
        })
    }
}

#[tokio::test]
async fn owned_post_commit_sink_finishes_after_caller_abort() {
    let locator = memory();
    let session_ref = SessionRef::scoped("thread-post-commit", "agent");
    let release = Arc::new(Notify::new());
    let finished = Arc::new(Notify::new());
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let sink = Arc::new(BlockingReadSink {
        locator: locator.clone(),
        session: session_ref.clone(),
        started: Mutex::new(Some(started_tx)),
        release: release.clone(),
        finished: finished.clone(),
    });
    let session = SessionBuilder::new(Arc::new(ReplyDriver))
        .codec(Arc::new(RoleCodec))
        .session(locator, session_ref, meta())
        .post_commit_sink(sink)
        .build()
        .unwrap();
    let turn = tokio::spawn(async move {
        let mut session = session;
        session
            .turn(
                SessionTurnRequest::new(Message::user("commit this")),
                TurnOptions::default(),
            )
            .await
    });
    started_rx.await.unwrap();
    turn.abort();
    release.notify_one();
    timeout(Duration::from_secs(1), finished.notified())
        .await
        .unwrap();
}

struct CountSink(Arc<AtomicUsize>);

#[async_trait]
impl crate::PostCommitSink for CountSink {
    async fn after_commit(&self, _: &crate::PostCommitObservation) -> Result<(), RuntimeError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct FailureTerminalRecorder(Arc<Mutex<Vec<SessionTerminal>>>);

#[async_trait]
impl SessionHooks for FailureTerminalRecorder {
    async fn before_turn(
        &self,
        _: &mut SessionTurnRequest,
        _: &mut TurnOptions,
        _: SessionStateView<'_>,
    ) -> Result<TurnPreparation, RuntimeError> {
        Ok(TurnPreparation::default())
    }
    async fn before_commit(
        &self,
        _: &SessionTurnOutcome,
        _: &TranscriptTurnOptions,
    ) -> Result<(), RuntimeError> {
        Ok(())
    }
    async fn on_terminal(&self, terminal: SessionTerminal) -> Result<(), RuntimeError> {
        self.0.lock().unwrap().push(terminal);
        Ok(())
    }
}

struct FailingPartialDriver;

#[async_trait]
impl SessionDriver for FailingPartialDriver {
    async fn execute(&self, request: DriverRequest) -> Result<DriverOutcome, DriverFailure> {
        let mut history = request.history;
        history.push(Message::assistant("durable partial"));
        Err(DriverFailure {
            error: RuntimeError::Driver("provider failed".into()),
            partial: Some(DriverOutcome {
                history,
                output: Some("durable partial".into()),
                partial: None,
                interrupted: false,
                outcome: None,
                tool_outcomes: Vec::new(),
            }),
            outcome: None,
        })
    }
}

#[tokio::test]
async fn persisted_failure_partial_skips_post_commit_sinks_and_fails_once() {
    let locator = memory();
    let session_ref = SessionRef::scoped("thread-failure-partial", "agent");
    let sink_calls = Arc::new(AtomicUsize::new(0));
    let terminals = Arc::new(Mutex::new(Vec::new()));
    let mut session = SessionBuilder::new(Arc::new(FailingPartialDriver))
        .codec(Arc::new(RoleCodec))
        .hooks(Arc::new(FailureTerminalRecorder(terminals.clone())))
        .post_commit_sink(Arc::new(CountSink(sink_calls.clone())))
        .session(locator.clone(), session_ref.clone(), meta())
        .build()
        .unwrap();
    assert!(
        session
            .turn(
                SessionTurnRequest::new(Message::user("fail")),
                TurnOptions::default()
            )
            .await
            .is_err()
    );
    assert_eq!(sink_calls.load(Ordering::SeqCst), 0);
    let terminals = terminals.lock().unwrap();
    assert_eq!(terminals.len(), 1);
    assert!(matches!(terminals[0], SessionTerminal::Failed(_)));
    let head = locator.head_generation(&session_ref);
    let transcript = locator
        .read_session_transcript(&head)
        .unwrap()
        .read_session()
        .unwrap()
        .unwrap();
    assert!(
        transcript
            .messages
            .iter()
            .any(|row| row.content == "durable partial")
    );
}

struct RejectBeforeCommit;

#[async_trait]
impl SessionHooks for RejectBeforeCommit {
    async fn before_turn(
        &self,
        _: &mut SessionTurnRequest,
        _: &mut TurnOptions,
        _: SessionStateView<'_>,
    ) -> Result<TurnPreparation, RuntimeError> {
        Ok(TurnPreparation::default())
    }
    async fn before_commit(
        &self,
        _: &SessionTurnOutcome,
        _: &TranscriptTurnOptions,
    ) -> Result<(), RuntimeError> {
        Err(RuntimeError::Driver("reject before durable append".into()))
    }
    async fn on_terminal(&self, _: SessionTerminal) -> Result<(), RuntimeError> {
        Ok(())
    }
}

#[tokio::test]
async fn precommit_error_does_not_call_post_commit_sink() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut session = SessionBuilder::new(Arc::new(ReplyDriver))
        .hooks(Arc::new(RejectBeforeCommit))
        .post_commit_sink(Arc::new(CountSink(calls.clone())))
        .build()
        .unwrap();
    assert!(
        session
            .turn(
                SessionTurnRequest::new(Message::user("reject")),
                TurnOptions::default()
            )
            .await
            .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

struct InterruptedDriver;

#[async_trait]
impl SessionDriver for InterruptedDriver {
    async fn execute(&self, request: DriverRequest) -> Result<DriverOutcome, DriverFailure> {
        Ok(DriverOutcome {
            history: request.history,
            output: None,
            partial: None,
            interrupted: true,
            outcome: None,
            tool_outcomes: Vec::new(),
        })
    }
}

#[tokio::test]
async fn normal_interrupted_commit_still_invokes_post_commit_sink() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut session = SessionBuilder::new(Arc::new(InterruptedDriver))
        .post_commit_sink(Arc::new(CountSink(calls.clone())))
        .build()
        .unwrap();
    let outcome = session
        .turn(
            SessionTurnRequest::new(Message::user("stop")),
            TurnOptions::default(),
        )
        .await
        .unwrap();
    assert!(outcome.interrupted);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

struct ErrorSink;

#[async_trait]
impl crate::PostCommitSink for ErrorSink {
    async fn after_commit(&self, _: &crate::PostCommitObservation) -> Result<(), RuntimeError> {
        Err(RuntimeError::Driver("sink failed after commit".into()))
    }
}

#[tokio::test]
async fn post_commit_sink_error_does_not_change_successful_turn_result() {
    let locator = memory();
    let session_ref = SessionRef::scoped("thread-sink-error", "agent");
    let mut session = SessionBuilder::new(Arc::new(ReplyDriver))
        .codec(Arc::new(RoleCodec))
        .post_commit_sink(Arc::new(ErrorSink))
        .session(locator, session_ref, meta())
        .build()
        .unwrap();
    let outcome = session
        .turn(
            SessionTurnRequest::new(Message::user("succeed")),
            TurnOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(outcome.output.as_deref(), Some("committed reply"));
}

fn meta() -> TranscriptMeta {
    TranscriptMeta {
        session_id: None,
        parent_session_id: None,
        agent_name: "agent".into(),
        agent_id: Some("agent".into()),
        agent_type: None,
        dispatcher: "test".into(),
        provider: None,
        model: None,
        created: "then".into(),
        updated: "then".into(),
        turn_count: 0,
        prefix_message_count: None,
        prefix_reset: false,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: None,
        task_id: None,
    }
}

/// Maps user/assistant rows one-to-one, keeping prior raw rows verbatim.
struct RoleCodec;

impl TranscriptCodec for RoleCodec {
    fn decode_history(&self, transcript: &SessionTranscript) -> Result<Vec<Message>, RuntimeError> {
        Ok(transcript
            .messages
            .iter()
            .map(|row| match row.role.as_str() {
                "system" => Message::system(&row.content),
                "user" => Message::user(&row.content),
                _ => Message::assistant(&row.content),
            })
            .collect())
    }

    fn reconcile(
        &self,
        prior: &[TranscriptMessage],
        previous: &[Message],
        next: &[Message],
        _: &crate::TranscriptTurnOptions,
    ) -> Result<Vec<TranscriptMessage>, RuntimeError> {
        let mut rows = prior.to_vec();
        let prior_system_count = prior.iter().take_while(|row| row.role == "system").count();
        let next_system_count = next
            .iter()
            .take_while(|message| matches!(message, Message::System(_)))
            .count();
        if prior_system_count == 0 && next_system_count > 0 {
            rows.splice(
                0..0,
                next[..next_system_count]
                    .iter()
                    .map(|message| TranscriptMessage::new("system", message.text())),
            );
        }
        rows.extend(
            next[next_system_count
                .saturating_add(previous.len())
                .min(next.len())..]
                .iter()
                .map(|message| match message {
                    Message::System(_) => TranscriptMessage::new("system", message.text()),
                    Message::User(_) => TranscriptMessage::user(message.text()),
                    _ => TranscriptMessage::assistant(message.text()),
                }),
        );
        Ok(rows)
    }
}

struct CapturingDriver(Arc<Mutex<Vec<Vec<Message>>>>);

#[async_trait]
impl SessionDriver for CapturingDriver {
    async fn execute(&self, request: DriverRequest) -> Result<DriverOutcome, DriverFailure> {
        self.0.lock().unwrap().push(request.history.clone());
        let mut history = request.history;
        history.push(Message::assistant("reply"));
        Ok(DriverOutcome {
            outcome: None,
            history,
            output: Some("reply".into()),
            partial: None,
            interrupted: false,
            tool_outcomes: Vec::new(),
        })
    }
}

struct PrefixWhenUnfrozen(&'static str);

#[async_trait]
impl SessionHooks for PrefixWhenUnfrozen {
    async fn before_turn(
        &self,
        _: &mut SessionTurnRequest,
        _: &mut TurnOptions,
        state: SessionStateView<'_>,
    ) -> Result<TurnPreparation, RuntimeError> {
        Ok(TurnPreparation {
            prefix: (!state.prefix_is_frozen)
                .then(|| PrefixSnapshot::new(vec![Message::system(self.0)])),
            ..TurnPreparation::default()
        })
    }

    async fn on_terminal(&self, _: SessionTerminal) -> Result<(), RuntimeError> {
        Ok(())
    }

    async fn before_commit(
        &self,
        _: &SessionTurnOutcome,
        _: &TranscriptTurnOptions,
    ) -> Result<(), RuntimeError> {
        Ok(())
    }
}

struct MutablePrefix(Arc<Mutex<String>>);

#[async_trait]
impl SessionHooks for MutablePrefix {
    async fn before_turn(
        &self,
        _: &mut SessionTurnRequest,
        _: &mut TurnOptions,
        state: SessionStateView<'_>,
    ) -> Result<TurnPreparation, RuntimeError> {
        let prompt = self.0.lock().unwrap().clone();
        Ok(TurnPreparation {
            prefix: (!state.prefix_is_frozen)
                .then(|| PrefixSnapshot::new(vec![Message::system(prompt)])),
            ..TurnPreparation::default()
        })
    }

    async fn on_terminal(&self, _: SessionTerminal) -> Result<(), RuntimeError> {
        Ok(())
    }

    async fn before_commit(
        &self,
        _: &SessionTurnOutcome,
        _: &TranscriptTurnOptions,
    ) -> Result<(), RuntimeError> {
        Ok(())
    }
}

#[tokio::test]
async fn reset_history_seals_old_turns_and_starts_a_fresh_resumable_context() {
    let locator = memory();
    let session_ref = SessionRef::scoped("reset-thread", "agent-id");
    let first_requests = Arc::new(Mutex::new(Vec::new()));
    let mut session = SessionBuilder::new(Arc::new(CapturingDriver(first_requests)))
        .codec(Arc::new(RoleCodec))
        .hooks(Arc::new(PrefixWhenUnfrozen("original prompt")))
        .session(locator.clone(), session_ref.clone(), meta())
        .build()
        .unwrap();
    session
        .turn(
            SessionTurnRequest::new(Message::user("before reset")),
            TurnOptions {
                resume: ResumeMode::Session,
                session: Some(session_ref.clone()),
                ..TurnOptions::default()
            },
        )
        .await
        .unwrap();

    let old_head = locator.head_generation(&session_ref);
    let old_before = locator
        .read_session_transcript(&old_head)
        .unwrap()
        .read_session()
        .unwrap()
        .unwrap();
    assert_eq!(old_before.messages[0].content, "original prompt");
    assert!(session.reset_history().await.unwrap());

    let new_head = locator.head_generation(&session_ref);
    assert_eq!(new_head.generation, old_head.generation + 1);
    assert_eq!(new_head.parent_session_id(), Some(old_head.session_id()));
    let old_after = locator
        .read_session_transcript(&old_head)
        .unwrap()
        .read_session()
        .unwrap()
        .unwrap();
    assert_eq!(
        serde_json::to_value((&old_after.meta, &old_after.messages, &old_after.tools,)).unwrap(),
        serde_json::to_value((&old_before.meta, &old_before.messages, &old_before.tools,)).unwrap(),
    );
    assert!(
        locator
            .read_session_transcript(&new_head)
            .unwrap()
            .read_session()
            .unwrap()
            .unwrap()
            .messages
            .is_empty()
    );

    assert!(!session.reset_history().await.unwrap());
    assert_eq!(locator.head_generation(&session_ref), new_head);

    let reset_requests = Arc::new(Mutex::new(Vec::new()));
    let mut after_reset = SessionBuilder::new(Arc::new(CapturingDriver(reset_requests.clone())))
        .codec(Arc::new(RoleCodec))
        .hooks(Arc::new(PrefixWhenUnfrozen("fresh prompt")))
        .session(locator.clone(), session_ref.clone(), meta())
        .build()
        .unwrap();
    after_reset
        .turn(
            SessionTurnRequest::new(Message::user("after reset")),
            TurnOptions {
                resume: ResumeMode::Session,
                session: Some(session_ref.clone()),
                ..TurnOptions::default()
            },
        )
        .await
        .unwrap();
    let reset_request = reset_requests.lock().unwrap()[0].clone();
    assert_eq!(reset_request[0].text(), "fresh prompt");
    assert_eq!(reset_request[1].text(), "after reset");
    assert!(
        !reset_request
            .iter()
            .any(|message| message.text() == "before reset")
    );

    let resumed_requests = Arc::new(Mutex::new(Vec::new()));
    let mut resumed = SessionBuilder::new(Arc::new(CapturingDriver(resumed_requests.clone())))
        .codec(Arc::new(RoleCodec))
        .hooks(Arc::new(PrefixWhenUnfrozen("replacement prompt")))
        .session(locator.clone(), session_ref.clone(), meta())
        .build()
        .unwrap();
    resumed
        .turn(
            SessionTurnRequest::new(Message::user("cold resume")),
            TurnOptions {
                resume: ResumeMode::Session,
                session: Some(session_ref.clone()),
                ..TurnOptions::default()
            },
        )
        .await
        .unwrap();
    let cold_request = resumed_requests.lock().unwrap()[0].clone();
    assert_eq!(cold_request[0].text(), "fresh prompt");
    assert!(
        !cold_request
            .iter()
            .any(|message| message.text() == "replacement prompt")
    );
    assert!(
        !cold_request
            .iter()
            .any(|message| message.text() == "before reset")
    );
}

#[tokio::test]
async fn reset_by_another_live_session_rebinds_cached_prompt_and_turn_state() {
    let locator = memory();
    let session_ref = SessionRef::scoped("live-reset", "agent-id");
    let prompt = Arc::new(Mutex::new("original prompt".to_owned()));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut live = SessionBuilder::new(Arc::new(CapturingDriver(requests.clone())))
        .codec(Arc::new(RoleCodec))
        .hooks(Arc::new(MutablePrefix(prompt.clone())))
        .session(locator.clone(), session_ref.clone(), meta())
        .build()
        .unwrap();
    live.turn(
        SessionTurnRequest::new(Message::user("before reset")),
        TurnOptions {
            resume: ResumeMode::Session,
            session: Some(session_ref.clone()),
            ..TurnOptions::default()
        },
    )
    .await
    .unwrap();

    let mut resetter =
        SessionBuilder::new(Arc::new(CapturingDriver(Arc::new(Mutex::new(Vec::new())))))
            .codec(Arc::new(RoleCodec))
            .session(locator.clone(), session_ref.clone(), meta())
            .build()
            .unwrap();
    assert!(resetter.reset_history().await.unwrap());
    *prompt.lock().unwrap() = "replacement prompt".to_owned();

    live.turn(
        SessionTurnRequest::new(Message::user("after reset")),
        TurnOptions {
            resume: ResumeMode::Session,
            session: Some(session_ref.clone()),
            ..TurnOptions::default()
        },
    )
    .await
    .unwrap();
    let history = requests.lock().unwrap()[1].clone();
    assert_eq!(history[0].text(), "replacement prompt");
    assert_eq!(history[1].text(), "after reset");
    assert!(
        !history
            .iter()
            .any(|message| message.text() == "original prompt")
    );
    assert!(
        !history
            .iter()
            .any(|message| message.text() == "before reset")
    );
    let head = locator.head_generation(&session_ref);
    let transcript = locator
        .read_session_transcript(&head)
        .unwrap()
        .read_session()
        .unwrap()
        .unwrap();
    assert_eq!(transcript.meta.turn_count, 1);
    assert_eq!(transcript.meta.prefix_message_count, Some(1));
}

#[tokio::test]
async fn a_live_session_rebinds_after_another_session_commits_the_reset_head() {
    let locator = memory();
    let session_ref = SessionRef::scoped("live-reset-commit", "agent-id");
    let prompt = Arc::new(Mutex::new("original prompt".to_owned()));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut live = SessionBuilder::new(Arc::new(CapturingDriver(requests.clone())))
        .codec(Arc::new(RoleCodec))
        .hooks(Arc::new(MutablePrefix(prompt.clone())))
        .session(locator.clone(), session_ref.clone(), meta())
        .build()
        .unwrap();
    live.turn(
        SessionTurnRequest::new(Message::user("before reset")),
        TurnOptions {
            resume: ResumeMode::Session,
            session: Some(session_ref.clone()),
            ..TurnOptions::default()
        },
    )
    .await
    .unwrap();

    let mut resetter =
        SessionBuilder::new(Arc::new(CapturingDriver(Arc::new(Mutex::new(Vec::new())))))
            .codec(Arc::new(RoleCodec))
            .session(locator.clone(), session_ref.clone(), meta())
            .build()
            .unwrap();
    assert!(resetter.reset_history().await.unwrap());

    let mut first_after_reset =
        SessionBuilder::new(Arc::new(CapturingDriver(Arc::new(Mutex::new(Vec::new())))))
            .codec(Arc::new(RoleCodec))
            .hooks(Arc::new(PrefixWhenUnfrozen("committed reset prompt")))
            .session(locator.clone(), session_ref.clone(), meta())
            .build()
            .unwrap();
    first_after_reset
        .turn(
            SessionTurnRequest::new(Message::user("first after reset")),
            TurnOptions {
                resume: ResumeMode::Session,
                session: Some(session_ref.clone()),
                ..TurnOptions::default()
            },
        )
        .await
        .unwrap();

    *prompt.lock().unwrap() = "stale replacement prompt".to_owned();
    live.turn(
        SessionTurnRequest::new(Message::user("after concurrent commit")),
        TurnOptions {
            resume: ResumeMode::Session,
            session: Some(session_ref.clone()),
            ..TurnOptions::default()
        },
    )
    .await
    .unwrap();
    let history = requests.lock().unwrap()[1].clone();
    assert_eq!(history[0].text(), "committed reset prompt");
    assert_eq!(history[1].text(), "first after reset");
    assert_eq!(history[2].text(), "reply");
    assert_eq!(history[3].text(), "after concurrent commit");
    assert!(!history.iter().any(|message| {
        matches!(message, Message::System(_))
            && (message.text() == "original prompt" || message.text() == "stale replacement prompt")
    }));
    let head = locator.head_generation(&session_ref);
    let transcript = locator
        .read_session_transcript(&head)
        .unwrap()
        .read_session()
        .unwrap()
        .unwrap();
    assert_eq!(transcript.meta.turn_count, 2);
}

#[tokio::test]
async fn reset_history_advances_an_explicit_frozen_empty_prefix() {
    let locator = memory();
    let session_ref = SessionRef::scoped("empty-prefix-reset", "agent-id");
    let mut empty_prefix_meta = meta();
    empty_prefix_meta.prefix_message_count = Some(0);
    let empty_head = locator
        .open_stem(&session_stem(&session_ref), empty_prefix_meta)
        .unwrap();
    empty_head.replace(&[]).unwrap();
    let mut session =
        SessionBuilder::new(Arc::new(CapturingDriver(Arc::new(Mutex::new(Vec::new())))))
            .codec(Arc::new(RoleCodec))
            .session(locator.clone(), session_ref.clone(), meta())
            .build()
            .unwrap();

    assert!(session.reset_history().await.unwrap());
    let reset_head = locator.head_generation(&session_ref);
    assert_eq!(reset_head.generation, 1);
    assert_eq!(
        locator
            .read_session_transcript(&reset_head)
            .unwrap()
            .read_session()
            .unwrap()
            .unwrap()
            .meta
            .prefix_message_count,
        None,
    );
    assert!(
        locator
            .read_session_transcript(&reset_head)
            .unwrap()
            .read_session()
            .unwrap()
            .unwrap()
            .meta
            .prefix_reset
    );

    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut cold = SessionBuilder::new(Arc::new(CapturingDriver(requests.clone())))
        .codec(Arc::new(RoleCodec))
        .hooks(Arc::new(PrefixWhenUnfrozen("after empty reset")))
        .session(locator, session_ref.clone(), meta())
        .build()
        .unwrap();
    cold.turn(
        SessionTurnRequest::new(Message::user("new turn")),
        TurnOptions {
            resume: ResumeMode::Session,
            session: Some(session_ref),
            ..TurnOptions::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(requests.lock().unwrap()[0][0].text(), "after empty reset");
}

#[tokio::test]
async fn an_empty_compaction_without_reset_marker_recovers_explicit_parent_boundary() {
    let locator = memory();
    let session_ref = SessionRef::scoped("empty-compaction", "agent-id");
    let mut root_meta = meta();
    root_meta.prefix_message_count = Some(0);
    let root = locator
        .open_stem(&session_stem(&session_ref), root_meta)
        .unwrap();
    root.replace(&[]).unwrap();
    let (successor, empty_head) = locator
        .begin_generation_from_baseline(&session_ref, meta(), &[])
        .unwrap();
    empty_head.replace(&[]).unwrap();

    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut resumed = SessionBuilder::new(Arc::new(CapturingDriver(requests.clone())))
        .codec(Arc::new(RoleCodec))
        .hooks(Arc::new(PrefixWhenUnfrozen(
            "must not replace frozen empty",
        )))
        .session(locator, session_ref.clone(), meta())
        .build()
        .unwrap();
    resumed
        .turn(
            SessionTurnRequest::new(Message::user("after compaction")),
            TurnOptions {
                resume: ResumeMode::Session,
                session: Some(session_ref),
                ..TurnOptions::default()
            },
        )
        .await
        .unwrap();
    let request = requests.lock().unwrap()[0].clone();
    assert_eq!(request.len(), 1);
    assert_eq!(request[0].text(), "after compaction");
    assert_eq!(successor.generation, 1);
}

#[tokio::test]
async fn reset_waits_for_an_in_flight_append_before_sealing_the_head() {
    let locator = memory();
    let session_ref = SessionRef::scoped("reset-race", "agent-id");
    seed(
        locator.as_ref(),
        &session_stem(&session_ref),
        &[
            TranscriptMessage::user("before concurrent turn"),
            TranscriptMessage::assistant("old answer"),
        ],
        &meta(),
    );
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let mut turning = SessionBuilder::new(Arc::new(GatedDriver {
        started: started.clone(),
        release: release.clone(),
    }))
    .codec(Arc::new(RoleCodec))
    .session(locator.clone(), session_ref.clone(), meta())
    .build()
    .unwrap();
    let turn_ref = session_ref.clone();
    let turn = tokio::spawn(async move {
        turning
            .turn(
                SessionTurnRequest::new(Message::user("commits before reset")),
                TurnOptions {
                    resume: ResumeMode::Session,
                    session: Some(turn_ref),
                    ..TurnOptions::default()
                },
            )
            .await
    });
    started.notified().await;

    let reset_ref = SessionRef::scoped("reset-race", "agent-id");
    let mut resetting =
        SessionBuilder::new(Arc::new(CapturingDriver(Arc::new(Mutex::new(Vec::new())))))
            .codec(Arc::new(RoleCodec))
            .session(locator.clone(), reset_ref, meta())
            .build()
            .unwrap();
    let reset = tokio::spawn(async move { resetting.reset_history().await });
    tokio::task::yield_now().await;
    assert!(
        !reset.is_finished(),
        "reset bypassed the active session turn lock"
    );

    release.notify_one();
    turn.await.unwrap().unwrap();
    assert!(reset.await.unwrap().unwrap());
    let old = locator
        .read_session_transcript(&session_ref)
        .unwrap()
        .read_session()
        .unwrap()
        .unwrap();
    assert!(
        old.messages
            .iter()
            .any(|message| message.content == "commits before reset")
    );
    assert_eq!(locator.head_generation(&session_ref).generation, 1);
}

#[tokio::test]
async fn reset_does_not_create_an_absent_session_or_cross_agent_boundary() {
    let locator = memory();
    let absent_ref = SessionRef::scoped("absent-reset", "agent-a");
    let mut absent =
        SessionBuilder::new(Arc::new(CapturingDriver(Arc::new(Mutex::new(Vec::new())))))
            .codec(Arc::new(RoleCodec))
            .session(locator.clone(), absent_ref.clone(), meta())
            .build()
            .unwrap();
    assert!(!absent.reset_history().await.unwrap());
    assert!(locator.read_session_transcript(&absent_ref).is_none());

    let first_ref = SessionRef::scoped("same-thread", "agent-a");
    let other_agent_ref = SessionRef::scoped("same-thread", "agent-b");
    seed(
        locator.as_ref(),
        &session_stem(&first_ref),
        &[
            TranscriptMessage::user("first agent"),
            TranscriptMessage::assistant("answer"),
        ],
        &meta(),
    );
    seed(
        locator.as_ref(),
        &session_stem(&other_agent_ref),
        &[
            TranscriptMessage::user("other agent"),
            TranscriptMessage::assistant("answer"),
        ],
        &meta(),
    );
    let mut first =
        SessionBuilder::new(Arc::new(CapturingDriver(Arc::new(Mutex::new(Vec::new())))))
            .codec(Arc::new(RoleCodec))
            .session(locator.clone(), first_ref.clone(), meta())
            .build()
            .unwrap();
    assert!(first.reset_history().await.unwrap());
    assert_eq!(locator.head_generation(&first_ref).generation, 1);
    assert_eq!(locator.head_generation(&other_agent_ref).generation, 0);
    assert!(
        locator
            .read_session_transcript(&other_agent_ref)
            .unwrap()
            .read_session()
            .unwrap()
            .unwrap()
            .messages
            .iter()
            .any(|message| message.content == "other agent")
    );
}

/// An atomic import may create a durable session before its first model turn.
/// The resumed history must still receive the host's initial prefix, while a
/// later process must reuse that committed prefix rather than its new prompt.
#[tokio::test]
async fn imported_history_gets_initial_prefix_before_the_first_turn() {
    let locator = memory();
    let session_ref = SessionRef::scoped("relay-import", "agent");
    let prior = vec![
        TranscriptMessage::user("legacy question"),
        TranscriptMessage::assistant("legacy answer"),
    ];
    assert!(
        locator
            .adopt_messages_if_absent(&session_ref, meta(), &prior)
            .unwrap()
    );

    let first_requests = Arc::new(Mutex::new(Vec::new()));
    let mut first = SessionBuilder::new(Arc::new(CapturingDriver(first_requests.clone())))
        .codec(Arc::new(RoleCodec))
        .hooks(Arc::new(PrefixWhenUnfrozen("configured initial prompt")))
        .session(locator.clone(), session_ref.clone(), meta())
        .build()
        .unwrap();
    first
        .turn(
            SessionTurnRequest::new(Message::user("current question")),
            TurnOptions {
                resume: ResumeMode::Session,
                session: Some(session_ref.clone()),
                thread_id: Some("relay-import".into()),
                ..TurnOptions::default()
            },
        )
        .await
        .unwrap();

    let first_history = first_requests.lock().unwrap()[0].clone();
    assert!(matches!(&first_history[0], Message::System(_)));
    assert_eq!(first_history[0].text(), "configured initial prompt");
    assert!(matches!(&first_history[1], Message::User(_)));
    assert_eq!(first_history[1].text(), "legacy question");
    assert!(matches!(&first_history[2], Message::Assistant(_)));
    assert_eq!(first_history[2].text(), "legacy answer");
    assert!(matches!(&first_history[3], Message::User(_)));
    assert_eq!(first_history[3].text(), "current question");
    let head = locator.head_generation(&session_ref);
    let persisted = locator
        .read_session_transcript(&head)
        .unwrap()
        .read_session()
        .unwrap()
        .unwrap();
    assert_eq!(persisted.messages[0].role, "system");
    assert_eq!(persisted.messages[0].content, "configured initial prompt");

    let second_requests = Arc::new(Mutex::new(Vec::new()));
    let mut second = SessionBuilder::new(Arc::new(CapturingDriver(second_requests.clone())))
        .codec(Arc::new(RoleCodec))
        .hooks(Arc::new(PrefixWhenUnfrozen("replacement prompt")))
        .session(locator, session_ref.clone(), meta())
        .build()
        .unwrap();
    second
        .turn(
            SessionTurnRequest::new(Message::user("after restart")),
            TurnOptions {
                resume: ResumeMode::Session,
                session: Some(session_ref),
                thread_id: Some("relay-import".into()),
                ..TurnOptions::default()
            },
        )
        .await
        .unwrap();
    let second_history = second_requests.lock().unwrap()[0].clone();
    assert!(matches!(&second_history[0], Message::System(_)));
    assert_eq!(second_history[0].text(), "configured initial prompt");
    assert!(matches!(&second_history[1], Message::User(_)));
    assert_eq!(second_history[1].text(), "legacy question");
    assert!(matches!(&second_history[2], Message::Assistant(_)));
    assert_eq!(second_history[2].text(), "legacy answer");
    assert!(
        matches!(&second_history[3], Message::User(_)),
        "resumed request history: {:?}",
        second_history.iter().map(Message::text).collect::<Vec<_>>()
    );
    assert_eq!(second_history[3].text(), "current question");
    assert!(matches!(second_history.last(), Some(Message::User(_))));
    assert_eq!(second_history.last().unwrap().text(), "after restart");
}

#[tokio::test]
async fn an_explicit_empty_frozen_prefix_stays_empty() {
    let locator = memory();
    let session_ref = SessionRef::scoped("empty-prefix", "agent");
    let mut explicit_empty_prefix = meta();
    explicit_empty_prefix.prefix_message_count = Some(0);
    seed(
        locator.as_ref(),
        &session_stem(&session_ref),
        &[TranscriptMessage::user("existing message")],
        &explicit_empty_prefix,
    );

    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut session = SessionBuilder::new(Arc::new(CapturingDriver(requests.clone())))
        .codec(Arc::new(RoleCodec))
        .hooks(Arc::new(PrefixWhenUnfrozen("must remain unused")))
        .session(locator, session_ref.clone(), explicit_empty_prefix)
        .build()
        .unwrap();
    session
        .turn(
            SessionTurnRequest::new(Message::user("next message")),
            TurnOptions {
                resume: ResumeMode::Session,
                session: Some(session_ref),
                thread_id: Some("empty-prefix".into()),
                ..TurnOptions::default()
            },
        )
        .await
        .unwrap();
    assert!(
        requests.lock().unwrap()[0]
            .iter()
            .all(|message| !matches!(message, Message::System(_)))
    );
}

/// Answers each turn with `reply`, but only after the test releases it.
struct GatedDriver {
    started: Arc<Notify>,
    release: Arc<Notify>,
}

#[async_trait]
impl SessionDriver for GatedDriver {
    async fn execute(&self, request: DriverRequest) -> Result<DriverOutcome, DriverFailure> {
        self.started.notify_one();
        self.release.notified().await;
        let mut history = request.history;
        history.push(Message::assistant("Sunny."));
        Ok(DriverOutcome {
            outcome: None,
            history,
            output: Some("Sunny.".into()),
            partial: None,
            interrupted: false,
            tool_outcomes: Vec::new(),
        })
    }
}

/// A background delivery issued while a turn is running waits for that turn
/// to persist, then lands after it — including when resume selects an absent
/// session and the write falls back to the target's existing session.
#[tokio::test]
async fn a_running_turn_holds_off_a_background_append_into_its_session() {
    let locator = memory();
    let session_ref = SessionRef::scoped("thread-1", "agent");
    let selected_session = SessionRef::scoped("thread-2", "agent");
    seed(
        locator.as_ref(),
        &session_stem(&session_ref),
        &[
            TranscriptMessage::user("remind me at 5"),
            TranscriptMessage::assistant("Scheduled."),
        ],
        &meta(),
    );
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let mut session = SessionBuilder::new(Arc::new(GatedDriver {
        started: started.clone(),
        release: release.clone(),
    }))
    .codec(Arc::new(RoleCodec))
    .session(locator.clone(), session_ref.clone(), meta())
    .build()
    .unwrap();
    session
        .seed_history(
            vec![
                Message::user("remind me at 5"),
                Message::assistant("Scheduled."),
            ],
            vec![
                TranscriptMessage::user("remind me at 5"),
                TranscriptMessage::assistant("Scheduled."),
            ],
        )
        .unwrap();

    let turn = tokio::spawn(async move {
        session
            .turn(
                SessionTurnRequest::new(Message::user("weather?")),
                TurnOptions {
                    resume: ResumeMode::Session,
                    session: Some(selected_session),
                    thread_id: Some("thread-2".into()),
                    ..TurnOptions::default()
                },
            )
            .await
    });
    started.notified().await;

    let delivery_locator = locator.clone();
    let delivery_session = session_ref.clone();
    let mut delivery = tokio::spawn(async move {
        tinyagents_session::transcript::append_background_message_in(
            delivery_locator.as_ref(),
            &delivery_session,
            TranscriptMessage::assistant("Time to stretch!"),
            BackgroundAppend::new("run-1", serde_json::json!({"kind": "cron"})),
        )
        .await
    });
    // Single-threaded runtime: yielding lets the delivery run up to the turn
    // lock it must wait on.
    for _ in 0..50 {
        tokio::task::yield_now().await;
    }
    assert!(
        timeout(Duration::from_secs(1), &mut delivery)
            .await
            .is_err(),
        "delivery must wait for the turn"
    );

    release.notify_one();
    turn.await.unwrap().unwrap();
    assert_eq!(
        delivery.await.unwrap().unwrap(),
        BackgroundAppendOutcome::Appended { generation: 1 }
    );
    let contents: Vec<String> = read(
        locator.as_ref(),
        &session_stem(&locator.head_generation(&session_ref)),
    )
    .messages
    .into_iter()
    .map(|message| message.content)
    .collect();
    assert_eq!(
        contents,
        [
            "remind me at 5",
            "Scheduled.",
            "weather?",
            "Sunny.",
            "Time to stretch!"
        ]
    );
}
