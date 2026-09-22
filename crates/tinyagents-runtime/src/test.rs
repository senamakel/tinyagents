#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use tinyagents_harness::{
    context::{RunConfig, RunContext},
    runtime::AgentHarness,
};
use tinyagents_session::transcript::{
    DisplayRecord, FileTranscriptLocator, SessionRef, SessionTranscript, TranscriptHistory,
    TranscriptLocator, TranscriptMessage, TranscriptMeta, TranscriptRead, TranscriptTurn,
    TurnUsage, read_transcript, read_transcript_display, session_stem,
};
use tinyinference_llm::message::Message;
use tinyinference_llm::providers::MockModel;
use tinytools::{Tool, ToolResult, ToolSpec};

use crate::{
    CommitReceipt, DriverFailure, DriverOutcome, DriverRequest, HarnessDriver, PrefixSnapshot,
    ResumeMode, ResumePreparation, RuntimeError, SessionBuilder, SessionDriver, SessionHooks,
    SessionStateView, SessionTerminal, SessionTurnOutcome, SessionTurnRequest, ToolSnapshot,
    TranscriptCodec, TranscriptTarget, TranscriptTurnOptions, TurnOptions, TurnPreparation,
};

struct Driver {
    results: Mutex<VecDeque<Result<DriverOutcome, DriverFailure>>>,
    requests: Mutex<Vec<DriverRequest>>,
}

impl Driver {
    fn new(results: Vec<Result<DriverOutcome, DriverFailure>>) -> Self {
        Self {
            results: Mutex::new(results.into()),
            requests: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl SessionDriver for Driver {
    async fn execute(&self, request: DriverRequest) -> Result<DriverOutcome, DriverFailure> {
        self.requests.lock().unwrap().push(request);
        self.results.lock().unwrap().pop_front().unwrap()
    }
}

struct WaitingDriver(Arc<tokio::sync::Notify>);

#[async_trait]
impl SessionDriver for WaitingDriver {
    async fn execute(&self, _: DriverRequest) -> Result<DriverOutcome, DriverFailure> {
        self.0.notify_one();
        std::future::pending().await
    }
}

struct RegisteredTool;

#[async_trait]
impl Tool for RegisteredTool {
    fn name(&self) -> &str {
        "registered"
    }
    fn description(&self) -> &str {
        "registered tool"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({})
    }
    async fn execute(&self, _: serde_json::Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success("ok"))
    }
}

fn outcome(history: Vec<Message>) -> DriverOutcome {
    DriverOutcome {
        history,
        output: Some("ok".into()),
        partial: None,
        interrupted: false,
    }
}

fn meta() -> TranscriptMeta {
    TranscriptMeta {
        session_id: None,
        parent_session_id: None,
        agent_name: "agent".into(),
        agent_id: Some("agent-id".into()),
        agent_type: None,
        dispatcher: "test".into(),
        provider: None,
        model: None,
        created: "then".into(),
        updated: "then".into(),
        turn_count: 0,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: None,
        task_id: None,
    }
}

#[derive(Default)]
struct MemoryHistory {
    path: PathBuf,
    state: Mutex<Option<SessionTranscript>>,
    opens: Mutex<usize>,
    fail: Mutex<bool>,
    cancel_after_append: Mutex<Option<tinyagents_harness::CancellationToken>>,
    cancel_after_read: Mutex<Option<tinyagents_harness::CancellationToken>>,
}

impl TranscriptRead for MemoryHistory {
    fn path(&self) -> &Path {
        &self.path
    }
    fn read_session(&self) -> anyhow::Result<Option<SessionTranscript>> {
        let transcript = self.state.lock().unwrap().clone();
        if let Some(cancellation) = self.cancel_after_read.lock().unwrap().as_ref() {
            cancellation.cancel();
        }
        Ok(transcript)
    }
}

impl TranscriptHistory for MemoryHistory {
    fn append_turn(&self, turn: TranscriptTurn<'_>) -> anyhow::Result<()> {
        if *self.fail.lock().unwrap() {
            anyhow::bail!("planned persistence failure");
        }
        *self.state.lock().unwrap() = Some(SessionTranscript {
            meta: turn.meta.clone(),
            messages: turn.next.to_vec(),
        });
        if let Some(cancellation) = self.cancel_after_append.lock().unwrap().as_ref() {
            cancellation.cancel();
        }
        Ok(())
    }
    fn messages(&self) -> anyhow::Result<Vec<TranscriptMessage>> {
        Ok(self
            .state
            .lock()
            .unwrap()
            .as_ref()
            .map(|value| value.messages.clone())
            .unwrap_or_default())
    }
    fn append(&self, _: TranscriptMessage) -> anyhow::Result<()> {
        Ok(())
    }
    fn replace(&self, _: &[TranscriptMessage]) -> anyhow::Result<()> {
        Ok(())
    }
    fn clear(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

struct Locator {
    history: Arc<MemoryHistory>,
    latest_agents: Mutex<Vec<String>>,
    scoped_threads: Mutex<Vec<(String, Option<String>)>>,
    opened_stems: Mutex<Vec<String>>,
    /// Sessions this double will answer a read for. Empty means "nothing has
    /// been written yet", which is how a first-turn session behaves.
    known_sessions: Mutex<Vec<SessionRef>>,
    generations: Mutex<Vec<SessionRef>>,
    adopted: Mutex<Vec<(SessionRef, String)>>,
}

impl TranscriptLocator for Locator {
    fn latest_for_agent(&self, agent: &str) -> Option<Arc<dyn TranscriptRead>> {
        self.latest_agents.lock().unwrap().push(agent.into());
        Some(self.history.clone())
    }
    fn root_for_thread(&self, _: &str) -> Option<Arc<dyn TranscriptRead>> {
        Some(self.history.clone())
    }
    fn root_for_thread_scoped(
        &self,
        thread: &str,
        agent_id: Option<&str>,
    ) -> Option<Arc<dyn TranscriptRead>> {
        self.scoped_threads
            .lock()
            .unwrap()
            .push((thread.into(), agent_id.map(str::to_owned)));
        Some(self.history.clone())
    }
    fn open_stem(
        &self,
        stem: &str,
        _: TranscriptMeta,
    ) -> anyhow::Result<Arc<dyn TranscriptHistory>> {
        self.opened_stems.lock().unwrap().push(stem.into());
        *self.history.opens.lock().unwrap() += 1;
        Ok(self.history.clone())
    }
    fn read_session_transcript(&self, session: &SessionRef) -> Option<Arc<dyn TranscriptRead>> {
        self.known_sessions
            .lock()
            .unwrap()
            .contains(session)
            .then(|| self.history.clone() as Arc<dyn TranscriptRead>)
    }
    fn adopt_legacy(
        &self,
        session: &SessionRef,
        thread_id: &str,
        _: &TranscriptMeta,
    ) -> anyhow::Result<Option<tinyagents_session::transcript::SessionAdoption>> {
        self.adopted
            .lock()
            .unwrap()
            .push((session.clone(), thread_id.to_string()));
        Ok(None)
    }
    fn begin_generation(
        &self,
        session: &SessionRef,
        _: TranscriptMeta,
    ) -> anyhow::Result<(SessionRef, Arc<dyn TranscriptHistory>)> {
        let successor = session.next_generation();
        self.generations.lock().unwrap().push(successor.clone());
        Ok((successor, self.history.clone()))
    }
}

fn locator(session: Option<SessionTranscript>) -> (Arc<Locator>, Arc<MemoryHistory>) {
    let dir = tempfile::tempdir().unwrap().keep();
    let history = Arc::new(MemoryHistory {
        path: dir.join("session.jsonl"),
        state: Mutex::new(session),
        opens: Mutex::new(0),
        fail: Mutex::new(false),
        cancel_after_append: Mutex::new(None),
        cancel_after_read: Mutex::new(None),
    });
    (
        Arc::new(Locator {
            history: history.clone(),
            latest_agents: Mutex::new(Vec::new()),
            scoped_threads: Mutex::new(Vec::new()),
            opened_stems: Mutex::new(Vec::new()),
            known_sessions: Mutex::new(Vec::new()),
            generations: Mutex::new(Vec::new()),
            adopted: Mutex::new(Vec::new()),
        }),
        history,
    )
}

#[derive(Default)]
struct Codec {
    seen_prior: Mutex<Vec<Vec<TranscriptMessage>>>,
    seen_options: Mutex<Vec<String>>,
    turn_usage: Mutex<Option<TurnUsage>>,
    fail_turn_usage: bool,
}

impl TranscriptCodec for Codec {
    fn decode_history(&self, transcript: &SessionTranscript) -> Result<Vec<Message>, RuntimeError> {
        Ok(transcript
            .messages
            .iter()
            .map(|row| Message::user(&row.content))
            .collect())
    }
    fn reconcile(
        &self,
        prior: &[TranscriptMessage],
        _: &[Message],
        next: &[Message],
        options: &TranscriptTurnOptions,
    ) -> Result<Vec<TranscriptMessage>, RuntimeError> {
        self.seen_prior.lock().unwrap().push(prior.to_vec());
        self.seen_options
            .lock()
            .unwrap()
            .push(format!("{:?}", options.request_id));
        if next.len() < prior.len() {
            return Ok(next
                .iter()
                .map(|message| TranscriptMessage::new("assistant", message.text()))
                .collect());
        }
        let mut rows = prior.to_vec();
        if rows.len() < next.len() {
            rows.extend(
                next[rows.len()..]
                    .iter()
                    .map(|message| TranscriptMessage::new("assistant", message.text())),
            );
        }
        Ok(rows)
    }

    fn turn_usage(&self, _: &TranscriptTurnOptions) -> Result<Option<TurnUsage>, RuntimeError> {
        if self.fail_turn_usage {
            return Err(RuntimeError::Driver("planned turn-usage failure".into()));
        }
        Ok(self.turn_usage.lock().unwrap().clone())
    }
}

fn turn_usage() -> TurnUsage {
    TurnUsage {
        provider: "provider".into(),
        model: "model".into(),
        usage: tinyagents_session::transcript::MessageUsage {
            input: 11,
            output: 7,
            cached_input: 3,
            context_window: 128,
            cost_usd: 0.42,
        },
        ts: "now".into(),
        reasoning_content: Some("because".into()),
        tool_calls: Vec::new(),
        iteration: 2,
    }
}

#[derive(Default)]
struct Events {
    terminal: Mutex<Vec<SessionTerminal>>,
    receipts: Mutex<Vec<CommitReceipt>>,
    before_commit: Mutex<usize>,
    order: Mutex<Vec<&'static str>>,
}

struct Hook {
    resume_preparations: Mutex<VecDeque<ResumePreparation>>,
    preparations: Mutex<VecDeque<TurnPreparation>>,
    events: Arc<Events>,
    fail_commit: bool,
    fail_post: bool,
}

#[async_trait]
impl SessionHooks for Hook {
    async fn before_resume(
        &self,
        _: &mut SessionTurnRequest,
        _: &mut TurnOptions,
        _: SessionStateView<'_>,
    ) -> Result<ResumePreparation, RuntimeError> {
        Ok(self
            .resume_preparations
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_default())
    }
    async fn before_turn(
        &self,
        _: &mut SessionTurnRequest,
        _: &mut TurnOptions,
        _: SessionStateView<'_>,
    ) -> Result<TurnPreparation, RuntimeError> {
        Ok(self
            .preparations
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_default())
    }
    async fn before_commit(
        &self,
        _: &SessionTurnOutcome,
        _: &TranscriptTurnOptions,
    ) -> Result<(), RuntimeError> {
        *self.events.before_commit.lock().unwrap() += 1;
        if self.fail_commit {
            Err(RuntimeError::Hook("rejected".into()))
        } else {
            Ok(())
        }
    }
    async fn after_commit(&self, receipt: CommitReceipt) -> Result<(), RuntimeError> {
        self.events.receipts.lock().unwrap().push(receipt);
        self.events.order.lock().unwrap().push("after_commit");
        if self.fail_post {
            Err(RuntimeError::Hook("post-commit".into()))
        } else {
            Ok(())
        }
    }
    async fn on_terminal(&self, terminal: SessionTerminal) -> Result<(), RuntimeError> {
        self.events.terminal.lock().unwrap().push(terminal);
        self.events.order.lock().unwrap().push("terminal");
        Ok(())
    }
}

fn hook(preparations: Vec<TurnPreparation>) -> (Arc<Hook>, Arc<Events>) {
    let events = Arc::new(Events::default());
    (
        Arc::new(Hook {
            resume_preparations: Mutex::new(VecDeque::new()),
            preparations: Mutex::new(preparations.into()),
            events: events.clone(),
            fail_commit: false,
            fail_post: false,
        }),
        events,
    )
}

fn hook_with_resume(
    resume_preparations: Vec<ResumePreparation>,
    preparations: Vec<TurnPreparation>,
) -> (Arc<Hook>, Arc<Events>) {
    let (hook, events) = hook(preparations);
    *hook.resume_preparations.lock().unwrap() = resume_preparations.into();
    (hook, events)
}

fn tools(name: &str) -> ToolSnapshot {
    ToolSnapshot::new(vec![ToolSpec {
        name: name.into(),
        description: name.into(),
        parameters: serde_json::json!({}),
    }])
    .unwrap()
}

#[tokio::test]
async fn prepared_first_turn_prefix_replaces_empty_builder_prefix() {
    let driver = Arc::new(Driver::new(vec![Ok(outcome(vec![
        Message::system("prepared"),
        Message::assistant("ok"),
    ]))]));
    let (hook, _) = hook(vec![TurnPreparation {
        prefix: Some(PrefixSnapshot::new(vec![Message::system("prepared")])),
        ..Default::default()
    }]);
    let mut session = SessionBuilder::new(driver.clone())
        .hooks(hook)
        .build()
        .unwrap();
    session
        .turn(
            SessionTurnRequest::new(Message::user("hi")),
            TurnOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(
        driver.requests.lock().unwrap()[0].history[0],
        Message::system("prepared")
    );
    assert_eq!(
        session.prefix_snapshot().messages(),
        &[Message::system("prepared")]
    );
}

#[tokio::test]
async fn prepared_tools_are_dynamic_and_never_reused() {
    let driver = Arc::new(Driver::new(vec![
        Ok(outcome(vec![Message::assistant("one")])),
        Ok(outcome(vec![Message::assistant("two")])),
    ]));
    let (hook, _) = hook(vec![
        TurnPreparation::with_tools(tools("one")),
        TurnPreparation::with_tools(tools("two")),
    ]);
    let mut session = SessionBuilder::new(driver.clone())
        .hooks(hook)
        .build()
        .unwrap();
    for input in ["a", "b"] {
        session
            .turn(
                SessionTurnRequest::new(Message::user(input)),
                TurnOptions::default(),
            )
            .await
            .unwrap();
    }
    let requests = driver.requests.lock().unwrap();
    assert_eq!(requests[0].tools.specs()[0].name, "one");
    assert_eq!(requests[1].tools.specs()[0].name, "two");
}

#[tokio::test]
async fn trailing_input_is_deduplicated_and_tool_collisions_fail_closed() {
    let driver = Arc::new(Driver::new(vec![
        Ok(outcome(vec![Message::user("same")])),
        Ok(outcome(vec![
            Message::user("same"),
            Message::assistant("ok"),
        ])),
    ]));
    let mut session = SessionBuilder::new(driver.clone()).build().unwrap();
    for _ in 0..2 {
        session
            .turn(
                SessionTurnRequest::new(Message::user("same")),
                TurnOptions::default(),
            )
            .await
            .unwrap();
    }
    assert_eq!(
        driver.requests.lock().unwrap()[1].history,
        vec![Message::user("same")]
    );
    assert!(matches!(
        ToolSnapshot::new(vec![
            ToolSpec {
                name: "same".into(),
                description: "one".into(),
                parameters: serde_json::json!({})
            },
            ToolSpec {
                name: "same".into(),
                description: "two".into(),
                parameters: serde_json::json!({})
            },
        ]),
        Err(RuntimeError::ToolNameCollision(_))
    ));
}

#[tokio::test]
async fn persistence_failure_rolls_back_and_a_partial_never_falls_back_to_two_writes() {
    let (failing_locator, history) = locator(None);
    *history.fail.lock().unwrap() = true;
    let mut session = SessionBuilder::new(Arc::new(Driver::new(vec![Ok(outcome(vec![
        Message::assistant("never"),
    ]))])))
    .codec(Arc::new(Codec::default()))
    .transcript(failing_locator, "agent", meta())
    .build()
    .unwrap();
    assert!(matches!(
        session
            .turn(
                SessionTurnRequest::new(Message::user("x")),
                TurnOptions::default()
            )
            .await,
        Err(RuntimeError::Persistence(_))
    ));
    assert!(session.history().is_empty());
    assert!(history.state.lock().unwrap().is_none());

    let (locator, history) = locator(None);
    let partial = DriverOutcome {
        history: vec![Message::assistant("recoverable")],
        output: None,
        partial: Some(crate::TranscriptPartial::new("display only")),
        interrupted: true,
    };
    let mut session = SessionBuilder::new(Arc::new(Driver::new(vec![Err(DriverFailure {
        error: RuntimeError::Driver("interrupted".into()),
        partial: Some(partial),
    })])))
    .codec(Arc::new(Codec::default()))
    .transcript(locator, "agent", meta())
    .build()
    .unwrap();
    assert!(matches!(
        session
            .turn(
                SessionTurnRequest::new(Message::user("x")),
                TurnOptions::default()
            )
            .await,
        Err(RuntimeError::Persistence(_))
    ));
    assert!(history.state.lock().unwrap().is_none());
}

#[tokio::test]
async fn prefix_reconciliation_preserves_maximal_overlap_after_driver_compaction() {
    let prefix = PrefixSnapshot::new(vec![Message::system("a"), Message::system("b")]);
    let driver = Arc::new(Driver::new(vec![Ok(outcome(vec![
        Message::system("b"),
        Message::assistant("answer"),
    ]))]));
    let mut session = SessionBuilder::new(driver).prefix(prefix).build().unwrap();
    let outcome = session
        .turn(
            SessionTurnRequest::new(Message::user("x")),
            TurnOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(
        outcome.history,
        vec![
            Message::system("a"),
            Message::system("b"),
            Message::assistant("answer")
        ]
    );
}

#[tokio::test]
async fn rejected_before_commit_leaves_no_durable_state() {
    let (locator, history) = locator(None);
    let events = Arc::new(Events::default());
    let hook = Arc::new(Hook {
        resume_preparations: Mutex::new(VecDeque::new()),
        preparations: Mutex::new(VecDeque::new()),
        events,
        fail_commit: true,
        fail_post: false,
    });
    let mut session = SessionBuilder::new(Arc::new(Driver::new(vec![Ok(outcome(vec![
        Message::assistant("candidate"),
    ]))])))
    .codec(Arc::new(Codec::default()))
    .transcript(locator, "agent", meta())
    .hooks(hook)
    .build()
    .unwrap();
    assert!(matches!(
        session
            .turn(
                SessionTurnRequest::new(Message::user("x")),
                TurnOptions::default()
            )
            .await,
        Err(RuntimeError::Hook(_))
    ));
    assert!(history.state.lock().unwrap().is_none());
    assert!(session.history().is_empty());
}

#[tokio::test]
async fn post_commit_errors_cannot_relabel_a_successful_turn() {
    let (locator, history) = locator(None);
    let events = Arc::new(Events::default());
    let hook = Arc::new(Hook {
        resume_preparations: Mutex::new(VecDeque::new()),
        preparations: Mutex::new(VecDeque::new()),
        events: events.clone(),
        fail_commit: false,
        fail_post: true,
    });
    let mut session = SessionBuilder::new(Arc::new(Driver::new(vec![Ok(outcome(vec![
        Message::assistant("done"),
    ]))])))
    .codec(Arc::new(Codec::default()))
    .transcript(locator, "agent", meta())
    .hooks(hook)
    .build()
    .unwrap();
    assert!(
        session
            .turn(
                SessionTurnRequest::new(Message::user("x")),
                TurnOptions::default()
            )
            .await
            .is_ok()
    );
    assert!(history.state.lock().unwrap().is_some());
    assert!(matches!(
        events.terminal.lock().unwrap().as_slice(),
        [SessionTerminal::Completed(_)]
    ));
}

#[tokio::test]
async fn harness_driver_keeps_the_explicit_model_and_tool_snapshot_boundary() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::constant("from harness")));
    harness.register_tool(Arc::new(RegisteredTool));
    let snapshot = ToolSnapshot::new(harness.tools().declared_specs()).unwrap();
    let driver = Arc::new(HarnessDriver::new(Arc::new(harness), Arc::new(())));
    let mut session = SessionBuilder::new(driver)
        .tool_snapshot(snapshot)
        .build()
        .unwrap();
    let committed = session
        .turn(
            SessionTurnRequest::new(Message::user("x")),
            TurnOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(committed.output.as_deref(), Some("from harness"));
}

#[tokio::test]
async fn harness_driver_rejects_a_snapshot_not_registered_by_the_harness() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::constant("unreachable")));
    let driver = Arc::new(HarnessDriver::new(Arc::new(harness), Arc::new(())));
    let mut session = SessionBuilder::new(driver)
        .tool_snapshot(tools("not-registered"))
        .build()
        .unwrap();
    assert_eq!(
        session
            .turn(
                SessionTurnRequest::new(Message::user("x")),
                TurnOptions::default()
            )
            .await,
        Err(RuntimeError::ToolSnapshotMismatch)
    );
}

#[tokio::test]
async fn file_history_commits_partial_model_history_and_display_only_partial_together() {
    let directory = tempfile::tempdir().unwrap();
    let locator = Arc::new(FileTranscriptLocator::new(directory.path()));
    let codec = Arc::new(Codec {
        turn_usage: Mutex::new(Some(turn_usage())),
        ..Default::default()
    });
    let partial = DriverOutcome {
        history: vec![Message::assistant("recoverable")],
        output: None,
        partial: Some(crate::TranscriptPartial::new("display partial")),
        interrupted: true,
    };
    let mut session = SessionBuilder::new(Arc::new(Driver::new(vec![Err(DriverFailure {
        error: RuntimeError::Driver("interrupted".into()),
        partial: Some(partial),
    })])))
    .codec(codec)
    .transcript(locator, "agent", meta())
    .build()
    .unwrap();
    assert!(
        session
            .turn(
                SessionTurnRequest::new(Message::user("x")),
                TurnOptions::default()
            )
            .await
            .is_err()
    );
    let path = directory.path().join("session_raw/agent.jsonl");
    let persisted = read_transcript(&path).unwrap();
    assert_eq!(persisted.messages[0].content, "recoverable");
    assert_eq!(persisted.messages[0].turn_usage, Some(turn_usage()));
    assert!(read_transcript_display(&path).unwrap().records.iter().any(|record| matches!(record,
        DisplayRecord::Message(message) if message.interrupted && message.message.content == "display partial"
    )));
}

#[tokio::test]
async fn codec_turn_usage_is_written_with_the_successful_turn_append() {
    let directory = tempfile::tempdir().unwrap();
    let locator = Arc::new(FileTranscriptLocator::new(directory.path()));
    let codec = Arc::new(Codec {
        turn_usage: Mutex::new(Some(turn_usage())),
        ..Default::default()
    });
    let mut session = SessionBuilder::new(Arc::new(Driver::new(vec![Ok(outcome(vec![
        Message::assistant("final"),
    ]))])))
    .codec(codec)
    .transcript(locator, "agent", meta())
    .build()
    .unwrap();

    session
        .turn(
            SessionTurnRequest::new(Message::user("x")),
            TurnOptions::default(),
        )
        .await
        .unwrap();

    let transcript = read_transcript(&directory.path().join("session_raw/agent.jsonl")).unwrap();
    assert_eq!(transcript.messages.len(), 1);
    assert_eq!(transcript.messages[0].content, "final");
    assert_eq!(transcript.messages[0].turn_usage, Some(turn_usage()));
}

#[tokio::test]
async fn codec_turn_usage_error_prevents_any_durable_commit() {
    let (locator, history) = locator(None);
    let codec = Arc::new(Codec {
        fail_turn_usage: true,
        ..Default::default()
    });
    let mut session = SessionBuilder::new(Arc::new(Driver::new(vec![Ok(outcome(vec![
        Message::assistant("final"),
    ]))])))
    .codec(codec)
    .transcript(locator, "agent", meta())
    .build()
    .unwrap();

    assert_eq!(
        session
            .turn(
                SessionTurnRequest::new(Message::user("x")),
                TurnOptions::default(),
            )
            .await,
        Err(RuntimeError::Driver("planned turn-usage failure".into()))
    );
    assert!(history.state.lock().unwrap().is_none());
    assert_eq!(*history.opens.lock().unwrap(), 0);
    assert!(session.history().is_empty());
}

#[tokio::test]
async fn partial_usage_error_leaves_the_session_and_target_entirely_uncommitted() {
    let (locator, history) = locator(None);
    let codec = Arc::new(Codec {
        fail_turn_usage: true,
        ..Default::default()
    });
    let partial = DriverOutcome {
        history: vec![Message::assistant("recoverable")],
        output: None,
        partial: Some(crate::TranscriptPartial::new("display partial")),
        interrupted: true,
    };
    let mut session = SessionBuilder::new(Arc::new(Driver::new(vec![Err(DriverFailure {
        error: RuntimeError::Driver("driver interrupted".into()),
        partial: Some(partial),
    })])))
    .codec(codec)
    .transcript(locator, "agent", meta())
    .build()
    .unwrap();

    assert_eq!(
        session
            .turn(
                SessionTurnRequest::new(Message::user("x")),
                TurnOptions::default(),
            )
            .await,
        Err(RuntimeError::Driver("planned turn-usage failure".into()))
    );
    // `turn_usage` runs before `persist`, so neither a lazy target bind nor
    // its atomic append can happen on this failure path.
    assert_eq!(*history.opens.lock().unwrap(), 0);
    assert!(history.state.lock().unwrap().is_none());
    assert!(session.history().is_empty());
    // A seed is only rejected after `committed_turns` advances. Its success
    // also replaces the runtime's raw persisted snapshot, proving the failed
    // partial never became the next in-memory durable baseline.
    assert!(
        session
            .seed_history(
                vec![Message::assistant("seed")],
                vec![TranscriptMessage::assistant("seed")],
            )
            .is_ok()
    );
}

#[tokio::test]
async fn cancellation_while_the_driver_is_waiting_has_one_cancelled_terminal() {
    let started = Arc::new(tokio::sync::Notify::new());
    let (hook, events) = hook(vec![]);
    let mut session = SessionBuilder::new(Arc::new(WaitingDriver(started.clone())))
        .hooks(hook)
        .build()
        .unwrap();
    let options = TurnOptions::default();
    let cancellation = options.cancellation.clone();
    let turn = tokio::spawn(async move {
        session
            .turn(SessionTurnRequest::new(Message::user("x")), options)
            .await
    });
    started.notified().await;
    cancellation.cancel();
    assert_eq!(turn.await.unwrap(), Err(RuntimeError::Cancelled));
    assert!(matches!(
        events.terminal.lock().unwrap().as_slice(),
        [SessionTerminal::Cancelled]
    ));
}

#[tokio::test]
async fn dropped_turn_future_has_one_failed_terminal() {
    let started = Arc::new(tokio::sync::Notify::new());
    let (hook, events) = hook(vec![]);
    let mut session = SessionBuilder::new(Arc::new(WaitingDriver(started.clone())))
        .hooks(hook)
        .build()
        .unwrap();
    let turn = tokio::spawn(async move {
        session
            .turn(
                SessionTurnRequest::new(Message::user("x")),
                TurnOptions::default(),
            )
            .await
    });
    started.notified().await;
    turn.abort();
    let _ = turn.await;
    tokio::task::yield_now().await;
    assert!(matches!(
        events.terminal.lock().unwrap().as_slice(),
        [SessionTerminal::Failed(_)]
    ));
}

struct BlockingAfterCommitHook {
    started: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
    terminal: Arc<tokio::sync::Notify>,
    events: Arc<Events>,
}

#[async_trait]
impl SessionHooks for BlockingAfterCommitHook {
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
    async fn after_commit(&self, _: CommitReceipt) -> Result<(), RuntimeError> {
        self.started.notify_waiters();
        self.release.notified().await;
        Ok(())
    }
    async fn on_terminal(&self, terminal: SessionTerminal) -> Result<(), RuntimeError> {
        self.events.terminal.lock().unwrap().push(terminal);
        self.terminal.notify_waiters();
        Ok(())
    }
}

#[tokio::test]
async fn dropped_turn_after_durable_append_keeps_a_completed_terminal() {
    let (locator, history) = locator(None);
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let terminal = Arc::new(tokio::sync::Notify::new());
    let events = Arc::new(Events::default());
    let hook = Arc::new(BlockingAfterCommitHook {
        started: started.clone(),
        release: release.clone(),
        terminal: terminal.clone(),
        events: events.clone(),
    });
    let mut session = SessionBuilder::new(Arc::new(Driver::new(vec![Ok(outcome(vec![
        Message::assistant("done"),
    ]))])))
    .codec(Arc::new(Codec::default()))
    .transcript(locator, "agent", meta())
    .hooks(hook)
    .build()
    .unwrap();
    let turn = tokio::spawn(async move {
        session
            .turn(
                SessionTurnRequest::new(Message::user("x")),
                TurnOptions::default(),
            )
            .await
    });
    started.notified().await;
    assert!(history.state.lock().unwrap().is_some());
    let observed_terminal = terminal.notified();
    turn.abort();
    let _ = turn.await;
    release.notify_one();
    observed_terminal.await;
    assert!(matches!(
        events.terminal.lock().unwrap().as_slice(),
        [SessionTerminal::Completed(_)]
    ));
}

#[tokio::test]
async fn resumed_history_restores_the_prefix_once_before_the_next_driver_call() {
    let raw = TranscriptMessage::new("user", "old");
    let (locator, _) = locator(Some(SessionTranscript {
        meta: meta(),
        messages: vec![raw],
    }));
    let driver = Arc::new(Driver::new(vec![Ok(outcome(vec![
        Message::user("old"),
        Message::assistant("new"),
    ]))]));
    let prefix = PrefixSnapshot::new(vec![Message::system("stable")]);
    let mut session = SessionBuilder::new(driver.clone())
        .codec(Arc::new(Codec::default()))
        .prefix(prefix)
        .transcript(locator, "agent", meta())
        .build()
        .unwrap();
    session
        .turn(
            SessionTurnRequest::new(Message::user("next")),
            TurnOptions {
                session: None,
                resume: ResumeMode::LatestForAgent,
                ..TurnOptions::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(
        driver.requests.lock().unwrap()[0].history[0],
        Message::system("stable")
    );
    assert_eq!(
        session
            .history()
            .iter()
            .filter(|message| **message == Message::system("stable"))
            .count(),
        1
    );
}

#[tokio::test]
async fn cancellation_signalled_by_a_successful_append_cannot_relabel_completion() {
    let (locator, history) = locator(None);
    let (hook, events) = hook(vec![]);
    let mut session = SessionBuilder::new(Arc::new(Driver::new(vec![Ok(outcome(vec![
        Message::assistant("done"),
    ]))])))
    .codec(Arc::new(Codec::default()))
    .transcript(locator, "agent", meta())
    .hooks(hook)
    .build()
    .unwrap();
    let options = TurnOptions::default();
    *history.cancel_after_append.lock().unwrap() = Some(options.cancellation.clone());
    assert!(
        session
            .turn(SessionTurnRequest::new(Message::user("x")), options)
            .await
            .is_ok()
    );
    assert!(history.state.lock().unwrap().is_some());
    assert_eq!(events.receipts.lock().unwrap().len(), 1);
    assert!(matches!(
        events.terminal.lock().unwrap().as_slice(),
        [SessionTerminal::Completed(_)]
    ));
}

struct BlockingCommitHook {
    started: Arc<tokio::sync::Notify>,
    events: Arc<Events>,
}
#[async_trait]
impl SessionHooks for BlockingCommitHook {
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
        self.started.notify_waiters();
        std::future::pending().await
    }
    async fn on_terminal(&self, terminal: SessionTerminal) -> Result<(), RuntimeError> {
        self.events.terminal.lock().unwrap().push(terminal);
        Ok(())
    }
}

#[tokio::test]
async fn cancellation_during_before_commit_leaves_no_append() {
    let (locator, history) = locator(None);
    let started = Arc::new(tokio::sync::Notify::new());
    let events = Arc::new(Events::default());
    let hook = Arc::new(BlockingCommitHook {
        started: started.clone(),
        events: events.clone(),
    });
    let mut session = SessionBuilder::new(Arc::new(Driver::new(vec![Ok(outcome(vec![
        Message::assistant("candidate"),
    ]))])))
    .codec(Arc::new(Codec::default()))
    .transcript(locator, "agent", meta())
    .hooks(hook)
    .build()
    .unwrap();
    let options = TurnOptions::default();
    let cancellation = options.cancellation.clone();
    let turn = tokio::spawn(async move {
        session
            .turn(SessionTurnRequest::new(Message::user("x")), options)
            .await
    });
    started.notified().await;
    cancellation.cancel();
    assert_eq!(turn.await.unwrap(), Err(RuntimeError::Cancelled));
    assert!(history.state.lock().unwrap().is_none());
    assert!(matches!(
        events.terminal.lock().unwrap().as_slice(),
        [SessionTerminal::Cancelled]
    ));
}

struct BlockingBeforeTurnHook {
    started: Arc<tokio::sync::Notify>,
    terminal: Arc<tokio::sync::Notify>,
    terminals: Mutex<Vec<SessionTerminal>>,
    after_commits: Mutex<usize>,
}

#[async_trait]
impl SessionHooks for BlockingBeforeTurnHook {
    async fn before_turn(
        &self,
        _: &mut SessionTurnRequest,
        _: &mut TurnOptions,
        _: SessionStateView<'_>,
    ) -> Result<TurnPreparation, RuntimeError> {
        self.started.notify_waiters();
        std::future::pending().await
    }

    async fn before_commit(
        &self,
        _: &SessionTurnOutcome,
        _: &TranscriptTurnOptions,
    ) -> Result<(), RuntimeError> {
        Ok(())
    }

    async fn after_commit(&self, _: CommitReceipt) -> Result<(), RuntimeError> {
        *self.after_commits.lock().unwrap() += 1;
        Ok(())
    }

    async fn on_terminal(&self, terminal: SessionTerminal) -> Result<(), RuntimeError> {
        self.terminals.lock().unwrap().push(terminal);
        self.terminal.notify_waiters();
        Ok(())
    }
}

#[tokio::test]
async fn cancellation_during_before_turn_has_no_driver_persist_or_after_commit() {
    let started = Arc::new(tokio::sync::Notify::new());
    let terminal = Arc::new(tokio::sync::Notify::new());
    let hook = Arc::new(BlockingBeforeTurnHook {
        started: started.clone(),
        terminal: terminal.clone(),
        terminals: Mutex::new(Vec::new()),
        after_commits: Mutex::new(0),
    });
    let (locator, history) = locator(None);
    let driver = Arc::new(Driver::new(vec![Ok(outcome(vec![Message::assistant(
        "never",
    )]))]));
    let mut session = SessionBuilder::new(driver.clone())
        .codec(Arc::new(Codec::default()))
        .transcript(locator, "agent", meta())
        .hooks(hook.clone())
        .build()
        .unwrap();
    let options = TurnOptions::default();
    let cancellation = options.cancellation.clone();
    let observed_terminal = terminal.notified();
    let turn = tokio::spawn(async move {
        session
            .turn(SessionTurnRequest::new(Message::user("x")), options)
            .await
    });
    started.notified().await;
    cancellation.cancel();
    assert_eq!(turn.await.unwrap(), Err(RuntimeError::Cancelled));
    observed_terminal.await;
    assert!(driver.requests.lock().unwrap().is_empty());
    assert!(history.state.lock().unwrap().is_none());
    assert_eq!(*hook.after_commits.lock().unwrap(), 0);
    assert!(matches!(
        hook.terminals.lock().unwrap().as_slice(),
        [SessionTerminal::Cancelled]
    ));
}

struct BlockingAfterCommitCancellationHook {
    started: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
    terminal: Arc<tokio::sync::Notify>,
    terminals: Mutex<Vec<SessionTerminal>>,
}

#[async_trait]
impl SessionHooks for BlockingAfterCommitCancellationHook {
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

    async fn after_commit(&self, _: CommitReceipt) -> Result<(), RuntimeError> {
        self.started.notify_waiters();
        self.release.notified().await;
        Ok(())
    }

    async fn on_terminal(&self, terminal: SessionTerminal) -> Result<(), RuntimeError> {
        self.terminals.lock().unwrap().push(terminal);
        self.terminal.notify_waiters();
        Ok(())
    }
}

#[tokio::test]
async fn cancellation_during_after_commit_keeps_the_completed_result_and_terminal() {
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let terminal = Arc::new(tokio::sync::Notify::new());
    let hook = Arc::new(BlockingAfterCommitCancellationHook {
        started: started.clone(),
        release: release.clone(),
        terminal: terminal.clone(),
        terminals: Mutex::new(Vec::new()),
    });
    let (locator, history) = locator(None);
    let mut session = SessionBuilder::new(Arc::new(Driver::new(vec![Ok(outcome(vec![
        Message::assistant("durable"),
    ]))])))
    .codec(Arc::new(Codec::default()))
    .transcript(locator, "agent", meta())
    .hooks(hook.clone())
    .build()
    .unwrap();
    let options = TurnOptions::default();
    let cancellation = options.cancellation.clone();
    let observed_terminal = terminal.notified();
    let turn = tokio::spawn(async move {
        session
            .turn(SessionTurnRequest::new(Message::user("x")), options)
            .await
    });
    started.notified().await;
    assert!(history.state.lock().unwrap().is_some());
    cancellation.cancel();
    release.notify_one();
    let committed = turn.await.unwrap().unwrap();
    assert_eq!(
        committed.history.last(),
        Some(&Message::assistant("durable"))
    );
    observed_terminal.await;
    assert!(
        matches!(hook.terminals.lock().unwrap().as_slice(), [SessionTerminal::Completed(outcome)] if outcome == &committed)
    );
}

#[tokio::test]
async fn prefix_mutation_after_a_commit_is_rejected() {
    let driver = Arc::new(Driver::new(vec![
        Ok(outcome(vec![Message::assistant("one")])),
        Ok(outcome(vec![Message::assistant("two")])),
    ]));
    let first = TurnPreparation {
        prefix: Some(PrefixSnapshot::new(vec![Message::system("p")])),
        ..Default::default()
    };
    let second = TurnPreparation {
        prefix: Some(PrefixSnapshot::new(vec![Message::system("changed")])),
        ..Default::default()
    };
    let (hook, _) = hook(vec![first, second]);
    let mut session = SessionBuilder::new(driver.clone())
        .hooks(hook)
        .build()
        .unwrap();
    session
        .turn(
            SessionTurnRequest::new(Message::user("a")),
            TurnOptions::default(),
        )
        .await
        .unwrap();
    assert!(matches!(
        session
            .turn(
                SessionTurnRequest::new(Message::user("b")),
                TurnOptions::default()
            )
            .await,
        Err(RuntimeError::InvalidSessionState(_))
    ));
    assert_eq!(driver.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn lazy_target_is_opened_only_after_before_resume_selects_it() {
    let (locator, history) = locator(None);
    let target = TranscriptTarget::new(locator, "late", meta());
    let (hook, _) = hook_with_resume(
        vec![ResumePreparation {
            transcript: Some(target),
        }],
        vec![],
    );
    let mut session = SessionBuilder::new(Arc::new(Driver::new(vec![Ok(outcome(vec![
        Message::assistant("ok"),
    ]))])))
    .codec(Arc::new(Codec::default()))
    .hooks(hook)
    .build()
    .unwrap();
    assert_eq!(*history.opens.lock().unwrap(), 0);
    session
        .turn(
            SessionTurnRequest::new(Message::user("x")),
            TurnOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(*history.opens.lock().unwrap(), 1);
}

/// `SessionBuilder::resume_agent` (new in this diff, distinct from
/// `TranscriptTarget::with_resume_agent` exercised by the test below) must
/// actually reach the resume lookup, not merely be stored and ignored.
#[tokio::test]
async fn session_builder_resume_agent_reaches_the_latest_for_agent_lookup() {
    let (locator, _) = locator(Some(SessionTranscript {
        meta: meta(),
        messages: vec![TranscriptMessage::new("user", "resumed")],
    }));

    let mut session = SessionBuilder::new(Arc::new(Driver::new(vec![Ok(outcome(vec![
        Message::user("resumed"),
        Message::assistant("next"),
    ]))])))
    .codec(Arc::new(Codec::default()))
    .session(
        locator.clone(),
        SessionRef::scoped("thread-1", "agent-id"),
        meta(),
    )
    .resume_agent("resume-agent")
    .build()
    .unwrap();

    session
        .turn(
            SessionTurnRequest::new(Message::user("next")),
            TurnOptions {
                session: None,
                resume: ResumeMode::LatestForAgent,
                ..TurnOptions::default()
            },
        )
        .await
        .unwrap();

    assert_eq!(
        locator.latest_agents.lock().unwrap().as_slice(),
        ["resume-agent"],
        "the configured resume_agent key, not the write stem, must drive the lookup"
    );
}

#[tokio::test]
async fn latest_resume_agent_is_distinct_from_the_write_stem() {
    let (locator, _) = locator(Some(SessionTranscript {
        meta: meta(),
        messages: vec![TranscriptMessage::new("user", "resumed")],
    }));
    let target = TranscriptTarget::new(locator.clone(), "write-stem", meta())
        .with_resume_agent("resume-agent");
    let (hook, _) = hook_with_resume(
        vec![ResumePreparation {
            transcript: Some(target),
        }],
        vec![],
    );
    let mut session = SessionBuilder::new(Arc::new(Driver::new(vec![Ok(outcome(vec![
        Message::user("resumed"),
        Message::assistant("next"),
    ]))])))
    .codec(Arc::new(Codec::default()))
    .hooks(hook)
    .build()
    .unwrap();

    session
        .turn(
            SessionTurnRequest::new(Message::user("next")),
            TurnOptions {
                session: None,
                resume: ResumeMode::LatestForAgent,
                ..TurnOptions::default()
            },
        )
        .await
        .unwrap();

    assert_eq!(
        locator.latest_agents.lock().unwrap().as_slice(),
        ["resume-agent"]
    );
    assert_eq!(
        locator.opened_stems.lock().unwrap().as_slice(),
        ["write-stem"]
    );
}

#[tokio::test]
async fn thread_resume_scopes_lookup_to_the_target_agent() {
    let (locator, _) = locator(Some(SessionTranscript {
        meta: meta(),
        messages: vec![TranscriptMessage::new("user", "resumed")],
    }));
    let target = TranscriptTarget::new(locator.clone(), "write-stem", meta());
    let (hook, _) = hook_with_resume(
        vec![ResumePreparation {
            transcript: Some(target),
        }],
        vec![],
    );
    let mut session = SessionBuilder::new(Arc::new(Driver::new(vec![Ok(outcome(vec![
        Message::user("resumed"),
        Message::assistant("next"),
    ]))])))
    .codec(Arc::new(Codec::default()))
    .hooks(hook)
    .build()
    .unwrap();

    session
        .turn(
            SessionTurnRequest::new(Message::user("next")),
            TurnOptions {
                session: None,
                resume: ResumeMode::Thread,
                thread_id: Some("thread-1".into()),
                ..TurnOptions::default()
            },
        )
        .await
        .unwrap();

    assert_eq!(
        locator.scoped_threads.lock().unwrap().as_slice(),
        [("thread-1".into(), Some("agent-id".into()))]
    );
}

type ObservedResumedState = (bool, Vec<Message>, Vec<TranscriptMessage>);

struct ResumedStateHook {
    observed: Mutex<Option<ObservedResumedState>>,
}

#[async_trait]
impl SessionHooks for ResumedStateHook {
    async fn before_turn(
        &self,
        _: &mut SessionTurnRequest,
        _: &mut TurnOptions,
        state: SessionStateView<'_>,
    ) -> Result<TurnPreparation, RuntimeError> {
        *self.observed.lock().unwrap() = Some((
            state.resumed,
            state.history.to_vec(),
            state.raw_history.to_vec(),
        ));
        Ok(TurnPreparation::default())
    }
    async fn before_commit(
        &self,
        _: &SessionTurnOutcome,
        _: &TranscriptTurnOptions,
    ) -> Result<(), RuntimeError> {
        Ok(())
    }
    async fn on_terminal(&self, _: SessionTerminal) -> Result<(), RuntimeError> {
        Ok(())
    }
}

#[tokio::test]
async fn before_turn_receives_resumed_decoded_history_and_raw_rows() {
    let mut raw = TranscriptMessage::new("user", "old");
    raw.extra_metadata = Some(serde_json::json!({"preserved": true}));
    let (locator, _) = locator(Some(SessionTranscript {
        meta: meta(),
        messages: vec![raw.clone()],
    }));
    let hook = Arc::new(ResumedStateHook {
        observed: Mutex::new(None),
    });
    let mut session = SessionBuilder::new(Arc::new(Driver::new(vec![Ok(outcome(vec![
        Message::user("old"),
        Message::assistant("new"),
    ]))])))
    .codec(Arc::new(Codec::default()))
    .transcript(locator, "agent", meta())
    .hooks(hook.clone())
    .build()
    .unwrap();

    session
        .turn(
            SessionTurnRequest::new(Message::user("next")),
            TurnOptions {
                session: None,
                resume: ResumeMode::LatestForAgent,
                ..TurnOptions::default()
            },
        )
        .await
        .unwrap();

    assert_eq!(
        *hook.observed.lock().unwrap(),
        Some((true, vec![Message::user("old")], vec![raw]))
    );
}

struct PrefixAfterResumeHook {
    prefix: PrefixSnapshot,
    calls: Mutex<usize>,
}

#[async_trait]
impl SessionHooks for PrefixAfterResumeHook {
    async fn before_turn(
        &self,
        _: &mut SessionTurnRequest,
        _: &mut TurnOptions,
        _: SessionStateView<'_>,
    ) -> Result<TurnPreparation, RuntimeError> {
        let mut calls = self.calls.lock().unwrap();
        *calls += 1;
        Ok(TurnPreparation {
            prefix: (*calls == 1).then(|| self.prefix.clone()),
            ..TurnPreparation::default()
        })
    }
    async fn before_commit(
        &self,
        _: &SessionTurnOutcome,
        _: &TranscriptTurnOptions,
    ) -> Result<(), RuntimeError> {
        Ok(())
    }
    async fn on_terminal(&self, _: SessionTerminal) -> Result<(), RuntimeError> {
        Ok(())
    }
}

struct SystemCodec;

impl TranscriptCodec for SystemCodec {
    fn decode_history(&self, transcript: &SessionTranscript) -> Result<Vec<Message>, RuntimeError> {
        Ok(transcript
            .messages
            .iter()
            .map(|row| match row.role.as_str() {
                "system" => Message::system(&row.content),
                _ => Message::user(&row.content),
            })
            .collect())
    }
    fn reconcile(
        &self,
        _: &[TranscriptMessage],
        _: &[Message],
        next: &[Message],
        _: &TranscriptTurnOptions,
    ) -> Result<Vec<TranscriptMessage>, RuntimeError> {
        Ok(next
            .iter()
            .map(|message| TranscriptMessage::new("assistant", message.text()))
            .collect())
    }
}

#[tokio::test]
async fn first_turn_prefix_accepts_an_exact_resumed_prefix_and_restores_it_after_compaction() {
    let prefix = PrefixSnapshot::new(vec![Message::system("stable")]);
    let (locator, _) = locator(Some(SessionTranscript {
        meta: meta(),
        messages: vec![
            TranscriptMessage::new("system", "stable"),
            TranscriptMessage::new("user", "old"),
        ],
    }));
    let driver = Arc::new(Driver::new(vec![
        Ok(outcome(vec![
            Message::system("stable"),
            Message::user("old"),
            Message::assistant("one"),
        ])),
        Ok(outcome(vec![Message::assistant("compacted")])),
    ]));
    let hook = Arc::new(PrefixAfterResumeHook {
        prefix: prefix.clone(),
        calls: Mutex::new(0),
    });
    let mut session = SessionBuilder::new(driver.clone())
        .codec(Arc::new(SystemCodec))
        .transcript(locator, "agent", meta())
        .hooks(hook)
        .build()
        .unwrap();

    session
        .turn(
            SessionTurnRequest::new(Message::user("first")),
            TurnOptions {
                session: None,
                resume: ResumeMode::LatestForAgent,
                ..TurnOptions::default()
            },
        )
        .await
        .unwrap();
    let compacted = session
        .turn(
            SessionTurnRequest::new(Message::user("second")),
            TurnOptions::default(),
        )
        .await
        .unwrap();

    assert_eq!(
        driver.requests.lock().unwrap()[0]
            .history
            .iter()
            .filter(|message| **message == Message::system("stable"))
            .count(),
        1
    );
    assert_eq!(
        compacted.history,
        vec![Message::system("stable"), Message::assistant("compacted")]
    );
}

#[tokio::test]
async fn changed_first_turn_prefix_replaces_a_builder_prefix_after_resume() {
    let old_prefix = PrefixSnapshot::new(vec![Message::system("old")]);
    let new_prefix = PrefixSnapshot::new(vec![Message::system("new")]);
    let (locator, _) = locator(Some(SessionTranscript {
        meta: meta(),
        messages: vec![
            TranscriptMessage::new("system", "old"),
            TranscriptMessage::new("user", "resumed"),
        ],
    }));
    let driver = Arc::new(Driver::new(vec![Ok(outcome(vec![
        Message::system("new"),
        Message::user("resumed"),
        Message::assistant("answer"),
    ]))]));
    let hook = Arc::new(PrefixAfterResumeHook {
        prefix: new_prefix,
        calls: Mutex::new(0),
    });
    let mut session = SessionBuilder::new(driver.clone())
        .codec(Arc::new(SystemCodec))
        .prefix(old_prefix)
        .transcript(locator, "agent", meta())
        .hooks(hook)
        .build()
        .unwrap();

    session
        .turn(
            SessionTurnRequest::new(Message::user("next")),
            TurnOptions {
                session: None,
                resume: ResumeMode::LatestForAgent,
                ..TurnOptions::default()
            },
        )
        .await
        .unwrap();

    assert_eq!(
        driver.requests.lock().unwrap()[0].history,
        vec![
            Message::system("new"),
            Message::user("resumed"),
            Message::user("next"),
        ]
    );
}

#[tokio::test]
async fn hook_selected_target_and_resume_mode_apply_before_driver_handoff() {
    let (locator, _) = locator(Some(SessionTranscript {
        meta: meta(),
        messages: vec![TranscriptMessage::new("user", "resumed")],
    }));
    let target = TranscriptTarget::new(locator, "agent", meta());
    let driver = Arc::new(Driver::new(vec![Ok(outcome(vec![
        Message::user("resumed"),
        Message::assistant("ok"),
    ]))]));
    struct ResumeHook(TranscriptTarget);
    #[async_trait]
    impl SessionHooks for ResumeHook {
        async fn before_resume(
            &self,
            _: &mut SessionTurnRequest,
            options: &mut TurnOptions,
            _: SessionStateView<'_>,
        ) -> Result<ResumePreparation, RuntimeError> {
            options.resume = ResumeMode::LatestForAgent;
            Ok(ResumePreparation {
                transcript: Some(self.0.clone()),
            })
        }
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
        async fn on_terminal(&self, _: SessionTerminal) -> Result<(), RuntimeError> {
            Ok(())
        }
    }
    let mut session = SessionBuilder::new(driver.clone())
        .codec(Arc::new(Codec::default()))
        .hooks(Arc::new(ResumeHook(target)))
        .build()
        .unwrap();
    session
        .turn(
            SessionTurnRequest::new(Message::user("next")),
            TurnOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(
        driver.requests.lock().unwrap()[0].history[0],
        Message::user("resumed")
    );
}

#[tokio::test]
async fn hook_selected_transcript_without_a_codec_fails_before_driver_execution() {
    let (locator, _) = locator(None);
    let target = TranscriptTarget::new(locator, "agent", meta());
    let driver = Arc::new(Driver::new(vec![Ok(outcome(vec![Message::assistant(
        "never",
    )]))]));
    let (hook, _) = hook_with_resume(
        vec![ResumePreparation {
            transcript: Some(target),
        }],
        vec![],
    );
    let mut session = SessionBuilder::new(driver.clone())
        .hooks(hook)
        .build()
        .unwrap();
    assert_eq!(
        session
            .turn(
                SessionTurnRequest::new(Message::user("x")),
                TurnOptions::default()
            )
            .await,
        Err(RuntimeError::MissingDependency("TranscriptCodec"))
    );
    assert!(driver.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn resumed_raw_rows_and_metadata_survive_the_append() {
    let mut raw = TranscriptMessage::new("user", "old");
    raw.extra_metadata = Some(serde_json::json!({"native": true}));
    let initial = SessionTranscript {
        meta: meta(),
        messages: vec![raw.clone()],
    };
    let (locator, history) = locator(Some(initial));
    let mut session = SessionBuilder::new(Arc::new(Driver::new(vec![Ok(outcome(vec![
        Message::user("old"),
        Message::assistant("new"),
    ]))])))
    .codec(Arc::new(Codec::default()))
    .transcript(locator, "agent", meta())
    .build()
    .unwrap();
    session
        .turn(
            SessionTurnRequest::new(Message::user("next")),
            TurnOptions {
                session: None,
                resume: ResumeMode::LatestForAgent,
                ..TurnOptions::default()
            },
        )
        .await
        .unwrap();
    let persisted = history.state.lock().unwrap().clone().unwrap();
    assert_eq!(persisted.meta.agent_id.as_deref(), Some("agent-id"));
    assert_eq!(persisted.messages[0], raw);
}

#[tokio::test]
async fn after_commit_receipt_observes_durable_append_and_terminal_follows_it() {
    let (locator, history) = locator(None);
    let (hook, events) = hook(vec![]);
    let mut session = SessionBuilder::new(Arc::new(Driver::new(vec![Ok(outcome(vec![
        Message::assistant("ok"),
    ]))])))
    .codec(Arc::new(Codec::default()))
    .transcript(locator, "agent", meta())
    .hooks(hook)
    .build()
    .unwrap();
    session
        .turn(
            SessionTurnRequest::new(Message::user("x")),
            TurnOptions::default(),
        )
        .await
        .unwrap();
    assert!(history.state.lock().unwrap().is_some());
    assert_eq!(events.receipts.lock().unwrap().len(), 1);
    let receipt = events.receipts.lock().unwrap().pop().unwrap();
    let transcript = receipt.transcript.unwrap();
    assert_eq!(transcript.path, history.path);
    assert!(matches!(
        transcript.delta,
        crate::TranscriptDelta::Append { .. }
    ));
    assert!(matches!(
        events.terminal.lock().unwrap().as_slice(),
        [SessionTerminal::Completed(_)]
    ));
    assert_eq!(
        *events.order.lock().unwrap(),
        vec!["after_commit", "terminal"]
    );
}

#[tokio::test]
async fn receipt_reports_a_compaction_as_replacement_not_an_append_range() {
    let (locator, _) = locator(None);
    let (hook, events) = hook(vec![]);
    let driver = Arc::new(Driver::new(vec![
        Ok(outcome(vec![
            Message::user("old"),
            Message::assistant("one"),
        ])),
        Ok(outcome(vec![Message::assistant("compacted")])),
    ]));
    let mut session = SessionBuilder::new(driver)
        .codec(Arc::new(Codec::default()))
        .transcript(locator, "agent", meta())
        .hooks(hook)
        .build()
        .unwrap();
    session
        .turn(
            SessionTurnRequest::new(Message::user("old")),
            TurnOptions::default(),
        )
        .await
        .unwrap();
    session
        .turn(
            SessionTurnRequest::new(Message::user("new")),
            TurnOptions::default(),
        )
        .await
        .unwrap();
    let receipts = events.receipts.lock().unwrap();
    assert!(matches!(
        receipts[1].transcript.as_ref().unwrap().delta,
        crate::TranscriptDelta::Replace { .. }
    ));
}

#[tokio::test]
async fn failure_and_cancellation_do_not_run_after_commit_and_emit_one_terminal() {
    let (failure_hook, events) = hook(vec![]);
    let mut failed = SessionBuilder::new(Arc::new(Driver::new(vec![Err(DriverFailure {
        error: RuntimeError::Driver("no".into()),
        partial: None,
    })])))
    .hooks(failure_hook)
    .build()
    .unwrap();
    assert!(
        failed
            .turn(
                SessionTurnRequest::new(Message::user("x")),
                TurnOptions::default()
            )
            .await
            .is_err()
    );
    assert!(events.receipts.lock().unwrap().is_empty());
    assert!(matches!(
        events.terminal.lock().unwrap().as_slice(),
        [SessionTerminal::Failed(_)]
    ));

    let (cancel_hook, events) = hook(vec![]);
    let mut cancelled = SessionBuilder::new(Arc::new(Driver::new(vec![])))
        .hooks(cancel_hook)
        .build()
        .unwrap();
    let options = TurnOptions::default();
    options.cancellation.cancel();
    assert_eq!(
        cancelled
            .turn(SessionTurnRequest::new(Message::user("x")), options)
            .await,
        Err(RuntimeError::Cancelled)
    );
    assert!(events.receipts.lock().unwrap().is_empty());
    assert!(matches!(
        events.terminal.lock().unwrap().as_slice(),
        [SessionTerminal::Cancelled]
    ));
}

#[tokio::test]
async fn seed_history_is_the_only_history_and_rejects_a_later_seed() {
    let codec = Arc::new(Codec::default());
    let mut session = SessionBuilder::new(Arc::new(Driver::new(vec![Ok(outcome(vec![
        Message::user("seed"),
        Message::assistant("next"),
    ]))])))
    .codec(codec.clone())
    .build()
    .unwrap();
    let raw = vec![TranscriptMessage::new("user", "seed")];
    session
        .seed_history(vec![Message::user("seed")], raw.clone())
        .unwrap();
    session
        .turn(
            SessionTurnRequest::new(Message::user("seed")),
            TurnOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(codec.seen_prior.lock().unwrap().as_slice(), [raw]);
    assert!(matches!(
        session.seed_history(vec![], vec![]),
        Err(RuntimeError::InvalidSessionState(_))
    ));
}

#[derive(Clone)]
struct Context(String);

struct ContextDriver(Mutex<Vec<String>>);
#[async_trait]
impl SessionDriver<Context> for ContextDriver {
    async fn execute(
        &self,
        request: DriverRequest<Context>,
    ) -> Result<DriverOutcome, DriverFailure> {
        self.0
            .lock()
            .unwrap()
            .push(request.run_context.data.0.clone());
        Ok(outcome(vec![Message::assistant("ok")]))
    }
}

struct ContextCodec(Mutex<Vec<String>>);
impl TranscriptCodec<Context> for ContextCodec {
    fn decode_history(&self, _: &SessionTranscript) -> Result<Vec<Message>, RuntimeError> {
        Ok(vec![])
    }
    fn reconcile(
        &self,
        _: &[TranscriptMessage],
        _: &[Message],
        _: &[Message],
        options: &TranscriptTurnOptions<Context>,
    ) -> Result<Vec<TranscriptMessage>, RuntimeError> {
        self.0.lock().unwrap().push(options.context.0.clone());
        Ok(vec![])
    }
}

struct ContextHook;
#[async_trait]
impl SessionHooks<Context> for ContextHook {
    async fn before_turn(
        &self,
        _: &mut SessionTurnRequest,
        options: &mut TurnOptions<Context>,
        _: SessionStateView<'_>,
    ) -> Result<TurnPreparation, RuntimeError> {
        options.run_context.data = Context("mutated".into());
        Ok(TurnPreparation::default())
    }
    async fn before_commit(
        &self,
        _: &SessionTurnOutcome,
        _: &TranscriptTurnOptions<Context>,
    ) -> Result<(), RuntimeError> {
        Ok(())
    }
    async fn on_terminal(&self, _: SessionTerminal) -> Result<(), RuntimeError> {
        Ok(())
    }
}

#[tokio::test]
async fn hook_option_context_mutation_reaches_driver_and_codec() {
    let (locator, _) = locator(None);
    let driver = Arc::new(ContextDriver(Mutex::new(vec![])));
    let codec = Arc::new(ContextCodec(Mutex::new(vec![])));
    let mut session = SessionBuilder::new(driver.clone())
        .codec(codec.clone())
        .transcript(locator, "agent", meta())
        .hooks(Arc::new(ContextHook))
        .build()
        .unwrap();
    let cancellation = tinyagents_harness::CancellationToken::new();
    session
        .turn(
            SessionTurnRequest::new(Message::user("x")),
            TurnOptions {
                request_id: None,
                thread_id: None,
                stream: false,
                session: None,
                resume: ResumeMode::Never,
                cancellation: cancellation.clone(),
                run_context: RunContext::new(RunConfig::new("test"), Context("before".into()))
                    .with_cancellation(cancellation),
            },
        )
        .await
        .unwrap();
    assert_eq!(driver.0.lock().unwrap().as_slice(), ["mutated"]);
    assert_eq!(codec.0.lock().unwrap().as_slice(), ["mutated"]);
}

struct BeforeResumeContextHook {
    target: TranscriptTarget,
    history: Arc<MemoryHistory>,
    before_turn_saw_lazy_target: Mutex<bool>,
}

#[async_trait]
impl SessionHooks<Context> for BeforeResumeContextHook {
    async fn before_resume(
        &self,
        _: &mut SessionTurnRequest,
        options: &mut TurnOptions<Context>,
        state: SessionStateView<'_>,
    ) -> Result<ResumePreparation, RuntimeError> {
        assert!(state.history.is_empty());
        options.request_id = Some("prepared-before-resume".into());
        options.run_context.data = Context("prepared-before-resume".into());
        Ok(ResumePreparation {
            transcript: Some(self.target.clone()),
        })
    }
    async fn before_turn(
        &self,
        _: &mut SessionTurnRequest,
        options: &mut TurnOptions<Context>,
        state: SessionStateView<'_>,
    ) -> Result<TurnPreparation, RuntimeError> {
        *self.before_turn_saw_lazy_target.lock().unwrap() = state
            .transcript_target
            .is_some_and(|target| target.stem == "late");
        assert_eq!(*self.history.opens.lock().unwrap(), 0);
        assert!(!state.resumed);
        assert_eq!(
            options.request_id.as_deref(),
            Some("prepared-before-resume")
        );
        assert_eq!(options.run_context.data.0, "prepared-before-resume");
        Ok(TurnPreparation::default())
    }
    async fn before_commit(
        &self,
        _: &SessionTurnOutcome,
        _: &TranscriptTurnOptions<Context>,
    ) -> Result<(), RuntimeError> {
        Ok(())
    }
    async fn on_terminal(&self, _: SessionTerminal) -> Result<(), RuntimeError> {
        Ok(())
    }
}

#[tokio::test]
async fn before_resume_mutates_context_and_options_while_target_remains_lazy() {
    let (locator, history) = locator(None);
    let driver = Arc::new(ContextDriver(Mutex::new(vec![])));
    let codec = Arc::new(ContextCodec(Mutex::new(vec![])));
    let hook = Arc::new(BeforeResumeContextHook {
        target: TranscriptTarget::new(locator, "late", meta()),
        history: history.clone(),
        before_turn_saw_lazy_target: Mutex::new(false),
    });
    let mut session = SessionBuilder::new(driver.clone())
        .codec(codec.clone())
        .hooks(hook.clone())
        .build()
        .unwrap();
    let cancellation = tinyagents_harness::CancellationToken::new();
    session
        .turn(
            SessionTurnRequest::new(Message::user("x")),
            TurnOptions {
                request_id: None,
                thread_id: None,
                stream: false,
                session: None,
                resume: ResumeMode::Never,
                cancellation: cancellation.clone(),
                run_context: RunContext::new(RunConfig::new("test"), Context("before".into()))
                    .with_cancellation(cancellation),
            },
        )
        .await
        .unwrap();

    assert!(*hook.before_turn_saw_lazy_target.lock().unwrap());
    assert_eq!(
        driver.0.lock().unwrap().as_slice(),
        ["prepared-before-resume"]
    );
    assert_eq!(
        codec.0.lock().unwrap().as_slice(),
        ["prepared-before-resume"]
    );
    assert_eq!(*history.opens.lock().unwrap(), 1);
}

struct ResumeCancellationHook {
    before_resume_calls: Mutex<usize>,
    before_turn_calls: Mutex<usize>,
    after_commit_calls: Mutex<usize>,
    terminals: Mutex<Vec<SessionTerminal>>,
}

#[async_trait]
impl SessionHooks for ResumeCancellationHook {
    async fn before_resume(
        &self,
        _: &mut SessionTurnRequest,
        _: &mut TurnOptions,
        _: SessionStateView<'_>,
    ) -> Result<ResumePreparation, RuntimeError> {
        *self.before_resume_calls.lock().unwrap() += 1;
        Ok(ResumePreparation::default())
    }
    async fn before_turn(
        &self,
        _: &mut SessionTurnRequest,
        _: &mut TurnOptions,
        _: SessionStateView<'_>,
    ) -> Result<TurnPreparation, RuntimeError> {
        *self.before_turn_calls.lock().unwrap() += 1;
        Ok(TurnPreparation::default())
    }
    async fn before_commit(
        &self,
        _: &SessionTurnOutcome,
        _: &TranscriptTurnOptions,
    ) -> Result<(), RuntimeError> {
        Ok(())
    }
    async fn after_commit(&self, _: CommitReceipt) -> Result<(), RuntimeError> {
        *self.after_commit_calls.lock().unwrap() += 1;
        Ok(())
    }
    async fn on_terminal(&self, terminal: SessionTerminal) -> Result<(), RuntimeError> {
        self.terminals.lock().unwrap().push(terminal);
        Ok(())
    }
}

#[tokio::test]
async fn cancellation_before_resume_skips_hooks_and_preserves_terminal_behavior() {
    let hook = Arc::new(ResumeCancellationHook {
        before_resume_calls: Mutex::new(0),
        before_turn_calls: Mutex::new(0),
        after_commit_calls: Mutex::new(0),
        terminals: Mutex::new(vec![]),
    });
    let driver = Arc::new(Driver::new(vec![Ok(outcome(vec![Message::assistant(
        "never",
    )]))]));
    let mut session = SessionBuilder::new(driver.clone())
        .hooks(hook.clone())
        .build()
        .unwrap();
    let options = TurnOptions::default();
    options.cancellation.cancel();

    assert_eq!(
        session
            .turn(SessionTurnRequest::new(Message::user("x")), options)
            .await,
        Err(RuntimeError::Cancelled)
    );
    assert_eq!(*hook.before_resume_calls.lock().unwrap(), 0);
    assert_eq!(*hook.before_turn_calls.lock().unwrap(), 0);
    assert_eq!(*hook.after_commit_calls.lock().unwrap(), 0);
    assert!(matches!(
        hook.terminals.lock().unwrap().as_slice(),
        [SessionTerminal::Cancelled]
    ));
    assert!(driver.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn cancellation_after_resume_before_before_turn_skips_driver_and_commit() {
    let (locator, history) = locator(Some(SessionTranscript {
        meta: meta(),
        messages: vec![TranscriptMessage::new("user", "old")],
    }));
    let hook = Arc::new(ResumeCancellationHook {
        before_resume_calls: Mutex::new(0),
        before_turn_calls: Mutex::new(0),
        after_commit_calls: Mutex::new(0),
        terminals: Mutex::new(vec![]),
    });
    let driver = Arc::new(Driver::new(vec![Ok(outcome(vec![Message::assistant(
        "never",
    )]))]));
    let mut session = SessionBuilder::new(driver.clone())
        .codec(Arc::new(Codec::default()))
        .transcript(locator, "agent", meta())
        .hooks(hook.clone())
        .build()
        .unwrap();
    let options = TurnOptions {
        resume: ResumeMode::LatestForAgent,
        ..TurnOptions::default()
    };
    *history.cancel_after_read.lock().unwrap() = Some(options.cancellation.clone());

    assert_eq!(
        session
            .turn(SessionTurnRequest::new(Message::user("x")), options)
            .await,
        Err(RuntimeError::Cancelled)
    );
    assert_eq!(*hook.before_resume_calls.lock().unwrap(), 1);
    assert_eq!(*hook.before_turn_calls.lock().unwrap(), 0);
    assert_eq!(*hook.after_commit_calls.lock().unwrap(), 0);
    assert!(matches!(
        hook.terminals.lock().unwrap().as_slice(),
        [SessionTerminal::Cancelled]
    ));
    assert!(driver.requests.lock().unwrap().is_empty());
    assert_eq!(
        history
            .state
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .meta
            .turn_count,
        0
    );
}

#[test]
fn runtime_stays_host_neutral() {
    assert!(
        !include_str!("../Cargo.toml")
            .to_ascii_lowercase()
            .contains("openhuman")
    );
}

// ── Session-identity resume ───────────────────────────────────────────

fn session_turn_options(resume: ResumeMode, thread: &str) -> TurnOptions {
    TurnOptions {
        thread_id: Some(thread.into()),
        resume,
        ..TurnOptions::default()
    }
}

fn session_outcome(history: Vec<Message>, output: &str) -> DriverOutcome {
    DriverOutcome {
        history,
        output: Some(output.into()),
        partial: None,
        interrupted: false,
    }
}

/// The regression this whole design exists for. Two cold sessions on one
/// conversation used to mint two stems, and resume loaded only the newest —
/// so a restart silently dropped the earlier turns. Now the second session
/// reads and appends to the file the first one wrote.
#[tokio::test]
async fn a_restarted_session_continues_the_same_transcript() {
    let directory = tempfile::tempdir().unwrap();
    let session_ref = SessionRef::scoped("thread-9fa08", "agent-id");

    let mut first = SessionBuilder::new(Arc::new(Driver::new(vec![Ok(session_outcome(
        vec![
            Message::user("plan a trip to kashmir"),
            Message::assistant("when?"),
        ],
        "when?",
    ))])))
    .codec(Arc::new(Codec::default()))
    .session(
        Arc::new(FileTranscriptLocator::new(directory.path())),
        session_ref.clone(),
        meta(),
    )
    .build()
    .unwrap();
    first
        .turn(
            SessionTurnRequest::new(Message::user("plan a trip to kashmir")),
            session_turn_options(ResumeMode::Session, "thread-9fa08"),
        )
        .await
        .unwrap();

    // A brand-new Session over the same identity, as a restarted core builds.
    let mut second = SessionBuilder::new(Arc::new(Driver::new(vec![Ok(session_outcome(
        vec![
            Message::user("plan a trip to kashmir"),
            Message::assistant("when?"),
            Message::user("what did i ask here?"),
            Message::assistant("about kashmir"),
        ],
        "about kashmir",
    ))])))
    .codec(Arc::new(Codec::default()))
    .session(
        Arc::new(FileTranscriptLocator::new(directory.path())),
        session_ref.clone(),
        meta(),
    )
    .build()
    .unwrap();
    let resumed = second
        .resume(&session_turn_options(ResumeMode::Session, "thread-9fa08"))
        .await
        .unwrap();

    assert!(resumed.loaded, "the restart must find the first session");
    assert_eq!(
        resumed.history.first().map(Message::text),
        Some("plan a trip to kashmir".to_string()),
        "the opening turn must survive the restart"
    );

    second
        .turn(
            SessionTurnRequest::new(Message::user("what did i ask here?")),
            session_turn_options(ResumeMode::Session, "thread-9fa08"),
        )
        .await
        .unwrap();

    let roots: Vec<_> = std::fs::read_dir(directory.path().join("session_raw"))
        .unwrap()
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().to_string())
        .collect();
    assert_eq!(
        roots.len(),
        1,
        "one conversation, one transcript: {roots:?}"
    );

    let persisted = read_transcript(&directory.path().join("session_raw").join(&roots[0])).unwrap();
    let contents: Vec<&str> = persisted
        .messages
        .iter()
        .map(|message| message.content.as_str())
        .collect();
    assert_eq!(
        contents,
        [
            "plan a trip to kashmir",
            "when?",
            "what did i ask here?",
            "about kashmir"
        ]
    );
    assert_eq!(
        persisted.meta.session_id.as_deref(),
        Some(session_stem(&session_ref).as_str())
    );
}

/// A compaction must not rewrite the sealed file: the turns it drops are the
/// conversation's own history.
#[tokio::test]
async fn a_compaction_opens_the_next_generation_and_leaves_the_sealed_one_intact() {
    let directory = tempfile::tempdir().unwrap();
    let session_ref = SessionRef::scoped("thread-1", "agent-id");
    let locator = Arc::new(FileTranscriptLocator::new(directory.path()));

    let mut session = SessionBuilder::new(Arc::new(Driver::new(vec![
        Ok(session_outcome(
            vec![
                Message::user("one"),
                Message::assistant("first"),
                Message::user("two"),
                Message::assistant("second"),
            ],
            "second",
        )),
        // The driver trimmed: the next set is no longer an extension.
        Ok(session_outcome(
            vec![Message::user("three"), Message::assistant("third")],
            "third",
        )),
    ])))
    .codec(Arc::new(Codec::default()))
    .session(locator.clone(), session_ref.clone(), meta())
    .build()
    .unwrap();

    session
        .turn(
            SessionTurnRequest::new(Message::user("one")),
            session_turn_options(ResumeMode::Session, "thread-1"),
        )
        .await
        .unwrap();
    let sealed = directory
        .path()
        .join("session_raw")
        .join(format!("{}.jsonl", session_stem(&session_ref)));
    let sealed_bytes = std::fs::read(&sealed).unwrap();

    session
        .turn(
            SessionTurnRequest::new(Message::user("three")),
            session_turn_options(ResumeMode::Never, "thread-1"),
        )
        .await
        .unwrap();

    assert_eq!(
        std::fs::read(&sealed).unwrap(),
        sealed_bytes,
        "the sealed generation must be byte-identical after a compaction"
    );
    let successor = directory.path().join("session_raw").join(format!(
        "{}.jsonl",
        session_stem(&session_ref.next_generation())
    ));
    let carried = read_transcript(&successor).unwrap();
    assert_eq!(
        carried
            .messages
            .iter()
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>(),
        ["three", "third"]
    );
    assert_eq!(
        carried.meta.parent_session_id.as_deref(),
        Some(session_stem(&session_ref).as_str())
    );
    assert_eq!(locator.head_generation(&session_ref).generation, 1);
}

/// After a compaction, a restart must land on the head generation rather than
/// replaying the sealed one.
#[tokio::test]
async fn a_restart_after_a_compaction_resumes_the_head_generation() {
    let directory = tempfile::tempdir().unwrap();
    let session_ref = SessionRef::scoped("thread-1", "agent-id");
    let locator = Arc::new(FileTranscriptLocator::new(directory.path()));
    let (_, handle) = locator.begin_generation(&session_ref, meta()).unwrap();
    locator
        .open_session(&session_ref, meta())
        .unwrap()
        .append(TranscriptMessage::new("user", "sealed"))
        .unwrap();
    handle
        .append(TranscriptMessage::new("user", "current"))
        .unwrap();

    let mut session = SessionBuilder::new(Arc::new(Driver::new(Vec::new())))
        .codec(Arc::new(Codec::default()))
        .session(locator, session_ref, meta())
        .build()
        .unwrap();
    let resumed = session
        .resume(&session_turn_options(ResumeMode::Session, "thread-1"))
        .await
        .unwrap();

    assert!(resumed.loaded);
    assert_eq!(
        resumed
            .history
            .iter()
            .map(Message::text)
            .collect::<Vec<_>>(),
        ["current"]
    );
}

/// A session-bound target's write destination is always its own session
/// file — construction and `rebind_session` keep `target.session`/`stem` in
/// lockstep, regardless of resume mode. `ResumeMode::Thread` can legitimately
/// read a *different* file than that destination (its contract is "find
/// this thread's newest root transcript", not "find the session's own
/// file"). The pre-turn append-diff baseline must still reflect what is
/// actually on the write destination — using the scanned read's rows
/// instead would diff the next append against the wrong file's content and
/// corrupt the destination.
#[tokio::test]
async fn thread_resume_on_a_session_bound_target_does_not_corrupt_its_own_destination() {
    let directory = tempfile::tempdir().unwrap();
    let session_ref = SessionRef::scoped("thread-1", "agent-id");
    let locator = Arc::new(FileTranscriptLocator::new(directory.path()));

    // The session's own destination already carries one real message from an
    // earlier session-mode turn.
    locator
        .open_session(&session_ref, meta())
        .unwrap()
        .append(TranscriptMessage::new(
            "user",
            "already on the session file",
        ))
        .unwrap();

    // A newer, *different* root transcript for the same thread, with a
    // *different* message count (2, not 1) — what a Thread-mode newest-wins
    // scan will find and seed `self.history` from instead.
    let mut legacy = meta();
    legacy.thread_id = Some("thread-1".into());
    legacy.created = "zzz-later".into();
    tinyagents_session::transcript::write_transcript(
        &directory.path().join("session_raw/legacy_other.jsonl"),
        &[
            TranscriptMessage::new("user", "legacy one"),
            TranscriptMessage::new("user", "legacy two"),
        ],
        &legacy,
        None,
    )
    .unwrap();

    // The driver extends whatever it was handed by exactly one message.
    let mut session = SessionBuilder::new(Arc::new(Driver::new(vec![Ok(session_outcome(
        vec![
            Message::user("legacy one"),
            Message::user("legacy two"),
            Message::assistant("brand new turn"),
        ],
        "brand new turn",
    ))])))
    .codec(Arc::new(Codec::default()))
    .session(locator.clone(), session_ref.clone(), meta())
    .build()
    .unwrap();

    session
        .turn(
            SessionTurnRequest::new(Message::user("legacy two")),
            session_turn_options(ResumeMode::Thread, "thread-1"),
        )
        .await
        .unwrap();

    // Sensitive invariant: with the append-diff baseline correctly re-derived
    // from the destination's own real prior content (1 message), appending
    // the driver's 1-message extension must leave the destination with
    // exactly as many messages as the model's full candidate history (3) —
    // regardless of what the unrelated scanned-from file contained. Using
    // the scanned file's row count (2) as the wrong baseline instead makes
    // the diff append too few tail rows, silently losing track of one
    // message: this assertion catches exactly that class of bug rather than
    // merely checking "some content survived", which an append-only writer
    // satisfies by construction even when it drops the wrong number of rows.
    let destination_path = directory
        .path()
        .join("session_raw")
        .join(format!("{}.jsonl", session_stem(&session_ref)));
    let on_disk = read_transcript(&destination_path).unwrap();
    assert_eq!(
        on_disk.messages.len(),
        3,
        "on-disk message count must match the full candidate history, not the \
         scanned-from file's unrelated row count: {:?}",
        on_disk
            .messages
            .iter()
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        on_disk.messages[0].content, "already on the session file",
        "the destination's own pre-existing message must never be overwritten"
    );
}

/// `session_binding` is only ever set on the `ResumeMode::Session` path, so
/// without rebinding for every mode, a session-bound target resumed through
/// `Thread`/`LatestForAgent` after an earlier compaction would still bind
/// generation 0 — a generation the design requires to stay sealed and
/// byte-for-byte unchanged — instead of the actual head.
#[tokio::test]
async fn thread_resume_on_a_session_bound_target_writes_the_head_not_a_sealed_generation() {
    let directory = tempfile::tempdir().unwrap();
    let session_ref = SessionRef::scoped("thread-1", "agent-id");
    let locator = Arc::new(FileTranscriptLocator::new(directory.path()));

    // Seal generation 0 and open generation 1, exactly what a prior
    // compaction does.
    locator
        .open_session(&session_ref, meta())
        .unwrap()
        .append(TranscriptMessage::new("user", "sealed generation 0"))
        .unwrap();
    let (_, head_handle) = locator.begin_generation(&session_ref, meta()).unwrap();
    head_handle
        .append(TranscriptMessage::new("user", "head generation 1"))
        .unwrap();
    let sealed_path = directory
        .path()
        .join("session_raw")
        .join(format!("{}.jsonl", session_stem(&session_ref)));
    let sealed_bytes_before = std::fs::read(&sealed_path).unwrap();

    let mut session = SessionBuilder::new(Arc::new(Driver::new(vec![Ok(session_outcome(
        vec![
            Message::user("head generation 1"),
            Message::assistant("new turn"),
        ],
        "new turn",
    ))])))
    .codec(Arc::new(Codec::default()))
    .session(locator.clone(), session_ref.clone(), meta())
    .build()
    .unwrap();

    session
        .turn(
            SessionTurnRequest::new(Message::user("continue")),
            session_turn_options(ResumeMode::Thread, "thread-1"),
        )
        .await
        .unwrap();

    assert_eq!(
        std::fs::read(&sealed_path).unwrap(),
        sealed_bytes_before,
        "generation 0 must stay sealed and byte-identical"
    );
    let head_path = directory.path().join("session_raw").join(format!(
        "{}.jsonl",
        session_stem(&session_ref.next_generation())
    ));
    let head_contents: Vec<String> = read_transcript(&head_path)
        .unwrap()
        .messages
        .into_iter()
        .map(|message| message.content)
        .collect();
    assert!(
        head_contents.contains(&"new turn".to_string()),
        "the new turn must land in the head generation, not the sealed one: {head_contents:?}"
    );
}

/// A conversation written before session identity existed is spread over
/// timestamped stems. The first session resume folds them in, so the model
/// regains the turns newest-wins lookup had stranded.
#[tokio::test]
async fn a_first_session_resume_adopts_a_pre_identity_conversation() {
    let directory = tempfile::tempdir().unwrap();
    let mut legacy = meta();
    legacy.thread_id = Some("thread-9fa08".into());
    legacy.created = "2026-09-22T07:30:47Z".into();
    tinyagents_session::transcript::write_transcript(
        &tinyagents_session::transcript::resolve_keyed_transcript_path(
            directory.path(),
            "1790062247_orchestrator",
        )
        .unwrap(),
        &[TranscriptMessage::new("user", "plan a trip to kashmir")],
        &legacy,
        None,
    )
    .unwrap();

    let mut session = SessionBuilder::new(Arc::new(Driver::new(Vec::new())))
        .codec(Arc::new(Codec::default()))
        .session(
            Arc::new(FileTranscriptLocator::new(directory.path())),
            SessionRef::scoped("thread-9fa08", "agent-id"),
            meta(),
        )
        .build()
        .unwrap();
    let resumed = session
        .resume(&session_turn_options(ResumeMode::Session, "thread-9fa08"))
        .await
        .unwrap();

    assert!(resumed.loaded, "the legacy conversation must be adopted");
    assert_eq!(
        resumed
            .history
            .iter()
            .map(Message::text)
            .collect::<Vec<_>>(),
        ["plan a trip to kashmir"]
    );
}

#[tokio::test]
async fn a_session_resume_with_no_history_anywhere_loads_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let mut session = SessionBuilder::new(Arc::new(Driver::new(Vec::new())))
        .codec(Arc::new(Codec::default()))
        .session(
            Arc::new(FileTranscriptLocator::new(directory.path())),
            SessionRef::scoped("thread-fresh", "agent-id"),
            meta(),
        )
        .build()
        .unwrap();

    let resumed = session
        .resume(&session_turn_options(ResumeMode::Session, "thread-fresh"))
        .await
        .unwrap();

    assert!(!resumed.loaded);
}
