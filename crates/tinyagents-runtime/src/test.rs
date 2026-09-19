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
    DisplayRecord, FileTranscriptHistory, FileTranscriptLocator, SessionTranscript,
    TranscriptHistory, TranscriptLocator, TranscriptMessage, TranscriptMeta, TranscriptRead,
    TranscriptTurn, read_transcript, read_transcript_display,
};
use tinyinference_llm::message::Message;
use tinyinference_llm::providers::MockModel;
use tinytools::{Tool, ToolResult, ToolSpec};

use crate::{
    DriverFailure, DriverOutcome, DriverRequest, HarnessDriver, PrefixSnapshot, ResumeMode,
    RuntimeError, SessionBuilder, SessionDriver, SessionHooks, SessionTerminal, SessionTurnOutcome,
    SessionTurnRequest, ToolSnapshot, TranscriptCodec, TurnOptions,
};

struct FakeDriver {
    results: Mutex<VecDeque<Result<DriverOutcome, DriverFailure>>>,
    requests: Mutex<Vec<DriverRequest>>,
}

impl FakeDriver {
    fn new(results: Vec<Result<DriverOutcome, DriverFailure>>) -> Self {
        Self {
            results: Mutex::new(results.into()),
            requests: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl SessionDriver for FakeDriver {
    async fn execute(&self, request: DriverRequest) -> Result<DriverOutcome, DriverFailure> {
        self.requests.lock().unwrap().push(request);
        self.results
            .lock()
            .unwrap()
            .pop_front()
            .expect("planned driver result")
    }
}

struct WaitingDriver;

#[async_trait]
impl SessionDriver for WaitingDriver {
    async fn execute(&self, _: DriverRequest) -> Result<DriverOutcome, DriverFailure> {
        std::future::pending().await
    }
}

struct DropDriver {
    started: Arc<tokio::sync::Notify>,
}

#[async_trait]
impl SessionDriver for DropDriver {
    async fn execute(&self, _: DriverRequest) -> Result<DriverOutcome, DriverFailure> {
        self.started.notify_waiters();
        std::future::pending().await
    }
}

// Reconciliation retains an explicit clone for the codec after the live
// `RunContext` is consumed by the driver.
#[derive(Clone)]
struct HostContext(String);

struct ContextDriver;

#[async_trait]
impl SessionDriver<HostContext> for ContextDriver {
    async fn execute(
        &self,
        request: DriverRequest<HostContext>,
    ) -> Result<DriverOutcome, DriverFailure> {
        Ok(outcome(vec![Message::assistant(
            request.run_context.data.0,
        )]))
    }
}

struct RegisteredTool;

#[async_trait]
impl Tool for RegisteredTool {
    fn name(&self) -> &str {
        "registered"
    }

    fn description(&self) -> &str {
        "a registered matching tool"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }

    async fn execute(&self, _: serde_json::Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::success("ok"))
    }
}

#[derive(Default)]
struct BasicCodec {
    decoded: Mutex<Vec<SessionTranscript>>,
}

impl TranscriptCodec for BasicCodec {
    fn decode_history(&self, transcript: &SessionTranscript) -> Result<Vec<Message>, RuntimeError> {
        self.decoded.lock().unwrap().push(transcript.clone());
        Ok(transcript
            .messages
            .iter()
            .map(|message| match message.role.as_str() {
                "assistant" => Message::assistant(&message.content),
                "system" => Message::system(&message.content),
                _ => Message::user(&message.content),
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
        // Preserve rows that still correspond to the prior model prefix;
        // fresh model suffixes receive only the codec's explicit projection.
        let raw_offset = (!prior.is_empty())
            .then(|| {
                previous.windows(prior.len()).position(|window| {
                    window
                        .iter()
                        .zip(prior)
                        .all(|(model, row)| model.text() == row.content)
                })
            })
            .flatten()
            .unwrap_or(usize::MAX);
        Ok(next
            .iter()
            .enumerate()
            .map(|(index, message)| {
                let raw_index = index.checked_sub(raw_offset);
                if let Some(raw_index) = raw_index.filter(|index| {
                    *index < prior.len() && previous.get(index + raw_offset) == Some(message)
                }) {
                    return prior[raw_index].clone();
                }
                let role = match message {
                    Message::System(_) => "system",
                    Message::User(_) => "user",
                    Message::Assistant(_) => "assistant",
                    Message::Tool(_) => "tool",
                };
                TranscriptMessage::new(role, message.text())
            })
            .collect())
    }
}

#[derive(Default)]
struct RecordingHooks {
    events: Mutex<Vec<String>>,
    fail_before: bool,
    fail_after: bool,
    terminal_notified: Option<Arc<tokio::sync::Notify>>,
}

struct WaitingAfterHooks {
    started: tokio::sync::Notify,
    events: Mutex<Vec<String>>,
}

struct WaitingBeforeHooks {
    started: tokio::sync::Notify,
}

struct WaitingPostCommitHooks {
    started: tokio::sync::Notify,
    terminal: tokio::sync::Notify,
    terminals: Mutex<Vec<SessionTerminal>>,
}

#[async_trait]
impl SessionHooks for WaitingBeforeHooks {
    async fn before_turn(&self, _: &mut SessionTurnRequest) -> Result<(), RuntimeError> {
        self.started.notify_waiters();
        std::future::pending().await
    }

    async fn after_turn(&self, _: &SessionTurnOutcome) -> Result<(), RuntimeError> {
        Ok(())
    }

    async fn on_terminal(&self, _: &SessionTerminal) -> Result<(), RuntimeError> {
        Ok(())
    }
}

#[async_trait]
impl SessionHooks for WaitingAfterHooks {
    async fn before_turn(&self, _: &mut SessionTurnRequest) -> Result<(), RuntimeError> {
        Ok(())
    }

    async fn after_turn(&self, _: &SessionTurnOutcome) -> Result<(), RuntimeError> {
        self.started.notify_waiters();
        std::future::pending().await
    }

    async fn on_terminal(&self, terminal: &SessionTerminal) -> Result<(), RuntimeError> {
        self.events.lock().unwrap().push(format!("{terminal:?}"));
        Ok(())
    }
}

#[async_trait]
impl SessionHooks for WaitingPostCommitHooks {
    async fn before_turn(&self, _: &mut SessionTurnRequest) -> Result<(), RuntimeError> {
        Ok(())
    }

    async fn after_turn(&self, _: &SessionTurnOutcome) -> Result<(), RuntimeError> {
        Ok(())
    }

    async fn after_commit(&self, _: &SessionTurnOutcome) -> Result<(), RuntimeError> {
        self.started.notify_waiters();
        std::future::pending().await
    }

    async fn on_terminal(&self, terminal: &SessionTerminal) -> Result<(), RuntimeError> {
        self.terminals.lock().unwrap().push(terminal.clone());
        self.terminal.notify_waiters();
        Ok(())
    }
}

#[async_trait]
impl SessionHooks for RecordingHooks {
    async fn before_turn(&self, _: &mut SessionTurnRequest) -> Result<(), RuntimeError> {
        self.events.lock().unwrap().push("before".into());
        if self.fail_before {
            Err(RuntimeError::Hook("before".into()))
        } else {
            Ok(())
        }
    }
    async fn after_turn(&self, _: &SessionTurnOutcome) -> Result<(), RuntimeError> {
        self.events.lock().unwrap().push("after".into());
        if self.fail_after {
            Err(RuntimeError::Hook("after".into()))
        } else {
            Ok(())
        }
    }
    async fn on_terminal(&self, terminal: &SessionTerminal) -> Result<(), RuntimeError> {
        let terminal = match terminal {
            SessionTerminal::Completed(_) => "Completed".to_owned(),
            other => format!("{other:?}"),
        };
        self.events
            .lock()
            .unwrap()
            .push(format!("terminal:{terminal}"));
        if let Some(notify) = &self.terminal_notified {
            notify.notify_waiters();
        }
        Ok(())
    }
}

#[derive(Default)]
struct FinalizationHooks {
    post_commits: Mutex<Vec<SessionTurnOutcome>>,
    terminals: Mutex<Vec<SessionTerminal>>,
    fail_post_commit: bool,
    cancel_on_post_commit: Option<tinyagents_harness::CancellationToken>,
}

#[async_trait]
impl SessionHooks for FinalizationHooks {
    async fn before_turn(&self, _: &mut SessionTurnRequest) -> Result<(), RuntimeError> {
        Ok(())
    }

    async fn after_turn(&self, _: &SessionTurnOutcome) -> Result<(), RuntimeError> {
        Ok(())
    }

    async fn after_commit(&self, outcome: &SessionTurnOutcome) -> Result<(), RuntimeError> {
        self.post_commits.lock().unwrap().push(outcome.clone());
        if let Some(cancellation) = &self.cancel_on_post_commit {
            cancellation.cancel();
        }
        if self.fail_post_commit {
            Err(RuntimeError::Hook("post-commit".into()))
        } else {
            Ok(())
        }
    }

    async fn on_terminal(&self, terminal: &SessionTerminal) -> Result<(), RuntimeError> {
        self.terminals.lock().unwrap().push(terminal.clone());
        Ok(())
    }
}

struct MemoryHistory {
    path: PathBuf,
    session: Mutex<Option<SessionTranscript>>,
    turns: Mutex<Vec<Vec<TranscriptMessage>>>,
    fail: bool,
    cancel_after_append: Mutex<Option<tinyagents_harness::CancellationToken>>,
}

impl TranscriptRead for MemoryHistory {
    fn path(&self) -> &Path {
        &self.path
    }
    fn read_session(&self) -> anyhow::Result<Option<SessionTranscript>> {
        Ok(self.session.lock().unwrap().clone())
    }
}

impl TranscriptHistory for MemoryHistory {
    fn append_turn(&self, turn: TranscriptTurn<'_>) -> anyhow::Result<()> {
        if self.fail {
            anyhow::bail!("planned persistence failure");
        }
        self.turns.lock().unwrap().push(turn.next.to_vec());
        *self.session.lock().unwrap() = Some(SessionTranscript {
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
            .session
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

struct MemoryLocator {
    history: Arc<MemoryHistory>,
}

/// A real transcript path whose single atomic turn operation fails before it
/// writes. This catches the old two-write partial path: if runtime wrote a
/// display partial first, the path below would exist after the failure.
struct FailingFileHistory(FileTranscriptHistory);

impl TranscriptRead for FailingFileHistory {
    fn path(&self) -> &Path {
        self.0.path()
    }

    fn read_session(&self) -> anyhow::Result<Option<SessionTranscript>> {
        TranscriptRead::read_session(&self.0)
    }
}

impl TranscriptHistory for FailingFileHistory {
    fn append_turn(&self, _: TranscriptTurn<'_>) -> anyhow::Result<()> {
        anyhow::bail!("planned atomic append failure")
    }

    fn messages(&self) -> anyhow::Result<Vec<TranscriptMessage>> {
        TranscriptHistory::messages(&self.0)
    }

    fn append(&self, _: TranscriptMessage) -> anyhow::Result<()> {
        anyhow::bail!("planned atomic append failure")
    }

    fn replace(&self, _: &[TranscriptMessage]) -> anyhow::Result<()> {
        anyhow::bail!("planned atomic append failure")
    }

    fn clear(&self) -> anyhow::Result<()> {
        anyhow::bail!("planned atomic append failure")
    }
}

struct FailingFileLocator {
    history: Arc<FailingFileHistory>,
}

impl TranscriptLocator for FailingFileLocator {
    fn latest_for_agent(&self, _: &str) -> Option<Arc<dyn TranscriptRead>> {
        Some(self.history.clone())
    }

    fn root_for_thread(&self, _: &str) -> Option<Arc<dyn TranscriptRead>> {
        Some(self.history.clone())
    }

    fn open_stem(&self, _: &str, _: TranscriptMeta) -> anyhow::Result<Arc<dyn TranscriptHistory>> {
        Ok(self.history.clone())
    }
}

impl TranscriptLocator for MemoryLocator {
    fn latest_for_agent(&self, _: &str) -> Option<Arc<dyn TranscriptRead>> {
        Some(self.history.clone())
    }
    fn root_for_thread(&self, _: &str) -> Option<Arc<dyn TranscriptRead>> {
        Some(self.history.clone())
    }
    fn open_stem(&self, _: &str, _: TranscriptMeta) -> anyhow::Result<Arc<dyn TranscriptHistory>> {
        Ok(self.history.clone())
    }
}

fn meta() -> TranscriptMeta {
    TranscriptMeta {
        agent_name: "agent".into(),
        agent_id: None,
        agent_type: None,
        dispatcher: "test".into(),
        provider: None,
        model: None,
        created: "now".into(),
        updated: "now".into(),
        turn_count: 0,
        input_tokens: 0,
        output_tokens: 0,
        cached_input_tokens: 0,
        charged_amount_usd: 0.0,
        thread_id: None,
        task_id: None,
    }
}

fn outcome(history: Vec<Message>) -> DriverOutcome {
    DriverOutcome {
        history,
        output: Some("done".into()),
        partial: None,
        interrupted: false,
    }
}

fn memory_locator(
    session: Option<SessionTranscript>,
    fail: bool,
) -> (Arc<MemoryLocator>, Arc<MemoryHistory>) {
    let dir = tempfile::tempdir().unwrap().keep();
    let history = Arc::new(MemoryHistory {
        path: dir.join("session.jsonl"),
        session: Mutex::new(session),
        turns: Mutex::new(Vec::new()),
        fail,
        cancel_after_append: Mutex::new(None),
    });
    (
        Arc::new(MemoryLocator {
            history: history.clone(),
        }),
        history,
    )
}

#[tokio::test]
async fn first_turn_commits_history_and_preserves_prefix() {
    let driver = Arc::new(FakeDriver::new(vec![Ok(outcome(vec![
        Message::system("stable"),
        Message::user("hi"),
        Message::assistant("hello"),
    ]))]));
    let mut session = SessionBuilder::new(driver.clone())
        .prefix(PrefixSnapshot::new(vec![Message::system("stable")]))
        .build()
        .unwrap();
    let committed = session
        .turn(
            SessionTurnRequest::new(Message::user("hi")),
            TurnOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(committed.history, session.history());
    assert_eq!(
        session.prefix_snapshot().messages(),
        &[Message::system("stable")]
    );
    assert_eq!(
        driver.requests.lock().unwrap()[0].history,
        vec![Message::system("stable"), Message::user("hi")]
    );
}

#[tokio::test]
async fn generic_session_passes_the_explicit_host_context_to_driver() {
    let mut session = SessionBuilder::<HostContext>::new(Arc::new(ContextDriver))
        .build()
        .unwrap();
    let options = TurnOptions {
        request_id: None,
        thread_id: None,
        stream: false,
        resume: ResumeMode::Never,
        cancellation: tinyagents_harness::CancellationToken::new(),
        run_context: RunContext::new(RunConfig::new("host"), HostContext("host context".into())),
    };
    let result = session
        .turn(SessionTurnRequest::new(Message::user("x")), options)
        .await
        .unwrap();
    assert_eq!(
        result.history.last().map(Message::text).as_deref(),
        Some("host context")
    );
}

#[tokio::test]
async fn harness_driver_uses_the_pinned_explicit_model_entry_point() {
    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::constant("from harness")));
    harness.register_tool(Arc::new(RegisteredTool));
    let snapshot = ToolSnapshot::new(harness.tools().declared_specs()).unwrap();
    let driver = Arc::new(HarnessDriver::new(Arc::new(harness), Arc::new(())));
    let mut session = SessionBuilder::new(driver)
        .tool_snapshot(snapshot)
        .build()
        .unwrap();
    let result = session
        .turn(
            SessionTurnRequest::new(Message::user("hello")),
            TurnOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(result.output.as_deref(), Some("from harness"));
    assert_eq!(
        result.history.last().map(Message::text).as_deref(),
        Some("from harness")
    );
}

#[tokio::test]
async fn trailing_input_is_deduplicated_and_tool_snapshot_is_immutable() {
    let driver = Arc::new(FakeDriver::new(vec![
        Ok(outcome(vec![Message::user("same")])),
        Ok(outcome(vec![
            Message::user("same"),
            Message::assistant("two"),
        ])),
    ]));
    let tools = ToolSnapshot::new(vec![ToolSpec {
        name: "echo".into(),
        description: "x".into(),
        parameters: serde_json::json!({}),
    }])
    .unwrap();
    let mut session = SessionBuilder::new(driver.clone())
        .tool_snapshot(tools)
        .build()
        .unwrap();
    session
        .turn(
            SessionTurnRequest::new(Message::user("same")),
            TurnOptions::default(),
        )
        .await
        .unwrap();
    session
        .turn(
            SessionTurnRequest::new(Message::user("same")),
            TurnOptions::default(),
        )
        .await
        .unwrap();
    let requests = driver.requests.lock().unwrap();
    assert_eq!(requests[1].history, vec![Message::user("same")]);
    assert_eq!(requests[0].tools.specs()[0].name, "echo");
    assert_eq!(session.tool_snapshot().specs()[0].name, "echo");
}

#[test]
fn tool_snapshots_dedup_identical_names_and_reject_collisions() {
    let spec = ToolSpec {
        name: "read".into(),
        description: "read".into(),
        parameters: serde_json::json!({}),
    };
    assert_eq!(
        ToolSnapshot::new(vec![spec.clone(), spec])
            .unwrap()
            .specs()
            .len(),
        1
    );
    assert!(matches!(
        ToolSnapshot::new(vec![
            ToolSpec {
                name: "read".into(),
                description: "one".into(),
                parameters: serde_json::json!({})
            },
            ToolSpec {
                name: "read".into(),
                description: "two".into(),
                parameters: serde_json::json!({})
            },
        ]),
        Err(RuntimeError::ToolNameCollision(name)) if name == "read"
    ));
}

#[tokio::test]
async fn resume_passes_full_durable_transcript_to_codec() {
    let mut durable = TranscriptMessage::new("user", "persisted");
    durable.extra_metadata = Some(serde_json::json!({"unmodified": true}));
    let transcript = SessionTranscript {
        meta: meta(),
        messages: vec![durable.clone()],
    };
    let (locator, _) = memory_locator(Some(transcript), false);
    let codec = Arc::new(BasicCodec::default());
    let mut session = SessionBuilder::new(Arc::new(FakeDriver::new(vec![])))
        .codec(codec.clone())
        .transcript(locator, "agent", meta())
        .build()
        .unwrap();
    let resumed = session
        .resume(&TurnOptions {
            resume: ResumeMode::LatestForAgent,
            ..TurnOptions::default()
        })
        .await
        .unwrap();
    assert!(resumed.loaded);
    assert_eq!(resumed.history, vec![Message::user("persisted")]);
    assert_eq!(
        codec.decoded.lock().unwrap()[0].messages[0].extra_metadata,
        durable.extra_metadata
    );
}

#[tokio::test]
async fn append_only_delta_and_failure_rollback_are_owned_by_session() {
    let (locator, history) = memory_locator(None, false);
    let codec = Arc::new(BasicCodec::default());
    let driver = Arc::new(FakeDriver::new(vec![
        Ok(outcome(vec![Message::user("one"), Message::assistant("a")])),
        Ok(outcome(vec![Message::assistant("compacted")])),
    ]));
    let mut session = SessionBuilder::new(driver)
        .codec(codec)
        .transcript(locator, "agent", meta())
        .build()
        .unwrap();
    session
        .turn(
            SessionTurnRequest::new(Message::user("one")),
            TurnOptions::default(),
        )
        .await
        .unwrap();
    session
        .turn(
            SessionTurnRequest::new(Message::user("two")),
            TurnOptions::default(),
        )
        .await
        .unwrap();
    {
        let turns = history.turns.lock().unwrap();
        assert_eq!(turns.len(), 2);
        assert_eq!(
            turns[1],
            vec![TranscriptMessage::new("assistant", "compacted")]
        );
    }

    let (bad_locator, _) = memory_locator(None, true);
    let bad = Arc::new(FakeDriver::new(vec![Ok(outcome(vec![Message::user(
        "will-not-commit",
    )]))]));
    let mut failing = SessionBuilder::new(bad)
        .codec(Arc::new(BasicCodec::default()))
        .transcript(bad_locator, "bad", meta())
        .build()
        .unwrap();
    assert!(matches!(
        failing
            .turn(
                SessionTurnRequest::new(Message::user("will-not-commit")),
                TurnOptions::default()
            )
            .await,
        Err(RuntimeError::Persistence(_))
    ));
    assert!(failing.history().is_empty());
}

#[tokio::test]
async fn partial_failure_never_leaves_a_display_partial_on_disk() {
    let directory = tempfile::tempdir().unwrap();
    let history = Arc::new(FailingFileHistory(
        FileTranscriptHistory::new(directory.path(), "agent", meta()).unwrap(),
    ));
    let path = history.path().to_path_buf();
    let locator = Arc::new(FailingFileLocator {
        history: history.clone(),
    });
    let driver = Arc::new(FakeDriver::new(vec![Err(DriverFailure {
        error: RuntimeError::Driver("interrupted".into()),
        partial: Some(DriverOutcome {
            history: vec![Message::assistant("partial model history")],
            output: None,
            partial: Some(crate::TranscriptPartial::new("display partial")),
            interrupted: true,
        }),
    })]));
    let mut session = SessionBuilder::new(driver)
        .codec(Arc::new(BasicCodec::default()))
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
    assert!(!path.exists(), "failed partial turn wrote {path:?}");
    assert!(session.history().is_empty());
}

#[tokio::test]
async fn partial_failure_commits_model_history_and_display_partial_together() {
    let directory = tempfile::tempdir().unwrap();
    let locator = Arc::new(FileTranscriptLocator::new(directory.path()));
    let driver = Arc::new(FakeDriver::new(vec![Err(DriverFailure {
        error: RuntimeError::Driver("interrupted".into()),
        partial: Some(DriverOutcome {
            history: vec![Message::assistant("recoverable model history")],
            output: None,
            partial: Some(crate::TranscriptPartial {
                content: "display partial".into(),
                reasoning_content: Some("thinking".into()),
                iteration: Some(3),
            }),
            interrupted: true,
        }),
    })]));
    let mut session = SessionBuilder::new(driver)
        .codec(Arc::new(BasicCodec::default()))
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
        Err(RuntimeError::Driver(_))
    ));

    let path = directory.path().join("session_raw/agent.jsonl");
    let model = read_transcript(&path).unwrap();
    assert_eq!(
        model
            .messages
            .iter()
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>(),
        vec!["recoverable model history"]
    );
    let display = read_transcript_display(&path).unwrap();
    assert!(display.records.iter().any(|record| matches!(
        record,
        DisplayRecord::Message(message)
            if message.interrupted
                && message.message.content == "display partial"
                && message.reasoning_content.as_deref() == Some("thinking")
                && message.iteration == Some(3)
    )));
    assert_eq!(
        session.history(),
        &[Message::assistant("recoverable model history")]
    );
}

#[tokio::test]
async fn resume_new_turn_retains_durable_metadata_and_restores_prefix_once() {
    let mut durable = TranscriptMessage::new("user", "persisted");
    durable.extra_metadata = Some(serde_json::json!({"provider": {"raw": true}}));
    let transcript = SessionTranscript {
        meta: meta(),
        messages: vec![durable.clone()],
    };
    let (locator, history) = memory_locator(Some(transcript), false);
    let driver = Arc::new(FakeDriver::new(vec![Ok(outcome(vec![
        // A compaction/faulty driver omitted the stable prefix.
        Message::user("persisted"),
        Message::assistant("new"),
    ]))]));
    let mut session = SessionBuilder::new(driver.clone())
        .codec(Arc::new(BasicCodec::default()))
        .prefix(PrefixSnapshot::new(vec![Message::system("stable")]))
        .transcript(locator, "agent", meta())
        .build()
        .unwrap();
    session
        .turn(
            SessionTurnRequest::new(Message::user("next")),
            TurnOptions {
                resume: ResumeMode::LatestForAgent,
                ..TurnOptions::default()
            },
        )
        .await
        .unwrap();
    let committed = history.session.lock().unwrap().clone().unwrap();
    assert_eq!(session.history()[0], Message::system("stable"));
    assert_eq!(
        driver.requests.lock().unwrap()[0].history[0],
        Message::system("stable")
    );
    assert_eq!(committed.messages[1].extra_metadata, durable.extra_metadata);
    assert_eq!(
        session
            .history()
            .iter()
            .filter(|m| **m == Message::system("stable"))
            .count(),
        1
    );
}

#[tokio::test]
async fn prefix_reconciliation_uses_maximal_suffix_prefix_overlap() {
    let prefix = PrefixSnapshot::new(vec![
        Message::system("stable-a"),
        Message::system("stable-b"),
        Message::system("stable-c"),
    ]);
    for (returned, expected) in [
        (
            vec![Message::system("stable-c"), Message::assistant("partial")],
            vec![
                Message::system("stable-a"),
                Message::system("stable-b"),
                Message::system("stable-c"),
                Message::assistant("partial"),
            ],
        ),
        (
            vec![
                Message::system("stable-a"),
                Message::system("stable-b"),
                Message::system("stable-c"),
                Message::assistant("complete"),
            ],
            vec![
                Message::system("stable-a"),
                Message::system("stable-b"),
                Message::system("stable-c"),
                Message::assistant("complete"),
            ],
        ),
        (
            vec![Message::assistant("missing")],
            vec![
                Message::system("stable-a"),
                Message::system("stable-b"),
                Message::system("stable-c"),
                Message::assistant("missing"),
            ],
        ),
    ] {
        let mut session =
            SessionBuilder::new(Arc::new(FakeDriver::new(vec![Ok(outcome(returned))])))
                .prefix(prefix.clone())
                .build()
                .unwrap();
        let committed = session
            .turn(
                SessionTurnRequest::new(Message::user("x")),
                TurnOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(committed.history, expected);
    }
}

#[tokio::test]
async fn hooks_are_ordered_terminal_once_and_driver_errors_are_terminal() {
    let hooks = Arc::new(RecordingHooks::default());
    let driver = Arc::new(FakeDriver::new(vec![Ok(outcome(vec![
        Message::assistant("ok"),
    ]))]));
    let mut session = SessionBuilder::new(driver)
        .hooks(hooks.clone())
        .build()
        .unwrap();
    session
        .turn(
            SessionTurnRequest::new(Message::user("x")),
            TurnOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(
        *hooks.events.lock().unwrap(),
        vec!["before", "after", "terminal:Completed"]
    );

    let hooks = Arc::new(RecordingHooks::default());
    let driver = Arc::new(FakeDriver::new(vec![Err(DriverFailure {
        error: RuntimeError::Driver("boom".into()),
        partial: None,
    })]));
    let mut session = SessionBuilder::new(driver)
        .hooks(hooks.clone())
        .build()
        .unwrap();
    assert!(matches!(
        session
            .turn(
                SessionTurnRequest::new(Message::user("x")),
                TurnOptions::default()
            )
            .await,
        Err(RuntimeError::Driver(_))
    ));
    assert_eq!(
        *hooks.events.lock().unwrap(),
        vec!["before", "terminal:Failed(\"driver failed: boom\")"]
    );
}

#[tokio::test]
async fn failed_precommit_hook_and_tool_mismatch_leave_no_commit() {
    let hooks = Arc::new(RecordingHooks {
        fail_after: true,
        ..RecordingHooks::default()
    });
    let (locator, history) = memory_locator(None, false);
    let mut session = SessionBuilder::new(Arc::new(FakeDriver::new(vec![Ok(outcome(vec![
        Message::assistant("x"),
    ]))])))
    .codec(Arc::new(BasicCodec::default()))
    .hooks(hooks)
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
        Err(RuntimeError::Hook(_))
    ));
    assert!(history.turns.lock().unwrap().is_empty());
    assert!(session.history().is_empty());

    let mut harness: AgentHarness<()> = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::constant("unreachable")));
    let mut mismatched = SessionBuilder::new(Arc::new(HarnessDriver::new(
        Arc::new(harness),
        Arc::new(()),
    )))
    .tool_snapshot(
        ToolSnapshot::new(vec![ToolSpec {
            name: "not-registered".into(),
            description: "x".into(),
            parameters: serde_json::json!({}),
        }])
        .unwrap(),
    )
    .build()
    .unwrap();
    assert!(matches!(
        mismatched
            .turn(
                SessionTurnRequest::new(Message::user("x")),
                TurnOptions::default()
            )
            .await,
        Err(RuntimeError::ToolSnapshotMismatch)
    ));
}

#[tokio::test]
async fn cancellation_at_driver_await_emits_one_terminal_hook() {
    let hooks = Arc::new(RecordingHooks::default());
    let mut session = SessionBuilder::new(Arc::new(WaitingDriver))
        .hooks(hooks.clone())
        .build()
        .unwrap();
    let options = TurnOptions::default();
    let cancellation = options.cancellation.clone();
    tokio::spawn(async move {
        tokio::task::yield_now().await;
        cancellation.cancel();
    });
    assert_eq!(
        session
            .turn(SessionTurnRequest::new(Message::user("x")), options)
            .await,
        Err(RuntimeError::Cancelled)
    );
    assert_eq!(
        *hooks.events.lock().unwrap(),
        vec!["before", "terminal:Cancelled"]
    );
}

#[tokio::test]
async fn dropped_turn_future_observes_one_failed_terminal() {
    let started = Arc::new(tokio::sync::Notify::new());
    let terminal = Arc::new(tokio::sync::Notify::new());
    let hooks = Arc::new(RecordingHooks {
        terminal_notified: Some(terminal.clone()),
        ..RecordingHooks::default()
    });
    let mut session = SessionBuilder::new(Arc::new(DropDriver {
        started: started.clone(),
    }))
    .hooks(hooks.clone())
    .build()
    .unwrap();
    let entered = started.notified();
    let observed = terminal.notified();
    let task = tokio::spawn(async move {
        session
            .turn(
                SessionTurnRequest::new(Message::user("x")),
                TurnOptions::default(),
            )
            .await
    });
    entered.await;
    task.abort();
    let _ = task.await;
    observed.await;
    assert_eq!(
        *hooks.events.lock().unwrap(),
        vec!["before", "terminal:Failed(\"session turn dropped\")"]
    );
}

#[tokio::test]
async fn dropped_turn_after_commit_keeps_a_truthful_completed_terminal() {
    let (locator, history) = memory_locator(None, false);
    let hooks = Arc::new(WaitingPostCommitHooks {
        started: tokio::sync::Notify::new(),
        terminal: tokio::sync::Notify::new(),
        terminals: Mutex::new(Vec::new()),
    });
    let mut session = SessionBuilder::new(Arc::new(FakeDriver::new(vec![Ok(outcome(vec![
        Message::assistant("durable"),
    ]))])))
    .codec(Arc::new(BasicCodec::default()))
    .hooks(hooks.clone())
    .transcript(locator, "agent", meta())
    .build()
    .unwrap();
    let entered = hooks.started.notified();
    let observed = hooks.terminal.notified();
    let task = tokio::spawn(async move {
        session
            .turn(
                SessionTurnRequest::new(Message::user("x")),
                TurnOptions::default(),
            )
            .await
    });
    entered.await;
    assert_eq!(history.turns.lock().unwrap().len(), 1);
    task.abort();
    let _ = task.await;
    observed.await;
    assert!(matches!(
        hooks.terminals.lock().unwrap().as_slice(),
        [SessionTerminal::Completed(outcome)] if outcome.history.last() == Some(&Message::assistant("durable"))
    ));
}

#[tokio::test]
async fn cancellation_during_before_hook_never_starts_or_commits_a_turn() {
    let hooks = Arc::new(WaitingBeforeHooks {
        started: tokio::sync::Notify::new(),
    });
    let driver = Arc::new(FakeDriver::new(vec![Ok(outcome(vec![
        Message::assistant("nope"),
    ]))]));
    let mut session = SessionBuilder::new(driver.clone())
        .hooks(hooks.clone())
        .build()
        .unwrap();
    let options = TurnOptions::default();
    let cancellation = options.cancellation.clone();
    let started = hooks.started.notified();
    let turn = tokio::spawn(async move {
        session
            .turn(SessionTurnRequest::new(Message::user("x")), options)
            .await
    });
    started.await;
    cancellation.cancel();
    assert_eq!(turn.await.unwrap(), Err(RuntimeError::Cancelled));
    assert!(driver.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn cancellation_during_precommit_hook_has_no_durable_commit() {
    let hooks = Arc::new(WaitingAfterHooks {
        started: tokio::sync::Notify::new(),
        events: Mutex::new(Vec::new()),
    });
    let (locator, history) = memory_locator(None, false);
    let mut session = SessionBuilder::new(Arc::new(FakeDriver::new(vec![Ok(outcome(vec![
        Message::assistant("candidate"),
    ]))])))
    .codec(Arc::new(BasicCodec::default()))
    .hooks(hooks.clone())
    .transcript(locator, "agent", meta())
    .build()
    .unwrap();
    let options = TurnOptions::default();
    let cancellation = options.cancellation.clone();
    let started = hooks.started.notified();
    let turn = tokio::spawn(async move {
        session
            .turn(SessionTurnRequest::new(Message::user("x")), options)
            .await
    });
    started.await;
    cancellation.cancel();
    assert_eq!(turn.await.unwrap(), Err(RuntimeError::Cancelled));
    assert!(history.turns.lock().unwrap().is_empty());
    assert_eq!(*hooks.events.lock().unwrap(), vec!["Cancelled"]);
}

#[tokio::test]
async fn cancellation_signalled_by_successful_commit_cannot_relabel_the_turn() {
    let (locator, history) = memory_locator(None, false);
    let driver = Arc::new(FakeDriver::new(vec![Ok(outcome(vec![
        Message::assistant("done"),
    ]))]));
    let mut session = SessionBuilder::new(driver)
        .codec(Arc::new(BasicCodec::default()))
        .transcript(locator, "agent", meta())
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
    assert_eq!(history.turns.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn post_commit_runs_once_after_durability_and_cannot_relabel_success() {
    let (locator, history) = memory_locator(None, false);
    let cancellation = tinyagents_harness::CancellationToken::new();
    let hooks = Arc::new(FinalizationHooks {
        fail_post_commit: true,
        cancel_on_post_commit: Some(cancellation.clone()),
        ..FinalizationHooks::default()
    });
    let mut session = SessionBuilder::new(Arc::new(FakeDriver::new(vec![Ok(outcome(vec![
        Message::assistant("done"),
    ]))])))
    .codec(Arc::new(BasicCodec::default()))
    .hooks(hooks.clone())
    .transcript(locator, "agent", meta())
    .build()
    .unwrap();
    let options = TurnOptions {
        cancellation,
        ..TurnOptions::default()
    };
    let committed = session
        .turn(SessionTurnRequest::new(Message::user("x")), options)
        .await
        .expect("post-commit failure/cancellation cannot revoke a durable success");
    assert_eq!(history.turns.lock().unwrap().len(), 1);
    assert_eq!(*hooks.post_commits.lock().unwrap(), vec![committed.clone()]);
    assert!(matches!(
        hooks.terminals.lock().unwrap().as_slice(),
        [SessionTerminal::Completed(outcome)] if outcome == &committed
    ));
}

#[tokio::test]
async fn persistence_failure_never_calls_post_commit() {
    let (locator, _) = memory_locator(None, true);
    let hooks = Arc::new(FinalizationHooks::default());
    let mut session = SessionBuilder::new(Arc::new(FakeDriver::new(vec![Ok(outcome(vec![
        Message::assistant("never durable"),
    ]))])))
    .codec(Arc::new(BasicCodec::default()))
    .hooks(hooks.clone())
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
    assert!(hooks.post_commits.lock().unwrap().is_empty());
    assert!(matches!(
        hooks.terminals.lock().unwrap().as_slice(),
        [SessionTerminal::Failed(_)]
    ));
}

type SeenCodecOptions = (String, Option<String>, Option<String>, bool, ResumeMode);

struct OptionsCodec {
    seen: Mutex<Vec<SeenCodecOptions>>,
}

impl TranscriptCodec<HostContext> for OptionsCodec {
    fn decode_history(&self, _: &SessionTranscript) -> Result<Vec<Message>, RuntimeError> {
        Ok(Vec::new())
    }

    fn reconcile(
        &self,
        _: &[TranscriptMessage],
        _: &[Message],
        next: &[Message],
        options: &crate::TranscriptTurnOptions<HostContext>,
    ) -> Result<Vec<TranscriptMessage>, RuntimeError> {
        self.seen.lock().unwrap().push((
            options.context.0.clone(),
            options.request_id.clone(),
            options.thread_id.clone(),
            options.stream,
            options.resume,
        ));
        Ok(next
            .iter()
            .map(|message| TranscriptMessage::assistant(message.text()))
            .collect())
    }
}

#[tokio::test]
async fn codec_reconciliation_receives_explicit_host_context_and_turn_options() {
    let (locator, _) = memory_locator(None, false);
    let codec = Arc::new(OptionsCodec {
        seen: Mutex::new(Vec::new()),
    });
    let mut session = SessionBuilder::<HostContext>::new(Arc::new(ContextDriver))
        .codec(codec.clone())
        .transcript(locator, "agent", meta())
        .build()
        .unwrap();
    let cancellation = tinyagents_harness::CancellationToken::new();
    session
        .turn(
            SessionTurnRequest::new(Message::user("x")),
            TurnOptions {
                request_id: Some("request-1".into()),
                thread_id: Some("thread-1".into()),
                stream: true,
                resume: ResumeMode::LatestForAgent,
                cancellation: cancellation.clone(),
                run_context: RunContext::new(
                    RunConfig::new("codec-host"),
                    HostContext("host-owned context".into()),
                )
                .with_cancellation(cancellation),
            },
        )
        .await
        .unwrap();
    assert_eq!(
        *codec.seen.lock().unwrap(),
        vec![(
            "host-owned context".into(),
            Some("request-1".into()),
            Some("thread-1".into()),
            true,
            ResumeMode::LatestForAgent,
        )]
    );
}

#[test]
fn dependency_boundary_has_no_product_dependency() {
    let manifest = include_str!("../Cargo.toml");
    assert!(manifest.contains("tinyagents-harness"));
    assert!(manifest.contains("tinyagents-session"));
    assert!(manifest.contains("tinytools"));
    assert!(!manifest.to_ascii_lowercase().contains("openhuman"));
}
