use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use async_trait::async_trait;
use tinyagents_harness::{
    CancellationToken,
    context::{RunConfig, RunContext},
};
use tinyagents_runtime::ToolSnapshot;
use tinyinference_llm::{message::Message, usage::UsageTotals};

use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Action {
    Load,
    Prepare,
    Execute,
    Pause,
    Terminal(SubagentStatusName),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum SubagentStatusName {
    Completed,
    Incomplete,
    Cancelled,
}

#[derive(Clone, Copy)]
enum ExecutorMode {
    Completed,
    Incomplete,
    Pause,
    WaitForCancellation,
    CancelAfterExecution,
    Error,
}

struct FakePlanner {
    calls: Mutex<usize>,
    saw_resume: Mutex<bool>,
    seen_resumes: Mutex<Vec<bool>>,
    reject: bool,
    actions: Arc<Mutex<Vec<Action>>>,
}

#[async_trait]
impl SubagentPlanner<String> for FakePlanner {
    async fn prepare(
        &self,
        request: SubagentRequest<String>,
    ) -> Result<PreparedSubagent<String>, SubagentError> {
        *self.calls.lock().unwrap() += 1;
        *self.saw_resume.lock().unwrap() = request.resume.is_some();
        self.seen_resumes
            .lock()
            .unwrap()
            .push(request.resume.is_some());
        self.actions.lock().unwrap().push(Action::Prepare);
        if self.reject {
            return Err(SubagentError::Planning("rejected".into()));
        }
        Ok(PreparedSubagent {
            task_id: request.task_id,
            agent_key: "resolved-agent".into(),
            input: vec![Message::user(request.input)],
            tools: ToolSnapshot::new(vec![]).unwrap(),
            run_context: request.parent_run,
        })
    }
}

struct FakeExecutor {
    calls: Mutex<usize>,
    context_ids: Mutex<Vec<u64>>,
    context_cancellations: Mutex<Vec<CancellationToken>>,
    mode: ExecutorMode,
    started: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    actions: Arc<Mutex<Vec<Action>>>,
}

#[async_trait]
impl SubagentExecutor<String> for FakeExecutor {
    async fn execute(
        &self,
        execution: SubagentExecution<String>,
    ) -> Result<SubagentOutcome, SubagentError> {
        *self.calls.lock().unwrap() += 1;
        self.context_ids
            .lock()
            .unwrap()
            .push(execution.prepared.run_context.instance_id());
        self.context_cancellations
            .lock()
            .unwrap()
            .push(execution.prepared.run_context.cancellation.clone());
        self.actions.lock().unwrap().push(Action::Execute);
        if matches!(self.mode, ExecutorMode::WaitForCancellation) {
            if let Some(sender) = self.started.lock().unwrap().take() {
                let _ = sender.send(());
            }
            execution.cancellation.cancelled().await;
        }
        if matches!(self.mode, ExecutorMode::CancelAfterExecution) {
            execution.cancellation.cancel();
        }
        if matches!(self.mode, ExecutorMode::Error) {
            return Err(SubagentError::Execution("executor failed".into()));
        }
        let status = match self.mode {
            ExecutorMode::Completed
            | ExecutorMode::WaitForCancellation
            | ExecutorMode::CancelAfterExecution => SubagentStatus::Completed,
            ExecutorMode::Incomplete => SubagentStatus::Incomplete(SubagentIncomplete {
                reason: "budget exhausted".into(),
            }),
            ExecutorMode::Pause => SubagentStatus::AwaitingInput(SubagentPause {
                reason: "need approval".into(),
                resume: SubagentResume::default(),
            }),
            ExecutorMode::Error => unreachable!(),
        };
        Ok(SubagentOutcome {
            task_id: execution.prepared.task_id,
            output: "result".into(),
            history: vec![Message::assistant("result")],
            status,
            usage: UsageTotals {
                calls: 7,
                ..UsageTotals::default()
            },
            artifacts: vec![ArtifactReference {
                id: "artifact-1".into(),
                ..ArtifactReference::default()
            }],
        })
    }
}

enum LoadMode {
    Empty,
    Resume,
    Error,
}

type Fakes = (
    Arc<FakePlanner>,
    Arc<FakeExecutor>,
    Arc<FakePersistence>,
    Arc<Mutex<Vec<Action>>>,
);

struct FakePersistence {
    actions: Arc<Mutex<Vec<Action>>>,
    load_mode: LoadMode,
    pause_error: bool,
    terminal_error: bool,
    outcomes: Mutex<Vec<SubagentOutcome>>,
    saved_pause: Mutex<Option<SubagentResume>>,
    keys: Mutex<Vec<SubagentTaskKey>>,
}

#[async_trait]
impl SubagentPersistence for FakePersistence {
    async fn load(&self, key: &SubagentTaskKey) -> Result<Option<SubagentResume>, SubagentError> {
        self.actions.lock().unwrap().push(Action::Load);
        self.keys.lock().unwrap().push(key.clone());
        match self.load_mode {
            LoadMode::Empty => Ok(self.saved_pause.lock().unwrap().clone()),
            LoadMode::Resume => Ok(Some(SubagentResume {
                checkpoint: Some("saved".into()),
                ..SubagentResume::default()
            })),
            LoadMode::Error => Err(SubagentError::Persistence("load failed".into())),
        }
    }

    async fn save_pause(&self, pause: PersistedSubagentPause) -> Result<(), SubagentError> {
        self.actions.lock().unwrap().push(Action::Pause);
        self.keys.lock().unwrap().push(pause.key);
        if self.pause_error {
            Err(SubagentError::Persistence("pause save failed".into()))
        } else {
            *self.saved_pause.lock().unwrap() = Some(pause.pause.resume);
            Ok(())
        }
    }

    async fn record_terminal(
        &self,
        key: &SubagentTaskKey,
        outcome: &SubagentOutcome,
    ) -> Result<(), SubagentError> {
        self.actions
            .lock()
            .unwrap()
            .push(Action::Terminal(match outcome.status {
                SubagentStatus::Completed => SubagentStatusName::Completed,
                SubagentStatus::Incomplete(_) => SubagentStatusName::Incomplete,
                SubagentStatus::Cancelled => SubagentStatusName::Cancelled,
                SubagentStatus::AwaitingInput(_) => unreachable!(),
            }));
        self.keys.lock().unwrap().push(key.clone());
        if self.terminal_error {
            Err(SubagentError::Persistence("terminal save failed".into()))
        } else {
            self.outcomes.lock().unwrap().push(outcome.clone());
            Ok(())
        }
    }
}

fn request(task_id: &str, data: &str) -> SubagentRequest<String> {
    request_with_parent(
        task_id,
        RunContext::new(RunConfig::new(format!("run-{task_id}")), data.into()),
    )
}

fn request_with_parent(task_id: &str, parent_run: RunContext<String>) -> SubagentRequest<String> {
    SubagentRequest {
        task_id: task_id.into(),
        parent_run,
        input: "do work".into(),
        thread_id: Some("thread-1".into()),
        resume: None,
    }
}

fn driver(
    planner: Arc<dyn SubagentPlanner<String>>,
    executor: Arc<dyn SubagentExecutor<String>>,
    persistence: Arc<dyn SubagentPersistence>,
) -> SubagentDriver<String> {
    SubagentDriver::new(SubagentCapabilities {
        planner: Some(planner),
        executor: Some(executor),
        persistence: Some(persistence),
    })
    .unwrap()
}

fn fakes(mode: ExecutorMode) -> Fakes {
    let actions = Arc::new(Mutex::new(Vec::new()));
    (
        Arc::new(FakePlanner {
            calls: Mutex::new(0),
            saw_resume: Mutex::new(false),
            seen_resumes: Mutex::new(Vec::new()),
            reject: false,
            actions: actions.clone(),
        }),
        Arc::new(FakeExecutor {
            calls: Mutex::new(0),
            context_ids: Mutex::new(Vec::new()),
            context_cancellations: Mutex::new(Vec::new()),
            mode,
            started: Mutex::new(None),
            actions: actions.clone(),
        }),
        Arc::new(FakePersistence {
            actions: actions.clone(),
            load_mode: LoadMode::Empty,
            pause_error: false,
            terminal_error: false,
            outcomes: Mutex::new(Vec::new()),
            saved_pause: Mutex::new(None),
            keys: Mutex::new(Vec::new()),
        }),
        actions,
    )
}

/// Persistence fake whose first selected operation cannot commit until the
/// test releases it. This makes the cancellation/commit boundary observable.
#[derive(Clone, Copy)]
enum BlockingStage {
    Pause,
    Terminal,
}

struct BlockingPersistence {
    stage: BlockingStage,
    started: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
    first: AtomicBool,
    outcomes: Mutex<Vec<SubagentOutcome>>,
}

#[async_trait]
impl SubagentPersistence for BlockingPersistence {
    async fn load(&self, _: &SubagentTaskKey) -> Result<Option<SubagentResume>, SubagentError> {
        Ok(None)
    }

    async fn save_pause(&self, _: PersistedSubagentPause) -> Result<(), SubagentError> {
        if matches!(self.stage, BlockingStage::Pause) && self.first.swap(false, Ordering::AcqRel) {
            self.started.notify_one();
            self.release.notified().await;
        }
        Ok(())
    }

    async fn record_terminal(
        &self,
        _: &SubagentTaskKey,
        outcome: &SubagentOutcome,
    ) -> Result<(), SubagentError> {
        if matches!(self.stage, BlockingStage::Terminal) && self.first.swap(false, Ordering::AcqRel)
        {
            self.started.notify_one();
            self.release.notified().await;
        }
        self.outcomes.lock().unwrap().push(outcome.clone());
        Ok(())
    }
}

struct MismatchedPlanner;

#[async_trait]
impl SubagentPlanner<String> for MismatchedPlanner {
    async fn prepare(
        &self,
        request: SubagentRequest<String>,
    ) -> Result<PreparedSubagent<String>, SubagentError> {
        Ok(PreparedSubagent {
            task_id: "other-task".into(),
            agent_key: "resolved-agent".into(),
            input: vec![Message::user(request.input)],
            tools: ToolSnapshot::new(vec![]).unwrap(),
            run_context: request.parent_run,
        })
    }
}

struct MismatchedExecutor;

#[async_trait]
impl SubagentExecutor<String> for MismatchedExecutor {
    async fn execute(
        &self,
        _: SubagentExecution<String>,
    ) -> Result<SubagentOutcome, SubagentError> {
        Ok(SubagentOutcome {
            task_id: "other-task".into(),
            output: String::new(),
            history: Vec::new(),
            status: SubagentStatus::Completed,
            usage: UsageTotals::default(),
            artifacts: Vec::new(),
        })
    }
}

struct NestedExecutor {
    driver: Mutex<Option<std::sync::Weak<SubagentDriver<String>>>>,
    calls: Mutex<Vec<String>>,
}

struct PauseThenCompleteExecutor {
    calls: Mutex<usize>,
    actions: Arc<Mutex<Vec<Action>>>,
}

#[async_trait]
impl SubagentExecutor<String> for PauseThenCompleteExecutor {
    async fn execute(
        &self,
        execution: SubagentExecution<String>,
    ) -> Result<SubagentOutcome, SubagentError> {
        let call = {
            let mut calls = self.calls.lock().unwrap();
            *calls += 1;
            *calls
        };
        self.actions.lock().unwrap().push(Action::Execute);
        Ok(SubagentOutcome {
            task_id: execution.prepared.task_id,
            output: if call == 1 {
                "waiting".into()
            } else {
                "completed".into()
            },
            history: Vec::new(),
            status: if call == 1 {
                SubagentStatus::AwaitingInput(SubagentPause {
                    reason: "need input".into(),
                    resume: SubagentResume {
                        checkpoint: Some("resume-token".into()),
                        ..SubagentResume::default()
                    },
                })
            } else {
                SubagentStatus::Completed
            },
            usage: UsageTotals::default(),
            artifacts: Vec::new(),
        })
    }
}

#[async_trait]
impl SubagentExecutor<String> for NestedExecutor {
    async fn execute(
        &self,
        execution: SubagentExecution<String>,
    ) -> Result<SubagentOutcome, SubagentError> {
        let task_id = execution.prepared.task_id.clone();
        self.calls.lock().unwrap().push(task_id.clone());
        if task_id == "parent" {
            let child_driver = self
                .driver
                .lock()
                .unwrap()
                .as_ref()
                .and_then(std::sync::Weak::upgrade)
                .expect("nested executor is attached to its driver");
            let child = child_driver
                .run(
                    SubagentRequest {
                        task_id: "child".into(),
                        parent_run: execution.prepared.run_context,
                        input: "nested work".into(),
                        thread_id: None,
                        resume: None,
                    },
                    execution.cancellation,
                )
                .await?;
            return Ok(SubagentOutcome {
                task_id,
                output: "parent result".into(),
                history: Vec::new(),
                status: SubagentStatus::Completed,
                // This models the host's parent-visible roll-up: the child is
                // added once alongside the parent's own model call.
                usage: UsageTotals {
                    calls: child.usage.calls + 1,
                    ..UsageTotals::default()
                },
                artifacts: Vec::new(),
            });
        }
        Ok(SubagentOutcome {
            task_id,
            output: "child result".into(),
            history: Vec::new(),
            status: SubagentStatus::Completed,
            usage: UsageTotals {
                calls: 7,
                ..UsageTotals::default()
            },
            artifacts: Vec::new(),
        })
    }
}

#[tokio::test]
async fn planner_rejection_does_not_execute_or_persist_terminal_state() {
    let (_, executor, persistence, actions) = fakes(ExecutorMode::Completed);
    let rejecting = Arc::new(FakePlanner {
        calls: Mutex::new(0),
        saw_resume: Mutex::new(false),
        seen_resumes: Mutex::new(Vec::new()),
        reject: true,
        actions: actions.clone(),
    });
    let result = driver(rejecting.clone(), executor.clone(), persistence.clone())
        .run(request("task", "ctx"), CancellationToken::new())
        .await;

    assert_eq!(result, Err(SubagentError::Planning("rejected".into())));
    assert_eq!(*executor.calls.lock().unwrap(), 0);
    assert!(persistence.outcomes.lock().unwrap().is_empty());
    assert_eq!(
        *actions.lock().unwrap(),
        vec![Action::Load, Action::Prepare]
    );
}

#[tokio::test]
async fn prepared_context_identity_reaches_executor() {
    let (planner, executor, persistence, _) = fakes(ExecutorMode::Completed);
    let incoming = request("task", "identity");
    let expected = incoming.parent_run.instance_id();
    let outcome = driver(planner, executor.clone(), persistence)
        .run(incoming, CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(outcome.status, SubagentStatus::Completed);
    assert_eq!(*executor.context_ids.lock().unwrap(), vec![expected]);
}

#[tokio::test]
async fn completed_incomplete_and_pause_use_one_mutually_exclusive_persistence_action() {
    for (mode, expected) in [
        (
            ExecutorMode::Completed,
            Action::Terminal(SubagentStatusName::Completed),
        ),
        (
            ExecutorMode::Incomplete,
            Action::Terminal(SubagentStatusName::Incomplete),
        ),
        (ExecutorMode::Pause, Action::Pause),
    ] {
        let (planner, executor, persistence, actions) = fakes(mode);
        driver(planner, executor, persistence)
            .run(request("task", "ctx"), CancellationToken::new())
            .await
            .unwrap();
        let records = actions.lock().unwrap();
        assert_eq!(
            records
                .iter()
                .filter(|action| matches!(action, Action::Pause | Action::Terminal(_)))
                .count(),
            1
        );
        assert_eq!(records.last(), Some(&expected));
    }
}

#[tokio::test]
async fn load_pause_and_terminal_errors_remain_typed() {
    let (planner, executor, persistence, _) = fakes(ExecutorMode::Completed);
    let load_failure = Arc::new(FakePersistence {
        actions: persistence.actions.clone(),
        load_mode: LoadMode::Error,
        pause_error: false,
        terminal_error: false,
        outcomes: Mutex::new(Vec::new()),
        saved_pause: Mutex::new(None),
        keys: Mutex::new(Vec::new()),
    });
    assert_eq!(
        driver(planner.clone(), executor.clone(), load_failure)
            .run(request("load", "ctx"), CancellationToken::new())
            .await,
        Err(SubagentError::Persistence("load failed".into()))
    );

    let (planner, executor, persistence, _) = fakes(ExecutorMode::Pause);
    let pause_failure = Arc::new(FakePersistence {
        pause_error: true,
        actions: persistence.actions.clone(),
        load_mode: LoadMode::Empty,
        terminal_error: false,
        outcomes: Mutex::new(Vec::new()),
        saved_pause: Mutex::new(None),
        keys: Mutex::new(Vec::new()),
    });
    assert_eq!(
        driver(planner, executor, pause_failure)
            .run(request("pause", "ctx"), CancellationToken::new())
            .await,
        Err(SubagentError::Persistence("pause save failed".into()))
    );

    let (planner, executor, persistence, _) = fakes(ExecutorMode::Completed);
    let terminal_failure = Arc::new(FakePersistence {
        terminal_error: true,
        actions: persistence.actions.clone(),
        load_mode: LoadMode::Empty,
        pause_error: false,
        outcomes: Mutex::new(Vec::new()),
        saved_pause: Mutex::new(None),
        keys: Mutex::new(Vec::new()),
    });
    assert_eq!(
        driver(planner, executor, terminal_failure)
            .run(request("terminal", "ctx"), CancellationToken::new())
            .await,
        Err(SubagentError::Persistence("terminal save failed".into()))
    );
}

#[tokio::test]
async fn loaded_resume_reaches_planner_before_execution_and_execution_errors_do_not_persist() {
    let (planner, executor, persistence, actions) = fakes(ExecutorMode::Completed);
    let resume_persistence = Arc::new(FakePersistence {
        actions: persistence.actions.clone(),
        load_mode: LoadMode::Resume,
        pause_error: false,
        terminal_error: false,
        outcomes: Mutex::new(Vec::new()),
        saved_pause: Mutex::new(None),
        keys: Mutex::new(Vec::new()),
    });
    driver(planner.clone(), executor, resume_persistence)
        .run(request("resume", "ctx"), CancellationToken::new())
        .await
        .unwrap();
    assert!(*planner.saw_resume.lock().unwrap());
    assert_eq!(
        actions.lock().unwrap()[..2],
        [Action::Load, Action::Prepare]
    );

    let (planner, executor, persistence, actions) = fakes(ExecutorMode::Error);
    assert_eq!(
        driver(planner, executor, persistence)
            .run(request("execution-error", "ctx"), CancellationToken::new())
            .await,
        Err(SubagentError::Execution("executor failed".into()))
    );
    assert_eq!(
        *actions.lock().unwrap(),
        vec![Action::Load, Action::Prepare, Action::Execute]
    );
}

#[tokio::test]
async fn duplicate_task_id_returns_recorded_outcome_without_second_execution_or_record() {
    let (planner, executor, persistence, _) = fakes(ExecutorMode::Completed);
    let driver = driver(planner.clone(), executor.clone(), persistence.clone());
    let first = driver
        .run(request("same", "one"), CancellationToken::new())
        .await
        .unwrap();
    let second = driver
        .run(request("same", "two"), CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(first, second);
    assert_eq!(*planner.calls.lock().unwrap(), 1);
    assert_eq!(*executor.calls.lock().unwrap(), 1);
    assert_eq!(persistence.outcomes.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn concurrent_same_task_calls_coalesce_to_one_lifecycle() {
    let (planner, executor, persistence, _) = fakes(ExecutorMode::WaitForCancellation);
    let driver = Arc::new(driver(
        planner.clone(),
        executor.clone(),
        persistence.clone(),
    ));
    let cancellation = CancellationToken::new();
    let (started_at_execution, started) = tokio::sync::oneshot::channel();
    *executor.started.lock().unwrap() = Some(started_at_execution);

    let first = tokio::spawn({
        let driver = driver.clone();
        let cancellation = cancellation.clone();
        async move {
            driver
                .run(request("same-in-flight", "one"), cancellation)
                .await
        }
    });
    started.await.unwrap();
    let second = tokio::spawn({
        let driver = driver.clone();
        async move {
            driver
                .run(request("same-in-flight", "two"), CancellationToken::new())
                .await
        }
    });
    tokio::task::yield_now().await;
    cancellation.cancel();

    assert_eq!(
        first.await.unwrap().unwrap().status,
        SubagentStatus::Cancelled
    );
    assert_eq!(
        second.await.unwrap().unwrap().status,
        SubagentStatus::Cancelled
    );
    assert_eq!(*planner.calls.lock().unwrap(), 1);
    assert_eq!(*executor.calls.lock().unwrap(), 1);
    assert_eq!(persistence.outcomes.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn cancelled_follower_returns_without_cancelling_the_leader_or_persisting() {
    let (planner, executor, persistence, _) = fakes(ExecutorMode::WaitForCancellation);
    let driver = Arc::new(driver(
        planner.clone(),
        executor.clone(),
        persistence.clone(),
    ));
    let leader_cancellation = CancellationToken::new();
    let follower_cancellation = CancellationToken::new();
    let (started_at_execution, started) = tokio::sync::oneshot::channel();
    *executor.started.lock().unwrap() = Some(started_at_execution);

    let leader = tokio::spawn({
        let driver = driver.clone();
        let cancellation = leader_cancellation.clone();
        async move {
            driver
                .run(request("follower-cancellation", "leader"), cancellation)
                .await
        }
    });
    started.await.unwrap();
    let follower = tokio::spawn({
        let driver = driver.clone();
        let cancellation = follower_cancellation.clone();
        async move {
            driver
                .run(request("follower-cancellation", "follower"), cancellation)
                .await
        }
    });
    follower_cancellation.cancel();

    let follower_outcome = tokio::time::timeout(Duration::from_secs(1), follower)
        .await
        .expect("cancelled follower must not wait for the leader")
        .unwrap()
        .unwrap();
    assert_eq!(follower_outcome.status, SubagentStatus::Cancelled);
    assert!(!leader_cancellation.is_cancelled());
    assert_eq!(*executor.calls.lock().unwrap(), 1);
    assert!(persistence.outcomes.lock().unwrap().is_empty());

    leader_cancellation.cancel();
    assert_eq!(
        leader.await.unwrap().unwrap().status,
        SubagentStatus::Cancelled
    );
    assert_eq!(persistence.outcomes.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn same_task_id_from_distinct_parent_runs_never_shares_lifecycle_state() {
    let (planner, executor, persistence, _) = fakes(ExecutorMode::Completed);
    let driver = driver(planner.clone(), executor.clone(), persistence.clone());
    let root = RunContext::new(RunConfig::new("root"), "root-data".to_owned());
    let first_parent = root
        .child(RunConfig::new("parent-one"), "first-parent".into())
        .unwrap();
    let second_parent = root
        .child(RunConfig::new("parent-two"), "second-parent".into())
        .unwrap();

    driver
        .run(
            request_with_parent("same-task", first_parent),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    driver
        .run(
            request_with_parent("same-task", second_parent),
            CancellationToken::new(),
        )
        .await
        .unwrap();

    assert_eq!(*planner.calls.lock().unwrap(), 2);
    assert_eq!(*executor.calls.lock().unwrap(), 2);
    assert_eq!(persistence.outcomes.lock().unwrap().len(), 2);
    let keys = persistence.keys.lock().unwrap();
    let terminal_keys = keys
        .iter()
        .filter(|key| key.task_id == "same-task")
        .collect::<Vec<_>>();
    assert_eq!(
        terminal_keys.len(),
        4,
        "load and terminal use each scoped key"
    );
    assert!(
        terminal_keys
            .iter()
            .any(|key| key.parent_run_id == "parent-one")
    );
    assert!(
        terminal_keys
            .iter()
            .any(|key| key.parent_run_id == "parent-two")
    );
    assert!(terminal_keys.iter().all(|key| key.root_run_id == "root"));
}

#[tokio::test]
async fn awaiting_input_is_not_cached_and_the_next_call_resumes_to_completion() {
    let (planner, _, persistence, actions) = fakes(ExecutorMode::Pause);
    let executor = Arc::new(PauseThenCompleteExecutor {
        calls: Mutex::new(0),
        actions: actions.clone(),
    });
    let driver = driver(planner.clone(), executor.clone(), persistence.clone());

    let first = driver
        .run(request("resumable", "first"), CancellationToken::new())
        .await
        .unwrap();
    let second = driver
        .run(request("resumable", "continued"), CancellationToken::new())
        .await
        .unwrap();

    assert!(matches!(first.status, SubagentStatus::AwaitingInput(_)));
    assert_eq!(second.status, SubagentStatus::Completed);
    assert_eq!(*planner.calls.lock().unwrap(), 2);
    assert_eq!(*executor.calls.lock().unwrap(), 2);
    assert_eq!(*planner.seen_resumes.lock().unwrap(), vec![false, true]);
    assert_eq!(
        actions
            .lock()
            .unwrap()
            .iter()
            .filter(|action| matches!(action, Action::Pause | Action::Terminal(_)))
            .cloned()
            .collect::<Vec<_>>(),
        vec![
            Action::Pause,
            Action::Terminal(SubagentStatusName::Completed)
        ]
    );
    assert_eq!(persistence.outcomes.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn driver_replaces_prepared_context_cancellation_with_execution_token() {
    let (planner, executor, persistence, _) = fakes(ExecutorMode::WaitForCancellation);
    let driver = Arc::new(driver(planner, executor.clone(), persistence));
    let cancellation = CancellationToken::new();
    let (started_at_execution, started) = tokio::sync::oneshot::channel();
    *executor.started.lock().unwrap() = Some(started_at_execution);

    let task = tokio::spawn({
        let driver = driver.clone();
        let cancellation = cancellation.clone();
        async move {
            driver
                .run(request("shared-cancellation", "ctx"), cancellation)
                .await
        }
    });
    started.await.unwrap();
    cancellation.cancel();
    let outcome = task.await.unwrap().unwrap();

    assert_eq!(outcome.status, SubagentStatus::Cancelled);
    assert!(executor.context_cancellations.lock().unwrap()[0].is_cancelled());
}

#[tokio::test]
async fn cancellation_before_persistence_commit_records_only_cancelled_terminal() {
    for (stage, mode, task_id) in [
        (
            BlockingStage::Pause,
            ExecutorMode::Pause,
            "cancel-save-pause",
        ),
        (
            BlockingStage::Terminal,
            ExecutorMode::Completed,
            "cancel-record-terminal",
        ),
    ] {
        let (planner, executor, _, _) = fakes(mode);
        let persistence = Arc::new(BlockingPersistence {
            stage,
            started: Arc::new(tokio::sync::Notify::new()),
            release: Arc::new(tokio::sync::Notify::new()),
            first: AtomicBool::new(true),
            outcomes: Mutex::new(Vec::new()),
        });
        let driver = Arc::new(driver(planner, executor, persistence.clone()));
        let cancellation = CancellationToken::new();
        let started = persistence.started.clone();
        let run = tokio::spawn({
            let driver = driver.clone();
            let cancellation = cancellation.clone();
            async move { driver.run(request(task_id, "ctx"), cancellation).await }
        });

        tokio::time::timeout(Duration::from_secs(1), started.notified())
            .await
            .expect("first persistence action must be pending");
        cancellation.cancel();
        let outcome = tokio::time::timeout(Duration::from_secs(1), run)
            .await
            .expect("cancellation must resolve the lifecycle")
            .unwrap()
            .unwrap();

        assert_eq!(outcome.status, SubagentStatus::Cancelled);
        assert_eq!(persistence.outcomes.lock().unwrap().len(), 1);
        assert_eq!(
            persistence.outcomes.lock().unwrap()[0].status,
            SubagentStatus::Cancelled
        );
    }
}

#[tokio::test]
async fn planner_and_executor_task_id_mismatches_do_not_persist_or_cache() {
    let (_, executor, persistence, _) = fakes(ExecutorMode::Completed);
    let planner_error = driver(Arc::new(MismatchedPlanner), executor, persistence.clone())
        .run(request("expected", "ctx"), CancellationToken::new())
        .await;
    assert_eq!(
        planner_error,
        Err(SubagentError::TaskIdMismatch {
            expected: "expected".into(),
            actual: "other-task".into(),
        })
    );
    assert!(persistence.outcomes.lock().unwrap().is_empty());

    let (planner, _, persistence, _) = fakes(ExecutorMode::Completed);
    let executor_error = driver(planner, Arc::new(MismatchedExecutor), persistence.clone())
        .run(request("expected", "ctx"), CancellationToken::new())
        .await;
    assert_eq!(
        executor_error,
        Err(SubagentError::TaskIdMismatch {
            expected: "expected".into(),
            actual: "other-task".into(),
        })
    );
    assert!(persistence.outcomes.lock().unwrap().is_empty());
}

#[tokio::test]
async fn nested_same_driver_task_uses_child_reservation_and_rolls_usage_up_once() {
    let (planner, _, persistence, _) = fakes(ExecutorMode::Completed);
    let executor = Arc::new(NestedExecutor {
        driver: Mutex::new(None),
        calls: Mutex::new(Vec::new()),
    });
    let driver = Arc::new(driver(planner, executor.clone(), persistence.clone()));
    *executor.driver.lock().unwrap() = Some(Arc::downgrade(&driver));

    let parent = tokio::time::timeout(
        Duration::from_secs(1),
        driver.run(request("parent", "ctx"), CancellationToken::new()),
    )
    .await
    .expect("a nested child task must not wait on a driver-global lock")
    .unwrap();

    assert_eq!(parent.usage.calls, 8);
    assert_eq!(*executor.calls.lock().unwrap(), vec!["parent", "child"]);
    let records = persistence.outcomes.lock().unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(
        records
            .iter()
            .find(|outcome| outcome.task_id == "parent")
            .expect("parent outcome is persisted")
            .usage
            .calls,
        8
    );
}

#[tokio::test]
async fn cancellation_before_execution_records_one_truthful_terminal() {
    let (planner, executor, persistence, actions) = fakes(ExecutorMode::Completed);
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let outcome = driver(planner.clone(), executor.clone(), persistence)
        .run(request("cancel-before", "ctx"), cancellation)
        .await
        .unwrap();

    assert_eq!(outcome.status, SubagentStatus::Cancelled);
    assert_eq!(*planner.calls.lock().unwrap(), 0);
    assert_eq!(*executor.calls.lock().unwrap(), 0);
    assert_eq!(
        *actions.lock().unwrap(),
        vec![Action::Terminal(SubagentStatusName::Cancelled)]
    );
}

#[tokio::test]
async fn cancellation_during_execution_is_truthful_and_terminal_once() {
    let (planner, executor, persistence, actions) = fakes(ExecutorMode::WaitForCancellation);
    let driver = Arc::new(driver(planner, executor.clone(), persistence));
    let cancellation = CancellationToken::new();
    let (started_at_execution, started) = tokio::sync::oneshot::channel();
    *executor.started.lock().unwrap() = Some(started_at_execution);
    let task = tokio::spawn({
        let driver = driver.clone();
        let cancellation = cancellation.clone();
        async move {
            driver
                .run(request("cancel-during", "ctx"), cancellation)
                .await
        }
    });
    started.await.unwrap();
    cancellation.cancel();
    let outcome = task.await.unwrap().unwrap();

    assert_eq!(outcome.status, SubagentStatus::Cancelled);
    assert_eq!(*executor.calls.lock().unwrap(), 1);
    assert_eq!(
        actions.lock().unwrap().last(),
        Some(&Action::Terminal(SubagentStatusName::Cancelled))
    );
}

#[tokio::test]
async fn cancellation_after_execution_preserves_lossless_result_data() {
    let (planner, executor, persistence, _) = fakes(ExecutorMode::CancelAfterExecution);
    let outcome = driver(planner, executor, persistence)
        .run(request("cancel-after", "ctx"), CancellationToken::new())
        .await
        .unwrap();

    assert_eq!(outcome.status, SubagentStatus::Cancelled);
    assert_eq!(outcome.output, "result");
    assert_eq!(outcome.history, vec![Message::assistant("result")]);
    assert_eq!(outcome.usage.calls, 7);
    assert_eq!(outcome.artifacts.len(), 1);
}

#[tokio::test]
async fn nested_usage_is_persisted_once_and_failure_order_is_load_prepare_execute_then_terminal() {
    let (planner, executor, persistence, actions) = fakes(ExecutorMode::Completed);
    driver(planner, executor, persistence.clone())
        .run(request("usage", "ctx"), CancellationToken::new())
        .await
        .unwrap();

    let outcomes = persistence.outcomes.lock().unwrap();
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].usage.calls, 7);
    assert_eq!(
        *actions.lock().unwrap(),
        vec![
            Action::Load,
            Action::Prepare,
            Action::Execute,
            Action::Terminal(SubagentStatusName::Completed),
        ]
    );
}

#[test]
fn absent_host_capabilities_fail_closed() {
    let result = SubagentDriver::<String>::new(SubagentCapabilities {
        planner: None,
        executor: None,
        persistence: None,
    });
    let error = match result {
        Err(error) => error,
        Ok(_) => panic!("missing host capabilities must fail closed"),
    };
    assert_eq!(error, SubagentError::MissingCapability("planner"));
}
