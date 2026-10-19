use std::{
    collections::HashMap,
    future::Future,
    hash::Hash,
    sync::{Arc, Weak},
    time::Duration,
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, mpsc};
use tokio::time::Instant;

use crate::progress::AgentProgress;
use crate::run_queue::{QueueLane, QueueMode, QueueStatus, QueuedMessage, RunQueue};

/// Identity that isolates session locks and active turns between agents.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ScopeKey {
    pub agent_id: String,
    pub profile_id: Option<String>,
}

/// A turn that currently holds a cancellation slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveTurn {
    pub scope: ScopeKey,
    pub thread_id: String,
    pub request_id: String,
    /// Whether it is a parallel turn rather than the thread's primary turn.
    pub parallel: bool,
}

/// A request after the host has supplied its channel-independent metadata.
#[derive(Debug, Clone)]
pub struct TurnRequest<P = serde_json::Value> {
    pub scope: ScopeKey,
    pub thread_id: String,
    pub request_id: String,
    pub text: String,
    /// Lossless host-defined input metadata such as attachments, access flags,
    /// locale, or channel attribution.
    pub payload: P,
    pub host_authored: bool,
    pub disposition: QueueDisposition,
    pub deadline: Option<Duration>,
    pub idle_timeout: Option<Duration>,
}

/// Host payloads that can stand in for the metadata of a steer the harness
/// recovered and pushed back onto the run queue.
///
/// The steering forwarder re-creates a recovered item through
/// [`QueuedMessage::requeued`], which knows only a fresh id, the text, the
/// thread label and the clock. [`TurnRequest`] implements that trait for every
/// payload that can build such a placeholder, so a turn's run queue can be the
/// harness's own queue. The runner restores the scope and thread of a recovered
/// item when it dispatches it as a fresh turn and lets the payload adopt the
/// metadata of the turn that was running.
pub trait RequeuePayload {
    /// A placeholder payload for a recovered item.
    fn requeued(thread_label: &str, queued_at_ms: u64) -> Self;

    /// Fill what a recovered placeholder lacks from the payload of the turn
    /// that was running when it was recovered. Payloads that are already
    /// complete must be left untouched.
    fn adopt(&mut self, _running: &Self) {}
}

impl RequeuePayload for serde_json::Value {
    fn requeued(_thread_label: &str, _queued_at_ms: u64) -> Self {
        serde_json::Value::Null
    }
}

impl<P: RequeuePayload + Send + 'static> QueuedMessage for TurnRequest<P> {
    fn id(&self) -> &str {
        &self.request_id
    }

    fn text(&self) -> &str {
        &self.text
    }

    fn requeued(id: String, text: String, thread_label: &str, queued_at_ms: u64) -> Self {
        Self {
            scope: ScopeKey {
                agent_id: String::new(),
                profile_id: None,
            },
            thread_id: thread_label.to_owned(),
            request_id: id,
            text,
            payload: P::requeued(thread_label, queued_at_ms),
            host_authored: false,
            disposition: QueueDisposition::Steer,
            deadline: None,
            idle_timeout: None,
        }
    }
}

/// Host-owned identity and fingerprint for the checked-out durable session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnSession {
    pub session_id: String,
    pub fingerprint: Option<String>,
}

/// Pipeline stages are called in this order for every submitted request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngressStage {
    StageAttachments,
    Guardrail,
    ApprovalInterception,
    BeforeSubmit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IngressDecision {
    Continue,
    /// The policy consumed the request, for example by routing an approval reply.
    Consumed,
    Reject(String),
}

#[async_trait]
pub trait IngressPolicy<P = serde_json::Value>: Send + Sync {
    async fn apply(
        &self,
        stage: IngressStage,
        request: &mut TurnRequest<P>,
    ) -> Result<IngressDecision, String>;
}

/// The five acceptance behaviors exposed by channel turn APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueDisposition {
    Interrupt,
    Steer,
    Followup,
    Collect,
    Parallel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminalEvent {
    Completed {
        request_id: String,
        reply_id: String,
    },
    Failed {
        request_id: String,
        message: String,
        class: String,
        /// The raw host or runner error the failure was classified from, for
        /// hosts that render richer copy than `class` and `message` carry.
        error: String,
    },
    Cancelled {
        request_id: String,
    },
}

/// Activity updates sent by the host so the idle guard distinguishes silence
/// from a model or tool call that is still running.
#[derive(Debug, Clone)]
pub enum TurnSignal {
    Progress(Box<AgentProgress>),
    WorkStarted,
    WorkFinished,
}

#[async_trait]
pub trait TurnHost<P = serde_json::Value>: Send + Sync {
    /// Check out or create the session for this request and verify its fingerprint.
    async fn checkout(&self, request: &TurnRequest<P>) -> Result<TurnSession, String>;
    /// Check out an independent transcript/session snapshot for Parallel mode.
    async fn checkout_fork(&self, request: &TurnRequest<P>) -> Result<TurnSession, String>;
    /// Confirm the checked-out session still matches the request's expected
    /// transcript/session identity before any model work begins.
    async fn validate_fingerprint(
        &self,
        _request: &TurnRequest<P>,
        _session: &TurnSession,
    ) -> Result<(), String> {
        Ok(())
    }
    /// Build the host's origin value (channel, actor and request attribution).
    fn run_origin(&self, request: &TurnRequest<P>) -> serde_json::Value;
    /// Run the actual harness/session turn. Progress is forwarded to the sink.
    async fn run(
        &self,
        request: &TurnRequest<P>,
        session: &TurnSession,
        origin: &serde_json::Value,
        cancellation: crate::CancellationToken,
        queue: Arc<RunQueue<TurnRequest<P>>>,
        progress: mpsc::Sender<TurnSignal>,
    ) -> Result<TurnResult, String>;
    /// Release the session after the run, including on cancellation or failure.
    async fn checkin(&self, request: &TurnRequest<P>);
    /// Cancel child runs belonging to this parent request.
    async fn cancel_children(&self, request: &TurnRequest<P>);
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TurnResult {
    pub reply: String,
    #[serde(default)]
    pub data: serde_json::Value,
}

#[async_trait]
pub trait ReplyStore<P = serde_json::Value>: Send + Sync {
    async fn persist(
        &self,
        request: &TurnRequest<P>,
        reply_id: &str,
        result: &TurnResult,
    ) -> Result<(), String>;
}

#[async_trait]
pub trait EventSink<P = serde_json::Value>: Send + Sync
where
    P: Send + Sync,
{
    async fn progress(&self, request: &TurnRequest<P>, event: AgentProgress);
    async fn terminal(&self, request: &TurnRequest<P>, event: TerminalEvent);
    async fn suggestions(&self, _request: &TurnRequest<P>, _suggestions: Vec<String>) {}
    async fn collected(&self, _request: &TurnRequest<P>, _items: Vec<TurnRequest<P>>) {}
}

pub trait FailureClassifier: Send + Sync {
    fn classify(&self, error: &str) -> (String, String);
}

#[async_trait]
pub trait Summarizer<P = serde_json::Value>: Send + Sync {
    /// Optional next-message suggestions for a successfully completed turn.
    async fn suggestions(&self, _request: &TurnRequest<P>, _result: &TurnResult) -> Vec<String> {
        Vec::new()
    }
    /// Ground a bounded close in the work completed so far after a guard fires.
    async fn grounded_close(&self, request: &TurnRequest<P>, reason: &str) -> Option<TurnResult>;
}

/// Runner guard defaults. The idle timeout catches silence; the deadline is a
/// separate total wall-clock ceiling and can be overridden per request.
#[derive(Debug, Clone, Copy)]
pub struct TurnRunnerConfig {
    pub idle_timeout: Duration,
    pub deadline: Option<Duration>,
    pub close_timeout: Duration,
    /// How long a cancelled host run keeps being polled, with its own
    /// cancellation token already fired, before it is dropped. A host uses the
    /// window to hand back and persist the rounds it completed. Zero drops the
    /// run immediately.
    pub cancel_grace: Duration,
}

impl Default for TurnRunnerConfig {
    fn default() -> Self {
        Self {
            idle_timeout: Duration::from_secs(900),
            deadline: None,
            close_timeout: Duration::from_secs(60),
            cancel_grace: Duration::ZERO,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TurnRunnerError {
    #[error("turn was consumed by ingress policy")]
    Consumed,
    #[error("turn was accepted into the host queue")]
    Queued,
    #[error("turn rejected: {0}")]
    Rejected(String),
    #[error("turn failed: {0}")]
    Failed(String),
    #[error("turn cancelled")]
    Cancelled,
}

#[derive(Clone)]
struct ActiveTurn<P> {
    request_id: String,
    cancel: crate::CancellationToken,
    generation: u64,
    queue: Arc<RunQueue<TurnRequest<P>>>,
}

type ThreadLockKey = (ScopeKey, String);
type ThreadLocks = Mutex<HashMap<ThreadLockKey, Weak<Mutex<()>>>>;
type ParallelQueueKey = (ScopeKey, String, String);
type ParallelQueueMap<P> = Mutex<HashMap<ParallelQueueKey, Arc<RunQueue<TurnRequest<P>>>>>;

fn queue_lane(disposition: QueueDisposition) -> Option<QueueLane> {
    match disposition {
        QueueDisposition::Steer => Some(QueueLane::Steer),
        QueueDisposition::Followup => Some(QueueLane::Followup),
        QueueDisposition::Collect => Some(QueueLane::Collect),
        QueueDisposition::Interrupt | QueueDisposition::Parallel => None,
    }
}

/// Reusable turn lifecycle coordinator, parameterized entirely by host ports.
pub struct TurnRunner<H, I, R, E, F, S, P = serde_json::Value> {
    host: Arc<H>,
    ingress: Arc<I>,
    replies: Arc<R>,
    events: Arc<E>,
    failures: Arc<F>,
    summarizer: Arc<S>,
    config: TurnRunnerConfig,
    locks: ThreadLocks,
    parallel_queues: ParallelQueueMap<P>,
    active: Mutex<HashMap<(ScopeKey, String), ActiveTurn<P>>>,
    parallel_active: Mutex<HashMap<(ScopeKey, String, String), crate::CancellationToken>>,
    next_generation: std::sync::atomic::AtomicU64,
}

impl<H, I, R, E, F, S, P> TurnRunner<H, I, R, E, F, S, P>
where
    H: TurnHost<P> + 'static,
    I: IngressPolicy<P> + 'static,
    R: ReplyStore<P> + 'static,
    E: EventSink<P> + 'static,
    F: FailureClassifier + 'static,
    S: Summarizer<P> + 'static,
    P: Clone + Send + Sync + RequeuePayload + 'static,
{
    pub fn new(
        host: Arc<H>,
        ingress: Arc<I>,
        replies: Arc<R>,
        events: Arc<E>,
        failures: Arc<F>,
        summarizer: Arc<S>,
        config: TurnRunnerConfig,
    ) -> Self {
        Self {
            host,
            ingress,
            replies,
            events,
            failures,
            summarizer,
            config,
            locks: Mutex::new(HashMap::new()),
            parallel_queues: Mutex::new(HashMap::new()),
            active: Mutex::new(HashMap::new()),
            parallel_active: Mutex::new(HashMap::new()),
            next_generation: std::sync::atomic::AtomicU64::new(1),
        }
    }

    /// Cancel only the active generation named by `request_id`; a stale cancel
    /// cannot target a newer turn that reused the same thread.
    pub async fn cancel(
        &self,
        scope: &ScopeKey,
        thread_id: &str,
        request_id: Option<&str>,
    ) -> bool {
        !self
            .cancel_requests(scope, thread_id, request_id)
            .await
            .is_empty()
    }

    /// [`cancel`](Self::cancel), returning the id of every turn it stopped: the
    /// primary turn first, then each parallel turn.
    pub async fn cancel_requests(
        &self,
        scope: &ScopeKey,
        thread_id: &str,
        request_id: Option<&str>,
    ) -> Vec<String> {
        let key = (scope.clone(), thread_id.to_owned());
        let mut cancelled = Vec::new();
        let active = self.active.lock().await;
        if let Some(turn) = active
            .get(&key)
            .filter(|turn| request_id.is_none_or(|id| turn.request_id == id))
        {
            turn.queue.clear().await;
            turn.cancel.cancel();
            cancelled.push(turn.request_id.clone());
        }
        drop(active);
        let mut parallel = self.parallel_active.lock().await;
        let keys: Vec<_> = parallel
            .keys()
            .filter(|(owner, thread, req)| {
                owner == scope && thread == thread_id && request_id.is_none_or(|id| req == id)
            })
            .cloned()
            .collect();
        for key in &keys {
            if let Some(token) = parallel.remove(key) {
                token.cancel();
            }
            self.parallel_queues.lock().await.remove(key);
            cancelled.push(key.2.clone());
        }
        cancelled
    }

    /// Every turn that currently holds a cancellation slot, primary and parallel.
    pub async fn live_turns(&self) -> Vec<LiveTurn> {
        let mut turns: Vec<LiveTurn> = self
            .active
            .lock()
            .await
            .iter()
            .map(|((scope, thread_id), turn)| LiveTurn {
                scope: scope.clone(),
                thread_id: thread_id.clone(),
                request_id: turn.request_id.clone(),
                parallel: false,
            })
            .collect();
        turns.extend(self.parallel_active.lock().await.keys().map(
            |(scope, thread_id, request_id)| LiveTurn {
                scope: scope.clone(),
                thread_id: thread_id.clone(),
                request_id: request_id.clone(),
                parallel: true,
            },
        ));
        turns
    }

    /// Pending items of the thread's active turn queue, oldest first per lane.
    pub async fn queue_snapshot(
        &self,
        scope: &ScopeKey,
        thread_id: &str,
    ) -> Vec<(QueueLane, TurnRequest<P>)> {
        let key = (scope.clone(), thread_id.to_owned());
        let queue = self
            .active
            .lock()
            .await
            .get(&key)
            .map(|turn| Arc::clone(&turn.queue));
        match queue {
            Some(queue) => queue.snapshot().await,
            None => Vec::new(),
        }
    }

    /// Whether the thread owns a primary user-authored turn cancellation slot.
    pub async fn is_in_flight(&self, scope: &ScopeKey, thread_id: &str) -> bool {
        self.active
            .lock()
            .await
            .contains_key(&(scope.clone(), thread_id.to_owned()))
    }

    pub async fn in_flight_request_id(&self, scope: &ScopeKey, thread_id: &str) -> Option<String> {
        self.active
            .lock()
            .await
            .get(&(scope.clone(), thread_id.to_owned()))
            .map(|turn| turn.request_id.clone())
    }

    pub async fn run(&self, request: TurnRequest<P>) -> Result<TurnResult, TurnRunnerError> {
        let mut state = None;
        let mut next = None;
        let result = self
            .run_one(request.clone(), None, &mut state, &mut next)
            .await;
        if result.is_ok() && request.disposition != QueueDisposition::Parallel {
            while let Some((followup, reserved)) = next.take() {
                let mut current = Some(reserved.clone());
                let mut following = None;
                if self
                    .run_one(
                        followup.clone(),
                        Some(reserved),
                        &mut current,
                        &mut following,
                    )
                    .await
                    .is_err()
                {
                    if let Some(state) = current.as_ref() {
                        self.clear_generation_queue(
                            &followup.scope,
                            &followup.thread_id,
                            state.generation,
                        )
                        .await;
                        self.clear_tracking(&followup, state.generation).await;
                    }
                    break;
                }
                next = following;
            }
        } else {
            if let Some(state) = state.as_ref() {
                if result.is_err() {
                    self.clear_generation_queue(
                        &request.scope,
                        &request.thread_id,
                        state.generation,
                    )
                    .await;
                }
                self.clear_tracking(&request, state.generation).await;
            } else if request.disposition == QueueDisposition::Parallel {
                self.clear_parallel_tracking(&request).await;
            }
        }
        result
    }

    async fn run_one(
        &self,
        mut request: TurnRequest<P>,
        reservation: Option<ActiveTurn<P>>,
        state_out: &mut Option<ActiveTurn<P>>,
        next_out: &mut Option<(TurnRequest<P>, ActiveTurn<P>)>,
    ) -> Result<TurnResult, TurnRunnerError> {
        let deadline = TurnDeadline::new(request.deadline.or(self.config.deadline));
        let cancellation = reservation
            .as_ref()
            .map(|state| state.cancel.clone())
            .unwrap_or_default();
        if reservation.is_none() {
            for stage in [
                IngressStage::StageAttachments,
                IngressStage::Guardrail,
                IngressStage::ApprovalInterception,
                IngressStage::BeforeSubmit,
            ] {
                match self
                    .before_deadline(
                        deadline.at,
                        &cancellation,
                        self.ingress.apply(stage, &mut request),
                    )
                    .await
                {
                    Ok(Ok(IngressDecision::Continue)) => {}
                    Ok(Ok(IngressDecision::Consumed)) => return Err(TurnRunnerError::Consumed),
                    Ok(Ok(IngressDecision::Reject(reason))) => {
                        return Err(TurnRunnerError::Rejected(reason));
                    }
                    Ok(Err(error)) => return self.fail_generation(&request, error, None).await,
                    Err(PreRunStop::Cancelled) => {
                        return self.cancel_generation(&request, None).await;
                    }
                    Err(PreRunStop::Deadline) => {
                        return self
                            .fail_generation(&request, "turn wall-clock deadline".into(), None)
                            .await;
                    }
                }
            }
        }

        let key = (request.scope.clone(), request.thread_id.clone());
        let mut primary_queue = reservation.as_ref().map(|state| state.queue.clone());
        let mut active_generation = reservation.as_ref().map(|state| state.generation);
        if !request.host_authored {
            if request.disposition == QueueDisposition::Parallel {
                self.parallel_active.lock().await.insert(
                    (
                        request.scope.clone(),
                        request.thread_id.clone(),
                        request.request_id.clone(),
                    ),
                    cancellation.clone(),
                );
            } else if let Some(reserved) = reservation.as_ref() {
                let mut active = self.active.lock().await;
                if active
                    .get(&key)
                    .is_none_or(|current| current.generation != reserved.generation)
                {
                    drop(active);
                    return self
                        .cancel_generation(&request, Some(reserved.generation))
                        .await;
                }
                let mut state = reserved.clone();
                state.request_id = request.request_id.clone();
                state.cancel = cancellation.clone();
                active.insert(key.clone(), state.clone());
                active_generation = Some(state.generation);
                *state_out = Some(state);
            } else {
                let mut active = self.active.lock().await;
                if let Some(lane) = queue_lane(request.disposition)
                    && let Some(current) = active.get(&key)
                {
                    current.queue.push(lane, request.clone()).await;
                    return Err(TurnRunnerError::Queued);
                }
                let queue = Arc::new(RunQueue::new());
                let state = ActiveTurn {
                    request_id: request.request_id.clone(),
                    cancel: cancellation.clone(),
                    generation: self
                        .next_generation
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                    queue: queue.clone(),
                };
                let previous = active.insert(key.clone(), state.clone());
                active_generation = Some(state.generation);
                *state_out = Some(state);
                primary_queue = Some(queue);
                if let Some(previous) = previous {
                    if request.disposition == QueueDisposition::Interrupt {
                        previous.queue.clear().await;
                    }
                    previous.cancel.cancel();
                }
            }
        }
        let run_queue = if request.disposition == QueueDisposition::Parallel {
            self.parallel_queue_for((
                request.scope.clone(),
                request.thread_id.clone(),
                request.request_id.clone(),
            ))
            .await
        } else if let Some(queue) = primary_queue {
            queue
        } else {
            Arc::new(RunQueue::new())
        };
        let lock = {
            let mut locks = self.locks.lock().await;
            locks.retain(|_, lock| lock.strong_count() > 0);
            if let Some(lock) = locks.get(&key).and_then(Weak::upgrade) {
                lock
            } else {
                let lock = Arc::new(Mutex::new(()));
                locks.insert(key.clone(), Arc::downgrade(&lock));
                lock
            }
        };
        let _session_guard = if request.disposition == QueueDisposition::Parallel {
            None
        } else {
            match self
                .before_deadline(deadline.at, &cancellation, lock.lock_owned())
                .await
            {
                Ok(guard) => Some(guard),
                Err(PreRunStop::Cancelled) => {
                    return self.cancel_generation(&request, active_generation).await;
                }
                Err(PreRunStop::Deadline) => {
                    return self
                        .fail_generation(
                            &request,
                            "turn wall-clock deadline".into(),
                            active_generation,
                        )
                        .await;
                }
            }
        };
        let checkout = if request.disposition == QueueDisposition::Parallel {
            self.before_deadline(
                deadline.at,
                &cancellation,
                self.host.checkout_fork(&request),
            )
            .await
        } else {
            self.before_deadline(deadline.at, &cancellation, self.host.checkout(&request))
                .await
        };
        let session = match checkout {
            Ok(Ok(session)) => session,
            Ok(Err(error)) => {
                return self
                    .fail_generation(&request, error, active_generation)
                    .await;
            }
            Err(PreRunStop::Cancelled) => {
                return self.cancel_generation(&request, active_generation).await;
            }
            Err(PreRunStop::Deadline) => {
                return self
                    .fail_generation(
                        &request,
                        "turn wall-clock deadline".into(),
                        active_generation,
                    )
                    .await;
            }
        };
        match self
            .before_deadline(
                deadline.at,
                &cancellation,
                self.host.validate_fingerprint(&request, &session),
            )
            .await
        {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                self.host.checkin(&request).await;
                return self
                    .fail_generation(&request, error, active_generation)
                    .await;
            }
            Err(PreRunStop::Cancelled) => {
                self.host.checkin(&request).await;
                return self.cancel_generation(&request, active_generation).await;
            }
            Err(PreRunStop::Deadline) => {
                self.host.checkin(&request).await;
                return self
                    .fail_generation(
                        &request,
                        "turn wall-clock deadline".into(),
                        active_generation,
                    )
                    .await;
            }
        }
        let origin = self.host.run_origin(&request);

        let result = self
            .run_guarded(
                &request,
                &session,
                &origin,
                cancellation.clone(),
                Arc::clone(&run_queue),
                deadline,
            )
            .await;
        self.host.checkin(&request).await;
        drop(_session_guard);

        match result {
            Ok(result) => {
                let reply_id =
                    deterministic_reply_id(&request.scope, &request.thread_id, &request.request_id);
                if let Err(error) = self.replies.persist(&request, &reply_id, &result).await {
                    return self
                        .fail_generation(&request, error, active_generation)
                        .await;
                }
                let suggestions = self.summarizer.suggestions(&request, &result).await;
                self.events.suggestions(&request, suggestions).await;
                if !request.host_authored && request.disposition != QueueDisposition::Parallel {
                    let (collected, next) = match active_generation {
                        Some(generation) => {
                            self.finalize_generation(
                                &request.scope,
                                &request.thread_id,
                                generation,
                                &request.payload,
                            )
                            .await
                        }
                        None => (Vec::new(), None),
                    };
                    *next_out = next.clone();
                    *state_out = next.map(|(_, state)| state);
                    self.events.collected(&request, collected).await;
                }
                self.events
                    .terminal(
                        &request,
                        TerminalEvent::Completed {
                            request_id: request.request_id.clone(),
                            reply_id,
                        },
                    )
                    .await;
                Ok(result)
            }
            Err(_error) if cancellation.is_cancelled() => {
                self.cancel_generation(&request, active_generation).await
            }
            Err(error) => {
                self.fail_generation(&request, error, active_generation)
                    .await
            }
        }
    }

    async fn parallel_queue_for(&self, key: ParallelQueueKey) -> Arc<RunQueue<TurnRequest<P>>> {
        let mut queues = self.parallel_queues.lock().await;
        Arc::clone(
            queues
                .entry(key)
                .or_insert_with(|| Arc::new(RunQueue::new())),
        )
    }

    async fn before_deadline<T>(
        &self,
        deadline: Option<Instant>,
        cancellation: &crate::CancellationToken,
        future: impl Future<Output = T>,
    ) -> Result<T, PreRunStop> {
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => Err(PreRunStop::Cancelled),
            _ = async {
                if let Some(deadline) = deadline {
                    tokio::time::sleep_until(deadline).await;
                } else {
                    std::future::pending::<()>().await;
                }
            }, if deadline.is_some() => Err(PreRunStop::Deadline),
            result = future => Ok(result),
        }
    }

    async fn retire_generation(
        &self,
        request: &TurnRequest<P>,
        generation: Option<u64>,
    ) -> Vec<TurnRequest<P>> {
        let Some(generation) = generation else {
            return Vec::new();
        };
        let key = (request.scope.clone(), request.thread_id.clone());
        let mut active = self.active.lock().await;
        let Some(turn) = active
            .get(&key)
            .filter(|turn| turn.generation == generation)
            .cloned()
        else {
            return Vec::new();
        };
        active.remove(&key);
        let mut queued = Vec::new();
        for lane in [QueueLane::Steer, QueueLane::Followup, QueueLane::Collect] {
            queued.extend(turn.queue.take(lane, QueueMode::All).await);
        }
        queued
    }

    async fn cancel_generation(
        &self,
        request: &TurnRequest<P>,
        generation: Option<u64>,
    ) -> Result<TurnResult, TurnRunnerError> {
        let queued = self.retire_generation(request, generation).await;
        if request.disposition == QueueDisposition::Parallel {
            self.clear_parallel_tracking(request).await;
        }
        for queued_request in queued {
            self.events
                .terminal(
                    &queued_request,
                    TerminalEvent::Cancelled {
                        request_id: queued_request.request_id.clone(),
                    },
                )
                .await;
        }
        self.events
            .terminal(
                request,
                TerminalEvent::Cancelled {
                    request_id: request.request_id.clone(),
                },
            )
            .await;
        Err(TurnRunnerError::Cancelled)
    }

    async fn fail_generation<T>(
        &self,
        request: &TurnRequest<P>,
        error: String,
        generation: Option<u64>,
    ) -> Result<T, TurnRunnerError> {
        let (class, message) = self.failures.classify(&error);
        let queued = self.retire_generation(request, generation).await;
        if request.disposition == QueueDisposition::Parallel {
            self.clear_parallel_tracking(request).await;
        }
        for queued_request in queued {
            self.events
                .terminal(
                    &queued_request,
                    TerminalEvent::Failed {
                        request_id: queued_request.request_id.clone(),
                        message: format!("queued turn was not run: {message}"),
                        class: class.clone(),
                        error: error.clone(),
                    },
                )
                .await;
        }
        self.events
            .terminal(
                request,
                TerminalEvent::Failed {
                    request_id: request.request_id.clone(),
                    message: message.clone(),
                    class,
                    error,
                },
            )
            .await;
        Err(TurnRunnerError::Failed(message))
    }

    async fn finalize_generation(
        &self,
        scope: &ScopeKey,
        thread_id: &str,
        owner_generation: u64,
        running: &P,
    ) -> (Vec<TurnRequest<P>>, Option<(TurnRequest<P>, ActiveTurn<P>)>) {
        let key = (scope.clone(), thread_id.to_owned());
        let mut active = self.active.lock().await;
        if active
            .get(&key)
            .is_none_or(|turn| turn.generation != owner_generation)
        {
            return (Vec::new(), None);
        }
        let Some(current) = active.get(&key).cloned() else {
            return (Vec::new(), None);
        };
        let collected = current.queue.take(QueueLane::Collect, QueueMode::All).await;
        let next = current
            .queue
            .take(QueueLane::Steer, QueueMode::OneAtATime)
            .await
            .into_iter()
            .next();
        let next = if next.is_none() {
            current
                .queue
                .take(QueueLane::Followup, QueueMode::OneAtATime)
                .await
                .into_iter()
                .next()
        } else {
            next
        };
        if let Some(mut request) = next {
            // A steer the harness recovered was rebuilt without its scope and
            // payload metadata; it belongs to this thread and inherits what the
            // finished turn knew.
            request.scope = scope.clone();
            request.thread_id = thread_id.to_owned();
            request.payload.adopt(running);
            let reserved = ActiveTurn {
                request_id: request.request_id.clone(),
                cancel: crate::CancellationToken::new(),
                generation: self
                    .next_generation
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                queue: current.queue,
            };
            active.insert(key, reserved.clone());
            (collected, Some((request, reserved)))
        } else {
            active.remove(&key);
            (collected, None)
        }
    }

    /// Take queued work from one lane. `OneAtATime` is useful for steering;
    /// `All` preserves FIFO while applying every accepted item at a boundary.
    pub async fn take_queued(
        &self,
        scope: &ScopeKey,
        thread_id: &str,
        lane: QueueLane,
        mode: QueueMode,
    ) -> Vec<TurnRequest<P>> {
        let key = (scope.clone(), thread_id.to_owned());
        let active = self.active.lock().await;
        match active.get(&key) {
            Some(turn) => turn.queue.take(lane, mode).await,
            None => Vec::new(),
        }
    }

    pub async fn queue_status(&self, scope: &ScopeKey, thread_id: &str) -> QueueStatus {
        let key = (scope.clone(), thread_id.to_owned());
        let active = self.active.lock().await;
        match active.get(&key) {
            Some(turn) => turn.queue.status().await,
            None => QueueStatus {
                steers: 0,
                followups: 0,
                collects: 0,
                total: 0,
            },
        }
    }

    pub async fn clear_queue(&self, scope: &ScopeKey, thread_id: &str) -> usize {
        let key = (scope.clone(), thread_id.to_owned());
        let active = self.active.lock().await;
        match active.get(&key) {
            Some(turn) => turn.queue.clear().await,
            None => 0,
        }
    }

    async fn clear_generation_queue(
        &self,
        scope: &ScopeKey,
        thread_id: &str,
        generation: u64,
    ) -> usize {
        let key = (scope.clone(), thread_id.to_owned());
        let active = self.active.lock().await;
        match active
            .get(&key)
            .filter(|turn| turn.generation == generation)
        {
            Some(turn) => turn.queue.clear().await,
            None => 0,
        }
    }

    pub async fn remove_queued(
        &self,
        scope: &ScopeKey,
        thread_id: &str,
        request_id: &str,
    ) -> usize {
        let key = (scope.clone(), thread_id.to_owned());
        let active = self.active.lock().await;
        match active.get(&key) {
            Some(turn) => {
                turn.queue
                    .remove_where(|turn| turn.request_id == request_id)
                    .await
            }
            None => 0,
        }
    }

    async fn run_guarded(
        &self,
        request: &TurnRequest<P>,
        session: &TurnSession,
        origin: &serde_json::Value,
        cancellation: crate::CancellationToken,
        queue: Arc<RunQueue<TurnRequest<P>>>,
        deadline: TurnDeadline,
    ) -> Result<TurnResult, String> {
        let (progress_tx, mut progress_rx) = mpsc::channel(32);
        let task_cancellation = crate::CancellationToken::new();
        let run = self.host.run(
            request,
            session,
            origin,
            task_cancellation.clone(),
            queue,
            progress_tx,
        );
        let idle = request.idle_timeout.unwrap_or(self.config.idle_timeout);
        let mut last_progress = deadline.started;
        let mut work_in_flight = 0usize;
        let mut progress_open = true;
        tokio::pin!(run);
        loop {
            if deadline
                .at
                .is_some_and(|deadline| Instant::now() >= deadline)
            {
                task_cancellation.cancel();
                self.host.cancel_children(request).await;
                return self
                    .close_or_timeout(request, "wall-clock deadline", &cancellation)
                    .await;
            }
            let idle_left = idle.saturating_sub(last_progress.elapsed());
            tokio::select! {
                biased;
                _ = cancellation.cancelled() => {
                    task_cancellation.cancel();
                    self.host.cancel_children(request).await;
                    if !self.config.cancel_grace.is_zero() {
                        // Let the host unwind with its token fired; its result
                        // is discarded either way.
                        let unwound = tokio::time::timeout(self.config.cancel_grace, &mut run).await;
                        tracing::debug!(
                            thread_id = %request.thread_id,
                            request_id = %request.request_id,
                            unwound = unwound.is_ok(),
                            "turn runner: cancelled run left its grace window"
                        );
                    }
                    return Err("turn cancelled".into());
                }
                _ = async { if let Some(deadline) = deadline.at { tokio::time::sleep_until(deadline).await } else { std::future::pending().await } }, if deadline.at.is_some() => {
                    task_cancellation.cancel();
                    self.host.cancel_children(request).await;
                    return self.close_or_timeout(request, "wall-clock deadline", &cancellation).await;
                }
                result = &mut run => {
                    // A host can emit progress and finish in the same poll. Close
                    // the channel before draining so retained child senders
                    // cannot extend completion indefinitely, then deliver every
                    // event already accepted by the bounded queue.
                    progress_rx.close();
                    while let Ok(signal) = progress_rx.try_recv() {
                        if let TurnSignal::Progress(event) = signal {
                            tokio::select! {
                                biased;
                                _ = cancellation.cancelled() => {
                                    task_cancellation.cancel();
                                    self.host.cancel_children(request).await;
                                    return Err("turn cancelled".into());
                                }
                                _ = async { if let Some(deadline) = deadline.at { tokio::time::sleep_until(deadline).await } else { std::future::pending().await } }, if deadline.at.is_some() => {
                                    task_cancellation.cancel();
                                    self.host.cancel_children(request).await;
                                    return self.close_or_timeout(request, "wall-clock deadline", &cancellation).await;
                                }
                                _ = self.events.progress(request, *event) => {}
                            }
                        }
                    }
                    return result;
                }
                signal = progress_rx.recv(), if progress_open => match signal {
                    Some(signal) => match signal {
                        TurnSignal::Progress(event) => {
                            last_progress = Instant::now();
                            tokio::select! {
                                biased;
                                _ = cancellation.cancelled() => {
                                    task_cancellation.cancel();
                                    self.host.cancel_children(request).await;
                                    return Err("turn cancelled".into());
                                }
                                _ = async { if let Some(deadline) = deadline.at { tokio::time::sleep_until(deadline).await } else { std::future::pending().await } }, if deadline.at.is_some() => {
                                    task_cancellation.cancel();
                                    self.host.cancel_children(request).await;
                                    return self.close_or_timeout(request, "wall-clock deadline", &cancellation).await;
                                }
                                _ = self.events.progress(request, *event) => {}
                            }
                        }
                        TurnSignal::WorkStarted => work_in_flight = work_in_flight.saturating_add(1),
                        TurnSignal::WorkFinished => {
                            work_in_flight = work_in_flight.saturating_sub(1);
                            if work_in_flight == 0 {
                                last_progress = Instant::now();
                            }
                        }
                    },
                    None => progress_open = false,
                },
                _ = tokio::time::sleep(idle_left), if work_in_flight == 0 => {
                    task_cancellation.cancel();
                    self.host.cancel_children(request).await;
                    return self.close_or_timeout(request, "idle timeout", &cancellation).await;
                }
            }
        }
    }

    async fn close_or_timeout(
        &self,
        request: &TurnRequest<P>,
        reason: &str,
        cancellation: &crate::CancellationToken,
    ) -> Result<TurnResult, String> {
        tokio::select! {
            biased;
            _ = cancellation.cancelled() => Err("turn cancelled".into()),
            result = tokio::time::timeout(
                self.config.close_timeout,
                self.summarizer.grounded_close(request, reason),
            ) => match result {
                Ok(Some(result)) if !cancellation.is_cancelled() => Ok(result),
                _ if cancellation.is_cancelled() => Err("turn cancelled".into()),
                _ => Err(format!("turn {reason}")),
            }
        }
    }

    async fn clear_tracking(&self, request: &TurnRequest<P>, generation: u64) {
        let key = (request.scope.clone(), request.thread_id.clone());
        let mut active = self.active.lock().await;
        if active
            .get(&key)
            .is_some_and(|turn| turn.generation == generation)
        {
            active.remove(&key);
        }
    }

    async fn clear_parallel_tracking(&self, request: &TurnRequest<P>) {
        let key = (
            request.scope.clone(),
            request.thread_id.clone(),
            request.request_id.clone(),
        );
        self.parallel_active.lock().await.remove(&key);
        self.parallel_queues.lock().await.remove(&key);
    }
}

#[derive(Debug, Clone, Copy)]
enum PreRunStop {
    Cancelled,
    Deadline,
}

#[derive(Debug, Clone, Copy)]
struct TurnDeadline {
    started: Instant,
    at: Option<Instant>,
}

impl TurnDeadline {
    fn new(duration: Option<Duration>) -> Self {
        let started = Instant::now();
        Self {
            started,
            at: duration.map(|duration| started + duration),
        }
    }
}

/// Stable reply identity for idempotent host persistence.
pub fn deterministic_reply_id(scope: &ScopeKey, thread_id: &str, request_id: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    hash.update(
        serde_json::to_vec(&(&scope.agent_id, &scope.profile_id, thread_id, request_id))
            .expect("serializing a tuple of strings cannot fail"),
    );
    let encoded = hash
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("reply-{encoded}")
}
