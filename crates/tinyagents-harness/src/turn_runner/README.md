# Turn runner

`TurnRunner` contains the reusable lifecycle around one channel-agnostic
conversation turn. The host implements session checkout and fingerprint
validation, supplies origin metadata, executes the turn, and checks the
session back in. Separate ports own ingress policy, reply persistence,
progress and terminal delivery, failure classification, and grounded closing
or suggestions.

Ingress policies run in a fixed order: stage attachments, apply the guardrail,
intercept approval replies, then run before-submit hooks. A consumed approval
reply or a policy rejection never starts a turn. The runner owns TinyAgents'
`RunQueue` and its Steer, Followup, and Collect lanes. Accepted work retains
its generic, typed host payload in that queue; Followup requests dispatch as
fresh turns after the current turn completes. Steer requests still pending at
that boundary also dispatch as fresh turns, so an accepted steer cannot be
lost after host execution has stopped. `Interrupt` cancels the prior
primary generation; `Parallel` uses a host-created fork session and a separate
request-scoped cancellation slot, without serializing against the primary.

Primary sessions serialize by `(agent_id, profile_id, thread_id)`. The active
slot is installed before waiting for that lock, so an interrupt can stop the
current owner and stale cancellation ids cannot stop its replacement.
Host-authored turns use the same session checkout and reply store but never
occupy the user cancellation slot. Hosts own session identity and checkout,
including fork isolation for parallel runs, and must check a session back in
after every successful checkout. The runner persists a deterministic reply
id before emitting exactly one terminal event.

The idle guard defaults to 900 seconds and resets on progress. Hosts can use
`WorkTracker` to fold `AgentProgress` into `WorkStarted` and `WorkFinished`
signals around model or tool work, so a quiet but healthy request is not
treated as idle. The optional wall deadline remains a separate total ceiling.
Either guard cancels child work and asks the `Summarizer` for a bounded
grounded close before classifying the turn as failed.

`AgentProgress` is the canonical channel-independent progress vocabulary.
Hosts translate it to their own event wire format; this module does not define
channel event names or transport payloads.

`TurnRequest<P>` is itself a [`QueuedMessage`](crate::run_queue::QueuedMessage)
for every payload that implements `RequeuePayload`, so a turn's run queue can be
handed straight to the steering forwarder. A steer the forwarder recovers and
pushes back is rebuilt from a bare id, text and thread label; the runner
restores the scope and thread when it dispatches that item as a fresh turn and
lets the payload adopt the metadata of the turn that was running.
`TurnRunnerConfig::cancel_grace` keeps a cancelled host run polled, with its own
cancellation token already fired, for a bounded time so the host can hand back
and persist the rounds it completed. It defaults to zero (drop at once).
