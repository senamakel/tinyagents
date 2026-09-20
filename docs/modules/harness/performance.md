# Harness performance and capacity testing

The harness ships an offline stress driver and process profiler for answering
three separate questions:

1. How does scheduler/runtime overhead scale from 1 to 100 concurrent agents?
2. What CPU, resident-memory, throughput, and tail-latency cost does
   observability add?
3. Which code paths consume CPU when the provider and network are removed from
   the measurement?

The deterministic mock provider completes locally. These measurements are
therefore a harness overhead test, not a model-provider capacity test. Run a
second workload against the production provider to size connection pools,
rate limits, and network capacity.

## Stress matrix

Build and run the default 1, 10, 25, 50, and 100 agent matrix in release mode:

```text
cargo run -p tinyagents-integration-tests --release \
  --example harness_stress -- --runs-per-agent 100
```

The report includes successful and failed runs, runs/second, p50/p95/p99/max
run latency, process CPU utilization, baseline/peak RSS, and sample count. One
fully occupied CPU core is 100%; a multi-threaded workload can exceed 100%.

Use `--json` to persist comparable results and select event overhead with
`--observe`:

```text
# Harness execution without listeners.
cargo run -p tinyagents-integration-tests --release \
  --example harness_stress -- --runs-per-agent 100 --observe off --json

# Event delivery to a non-retaining atomic counter.
cargo run -p tinyagents-integration-tests --release \
  --example harness_stress -- --runs-per-agent 100 --observe count --json

# Event delivery plus full in-memory retention.
cargo run -p tinyagents-integration-tests --release \
  --example harness_stress -- --runs-per-agent 100 --observe record --json
```

`--concurrency 100` runs only the maximum case. Concurrency values above 100
are rejected deliberately so the checked-in workload matches the supported
capacity target. `--sample-ms` controls RSS sampling frequency (5 ms by
default); very small intervals increase profiler overhead.

## Correctness stress test

The ordinary suite runs a short 16-agent shared-harness test. The 100-agent
capacity test is ignored so it does not become a timing-sensitive CI gate:

```text
cargo test -p tinyagents-integration-tests --test harness_stress \
  --release -- --ignored --nocapture
```

This test requires every invocation to finish successfully; the stress driver
is the richer measurement surface.

## CPU profiling

Build once, then collect Linux `perf` data under `target/` so generated output
stays inside the checkout and can be cleaned with the normal target cleanup:

```text
cargo build -p tinyagents-integration-tests --release --example harness_stress
mkdir -p target/profiles
perf stat -d target/release/examples/harness_stress \
  --concurrency 100 --runs-per-agent 1000 --observe off
perf record -g --call-graph dwarf -o target/profiles/harness-stress.data -- \
  target/release/examples/harness_stress \
  --concurrency 100 --runs-per-agent 1000 --observe count
perf report -i target/profiles/harness-stress.data
```

Use `--observe off`, `count`, and `record` as separate profiles. A combined
profile obscures whether time belongs to core execution, event dispatch, or
retention. Provider-backed profiling should also be separate because waiting
on network I/O dominates wall time and hides local CPU hot spots.

## ProcessProfiler API

`tinyagents_harness::observability::ProcessProfiler` is independent of the
stress binary and can wrap application-specific workloads. It samples current
RSS on a low-frequency helper thread and reads user/system process CPU at the
measurement boundaries. `finish()` returns a serializable `ProcessProfile`.
RSS is currently available on Linux and CPU time on Unix; unsupported counters
are `None`, not zero.

The sampler reports process-wide consumption. Do not run unrelated workloads
in the same process while interpreting a profile as belonging to one agent
batch.

## Initial bottleneck and fix

The first 100-agent release run exposed an event-bus allocation/backlog issue:
events were cloned and inserted into the ordered dispatch queue even when the
sink had no listeners. Fast agents could produce records faster than the one
drainer removed them, so an observability-disabled run still accumulated a
large transient queue. Listener vectors were also cloned for every event.

The event sink now returns directly after assigning an id/offset when there are
no listeners, and listener snapshots use `Arc<Vec<_>>`, making the per-event
snapshot an atomic reference-count increment. On the development host, the
same 100-agent/100-runs-each workload changed as follows (single samples; use
the JSON matrix for rigorous repeated comparisons):

| Mode | Throughput before | Throughput after | Peak RSS before | Peak RSS after |
| --- | ---: | ---: | ---: | ---: |
| off | 172k runs/s | 439k runs/s | 38.0 MiB | 9.0 MiB |
| count | 156k runs/s | 190k runs/s | 40.5 MiB | 37.1 MiB |
| record | 130k runs/s | 191k runs/s | 60.4 MiB | 57.0 MiB |

Full recording intentionally grows with retained event count. For long-lived
processes, use the bounded durable journal controls and export/drain events
instead of retaining an unbounded `RecordingListener`.

## Reading results

- Compare p95/p99 before averages; lock contention and scheduler stalls appear
  first in the tail.
- Compare peak RSS deltas, not only absolute RSS, when the process has already
  initialized large provider clients or caches.
- Treat zero failures as a requirement. Throughput from a run that shed work is
  not a valid improvement.
- Repeat release runs on an otherwise idle machine. CPU frequency scaling and
  unrelated processes make single sub-millisecond measurements noisy.
- Use realistic message/tool payloads in a second benchmark. This driver
  isolates orchestration overhead and does not predict context-sized cloning or
  provider serialization costs.
