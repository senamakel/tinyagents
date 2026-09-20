//! Offline stress/profiling driver for `AgentHarness`.
//!
//! The workload uses the deterministic in-process mock model, so results expose
//! harness/runtime overhead rather than network latency. Run the default matrix:
//!
//! ```text
//! cargo run -p tinyagents-integration-tests --release --example harness_stress
//! ```
//!
//! Use `--json` for machine-readable output and `--observe count|record` to
//! measure event-pipeline overhead. Concurrency is intentionally capped at 100.

use std::env;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde::Serialize;
use tinyagents_harness::context::{RunConfig, RunContext};
use tinyagents_harness::events::{EventListener, EventRecord, EventSink, RecordingListener};
use tinyagents_harness::observability::{ProcessProfile, ProcessProfiler};
use tinyagents_harness::runtime::AgentHarness;
use tinyinference_llm::message::Message;
use tinyinference_llm::providers::MockModel;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ObserveMode {
    Off,
    Count,
    Record,
}

impl ObserveMode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "off" => Ok(Self::Off),
            "count" => Ok(Self::Count),
            "record" => Ok(Self::Record),
            _ => Err(format!(
                "invalid --observe value `{value}`; expected off, count, or record"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Count => "count",
            Self::Record => "record",
        }
    }
}

struct Config {
    concurrency: Vec<usize>,
    runs_per_agent: usize,
    sample_interval: Duration,
    observe: ObserveMode,
    json: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            concurrency: vec![1, 10, 25, 50, 100],
            runs_per_agent: 20,
            sample_interval: Duration::from_millis(5),
            observe: ObserveMode::Off,
            json: false,
        }
    }
}

impl Config {
    fn from_args() -> Result<Self, String> {
        let mut config = Self::default();
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--concurrency" => {
                    let raw = args.next().ok_or("--concurrency requires a value")?;
                    config.concurrency = raw
                        .split(',')
                        .map(|value| {
                            value
                                .parse::<usize>()
                                .map_err(|_| format!("invalid concurrency `{value}`"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                }
                "--runs-per-agent" => {
                    let raw = args.next().ok_or("--runs-per-agent requires a value")?;
                    config.runs_per_agent = raw
                        .parse()
                        .map_err(|_| format!("invalid runs-per-agent `{raw}`"))?;
                }
                "--sample-ms" => {
                    let raw = args.next().ok_or("--sample-ms requires a value")?;
                    let millis = raw
                        .parse::<u64>()
                        .map_err(|_| format!("invalid sample interval `{raw}`"))?;
                    config.sample_interval = Duration::from_millis(millis.max(1));
                }
                "--observe" => {
                    let raw = args.next().ok_or("--observe requires a value")?;
                    config.observe = ObserveMode::parse(&raw)?;
                }
                "--json" => config.json = true,
                "--help" | "-h" => {
                    println!(
                        "harness_stress [--concurrency 1,10,25,50,100] \
                         [--runs-per-agent 20] [--sample-ms 5] \
                         [--observe off|count|record] [--json]"
                    );
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument `{arg}`; use --help")),
            }
        }
        if config.concurrency.is_empty()
            || config
                .concurrency
                .iter()
                .any(|&value| value == 0 || value > 100)
        {
            return Err("concurrency values must be in 1..=100".to_string());
        }
        if config.runs_per_agent == 0 {
            return Err("--runs-per-agent must be greater than zero".to_string());
        }
        Ok(config)
    }
}

#[derive(Default)]
struct CountingListener {
    events: AtomicU64,
}

impl EventListener for CountingListener {
    fn on_event(&self, _record: &EventRecord) {
        self.events.fetch_add(1, Ordering::Relaxed);
    }
}

enum Observer {
    Off,
    Count(Arc<CountingListener>),
    Record(Arc<RecordingListener>),
}

impl Observer {
    fn new(mode: ObserveMode, sink: &EventSink) -> Self {
        match mode {
            ObserveMode::Off => Self::Off,
            ObserveMode::Count => {
                let listener = Arc::new(CountingListener::default());
                sink.subscribe(listener.clone());
                Self::Count(listener)
            }
            ObserveMode::Record => {
                let listener = Arc::new(RecordingListener::new());
                sink.subscribe(listener.clone());
                Self::Record(listener)
            }
        }
    }

    fn event_count(&self) -> u64 {
        match self {
            Self::Off => 0,
            Self::Count(listener) => listener.events.load(Ordering::Relaxed),
            Self::Record(listener) => listener.len() as u64,
        }
    }
}

#[derive(Debug, Serialize)]
struct StressReport {
    concurrency: usize,
    runs_per_agent: usize,
    total_runs: usize,
    succeeded: usize,
    failed: usize,
    observe: String,
    observed_events: u64,
    throughput_runs_per_second: f64,
    latency_ms_p50: f64,
    latency_ms_p95: f64,
    latency_ms_p99: f64,
    latency_ms_max: f64,
    process: ProcessProfile,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let config = Config::from_args().unwrap_or_else(|error| {
        eprintln!("error: {error}");
        std::process::exit(2);
    });
    let harness = Arc::new(build_harness());
    let mut reports = Vec::with_capacity(config.concurrency.len());
    for &concurrency in &config.concurrency {
        reports.push(run_scenario(&harness, &config, concurrency).await);
    }

    if config.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&reports).expect("serialize stress reports")
        );
    } else {
        print_table(&reports);
    }
}

fn build_harness() -> AgentHarness<()> {
    let mut harness = AgentHarness::new();
    harness.register_model("mock", Arc::new(MockModel::constant("done")));
    harness
}

async fn run_scenario(
    harness: &Arc<AgentHarness<()>>,
    config: &Config,
    concurrency: usize,
) -> StressReport {
    let event_sink = EventSink::with_stream_id(format!("stress-{concurrency}"));
    let observer = Observer::new(config.observe, &event_sink);
    let barrier = Arc::new(tokio::sync::Barrier::new(concurrency + 1));
    let profiler = ProcessProfiler::start(config.sample_interval);
    let mut tasks = tokio::task::JoinSet::new();

    for agent_index in 0..concurrency {
        let harness = Arc::clone(harness);
        let barrier = Arc::clone(&barrier);
        let event_sink = event_sink.clone();
        let runs_per_agent = config.runs_per_agent;
        tasks.spawn(async move {
            barrier.wait().await;
            let mut latencies = Vec::with_capacity(runs_per_agent);
            let mut failures = 0;
            for iteration in 0..runs_per_agent {
                let run_id = format!("stress-{concurrency}-{agent_index}-{iteration}");
                let context =
                    RunContext::new(RunConfig::new(run_id), ()).with_events(event_sink.clone());
                let started = Instant::now();
                if harness
                    .invoke_in_context(&(), context, vec![Message::user("benchmark")])
                    .await
                    .is_err()
                {
                    failures += 1;
                }
                latencies.push(started.elapsed().as_secs_f64() * 1_000.0);
            }
            (latencies, failures)
        });
    }

    barrier.wait().await;
    let mut latencies = Vec::with_capacity(concurrency * config.runs_per_agent);
    let mut failed = 0;
    while let Some(result) = tasks.join_next().await {
        let (mut task_latencies, task_failures) = result.expect("stress task panicked");
        latencies.append(&mut task_latencies);
        failed += task_failures;
    }
    let process = profiler.finish();
    latencies.sort_by(f64::total_cmp);
    let total_runs = latencies.len();
    StressReport {
        concurrency,
        runs_per_agent: config.runs_per_agent,
        total_runs,
        succeeded: total_runs - failed,
        failed,
        observe: config.observe.as_str().to_string(),
        observed_events: observer.event_count(),
        throughput_runs_per_second: total_runs as f64 / (process.wall_time_ms / 1_000.0),
        latency_ms_p50: percentile(&latencies, 0.50),
        latency_ms_p95: percentile(&latencies, 0.95),
        latency_ms_p99: percentile(&latencies, 0.99),
        latency_ms_max: latencies.last().copied().unwrap_or(0.0),
        process,
    }
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[index]
}

fn print_table(reports: &[StressReport]) {
    println!(
        " conc | runs | obs    | runs/s | p50 ms | p95 ms | p99 ms | CPU % | peak RSS | failures"
    );
    println!(
        "------+------|--------|--------|--------|--------|--------|-------|----------|---------"
    );
    for report in reports {
        let peak_rss = report
            .process
            .peak_rss_bytes
            .map(|bytes| format!("{:.1} MiB", bytes as f64 / 1_048_576.0))
            .unwrap_or_else(|| "n/a".to_string());
        let cpu = report
            .process
            .cpu_utilization_percent
            .map(|value| format!("{value:.1}"))
            .unwrap_or_else(|| "n/a".to_string());
        println!(
            "{:>5} | {:>4} | {:<6} | {:>6.0} | {:>6.3} | {:>6.3} | {:>6.3} | {:>5} | {:>8} | {:>8}",
            report.concurrency,
            report.total_runs,
            report.observe,
            report.throughput_runs_per_second,
            report.latency_ms_p50,
            report.latency_ms_p95,
            report.latency_ms_p99,
            cpu,
            peak_rss,
            report.failed,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_selects_distribution_boundaries() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&values, 0.0), 1.0);
        assert_eq!(percentile(&values, 0.5), 3.0);
        assert_eq!(percentile(&values, 1.0), 5.0);
    }
}
