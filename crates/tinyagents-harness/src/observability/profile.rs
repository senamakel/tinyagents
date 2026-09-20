//! Lightweight process profiling for repeatable harness load tests.
//!
//! This module intentionally avoids a metrics runtime or global recorder. A
//! caller starts a [`ProcessProfiler`] around the workload it wants to measure
//! and receives one serializable [`ProcessProfile`] when the workload ends.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// A point-in-time view of process resource consumption.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ProcessSnapshot {
    /// Resident memory in bytes, when the platform exposes it.
    pub rss_bytes: Option<u64>,
    /// User-mode CPU consumed by this process in microseconds.
    pub user_cpu_us: Option<u64>,
    /// Kernel-mode CPU consumed by this process in microseconds.
    pub system_cpu_us: Option<u64>,
}

impl ProcessSnapshot {
    /// Reads the current process counters without starting a sampler thread.
    pub fn capture() -> Self {
        let cpu = cpu_time_us();
        Self {
            rss_bytes: current_rss_bytes(),
            user_cpu_us: cpu.map(|(user, _)| user),
            system_cpu_us: cpu.map(|(_, system)| system),
        }
    }
}

/// Process-level measurements collected around one workload.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ProcessProfile {
    /// Wall-clock duration of the measured region.
    pub wall_time_ms: f64,
    /// Resident memory at profiler start.
    pub baseline_rss_bytes: Option<u64>,
    /// Highest sampled resident memory during the measured region.
    pub peak_rss_bytes: Option<u64>,
    /// Signed peak increase over the baseline.
    pub peak_rss_delta_bytes: Option<i64>,
    /// User-mode CPU consumed during the measured region.
    pub user_cpu_ms: Option<f64>,
    /// Kernel-mode CPU consumed during the measured region.
    pub system_cpu_ms: Option<f64>,
    /// Total CPU time divided by wall time. One saturated core is 100%; a
    /// multi-threaded workload may exceed 100%.
    pub cpu_utilization_percent: Option<f64>,
    /// Number of RSS samples taken, including the initial sample.
    pub sample_count: u64,
}

struct SharedSamples {
    running: AtomicBool,
    peak_rss_bytes: AtomicU64,
    sample_count: AtomicU64,
}

/// A scoped, low-overhead process CPU/RSS sampler.
///
/// RSS is sampled on a dedicated thread because a start/end pair misses short
/// allocation peaks. CPU is read only at the boundaries. Dropping the profiler
/// stops and joins the sampler; call [`Self::finish`] to retain the report.
pub struct ProcessProfiler {
    started: Instant,
    initial: ProcessSnapshot,
    shared: Arc<SharedSamples>,
    sampler: Option<std::thread::JoinHandle<()>>,
}

impl ProcessProfiler {
    /// Starts profiling with `sample_interval` clamped to at least 1 ms.
    pub fn start(sample_interval: Duration) -> Self {
        let initial = ProcessSnapshot::capture();
        let shared = Arc::new(SharedSamples {
            running: AtomicBool::new(true),
            peak_rss_bytes: AtomicU64::new(initial.rss_bytes.unwrap_or(0)),
            sample_count: AtomicU64::new(1),
        });
        let worker = Arc::clone(&shared);
        let interval = sample_interval.max(Duration::from_millis(1));
        let sampler = std::thread::Builder::new()
            .name("tinyagents-process-profiler".to_string())
            .spawn(move || {
                while worker.running.load(Ordering::Relaxed) {
                    std::thread::park_timeout(interval);
                    sample_rss(&worker);
                }
                sample_rss(&worker);
            })
            .ok();
        Self {
            started: Instant::now(),
            initial,
            shared,
            sampler,
        }
    }

    /// Stops sampling and returns the completed profile.
    pub fn finish(mut self) -> ProcessProfile {
        self.stop_sampler();
        let wall_time = self.started.elapsed();
        let final_snapshot = ProcessSnapshot::capture();
        let peak = self.shared.peak_rss_bytes.load(Ordering::Relaxed);
        let peak = (peak > 0).then_some(peak);
        let user_cpu_ms = elapsed_us(self.initial.user_cpu_us, final_snapshot.user_cpu_us)
            .map(|value| value as f64 / 1_000.0);
        let system_cpu_ms = elapsed_us(self.initial.system_cpu_us, final_snapshot.system_cpu_us)
            .map(|value| value as f64 / 1_000.0);
        let wall_time_ms = wall_time.as_secs_f64() * 1_000.0;
        let cpu_utilization_percent = match (user_cpu_ms, system_cpu_ms) {
            (Some(user), Some(system)) if wall_time_ms > 0.0 => {
                Some((user + system) / wall_time_ms * 100.0)
            }
            _ => None,
        };
        ProcessProfile {
            wall_time_ms,
            baseline_rss_bytes: self.initial.rss_bytes,
            peak_rss_bytes: peak,
            peak_rss_delta_bytes: self
                .initial
                .rss_bytes
                .zip(peak)
                .map(|(start, peak)| i64::try_from(peak.saturating_sub(start)).unwrap_or(i64::MAX)),
            user_cpu_ms,
            system_cpu_ms,
            cpu_utilization_percent,
            sample_count: self.shared.sample_count.load(Ordering::Relaxed),
        }
    }

    fn stop_sampler(&mut self) {
        self.shared.running.store(false, Ordering::Relaxed);
        if let Some(sampler) = self.sampler.take() {
            sampler.thread().unpark();
            let _ = sampler.join();
        }
    }
}

impl Drop for ProcessProfiler {
    fn drop(&mut self) {
        self.stop_sampler();
    }
}

fn elapsed_us(start: Option<u64>, end: Option<u64>) -> Option<u64> {
    Some(end?.saturating_sub(start?))
}

fn sample_rss(samples: &SharedSamples) {
    if let Some(rss) = current_rss_bytes() {
        samples.peak_rss_bytes.fetch_max(rss, Ordering::Relaxed);
    }
    samples.sample_count.fetch_add(1, Ordering::Relaxed);
}

#[cfg(target_os = "linux")]
fn current_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmRSS:"))?;
    let kib = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    kib.checked_mul(1024)
}

#[cfg(not(target_os = "linux"))]
fn current_rss_bytes() -> Option<u64> {
    None
}

#[cfg(unix)]
fn cpu_time_us() -> Option<(u64, u64)> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `getrusage` initializes the provided `rusage` on success and we
    // only call `assume_init` after checking its zero return code.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: established by the successful `getrusage` call above.
    let usage = unsafe { usage.assume_init() };
    Some((timeval_us(usage.ru_utime), timeval_us(usage.ru_stime)))
}

#[cfg(unix)]
fn timeval_us(value: libc::timeval) -> u64 {
    let seconds = u64::try_from(value.tv_sec).unwrap_or(0);
    let micros = u64::try_from(value.tv_usec).unwrap_or(0);
    seconds.saturating_mul(1_000_000).saturating_add(micros)
}

#[cfg(not(unix))]
fn cpu_time_us() -> Option<(u64, u64)> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiler_reports_a_bounded_region() {
        let profiler = ProcessProfiler::start(Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(3));
        let profile = profiler.finish();

        assert!(profile.wall_time_ms >= 2.0);
        assert!(profile.sample_count >= 2);
        if let (Some(baseline), Some(peak)) = (profile.baseline_rss_bytes, profile.peak_rss_bytes) {
            assert!(peak >= baseline);
        }
    }
}
