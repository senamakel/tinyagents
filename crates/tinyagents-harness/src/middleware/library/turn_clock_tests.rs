use std::time::Duration;

use super::*;
use crate::context::{RunConfig, RunContext};
use crate::middleware::{Middleware, ToolInvocationIdentity};
use tinytools::{ToolContent, ToolResult};

const MINUTE: Duration = Duration::from_secs(60);

#[test]
fn clock_durations_render_compactly() {
    assert_eq!(format_clock(Duration::from_secs(45)), "45s");
    assert_eq!(format_clock(Duration::from_secs(270)), "4m30s");
    assert_eq!(format_clock(Duration::from_secs(600)), "10m");
    assert_eq!(format_clock(Duration::from_secs(32 * 60 + 59)), "32m");
    assert_eq!(format_clock(Duration::from_secs(65 * 60)), "1h05m");
}

#[test]
fn the_note_names_elapsed_and_remaining_time() {
    let clock = TurnClock {
        elapsed: 32 * MINUTE,
        budget: 60 * MINUTE,
    };
    assert_eq!(clock.remaining(), 28 * MINUTE);
    assert_eq!(clock.note(), "[turn budget: 32m elapsed / 28m remaining]");
}

#[test]
fn remaining_saturates_once_the_budget_is_spent() {
    let clock = TurnClock {
        elapsed: 61 * MINUTE,
        budget: 60 * MINUTE,
    };
    assert_eq!(clock.remaining(), Duration::ZERO);
}

#[test]
fn no_band_before_half_the_budget_is_used() {
    let budget = 60 * MINUTE;
    for elapsed in [
        Duration::ZERO,
        10 * MINUTE,
        29 * MINUTE + Duration::from_secs(59),
    ] {
        let clock = TurnClock { elapsed, budget };
        assert_eq!(clock.band(), None, "elapsed {elapsed:?}");
    }
}

#[test]
fn bands_advance_every_tenth_of_the_budget_past_half() {
    let budget = 60 * MINUTE;
    let band = |minutes: u32| {
        TurnClock {
            elapsed: minutes * MINUTE,
            budget,
        }
        .band()
    };
    assert_eq!(band(30), Some(5));
    assert_eq!(band(35), Some(5), "same tenth: no refresh");
    assert_eq!(band(36), Some(6));
    assert_eq!(band(54), Some(9));
    assert_eq!(band(60), Some(10));
    assert_eq!(
        band(90),
        Some(10),
        "past the deadline the band stops moving"
    );
}

#[test]
fn a_zero_budget_has_no_band() {
    let clock = TurnClock {
        elapsed: MINUTE,
        budget: Duration::ZERO,
    };
    assert_eq!(clock.band(), None);
}

#[test]
fn the_clock_takes_the_tighter_of_the_run_and_host_budgets() {
    let ctx = RunContext::new(RunConfig::new("r").with_timeout_ms(120_000), ());
    let host = TurnClock::of(&ctx, Some(60 * MINUTE)).expect("a budget is set");
    assert_eq!(host.budget, Duration::from_secs(120));

    let ctx = RunContext::new(RunConfig::new("r"), ());
    assert!(TurnClock::of(&ctx, None).is_none(), "no budget, no clock");
    let host = TurnClock::of(&ctx, Some(MINUTE)).expect("the host budget applies");
    assert_eq!(host.budget, MINUTE);
}

fn result_text(result: &ToolResult) -> String {
    result.output()
}

async fn run_after_tool(middleware: &TurnClockMiddleware, ctx: &mut RunContext<()>) -> ToolResult {
    let mut result = ToolResult::success("output");
    let identity = ToolInvocationIdentity::new("call-1", "shell");
    Middleware::<(), ()>::after_tool(middleware, ctx, &(), &identity, &mut result)
        .await
        .expect("after_tool succeeds");
    result
}

#[tokio::test]
async fn early_in_the_run_results_are_left_alone() {
    let middleware = TurnClockMiddleware::new(Some(60 * MINUTE));
    let mut ctx = RunContext::new(RunConfig::new("r"), ());
    let result = run_after_tool(&middleware, &mut ctx).await;
    assert_eq!(result_text(&result), "output");
}

#[tokio::test]
async fn without_a_budget_results_are_left_alone() {
    let middleware = TurnClockMiddleware::new(None);
    let mut ctx = RunContext::new(RunConfig::new("r"), ());
    let result = run_after_tool(&middleware, &mut ctx).await;
    assert_eq!(result_text(&result), "output");
}

#[tokio::test]
async fn past_half_the_budget_the_result_carries_the_clock_once_per_band() {
    // A 1 ms budget is spent by the time the hook runs, so the clock is past
    // half with no dependence on scheduling.
    let middleware = TurnClockMiddleware::new(Some(Duration::from_millis(1)));
    let mut ctx = RunContext::new(RunConfig::new("r"), ());
    std::thread::sleep(Duration::from_millis(5));

    let first = run_after_tool(&middleware, &mut ctx).await;
    assert!(
        matches!(
            first.content.last(),
            Some(ToolContent::Text { text }) if text.starts_with("\n[turn budget: ")
                && text.ends_with(" remaining]")
        ),
        "the note is appended as a trailing block: {:?}",
        first.content
    );
    assert!(result_text(&first).starts_with("output"));

    let second = run_after_tool(&middleware, &mut ctx).await;
    assert_eq!(
        result_text(&second),
        "output",
        "the same band is not repeated"
    );

    // A different run has its own clock state.
    let mut other = RunContext::new(RunConfig::new("r2"), ());
    std::thread::sleep(Duration::from_millis(5));
    let fresh = run_after_tool(&middleware, &mut other).await;
    assert_ne!(result_text(&fresh), "output");
}

#[tokio::test]
async fn a_markdown_rendering_carries_the_note_too() {
    let middleware = TurnClockMiddleware::new(Some(Duration::from_millis(1)));
    let mut ctx = RunContext::new(RunConfig::new("r"), ());
    std::thread::sleep(Duration::from_millis(5));
    let mut result = ToolResult::success("plain").with_markdown("## rendered");
    let identity = ToolInvocationIdentity::new("call-1", "shell");
    Middleware::<(), ()>::after_tool(&middleware, &mut ctx, &(), &identity, &mut result)
        .await
        .expect("after_tool succeeds");
    let markdown = result.markdown_formatted.expect("markdown kept");
    assert!(
        markdown.starts_with("## rendered\n[turn budget: "),
        "{markdown:?}"
    );
}

#[tokio::test]
async fn a_json_result_is_skipped_and_the_band_waits_for_the_next_text_result() {
    let middleware = TurnClockMiddleware::new(Some(Duration::from_millis(1)));
    let mut ctx = RunContext::new(RunConfig::new("r"), ());
    std::thread::sleep(Duration::from_millis(5));
    let identity = ToolInvocationIdentity::new("call-1", "propose_workflow");

    let mut json_result = ToolResult::success("{\"type\":\"workflow_proposal\"}");
    Middleware::<(), ()>::after_tool(&middleware, &mut ctx, &(), &identity, &mut json_result)
        .await
        .expect("after_tool succeeds");
    assert_eq!(
        result_text(&json_result),
        "{\"type\":\"workflow_proposal\"}"
    );

    let text = run_after_tool(&middleware, &mut ctx).await;
    assert_ne!(
        result_text(&text),
        "output",
        "the band was not spent on the JSON row"
    );
}
