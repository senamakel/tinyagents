//! Unit tests for cost accounting.
//!
//! These cover [`super::estimate_cost`] pricing each token class (input,
//! output, cache read/creation, reasoning) against a [`ModelPricing`] entry,
//! the handling of missing prices as zero cost, and [`super::CostTotals`]
//! `+`/`+=` accumulation with total recomputation.

use super::*;
use crate::cost::ModelPricing;
use tinyinference_llm::usage::Usage;

fn pricing() -> ModelPricing {
    ModelPricing {
        input_per_token: Some(0.001),
        output_per_token: Some(0.002),
        cache_read_input_per_token: Some(0.0001),
        cache_creation_input_per_token: Some(0.0005),
        output_reasoning_per_token: Some(0.003),
        ..ModelPricing::default()
    }
}

#[test]
fn estimates_each_component() {
    let usage = Usage {
        input_tokens: 1000,
        output_tokens: 500,
        total_tokens: 1500,
        cache_read_tokens: 100,
        cache_creation_tokens: 200,
        reasoning_tokens: 10,
        charged_amount: None,
        context_window_tokens: None,
    };
    let cost = estimate_cost(&pricing(), &usage);
    // cache_read_tokens (100) and reasoning_tokens (10) are subsets of
    // input_tokens/output_tokens, so the standard-rate cost only applies to
    // the non-cached/non-reasoning remainder: (1000-100)*0.001 = 0.9,
    // (500-10)*0.002 = 0.98.
    assert!((cost.input_cost - 0.9).abs() < 1e-9);
    assert!((cost.output_cost - 0.98).abs() < 1e-9);
    // 100*0.0001 + 200*0.0005 = 0.01 + 0.1 = 0.11
    assert!((cost.cache_cost - 0.11).abs() < 1e-9);
    assert!((cost.reasoning_cost - 0.03).abs() < 1e-9);
    assert!((cost.total_cost - 2.02).abs() < 1e-9);
}

#[test]
fn cache_and_reasoning_tokens_are_not_double_counted() {
    // A fully-cached prompt: all 90k input tokens were served from cache.
    // Charging the full input_tokens at the standard rate *and* the
    // cache_read_tokens at the cache rate would double-bill the cached
    // tokens; only the cache rate should apply to them.
    let usage = Usage {
        input_tokens: 90_000,
        output_tokens: 100,
        total_tokens: 90_100,
        cache_read_tokens: 90_000,
        cache_creation_tokens: 0,
        reasoning_tokens: 0,
        charged_amount: None,
        context_window_tokens: None,
    };
    let cost = estimate_cost(&pricing(), &usage);
    assert_eq!(cost.input_cost, 0.0);
    assert!((cost.cache_cost - 9.0).abs() < 1e-9);
}

#[test]
fn missing_prices_are_zero() {
    let usage = Usage::new(100, 100);
    let cost = estimate_cost(&ModelPricing::default(), &usage);
    assert_eq!(cost.total_cost, 0.0);
}

#[test]
fn cost_totals_accumulate() {
    let mut totals = CostTotals::new();
    let usage = Usage::new(1000, 0);
    totals += estimate_cost(&pricing(), &usage);
    totals += estimate_cost(&pricing(), &usage);
    assert!((totals.input_cost - 2.0).abs() < 1e-9);
    assert!((totals.total_cost - 2.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// Tiered pricing
// ---------------------------------------------------------------------------

fn tiered_pricing() -> ModelPricing {
    ModelPricing {
        // Flat rate is the fallback the highest tier partially relies on.
        input_per_token: Some(0.001),
        output_per_token: Some(0.002),
        tiers: vec![
            PriceTier {
                up_to_tokens: Some(200_000),
                input: Some(0.001),
                output: Some(0.002),
                cache_read: Some(0.0001),
                cache_write: None,
            },
            PriceTier {
                up_to_tokens: None,
                input: Some(0.002),
                output: Some(0.004),
                cache_read: None,
                cache_write: None,
            },
        ],
        ..ModelPricing::default()
    }
}

#[test]
fn tiered_pricing_uses_the_matching_capped_tier() {
    let usage = Usage::new(100_000, 1000);
    let cost = estimate_cost(&tiered_pricing(), &usage);
    assert!((cost.input_cost - 100.0).abs() < 1e-9);
    assert!((cost.output_cost - 2.0).abs() < 1e-9);
}

#[test]
fn tiered_pricing_falls_through_to_the_unlimited_tier_above_the_cap() {
    let usage = Usage::new(300_000, 1000);
    let cost = estimate_cost(&tiered_pricing(), &usage);
    // The unlimited tier's higher input rate (0.002) applies once the call's
    // input tokens exceed the capped tier's 200K threshold.
    assert!((cost.input_cost - 600.0).abs() < 1e-9);
    assert!((cost.output_cost - 4.0).abs() < 1e-9);
}

#[test]
fn tiered_pricing_falls_back_to_flat_rate_for_unset_tier_fields() {
    // The unlimited tier leaves `cache_read` unset; it must fall back to the
    // flat `cache_read_input_per_token` rather than pricing at zero.
    let mut pricing = tiered_pricing();
    pricing.cache_read_input_per_token = Some(0.0002);
    let usage = Usage {
        input_tokens: 300_000,
        output_tokens: 0,
        total_tokens: 300_000,
        cache_read_tokens: 50_000,
        cache_creation_tokens: 0,
        reasoning_tokens: 0,
        charged_amount: None,
        context_window_tokens: None,
    };
    let cost = estimate_cost(&pricing, &usage);
    assert!((cost.cache_cost - 10.0).abs() < 1e-9);
}

#[test]
fn no_tiers_declared_uses_flat_pricing() {
    assert!(select_tier(&pricing(), 1_000_000).is_none());
}

#[test]
fn select_tier_picks_the_smallest_sufficient_cap() {
    let pricing = tiered_pricing();
    assert_eq!(select_tier(&pricing, 50_000).unwrap().up_to_tokens, Some(200_000));
    assert_eq!(select_tier(&pricing, 200_000).unwrap().up_to_tokens, Some(200_000));
    assert_eq!(select_tier(&pricing, 200_001).unwrap().up_to_tokens, None);
}
