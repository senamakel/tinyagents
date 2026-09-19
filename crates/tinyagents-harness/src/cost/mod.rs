//! Cost accounting.
//!
//! Because [`CostTotals`] is additive, cost composes the same way the runtime
//! recurses: each model call's cost folds into its run, and a parent run rolls
//! up the cost of every nested sub-agent and sub-graph beneath it into one
//! total.
//!
//! [`estimate_cost`] prices a [`Usage`] record against a [`ModelPricing`] entry
//! from the registry catalog. [`CostTotals`] supports `+`/`+=` accumulation so
//! a run can roll up cost across many calls.

mod types;

use std::ops::{Add, AddAssign};

use tinyinference_llm::usage::Usage;
pub use types::*;

impl CostTotals {
    /// Creates an empty cost accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Recomputes `total_cost` from the component costs.
    fn recompute_total(&mut self) {
        self.total_cost =
            self.input_cost + self.output_cost + self.cache_cost + self.reasoning_cost;
    }
}

impl Add for CostTotals {
    type Output = CostTotals;

    fn add(mut self, rhs: CostTotals) -> CostTotals {
        self += rhs;
        self
    }
}

impl AddAssign for CostTotals {
    fn add_assign(&mut self, rhs: CostTotals) {
        self.input_cost += rhs.input_cost;
        self.output_cost += rhs.output_cost;
        self.cache_cost += rhs.cache_cost;
        self.reasoning_cost += rhs.reasoning_cost;
        self.recompute_total();
    }
}

/// Selects the [`PriceTier`] matching `input_tokens`, when `pricing` declares
/// any. The match is the tier with the smallest `up_to_tokens` that is still
/// `>= input_tokens`, or the unlimited tier (`up_to_tokens: None`) when
/// `input_tokens` exceeds every capped tier. Returns `None` when `pricing`
/// declares no tiers.
///
/// See [`ModelPricing::tiers`].
fn select_tier(pricing: &ModelPricing, input_tokens: u64) -> Option<&PriceTier> {
    if pricing.tiers.is_empty() {
        return None;
    }
    pricing
        .tiers
        .iter()
        .filter(|tier| tier.up_to_tokens.is_none_or(|cap| input_tokens <= cap))
        .min_by_key(|tier| tier.up_to_tokens.unwrap_or(u64::MAX))
}

/// Estimates the cost of a [`Usage`] record using per-token [`ModelPricing`].
///
/// Missing prices contribute zero. Cache read and cache creation tokens are
/// priced independently when the catalog provides those rates and folded into
/// `cache_cost`.
///
/// Providers report `cache_read_tokens` as a *subset* of `input_tokens` (and
/// `reasoning_tokens` as a subset of `output_tokens`), not an addition to
/// them. Pricing the full `input_tokens`/`output_tokens` count at the
/// standard rate *and* separately pricing the cached/reasoning subset would
/// double-charge those tokens, so the standard-rate cost is computed on the
/// non-cached/non-reasoning remainder only.
///
/// When `pricing` declares [`ModelPricing::tiers`], the tier matching this
/// call's `usage.input_tokens` (see [`select_tier`]) supplies the input,
/// output, cache-read, and cache-write rates, falling back to the flat
/// [`ModelPricing`] fields for any rate the tier leaves unset. Reasoning
/// tokens are always priced at the flat
/// [`ModelPricing::output_reasoning_per_token`] rate; tiers do not currently
/// carry a reasoning override.
pub fn estimate_cost(pricing: &ModelPricing, usage: &Usage) -> CostTotals {
    let price = |rate: Option<f64>, tokens: u64| rate.unwrap_or(0.0) * tokens as f64;

    let tier = select_tier(pricing, usage.input_tokens);
    let input_rate = tier.and_then(|t| t.input).or(pricing.input_per_token);
    let output_rate = tier.and_then(|t| t.output).or(pricing.output_per_token);
    let cache_read_rate = tier
        .and_then(|t| t.cache_read)
        .or(pricing.cache_read_input_per_token);
    let cache_write_rate = tier
        .and_then(|t| t.cache_write)
        .or(pricing.cache_creation_input_per_token);

    let billable_input_tokens = usage.input_tokens.saturating_sub(usage.cache_read_tokens);
    let billable_output_tokens = usage.output_tokens.saturating_sub(usage.reasoning_tokens);

    let mut totals = CostTotals {
        input_cost: price(input_rate, billable_input_tokens),
        output_cost: price(output_rate, billable_output_tokens),
        cache_cost: price(cache_read_rate, usage.cache_read_tokens)
            + price(cache_write_rate, usage.cache_creation_tokens),
        reasoning_cost: price(pricing.output_reasoning_per_token, usage.reasoning_tokens),
        total_cost: 0.0,
    };
    totals.recompute_total();
    totals
}

#[cfg(test)]
mod test;
