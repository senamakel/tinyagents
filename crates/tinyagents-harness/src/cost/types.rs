//! Cost accounting types.
//!
//! [`CostTotals`] is the additive value that lets cost roll up across a
//! recursive run tree (model call → run → parent run).

use serde::{Deserialize, Serialize};

/// Per-token pricing for a model.
///
/// Every field is optional: `None` means the price is unknown or does not
/// apply, rather than that the token class is free.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ModelPricing {
    /// Price per input token.
    #[serde(default)]
    pub input_per_token: Option<f64>,
    /// Price per output token.
    #[serde(default)]
    pub output_per_token: Option<f64>,
    /// Discounted price per cached input token.
    #[serde(default)]
    pub cache_read_input_per_token: Option<f64>,
    /// Price per input token written to a prompt cache.
    #[serde(default)]
    pub cache_creation_input_per_token: Option<f64>,
    /// Price per audio input token.
    #[serde(default)]
    pub input_audio_per_token: Option<f64>,
    /// Price per reasoning output token.
    #[serde(default)]
    pub output_reasoning_per_token: Option<f64>,
    /// Context-size-tiered pricing (some providers charge more once a call's
    /// context crosses a threshold, e.g. Gemini's price step above 200K
    /// input tokens). Empty by default: when set, [`crate::cost::estimate_cost`]
    /// selects the tier matching the call's input-token count and uses its
    /// rates in place of the flat fields above (falling back to the flat
    /// fields for any rate a matched tier leaves `None`). Tiers do not need
    /// to be pre-sorted; the matching tier is the one with the smallest
    /// [`PriceTier::up_to_tokens`] that is still `>=` the call's input-token
    /// count, or the tier with no `up_to_tokens` (unlimited) when the count
    /// exceeds every capped tier.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tiers: Vec<PriceTier>,
}

/// One context-size pricing tier. See [`ModelPricing::tiers`].
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct PriceTier {
    /// Upper bound (inclusive) on input tokens this tier applies to. `None`
    /// means "no upper bound" — the tier for any call whose input-token
    /// count exceeds every other tier's `up_to_tokens`.
    #[serde(default)]
    pub up_to_tokens: Option<u64>,
    /// Price per input token in this tier. Falls back to
    /// [`ModelPricing::input_per_token`] when `None`.
    #[serde(default)]
    pub input: Option<f64>,
    /// Price per output token in this tier. Falls back to
    /// [`ModelPricing::output_per_token`] when `None`.
    #[serde(default)]
    pub output: Option<f64>,
    /// Price per cached input token in this tier. Falls back to
    /// [`ModelPricing::cache_read_input_per_token`] when `None`.
    #[serde(default)]
    pub cache_read: Option<f64>,
    /// Price per input token written to a prompt cache in this tier. Falls
    /// back to [`ModelPricing::cache_creation_input_per_token`] when `None`.
    #[serde(default)]
    pub cache_write: Option<f64>,
}

/// A breakdown of estimated cost for one or more model calls, in the pricing
/// table's currency (typically USD). All values are accumulating sums.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CostTotals {
    /// Cost attributed to (non-cached) input tokens.
    #[serde(default)]
    pub input_cost: f64,
    /// Cost attributed to output tokens.
    #[serde(default)]
    pub output_cost: f64,
    /// Cost attributed to cache read and cache creation tokens.
    #[serde(default)]
    pub cache_cost: f64,
    /// Cost attributed to reasoning tokens.
    #[serde(default)]
    pub reasoning_cost: f64,
    /// Sum of all component costs.
    #[serde(default)]
    pub total_cost: f64,
}
