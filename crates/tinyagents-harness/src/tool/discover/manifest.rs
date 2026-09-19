//! The budgeted listing of deferred tools embedded in `tool_search`'s
//! description.
//!
//! The listing is the only thing a model sees of a deferred tool before it
//! searches, so it has to exist; but it is paid on every request, so it has to
//! be bounded. Three forms, tried largest first until one fits the budget:
//!
//! 1. `- name: first sentence of the description (≤ 60 chars)`
//! 2. `- name`
//! 3. `N tools are searchable.`
//!
//! The catalogue is name-sorted, so the rendered bytes are identical from one
//! turn to the next.

use super::types::DeferredCatalog;

/// Longest description excerpt kept per tool in the full form.
pub const MANIFEST_DESCRIPTION_CHARS: usize = 60;

/// Rough bytes-per-token used to turn a token budget into a byte budget.
const BYTES_PER_TOKEN: usize = 4;

/// Renders the deferred-tool manifest within `token_budget` tokens.
#[must_use]
pub fn render_manifest(catalog: &DeferredCatalog, token_budget: usize) -> String {
    let byte_budget = token_budget.saturating_mul(BYTES_PER_TOKEN);
    let header = format!("{} deferred tool(s) are searchable:\n", catalog.len());

    let full: String = catalog
        .schemas()
        .map(|schema| {
            format!(
                "- {}: {}\n",
                schema.name,
                first_sentence(&schema.description, MANIFEST_DESCRIPTION_CHARS)
            )
        })
        .collect();
    if header.len() + full.len() <= byte_budget {
        return format!("{header}{full}");
    }

    let names: String = catalog
        .schemas()
        .map(|schema| format!("- {}\n", schema.name))
        .collect();
    if header.len() + names.len() <= byte_budget {
        return format!("{header}{names}");
    }

    format!("{} deferred tool(s) are searchable.\n", catalog.len())
}

/// The first sentence of `text`, clipped to `max_chars`, whitespace collapsed.
#[must_use]
pub fn first_sentence(text: &str, max_chars: usize) -> String {
    let collapsed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    // A sentence ends at `.`/`!`/`?` followed by a space (or the end), so
    // "e.g." and "v1.2" inside a sentence do not cut it short.
    let end = collapsed
        .char_indices()
        .find(|&(index, ch)| {
            matches!(ch, '.' | '!' | '?')
                && collapsed[index + ch.len_utf8()..]
                    .chars()
                    .next()
                    .is_none_or(char::is_whitespace)
        })
        .map(|(index, _)| index);
    let sentence = end.map_or(collapsed.as_str(), |end| &collapsed[..end]);
    let mut clipped: String = sentence.chars().take(max_chars).collect();
    if clipped.len() < sentence.len() {
        clipped.push('…');
    }
    clipped
}
