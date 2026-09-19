//! BM25 ranking over deferred tool declarations.
//!
//! Names nothing from the rest of the harness: it takes `(sort_key, text)`
//! pairs and returns ranked indices, so the tool-specific half (what text is
//! searchable, what a hit looks like) stays in [`super::types`].
//!
//! Hand-rolled rather than the `bm25` crate: the arithmetic is ~80 lines and
//! the harness is a dependency floor for every embedder.

use std::collections::HashMap;

/// Words that carry no capability meaning, dropped from a query before ranking.
///
/// A document-frequency threshold alone cannot do this on a small corpus:
/// with three deferred tools, "a" may appear in exactly one description and
/// so rank as the *most* distinguishing term in "send a calendar invite".
/// Deliberately short and English-only — it can only remove terms, so a
/// description in another language ranks exactly as it would without it.
/// Words that could name a capability ("up", as in "look up") are left in.
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "for", "from", "how", "i",
    "if", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or", "our", "so", "that",
    "the", "their", "them", "then", "there", "these", "they", "this", "to", "was", "we", "were",
    "what", "when", "which", "who", "will", "with", "would", "you", "your",
];

/// BM25 term-frequency saturation (the standard default).
const K1: f64 = 1.2;
/// BM25 length normalisation (the standard default).
const B: f64 = 0.75;

/// Splits text into search terms.
///
/// Splits on non-alphanumerics **and** on a lower→upper transition, so
/// `memory_hybrid_search` and `readWorkflowResource` both yield the words a
/// person would type. Without the camel-case rule a query for "workflow"
/// misses a tool whose only mention of it is inside an identifier.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut previous_lower = false;
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            if ch.is_uppercase() && previous_lower && !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
            current.extend(ch.to_lowercase());
            previous_lower = ch.is_lowercase() || ch.is_numeric();
        } else if !current.is_empty() {
            out.push(std::mem::take(&mut current));
            previous_lower = false;
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// A ranked corpus of short documents, identified by their index into the
/// slice the caller built the index from.
#[derive(Default, Debug, Clone)]
pub struct Bm25Index {
    documents: Vec<Document>,
    document_frequency: HashMap<String, usize>,
    average_length: f64,
}

#[derive(Debug, Clone)]
struct Document {
    /// Used only to break score ties deterministically.
    sort_key: String,
    tokens: Vec<String>,
}

impl Bm25Index {
    /// Builds from `(sort_key, searchable_text)` pairs, in caller order.
    ///
    /// `sort_key` breaks ties; make it the id the caller would print, so two
    /// identical queries produce identical output. An unstable order would make
    /// a model's transcript non-reproducible for no benefit.
    pub fn build<'a>(documents: impl IntoIterator<Item = (&'a str, &'a str)>) -> Self {
        let documents: Vec<Document> = documents
            .into_iter()
            .map(|(sort_key, text)| Document {
                sort_key: sort_key.to_string(),
                tokens: tokenize(text),
            })
            .collect();

        let mut document_frequency: HashMap<String, usize> = HashMap::new();
        for doc in &documents {
            let mut seen: Vec<&str> = Vec::new();
            for token in &doc.tokens {
                if !seen.contains(&token.as_str()) {
                    seen.push(token);
                    *document_frequency.entry(token.clone()).or_insert(0) += 1;
                }
            }
        }

        let total: usize = documents.iter().map(|d| d.tokens.len()).sum();
        let average_length = if documents.is_empty() {
            0.0
        } else {
            total as f64 / documents.len() as f64
        };

        Self {
            documents,
            document_frequency,
            average_length,
        }
    }

    /// `true` when the corpus holds no documents.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Number of documents in the corpus.
    #[must_use]
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Ranks against `query`, best first, returning **indices** into the corpus.
    ///
    /// Only documents scoring above zero are returned. Padding the list out to
    /// `limit` with unrelated entries would spend exactly the tokens deferral
    /// exists to save, and would invite the model to call something unrelated
    /// to what it asked for.
    #[must_use]
    pub fn search(&self, query: &str, limit: usize) -> Vec<usize> {
        let terms = self.significant(tokenize(query));
        if terms.is_empty() || self.documents.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(f64, usize)> = self
            .documents
            .iter()
            .enumerate()
            .map(|(index, doc)| (self.score(doc, &terms), index))
            .filter(|(score, _)| *score > 0.0)
            .collect();

        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    self.documents[a.1]
                        .sort_key
                        .cmp(&self.documents[b.1].sort_key)
                })
        });
        scored.into_iter().take(limit).map(|(_, i)| i).collect()
    }

    /// Drops query terms too common in this corpus to mean anything.
    ///
    /// The IDF below carries the standard `+ 1`, which keeps a term present in
    /// every document at a small **positive** weight — without it a corpus of
    /// one document scores every term at zero and nothing is ever findable.
    /// The cost is that "a" and "the" score, so a document-frequency filter
    /// runs first: drop the term when `df >= max(2, ceil(0.8 * n))`. The floor
    /// of 2 makes it inert on a one-document corpus; [`STOPWORDS`] covers the
    /// small-corpus case the threshold cannot.
    fn significant(&self, terms: Vec<String>) -> Vec<String> {
        let n = self.documents.len();
        if n == 0 {
            return Vec::new();
        }
        let threshold = std::cmp::max(2, (0.8 * n as f64).ceil() as usize);
        terms
            .into_iter()
            .filter(|term| !STOPWORDS.contains(&term.as_str()))
            .filter(|term| *self.document_frequency.get(term).unwrap_or(&0) < threshold)
            .collect()
    }

    fn score(&self, doc: &Document, terms: &[String]) -> f64 {
        let length = doc.tokens.len() as f64;
        let count = self.documents.len() as f64;
        terms
            .iter()
            .map(|term| {
                let frequency = doc.tokens.iter().filter(|t| *t == term).count() as f64;
                if frequency == 0.0 {
                    return 0.0;
                }
                let df = *self.document_frequency.get(term).unwrap_or(&0) as f64;
                // Standard BM25 IDF with the +1 that keeps a term present in
                // every document at a small positive weight (see `significant`).
                let idf = ((count - df + 0.5) / (df + 0.5) + 1.0).ln();
                let normalised = frequency * (K1 + 1.0)
                    / (frequency + K1 * (1.0 - B + B * length / self.average_length.max(1.0)));
                idf * normalised
            })
            .sum()
    }
}
