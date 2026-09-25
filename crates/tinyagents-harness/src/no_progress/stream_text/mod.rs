//! Detect a streamed response that keeps starting sentences the same way.
//!
//! The provider can stream an open-ended sequence of process narration without
//! ever completing a model call. Tool-call repeat guards run only after that
//! call, so they cannot stop this shape of stall.

use std::collections::VecDeque;

const MIN_CHARS: usize = 600;
const WINDOW: usize = 10;
const MATCHES_TO_STALL: usize = 8;

/// Per-model-call detector for a long run of similarly opened sentences.
/// Feed visible text fragments in stream order, regardless of chunk boundaries.
#[derive(Default)]
pub struct StreamTextStallDetector {
    sentence: String,
    recent_starts: VecDeque<String>,
    total_chars: usize,
}

impl StreamTextStallDetector {
    /// Observe the next visible text fragment. Returns `true` once a strong
    /// sentence-start recurrence is present; callers should stop that stream.
    pub fn observe(&mut self, fragment: &str) -> bool {
        for ch in fragment.chars() {
            self.total_chars += 1;
            if matches!(ch, '.' | '!' | '?' | '\n') {
                self.finish_sentence();
            } else if self.sentence.len() < 1024 {
                self.sentence.push(ch);
            }
        }
        if self.total_chars < MIN_CHARS || self.recent_starts.len() < WINDOW {
            return false;
        }
        self.recent_starts
            .iter()
            .filter(|start| is_process_start(start))
            .any(|candidate| {
                self.recent_starts
                    .iter()
                    .filter(|start| *start == candidate)
                    .count()
                    >= MATCHES_TO_STALL
            })
    }

    /// A structured tool call is progress; any later visible text starts a
    /// fresh window instead of inheriting narration before that call.
    pub fn reset(&mut self) {
        self.sentence.clear();
        self.recent_starts.clear();
        self.total_chars = 0;
    }

    fn finish_sentence(&mut self) {
        let words: Vec<&str> = self.sentence.split_whitespace().collect();
        if words.len() >= 4 && self.sentence.chars().count() >= 24 {
            let start = words[..2]
                .iter()
                .map(|word| {
                    word.chars()
                        .filter(|ch| ch.is_alphanumeric())
                        .collect::<String>()
                        .to_lowercase()
                })
                .collect::<Vec<_>>()
                .join(" ");
            if !start.trim().is_empty() {
                self.recent_starts.push_back(start);
                if self.recent_starts.len() > WINDOW {
                    self.recent_starts.pop_front();
                }
            }
        }
        self.sentence.clear();
    }
}

/// A repeated subject in an explanation ("Jev is ...") may be intentional.
/// The failure shape is repeated self-narration of actions never taken.
fn is_process_start(start: &str) -> bool {
    matches!(
        start,
        "let me" | "i will" | "i need" | "i should" | "i can" | "ill now"
    )
}

#[cfg(test)]
mod test;
