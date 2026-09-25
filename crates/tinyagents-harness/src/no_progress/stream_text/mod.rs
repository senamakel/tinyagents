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
    stalled: bool,
}

impl StreamTextStallDetector {
    /// Observe the next visible text fragment. Returns `true` once a strong
    /// sentence-start recurrence is present; callers should stop that stream.
    pub fn observe(&mut self, fragment: &str) -> bool {
        if self.stalled {
            return true;
        }
        for ch in fragment.chars() {
            self.total_chars += 1;
            if matches!(ch, '.' | '!' | '?' | '\n') {
                self.finish_sentence();
                self.stalled |= self.has_stalled_window();
                if self.stalled {
                    return true;
                }
            } else if self.sentence.len() < 1024 {
                self.sentence.push(ch);
            }
        }
        self.stalled |= self.has_stalled_window();
        self.stalled
    }

    fn has_stalled_window(&self) -> bool {
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
        self.stalled = false;
    }

    fn finish_sentence(&mut self) {
        let words: Vec<String> = self
            .sentence
            .split_whitespace()
            .map(|word| {
                word.chars()
                    .filter(|ch| ch.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase()
            })
            .filter(|word| !word.is_empty())
            .take(2)
            .collect();
        if words.len() == 2 {
            self.recent_starts.push_back(words.join(" "));
            if self.recent_starts.len() > WINDOW {
                self.recent_starts.pop_front();
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
