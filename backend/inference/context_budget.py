"""Shared tokenizer-free serving-budget estimate (not an exact token count)."""

from dataclasses import dataclass

CONTEXT_SAFETY_MARGIN_TOKENS = 512


def estimated_tokens(text: str) -> int:
    non_ascii = sum(1 for char in text if ord(char) > 127)
    ascii_chars = len(text) - non_ascii
    return non_ascii + (ascii_chars + 3) // 4


@dataclass(frozen=True)
class ReviewContextBudget:
    window_tokens: int
    history_messages: int = 128

    def __post_init__(self):
        if self.window_tokens < 1024 or self.history_messages < 1:
            raise ValueError('Invalid review context budget')

    def fits(self, messages, output_tokens):
        return (sum(estimated_tokens(m['content']) + 4 for m in messages)
                + output_tokens + CONTEXT_SAFETY_MARGIN_TOKENS <= self.window_tokens)
