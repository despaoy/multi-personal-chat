"""Bounded original-text slices, preferring paragraph and sentence boundaries."""

import re
from bisect import bisect_right


def smart_text_split(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    """Keep source characters and exact overlap; budgets are measured in characters."""
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if type(overlap) is not int or not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be an integer from zero to chunk_size - 1")
    if not text.strip():
        return []

    paragraph_ends = [match.end() for match in re.finditer(r"\r?\n[ \t]*\r?\n", text)]
    sentence_ends = [match.end() for match in re.finditer(r"[。！？；.!?;\n]", text)]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            minimum = start + max(overlap + 1, chunk_size // 2)
            for boundaries in (paragraph_ends, sentence_ends):
                position = bisect_right(boundaries, end) - 1
                if position >= 0 and boundaries[position] >= minimum:
                    end = boundaries[position]
                    break
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


# Existing API and evaluation callers share this implementation.
simple_text_split = smart_text_split
