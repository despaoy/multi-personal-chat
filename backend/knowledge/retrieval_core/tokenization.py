"""Shared keyword tokens for persisted and domain BM25 indexes."""
import re


def tokenize(text: str) -> list[str]:
    import jieba

    tokens = re.findall(r"\w+", text.lower())
    tokens.extend(token for token in jieba.cut(text) if token.strip() and not token.isascii())
    return tokens
