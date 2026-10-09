"""Query segmentation and keyword tokens for retrieval."""
import re


def tokenize(text: str) -> list[str]:
    import jieba

    tokens = re.findall(r"\w+", text.lower())
    tokens.extend(token for token in jieba.cut(text) if token.strip() and not token.isascii())
    return tokens


def segment(text: str) -> list[str]:
    """Required word segmentation for query scoring and reformulation."""
    import jieba

    return [token for part in jieba.cut(text) if (token := part.strip())]
