"""One bounded keyword-append rule for corrective retrieval adapters."""
from collections.abc import Iterable

from .retrieval_core.tokenization import segment

_STOPWORDS = {
    "的",
    "了",
    "是",
    "在",
    "我",
    "你",
    "他",
    "她",
    "它",
    "们",
    "这",
    "那",
    "怎么",
    "什么",
    "为什么",
    "哪里",
    "哪个",
    "请问",
    "一下",
    "可能",
    "应该",
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "what",
    "how",
    "why",
}


def append_retrieval_keywords(query: str, texts: Iterable[str]) -> str:
    keywords: list[str] = []
    query_folded = query.casefold()
    for text in texts:
        for token in segment(text):
            if (len(token) > 1 and token not in keywords and token not in _STOPWORDS
                    and token.casefold() not in query_folded):
                keywords.append(token)
        if len(keywords) >= 8:
            break
    return f"{query} {' '.join(keywords[:5])}" if keywords else query
