"""Resolve bounded knowledge follow-ups using user topics, never model claims.

This is retrieval query construction, not fact extraction. Only registered
domain anchors are inherited; unrelated turns terminate the chain. Ambiguous
pronouns are left intact for the answerer to clarify, not resolved to a person.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from knowledge.query_tasks import requests_source_text
from knowledge.retrieval_core.query import QueryAnalyzer

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_FOLLOWUP = re.compile(
    r'^(?:那|那么|所以|但是|不过|确定|请)?(?:他|她|它|这|那|谁|为什么|怎么|具体|能否|能不能)'
)
_RESET = re.compile(r'换.{0,3}话题|不聊|先不说|与.{0,12}无关')


def _is_followup(text: str) -> bool:
    # A source-request is a retrieval task, independent of an exact sentence
    # prefix. Mentioning a source in an unrelated statement is not that task.
    return bool(requests_source_text(text) or _FOLLOWUP.search(text))


def contextual_retrieval_query(message: str, history: Sequence[Mapping[str, str]], configs) -> str:
    if _RESET.search(message) or not _is_followup(message):
        return message
    analyzer = QueryAnalyzer(configs)
    current = analyzer.analyze(message)
    # A current explicit entity is self-contained; do not mix unrelated people.
    if current.entities:
        return message
    topics = []
    for row in reversed(history[-8:]):
        if row.get('role') != 'user':
            continue
        text = str(row.get('content', '')).strip()
        if not text or len(text) > 600 or _RESET.search(text):
            break
        analysis = analyzer.analyze(text)
        if analysis.matched_domains:
            topics.append(text)
            if analysis.entities:
                if len(analysis.matched_domains) != 1:
                    return message
                return '\n'.join([*reversed(topics), message])
        elif _is_followup(text):
            topics.append(text)
        else:
            break
    return message
