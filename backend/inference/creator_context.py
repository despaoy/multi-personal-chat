"""Fit role-library examples and reference excerpts after the primary prompt."""

from __future__ import annotations

import json
from dataclasses import replace
from html import escape
from typing import TYPE_CHECKING

from inference.context_budget import CONTEXT_SAFETY_MARGIN_TOKENS, estimated_tokens
from knowledge.retrieval_core.tokenization import segment
from knowledge.text_splitter import smart_text_split

if TYPE_CHECKING:
    from inference.generation_request import GenerationPlan, GenerationRequest

_OPEN = '<creator_material trust="untrusted" purpose="style_and_reference">\n'
_CLOSE = '</creator_material>\n\n'


def _examples(messages):
    position = 0
    while position < len(messages):
        end = position + 1
        if (messages[position]["role"] == "user" and end < len(messages)
                and messages[end]["role"] == "assistant"):
            end += 1
        yield messages[position:end]
        position = end


def _reference_excerpts(query, text):
    terms = {term for term in segment(query.casefold()) if len(term) > 1 and term.isalnum()}
    if not terms:
        return ()
    matches = []
    for line in text.splitlines():
        for excerpt in smart_text_split(line.strip(), chunk_size=700, overlap=100):
            folded = excerpt.casefold()
            score = sum(len(term) for term in terms if term in folded)
            if score:
                matches.append((score, excerpt))
    return (excerpt for _, excerpt in sorted(matches, key=lambda match: -match[0]))


def add_creator_context(request: GenerationRequest, plan: GenerationPlan) -> GenerationPlan:
    """Use only spare capacity and keep creator material within the user role."""
    limit = request.context_window_tokens - request.max_tokens - CONTEXT_SAFETY_MARGIN_TOKENS
    cost = sum(estimated_tokens(message["content"]) + 4 for message in plan.messages)
    allowance = min(limit - cost, 2048)
    remaining = allowance - estimated_tokens(_OPEN + _CLOSE)
    if remaining <= 0:
        return plan

    blocks = {}
    references = _reference_excerpts(request.message, request.creator_reference) if request.creator_reference else ()
    candidates = (
        ("参考资料节选", references),
        ("示范对话（供参考表达风格）", _examples(request.creator_examples)),
    )
    for label, items in candidates:
        for item in items:
            block = escape(f"{label}：{json.dumps(item, ensure_ascii=False)}\n", quote=False)
            if block in blocks:
                continue
            block_cost = estimated_tokens(block)
            if block_cost <= remaining:
                blocks[block] = None
                remaining -= block_cost
    if not blocks:
        return plan

    current = plan.messages[-1]
    content = _OPEN + "".join(blocks) + _CLOSE + current["content"]
    if estimated_tokens(content) - estimated_tokens(current["content"]) > allowance:
        return plan
    return replace(plan, messages=(*plan.messages[:-1], {**current, "content": content}))
