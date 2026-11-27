"""Highlight related utterances already admitted to a long character chat."""

import json
import math
import re
from collections import Counter
from heapq import nlargest

from character.memory_extractor import memory_write_allowed
from inference.context_budget import CONTEXT_SAFETY_MARGIN_TOKENS, estimated_tokens
from inference.prompt_policy import dialogue_evidence_block

_QUESTION = re.compile(r"[?？]|哪|什么|谁|是否|有没有")
_MIN_USER_HISTORY_CHARS = 128_000


def focus_dialogue(request, messages):
    question = request.message.strip()
    if messages[0]["role"] != "system" or not (2 <= len(question) <= 256 and _QUESTION.search(question)):
        return messages
    if sum(len(m["content"]) for m in messages[:-1] if m["role"] == "user") < _MIN_USER_HISTORY_CHARS:
        return messages
    history = request.complete_history or request.history
    if sum(m["role"] == "user" for m in history) != sum(m["role"] == "user" for m in messages[:-1]):
        return messages
    terms = sorted({question[i:i + size].casefold() for size in (2, 3, 4)
                    for i in range(len(question) - size + 1) if question[i:i + size].isalnum()})
    if not terms:
        return messages
    frequency = Counter()
    candidates = []
    for index, (user, reply) in enumerate(zip(messages, messages[1:])):
        if user["role"] != "user" or reply["role"] != "assistant":
            continue
        if not all(memory_write_allowed(m["content"]) for m in (user, reply)):
            continue
        for match in re.finditer(r"[^。！？!?\n]+[。！？!?]?", user["content"]):
            sentence = match.group().strip()
            if len(sentence) > 160:
                continue
            lower = sentence.casefold()
            overlap = [t for t in terms if t in lower]
            if overlap:
                frequency.update(overlap)
                candidates.append((index, match.start(), sentence, overlap))
    weights = {term: math.log((1 + len(candidates)) / (1 + count)) for term, count in frequency.items()}
    ranked = nlargest(4, candidates, key=lambda item: sum(weights[t] for t in item[3]))
    quotes = [{"sequence": i, "role": "user", "quote": sentence} for i, _, sentence, overlap
              in sorted(ranked) if any(weights[t] for t in overlap)]
    if len({quote["sequence"] for quote in quotes}) < 2:
        return messages
    block = dialogue_evidence_block(json.dumps(quotes, ensure_ascii=False))
    current = {**messages[-1], "content": block + "\n\n" + messages[-1]["content"]}
    system = {**messages[0], "content": messages[0]["content"] + "\n本轮附有按时间排列的对话原句。"
              "根据前后指代核对人物和事项；沿用原安排指向原来的约定，明确取消或替换则以之后的说法为准。"}
    focused = (system, *messages[1:-1], current)
    cost = sum(estimated_tokens(m["content"]) + 4 for m in focused)
    if cost + request.max_tokens + CONTEXT_SAFETY_MARGIN_TOKENS > request.context_window_tokens:
        return messages
    return focused
