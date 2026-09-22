"""Offline input-representation ablations; no gold, query parsing or new facts.

Lifecycle eligibility remains upstream. The model still decides relevance and
temporal applicability; this adapter never forces a label or restores a record.
"""

from __future__ import annotations

import json

from character.evidence_selector import MAX_INPUT_CHARS, InputBudgetError

VIEWS = frozenset({"status_neutral", "interval_prose"})


def transform_messages(messages, view):
    if view not in VIEWS:
        raise ValueError("unknown metadata view")
    if len(messages) != 2 or messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise ValueError("unexpected selector message structure")
    # Decode a fresh object: never mutate original evidence, dates or status in
    # the service/selector. Omitted fields remain available to final compilation.
    payload = json.loads(messages[1]["content"])
    for candidate in payload["candidates"]:
        status = candidate.get("status")
        if status not in {"active", "current", "superseded", "archived"}:
            raise ValueError("metadata ablation only accepts upstream-eligible lifecycle states")
        if status in {"superseded", "archived"} and candidate.get("historical") is not True:
            raise ValueError("inactive versions require upstream historical eligibility")
        del candidate["status"]
        if view == "interval_prose":
            start, end = candidate["valid_from"], candidate["valid_to"]
            if not isinstance(start, str) or not isinstance(end, str):
                raise ValueError("temporal metadata must retain textual source dates")
            candidate["time_scope_explanation"] = (
                f"这条记录所述事实的有效时段：起点={start or '未注明'}，终点={end or '未注明'}。"
                "请比较问题所问时间与这个时段；不是将终点与今天比较。"
            )
            if candidate.get("historical") is True:
                candidate["time_scope_explanation"] += "这是历史版本；历史版本本身不表示在其有效时段内不成立。"
    encoded = json.dumps(payload, ensure_ascii=False)
    if len(encoded) > MAX_INPUT_CHARS:
        raise InputBudgetError("metadata representation exceeds full-input budget")
    return [dict(messages[0]), {"role": "user", "content": encoded}]


class TemporalMetadataReviewer:
    def __init__(self, reviewer, view):
        if view not in VIEWS:
            raise ValueError("unknown metadata view")
        self.reviewer = reviewer
        self.view = view

    async def __call__(self, messages):
        return await self.reviewer(transform_messages(messages, self.view))
