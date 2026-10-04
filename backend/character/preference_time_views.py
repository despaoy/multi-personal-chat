"""Task-local time for explicit self preference lookups in a complete query.

Only a parsed self lookup selects a time view. Unparsed clauses are retained
in the original query; they cannot donate dates to an independent lookup.
This is a retrieval view, never source authority or a deterministic answer.
"""

import re
from dataclasses import dataclass

from character.memory_subject import OWNER_PATTERN, SELF_OWNERS

_PERIOD = r"当前|目前|现在|过去|以前|之前|原来|上次|去年|前年|\d{4}年(?:\d{1,2}月)?"
_LOOKUP = re.compile(
    r"^(?:请)?(?P<verb>读取|回忆|查询|核对)?"
    r"(?P<leading>" + _PERIOD + r")?我(?:的)?"
    r"(?P<period>" + _PERIOD + r")?"
    r"(?:仍在所存有效区间的|保存的|的)?"
    r"(?P<subjects>[\w\u4e00-\u9fff、]{0,30}?)偏好"
    r"(?P<question>(?:分别)?是什么)?"
    r"(?:[，,]返回[A-Za-z_][A-Za-z0-9_]*(?:(?:与|和|、)[A-Za-z_][A-Za-z0-9_]*)*)?$"
)


@dataclass(frozen=True)
class PreferenceTimeTask:
    query: str
    subjects: tuple[str, ...]
    time_expression: str
    fields: tuple[str, ...] = ("preference",)

    def matches(self, row, field=None):
        key = str(row.get("memory_key") or "")
        return (
            field in (None, "preference")
            and key.startswith("preference_")
            and (not self.subjects or any(key == "preference_" + subject for subject in self.subjects))
        )


def preference_time_tasks(message):
    tasks = []
    for original in re.split(r"[。！？!?；;\n]", message):
        text = "".join(original.split())
        match = _LOOKUP.fullmatch(text)
        if match is None or not (match["verb"] or match["question"]):
            continue
        owners = list(OWNER_PATTERN.finditer(text))
        if not owners or owners[0].group() not in SELF_OWNERS:
            continue
        if match["leading"] and match["period"] and match["leading"] != match["period"]:
            continue
        subjects = tuple(part for part in re.split(r"和|与|、", match["subjects"]) if part)
        tasks.append(PreferenceTimeTask(original, subjects, match["period"] or match["leading"] or ""))
    return tuple(tasks)
