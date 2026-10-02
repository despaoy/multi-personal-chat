"""Time focus for a complete, explicitly current personal lookup.

This is a read-routing view. The complete original message must continue to
reach ranking, evidence reviewers, source attribution and generation.
"""

import re
from dataclasses import dataclass

from character.memory_query import MemoryQueryPlan, lookup_fields, profile_lookup_fields


def current_lookup_time_text(message: str) -> str | None:
    """Return a locally current task; defer unknown or multiple tasks unchanged.

    A date outside a complete explicit-current lookup does not date that lookup.
    A historical modifier inside the task cannot pass the shared closed grammar.
    Question/command backgrounds and standalone temporal frames are deliberately
    not interpreted here: their dependencies require a richer multi-task plan.
    """
    parts = re.split(r"([。！？!?；;\n])", message)
    clauses = [
        (parts[i], parts[i + 1] if i + 1 < len(parts) else "") for i in range(0, len(parts), 2) if parts[i].strip()
    ]
    if len(clauses) < 2:
        return None
    original, _ = clauses[-1]
    task = "".join(original.split())
    task = re.sub(r"^(?:请告诉我|你还记得|你记得)[，,]?", "", task)
    explicit_current = bool(re.search(r"现居地|当前|目前|现在", task))
    if not explicit_current:
        return None
    lookup = re.sub(r"^我的(?:当前|目前|现在)", "我的", task)
    if not (lookup_fields(lookup) or profile_lookup_fields(task)):
        return None
    # Never erase earlier requests or a detached date that might frame the task.
    # These are conservative exclusions, not claims that other text is factual.
    for background, delimiter in clauses[:-1]:
        text = "".join(background.split())
        if delimiter in {"？", "?"} or re.search(
            r"什么|哪里|哪儿|何时|为何|为什么|怎么|是否|吗|请|告诉|列出|核对|回忆|整理|说明|介绍", text
        ):
            return None
        if re.fullmatch(
            r"(?:在|于)?(?:\d{4}年(?:[一二三四五六七八九十\d]{1,3}月(?:\d{1,2}日)?)?|"
            r"以前|之前|过去|当时|那时|那年|去年|前年)(?:时|的时候|期间)?",
            text,
        ):
            return None
    return original


@dataclass(frozen=True)
class PersonalTimeTask:
    """An original closed lookup clause and the personal fields it requests."""

    query: str
    fields: tuple[str, ...]

    @property
    def time_expression(self) -> str:
        text = "".join(self.query.split())
        text = re.sub(r"^(?:请告诉我|你还记得|你记得)[，,]?", "", text)
        match = _TASK_TIME_PREFIX.match(text)
        return match.group() if match else ""

    def matches(self, row: dict, field: str | None = None) -> bool:
        plan = MemoryQueryPlan(self.fields)
        return plan.matches(row, field) if field is not None else bool(plan.matched_fields(row))


_TASK_TIME_PREFIX = re.compile(
    r"^(?:在|于)?(?:\d{4}年(?:[一二三四五六七八九十\d]{1,3}月)?|"
    r"今年[一二三四五六七八九十\d]{1,3}月|去年(?:[一二三四五六七八九十\d]{1,3}月)?|"
    r"前年(?:[一二三四五六七八九十\d]{1,3}月)?|以前|之前|过去|当时|曾经|上次|原来)"
)


def personal_time_tasks(message: str) -> tuple[PersonalTimeTask, ...]:
    """Parse every clause, or defer the entire compound query unchanged.

    Only independent complete self-lookups have task-local constraints. Unknown
    background/assertion, detached dates, foreign owners and unresolved time
    reference cannot be partially dropped to manufacture a simpler query.
    Temporal expressions stay in original query text for the existing resolver.
    """
    clauses = [part for part in re.split(r"[。！？!?；;\n]", message) if part.strip()]
    if len(clauses) < 2:
        return ()
    tasks = []
    for original in clauses:
        text = "".join(original.split())
        text = re.sub(r"^(?:请告诉我|你还记得|你记得)[，,]?", "", text)
        prefix = _TASK_TIME_PREFIX.match(text)
        if prefix:
            text = text[prefix.end() :]
        text = re.sub(r"^我的(?:当前|目前|现在)", "我的", text)
        fields = lookup_fields(text) or profile_lookup_fields(text)
        if not fields:
            return ()
        tasks.append(PersonalTimeTask(original, fields))
    return tuple(tasks)
