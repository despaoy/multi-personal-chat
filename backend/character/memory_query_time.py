"""Time focus for a complete, explicitly current personal lookup.

This is a read-routing view. The complete original message must continue to
reach ranking, evidence reviewers, source attribution and generation.
"""

import re

from character.memory_query import lookup_fields, profile_lookup_fields


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
