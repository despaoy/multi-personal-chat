"""Historical keyword-scope ablation; not an application classifier."""

import re

from knowledge.query_tasks import independent_content_query
from knowledge.task_plan import plan_turn_tasks


def explicit_identity_subject(analysis) -> str:
    entities = tuple(dict.fromkeys(analysis.entities))
    text = independent_content_query(getattr(analysis, "normalized_query", "").strip())
    return _identity_subject(entities, text)


def identity_evidence_subject(analysis) -> str:
    """An identity subtask can own RAG evidence without owning the whole turn."""
    independent = explicit_identity_subject(analysis)
    if independent:
        return independent
    tasks = plan_turn_tasks(getattr(analysis, "normalized_query", "").strip())
    content = [task for task in tasks if task.kind == "content"]
    if len(content) != 1 or not any(task.kind == "memory" for task in tasks):
        return ""
    return _identity_subject(tuple(dict.fromkeys(analysis.entities)), content[0].query)


def _identity_subject(entities: tuple[str, ...], text: str) -> str:
    if len(entities) != 1 or not text:
        return ""
    entity = entities[0]
    pattern = (
        r"(?:请问)?" + re.escape(entity) + r"(?:到底|究竟)?(?:是谁|是什么人|是什么身份|是什么人物|的身份是什么)[？?。]?"
    )
    return entity if re.fullmatch(pattern, text) else ""


_OPEN_SCOPE = re.compile(r"为什么|为何|怎么|原因|导致|造成|影响|通过|第三|其他|各自|分别|共同|所有")


def explicit_relation_pair(analysis) -> frozenset[str]:
    entities = tuple(dict.fromkeys(analysis.entities))
    text = getattr(analysis, "normalized_query", "")
    if len(entities) != 2 or not text or _OPEN_SCOPE.search(text):
        return frozenset()
    for first, second in (entities, entities[::-1]):
        pattern = (
            re.escape(first)
            + r"(?:与|和|跟|、)"
            + re.escape(second)
            + r"(?:之间|两人|二人)?"
            # A source-domain adjunct does not add a relation endpoint.
            # This scopes ownership, not truth across narrative layers.
            + r"(?:在(?:原作|原著|作品|故事|小说)(?:里|中))?"
            + r"的?(?:到底|究竟)?(?:是|有)?(?:什么)?关系"
        )
        if re.search(pattern, text):
            return frozenset(entities)
    return frozenset()
