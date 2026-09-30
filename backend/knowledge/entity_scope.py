"""Structured ownership for complete, single-entity identity questions."""
from __future__ import annotations

import re

from knowledge.query_tasks import independent_content_query
from knowledge.relation_scope import relation_endpoints
from knowledge.task_plan import plan_turn_tasks


def explicit_identity_subject(analysis) -> str:
    entities = tuple(dict.fromkeys(analysis.entities))
    text = independent_content_query(getattr(analysis, 'normalized_query', '').strip())
    return _identity_subject(entities, text)


def identity_evidence_subject(analysis) -> str:
    """An identity subtask can own RAG evidence without owning the whole turn."""
    independent = explicit_identity_subject(analysis)
    if independent:
        return independent
    tasks = plan_turn_tasks(getattr(analysis, 'normalized_query', '').strip())
    content = [task for task in tasks if task.kind == 'content']
    if len(content) != 1 or not any(task.kind == 'memory' for task in tasks):
        return ''
    return _identity_subject(tuple(dict.fromkeys(analysis.entities)), content[0].query)


def _identity_subject(entities: tuple[str, ...], text: str) -> str:
    if len(entities) != 1 or not text:
        return ''
    entity = entities[0]
    pattern = (r'(?:请问)?' + re.escape(entity)
               + r'(?:到底|究竟)?(?:是谁|是什么人|是什么身份|是什么人物|的身份是什么)[？?。]?')
    return entity if re.fullmatch(pattern, text) else ''


def in_identity_scope(document, subject: str) -> bool:
    if not subject:
        return True
    if document.document_type == 'fact':
        owner = document.metadata.get('subject')
        # Unknown legacy ownership stays unknown; do not manufacture it from
        # incidental mentions, title fragments, or a model-generated summary.
        return not isinstance(owner, str) or not owner or owner == subject
    if document.document_type == 'relation':
        endpoints = relation_endpoints(document)
        return not endpoints or subject in endpoints
    return True
