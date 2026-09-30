"""Conservative subject scope for explicitly named two-person relations.

Only direct pair requests are scoped. Causal, bridge and multi-party queries
continue through ordinary retrieval; a mention is not a relation endpoint.
"""
from __future__ import annotations

import re

_OPEN_SCOPE = re.compile(r'为什么|为何|怎么|原因|导致|造成|影响|通过|第三|其他|各自|分别|共同|所有')


def explicit_relation_pair(analysis) -> frozenset[str]:
    entities = tuple(dict.fromkeys(analysis.entities))
    text = getattr(analysis, 'normalized_query', '')
    if len(entities) != 2 or not text or _OPEN_SCOPE.search(text):
        return frozenset()
    for first, second in (entities, entities[::-1]):
        pattern = (re.escape(first) + r'(?:与|和|跟|、)' + re.escape(second)
                   + r'(?:之间|两人|二人)?'
                   # A source-domain adjunct does not add a relation endpoint.
                   # This scopes ownership, not truth across narrative layers.
                   + r'(?:在(?:原作|原著|作品|故事|小说)(?:里|中))?'
                   + r'的?(?:到底|究竟)?(?:是|有)?(?:什么)?关系')
        if re.search(pattern, text):
            return frozenset(entities)
    return frozenset()


def relation_endpoints(document) -> frozenset[str]:
    if document.document_type != 'relation':
        return frozenset()
    metadata = document.metadata
    subject, target = metadata.get('subject'), metadata.get('target')
    if isinstance(subject, str) and subject and isinstance(target, str) and target:
        return frozenset((subject, target))
    # Missing structured endpoints are unknown, not authority to infer a pair
    # from names merely mentioned somewhere in an evidence paragraph.
    return frozenset()


def in_relation_scope(document, pair: frozenset[str]) -> bool:
    endpoints = relation_endpoints(document)
    return not pair or not endpoints or endpoints == pair
