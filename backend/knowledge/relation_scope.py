"""Structured relationship endpoints; query scope is supplied by the caller."""
from __future__ import annotations


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
