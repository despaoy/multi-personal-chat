"""Structured ownership for explicitly scoped identity evidence."""
from __future__ import annotations

from knowledge.relation_scope import relation_endpoints


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
