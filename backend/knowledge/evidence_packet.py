"""Preserve indexed claim ownership and narrative scope beside its evidence.

These are source annotations, not independently verified facts or instructions.
Unknown or malformed fields are omitted, never inferred from nearby names.
"""
from __future__ import annotations

import json
from collections.abc import Mapping


def render_card_evidence(document: Mapping, evidence: str) -> str:
    metadata = document.get('metadata')
    metadata = metadata if isinstance(metadata, Mapping) else {}
    annotations = {}
    for key in ('subject', 'predicate', 'value', 'relation', 'target', 'viewpoint',
                'story_title', 'semantic_temporal_scope', 'source_temporal_scope'):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            annotations[key] = value
    for key in ('reality_status', 'temporal_scope', 'content_scope'):
        value = document.get(key)
        if isinstance(value, str) and value.strip():
            annotations[key] = value
    source = document.get('source')
    if isinstance(source, Mapping):
        location = {key: source[key] for key in ('source_path', 'line_start', 'line_end')
                    if key in source and (isinstance(source[key], str)
                                           or type(source[key]) is int)}
        if location:
            annotations['source'] = location
    header = ('索引标注（未独立核验）：' + json.dumps(annotations, ensure_ascii=False) + '\n') if annotations else ''
    return f"【卡片关联证据】{document.get('title', '')}\n{header}{evidence}"
