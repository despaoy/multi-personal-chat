"""Read statement provenance separately from currently accepted facts.

Only explicit, complete personal record-review tasks use this path. Pending
statements are quoted with their lifecycle label, never admitted as facts.
"""
from __future__ import annotations

import json
import re

from character.memory_extractor import extract_memories
from character.memory_query import PERSONAL_MEMORY_KEYS

_FIELDS = {'姓名': 'name', '名字': 'name', '专业': 'major', '籍贯': 'origin',
           '现居地': 'residence', '居住地': 'residence', '工作地点': 'workplace',
           '工作单位': 'workplace', '年级': 'study_stage'}
_STATUS = {'active': '当前记录', 'current': '当前记录', 'pending': '待确认，不能当作当前事实',
           'superseded': '旧版本', 'archived': '归档版本'}


def mention_query(message: str) -> tuple[str, str] | None:
    match = re.fullmatch(r'(?:请列出)?我(?:之前|以前)?(?:说过|提过)(?:哪些|什么)([^，,。！？!?；;\n]{1,30})[。？?]?', message.strip())
    if not match:
        return None
    subject = match[1]
    if subject in _FIELDS:
        return _FIELDS[subject], ''
    constraint = re.fullmatch(r'(.{0,20}?)(?:限制|禁忌|注意事项)', subject)
    if constraint:
        topic = constraint[1].removesuffix('的')
        if not re.search(r'和|与|以及|并|然后|顺便|请|帮|你|他|她|朋友|同事', topic):
            return 'constraints', topic
    return None


def review_mentions(message: str, records: list[dict], *, complete_read: bool) -> str | None:
    task = mention_query(message)
    if task is None:
        return None
    field, topic = task
    entries: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    skipped = False
    size = 0
    for row in records:
        status = str(row.get('status', 'active')).lower()
        if status not in _STATUS or str(row.get('relation_type', '')).upper() in {'RETRACT', 'ERASE', 'NOOP'}:
            continue
        metadata = row.get('metadata') or {}
        if not isinstance(metadata, dict):
            continue
        temporal = metadata.get('temporal_provenance')
        semantic_source = (isinstance(temporal, dict) and temporal.get('version') == 1
                           and temporal.get('producer') == 'semantic_memory'
                           and row.get('temporal_mode') in {'observation', 'asserted_state'})
        observation = semantic_source and row.get('temporal_mode') == 'observation'
        if metadata.get('origin') not in {'rule_v2', 'rule_candidate'} and not semantic_source:
            continue
        key = str(row.get('memory_key') or '')
        if field == 'constraints':
            if not metadata.get('qualifiers'):
                continue
        elif key != PERSONAL_MEMORY_KEYS[field]:
            continue
        sources = row.get('source_message_ids')
        if not isinstance(sources, (list, tuple)) or not sources or not all(isinstance(s, str) and s for s in sources):
            continue
        evidence = row.get('evidence')
        if not isinstance(evidence, (list, tuple)):
            continue
        for quote in evidence:
            if not isinstance(quote, str) or (topic and topic not in quote):
                continue
            # Quote only independently recoverable user assertions, not a
            # generated summary, a provenance parent's text, or forged slots.
            if not observation and not any(item.memory_key == key for item in extract_memories(quote)):
                continue
            observed_at = str(row.get('temporal_observed_at') or '') if observation else ''
            identity = (status, quote, observed_at)
            if identity in seen:
                continue
            seen.add(identity)
            label = (f'原话记录，时效未核实，记录于{observed_at}；{_STATUS[status]}'
                     if observation else _STATUS[status])
            line = f'- [{label}] ' + json.dumps(quote, ensure_ascii=False)
            if len(entries) >= 8 or size + len(line) > 3000:
                skipped = True
                continue
            entries.append(line)
            size += len(line)
    if not entries:
        return '这次没有找到能核对来源的相关记忆记录；这不代表你从未说过。'
    suffix = '\n这里只展示部分可核对记录。' if skipped or not complete_read else ''
    return '本次可核对的记忆原话如下（不等于完整聊天历史）：\n' + '\n'.join(entries) + suffix
