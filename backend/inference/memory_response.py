"""Execute bounded personal-field reads without asking a model to copy values.

This is not a general intent classifier or a substitute for conversational
generation. Entire requests must match the read grammar. Values require an
unambiguous current evidence-backed packet in the final budget; proven missing
fields may be rendered only without conversation history or branch context.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from character.memory_extractor import complete_self_assertions, extract_memories
from character.memory_query import lookup_fields as lookup_fields
from character.memory_query import storage_fields as storage_fields

if TYPE_CHECKING:
    from character.models import CompiledCharacterContext, MemoryItem

_SLOTS = {
    'name': ('user_name', '用户说自己叫', '', '你叫'),
    'origin': ('user_origin', '用户说自己来自', '', '你来自'),
    'residence': ('user_residence', '用户说自己居住在', '', '现在住在'),
    'major': ('user_major', '用户说自己的专业是', '', '你的专业是'),
    'workplace': ('user_workplace', '用户说自己在', '工作', '你在'),
    'study_stage': ('user_study_stage', '用户说自己是', '', '你是'),
}
_LABELS = {'name': '姓名', 'origin': '来源地', 'residence': '现居地', 'major': '专业',
           'workplace': '工作地点', 'study_stage': '年级'}


def _value(item: MemoryItem, field: str) -> str | None:
    key, prefix, suffix, _ = _SLOTS[field]
    if (item.memory_key != key or item.memory_type != 'user_fact' or item.historical
            or item.temporal_mode == 'observation'
            or item.status not in {'active', 'current'} or item.valid_to
            or item.confidence < .8 or not item.source_message_ids):
        return None
    match = re.fullmatch(re.escape(prefix) + r'([\w· -]{1,30})' + re.escape(suffix), item.content)
    if not match:
        return None
    # Recheck against actual user evidence, never assistant history. Legacy
    # paraphrases and qualified claims are deliberately left to generation.
    for evidence in item.evidence:
        if re.fullmatch(r'(?:现在|目前)(?:住在[\w· -]{1,30}|在[\w· -]{1,30}工作)[。]?', evidence):
            evidence = '我' + evidence
        facts = (complete_self_assertions(evidence) if item.temporal_mode == 'asserted_state'
                 else extract_memories(evidence))
        values = {f.content for f in facts if f.memory_key == key and not f.qualifiers}
        if values == {item.content}:
            return match.group(1)
    return None


@dataclass(frozen=True)
class MemoryFieldRead:
    field: str
    status: str
    value: str | None = None


def read_memory_fields(message: str, context: CompiledCharacterContext | None) -> tuple[MemoryFieldRead, ...]:
    """Preserve why a closed slot cannot be answered, without inferring absence.

    Absence requires the upstream complete scoped read. Missing final packets
    with positive presence are unadmitted, not absent. This reads saved memory
    only; callers must still consider current input and conversation history.
    """
    result = []
    for field in lookup_fields(message):
        if context is None or context.memory_status == 'retrieval_error':
            result.append(MemoryFieldRead(field, 'unavailable'))
            continue
        presence_by_field = dict(context.memory_field_presence)
        if field in presence_by_field and presence_by_field[field] is None:
            result.append(MemoryFieldRead(field, 'unverified'))
            continue
        key = _SLOTS[field][0]
        packets = [item for item in context.memory_packets
                   if item.memory_id in context.used_memory_ids and item.memory_key == key]
        if not packets:
            presence = dict(context.memory_field_presence).get(field)
            status = ('unavailable' if context.memory_status == 'not_checked' else
                      'absent' if presence is False else 'unadmitted' if presence is True else 'unknown')
            result.append(MemoryFieldRead(field, status))
            continue
        values = [_value(item, field) for item in packets]
        status = ('unverified' if None in values else 'conflicting' if len(set(values)) != 1 else 'known')
        result.append(MemoryFieldRead(field, status, values[0] if status == 'known' else None))
    return tuple(result)


def render_absent_memory_response(message: str, context: CompiledCharacterContext | None) -> str | None:
    if (getattr(context, 'episodic_reference_context', '')
            or getattr(context, 'conversation_reference_context', '')):
        return None
    reads = read_memory_fields(message, context)
    if not reads or any(item.status != 'absent' for item in reads):
        return None
    return '我暂时没有找到你的' + '、'.join(_LABELS[item.field] for item in reads) + '记录。'


def render_complete_memory_read(message: str, context: CompiledCharacterContext | None, history) -> str | None:
    """Render a bounded saved-field read, not an open conversation.

    Any history may contain an unsaved fact or correction, so leave that case
    to the ordinary path. Explicit typed values may accompany proven absence
    or a legacy-unresolved field, reported separately rather than inferred.
    Failed, clipped, conflicting and unchecked reads are not absence.
    """
    if (history or context is None or getattr(context, 'branch_context', '')
            or getattr(context, 'episodic_reference_context', '')
            or getattr(context, 'conversation_reference_context', '')):
        return None
    reads = read_memory_fields(message, context)
    legacy_unknown = {field for field, present in context.memory_field_presence if present is None}
    mixed_legacy = (any(item.status == 'known' for item in reads)
                    and any(item.field in legacy_unknown for item in reads))
    if (not reads or not (any(item.status == 'absent' for item in reads) or mixed_legacy)
            or any(item.status not in {'known', 'absent'}
                   and not (mixed_legacy and item.field in legacy_unknown) for item in reads)):
        return None
    known = []
    absent = []
    unresolved = []
    for item in reads:
        if item.status == 'known':
            _, _, suffix, label = _SLOTS[item.field]
            known.append(label + str(item.value) + suffix)
        elif item.field in legacy_unknown:
            unresolved.append(_LABELS[item.field])
        else:
            absent.append(_LABELS[item.field])
    reply = ('我这里记着的是：' + '，'.join(known) + '。') if known else ''
    if absent:
        reply += '我这里暂时没有你的' + '、'.join(absent) + '信息。'
    if unresolved:
        reply += '关于你的' + '、'.join(unresolved) + '，有相关记忆，但暂时还不能明确确认。'
    return reply


def memory_query_result(message: str, context: CompiledCharacterContext | None) -> str:
    """Expose unresolved saved-memory reads without rewriting the user's tasks.

    Only closed personal reads supply fields. No raw user text, guessed values,
    historical conclusions, or claim that visible conversation was searched.
    """
    from knowledge.task_plan import plan_turn_tasks

    if context is None or getattr(context, 'memory_status', 'not_checked') == 'not_checked':
        return ''
    labels = {'absent': '未找到当前可用记录', 'unadmitted': '存在记录但未进入本轮记忆证据',
              'unavailable': '读取失败', 'unverified': '证据不足以直接确认',
              'conflicting': '当前证据存在冲突'}
    fields = {}
    for task in plan_turn_tasks(message):
        if task.kind == 'memory':
            for result in read_memory_fields(task.query, context):
                if result.status in labels:
                    fields[result.field] = {'字段': _LABELS[result.field], '查询状态': result.status,
                                            '可用值': None, '诊断': labels[result.status]}
    if not fields:
        return ''
    return json.dumps({'主体': '当前对话者（用户）', '查询范围': '已保存的长期记忆，不包含当前消息及聊天历史',
                       '字段结果': list(fields.values())}, ensure_ascii=False)


def render_memory_response(message: str, context: CompiledCharacterContext | None, *, history=()) -> str | None:
    from character.memory_mentions import mention_query
    from inference.constraint_response import render_constraint_response

    # No causal watermark links these user turns to the saved projection.
    # They may contain an unsaved update, withdrawal or ownership correction.
    # Do not guess relevance or promote a historical sentence to a new fact.
    # Assistant-only history cannot authoritatively replace user facts.
    has_context_dependency = (
        any(getattr(context, key, '') for key in
            ('episodic_reference_context', 'conversation_reference_context', 'branch_context'))
        or any(row.get('role') == 'user' and str(row.get('content', '')).strip() for row in history)
    )
    constraint_reply = None if has_context_dependency else render_constraint_response(message, context)
    if constraint_reply is not None:
        return constraint_reply

    if mention_query(message) is not None and context is not None:
        if context.memory_review_query == message and context.memory_review_text:
            return context.memory_review_text
        return '这次没有取得可核对的记忆原话，暂时不能确认你此前提过哪些内容。'
    requested_status = storage_fields(message)
    if requested_status and context is not None:
        presence = dict(context.memory_field_presence)
        if not all(field in presence for field in requested_status):
            return '这次没有取得完整的长期记忆保存状态，暂时不能确认是否有这些记录。'
        if any(presence[field] is None for field in requested_status):
            if context.memory_source_status in {'budget_omitted', 'retrieval_error'}:
                return '本轮未能完整核对已保存的原话来源，暂时不能确认这些信息是否有保存；这不表示你没有说过。'
            return '有尚未明确归类的相关记忆，暂时不能确认这些字段的保存状态。'
        present = '、'.join(_LABELS[field] for field in requested_status if presence[field])
        absent = '、'.join(_LABELS[field] for field in requested_status if not presence[field])
        parts = []
        if present:
            parts.append(f'有你的{present}记录')
        if absent:
            parts.append(f'没有你的{absent}记录')
        return '当前可用的长期记忆里，' + '；'.join(parts) + '。'
    if has_context_dependency:
        return None
    reads = read_memory_fields(message, context)
    if not reads or any(item.status != 'known' for item in reads):
        return None
    parts: list[str] = []
    for item in reads:
        _, _, suffix, label = _SLOTS[item.field]
        parts.append(label + str(item.value) + suffix)
    return '我这里记着的是：' + '，'.join(parts) + '。'
