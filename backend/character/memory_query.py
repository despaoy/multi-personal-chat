"""Typed query plan over existing memory fields, never generated facts.

Unknown language stays on hybrid retrieval. Recognized personal-field requests
get field coverage, rather than relying on similarity to stored values.
Authorization and lifecycle filters remain owned by the repository/service.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from character.memory_subject import OWNER_PATTERN, SELF_OWNERS

_FIELD_NOUNS = {
    'name': ('姓名', '名字'),
    'origin': ('来源地', '籍贯', '故乡', '家乡'),
    'residence': ('现居地', '住址', '居住地'),
    'major': ('专业', '学科', '大学专业'),
    'workplace': ('工作地点', '工作单位', '工作场所'),
    'study_stage': ('年级', '学习阶段'),
}
_FIELD_KEYS = {'name': 'user_name', 'origin': 'user_origin', 'residence': 'user_residence',
               'major': 'user_major', 'workplace': 'user_workplace', 'study_stage': 'user_study_stage'}
# Retrieval may match a predicate fragment; execution still requires a complete
# personal lookup. Keep verbal fragments out of the noun grammar.
_RETRIEVAL_VERBS = {'origin': ('来自',), 'residence': ('住哪里', '住哪儿', '住在')}
_FIELDS = {
    **{field: ('|'.join(re.escape(word) for word in (*nouns, *_RETRIEVAL_VERBS.get(field, ()))),
               (_FIELD_KEYS[field],)) for field, nouns in _FIELD_NOUNS.items()},
    'profile': (r'基本信息|基本资料|个人资料|个人信息',
                ('user_name', 'user_origin', 'user_residence', 'user_major', 'user_workplace', 'user_study_stage')),
    'constraints': (r'限制|禁忌|注意事项', ()),
}
_PATTERNS = {key: re.compile(pattern) for key, (pattern, _) in _FIELDS.items()}
PERSONAL_MEMORY_KEYS = {field: keys[0] for field, (_, keys) in _FIELDS.items()
                        if field not in {'profile', 'constraints'}}
# Legacy schema combines distinct predicates; its presence proves neither
# typed predicate, but prevents proving either one absent.
AMBIGUOUS_MEMORY_KEYS = {'user_location': ('origin', 'residence')}
_OWNERS = OWNER_PATTERN
_SELF = SELF_OWNERS
MEMORY_FIELD_NAMES = {noun: field for field, nouns in _FIELD_NOUNS.items() for noun in nouns}
_CLOSED_VERBAL = (
    (re.compile(r'我来自(?:哪里|哪儿)'), 'origin'),
    (re.compile(r'我(?:(?:目前|现在))?住(?:在)?(?:哪里|哪儿)'), 'residence'),
    (re.compile(r'我叫(?:什么|什么名字)'), 'name'),
)


def storage_fields(message: str) -> tuple[str, ...]:
    """Complete personal storage-status questions, shared by routing/rendering."""
    text = ''.join(message.split()).rstrip('。？?')
    prefix = r'(?:请问)?你(?:现在|目前)?(?:还|已经)?'
    match = re.fullmatch(prefix + r'(?:是否|有没有)(?:保存|记录|记住)(?:了|着)?我的(.+?)', text)
    if match is None:
        match = re.fullmatch(prefix + r'(?:保存|记录|记住)(?:了|着)?我的(.+?)(?:了吗|吗|了没有|没有)', text)
    if match is None:
        return ()
    parts = re.split(r'和|与|、', match.group(1))
    if not all(part in MEMORY_FIELD_NAMES for part in parts):
        return ()
    return tuple(dict.fromkeys(MEMORY_FIELD_NAMES[part] for part in parts))


def lookup_fields(message: str) -> tuple[str, ...]:
    """Closed personal lookup syntax shared by task planning and rendering."""
    text = ''.join(message.split()).rstrip('。？?')
    text = re.sub(r'^(?:请告诉我|你还记得|你记得)', '', text)
    text = re.sub(r'(?:吗|呢)$', '', text)
    nouns = re.fullmatch(r'我的(.+?)(?:(?:分别)?是(?:什么|哪里|哪儿))?', text)
    if nouns:
        parts = re.split(r'和|与|、', nouns.group(1))
        if parts and all(part in MEMORY_FIELD_NAMES for part in parts):
            return tuple(dict.fromkeys(MEMORY_FIELD_NAMES[part] for part in parts))
    fields: list[str] = []
    for index, clause in enumerate(re.split(r'[，,]', text)):
        if index and re.fullmatch(r'(?:目前|现在)住(?:在)?(?:哪里|哪儿)', clause):
            clause = '我' + clause
        field = next((field for pattern, field in _CLOSED_VERBAL if pattern.fullmatch(clause)), None)
        if field is None:
            return ()
        fields.append(field)
    return tuple(dict.fromkeys(fields))


def profile_lookup_fields(message: str) -> tuple[str, ...]:
    """A complete self-profile task, including only parsed output controls.

    This is a dependency/read plan, not a deterministic response. Any unknown
    field, other owner or additional content task defers to ordinary routing.
    """
    text = ''.join(message.split())
    clauses = [part for part in re.split(r'[。！？!?；;\n]', text) if part]
    if not clauses:
        return ()
    match = re.fullmatch(
        r'(?:请)?(?:用(?:一句话|一段话|列表|表格))?'
        r'(?:核对|回忆|列出|整理|介绍|说明)(?:一下)?我的(?:当前)?'
        r'(?:个人资料|个人信息|基本资料|基本信息)'
        r'(?:[：:](.+?)(?:各|分别)?是什么)?', clauses[0])
    if match is None:
        return ()
    for control in clauses[1:]:
        for part in re.split(r'[，,]', control):
            if not re.fullmatch(
                r'(?:不做建议|不要建议|不提供建议|不要提供建议|'
                r'(?:只写|只列|只回答)(?:我本人|本人|我)(?:的)?(?:当前)?(?:资料|信息)|'
                r'(?:只写|只列|只回答)当前本人(?:资料|信息))', part):
                return ()
    if match[1] is None:
        return tuple(PERSONAL_MEMORY_KEYS)
    parts = re.split(r'和|与|、', match[1])
    if not parts or any(part not in MEMORY_FIELD_NAMES for part in parts):
        return ()
    fields = {MEMORY_FIELD_NAMES[part] for part in parts}
    return tuple(field for field in PERSONAL_MEMORY_KEYS if field in fields)


@dataclass(frozen=True)
class MemoryQueryPlan:
    fields: tuple[str, ...] = ()
    excluded_fields: tuple[str, ...] = ()

    def matches(self, row: dict[str, Any], field: str) -> bool:
        if field == 'constraints':
            metadata = row.get('metadata') or {}
            return isinstance(metadata, dict) and bool(metadata.get('qualifiers'))
        return (str(row.get('memory_key') or '') in _FIELDS[field][1]
                or bool(set(row.get('legacy_field_keys') or ()) & set(_FIELDS[field][1])))

    def matched_fields(self, row: dict[str, Any]) -> tuple[str, ...]:
        return tuple(field for field in self.fields if self.matches(row, field))

    def suppresses(self, row: dict[str, Any]) -> bool:
        return not self.matched_fields(row) and any(self.matches(row, field) for field in self.excluded_fields)


def plan_memory_query(query: str) -> MemoryQueryPlan:
    profile_fields = profile_lookup_fields(query)
    if profile_fields:
        return MemoryQueryPlan(profile_fields)
    requested: set[str] = set()
    excluded: set[str] = set()
    for clause in re.split(r'[，,。！？!?；;\n]', query):
        clause = ''.join(clause.split())
        for field, pattern in _PATTERNS.items():
            for match in pattern.finditer(clause):
                owners = list(_OWNERS.finditer(clause[:match.start()]))
                if not owners:
                    continue
                owner = owners[-1].group()
                (requested if owner in _SELF else excluded).add(field)
    # Closed compound reads already resolve inherited self-ownership in the
    # shared grammar (e.g. "我来自哪里，目前住哪里"). Retrieval must cover every
    # field the executor will answer, without extending ownership heuristics.
    requested.update(lookup_fields(query))
    # A profile is a set of slots, not one all-or-nothing retrieval target.
    profile_fields = set(_FIELDS) - {'profile', 'constraints'}
    if 'profile' in requested:
        requested = (requested - {'profile'}) | profile_fields
    if 'profile' in excluded:
        excluded = (excluded - {'profile'}) | profile_fields
    return MemoryQueryPlan(tuple(f for f in _FIELDS if f in requested),
                           tuple(f for f in _FIELDS if f in excluded and f not in requested))
