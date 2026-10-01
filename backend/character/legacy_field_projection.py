"""Read-only personal views of legacy generic claims, backed by complete speech.

No keys or version endpoints are migrated. A model label, clipped evidence or
an unrelated source cannot supply field authority; uncertainty remains visible.
"""

from __future__ import annotations

import json
import re
from datetime import datetime

from character.memory_extractor import extract_memories
from character.memory_query import PERSONAL_MEMORY_KEYS
from character.memory_subject import is_source_observation

_VALUES = {
    "user_name": ("用户说自己叫", ""),
    "user_origin": ("用户说自己来自", ""),
    "user_residence": ("用户说自己居住在", ""),
    "user_major": ("用户说自己的专业是", ""),
    "user_workplace": ("用户说自己在", "工作"),
    "user_study_stage": ("用户说自己是", ""),
}


def _normalized(text):
    return re.sub(r"\s+", " ", text).strip()


def _fields(text, value=None):
    result = {}
    for item in extract_memories(text):
        if item.memory_key not in PERSONAL_MEMORY_KEYS.values() or item.qualifiers:
            continue
        if value is not None:
            prefix, suffix = _VALUES[item.memory_key]
            if item.content != prefix + value + suffix:
                continue
        result[(item.memory_key, item.content)] = item
    return result


def project_legacy_personal_record(row, sources):
    """Project only a uniquely supported field; preserve every persisted row."""
    key = str(row.get("memory_key") or "")
    if not key.startswith("fact_") or row.get("memory_type") != "user_fact":
        return row
    quotes = row.get("evidence")
    if not isinstance(quotes, (list, tuple)) or not quotes or not all(isinstance(q, str) for q in quotes):
        return row
    value = key[5:]
    candidates = {}
    for quote in quotes:
        candidates.update(_fields(quote, value))
    # An explicit version edge may retain the old generic key after replacing
    # its value. Recover from one complete assertion, never collateral fields.
    if (
        not candidates
        and str(row.get("relation_type") or "").upper() == "SUPERSEDE"
        and row.get("supersedes_memory_id")
    ):
        for quote in quotes:
            candidates.update(_fields(quote))
        if len(candidates) != 1:
            candidates = {}
    if not candidates:
        return row
    view = dict(row)
    view["legacy_field_keys"] = tuple(dict.fromkeys(item.memory_key for item in candidates.values()))
    source_ids = row.get("source_message_ids") or ()
    complete = bool(source_ids) and all(str(source_id) in sources for source_id in source_ids)
    bodies = tuple(sources[str(source_id)]["body"] for source_id in source_ids if str(source_id) in sources)
    supported = {}
    if complete:
        for body in bodies:
            if not any(_normalized(quote) in _normalized(body) for quote in quotes):
                complete = False
                break
            full_fields = _fields(body)
            if not set(candidates) <= set(full_fields):
                complete = False
                break
            requested_keys = {identity[0] for identity in candidates}
            if any(identity[0] in requested_keys and identity not in candidates for identity in full_fields):
                complete = False
                break
            for identity, item in full_fields.items():
                if identity in candidates:
                    supported[identity] = item
        complete = complete and set(supported) == set(candidates)
    metadata = row.get("metadata") or {}
    if (
        complete
        and len(supported) == 1
        and isinstance(metadata, dict)
        and not metadata.get("qualifiers")
        and not is_source_observation(row)
        and float(row.get("confidence", 1)) >= 0.8
    ):
        (item,) = supported.values()
        view.update(memory_key=item.memory_key, content=item.content, legacy_memory_key=key)
        return view
    # A recognizable quote is related evidence, not proof of absence or a
    # current assertion. Preserve late qualifications in the full source.
    view["metadata"] = dict(metadata) if isinstance(metadata, dict) else {}
    view["metadata"].update(content_semantics="quoted_source", speaker_role="user", described_subject="not_resolved")
    view.update(
        temporal_mode="observation",
        content="用户原话记录（旧字段尚未核实，不代表当前状态）：" + json.dumps(bodies or quotes, ensure_ascii=False),
    )
    view["retrieval_content"] = "\n".join(bodies or quotes)
    view["evidence"] = tuple(dict.fromkeys((*bodies, *quotes)))
    # The compiler admits observations only with a trusted aware receipt.
    # Preserve source time; never invent applicability or use the read clock.
    receipts = []
    for source_id in source_ids:
        try:
            receipt = datetime.fromisoformat(str(sources[str(source_id)]["observed_at"]))
        except (KeyError, ValueError, TypeError):
            continue
        if receipt.utcoffset() is not None:
            receipts.append(receipt)
    if receipts:
        view["temporal_observed_at"] = max(receipts).isoformat()

    return view
