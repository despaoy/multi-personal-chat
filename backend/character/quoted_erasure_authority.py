"""Literal source-prefix selectors for already scoped partial observations.

Matching source text identifies an observation, never its subject or truth.
The scheduler resolves selectors across all visible rows before Top-K; ranking
and model IDs do not supply authority. Ambiguous selectors authorize no write.
"""

import json
import re
from datetime import datetime

_PAIRS = {"“": "”", "‘": "’", "「": "」", "『": "』", '"': '"'}


def has_source_selector(message):
    return bool(re.search(r"原话以\s*[“‘「『\"]", message or ""))


def masked_quotes(message):
    masked = list(message)
    stack, quotes = [], []
    start = None
    escaped = False
    for index, char in enumerate(message):
        if escaped:
            masked[index] = " "
            escaped = False
            continue
        if stack and char == "\\":
            escaped = True
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:
                quotes.append((start, index + 1, message[start + 1 : index]))
        elif char in _PAIRS:
            if not stack:
                start = index
            stack.append(_PAIRS[char])
        elif char in _PAIRS.values():
            raise ValueError("unmatched quotation boundary")
        if stack or (quotes and quotes[-1][1] == index + 1):
            masked[index] = " "
    if stack or escaped:
        raise ValueError("incomplete quotation boundary")
    return "".join(masked), quotes


def partial_source_packet(record):
    """Only app-produced original source fragments with checkable provenance."""
    try:
        metadata = record.get("metadata") or json.loads(record.get("metadata_json") or "{}")
        evidence = record.get("evidence") or json.loads(record.get("evidence_json") or "[]")
        source_ids = record.get("source_message_ids") or json.loads(record.get("source_message_ids_json") or "[]")
        original_id = record.get("source_message_id")
        projection = metadata["erasure_evidence_projection"]
        (source,) = projection["sources"]
        spans = source["spans"]
        clock = datetime.fromisoformat(record["observed_at"])
        if (
            record.get("status", "active") != "active"
            or record.get("memory_type") != "shared_event"
            or metadata.get("content_semantics") != "quoted_source"
            or metadata.get("speaker_role") != "user"
            or metadata.get("described_subject") != "not_resolved"
            or projection.get("version") != 1
            or projection.get("kind") != "original_source_fragments"
            or projection.get("complete_original_source") is not False
            or not original_id
            or source_ids != [original_id]
            or source["source_message_id"] != original_id
            or not isinstance(evidence, list)
            or not evidence
            or not isinstance(spans, list)
            or len(spans) != len(evidence)
            or clock.utcoffset() is None
        ):
            return None
        previous = 0
        for item, span in zip(evidence, spans, strict=True):
            a, z = span
            if not isinstance(item, str) or not item or type(a) is not int or type(z) is not int:
                return None
            if a < previous or z - a != len(item):
                return None
            previous = z
        if spans[0][0] != 0:
            return None  # A surviving middle fragment is not the original source prefix.
        return dict(
            speaker_role="user",
            described_subject="not_resolved",
            content_semantics="quoted_source",
            complete_original_source=False,
            evidence=list(evidence),
            source_message_ids=list(source_ids),
            observed_at=record["observed_at"],
        )
    except (TypeError, ValueError, KeyError, AttributeError):
        return None


def quoted_erasure_plan(message, records=None):
    if not has_source_selector(message):
        return None
    from character.erasure_authority import (
        _ARCHIVE,
        _NEGATIVE,
        _PROTECTED,
        PartialErasurePlan,
        _identity,
        _normalized,
        _object,
    )
    from character.memory_llm import _ARCHIVE_ONLY_ERASURE_NEGATION, _ERASE_REQUEST_PATTERN

    try:
        masked, quotes = masked_quotes(message)
    except ValueError:
        return PartialErasurePlan(False, (), ())
    if re.search(r"^(?:如果|假如|假设|要是)|(?:他说|她说|朋友说|你说过)", masked.strip()):
        return PartialErasurePlan(False, (), ())
    affirmative, protected, allowed, kept = [], [], [], []
    unresolved = False
    invalid = False
    for match in re.finditer(r"[^，,。；;！？!?\n]+(?:[，,。；;！？!?\n]|$)", masked):
        clause = match.group().rstrip("，,。；;！？!?\n").strip()
        raw = message[match.start() : match.end()].rstrip("，,。；;！？!?\n").strip()
        if _ARCHIVE_ONLY_ERASURE_NEGATION.fullmatch(clause):
            continue
        retention = _PROTECTED.fullmatch(clause)
        positive = bool(_ERASE_REQUEST_PATTERN.search(clause))
        if retention:
            if _ARCHIVE.search(retention.group("object")):
                invalid = True
            protected.append(raw)
        elif _NEGATIVE.search(clause):
            invalid = True
            continue
        elif positive:
            invalid |= match.group().endswith(("?", "？")) or bool(re.search(r"如果|假如|假设|要是", clause))
            affirmative.append(raw)
        else:
            continue
        if records is None:
            continue
        selectors = [
            text
            for a, z, text in quotes
            if match.start() <= a < z <= match.end()
            and re.search(r"原话以\s*$", message[match.start() : a])
            and re.match(r"\s*(?:为)?开头", message[z : match.end()])
        ]
        if selectors:
            # A full first fragment or a complete original first sentence is
            # required; a short shared word cannot authorize a destructive ID.
            matches = []
            for record in records:
                packet = partial_source_packet(record)
                if packet and all(
                    len(anchor) >= 24
                    and packet["evidence"][0].startswith(anchor)
                    and (anchor == packet["evidence"][0] or anchor.endswith(("。", "！", "!", ".")))
                    for anchor in selectors
                ):
                    matches.append(record)
            if len(matches) != 1:
                unresolved = True
                matches = []
        else:
            matches = [record for record in records if _object(record) and _object(record) in _normalized(clause)]
            if retention and not matches:
                unresolved = True
        (kept if retention else allowed).extend(matches)
    if not affirmative:
        return PartialErasurePlan(False, (), tuple(protected))
    kept_ids = tuple(dict.fromkeys(_identity(row) for row in kept))
    allowed_ids = tuple(dict.fromkeys(_identity(row) for row in allowed if _identity(row) not in kept_ids))
    return PartialErasurePlan(
        not invalid,
        tuple(affirmative),
        tuple(protected),
        () if unresolved else allowed_ids,
        kept_ids,
        tuple(dict.fromkeys(str(row.get("memory_key") or "") for row in kept)),
        tuple(dict.fromkeys(_object(row) for row in kept)),
        unresolved,
    )
