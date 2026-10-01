"""Purge copied erased statements without inventing a replacement full source.

The caller holds the existing owner lock and commits these evidence projections,
source revocation and claim deletion together. Only exact, whole statements from
the actual shared source can be removed; ambiguous overlap aborts the transaction.
Independent normalized siblings are unchanged. Surviving quoted material is
explicitly a set of original fragments, never a complete reconstructed utterance.
"""

import json
import re

from db.memory_claim_guard import MemoryClaimConflict

_ENDS = frozenset("。！？!?.;；\n")


def _evidence(record):
    try:
        value = json.loads(record.get("evidence_json") or "[]")
    except (TypeError, ValueError) as exc:
        raise MemoryClaimConflict("invalid retained evidence") from exc
    if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
        raise MemoryClaimConflict("unsupported retained evidence shape")
    return value


def _metadata(record):
    try:
        value = json.loads(record.get("metadata_json") or "{}")
    except (TypeError, ValueError) as exc:
        raise MemoryClaimConflict("invalid evidence provenance") from exc
    if not isinstance(value, dict):
        raise MemoryClaimConflict("unsupported evidence provenance")
    return value


def _outside_quotes(body, end):
    stack = []
    pairs = {"“": "”", "「": "」", "『": "』"}
    escaped = False
    for character in body[:end]:
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
        elif character == '"':
            if stack and stack[-1] == '"':
                stack.pop()
            else:
                stack.append('"')
        elif character in pairs:
            stack.append(pairs[character])
        elif character in pairs.values():  # noqa: SIM102 - isolate close-token detection from stack mutation
            if not stack or stack.pop() != character:
                return False
    return not stack


def _statement_span(body, snippet, start):
    end = start + len(snippet)
    prefix = body[:start].rstrip()
    text = snippet.rstrip()
    return (
        bool(text)
        and (not prefix or prefix[-1] in _ENDS)
        and (end == len(body) or text[-1] in _ENDS)
        and _outside_quotes(body, start)
        and _outside_quotes(body, end)
    )


def _positions(body, snippet):
    return [m.start() for m in re.finditer(re.escape(snippet), body)]


def _contains(value, needles):
    if isinstance(value, str):
        return any(needle in value for needle in needles)
    if isinstance(value, dict):
        return any(_contains(k, needles) or _contains(v, needles) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains(v, needles) for v in value)
    return False


def _preference_cuts(body, record):
    """Locate one literal normalized preference predicate, not its broad evidence.

    The original writer may cite an entire multi-fact sentence. Evidence is
    still required, but its breadth never grants deletion of other predicates.
    Unknown clauses/roles remain conflicts; this does not infer event actors.
    """
    key = str(record.get("memory_key") or "")
    value = key.removeprefix("preference_")
    metadata = _metadata(record)
    content = str(record.get("content") or "")
    if (
        not value
        or any(character in "，,。！？!?；;\n" for character in value)
        or metadata.get("attributed_to", "user") != "user"
        or metadata.get("content_semantics") == "quoted_source"
        or not any(content.startswith("用户" + verb + value) for verb in ("喜欢", "偏好", "偏爱"))
    ):
        raise MemoryClaimConflict("erased evidence is not a complete source statement or focused preference")
    evidence = _evidence(record)
    if not any(value in snippet and snippet in body for snippet in evidence):
        raise MemoryClaimConflict("erased evidence is not a complete source statement or grounded preference")
    boundary = r"(?:(?<=[，,。！？!?；;\n])|^)"
    connectors = r"(?:(?:并且|而且|同时|另外|此外|也)\s*)?"
    pattern = re.compile(boundary + r"\s*" + connectors + r"我(?:本人)?(?:喜欢|偏好|偏爱)" + re.escape(value))
    qualifier = re.compile(r"[，,]\s*这是我(?:本人)?(?:明确(?:且|而且))?(?:长期)?的(?:个人)?偏好")
    intervals = []
    for match in pattern.finditer(body):
        start, end = match.span()
        if not _outside_quotes(body, start):
            continue
        if end < len(body) and body[end] not in "，,。！？!?；;\n":
            continue  # A conjunction/qualification is not a literal value edge.
        qualification = qualifier.match(body, end)
        if qualification is not None:
            candidate = qualification.end()
            if candidate < len(body) and body[candidate] not in "，,。！？!?；;\n":
                continue
            end = candidate
        # A comma may introduce a retained predicate; keep it and everything
        # after it. Only a completed terminal punctuation belongs to the cut.
        if end < len(body) and body[end] in "。！？!?；;\n":
            end += 1
        if _outside_quotes(body, end):
            intervals.append((start, end))
    if not intervals:
        raise MemoryClaimConflict("erased evidence is not a complete source statement or separable preference")
    return intervals


def _source_cuts(body, erased_records):
    cuts = set()
    needles = []
    for record in erased_records:
        if str(record.get("memory_key") or "").startswith("preference_"):
            intervals = _preference_cuts(body, record)
            cuts.update(intervals)
            needles.append(str(record["memory_key"]).removeprefix("preference_"))
        else:
            snippets = [snippet for snippet in _evidence(record) if snippet != body]
            if not snippets:
                raise MemoryClaimConflict("cannot separate retained source evidence")
            for snippet in snippets:
                positions = _positions(body, snippet)
                if not positions or any(not _statement_span(body, snippet, start) for start in positions):
                    raise MemoryClaimConflict("erased evidence is not a complete source statement")
                cuts.update((start, start + len(snippet)) for start in positions)
                needles.append(snippet)
    intervals = []
    for start, end in sorted(cuts):
        if intervals and start < intervals[-1][1]:
            raise MemoryClaimConflict("overlapping erased source statements")
        intervals.append((start, end))
    return intervals, tuple(needles)


def project_retained(record, source, erased_records):
    """Return a precise evidence patch, or None; never choose deleted targets."""
    body = source.get("body")
    erased_evidence = tuple(dict.fromkeys(x for r in erased_records for x in _evidence(r)))
    evidence = _evidence(record)
    metadata = _metadata(record)
    # A full quote used to erase one topic cannot authorize removing other
    # independent facts from that same quote. Such overlap requires resolution.
    narrow = tuple(x for x in erased_evidence if isinstance(body, str) and x != body)
    objects = tuple(
        value
        for r in erased_records
        if len(value := str(r.get("memory_key") or "").partition("_")[2]) >= 2
        and any(ord(character) > 127 for character in value)
    )
    content = str(record.get("content") or "")
    quoted = metadata.get("content_semantics") == "quoted_source"
    if _contains(metadata, (*narrow, *objects)):
        raise MemoryClaimConflict("erased material remains in retained metadata")
    if not quoted and _contains(content, (*narrow, *objects)):
        raise MemoryClaimConflict("erased material remains in retained assertion")
    copied_content = quoted and ((isinstance(body, str) and body in content) or _contains(content, narrow))
    if not evidence:
        if copied_content:
            raise MemoryClaimConflict("copied retained text has no separable evidence")
        return None
    affected = copied_content or any(
        (isinstance(body, str) and body in item) or _contains(item, (*narrow, *objects)) for item in evidence
    )
    if not affected:
        if not isinstance(body, str) and (_contains(evidence, erased_evidence) or _contains(content, erased_evidence)):
            raise MemoryClaimConflict("shared source unavailable for retained evidence")
        return None
    if not isinstance(body, str) or source.get("state") != "recorded":
        raise MemoryClaimConflict("cannot separate retained source evidence")
    intervals, needles = _source_cuts(body, erased_records)
    projected = []
    spans = []
    changed = bool(copied_content)
    for item in evidence:
        if not _contains(item, needles):
            projected.append(item)
            continue
        starts = _positions(body, item)
        if len(starts) != 1:
            raise MemoryClaimConflict("retained evidence is not an exact unique source span")
        left = starts[0]
        right = left + len(item)
        local = [(a, z) for a, z in intervals if a < right and z > left]
        if any(a < left or z > right for a, z in local):
            raise MemoryClaimConflict("retained evidence clips an erased statement")
        cursor = left
        for start, end in [*local, (right, right)]:
            if cursor < start:
                fragment = body[cursor:start]
                if fragment.strip():
                    projected.append(fragment)
                    spans.append([cursor, start])
            cursor = end
        changed = True
    if not changed:
        raise MemoryClaimConflict("retained source evidence could not be projected")
    if not projected:
        raise MemoryClaimConflict("erasure would leave a retained item without evidence")
    # Evidence surgery cannot silently rewrite a retained factual assertion or
    # conceal a second copy in arbitrary metadata. Conflicts remain atomic.
    if quoted:
        content = "用户原话保留片段（仅保留内容见证据）"
    previous = metadata.get("erasure_evidence_projection", {})
    projections = list(previous.get("sources", [])) if isinstance(previous, dict) else []
    projections.append({"source_message_id": source["source_message_id"], "spans": spans})
    metadata = dict(
        metadata,
        erasure_evidence_projection={
            "version": 1,
            "kind": "original_source_fragments",
            "sources": projections,
            "complete_original_source": False,
        },
    )
    return dict(
        content=content,
        evidence_json=json.dumps(projected, ensure_ascii=False),
        metadata_json=json.dumps(metadata, ensure_ascii=False),
    )


def purge_plan(scope, erased_records):
    """Same-owner, actual linked sources; bounded binds, no model-selected IDs."""
    erased = {int(r["id"]): dict(r) for r in erased_records}
    if not erased:
        return
    sources = {}
    ids = tuple(erased)
    for offset in range(0, len(ids), 200):
        bound = {f"erase{i}": value for i, value in enumerate(ids[offset : offset + 200])}
        placeholders = ",".join(":" + key for key in bound)
        rows = yield (
            "SELECT s.source_key,s.source_message_id,s.body,s.state,l.memory_id FROM memory_sources s "
            "JOIN memory_source_links l ON l.source_key=s.source_key "
            f"WHERE s.owner_key=:owner_key AND l.memory_id IN ({placeholders})",
            dict(scope, **bound),
        )
        for row in rows:
            entry = sources.setdefault(row["source_key"], dict(source=row, erased_ids=set()))
            entry["erased_ids"].add(int(row["memory_id"]))
    patches = {}
    for key, entry in sources.items():
        rows = yield (
            "SELECT DISTINCT m.* FROM character_memories m JOIN memory_source_links l "
            "ON l.memory_id=m.id WHERE l.source_key=:source_key",
            {"source_key": key},
        )
        for row in rows:
            identity = int(row["id"])
            if identity in erased:
                continue
            current = dict(row, **patches.get(identity, {}))
            patch = project_retained(current, entry["source"], [erased[i] for i in entry["erased_ids"]])
            if patch is not None:
                patches[identity] = patch
    # Compute every projection before any update. The outer transaction rolls
    # back a later source/fence failure as well; no partial deletion receipt.
    for identity, patch in patches.items():
        yield (
            "UPDATE character_memories SET content=:content,evidence_json=:evidence_json,"
            "metadata_json=:metadata_json WHERE id=:retained_id",
            dict(patch, retained_id=identity),
        )
