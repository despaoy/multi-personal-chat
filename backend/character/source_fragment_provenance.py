"""Read application-owned partial-source metadata without restoring raw text."""

from collections.abc import Mapping


def source_fragment_fields(row):
    """Unknown completeness never means full; coordinates must fit retained evidence.

    This is a read annotation, not a grant to hydrate sources or infer missing
    content, the described subject, current validity, or deletion authority.
    Offsets refer to original Python string positions, never the joined quote.
    A recognized partial marker survives malformed coordinates, while those
    coordinates remain unavailable rather than being repaired or invented.
    """
    result = dict(complete_original_source=None, source_fragments=())
    metadata = row.get("metadata")
    projection = metadata.get("erasure_evidence_projection") if isinstance(metadata, Mapping) else None
    if (
        not isinstance(projection, Mapping)
        or type(projection.get("version")) is not int
        or projection["version"] != 1
        or projection.get("kind") != "original_source_fragments"
        or projection.get("complete_original_source") is not False
    ):
        return result
    result["complete_original_source"] = False
    sources = projection.get("sources")
    ids = row.get("source_message_ids")
    evidence = row.get("evidence")
    if (
        not isinstance(sources, (list, tuple))
        or not sources
        or not isinstance(ids, (list, tuple))
        or not all(isinstance(x, str) and x for x in ids)
        or not isinstance(evidence, (list, tuple))
        or not evidence
        or not all(isinstance(x, str) and x for x in evidence)
    ):
        return result
    fragments = []
    lengths = []
    seen = set()
    for source in sources:
        if not isinstance(source, Mapping):
            return result
        source_id, spans = source.get("source_message_id"), source.get("spans")
        if (
            not isinstance(source_id, str)
            or source_id not in ids
            or source_id in seen
            or not isinstance(spans, (list, tuple))
            or not spans
        ):
            return result
        seen.add(source_id)
        checked = []
        previous_end = 0
        for span in spans:
            if not isinstance(span, (list, tuple)) or len(span) != 2 or any(type(x) is not int for x in span):
                return result
            start, end = span
            if start < previous_end or end <= start:
                return result
            checked.append((start, end))
            lengths.append(end - start)
            previous_end = end
        fragments.append((source_id, tuple(checked)))
    # The producer stores retained fragments in source/span order. There is no
    # original body here to validate or reconstruct any removed character.
    if lengths != [len(x) for x in evidence]:
        return result
    result["source_fragments"] = tuple(fragments)
    return result


def source_completeness_payload(item):
    partial = item.complete_original_source is False
    return dict(
        source_fragment_position_note="原始来源的Unicode字符位置，起点包含、终点不含；同来源前片段终点小于后片段起点表示已知空缺，空缺文字不可见。",
        known_source_gaps=[
            dict(source_message_id=source_id, span=[left[1], right[0]])
            for source_id, spans in item.source_fragments
            for left, right in zip(spans, spans[1:])
            if left[1] < right[0]
        ]
        if partial
        else [],
        source_completeness="partial" if partial else "unverified",
        complete_original_source=False if partial else None,
        source_fragments=[
            dict(source_message_id=source_id, spans=[list(span) for span in spans])
            for source_id, spans in item.source_fragments
        ]
        if partial
        else [],
    )
