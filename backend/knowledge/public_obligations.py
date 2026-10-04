"""Literal partitions preserve all public query data without model-made offsets."""

from knowledge.turn_dependencies import query_segments


def validate_public_obligations(tasks, query, *, public_indices=None):
    segments = query_segments(query)
    offsets, cursor = [], 0
    for segment in segments:
        offsets.append(cursor)
        cursor += len(segment)
    if not isinstance(tasks, (list, tuple)) or not tasks or len(tasks) > 32:
        raise ValueError("Invalid public obligation count")
    grouped = {}
    for task in tasks:
        if not isinstance(task, dict) or set(task) != {"index", "query", "segment_index", "start", "end"}:
            raise ValueError("Invalid public obligation fields")
        parent, start, end = task["segment_index"], task["start"], task["end"]
        if (
            type(parent) is not int
            or not 0 <= parent < len(segments)
            or type(start) is not int
            or type(end) is not int
            or not offsets[parent] <= start < end <= offsets[parent] + len(segments[parent])
            or task["query"] != query[start:end]
            or not task["query"].strip()
        ):
            raise ValueError("Public obligation is not exact original query data")
        grouped.setdefault(parent, []).append(task)
    if public_indices is not None and set(grouped) != set(public_indices):
        raise ValueError("Public partition lost a source dependency")
    expected_order = []
    for parent in sorted(grouped):
        position = offsets[parent]
        for part, task in enumerate(grouped[parent]):
            if task["index"] != f"public:{parent}:{part}" or task["start"] != position:
                raise ValueError("Public partition has gaps, overlap or invalid identity")
            position = task["end"]
            expected_order.append(task)
        if position != offsets[parent] + len(segments[parent]):
            raise ValueError("Public partition dropped original suffix")
    if list(tasks) != expected_order:
        raise ValueError("Public obligations changed original order")
    return tuple(tasks)


def parse_public_partitions(value, dependencies, query):
    if "".join(dependencies.segments) != query:
        raise ValueError("Public partition changed query")
    public = dict(dependencies.groups)["public_knowledge"]
    if not isinstance(value, list) or len(value) != len(public):
        raise ValueError("Every public segment needs a partition")
    offsets, cursor = [], 0
    for segment in dependencies.segments:
        offsets.append(cursor)
        cursor += len(segment)
    seen, partitions = set(), {}
    for row in value:
        if not isinstance(row, dict) or set(row) != {"segment_id", "cuts"}:
            raise ValueError("Invalid public partition fields")
        parent, cuts = row["segment_id"], row["cuts"]
        if type(parent) is not int or parent not in public or parent in seen:
            raise ValueError("Invalid public partition parent")
        if not isinstance(cuts, list) or len(cuts) > 15:
            raise ValueError("Invalid public partition cuts")
        segment = dependencies.segments[parent]
        positions = [0]
        for marker in cuts:
            if (
                not isinstance(marker, str)
                or not marker.strip()
                or not 1 <= len(marker) <= 64
                or segment.count(marker) != 1
            ):
                raise ValueError("Partition cuts must be unique literal markers")
            position = segment.index(marker)
            if position <= positions[-1]:
                raise ValueError("Partition cuts must preserve original order")
            positions.append(position)
        positions.append(len(segment))
        partitions[parent] = [
            dict(
                index=f"public:{parent}:{part}",
                query=segment[start:end],
                segment_index=parent,
                start=offsets[parent] + start,
                end=offsets[parent] + end,
            )
            for part, (start, end) in enumerate(zip(positions, positions[1:]))
        ]
        seen.add(parent)
    if not public:
        return ()
    tasks = tuple(task for parent in sorted(public) for task in partitions[parent])
    return validate_public_obligations(tasks, query, public_indices=public)
