"""Complete extractive turn dependencies; source labels never grant access or facts."""

from __future__ import annotations

import re
from dataclasses import dataclass

KINDS = ("private_memory", "current_input", "public_knowledge", "control", "unresolved_source")


def query_segments(query: str) -> tuple[str, ...]:
    segments = tuple(match[0] for match in re.finditer(r"[^。！？!?；;\n]*[。！？!?；;\n]+|[^。！？!?；;\n]+$", query))
    if not segments or len(segments) > 128 or "".join(segments) != query:
        raise ValueError("Query cannot be completely represented")
    return segments


@dataclass(frozen=True)
class TurnDependencies:
    segments: tuple[str, ...]
    groups: tuple[tuple[str, tuple[int, ...]], ...]

    @property
    def private_context_only(self) -> bool:
        groups = dict(self.groups)
        return bool(groups["private_memory"]) and not groups["public_knowledge"] and not groups["unresolved_source"]


def parse_dependencies(value: object, query: str) -> TurnDependencies:
    segments = query_segments(query)
    if isinstance(value, dict) and set(value) == {
        "private_memory",
        "current_input",
        "public_knowledge",
        "control",
        "unknown",
    }:
        # Old ambiguous labels retain every unresolved index, never become a
        # private-only exemption merely because values are unknown.
        value = {("unresolved_source" if key == "unknown" else key): indices for key, indices in value.items()}
    if not isinstance(value, dict) or set(value) != set(KINDS):
        raise ValueError("Unexpected dependency groups")
    groups = []
    coverage = set()
    for kind in KINDS:
        indices = value[kind]
        if not isinstance(indices, list) or len(indices) > len(segments):
            raise ValueError("Invalid dependency index list")
        if any(type(index) is not int or not 0 <= index < len(segments) for index in indices):
            raise ValueError("Dependency references must be actual segment indices")
        if len(indices) != len(set(indices)):
            raise ValueError("Duplicate dependency index")
        coverage.update(indices)
        groups.append((kind, tuple(indices)))
    if coverage != set(range(len(segments))):
        raise ValueError("Every original segment must retain a dependency")
    # One segment may require both private and public evidence. Never force a
    # compound sentence into a single source or drop its hypothetical inputs.
    # Controls may share a complete sentence with a content task. An unknown
    # dependency must remain present even when another part is resolved; the
    # private-only predicate always vetoes any unknown or public dependency.
    return TurnDependencies(segments, tuple(groups))
