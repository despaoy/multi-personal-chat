"""Explicit character-knowledge boundaries, independent of relevance scores.

Curators annotate knowledge_access on documents. Entity occurrence is never
treated as proof that a character knows the fact. Unannotated, unreviewed or
malformed documents are invisible when a boundary is requested.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowledge.retrieval_core.documents import KnowledgeIndexDocument


@dataclass(frozen=True)
class KnowledgeBoundary:
    character_id: str
    scene_position: int
    continuity_id: str = "canonical"

    def __post_init__(self):
        if not isinstance(self.character_id, str) or not self.character_id.strip():
            raise ValueError("character_id must be nonempty")
        if isinstance(self.scene_position, bool) or not isinstance(self.scene_position, int) or self.scene_position < 0:
            raise ValueError("scene_position must be a nonnegative integer")
        if not isinstance(self.continuity_id, str) or not self.continuity_id.strip():
            raise ValueError("continuity_id must be nonempty")


def visible_to(document: KnowledgeIndexDocument, boundary: KnowledgeBoundary | None) -> bool:
    if boundary is None:
        return True
    if document.review_status != "approved":
        return False
    access = document.metadata.get("knowledge_access")
    if not isinstance(access, Mapping) or type(access.get("version")) is not int or access["version"] != 1:
        return False
    if access.get("continuity_id") != boundary.continuity_id:
        return False
    known_by = access.get("known_by")
    if not isinstance(known_by, Mapping):
        return False
    window = known_by.get(boundary.character_id)
    if not isinstance(window, Mapping):
        return False
    start, end = window.get("from_scene"), window.get("until_scene")
    if type(start) is not int or start < 0:
        return False
    if end is not None and (type(end) is not int or end <= start):
        return False
    return start <= boundary.scene_position and (end is None or boundary.scene_position < end)
