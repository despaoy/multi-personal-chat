"""Build an isolated story -> scene -> card -> evidence document hierarchy."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from knowledge.retrieval_core.documents import KnowledgeIndexDocument, SourceReference
from knowledge.retrieval_core.loaders import AliasEntityNormalizer, ApprovedCardsLoader

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .source_text import OriginalTextExtractor


def _clip(text: str, limit: int) -> str:
    normalized = "\n".join(line.rstrip() for line in (text or "").strip().splitlines())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "…"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("review_status") != "approved":
                raise ValueError(f"{path.name}:{line_no} 未通过 approved 门禁")
            records.append(record)
    return records


@dataclass(frozen=True)
class MultiScaleBuildResult:
    documents: tuple[KnowledgeIndexDocument, ...]
    counts: dict[str, int]
    exact_evidence_matches: int


class HierarchyDocumentBuilder:
    """Create hierarchical documents without mutating approved source files."""

    def __init__(
        self,
        *,
        domain_id: str,
        index_version: str,
        aliases: dict[str, str],
        source_extractor: OriginalTextExtractor | None = None,
    ) -> None:
        self.domain_id = domain_id
        self.index_version = index_version
        self.normalizer = AliasEntityNormalizer(aliases)
        self.source_extractor = source_extractor

    def build(self, approved_root: Path, enriched_scenes_path: Path) -> MultiScaleBuildResult:
        card_loader = ApprovedCardsLoader(
            domain_id=self.domain_id,
            index_version=self.index_version,
            entity_normalizer=self.normalizer,
        )
        cards = card_loader.load(Path(approved_root))
        scenes = self._scene_documents(_read_jsonl(Path(enriched_scenes_path)))
        cards_with_parents: list[KnowledgeIndexDocument] = []
        evidence_docs: list[KnowledgeIndexDocument] = []
        exact_matches = 0
        for card in cards:
            parent = self._find_parent_scene(card, scenes)
            data = card.to_dict()
            data["metadata"] = {
                **data["metadata"],
                "scale": "card",
                "parent_id": parent.id if parent else "",
                "scene_id": parent.id if parent else "",
            }
            card_with_parent = KnowledgeIndexDocument.from_dict(data)
            cards_with_parents.append(card_with_parent)

            evidence = self._card_evidence(card)
            if not evidence:
                continue
            evidence_source = card.source
            match_status = "approved_evidence_fallback"
            if self.source_extractor is not None:
                evidence_source, match_status = self.source_extractor.tighten_to_evidence(card.source, evidence)
            exact_matches += int(match_status == "exact_source_match")
            evidence_docs.append(
                KnowledgeIndexDocument(
                    id=f"evidence:{card.id}",
                    domain_id=self.domain_id,
                    document_type="evidence",
                    title=f"原文证据：{card.title}",
                    summary=_clip(evidence, 240),
                    content=evidence,
                    embedding_text=(f"{card.title}。{card.summary}。原文证据：{_clip(evidence, 480)}"),
                    keywords=list(dict.fromkeys([*card.keywords, "原文", "证据"])),
                    entities=list(card.entities),
                    relations=list(card.relations),
                    source=evidence_source,
                    metadata={
                        **card.metadata,
                        "scale": "evidence",
                        "parent_id": card.id,
                        "scene_id": parent.id if parent else "",
                        "evidence_match_status": match_status,
                    },
                    reality_status=card.reality_status,
                    temporal_scope=card.temporal_scope,
                    content_scope=card.content_scope,
                    review_status="approved",
                    index_version=self.index_version,
                )
            )

        stories = self._story_documents(scenes)
        documents = [*stories, *scenes, *cards_with_parents, *evidence_docs]
        ids = [doc.id for doc in documents]
        if len(ids) != len(set(ids)):
            raise ValueError("多粒度文档 ID 不唯一")
        counts: dict[str, int] = defaultdict(int)
        for doc in documents:
            counts[str(doc.metadata.get("scale") or doc.document_type)] += 1
        return MultiScaleBuildResult(tuple(documents), dict(counts), exact_matches)

    def _scene_documents(self, records: Iterable[dict[str, Any]]) -> list[KnowledgeIndexDocument]:
        docs: list[KnowledgeIndexDocument] = []
        for scene in records:
            story = scene.get("story") or {}
            src = scene.get("source") or {}
            text = str(scene.get("text") or "")
            characters = list(
                dict.fromkeys(
                    str(item)
                    for key in ("present_characters", "speakers", "mentioned_characters")
                    for item in (scene.get(key) or [])
                    if item
                )
            )
            entities = self.normalizer.scan_text(" ".join([*characters, scene.get("title", ""), _clip(text, 800)]))
            story_unit_id = str(story.get("story_unit_id") or "")
            title = str(scene.get("title") or f"场景 {scene['id']}")
            docs.append(
                KnowledgeIndexDocument(
                    id=str(scene["id"]),
                    domain_id=self.domain_id,
                    document_type="scene",
                    title=title,
                    summary=_clip(text, 360),
                    content=text,
                    embedding_text=(
                        f"场景：{title}。故事：{story.get('story_title', '')}。"
                        f"人物：{'、'.join(characters)}。剧情：{_clip(text, 900)}"
                    ),
                    keywords=[title, str(story.get("story_title") or ""), *characters],
                    entities=entities,
                    relations=[],
                    source=SourceReference(
                        source_path=str(src.get("source_path") or ""),
                        line_start=src.get("line_start"),
                        line_end=src.get("line_end"),
                        extra={"story_unit_id": story_unit_id, "scene_id": str(scene["id"])},
                    ),
                    metadata={
                        "scale": "scene",
                        "parent_id": f"story:{story_unit_id}" if story_unit_id else "",
                        "story_unit_id": story_unit_id,
                        "story_title": story.get("story_title"),
                        "volume_number": story.get("volume_number"),
                        "viewpoint": story.get("viewpoint"),
                    },
                    reality_status=str(scene.get("reality_status") or "unknown"),
                    temporal_scope=str(story.get("temporal_scope") or "unknown"),
                    content_scope=str(story.get("content_scope") or "unknown"),
                    review_status="approved",
                    index_version=self.index_version,
                )
            )
        return docs

    def _story_documents(self, scenes: list[KnowledgeIndexDocument]) -> list[KnowledgeIndexDocument]:
        grouped: dict[str, list[KnowledgeIndexDocument]] = defaultdict(list)
        for scene in scenes:
            unit_id = str(scene.metadata.get("story_unit_id") or "")
            if unit_id:
                grouped[unit_id].append(scene)

        docs: list[KnowledgeIndexDocument] = []
        for unit_id, children in sorted(grouped.items()):
            children.sort(key=lambda doc: (doc.source.line_start or 0, doc.id))
            title = str(children[0].metadata.get("story_title") or unit_id)
            entities = list(dict.fromkeys(entity for child in children for entity in child.entities))
            digest = "\n".join(f"- {child.title}：{_clip(child.summary, 180)}" for child in children)
            docs.append(
                KnowledgeIndexDocument(
                    id=f"story:{unit_id}",
                    domain_id=self.domain_id,
                    document_type="story",
                    title=title,
                    summary=_clip(digest, 900),
                    content=digest,
                    embedding_text=f"故事单元：{title}。人物：{'、'.join(entities)}。内容：{_clip(digest, 1200)}",
                    keywords=[title, *entities],
                    entities=entities,
                    source=SourceReference(
                        source_path=children[0].source.source_path,
                        line_start=min(child.source.line_start or 1 for child in children),
                        line_end=max(child.source.line_end or 1 for child in children),
                        extra={"story_unit_id": unit_id},
                    ),
                    metadata={
                        "scale": "story",
                        "parent_id": "",
                        "story_unit_id": unit_id,
                        "story_title": title,
                        "child_ids": [child.id for child in children],
                    },
                    reality_status=children[0].reality_status,
                    temporal_scope=children[0].temporal_scope,
                    content_scope=children[0].content_scope,
                    review_status="approved",
                    index_version=self.index_version,
                )
            )
        return docs

    @staticmethod
    def _find_parent_scene(
        card: KnowledgeIndexDocument,
        scenes: list[KnowledgeIndexDocument],
    ) -> KnowledgeIndexDocument | None:
        start, end = card.source.line_start, card.source.line_end
        candidates = [
            scene
            for scene in scenes
            if scene.source.source_path == card.source.source_path
            and start is not None
            and end is not None
            and scene.source.line_start is not None
            and scene.source.line_end is not None
            and scene.source.line_start <= start
            and end <= scene.source.line_end
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda scene: (scene.source.line_end or 0) - (scene.source.line_start or 0))

    @staticmethod
    def _card_evidence(card: KnowledgeIndexDocument) -> str:
        marker = "\n证据："
        return card.content.split(marker, 1)[1].strip() if marker in card.content else ""
