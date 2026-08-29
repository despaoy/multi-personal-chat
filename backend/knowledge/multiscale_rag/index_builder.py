"""Build scale-specific semantic text for the character knowledge index."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from knowledge.retrieval_core.documents import KnowledgeIndexDocument

from .constants import EMBEDDING_TEXT_VERSION
from .hierarchy_builder import HierarchyDocumentBuilder
from .source_text import OriginalTextExtractor

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

_CAUSAL_MARKERS = ("因为", "由于", "为了", "为反抗", "为抵抗", "因此", "所以", "导致", "以此")
_SETTING_MARKERS = ("设定", "安排", "职责", "结局", "真相", "意味着", "那就是")
_FLASHBACK_MARKERS = ("儿时", "小时候", "童年", "当年")


def _clean(text: str) -> str:
    return " ".join((text or "").replace("\r", "").split())


def _clip(text: str, limit: int) -> str:
    text = _clean(text)
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _card_vector_text(doc: KnowledgeIndexDocument) -> str:
    meta = doc.metadata
    story = str(meta.get("story_title") or "")
    prefix = f"标题：{doc.title}。摘要：{doc.summary}。"
    if doc.document_type == "fact":
        subject = str(meta.get("subject") or "")
        predicate = str(meta.get("predicate") or doc.title)
        value = str(meta.get("value") or doc.summary)
        core = f"事实：{subject}的{predicate}是{value}。{subject}是什么：{value}。"
        if predicate == "设定":
            core += f"关于{subject}的故事设定、安排或结局：{value}。"
        elif predicate == "死因":
            death_aliases = "上吊、自杀、自缢" if "自缢" in value else "死亡方式、怎么死、死因"
            core += f"{subject}如何死亡，{subject}的死亡方式：{value}。相关说法：{death_aliases}。"
    elif doc.document_type == "relation":
        subject = str(meta.get("subject") or "")
        relation = str(meta.get("relation") or "关系")
        target = str(meta.get("target") or "")
        core = f"人物关系：{target}是{subject}的{relation}。{subject}和{target}的关系是{relation}。"
    else:
        participants = "、".join(str(x) for x in (meta.get("participants") or doc.entities))
        causes = "；".join(str(x) for x in (meta.get("causes") or []))
        outcomes = "；".join(str(x) for x in (meta.get("outcomes") or []))
        core = f"剧情事件：{doc.title}。参与人物：{participants}。{doc.summary}。"
        if causes:
            core += f"原因：{causes}。"
        if outcomes:
            core += f"结果：{outcomes}。"
    evidence = _card_evidence(doc)
    markers: tuple[str, ...] = ()
    if doc.document_type == "event":
        markers = _CAUSAL_MARKERS
    elif doc.document_type == "fact" and str(meta.get("predicate") or "") == "设定":
        markers = _SETTING_MARKERS
    elif doc.document_type == "fact" and str(meta.get("predicate") or "") == "经历":
        markers = _CAUSAL_MARKERS
    semantic_evidence = _select_evidence_lines(evidence, markers, limit=105) if markers else ""
    evidence_part = f"关键语义证据：{semantic_evidence}。" if semantic_evidence else ""
    suffix = f"故事范围：{story}。叙事状态：{doc.reality_status}，时间：{doc.temporal_scope}。"
    return _clip(prefix + core + evidence_part + suffix, 340)


def _card_evidence(doc: KnowledgeIndexDocument) -> str:
    marker = "\n证据："
    return doc.content.split(marker, 1)[1].strip() if marker in doc.content else ""


def _select_evidence_lines(evidence: str, markers: tuple[str, ...], *, limit: int) -> str:
    if not evidence:
        return ""
    lines = [_clean(line) for line in evidence.splitlines() if _clean(line)]
    selected = [line for line in lines if any(marker in line for marker in markers)]
    return _clip("；".join(selected[:3]), limit) if selected else ""


def _evidence_vector_text(doc: KnowledgeIndexDocument, parent: KnowledgeIndexDocument | None) -> str:
    if parent is None:
        return _clip(f"原文证据：{doc.summary}。{doc.content}", 300)
    return _clip(f"查询原文、出处、引用。{parent.title}。{parent.summary}。原文：{doc.content}", 340)


def _scene_vector_text(doc: KnowledgeIndexDocument, children: list[KnowledgeIndexDocument]) -> str:
    characters = "、".join(doc.entities)
    digest = "；".join(child.summary for child in children)
    if not digest:
        digest = _clip(doc.content, 260)
    return _clip(
        f"场景：{doc.title}。所属故事：{doc.metadata.get('story_title', '')}。"
        f"出场人物：{characters}。场景剧情：{digest}。",
        380,
    )


def _story_vector_text(doc: KnowledgeIndexDocument, children: list[KnowledgeIndexDocument]) -> str:
    facts = [child.summary for child in children if child.document_type == "fact"]
    relations = [child.summary for child in children if child.document_type == "relation"]
    events = [child.summary for child in children if child.document_type == "event"]
    entities = "、".join(_ordered_unique(entity for child in children for entity in child.entities))
    return _clip(
        f"完整故事与卷概述：{doc.title}。主要人物：{entities}。"
        f"关键事件：{'；'.join(events)}。人物关系：{'；'.join(relations)}。关键事实：{'；'.join(facts)}。",
        440,
    )


@dataclass(frozen=True)
class CharacterKnowledgeBuildResult:
    documents: tuple[KnowledgeIndexDocument, ...]
    counts: dict[str, int]
    exact_evidence_matches: int
    parented_cards: int


class CharacterKnowledgeIndexBuilder:
    """Build hierarchy first, then create scale-specific semantic texts."""

    def __init__(self, *, domain_id: str, index_version: str, aliases: dict[str, str], corpus_root: Path) -> None:
        self.domain_id = domain_id
        self.index_version = index_version
        self.extractor = OriginalTextExtractor(corpus_root)
        self.hierarchy_builder = HierarchyDocumentBuilder(
            domain_id=domain_id,
            index_version=index_version,
            aliases=aliases,
            source_extractor=self.extractor,
        )

    def build(self, approved_root: Path, enriched_scenes_path: Path) -> CharacterKnowledgeBuildResult:
        base = self.hierarchy_builder.build(approved_root, enriched_scenes_path)
        by_id = {doc.id: doc for doc in base.documents}
        cards = [doc for doc in base.documents if doc.document_type in {"fact", "relation", "event"}]
        cards_by_scene: dict[str, list[KnowledgeIndexDocument]] = defaultdict(list)
        cards_by_story: dict[str, list[KnowledgeIndexDocument]] = defaultdict(list)
        for card in cards:
            cards_by_scene[str(card.metadata.get("scene_id") or "")].append(card)
            cards_by_story[str(card.metadata.get("story_unit_id") or "")].append(card)

        processed: list[KnowledgeIndexDocument] = []
        for document in base.documents:
            data = document.to_dict()
            scale = str(document.metadata.get("scale") or document.document_type)
            if scale == "card":
                vector_text = _card_vector_text(document)
                profile = f"{document.document_type}_query_aligned"
            elif scale == "evidence":
                parent = by_id.get(str(document.metadata.get("parent_id") or ""))
                vector_text = _evidence_vector_text(document, parent)
                profile = "source_request_only"
            elif scale == "scene":
                vector_text = _scene_vector_text(document, cards_by_scene.get(document.id, []))
                profile = "scene_card_digest"
            else:
                unit_id = str(document.metadata.get("story_unit_id") or "")
                vector_text = _story_vector_text(document, cards_by_story.get(unit_id, []))
                profile = "story_card_digest"
            data["embedding_text"] = vector_text
            semantic_temporal = document.temporal_scope
            if scale == "card" and any(
                marker in f"{document.title} {document.summary} {document.metadata.get('relation', '')}"
                for marker in _FLASHBACK_MARKERS
            ):
                semantic_temporal = "flashback"
                data["temporal_scope"] = semantic_temporal
            data["metadata"] = {
                **data["metadata"],
                "embedding_text_version": EMBEDDING_TEXT_VERSION,
                "embedding_profile": profile,
                "embedding_text_chars": len(vector_text),
                "source_temporal_scope": document.temporal_scope,
                "semantic_temporal_scope": semantic_temporal,
            }
            processed.append(KnowledgeIndexDocument.from_dict(data))

        return CharacterKnowledgeBuildResult(
            documents=tuple(processed),
            counts=dict(base.counts),
            exact_evidence_matches=base.exact_evidence_matches,
            parented_cards=sum(bool(card.metadata.get("scene_id")) for card in cards),
        )
