"""Routed retrieval over physically separated character-knowledge indexes."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from knowledge.retrieval_core.query import QueryAnalysis, QueryAnalyzer
from knowledge.retrieval_core.rerank import PipelineReranker
from knowledge.retrieval_core.retrieval import HybridRetriever, RetrievalCandidate

if TYPE_CHECKING:
    from .source_text import OriginalTextExtractor

CARD_TYPES = frozenset({"fact", "relation", "event"})
_RAW_WORDS = ("原文", "出处", "引用", "哪一段", "原句")
_BROAD_WORDS = ("整卷", "整个故事", "故事概述", "剧情概述", "主要剧情", "总体讲了", "完整回顾")
_RELATION_FOCUS_WORDS = ("什么关系", "是什么关系", "关系经历", "关系变化", "感情变成", "后来变成")
_IDENTITY_FOCUS_WORDS = ("什么人", "什么身份", "是什么人物", "个人资料")
_EVENT_FOCUS_WORDS = ("怎么回事", "发生了什么", "为何", "为什么", "袭击", "遇害", "死于")


def analyze_explicit_domain(analyzer: QueryAnalyzer, domain_id: str, query: str) -> QueryAnalysis:
    """Analyze a caller-selected domain without losing one-character entities.

    The production analyzer intentionally applies automatic-domain gating even
    when a domain is supplied.  Offline/domain-scoped retrieval already knows
    the domain, so a single-character canonical entity must still participate
    in entity recall and reranking.
    """
    analysis = analyzer.analyze(query, domain_id=domain_id)
    entities, _matched_tokens = analyzer._scan_entities(domain_id, query)
    if entities:
        analysis.entities = list(entities)
        analysis.normalized_query = analyzer._normalize_query(domain_id, query)
        if domain_id not in analysis.matched_domains:
            analysis.matched_domains.append(domain_id)
            analysis.domain_hit_reasons[domain_id] = "explicit_domain"
    return analysis


def choose_card_types(analysis: QueryAnalysis, query: str) -> frozenset[str]:
    """Choose a schema partition before recall, not after top-k truncation."""
    # “什么人/什么身份”既可能由身份事实卡回答，也可能需要家族关系卡；
    # 不能因查询中出现“家”便提前截断到 relation。
    if any(word in query for word in _IDENTITY_FOCUS_WORDS):
        return frozenset({"fact", "relation"})
    # “父亲袭击某人是怎么回事”虽然含亲属词，但核心是在询问事件。
    # 事件提示优先于 relation_type，保留事实卡用于原因或结果补充。
    if any(word in query for word in _EVENT_FOCUS_WORDS):
        return frozenset({"fact", "event"})
    if analysis.reality_preferences:
        return CARD_TYPES
    if analysis.relation_type_preferences:
        return frozenset({"relation"})
    if (
        "relation" in analysis.doc_type_preferences
        and len(analysis.entities) >= 2
        and any(word in query for word in _RELATION_FOCUS_WORDS)
    ):
        return frozenset({"relation"})
    preferred = frozenset(CARD_TYPES & set(analysis.doc_type_preferences))
    return preferred or CARD_TYPES


def rerank_with_title_frames(
    analysis: QueryAnalysis,
    candidates: list[RetrievalCandidate],
    *,
    top_k: int,
    reranker: PipelineReranker,
) -> list[RetrievalCandidate]:
    """Shared deterministic rerank plus generic quoted-title alignment."""
    ranked = reranker.rerank(analysis, candidates, top_k=max(top_k * 4, 20))
    quoted = [match.strip() for match in re.findall(r"《([^》]{1,80})》", analysis.original_query) if match.strip()]
    query_cn = "".join(re.findall(r"[\u4e00-\u9fff]", analysis.normalized_query or analysis.original_query))
    query_bigrams = {query_cn[index : index + 2] for index in range(max(0, len(query_cn) - 1))}
    for candidate in ranked:
        base = float(candidate.rerank_score or 0.0)
        text = f"{candidate.document.title} {candidate.document.summary}"
        quote_bonus = 0.24 if quoted and any(title in text for title in quoted) else 0.0
        text_cn = "".join(re.findall(r"[\u4e00-\u9fff]", text))
        text_bigrams = {text_cn[index : index + 2] for index in range(max(0, len(text_cn) - 1))}
        lexical_bonus = 0.12 * (len(query_bigrams & text_bigrams) / len(query_bigrams)) if query_bigrams else 0.0
        candidate.rerank_score = round(base + quote_bonus + lexical_bonus, 4)
    ranked.sort(key=lambda candidate: (-(candidate.rerank_score or 0.0), -candidate.fused_score, candidate.row))
    return ranked[:top_k]


class RoutedMultiScaleService:
    """Route first, retrieve second; fine and coarse scales never compete."""

    def __init__(
        self,
        config,
        indexes: dict[frozenset[str], Any],
        embedding_provider,
        *,
        all_documents: list,
        source_extractor: OriginalTextExtractor | None = None,
    ) -> None:
        self.config = config
        self.indexes = indexes
        self.analyzer = QueryAnalyzer([config])
        self.retrievers = {key: HybridRetriever(config, index, embedding_provider) for key, index in indexes.items()}
        self.reranker = PipelineReranker(cross_encoder_enabled=False)
        self.extractor = source_extractor
        self.by_id = {doc.id: doc for doc in all_documents}
        self.evidence_by_parent = {
            str(doc.metadata.get("parent_id")): doc
            for doc in all_documents
            if doc.document_type == "evidence" and doc.metadata.get("parent_id")
        }

    def retrieve(self, query: str, *, top_k: int = 5, raw_text: bool = False) -> dict[str, Any]:
        analysis = analyze_explicit_domain(self.analyzer, self.config.domain_id, query)
        broad = any(word in query for word in _BROAD_WORDS)
        route = frozenset({"story", "scene"}) if broad else choose_card_types(analysis, query)
        retriever = self.retrievers.get(route) or self.retrievers[CARD_TYPES]
        recall_k = max(top_k * 12, 60)
        recalled = retriever.search(analysis, top_k=recall_k, recall_k=recall_k, mode="hybrid")
        selected = rerank_with_title_frames(
            analysis,
            recalled,
            top_k=top_k,
            reranker=self.reranker,
        )

        context_blocks: list[str] = []
        citations: list[dict[str, Any]] = []
        for candidate in selected:
            doc = candidate.document
            context_blocks.append(f"【{doc.document_type}】{doc.title}\n{doc.summary}")
            citations.append({"id": doc.id, **doc.source.to_dict()})
            scene = self.by_id.get(str(doc.metadata.get("scene_id") or ""))
            if scene is not None:
                context_blocks.append(f"【父场景】{scene.title}\n{scene.summary}")

        timeline: list[dict[str, Any]] = []
        if route == frozenset({"relation"}) and len(analysis.entities) >= 2:
            wanted = set(analysis.entities)
            relation_docs = [doc for doc in self.indexes[route].documents if wanted <= set(doc.entities)]
            relation_docs.sort(
                key=lambda doc: (
                    int(doc.metadata.get("volume_number") or 0),
                    doc.source.line_start or 0,
                    doc.id,
                )
            )
            timeline = [
                {
                    "id": doc.id,
                    "volume": doc.metadata.get("volume_number"),
                    "summary": doc.summary,
                    "reality_status": doc.reality_status,
                }
                for doc in relation_docs
            ]

        wants_raw = raw_text or any(word in query for word in _RAW_WORDS)
        raw_excerpt = None
        if wants_raw and self.extractor is not None:
            for candidate in selected:
                evidence = self.evidence_by_parent.get(candidate.document.id)
                if evidence is None:
                    continue
                try:
                    raw_excerpt = self.extractor.extract(evidence.source).to_dict()
                    raw_excerpt.update({"evidence_id": evidence.id, "parent_id": candidate.document.id})
                    break
                except (ValueError, FileNotFoundError, UnicodeError):
                    continue

        return {
            "retrieval_strategy": "multi_scale_character",
            "route_types": sorted(route),
            "results": [candidate.to_dict() for candidate in selected],
            "relation_timeline": timeline,
            "context_text": "\n\n".join(context_blocks),
            "citations": citations,
            "raw_excerpt": raw_excerpt,
            "context_trust": "untrusted_retrieved_evidence",
        }
