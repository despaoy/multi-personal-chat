"""Parent-aware retrieval service for the isolated multi-scale index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from knowledge.rag_pipeline.index import DomainIndex
from knowledge.rag_pipeline.query import QueryAnalysis, QueryAnalyzer
from knowledge.rag_pipeline.registry import KnowledgeDomainConfig
from knowledge.rag_pipeline.rerank import PipelineReranker
from knowledge.rag_pipeline.retrieval import HybridRetriever, RetrievalCandidate

from .source_text import OriginalTextExtractor


@dataclass(frozen=True)
class _ScoredCandidate:
    candidate: RetrievalCandidate
    experiment_score: float
    reasons: tuple[str, ...]


class MultiScaleRagService:
    """Explicit-only experimental service; it never replaces the P6 singleton."""

    def __init__(
        self,
        config: KnowledgeDomainConfig,
        index: DomainIndex,
        embedding_provider: Any,
        *,
        source_extractor: OriginalTextExtractor | None = None,
        reranker: PipelineReranker | None = None,
        context_budget_chars: int = 2400,
    ) -> None:
        self.config = config
        self.index = index
        self.retriever = HybridRetriever(config, index, embedding_provider)
        self.query_analyzer = QueryAnalyzer([config])
        self.reranker = reranker or PipelineReranker(cross_encoder_enabled=False)
        self.source_extractor = source_extractor
        self.context_budget_chars = max(400, int(context_budget_chars))
        self._documents_by_id = {doc.id: doc for doc in index.documents}

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        raw_text: bool = False,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        analysis = self.query_analyzer.analyze(query, domain_id=self.config.domain_id)
        recall_k = max(top_k * 8, 40)
        recalled = self.retriever.search(
            analysis,
            top_k=recall_k,
            recall_k=recall_k,
            filters=filters,
            mode="hybrid",
        )
        reranked = self.reranker.rerank(analysis, recalled, top_k=recall_k)
        scored = self._parent_aware_scores(analysis, reranked)
        selected = scored[: max(1, top_k)]
        context_text, citations = self._build_context(selected)

        raw_excerpt: dict[str, object] | None = None
        raw_reason = ""
        if raw_text:
            # Raw extraction may use a relevant evidence candidate just beyond
            # the displayed top-k; it does not change the visible ranking.
            raw_excerpt, raw_reason = self._extract_raw(scored)

        return {
            "experiment": "multiscale_parent_child_v1",
            "baseline_untouched": True,
            "query": query,
            "query_analysis": {
                "entities": list(analysis.entities),
                "doc_type_preferences": list(analysis.doc_type_preferences),
                "story_hits": list(analysis.story_hits),
            },
            "results": [self._result_item(item, rank + 1) for rank, item in enumerate(selected)],
            "context_text": context_text,
            "context_trust": "untrusted_retrieved_evidence",
            "citations": citations,
            "raw_excerpt": raw_excerpt,
            "raw_reason": raw_reason,
        }

    def _parent_aware_scores(
        self,
        analysis: QueryAnalysis,
        candidates: Sequence[RetrievalCandidate],
    ) -> list[_ScoredCandidate]:
        candidate_ids = {candidate.document.id for candidate in candidates}
        preferred = set(analysis.doc_type_preferences)
        broad_query = not preferred
        wants_source = any(word in analysis.original_query for word in ("原文", "出处", "引用", "哪一段"))
        scored: list[_ScoredCandidate] = []
        for candidate in candidates:
            doc = candidate.document
            scale = str(doc.metadata.get("scale") or doc.document_type)
            base = (
                float(candidate.rerank_score)
                if candidate.rerank_score is not None
                else min(1.0, candidate.fused_score * 8.0)
            )
            reasons: list[str] = ["base_rerank"]
            bonus = 0.0
            if doc.document_type in preferred:
                bonus += 0.16
                reasons.append("intent_type_match")
            if scale == "evidence":
                bonus += 0.07
                reasons.append("source_evidence")
                if wants_source:
                    bonus += 0.18
                    reasons.append("raw_source_intent")
            elif broad_query and scale == "scene":
                bonus += 0.08
                reasons.append("broad_scene_match")
            elif broad_query and scale == "story":
                bonus += 0.03
                reasons.append("broad_story_match")

            parent_id = str(doc.metadata.get("parent_id") or "")
            if parent_id and parent_id in candidate_ids:
                bonus += 0.08
                reasons.append("parent_recalled")
            scene_id = str(doc.metadata.get("scene_id") or "")
            if scene_id and scene_id in candidate_ids:
                bonus += 0.06
                reasons.append("scene_support")
            scored.append(_ScoredCandidate(candidate, round(base + bonus, 6), tuple(reasons)))
        return sorted(
            scored,
            key=lambda item: (-item.experiment_score, -item.candidate.fused_score, item.candidate.row),
        )

    def _build_context(self, selected: Sequence[_ScoredCandidate]) -> tuple[str, list[dict[str, Any]]]:
        blocks: list[str] = []
        citations: list[dict[str, Any]] = []
        seen: set[str] = set()
        used = 0

        def append_doc(doc, label: str, score: float) -> bool:
            nonlocal used
            if doc.id in seen:
                return True
            body = doc.summary or doc.content
            if doc.document_type == "evidence":
                body = f"原文：「{doc.content[:700]}{'…' if len(doc.content) > 700 else ''}」"
            else:
                body = body[:700] + ("…" if len(body) > 700 else "")
            block = f"【{label}】{doc.title}\n{body}"
            if used + len(block) + 2 > self.context_budget_chars:
                return False
            blocks.append(block)
            used += len(block) + 2
            seen.add(doc.id)
            citations.append(
                {
                    "source_id": doc.id,
                    "document_type": doc.document_type,
                    "source_path": doc.source.source_path,
                    "line_start": doc.source.line_start,
                    "line_end": doc.source.line_end,
                    "score": round(score, 6),
                }
            )
            return True

        for item in selected:
            doc = item.candidate.document
            label = {
                "story": "故事范围",
                "scene": "父场景",
                "fact": "事实卡",
                "relation": "关系卡",
                "event": "事件卡",
                "evidence": "原文证据",
            }.get(doc.document_type, doc.document_type)
            if not append_doc(doc, label, item.experiment_score):
                break

            # A selected fine-grained document brings back only a compact parent
            # scene, not the entire scene body.
            scene_id = str(doc.metadata.get("scene_id") or "")
            parent = self._document_by_id(scene_id) if scene_id else None
            if parent is not None:
                append_doc(parent, "父场景补充", item.experiment_score - 0.01)
        return "\n\n".join(blocks), citations

    def _extract_raw(
        self,
        selected: Sequence[_ScoredCandidate],
    ) -> tuple[dict[str, object] | None, str]:
        if self.source_extractor is None:
            return None, "source_extractor_not_configured"
        preferred = sorted(
            selected,
            key=lambda item: (
                0 if item.candidate.document.document_type == "evidence" else 1,
                -item.experiment_score,
            ),
        )
        for item in preferred:
            doc = item.candidate.document
            if not doc.source.source_path or doc.source.line_start is None or doc.source.line_end is None:
                continue
            try:
                return self.source_extractor.extract(doc.source).to_dict(), "exact_source_lines"
            except (ValueError, FileNotFoundError, UnicodeError):
                continue
        return None, "no_resolvable_source"

    def _document_by_id(self, document_id: str):
        return self._documents_by_id.get(document_id)

    @staticmethod
    def _result_item(item: _ScoredCandidate, rank: int) -> dict[str, Any]:
        result = item.candidate.to_dict()
        result.update(
            {
                "retrieval_rank": rank,
                "experiment_score": item.experiment_score,
                "experiment_reasons": list(item.reasons),
                "scale": item.candidate.document.metadata.get("scale"),
                "parent_id": item.candidate.document.metadata.get("parent_id"),
                "scene_id": item.candidate.document.metadata.get("scene_id"),
            }
        )
        return result
