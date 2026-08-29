"""Intent-routed V2 multi-scale retrieval without cross-scale top-k competition."""

from __future__ import annotations

from typing import Any

from knowledge.rag_pipeline.query import QueryAnalyzer
from knowledge.rag_pipeline.rerank import PipelineReranker
from knowledge.rag_pipeline.retrieval import HybridRetriever

from .source_text import OriginalTextExtractor

_RAW_WORDS = ("原文", "出处", "引用", "哪一段", "原句")
_BROAD_WORDS = ("整卷", "整个故事", "故事概述", "剧情概述", "主要剧情", "总体讲了", "完整回顾")


class ReprocessedMultiScaleService:
    def __init__(self, config, index, embedding_provider, *, source_extractor: OriginalTextExtractor) -> None:
        self.config = config
        self.index = index
        self.analyzer = QueryAnalyzer([config])
        self.retriever = HybridRetriever(config, index, embedding_provider)
        self.reranker = PipelineReranker(cross_encoder_enabled=False)
        self.extractor = source_extractor
        self.by_id = {doc.id: doc for doc in index.documents}
        self.evidence_by_parent = {
            str(doc.metadata.get("parent_id")): doc
            for doc in index.documents
            if doc.document_type == "evidence" and doc.metadata.get("parent_id")
        }

    def retrieve(self, query: str, *, top_k: int = 5, raw_text: bool = False) -> dict[str, Any]:
        analysis = self.analyzer.analyze(query, domain_id=self.config.domain_id)
        wants_raw = raw_text or any(word in query for word in _RAW_WORDS)
        broad = any(word in query for word in _BROAD_WORDS)
        if broad:
            allowed_types = ["story", "scene"]
        elif analysis.reality_preferences:
            # Claims, fictional layers and objective-truth questions may be
            # represented by any card schema; lexical words such as "关系"
            # must not exclude a fact card that records a character claim.
            allowed_types = ["fact", "relation", "event"]
        elif analysis.relation_type_preferences:
            # Explicit kinship/role wording has an unambiguous relation-card
            # schema.  Facts and events remain available for less explicit
            # multi-entity or relationship-evolution questions.
            allowed_types = ["relation"]
        elif analysis.doc_type_preferences:
            allowed_types = [
                document_type
                for document_type in analysis.doc_type_preferences
                if document_type in {"fact", "relation", "event"}
            ]
        else:
            allowed_types = ["fact", "relation", "event"]
        recall_k = max(top_k * 12, 60)
        recalled = self.retriever.search(
            analysis,
            top_k=recall_k,
            recall_k=recall_k,
            mode="hybrid",
            filters={"document_type": allowed_types},
        )
        ranked = self.reranker.rerank(analysis, recalled, top_k=max(top_k * 4, 20))
        selected = ranked[:top_k]

        context_blocks: list[str] = []
        citations: list[dict[str, Any]] = []
        raw_excerpt = None
        for candidate in selected:
            doc = candidate.document
            context_blocks.append(f"【{doc.document_type}】{doc.title}\n{doc.summary}")
            citations.append({"id": doc.id, **doc.source.to_dict()})
            scene_id = str(doc.metadata.get("scene_id") or "")
            scene = self.by_id.get(scene_id)
            if scene is not None:
                context_blocks.append(f"【父场景】{scene.title}\n{scene.summary}")

        if wants_raw:
            for candidate in selected:
                evidence = self.evidence_by_parent.get(candidate.document.id)
                if evidence is None:
                    continue
                try:
                    raw_excerpt = self.extractor.extract(evidence.source).to_dict()
                    raw_excerpt["evidence_id"] = evidence.id
                    raw_excerpt["parent_id"] = candidate.document.id
                    break
                except (ValueError, FileNotFoundError, UnicodeError):
                    continue

        return {
            "experiment": "multiscale_semantic_v2",
            "baseline_untouched": True,
            "routing": "broad_story_scene" if broad else "answer_card",
            "results": [candidate.to_dict() for candidate in selected],
            "context_text": "\n\n".join(context_blocks),
            "citations": citations,
            "raw_excerpt": raw_excerpt,
            "context_trust": "untrusted_retrieved_evidence",
        }
