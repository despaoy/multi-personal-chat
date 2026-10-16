"""Retrieve fine and coarse character sources in one shared candidate pool."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any

from infra.environment import read_bool
from knowledge.entity_scope import explicit_identity_subject, identity_evidence_subject, in_identity_scope
from knowledge.evidence_packet import render_card_evidence
from knowledge.query_tasks import requests_source_text
from knowledge.relation_scope import explicit_relation_pair, in_relation_scope, relation_endpoints
from knowledge.retrieval_core.query import QueryAnalysis, QueryAnalyzer
from knowledge.retrieval_core.rerank import PipelineReranker
from knowledge.retrieval_core.retrieval import HybridRetriever, RetrievalCandidate

from .visibility import KnowledgeBoundary, visible_to

if TYPE_CHECKING:
    from .source_text import OriginalTextExtractor

RETRIEVAL_TYPES = frozenset({"fact", "relation", "event", "scene", "story"})


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


def rerank_with_title_frames(
    analysis: QueryAnalysis,
    candidates: list[RetrievalCandidate],
    *,
    top_k: int,
    reranker: PipelineReranker,
    identity_coverage: bool = False,
) -> list[RetrievalCandidate]:
    """Preserve model ordering; literal title matches belong to deterministic mode."""
    subject = identity_evidence_subject(analysis) if identity_coverage and top_k >= 2 else ""
    quoted = [match.strip() for match in re.findall(r"《([^》]{1,80})》", analysis.original_query) if match.strip()]
    # Scope alignment must see recalled candidates before intermediate top-k
    # drops them. The deterministic reranker already scores the same pool.
    ranked = reranker.rerank(
        analysis, candidates, top_k=max(top_k * 4, 20, len(candidates) if subject or quoted else 0)
    )
    if ranked and all(candidate.rerank_method == "cross_encoder" for candidate in ranked):
        return select_identity_coverage(ranked, top_k, subject)
    query_cn = "".join(re.findall(r"[\u4e00-\u9fff]", analysis.normalized_query or analysis.original_query))
    query_bigrams = {query_cn[index : index + 2] for index in range(max(0, len(query_cn) - 1))}
    for candidate in ranked:
        base = float(candidate.rerank_score or 0.0)
        text = f"{candidate.document.title} {candidate.document.summary}"
        # Card titles name events, not necessarily their containing story.
        # Use the indexed story title too; keep this a preference so explicit
        # comparisons and retrospective evidence are not hard-filtered out.
        story_title = candidate.document.metadata.get("story_title")
        title_scope = text + "\n" + (story_title if isinstance(story_title, str) else "")
        quote_bonus = 0.24 if quoted and any(title in title_scope for title in quoted) else 0.0
        text_cn = "".join(re.findall(r"[\u4e00-\u9fff]", text))
        text_bigrams = {text_cn[index : index + 2] for index in range(max(0, len(text_cn) - 1))}
        lexical_bonus = 0.12 * (len(query_bigrams & text_bigrams) / len(query_bigrams)) if query_bigrams else 0.0
        candidate.rerank_score = round(base + quote_bonus + lexical_bonus, 4)
    ranked.sort(key=lambda candidate: (-(candidate.rerank_score or 0.0), -candidate.fused_score, candidate.row))
    return select_identity_coverage(ranked, top_k, subject)


def select_identity_coverage(ranked, top_k: int, subject: str):
    """Experimental type coverage, not a factual or temporal validity verdict.

    Keep the best overall evidence and reserve at most one existing slot for a
    directly owned relation. Do not synthesize ownership from text mentions or
    fetch hidden/out-of-scope cards. Callers must filter visibility beforehand.
    """
    selected = ranked[:top_k]
    if not subject or top_k < 2 or not selected:
        return selected
    eligible = [c for c in ranked if subject in relation_endpoints(c.document)]
    if eligible and not any(subject in relation_endpoints(c.document) for c in selected):
        selected = [*selected[:-1], eligible[0]]
    return selected


class RoutedMultiScaleService:
    """Recall all source scales before visibility, ownership and ranking."""

    context_max_chars = 6000

    def __init__(
        self,
        config,
        index: Any,
        embedding_provider,
        *,
        all_documents: list,
        source_extractor: OriginalTextExtractor | None = None,
        reranker: PipelineReranker | None = None,
        identity_coverage: bool = False,
        context_max_chars: int | None = 6000,
    ) -> None:
        if context_max_chars is not None and (type(context_max_chars) is not int or context_max_chars < 1):
            raise ValueError("context_max_chars must be a positive integer or None")
        self.context_max_chars = context_max_chars
        self.config = config
        self.identity_coverage = identity_coverage
        self.analyzer = QueryAnalyzer([config])
        self.retriever = HybridRetriever(config, index, embedding_provider)
        # Independent opt-in: the generic KB switch must not silently change
        # character retrieval. The first real dev ablation regressed on raw text.
        enabled = read_bool(os.environ, "CHARACTER_RAG_RERANKER_ENABLED")
        self.reranker = reranker or PipelineReranker(
            cross_encoder_enabled=enabled,
            text_view=os.getenv("CHARACTER_RAG_RERANK_TEXT_VIEW", "content"),
        )
        self.extractor = source_extractor
        self.by_id = {doc.id: doc for doc in all_documents}
        self.evidence_by_parent = {
            str(doc.metadata.get("parent_id")): doc
            for doc in all_documents
            if doc.document_type == "evidence" and doc.metadata.get("parent_id")
        }

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        raw_text: bool = False,
        knowledge_boundary: KnowledgeBoundary | None = None,
        object_names: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        analysis = analyze_explicit_domain(self.analyzer, self.config.domain_id, query)
        if object_names is not None:
            if not isinstance(object_names, tuple) or not object_names or any(not isinstance(name, str) or name not in query or self.config.canonical_entity(name) is None and name not in self.config.story_titles for name in object_names):
                raise ValueError("Curated scope requires literal registered whole question objects")
            analysis.entities = list(dict.fromkeys(self.config.canonical_entity(name) for name in object_names if self.config.canonical_entity(name) is not None))
            analysis.story_hits = [name for name in object_names if name in self.config.story_titles]
        retriever = self.retriever
        recall_k = max(top_k * 12, 60)
        if knowledge_boundary is not None:
            # Until visibility-aware ANN indexes are built, recall the complete
            # route so hidden documents cannot exhaust the pre-filter top-k.
            # This opt-in correctness path is deliberately not a latency claim.
            recall_k = max(recall_k, retriever.index.count())
        recalled = retriever.search(analysis, top_k=recall_k, recall_k=recall_k, mode="hybrid")
        recalled = [candidate for candidate in recalled if visible_to(candidate.document, knowledge_boundary)]
        pair = explicit_relation_pair(analysis)
        before_scope = len(recalled)
        recalled = [candidate for candidate in recalled if in_relation_scope(candidate.document, pair)]
        identity_subject = identity_evidence_subject(analysis)
        before_identity_scope = len(recalled)
        recalled = [candidate for candidate in recalled if in_identity_scope(candidate.document, identity_subject)]
        selected = rerank_with_title_frames(
            analysis,
            recalled,
            top_k=top_k,
            reranker=self.reranker,
            identity_coverage=self.identity_coverage,
        )

        context_blocks: list[str] = []
        background_blocks: list[str] = []
        background_support: dict[str, list[str]] = {}
        block_ids: dict[str, list[str]] = {}
        citations: list[dict[str, Any]] = []
        emitted_scenes: set[str] = set()
        for candidate in selected:
            doc = candidate.document
            evidence = self.evidence_by_parent.get(doc.id)
            evidence_text = getattr(evidence, "content", "") if evidence is not None else ""
            if evidence_text and visible_to(evidence, knowledge_boundary):
                # Stored claim-specific evidence is already in the index. Do
                # not replace it with the beginning of a much broader scene.
                # This is not a fresh exact-source verification or quotation.
                context_blocks.append(
                    render_card_evidence(
                        {
                            "title": doc.title,
                            "metadata": doc.metadata,
                            "source": doc.source.to_dict(),
                            **{
                                key: getattr(doc, key, "")
                                for key in ("reality_status", "temporal_scope", "content_scope")
                            },
                        },
                        evidence_text,
                    )
                )
            else:
                context_blocks.append(f"【{doc.document_type}摘要】{doc.title}\n{doc.summary}")
            block_ids.setdefault(context_blocks[-1], []).append(doc.id)
            citations.append({"id": doc.id, **doc.source.to_dict()})
            scene = self.by_id.get(str(doc.metadata.get("scene_id") or ""))
            if scene is not None and visible_to(scene, knowledge_boundary):
                block = f"【父场景】{scene.title}\n{scene.summary}"
                background_support.setdefault(block, []).append(doc.id)
                if scene.id not in emitted_scenes:
                    background_blocks.append(block)
                    emitted_scenes.add(scene.id)

        timeline: list[dict[str, Any]] = []
        if (
            len(analysis.entities) >= 2
            and any(c.document.document_type == "relation" for c in selected)
        ):
            wanted = set(analysis.entities)
            relation_docs = [
                doc
                for doc in retriever.index.documents
                if doc.document_type == "relation"
                and wanted <= set(doc.entities)
                and visible_to(doc, knowledge_boundary)
                and in_relation_scope(doc, pair)
            ]
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

        from knowledge.curated_sources import collect_curated_sources, curated_reference

        source_catalog = collect_curated_sources(selected, self.evidence_by_parent, self.extractor, knowledge_boundary)
        sources_by_id = {source["source_id"]: source for source in source_catalog["sources"]}
        wants_raw = raw_text or requests_source_text(query)
        raw_excerpt = None
        raw_status = "not_requested"
        if wants_raw:
            raw_status = "no_visible_source"
            for candidate in selected:
                evidence = curated_reference(candidate.document, self.evidence_by_parent)
                if evidence is None or not visible_to(evidence, knowledge_boundary):
                    continue
                if self.extractor is None:
                    raise RuntimeError("Original source extraction is required; configure the source extractor")
                raw_excerpt = dict(sources_by_id[evidence.id]["excerpt"])
                raw_excerpt.update({"evidence_id": evidence.id, "parent_id": candidate.document.id})
                raw_status = "truncated" if raw_excerpt.get("truncated") else "available"
                break

        if raw_excerpt and raw_excerpt.get("text"):
            # Put exact source first so a long background cannot crowd it out
            # of the generation evidence budget. It remains untrusted data.
            context_blocks.insert(
                0,
                (
                    f"【原文摘录】{raw_excerpt['source_path']} "
                    f"L{raw_excerpt['line_start']}-{raw_excerpt['line_end']}\n{raw_excerpt['text']}"
                ),
            )
            block_ids.setdefault(context_blocks[0], []).append(raw_excerpt["parent_id"])

        # Use the instance's generation evidence budget. Admit complete packets,
        # then background, so a late negation/qualification is never clipped.
        admitted: list[str] = []
        used_chars = 0
        skipped_blocks = 0
        admitted_ids: set[str] = set()
        for block in [*context_blocks, *background_blocks]:
            size = len(block) + (2 if admitted else 0)
            supported_background = bool(admitted_ids.intersection(background_support.get(block, ())))
            over_budget = self.context_max_chars is not None and used_chars + size > self.context_max_chars
            if over_budget or (block not in block_ids and not supported_background):
                skipped_blocks += 1
                continue
            admitted.append(block)
            admitted_ids.update(block_ids.get(block, ()))
            used_chars += size

        return {
            "curated_source_catalog": source_catalog,
            "retrieval_strategy": "multi_scale_character",
            "identity_coverage": self.identity_coverage,
            "candidate_types": sorted(RETRIEVAL_TYPES),
            "relation_scope": {"pair": sorted(pair), "excluded_candidates": before_scope - before_identity_scope},
            "identity_scope": {
                "subject": identity_subject,
                "excluded_candidates": before_identity_scope - len(recalled),
            },
            # Bind the dependency decision to the actual retrieval input, not
            # an entity label that might have been resolved from a follow-up.
            "identity_task": {"query": query, "subject": identity_subject}
            if explicit_identity_subject(analysis)
            else {},
            "identity_subtask": (
                {"query": query, "subject": identity_subject}
                if identity_subject and not explicit_identity_subject(analysis)
                else {}
            ),
            "results": [candidate.to_dict() for candidate in selected],
            "relation_timeline": timeline,
            "context_text": "\n\n".join(admitted),
            "evidence_packets": [
                {
                    "text": block,
                    "document_ids": block_ids.get(block, []),
                    "supporting_document_ids": background_support.get(block, []),
                    "kind": "evidence" if block in block_ids else "background",
                }
                for block in admitted
            ],
            "context_budget": {
                "used_chars": used_chars,
                "skipped_blocks": skipped_blocks,
                "max_chars": self.context_max_chars,
                "admitted_ids": sorted(admitted_ids),
            },
            "citations": [item for item in citations if item["id"] in admitted_ids],
            "raw_excerpt": raw_excerpt,
            "raw_source_status": raw_status,
            "context_trust": "untrusted_retrieved_evidence",
            "knowledge_boundary_applied": knowledge_boundary is not None,
            "rerank_text_view": getattr(self.reranker, "text_view", "content"),
        }
