"""Retrieve chat knowledge from its selected sources and fresh original authority."""

import asyncio
from typing import Any

from db.adapter import db
from inference.provider_context import get_provider_context_budget
from knowledge import index_lifecycle


async def retrieve_chat_knowledge(
    query: str, top_k: int, filters: dict[str, Any] | None, *,
    question_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retrieve authorized knowledge with fresh source authority."""

    def retrieve() -> dict[str, Any]:
        from knowledge.source_expansion import requested_document_titles

        # An explicit named-document read owns its evidence targets. Incidental
        # character names must not replace those sources with curated lore.
        # Ordinary character questions retain the strict domain availability gate.
        use_curated = not filters and not requested_document_titles(query)
        if question_binding is not None or use_curated:
            from knowledge.multiscale_rag.runtime import get_multiscale_rag_service

            character_rag = get_multiscale_rag_service()
        if question_binding is not None:
            from knowledge.public_domains import (
                all_bound_objects_generic,
                domain_plan,
                requires_independent_domains,
                retrieve_curated_scope,
                retrieve_domains,
            )

            plan = domain_plan(question_binding, character_rag.config)
            if requires_independent_domains(plan):
                return retrieve_domains(plan, query, top_k, curated=character_rag.retrieve_with_citations, generic=retrieve_generic)
            if use_curated:
                if any(obj["authority"] == "curated_character" for obj in plan["objects"]):
                    return retrieve_curated_scope(plan, query, top_k, curated=character_rag.retrieve_with_citations)
                if all_bound_objects_generic(plan):
                    return retrieve_generic()
        if use_curated:
            # retrieve_with_citations 内部完成惰性加载和域门控。不能先用
            # is_available() 短路，否则冷启动首条请求无法触发索引加载。
            bundle = character_rag.retrieve_with_citations(query, top_k=top_k, filters=filters)
            if bundle is not None:
                return bundle

        return retrieve_generic()

    def retrieve_generic() -> dict[str, Any]:
        from knowledge.rag_helper import get_rag_helper
        from knowledge.source_expansion import expand_source_context
        from knowledge.vector_db import get_vector_db

        # A normal generation does not necessarily follow the search API.
        # Resolve dirty metadata/content before consulting cached RAG
        # lookup; an incomplete rebuild cannot authorize stale source evidence.
        if not index_lifecycle._ensure_vector_index():
            raise RuntimeError("Generic knowledge index is not ready")

        vector_db = get_vector_db()
        generation = vector_db.cache_generation
        revision = index_lifecycle._vector_index_revision
        binding_options = {"question_binding": question_binding} if question_binding is not None else {}
        bundle = get_rag_helper().retrieve_with_citations(query, top_k=top_k, filters=filters, **binding_options)
        budget = get_provider_context_budget()
        expanded = expand_source_context(
            bundle, vector_db, expected_generation=generation,
            source_budget_tokens=budget.window_tokens, filters=filters, query=query,
        )
        if index_lifecycle._get_rebuild_revision() != revision:
            raise RuntimeError("Knowledge authority changed during retrieval")
        from knowledge.original_sources import attach_original_sources

        expanded = attach_original_sources(
            expanded, db.get_knowledge_document,
            source_budget_tokens=budget.window_tokens,
            authority_revision=revision,
        )
        if index_lifecycle._get_rebuild_revision() != revision:
            raise RuntimeError("Knowledge authority changed during retrieval")
        if expanded is not bundle:
            expanded["citations"] = get_rag_helper().build_citations(expanded["results"])
        return expanded

    return await asyncio.to_thread(retrieve)
