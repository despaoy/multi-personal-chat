"""Focused tests for production multi-scale character knowledge retrieval."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

from knowledge.multiscale_rag.runtime import MultiScaleRagRuntime
from knowledge.multiscale_rag.service import analyze_explicit_domain, choose_card_types
from knowledge.multiscale_rag.source_text import OriginalTextExtractor
from knowledge.retrieval_core.query import QueryAnalyzer

if TYPE_CHECKING:
    from pathlib import Path


class FakeQueryEmbeddingProvider:
    dimension = 384
    model_id = "test-384"

    def __init__(self) -> None:
        self._model = object()

    def embed_query(self, query: str) -> np.ndarray:
        digest = hashlib.sha256(query.encode("utf-8")).digest()
        raw = np.resize(np.frombuffer(digest, dtype=np.uint8).astype(np.float32), self.dimension)
        norm = np.linalg.norm(raw)
        return raw / norm if norm else raw


def test_runtime_loads_all_four_scales_and_returns_bundle():
    runtime = MultiScaleRagRuntime()
    runtime._provider = FakeQueryEmbeddingProvider()

    assert runtime.is_available() is True
    stats = runtime.stats()
    assert stats["documents"] == 1098
    assert stats["document_type_counts"] == {
        "fact": 218,
        "relation": 32,
        "event": 159,
        "story": 18,
        "scene": 262,
        "evidence": 409,
    }

    bundle = runtime.retrieve_with_citations("妃和琉璃是什么关系？", top_k=3)

    assert bundle is not None
    assert bundle["results"]
    assert bundle["domains"] == ["tsukiyashiro_kisaki"]
    assert bundle["query_analysis"]["route_types"] == ["relation"]
    assert bundle["context_trust"] == "untrusted_retrieved_evidence"
    assert isinstance(bundle["abstained"], bool)


def test_runtime_keeps_unrelated_chat_out_of_game_domain():
    runtime = MultiScaleRagRuntime()
    runtime._provider = FakeQueryEmbeddingProvider()

    assert runtime.retrieve_with_citations("今天天气怎么样", top_k=3) is None


def test_runtime_does_not_ignore_generic_kb_filter():
    runtime = MultiScaleRagRuntime()
    runtime._provider = FakeQueryEmbeddingProvider()

    assert (
        runtime.retrieve_with_citations(
            "妃和琉璃是什么关系？",
            top_k=3,
            filters={"knowledge_base_id": 7},
        )
        is None
    )


def test_event_intent_overrides_incidental_relation_word():
    runtime = MultiScaleRagRuntime()
    analyzer = QueryAnalyzer([runtime.config])
    query = "父亲袭击妃是怎么回事"
    analysis = analyze_explicit_domain(analyzer, runtime.config.domain_id, query)

    assert choose_card_types(analysis, query) == frozenset({"fact", "event"})


def test_source_extractor_rejects_path_traversal(tmp_path: Path):
    extractor = OriginalTextExtractor(tmp_path)
    with pytest.raises(ValueError, match="超出语料根目录"):
        extractor.resolve_path("../outside.txt")
