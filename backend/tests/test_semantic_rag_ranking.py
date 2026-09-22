"""Semantic scores must retain their own ordering, scale and provenance."""

from types import SimpleNamespace

import pytest

from knowledge.multiscale_rag.runtime import MultiScaleRagRuntime
from knowledge.multiscale_rag.service import rerank_with_title_frames
from knowledge.retrieval_core.documents import KnowledgeIndexDocument
from knowledge.retrieval_core.rerank import PipelineReranker
from knowledge.retrieval_core.retrieval import RetrievalCandidate


def _candidate(key, title="unrelated"):
    doc = KnowledgeIndexDocument(key, "test", "fact", title, title, title, title)
    return RetrievalCandidate(int(key), doc, fused_score=0.9)


def _analysis():
    return SimpleNamespace(
        original_query="《标题》发生什么？",
        normalized_query="标题",
        entities=[],
        doc_type_preferences=[],
        relation_type_preferences=[],
        reality_preferences=[],
        temporal_preferences=[],
        content_scope_preferences=[],
    )


class Encoder:
    def rerank(self, query, candidates, top_k):
        return [dict(item, rerank_score=0.1 if item["id"] == "1" else 0.09) for item in candidates]


def test_semantic_order_not_overridden_by_title_bonus_and_inputs_not_mutated():
    candidates = [_candidate("1"), _candidate("2", "标题")]
    reranker = PipelineReranker(cross_encoder=Encoder())
    result = rerank_with_title_frames(_analysis(), candidates, top_k=2, reranker=reranker)
    assert [item.document.id for item in result] == ["1", "2"]
    assert [item.rerank_score for item in result] == [0.1, 0.09]
    assert all(item.rerank_method == "cross_encoder" for item in result)
    assert all(item.rerank_score is None and item.rerank_method == "none" for item in candidates)


def test_embedding_text_view_changes_only_model_input_not_returned_evidence():
    class InspectEncoder:
        def rerank(self, query, candidates, top_k):
            assert [item["content"] for item in candidates] == ["structured-1", "structured-2"]
            return [dict(item, rerank_score=2 - index) for index, item in enumerate(candidates)]

    candidates = [_candidate("1", "source-1"), _candidate("2", "source-2")]
    for index, candidate in enumerate(candidates, 1):
        candidate.document.embedding_text = f"structured-{index}"
    result = PipelineReranker(cross_encoder=InspectEncoder(), text_view="embedding_text").rerank(_analysis(), candidates, 2)
    assert [item.document.content for item in result] == ["source-1", "source-2"]


def test_invalid_reranker_view_is_not_silently_accepted():
    with pytest.raises(ValueError, match="text view"):
        PipelineReranker(text_view="invented")


@pytest.mark.parametrize("kind", ["nan", "missing", "duplicate", "unknown", "unscored"])
def test_invalid_encoder_results_use_explicit_deterministic_fallback(kind):
    class Broken:
        def rerank(self, query, candidates, top_k):
            values = [dict(item, rerank_score=0.4) for item in candidates]
            if kind == "nan":
                values[0]["rerank_score"] = float("nan")
            elif kind == "missing":
                values.pop()
            elif kind == "duplicate":
                values[1] = values[0]
            elif kind == "unknown":
                values[0]["id"] = "unseen"
            else:
                values[0].pop("rerank_score")
            return values

    reranker = PipelineReranker(cross_encoder=Broken())
    reranker.deterministic = SimpleNamespace(rerank=lambda analysis, candidates: [(item, 0.2) for item in candidates])
    result = reranker.rerank(_analysis(), [_candidate("1"), _candidate("2")], 2)
    assert len(result) == 2
    assert all(item.rerank_method == "deterministic" and item.rerank_score == 0.2 for item in result)
    assert not reranker.uses_cross_encoder


def _runtime(score, method):
    runtime = object.__new__(MultiScaleRagRuntime)
    runtime._ensure_loaded = lambda: True
    runtime._base_config = SimpleNamespace(domain_id="test")
    runtime._gate = SimpleNamespace(analyze=lambda query: SimpleNamespace(entities=[], matched_domains=["test"]))
    runtime._service = SimpleNamespace(
        config=SimpleNamespace(domain_id="test"),
        retrieve=lambda query, top_k: {
            "results": [
                {
                    "id": "1",
                    "rerank_score": score,
                    "rerank_method": method,
                    "fused_score": 0.99,
                }
            ]
        },
    )
    return runtime


def test_negative_semantic_score_never_replaced_by_high_recall_score(monkeypatch):
    monkeypatch.setenv("CHARACTER_RAG_CROSS_ENCODER_MIN_LOGIT", "0")
    monkeypatch.setenv("CHARACTER_RAG_ABSTAIN_THRESHOLD", "0")
    result = _runtime(-4.0, "cross_encoder").retrieve_with_citations("q")
    assert result["abstained"]
    assert result["confidence"] < 0.02
    assert result["confidence_kind"] == "uncalibrated_sigmoid"
    assert result["warnings"]


def test_semantic_threshold_is_independent_of_rule_threshold(monkeypatch):
    monkeypatch.setenv("CHARACTER_RAG_CROSS_ENCODER_MIN_LOGIT", "2.0")
    monkeypatch.setenv("CHARACTER_RAG_ABSTAIN_THRESHOLD", "0.01")
    assert _runtime(1.0, "cross_encoder").retrieve_with_citations("q")["abstained"]
    assert not _runtime(3.0, "cross_encoder").retrieve_with_citations("q")["abstained"]


def test_invalid_threshold_uses_finite_default(monkeypatch):
    monkeypatch.setenv("CHARACTER_RAG_CROSS_ENCODER_MIN_LOGIT", "nan")
    assert _runtime(-1.0, "cross_encoder").retrieve_with_citations("q")["abstained"]


def test_generic_kb_switch_cannot_enable_character_reranking(monkeypatch):
    from knowledge.multiscale_rag import service

    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.delenv("CHARACTER_RAG_RERANKER_ENABLED", raising=False)
    monkeypatch.setenv("CHARACTER_RAG_RERANK_TEXT_VIEW", "embedding_text")
    monkeypatch.setattr(service, "QueryAnalyzer", lambda configs: None)
    instance = service.RoutedMultiScaleService(SimpleNamespace(domain_id="test"), {}, None, all_documents=[])
    assert instance.reranker._resolve_cross_encoder() is None
    assert instance.reranker.text_view == "embedding_text"


def test_character_reranking_has_independent_explicit_opt_in(monkeypatch):
    from knowledge.multiscale_rag import service

    monkeypatch.setenv("RERANKER_ENABLED", "false")
    monkeypatch.setenv("CHARACTER_RAG_RERANKER_ENABLED", "true")
    monkeypatch.setattr(service, "QueryAnalyzer", lambda configs: None)
    instance = service.RoutedMultiScaleService(SimpleNamespace(domain_id="test"), {}, None, all_documents=[])
    assert instance.reranker._cross_encoder_enabled is True
