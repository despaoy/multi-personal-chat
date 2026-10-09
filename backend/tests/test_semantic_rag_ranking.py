"""Semantic scores must retain their own ordering, scale and provenance."""

from types import SimpleNamespace
from unittest.mock import Mock

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


def test_quoted_story_uses_source_metadata_before_intermediate_candidate_cutoff():
    candidates = [_candidate(str(i), '事件') for i in range(1, 26)]
    candidates[-1].document.metadata['story_title'] = '第三卷 标题的延续'

    class RankedPool:
        def rerank(self, analysis, pool, *, top_k):
            assert top_k >= len(pool)
            for candidate in pool:
                candidate.rerank_score = .2
                candidate.rerank_method = 'deterministic'
            return pool[:top_k]

    result = rerank_with_title_frames(_analysis(), candidates, top_k=3, reranker=RankedPool())
    assert result[0].document.id == '25'
    assert result[0].document.content == '事件'
    assert len(result) == 3  # not a hard story filter


@pytest.mark.parametrize('query', ['比较《晨星》和《暮雨》的结局', '解释人物经历'])
def test_story_metadata_preference_keeps_comparisons_and_unscoped_order(query):
    candidates = [_candidate('1', '事件'), _candidate('2', '事件'), _candidate('3', '事件')]
    candidates[1].document.metadata['story_title'] = '暮雨'
    candidates[2].document.metadata['story_title'] = '晨星'
    analysis = _analysis()
    analysis.original_query = query
    analysis.normalized_query = query
    reranker = SimpleNamespace(rerank=lambda analysis, candidates, top_k: candidates)
    result = rerank_with_title_frames(analysis, candidates, top_k=3, reranker=reranker)
    expected = ['2', '3', '1'] if '《' in query else ['1', '2', '3']
    assert [candidate.document.id for candidate in result] == expected


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
def test_invalid_encoder_results_raise_without_changing_ranking_method(kind):
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
    reranker.deterministic.rerank = Mock(side_effect=AssertionError('unexpected downgrade'))
    candidates = [_candidate("1"), _candidate("2")]
    with pytest.raises(RuntimeError):
        reranker.rerank(_analysis(), candidates, 2)
    reranker.deterministic.rerank.assert_not_called()
    assert all(item.rerank_score is None and item.rerank_method == 'none' for item in candidates)


@pytest.mark.parametrize('stage', ['initialization', 'inference'])
def test_required_encoder_errors_propagate_without_sticky_disable(monkeypatch, stage, caplog):
    error = OSError('private-reranker-error')
    encoder = SimpleNamespace(rerank=Mock(side_effect=error))
    factory = Mock(side_effect=error) if stage == 'initialization' else Mock(return_value=encoder)
    monkeypatch.setattr('knowledge.reranker.get_reranker', factory)
    reranker = PipelineReranker(cross_encoder_enabled=True)
    reranker.deterministic.rerank = Mock(side_effect=AssertionError('unexpected downgrade'))
    for _ in range(2):
        with pytest.raises(OSError) as caught:
            reranker.rerank(_analysis(), [_candidate('1'), _candidate('2')], 2)
        assert caught.value is error
    assert factory.call_count == 2
    assert encoder.rerank.call_count == (2 if stage == 'inference' else 0)
    reranker.deterministic.rerank.assert_not_called()
    assert 'private-reranker-error' not in caplog.text


def test_enabled_encoder_runs_only_actual_query(monkeypatch):
    encoder = SimpleNamespace(rerank=Mock(wraps=Encoder().rerank))
    monkeypatch.setattr('knowledge.reranker.get_reranker', lambda: encoder)
    result = PipelineReranker(cross_encoder_enabled=True).rerank(_analysis(), [_candidate('1')], 1)
    assert result[0].rerank_method == 'cross_encoder'
    encoder.rerank.assert_called_once()
    assert encoder.rerank.call_args.args[0] == _analysis().original_query
    assert encoder.rerank.call_args.args[1][0]['id'] == '1'


def test_explicit_disabled_encoder_uses_deterministic_ranking(monkeypatch):
    factory = Mock(side_effect=AssertionError('disabled model must not load'))
    monkeypatch.setattr('knowledge.reranker.get_reranker', factory)
    analysis = _analysis()
    analysis.predicate_preferences = []
    analysis.causal_intent = False
    analysis.story_hits = []
    candidates = [_candidate('1'), _candidate('2')]
    result = PipelineReranker(cross_encoder_enabled=False).rerank(analysis, candidates, 2)
    assert len(result) == 2 and all(item.rerank_method == 'deterministic' for item in result)
    assert all(item.rerank_method == 'none' for item in candidates)
    factory.assert_not_called()


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
