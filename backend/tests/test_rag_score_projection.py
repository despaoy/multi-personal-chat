"""Project actual producer scores once; controlled logits, no model assets."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_reranker import _FakeModel, _FakeTokenizer, _install_fake_transformers

from knowledge import rag_helper
from knowledge.reranker import CrossEncoderReranker, RerankConfig

QUERY = "各项登记的办理日期和原件要求是什么？"
ROWS = (
    dict(id="a", title="竹湾登记", content="竹湾登记周二办理，须持申请卡原件，预约不能替代原件。", score=.8, fused_score=.75),
    dict(id="b", title="枫台登记", content="枫台登记周四办理，须持确认函原件，预约不能替代原件。", score=.7, fused_score=.65),
    dict(id="c", title="松岭登记", content="松岭登记周六办理，须持证件原件，预约不能替代原件。", score=.6, fused_score=.55),
)


@pytest.fixture
def rig(monkeypatch):
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    index = SimpleNamespace(cache_generation=1, hybrid_search=Mock(return_value=ROWS))
    monkeypatch.setattr(rag_helper, "get_vector_db", lambda: index)
    return SimpleNamespace(helper=rag_helper.RAGHelper(), index=index)


def attach_encoder(rig, monkeypatch, *, normalize=True):
    model = _FakeModel({row["content"]: score for row, score in zip(ROWS, (4., 2., 0.), strict=True)})
    _install_fake_transformers(monkeypatch, tokenizer=_FakeTokenizer(), model=model)
    encoder = CrossEncoderReranker(RerankConfig(
        model_name="synthetic-score-projection", device="cpu", score_normalize=normalize,
    ))
    normalizer = Mock(wraps=encoder._normalize_scores)
    monkeypatch.setattr(encoder, "_normalize_scores", normalizer)
    rig.helper.reranker, rig.helper.enable_reranking = encoder, True
    return model, normalizer


def test_rerank_score_is_stable_across_top_k_and_matches_citations(rig, monkeypatch):
    model, normalizer = attach_encoder(rig, monkeypatch)
    first = rig.helper.retrieve_with_citations(QUERY, top_k=2)
    assert [row["normalized_score"] for row in first["results"]] == [1., .5]
    assert [row["rerank_score"] for row in first["results"]] == [4., 2.]
    assert [citation["score"] for citation in first["citations"]] == [1., .5]
    assert normalizer.call_count == len(model.forward_calls) == 1
    second = rig.helper.retrieve_with_citations(QUERY, top_k=3)
    assert [row["normalized_score"] for row in second["results"]] == [1., .5, 0.]
    assert all(row["normalized_score"] == row["rerank_normalized_score"] for row in second["results"])
    assert normalizer.call_count == len(model.forward_calls) == 2
    assert all("normalized_score" not in row and "rerank_score" not in row for row in ROWS)


def test_explicit_encoder_normalization_setting_is_not_overridden(rig, monkeypatch):
    model, normalizer = attach_encoder(rig, monkeypatch, normalize=False)
    result = rig.helper.retrieve_context(QUERY, top_k=2)
    assert [row["normalized_score"] for row in result] == [4., 2.]
    assert normalizer.call_count == len(model.forward_calls) == 1


def test_unranked_fusion_scores_are_not_rescaled_to_a_relative_maximum(rig):
    result = rig.helper.retrieve_with_citations(QUERY, top_k=2)
    assert [row["normalized_score"] for row in result["results"]] == [.75, .65]
    assert [citation["score"] for citation in result["citations"]] == [.75, .65]
    assert [row["score"] for row in result["results"]] == [.8, .7]


def test_missing_required_encoder_projection_is_not_fabricated_or_cached(rig):
    rig.helper.enable_reranking = True
    rig.helper.reranker = SimpleNamespace(rerank=Mock(return_value=[dict(ROWS[0], rerank_score=4.)]))
    with pytest.raises(KeyError, match="rerank_normalized_score"):
        rig.helper.retrieve_context(QUERY)
    assert not rig.helper._query_cache
