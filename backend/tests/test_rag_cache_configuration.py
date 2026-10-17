"""Invalid cache settings cannot be silently repaired or expose their value."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from knowledge import rag_helper

SETTING = "RAG_RETRIEVAL_CACHE_TTL"
QUERY = "合成档案馆开门时间"
ROW = {
    "id": "synthetic-opening",
    "title": "完整合成规程",
    "content": "合成档案馆周一九点开门，周三不开门。",
    "score": 0.8,
}


@pytest.mark.parametrize("value", ["0", "-1", "-12", "", "1.5", "synthetic-private-cache-value"])
def test_invalid_ttl_fails_before_retrieval_resources_and_redacts_value(monkeypatch, value):
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv(SETTING, value)
    reranker = Mock(side_effect=AssertionError("invalid configuration allocated a reranker"))
    vector = Mock(side_effect=AssertionError("invalid configuration loaded an index"))
    monkeypatch.setattr(rag_helper, "get_reranker", reranker)
    monkeypatch.setattr(rag_helper, "get_vector_db", vector)
    with pytest.raises(RuntimeError, match=SETTING) as caught:
        rag_helper.RAGHelper()
    assert "positive integer" in str(caught.value)
    assert "synthetic-private" not in str(caught.value)
    reranker.assert_not_called()
    vector.assert_not_called()


@pytest.mark.parametrize("value,expected", [(None, 60), ("1", 1), ("60", 60)])
def test_missing_and_valid_positive_ttl_preserve_configured_lifetime(monkeypatch, value, expected):
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    if value is None:
        monkeypatch.delenv(SETTING, raising=False)
    else:
        monkeypatch.setenv(SETTING, value)
    assert rag_helper.RAGHelper()._cache_ttl == expected


def test_expired_cache_failure_does_not_return_old_evidence_and_can_recover(monkeypatch):
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    monkeypatch.setenv(SETTING, "2")
    clock = [100.0]
    monkeypatch.setattr(rag_helper.time, "monotonic", lambda: clock[0])
    index = SimpleNamespace(cache_generation=1, hybrid_search=Mock(return_value=[ROW]))
    monkeypatch.setattr(rag_helper, "get_vector_db", lambda: index)
    helper = rag_helper.RAGHelper()
    first = helper.retrieve_with_citations(QUERY)
    assert not first["abstained"] and first["results"][0]["content"] == ROW["content"]
    assert helper.retrieve_with_citations(QUERY) == first
    assert index.hybrid_search.call_count == 1
    clock[0] = 102.0
    error = OSError("synthetic-private-index-detail")
    index.hybrid_search.side_effect = error
    with pytest.raises(OSError) as caught:
        helper.retrieve_with_citations(QUERY)
    assert caught.value is error and not helper._query_cache
    index.hybrid_search.side_effect = None
    index.hybrid_search.return_value = [dict(ROW, content="新合成规程：周一十点开门，周三仍不开门。")]
    recovered = helper.retrieve_with_citations(QUERY)
    assert "十点" in recovered["results"][0]["content"]
    assert index.hybrid_search.call_count == 3


def test_invalid_configuration_cannot_publish_a_partial_singleton(monkeypatch):
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    monkeypatch.setenv(SETTING, "-1")
    monkeypatch.setattr(rag_helper, "_rag_helper", None)
    with pytest.raises(RuntimeError, match=SETTING):
        rag_helper.get_rag_helper()
    assert rag_helper._rag_helper is None
    monkeypatch.setenv(SETTING, "3")
    helper = rag_helper.get_rag_helper()
    assert helper._cache_ttl == 3 and rag_helper.get_rag_helper() is helper
