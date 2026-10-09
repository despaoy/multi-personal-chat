"""Malformed evidence scores cannot masquerade as confidence decisions."""
import pytest

from knowledge.rag_helper import RAGHelper


@pytest.fixture
def helper():
    return object.__new__(RAGHelper)


@pytest.mark.parametrize("fields", [
    {"score": float("nan"), "fused_score": 0.8},
    {"fused_score": float("inf")},
    {"final_score": float("-inf")},
    {"score": None, "fused_score": 0.8},
    {"score": "invalid"},
    {"score": True},
    {},
])
def test_invalid_score_cannot_return_a_citation_or_abstention(helper, monkeypatch, fields):
    record = dict(id=7, title="Permit", content="Written approval is required.", **fields)
    monkeypatch.setattr(helper, "retrieve_context", lambda *args, **kwargs: [record])
    with pytest.raises(ValueError, match="RAG retrieval score must be a finite number"):
        helper.retrieve_with_citations("Is written approval required?")


@pytest.mark.parametrize("fields,confidence", [
    ({"score": 0.8}, 0.8),
    ({"fused_score": 0.1}, 0.1),
    ({"final_score": 1.2}, 1.0),
    ({"score": -0.2}, 0.0),
    ({"score": 0.0, "fused_score": 0.9}, 0.0),
])
def test_valid_scores_keep_precedence_and_decision(helper, monkeypatch, fields, confidence):
    record = dict(id=7, title="Permit", content="Written approval is required.", **fields)
    monkeypatch.setattr(helper, "retrieve_context", lambda *args, **kwargs: [record])
    result = helper.retrieve_with_citations("Is written approval required?")
    assert result["confidence"] == confidence
    assert result["abstained"] is (confidence < 0.3)
    assert bool(result["citations"]) is (confidence >= 0.3)


def test_no_evidence_remains_a_normal_abstention(helper, monkeypatch):
    monkeypatch.setattr(helper, "retrieve_context", lambda *args, **kwargs: [])
    result = helper.retrieve_with_citations("Is written approval required?")
    assert result == dict(results=[], citations=[], confidence=0.0, abstained=True)
