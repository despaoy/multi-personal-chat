"""Structured correction reports all attempted rewrites without default success."""
from copy import deepcopy
from unittest.mock import Mock

import pytest

from knowledge.grounded_answer.corrective import CorrectiveRetrievalAdapter


def bundle(key="permit", *, abstained=True):
    return dict(results=[dict(id=key, title=key, summary=key, fused_score=0.2)],
                citations=[dict(source_id=key)], confidence=0.2 if abstained else 0.8, abstained=abstained)


@pytest.mark.parametrize("outcome", ["low", "outside", "success"])
def test_attempted_rewrite_is_recorded_for_every_outcome(outcome):
    first = bundle()
    second = None if outcome == "outside" else bundle("training", abstained=outcome != "success")
    originals = deepcopy([first, second])
    retrieve = Mock(side_effect=[first, second])
    result, info = CorrectiveRetrievalAdapter(retrieve).retrieve_with_correction("entry", top_k=3, filters={"scope": "lab"})
    assert info["reformulated"] is True
    assert info["reformulated_query"] == "entry permit"
    assert [r["query"] for r in info["rounds"]] == ["entry", "entry permit"]
    assert all(c.kwargs == dict(top_k=3, filters={"scope": "lab"}) for c in retrieve.call_args_list)
    if outcome == "success":
        assert result["abstained"] is False
        assert [r["id"] for r in result["results"]] == ["permit", "training"]
        assert [r["source_id"] for r in result["citations"]] == ["permit", "training"]
    else:
        assert result == (first if outcome == "outside" else second)
    assert [first, second] == originals


@pytest.mark.parametrize("mode", ["outside", "success", "unchanged", "zero"])
def test_no_rewrite_paths_remain_single_retrieval(mode):
    first = None if mode == "outside" else bundle(abstained=mode != "success")
    query = "permit" if mode == "unchanged" else "entry"
    retrieve = Mock(return_value=first)
    result, info = CorrectiveRetrievalAdapter(retrieve, max_retries=0 if mode == "zero" else 1).retrieve_with_correction(query, top_k=3, filters=None)
    assert result == first and retrieve.call_count == 1
    assert not info["reformulated"] and info["reformulated_query"] is None


@pytest.mark.parametrize("field", ["abstained", "confidence"])
@pytest.mark.parametrize("round_index", [0, 1])
def test_missing_required_decision_cannot_be_success(field, round_index):
    bad = bundle()
    del bad[field]
    retrieve = Mock(side_effect=[bundle()] * round_index + [bad])
    with pytest.raises(KeyError, match=field):
        CorrectiveRetrievalAdapter(retrieve).retrieve_with_correction("entry", top_k=3, filters=None)


@pytest.mark.parametrize("limit", [-1, 0.5, True, "2", None])
def test_invalid_retry_limit_is_rejected(limit):
    with pytest.raises(ValueError, match="max_retries"):
        CorrectiveRetrievalAdapter(Mock(), max_retries=limit)


def test_later_failure_propagates_instead_of_returning_prior_evidence():
    failure = TimeoutError("synthetic timeout")
    retrieve = Mock(side_effect=[bundle(), failure])
    with pytest.raises(TimeoutError) as caught:
        CorrectiveRetrievalAdapter(retrieve).retrieve_with_correction("entry", top_k=3, filters=None)
    assert caught.value is failure
