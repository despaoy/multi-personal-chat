"""Corrective retrieval has one bounded loop and no default-success fields."""
from unittest.mock import Mock

import pytest

from knowledge.corrective_rag import CorrectiveRAG


def response(title="书面许可", *, abstained=True):
    return dict(results=[dict(id=7, title=title, content=title, score=0.1 if abstained else 0.8)],
                citations=[], confidence=0.1 if abstained else 0.8, abstained=abstained)


@pytest.mark.parametrize("field", ["abstained", "confidence"])
@pytest.mark.parametrize("round_index", [0, 1])
def test_missing_decision_field_raises_in_either_round(field, round_index):
    malformed = response()
    del malformed[field]
    helper = Mock()
    helper.retrieve_with_citations.side_effect = [response()] * round_index + [malformed]
    with pytest.raises(KeyError, match=field):
        CorrectiveRAG(helper).retrieve_with_correction("进入实验室的条件")
    assert helper.retrieve_with_citations.call_count == round_index + 1


def test_later_retrieval_error_is_not_a_partial_success():
    error = TimeoutError("synthetic retrieval timeout")
    helper = Mock()
    helper.retrieve_with_citations.side_effect = [response(), error]
    with pytest.raises(TimeoutError) as caught:
        CorrectiveRAG(helper).retrieve_with_correction("进入实验室的条件")
    assert caught.value is error
    assert helper.retrieve_with_citations.call_count == 2


def test_two_rewrites_keep_scope_rounds_and_original_input(caplog):
    helper = Mock()
    helper.retrieve_with_citations.side_effect = [response("书面许可"), response("安全培训"), response("已满足", abstained=False)]
    filters = {"knowledge_base_id": 7}
    query = "进入实验室的条件"
    with caplog.at_level("INFO"):
        result = CorrectiveRAG(helper, max_retries=2).retrieve_with_correction(query, top_k=3, filters=filters)
    assert not result["abstained"] and result["reformulated"]
    assert result["original_query"] == query
    assert len(result["rounds"]) == 3
    assert result["reformulated_query"] == result["rounds"][-1]["query"]
    assert all(item["query"].startswith(query) for item in result["rounds"])
    assert all(call.kwargs == dict(top_k=3, threshold=0.3, filters=filters) for call in helper.retrieve_with_citations.call_args_list)
    assert query not in caplog.text
