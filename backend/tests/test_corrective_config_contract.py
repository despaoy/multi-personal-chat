"""Invalid corrective settings must fail before retrieval."""
from unittest.mock import Mock

import pytest

from knowledge.corrective_rag import CorrectiveRAG


@pytest.mark.parametrize("value", [-1, 1.5, True, "2", None])
def test_invalid_retry_limit_cannot_be_coerced(value):
    helper = Mock()
    with pytest.raises(ValueError, match="CorrectiveRAG.max_retries"):
        CorrectiveRAG(helper, max_retries=value)
    helper.retrieve_with_citations.assert_not_called()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True, None, "invalid"])
def test_invalid_confidence_threshold_fails_before_retrieval(value):
    helper = Mock()
    with pytest.raises(ValueError, match="CorrectiveRAG.threshold"):
        CorrectiveRAG(helper, threshold=value)
    helper.retrieve_with_citations.assert_not_called()


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_valid_threshold_boundary_reaches_retrieval_unchanged(threshold):
    helper = Mock()
    helper.retrieve_with_citations.return_value = dict(results=[], citations=[], confidence=0.0, abstained=True)
    result = CorrectiveRAG(helper, threshold=threshold, max_retries=0).retrieve_with_correction("许可要求")
    helper.retrieve_with_citations.assert_called_once_with("许可要求", top_k=None, threshold=threshold, filters=None)
    assert result["abstained"] and len(result["rounds"]) == 1
