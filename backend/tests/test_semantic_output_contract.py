"""Only the declared state envelope can produce successful review diagnostics."""

import json
from copy import deepcopy

import pytest
from fastapi import HTTPException
from test_semantic_character_context import _service
from test_semantic_review_output import HISTORY, QUERY
from test_semantic_state_estimator import _review_payload, _Reviewer, _state

from character.semantic_state_estimator import SemanticStateEstimator
from db.schemas import MessageRequest


def invalid_shape(defect):
    value = deepcopy(_review_payload())
    if defect == "flat":
        return value["state"]
    if defect == "extra_root":
        value["error"] = "synthetic incomplete review"
    elif defect == "extra_state":
        value["state"]["rationale"] = "synthetic undeclared text"
    elif defect == "missing_field":
        del value["state"]["arousal"]
    elif defect == "signal_list":
        value["state"]["user_acts"] = [dict(signal_id="seek_support", score=0.76)]
    elif defect == "safety_field":
        value["state"]["safety_triggered"] = False
    else:
        raise AssertionError("Unknown explicit schema defect")
    return value


@pytest.mark.parametrize("mode", ["selective", "all_non_safety"])
@pytest.mark.parametrize(
    "defect", ["flat", "extra_root", "extra_state", "missing_field", "signal_list", "safety_field"]
)
async def test_undeclared_shape_fails_and_healthy_retry_releases_guard(mode, defect):
    reviewer = _Reviewer(json.dumps(invalid_shape(defect)))
    estimator = SemanticStateEstimator(reviewer, review_mode=mode)
    with pytest.raises(ValueError, match="semantic review"):
        await estimator.refine_with_diagnostics(QUERY, HISTORY, _state(confidence=0.2))
    assert len(reviewer.calls) == 1
    assert json.loads(reviewer.calls[0][-1]["content"])["current_message"] == QUERY
    reviewer.result = json.dumps(_review_payload())
    result = await estimator.refine_with_diagnostics(QUERY, HISTORY, _state(confidence=0.2))
    assert result.status == "applied" and len(reviewer.calls) == 2
    assert result.state.primary_situation == "emotional"


@pytest.mark.parametrize("defect", ["extra_root", "signal_list"])
async def test_actual_api_preparation_exposes_schema_failure_without_returning_context(defect):
    from api import generate

    reviewer = _Reviewer(invalid_shape(defect))
    service = _service(SemanticStateEstimator(reviewer, review_mode="all_non_safety"))
    request = MessageRequest(message=QUERY, history=list(HISTORY), senderId="schema-test", conversationId="schema-test")
    with pytest.raises(HTTPException) as caught:
        await generate._prepare_character_turn(request, "tsukiyashiro_kisaki", character_service=service)
    assert caught.value.status_code == 503 and isinstance(caught.value.__cause__, ValueError)
    assert len(reviewer.calls) == 1
    reviewer.result = _review_payload()
    prepared = await generate._prepare_character_turn(request, "tsukiyashiro_kisaki", character_service=service)
    assert prepared.semantic_review_status == "applied" and len(reviewer.calls) == 2
    assert QUERY not in prepared.compiled.dynamic_context
