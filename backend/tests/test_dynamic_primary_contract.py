"""Unknown primary states cannot become a successful daily conversation context."""

from unittest.mock import Mock

import pytest
from test_semantic_character_context import _review_payload, _service, _turn

from character.models import InteractionState, WeightedSignal
from character.semantic_state_estimator import SemanticReviewOutcome, _parse_reviewed_state
from character.situation_analyzer import SituationAnalyzer


def state(primary, soft):
    return InteractionState(
        primary_situation=primary,
        situation_scores=(WeightedSignal("daily", 0.8),) if soft else (),
        user_acts=(WeightedSignal("seek_support", 0.8), WeightedSignal("information_request", 0.8)) if soft else (),
    )


@pytest.mark.parametrize("soft", [False, True])
def test_invalid_primary_cannot_be_hidden_by_daily_or_mixed_goal(soft):
    with pytest.raises(KeyError):
        SituationAnalyzer().response_goal(state("invalid-primary", soft))


@pytest.mark.asyncio
@pytest.mark.parametrize("soft", [False, True])
@pytest.mark.parametrize("producer", ["analyzer", "custom_estimator"])
async def test_preparation_fails_before_decision_with_invalid_primary(soft, producer):
    invalid = state("invalid-primary", soft)

    class Analyzer(SituationAnalyzer):
        def estimate(self, *args, **kwargs):
            return invalid if producer == "analyzer" else InteractionState()

    class Estimator:
        async def refine_with_diagnostics(self, message, history, original):
            return SemanticReviewOutcome(
                invalid, "applied", (), 0.0, len(history), original.confidence, invalid.confidence
            )

    service = _service(Estimator() if producer == "custom_estimator" else None, analyzer=Analyzer())
    decide = Mock(side_effect=AssertionError("Invalid state reached decision policy"))
    service._decision_policy.decide = decide
    with pytest.raises(KeyError):
        await service.prepare_turn(_turn("这是完整的隔离状态契约检查。"), "tsukiyashiro_kisaki")
    decide.assert_not_called()


def test_model_unknown_primary_is_rejected_at_existing_protocol_boundary():
    payload = _review_payload()
    payload["state"]["primary_situation"] = "private-invalid-model-value"
    with pytest.raises(ValueError, match="primary situation") as caught:
        _parse_reviewed_state(payload)
    assert "private-" not in str(caught.value)
