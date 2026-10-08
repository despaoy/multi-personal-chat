"""Service-level contracts for selective semantic context review."""

from __future__ import annotations

import asyncio

import pytest

from character.models import CharacterProfile, RelationshipState
from character.semantic_state_estimator import SemanticInputBudgetError, SemanticStateEstimator
from services.character_context import CharacterContextService, TurnInput, build_character_context_service


class _Profiles:
    def get_profile(self, character_id: str) -> CharacterProfile:
        return CharacterProfile(
            character_id=character_id,
            display_name="月社妃",
            identity="《纸上的魔法使》中的人物",
            traits=("自尊心强",),
            values=("认真对待承诺",),
            speaking_style=("简短直接",),
            boundaries=("不编造事实",),
        )


class _MemoryRepository:
    async def get_relationship(self, _character_id, _user_scope):
        return RelationshipState(stage="familiar")

    async def get_relationship_record(self, _character_id, _user_scope):
        return {"interaction_count": 8}

    async def list_relationship_notes(self, *args):
        return []


class _MemoryService:
    async def recall_with_diagnostics(self, *args, **kwargs):
        return (), 0, {"status": "no_records_returned"}


async def test_retrieval_failure_is_visible_through_preparation():
    class FailedRecall:
        async def recall_with_diagnostics(self, *args, **kwargs):
            raise RuntimeError("synthetic read failure")

    service = _service()
    service._memory_service = FailedRecall()
    with pytest.raises(RuntimeError, match="synthetic read failure"):
        await service.prepare_turn(_turn('我的专业是什么？'), 'tsukiyashiro_kisaki')


async def test_storage_snapshot_reaches_generation_without_selected_memory():
    class RecallSnapshot:
        async def recall_with_diagnostics(self, *args, **kwargs):
            return (), 1, {'status': 'no_relevant_candidates',
                           'field_presence': {'name': False, 'major': True, 'workplace': 'false', 'residence': None}}

    service = _service()
    service._memory_service = RecallSnapshot()
    prepared = await service.prepare_turn(_turn('你保存了我的姓名和专业吗？'), 'tsukiyashiro_kisaki')
    assert dict(prepared.compiled.memory_field_presence) == {'name': False, 'major': True, 'residence': None}
    assert prepared.compiled.used_memory_ids == ()


class _Messages:
    async def list_recent_conversation_history(self, _user_scope, *, limit, max_chars, character_id=None):
        del limit, max_chars
        return ()


class _Reviewer:
    def __init__(self, result) -> None:
        self.result = result
        self.calls = []

    async def __call__(self, messages):
        self.calls.append(messages)
        return self.result


class _BrokenAnalyzer:
    def estimate(self, _message, _history=()):
        raise RuntimeError("synthetic analyzer failure")


class _BrokenSemanticEstimator:
    async def refine_with_diagnostics(self, _message, _history, _state):
        raise RuntimeError("synthetic estimator failure")


def _review_payload() -> dict:
    return {
        "state": {
            "primary_situation": "emotional",
            "situation_scores": {"emotional": 0.86, "conflict": 0.61},
            "user_acts": {"self_disclosure": 0.82, "seek_support": 0.72},
            "user_needs": {"validation": 0.78},
            "valence": -0.65,
            "arousal": 0.55,
            "warmth": -0.1,
            "face_threat": 0.35,
            "conversation_phase": "deepening",
            "confidence": 0.8,
        }
    }


def _turn(message: str, history=()) -> TurnInput:
    return TurnInput(
        message=message,
        platform="qq",
        adapter="onebot",
        sender_id="semantic-review-user",
        conversation_id="semantic-review-user",
        conversation_type="private",
        history=tuple(history),
    )


def _service(estimator=None, *, analyzer=None) -> CharacterContextService:
    return CharacterContextService(
        _Profiles(),
        _MemoryRepository(),  # type: ignore[arg-type]
        _Messages(),  # type: ignore[arg-type]
        memory_service=_MemoryService(),  # type: ignore[arg-type]
        situation_analyzer=analyzer,
        semantic_estimator=estimator,
    )


async def test_ambiguous_turn_is_reviewed_before_decision_and_compilation():
    reviewer = _Reviewer(_review_payload())
    prepared = await _service(SemanticStateEstimator(reviewer)).prepare_turn(
        _turn("我当然开心，毕竟又被放鸽子了"),
        "tsukiyashiro_kisaki",
    )

    assert len(reviewer.calls) == 1
    assert prepared.semantic_review_status == "applied"
    assert "sarcasm" in prepared.semantic_review_reasons
    assert prepared.semantic_review_confidence == 0.8
    assert prepared.interaction.primary_situation == "emotional"
    assert prepared.decision.strategy_ids
    assert "情景类型" not in prepared.compiled.dynamic_context
    assert "我当然开心" not in prepared.compiled.dynamic_context


async def test_clear_factual_turn_stays_on_rule_fast_path():
    reviewer = _Reviewer(_review_payload())
    prepared = await _service(SemanticStateEstimator(reviewer)).prepare_turn(
        _turn("北京现在几点？"),
        "tsukiyashiro_kisaki",
    )

    assert reviewer.calls == []
    assert prepared.semantic_review_status == "not_needed"
    assert prepared.interaction.primary_situation == "factual"


async def test_hard_safety_turn_never_calls_semantic_reviewer():
    reviewer = _Reviewer(_review_payload())
    prepared = await _service(SemanticStateEstimator(reviewer)).prepare_turn(
        _turn("我已经准备伤害自己了，你别管。"),
        "tsukiyashiro_kisaki",
    )

    assert reviewer.calls == []
    assert prepared.semantic_review_status == "not_needed"
    assert prepared.interaction.safety_triggered is True
    assert prepared.interaction.primary_situation == "safety"


async def test_timeout_stops_character_preparation():
    async def slow(_messages):
        await asyncio.sleep(0.1)
        return _review_payload()
    service = _service(SemanticStateEstimator(slow, timeout_seconds=0.01))
    with pytest.raises(TimeoutError):
        await service.prepare_turn(_turn("我当然开心，毕竟又被放鸽子了"), "tsukiyashiro_kisaki")


async def test_rule_analyzer_failure_stops_before_semantic_review():
    reviewer = _Reviewer(_review_payload())
    service = _service(SemanticStateEstimator(reviewer), analyzer=_BrokenAnalyzer())
    with pytest.raises(RuntimeError, match="synthetic analyzer failure"):
        await service.prepare_turn(_turn("这轮规则分析器会失败"), "tsukiyashiro_kisaki")
    assert reviewer.calls == []


async def test_input_budget_failure_remains_distinct_at_service_boundary():
    reviewer = _Reviewer(_review_payload())
    service = _service(SemanticStateEstimator(reviewer, review_mode="all_non_safety"))
    message = "背景。" * 1400 + "我只想安静待一会。"
    with pytest.raises(SemanticInputBudgetError):
        await service.prepare_turn(_turn(message), "tsukiyashiro_kisaki")
    assert reviewer.calls == []


async def test_custom_estimator_failure_is_not_replaced_by_rule_state():
    with pytest.raises(RuntimeError, match="synthetic estimator failure"):
        await _service(_BrokenSemanticEstimator()).prepare_turn(
            _turn("我当然开心，毕竟又被放鸽子了"), "tsukiyashiro_kisaki",
        )


async def test_service_without_estimator_keeps_the_existing_constructor_contract():
    prepared = await _service().prepare_turn(
        _turn("我当然开心，毕竟又被放鸽子了"),
        "tsukiyashiro_kisaki",
    )

    assert prepared.semantic_review_status == "disabled"
    assert prepared.semantic_review_reasons == ()
    assert prepared.interaction.primary_situation


def test_production_service_factory_wires_the_environment_controlled_reviewer(monkeypatch):
    monkeypatch.setenv("DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED", "true")
    monkeypatch.setenv("DYNAMIC_CONTEXT_SEMANTIC_REVIEW_TIMEOUT_SECONDS", "1.75")

    service = build_character_context_service(object())

    assert service._semantic_estimator is not None
    assert service._semantic_estimator._reviewer is not None
    assert service._semantic_estimator._timeout_seconds == 1.75


async def test_decision_failure_is_not_retried_with_fewer_arguments():
    from unittest.mock import Mock

    service = _service()
    failure = RuntimeError("synthetic decision failure")
    decide = Mock(side_effect=failure)
    service._decision_policy.decide = decide
    with pytest.raises(RuntimeError) as caught:
        await service.prepare_turn(_turn("你好"), "tsukiyashiro_kisaki")
    assert caught.value is failure
    decide.assert_called_once()


@pytest.mark.parametrize("selection_enabled", [False, True])
async def test_missing_current_recall_interface_does_not_use_legacy_result(selection_enabled):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    legacy = AsyncMock(return_value=((), 0))
    service = _service()
    service._memory_service = SimpleNamespace(load_relevant_memories=legacy)
    selector = SimpleNamespace(select=AsyncMock())
    service._memory_selector = selector if selection_enabled else None
    with pytest.raises(AttributeError, match="recall_with_diagnostics"):
        await service.prepare_turn(_turn("我的专业是什么？"), "tsukiyashiro_kisaki")
    legacy.assert_not_awaited()
    selector.select.assert_not_awaited()


@pytest.mark.parametrize("selection_enabled", [False, True])
async def test_single_recall_preserves_context_and_storage_diagnostics(selection_enabled):
    from dataclasses import replace
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from character.evidence_selector import SelectionOutcome

    stamp = datetime(2026, 10, 9, tzinfo=timezone.utc)
    trace = {"status": "no_records_returned", "field_presence": {"major": False}}
    recall = AsyncMock(return_value=((), 0, trace))
    service = _service()
    service._memory_service = SimpleNamespace(recall_with_diagnostics=recall)
    service._memory_selector = SimpleNamespace(select=AsyncMock(return_value=SelectionOutcome(status="empty"))) if selection_enabled else None
    turn = replace(_turn("我晚上喝咖啡吗？", ({"role": "user", "content": "我晚上不喝咖啡。"},)), received_at=stamp)
    prepared = await service.prepare_turn(turn, "tsukiyashiro_kisaki")
    recall.assert_awaited_once()
    assert prepared.memory_recall is trace
    assert prepared.compiled.memory_status == "no_match"
    assert dict(prepared.compiled.memory_field_presence) == {"major": False}
    if selection_enabled:
        assert recall.call_args.kwargs["for_contextual_selection"] is True
        assert "我晚上不喝咖啡。" in recall.call_args.kwargs["retrieval_context"]
        assert recall.call_args.kwargs["reference_time"] == stamp
    else:
        assert recall.call_args.kwargs == {}


async def test_manual_review_fixture_supports_current_preparation_contract():
    import runpy
    from pathlib import Path

    fixture = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/run_interaction_reply_review.py"))
    service = CharacterContextService(_Profiles(), fixture["_MemoryRepository"](), fixture["_Messages"](),
        memory_service=fixture["_MemoryService"](), source_recall_enabled=False)
    prepared = await service.prepare_turn(_turn("你好"), "tsukiyashiro_kisaki")
    assert prepared.history == () and prepared.compiled.memory_packets == ()
    assert prepared.memory_recall["status"] == "no_records_returned"
