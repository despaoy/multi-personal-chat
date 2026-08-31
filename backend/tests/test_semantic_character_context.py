"""Service-level contracts for selective semantic context review."""

from __future__ import annotations

import asyncio

from character.models import CharacterProfile, InteractionState, RelationshipState
from character.semantic_state_estimator import SemanticStateEstimator
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


class _MemoryService:
    async def load_relevant_memories(self, _character_id, _user_scope, _message):
        return (), 0


class _Messages:
    async def list_recent_conversation_history(self, _user_scope, *, limit, max_chars):
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


async def test_timeout_falls_back_to_the_exact_rule_state_and_keeps_character_context():
    async def slow(_messages):
        await asyncio.sleep(0.1)
        return _review_payload()

    service = _service(SemanticStateEstimator(slow, timeout_seconds=0.01))
    rule_state = service._situation_analyzer.estimate("我当然开心，毕竟又被放鸽子了", ())
    prepared = await service.prepare_turn(
        _turn("我当然开心，毕竟又被放鸽子了"),
        "tsukiyashiro_kisaki",
    )

    assert prepared.semantic_review_status == "fallback"
    assert prepared.semantic_review_fallback_reason == "timeout"
    assert prepared.interaction == rule_state
    assert prepared.compiled.profile_context
    assert prepared.compiled.dynamic_context


async def test_rule_analyzer_failure_skips_semantic_review_and_preserves_legacy_fallback():
    reviewer = _Reviewer(_review_payload())
    prepared = await _service(
        SemanticStateEstimator(reviewer),
        analyzer=_BrokenAnalyzer(),
    ).prepare_turn(
        _turn("这轮规则分析器会失败"),
        "tsukiyashiro_kisaki",
    )

    assert reviewer.calls == []
    assert prepared.semantic_review_status == "analysis_failed"
    assert prepared.interaction == InteractionState()
    assert "日常互动" in prepared.compiled.dynamic_context


async def test_unexpected_custom_estimator_failure_keeps_the_rule_state():
    service = _service(_BrokenSemanticEstimator())
    rule_state = service._situation_analyzer.estimate("我当然开心，毕竟又被放鸽子了", ())

    prepared = await service.prepare_turn(
        _turn("我当然开心，毕竟又被放鸽子了"),
        "tsukiyashiro_kisaki",
    )

    assert prepared.semantic_review_status == "fallback"
    assert prepared.semantic_review_fallback_reason == "error"
    assert prepared.interaction == rule_state
    assert prepared.compiled.profile_context


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
