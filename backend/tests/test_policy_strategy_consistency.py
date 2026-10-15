"""A successful policy cannot contain contradictory strategies.

Reuse the real SQLite/cloud orchestration fixture. State and policy reviewer
outputs are synthetic but use the actual parsers; baseline DecisionPolicy,
service, compiler, API and final prompt are real.
"""

import json

import pytest
import test_circuit_strict_generation as transport_fixture
from fastapi import HTTPException
from repositories.character_memory import DatabaseCharacterMemoryRepository
from repositories.messages import DatabaseMessageRepository
from services.character_context import CharacterContextService, TurnInput
from test_circuit_strict_generation import circuit_rig as circuit_rig

from api import generate
from character import profile_registry
from character.contextual_policy import ContextualDecisionPolicy
from character.decision_policy import STRATEGY_INSTRUCTIONS, DecisionPolicy
from character.memory_service import CharacterMemoryService
from character.semantic_state_estimator import SemanticStateEstimator
from db.schemas import MessageRequest
from inference.context_budget import ReviewContextBudget

pytestmark = pytest.mark.usefixtures("no_live_context_review")
ROLE = "fiction-policy-consistency"
OWNER = "fiction-policy-owner"
QUERY = "完整题设：我计划本周四14:00至15:00排练，尚未报名；只有取得书面确认且当天不下雨才参加，预约不豁免雨天限制。请给我一至两项准备建议，保留这些条件。"
ANSWER = "先向排练方确认书面报名流程，再检查周四天气；尚未获得确认或当天有雨时都不参加。"


def arrange_policy_conflict(tmp_path, monkeypatch, circuit_rig, strategy_ids):
    """Isolate external model semantics, never replace the merge/check logic."""
    rig = circuit_rig
    monkeypatch.setattr(transport_fixture, "FACTS", QUERY)
    monkeypatch.setattr(transport_fixture, "REPLY", ANSWER)
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "fixture.json").write_text(json.dumps({
        "character_id": ROLE, "display_name": "虚构策略核对角色",
        "identity": "给出条件完整的建议", "values": ["用户明确任务优先"],
        "boundaries": ["未确认的预约不得当作已完成，不代用户执行现实操作"],
    }, ensure_ascii=False), encoding="utf-8")
    registry = profile_registry.CharacterProfileRegistry(profiles)
    registry.load_profiles()
    monkeypatch.setattr(profile_registry, "_default_registry", registry)
    for name, value in {
        "CAHM_SEMANTIC_MEMORY_ENABLED": "false", "CONTEXTUAL_MEMORY_SELECTION_ENABLED": "false",
        "CONTEXTUAL_DECISION_POLICY_ENABLED": "false", "MEMORY_SOURCE_RECALL_ENABLED": "false",
    }.items():
        monkeypatch.setenv(name, value)
    rig.state_calls, rig.policy_calls, rig.baselines, rig.prepared = [], [], [], []

    async def state_reviewer(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["current_message"] == QUERY and payload["recent_history"] == []
        rig.state_calls.append(payload)
        return json.dumps({"state": {
            "primary_situation": "factual", "situation_scores": {"factual": .9},
            "user_acts": {"advice_request": .9}, "user_needs": {"guidance": .9},
            "valence": 0.0, "arousal": .2, "warmth": 0.0, "face_threat": 0.0,
            "conversation_phase": "sustaining", "confidence": .9,
        }})

    async def policy_reviewer(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == QUERY and payload["history"] == []
        assert payload["profile"]["values"] == ["用户明确任务优先"]
        assert payload["state_hint"]["acts"]["advice_request"] >= .5
        assert all(identity in payload["allowed_strategies"] for identity in strategy_ids)
        rig.policy_calls.append(payload)
        return json.dumps({"strategy_ids": list(strategy_ids)})

    decision_policy = DecisionPolicy()
    actual_decide = decision_policy.decide

    def capture_baseline(*args, **kwargs):
        baseline = actual_decide(*args, **kwargs)
        rig.baselines.append(baseline)
        return baseline

    decision_policy.decide = capture_baseline
    repository = DatabaseCharacterMemoryRepository(rig.database)
    rig.service = CharacterContextService(
        registry, repository, DatabaseMessageRepository(rig.database),
        memory_service=CharacterMemoryService(repository, semantic_enabled=False), decision_policy=decision_policy,
        semantic_estimator=SemanticStateEstimator(state_reviewer, review_mode="all_non_safety", context_budget=ReviewContextBudget(65536)),
        contextual_policy=ContextualDecisionPolicy(policy_reviewer, context_budget=ReviewContextBudget(65536)),
        source_recall_enabled=False,
    )
    actual_prepare = rig.service.prepare_turn

    async def capture_prepared(*args, **kwargs):
        prepared = await actual_prepare(*args, **kwargs)
        rig.prepared.append(prepared)
        return prepared

    rig.service.prepare_turn = capture_prepared
    return rig


async def baseline_preflight(rig):
    """Return only policy metadata and instruction-presence booleans."""
    prepared = await rig.service.prepare_turn(TurnInput(QUERY, "web", "policy-consistency", OWNER, OWNER, "private"), ROLE)
    dynamic = prepared.compiled.dynamic_context
    return {
        "status": prepared.contextual_policy_status,
        "strategy_ids": list(prepared.decision.strategy_ids),
        "baseline_strategy_ids": list(rig.baselines[-1].strategy_ids),
        "offer_suggestion_present": STRATEGY_INSTRUCTIONS["offer_suggestion"] in dynamic,
        "stay_present_present": STRATEGY_INSTRUCTIONS["stay_present"] in dynamic,
        "state_reviews": len(rig.state_calls), "policy_reviews": len(rig.policy_calls), "answer_calls": len(rig.calls),
    }


async def api_turn(rig):
    return await generate._generate_reply_impl(
        MessageRequest(
            message=QUERY, characterId=ROLE, loraName="default", platform="web", adapter="policy-consistency",
            senderId=OWNER, userId=OWNER, conversationId=OWNER, conversationType="private", sessionId="fiction-policy-session",
        ),
        character_service=rig.service, message_db=rig.database,
        persist_message=False, enable_rag=False, record_invocation=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy_ids", [
    ("offer_suggestion", "stay_present"),  # Contradictory raw model pair.
    ("stay_present",),  # Pair becomes contradictory after required-action merge.
])
async def test_conflicting_policy_stops_real_api_before_answer_model(tmp_path, monkeypatch, circuit_rig, strategy_ids):
    rig = arrange_policy_conflict(tmp_path, monkeypatch, circuit_rig, strategy_ids)
    with pytest.raises(HTTPException) as caught:
        await api_turn(rig)
    assert caught.value.status_code == 503 and isinstance(caught.value.__cause__, ValueError)
    assert "incompatible" in str(caught.value.__cause__).lower()
    assert len(rig.state_calls) == len(rig.policy_calls) == len(rig.baselines) == 1
    assert "offer_suggestion" in rig.baselines[0].strategy_ids
    assert not rig.calls and not rig.prepared
    assert rig.database.execute_sql("SELECT * FROM messages") == []
    assert rig.database.execute_sql("SELECT * FROM model_invocations") == []


@pytest.mark.asyncio
async def test_compatible_policy_reaches_real_model_system_context(tmp_path, monkeypatch, circuit_rig):
    rig = arrange_policy_conflict(tmp_path, monkeypatch, circuit_rig, ("offer_suggestion", "acknowledge_emotion"))
    result = await api_turn(rig)
    assert result.reply == ANSWER and result.modelInvoked is True
    assert len(rig.state_calls) == len(rig.policy_calls) == len(rig.baselines) == len(rig.prepared) == len(rig.calls) == 1
    prepared = rig.prepared[0]
    assert prepared.contextual_policy_status == "applied"
    assert prepared.decision.selection_source == "semantic"
    assert prepared.decision.strategy_ids == ("offer_suggestion", "acknowledge_emotion")
    system = "\n".join(message["content"] for message in rig.calls[0]["messages"] if message["role"] == "system")
    assert STRATEGY_INSTRUCTIONS["offer_suggestion"] in system
    assert STRATEGY_INSTRUCTIONS["acknowledge_emotion"] in system
    assert STRATEGY_INSTRUCTIONS["stay_present"] not in system


@pytest.mark.asyncio
async def test_explicit_advice_before_requested_close_remains_compatible():
    from character.context_builder import _compact_dynamic_projection
    from character.models import CharacterProfile, InteractionState, RelationshipState, WeightedSignal

    state = InteractionState(
        primary_situation="factual", situation_scores=(WeightedSignal("factual", .9),),
        user_acts=(WeightedSignal("advice_request", .9), WeightedSignal("closing", .9)),
        user_needs=(WeightedSignal("guidance", .9),), conversation_phase="closing", confidence=.9,
    )
    profile = CharacterProfile("fiction-close", "虚构角色", values=("先完成用户任务，再按请求结束对话",))
    relationship = RelationshipState()
    baseline = DecisionPolicy().decide(profile, relationship, "factual", interaction=state)
    assert baseline.strategy_ids == ("offer_suggestion", "graceful_close")

    async def reviewer(messages):
        payload = json.loads(messages[-1]["content"])
        assert "先给准备建议" in payload["query"] and "今天就聊到这里" in payload["query"]
        return '{"strategy_ids":["graceful_close"]}'

    outcome = await ContextualDecisionPolicy(reviewer).refine(
        baseline,
        query="完整题设：周四14至15点排练，尚未报名，雨天不参加。请先给准备建议，然后今天就聊到这里。",
        history=(), profile=profile, relationship=relationship, interaction=state, has_relevant_memory=False,
    )
    assert outcome.status == "applied" and outcome.plan.strategy_ids == baseline.strategy_ids
    priorities, _ = _compact_dynamic_projection(state, outcome.plan)
    text = "\n".join(priorities)
    assert STRATEGY_INSTRUCTIONS["offer_suggestion"] in text
    assert STRATEGY_INSTRUCTIONS["graceful_close"] in text
