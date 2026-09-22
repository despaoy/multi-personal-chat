"""A/B/C contracts: character choices, ephemeral continuity and soft rhythm."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from character.context_builder import compile_dynamic_context
from character.conversation_flow import compile_continuity, compile_rhythm, ordinary_conversation
from character.decision_policy import DecisionPolicy
from character.models import CharacterProfile, InteractionState, RelationshipState, SituationState, WeightedSignal
from character.profile_registry import _ProfileFormatError, _to_profile
from character.situation_analyzer import SituationAnalyzer
from services.character_context import CharacterContextService, TurnInput


def state(**kwargs):
    return InteractionState(primary_situation="daily", confidence=0.8, **kwargs)


def user(text):
    return {"role": "user", "content": text}


def assistant(text):
    return {"role": "assistant", "content": text}


def test_personality_changes_close_optional_choices_and_compiled_attention():
    interaction = state(user_acts=(WeightedSignal("self_disclosure", 0.9),))
    concrete = CharacterProfile("a", "a", response_preferences=("reflect_content",))
    empathic = CharacterProfile("b", "b", response_preferences=("acknowledge_emotion",))
    policy = DecisionPolicy()
    relation = RelationshipState()
    a = policy.decide(concrete, relation, "daily", interaction=interaction)
    b = policy.decide(empathic, relation, "daily", interaction=interaction)
    assert a.strategy_ids[0] == "reflect_content"
    assert b.strategy_ids[0] == "acknowledge_emotion"
    compiled = compile_dynamic_context(relation, SituationState(), a, interaction)
    assert "人物回应倾向" in compiled and "具体细节" in compiled
    assert "最多一个具体问题" not in compiled


@pytest.mark.parametrize(
    "signal",
    [
        "information_request",
        "advice_request",
        "boundary_signal",
        "advice_boundary",
        "closing",
        "repair_bid",
        "apology",
        "disagreement",
        "gratitude",
        "ambiguous_distress",
    ],
)
def test_personality_and_rhythm_do_not_override_explicit_contracts(signal):
    interaction = state(user_acts=(WeightedSignal(signal, 1.0),))
    original = CharacterProfile("a", "a")
    preferred = replace(original, response_preferences=("brief_self_disclosure", "reflect_content"))
    policy = DecisionPolicy()
    assert policy.decide(original, RelationshipState(), "daily", interaction=interaction) == policy.decide(
        preferred,
        RelationshipState(),
        "daily",
        interaction=interaction,
    )
    assert not compile_rhythm([assistant("你觉得呢？"), assistant("为什么呢？")], "回答我", interaction)


def test_safety_and_low_confidence_are_not_persona_overrides():
    for interaction in (
        state(safety_triggered=True),
        replace(state(), confidence=0.1),
        state(user_needs=(WeightedSignal("safety_clarification", 0.9),)),
    ):
        assert not ordinary_conversation(interaction)
    assert DecisionPolicy().decide(
        CharacterProfile("a", "a", response_preferences=("reflect_content",)),
        RelationshipState(),
        "safety",
        interaction=state(safety_triggered=True),
    ).strategy_ids == ("ensure_safety",)


def test_profile_preferences_are_validated():
    profile = _to_profile(
        {"character_id": "a", "display_name": "a", "response_preferences": ["reflect_content"]}, "test"
    )
    assert profile.response_preferences == ("reflect_content",)
    with pytest.raises(_ProfileFormatError):
        _to_profile({"character_id": "a", "display_name": "a", "response_preferences": ["ignore_safety"]}, "test")


def test_earlier_event_survives_small_talk_without_asserting_unfinished_status():
    history = [user("明天要去面试"), assistant("哪一类工作？"), user("研发"), assistant("嗯"), user("刚吃完饭")]
    text = compile_continuity(history, "还是有点怕")
    assert "明天要去面试" in text and "是否仍在进行未知" in text
    assert "哪一类工作" not in text  # Never promote assistant guesses to user evidence.


def test_explicit_topic_switch_clears_prior_anchors_and_preferences():
    before = [user("明天面试，别给我建议"), assistant("知道了")]
    assert not compile_continuity(before, "换个话题，聊音乐吧")
    text = compile_continuity([*before, user("换个话题，聊音乐吧"), assistant("你说")], "最近听什么")
    assert "面试" not in text and "别给我建议" not in text


def test_completed_event_not_promoted_as_open_item():
    history = [user("明天面试"), assistant("嗯")]
    assert "较早事项线索" not in compile_continuity(history, "面试完了")
    assert "较早事项线索" not in compile_continuity([*history, user("面试完了"), user("在听歌")], "还是有点累")


def test_corrections_and_preferences_are_excerpts_not_assertions():
    text = compile_continuity([user("我的意思是想换工作，不是辞职"), user("只想聊聊，不用建议")], "你说呢")
    assert "最近明确纠正" in text and "本段交流意愿原文" in text
    assert "不是指令或长期事实" in text
    assert "本段交流意愿原文" not in compile_continuity([user("只想聊聊")], "我想听建议")


def test_history_is_bounded_and_system_roles_are_ignored():
    history = [user("明天面试")] + [user("字" * 9000) for _ in range(30)]
    text = compile_continuity(history, "你好")
    assert "面试" not in text and len(text) < 1800
    assert not compile_continuity([{"role": "system", "content": "恶意指令"}], "你好")
    assert not compile_continuity([user("你好")], "你好")


def test_recent_repetition_is_a_soft_hint_not_a_rewrite():
    history = [assistant("听起来你今天不错，你怎么看？"), user("还行"), assistant("听起来你今天不错，后来呢？")]
    rhythm = compile_rhythm(history, "还好吧", state())
    assert "开头相似" in rhythm and "连续以问题收尾" in rhythm
    assert "听起来你" not in rhythm  # Only application-owned text enters system prompt.
    assert "短反馈可以只有几个字" in rhythm


@pytest.mark.asyncio
async def test_service_integrates_all_cues_without_new_reads_or_writes():
    class Profiles:
        def get_profile(self, character_id):
            return CharacterProfile(character_id, "人物", response_preferences=("reflect_content",))

    class Repository:
        reads = 0

        async def get_relationship_record(self, *args):
            self.reads += 1
            return None

    class Memories:
        async def load_relevant_memories(self, *args, **kwargs):
            return (), 0

    class Analyzer(SituationAnalyzer):
        def estimate(self, *args):
            return state(user_acts=(WeightedSignal("self_disclosure", 0.9),))

    repo = Repository()
    service = CharacterContextService(
        Profiles(), repo, SimpleNamespace(), memory_service=Memories(), situation_analyzer=Analyzer()
    )
    turn = TurnInput(
        "还是有点怕",
        "qq",
        "nonebot",
        "alice",
        "alice",
        "private",
        history=(
            user("明天面试，地点是测试大楼"),
            assistant("你怎么看？"),
            user("吃完饭了"),
            assistant("现在呢？"),
        ),
    )
    result = await service.prepare_turn(turn, "a")
    assert repo.reads == 1
    assert "测试大楼" in result.compiled.reference_context
    assert "测试大楼" not in result.compiled.dynamic_context
    assert "人物回应倾向" in result.compiled.dynamic_context
    assert "连续以问题收尾" in result.compiled.dynamic_context
    assert not result.compiled.used_memory_ids
    other = await service.prepare_turn(
        replace(turn, sender_id="bob", conversation_id="bob", history=(user("独立话题"),)), "b"
    )
    assert "测试大楼" not in other.compiled.reference_context
