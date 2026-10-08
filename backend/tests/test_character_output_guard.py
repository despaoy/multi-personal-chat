from __future__ import annotations

import pytest

from character.decision_policy import DecisionPolicy
from character.models import (
    CharacterProfile,
    DecisionPlan,
    InteractionState,
    RelationshipState,
    WeightedSignal,
)
from character.output_guard import (
    AFFILIATION_MISREAD_AS_SAFETY,
    AFFILIATION_NOT_RECIPROCATED,
    AUTONOMY_BOUNDARY_IGNORED,
    CLOSING_WITH_QUESTION,
    FACTUAL_TASK_STYLE_DRIFT,
    GENERIC_ASSISTANT_TEMPLATE,
    IGNORED_ADVICE_BOUNDARY,
    MECHANICAL_REPAIR,
    MISSING_GENTLE_SAFETY_CHECK,
    MISSING_NEGATIVE_EMOTION_ACKNOWLEDGEMENT,
    MISSING_SELF_ANSWER,
    MISSING_URGENT_SAFETY_CHECK,
    RESOLVED_THIRD_PARTY_CRISIS_ESCALATION,
    THIRD_PARTY_SAFETY_INCOMPLETE,
    UNPROMPTED_ADVICE,
    UNPROMPTED_CANONICAL_IDENTITY,
    UNPROMPTED_LORE_FLOURISH,
    UNSUPPORTED_FACTUAL_CLAIM,
    UNSUPPORTED_THIRD_PARTY_GENDER,
    UNSUPPORTED_USER_FACT,
    ReplyGuard,
    build_reply_guard,
    retry_instruction,
    validate_reply,
)
from character.situation_analyzer import SituationAnalyzer
from inference.generation_request import GenerationRequest, generate_character_response


def _profile() -> CharacterProfile:
    return CharacterProfile(
        character_id="kisaki",
        display_name="月社妃",
        canonical_relationships=("琉璃：哥哥", "夜子：朋友"),
    )


def _guard_for(message: str) -> ReplyGuard:
    interaction = SituationAnalyzer().estimate(message)
    decision = DecisionPolicy().decide(
        _profile(),
        RelationshipState(stage="familiar"),
        interaction.primary_situation,
        interaction=interaction,
    )
    return build_reply_guard(_profile(), message, (), interaction, decision)


def test_unprompted_canonical_names_are_forbidden_but_user_requested_name_is_allowed():
    interaction = InteractionState(situation_scores=(WeightedSignal("daily", 0.8),))
    plan = DecisionPlan(strategy_ids=("reflect_content",))

    generic = build_reply_guard(_profile(), "今天吃了蛋糕。", (), interaction, plan)
    asked = build_reply_guard(_profile(), "琉璃是谁？", (), interaction, plan)

    assert generic.forbidden_terms == ("琉璃", "夜子")
    assert asked.forbidden_terms == ("夜子",)
    assert validate_reply("琉璃最近在减肥。", generic) == (UNPROMPTED_CANONICAL_IDENTITY,)
    assert validate_reply("琉璃是她的哥哥。", asked) == ()


def test_declarative_question_words_do_not_trigger_closing_guard():
    guard = ReplyGuard(closing=True)

    assert validate_reply("我知道怎么回事了，晚安。", guard) == ()
    assert validate_reply("我知道为什么会这样了，今晚先到这里。", guard) == ()
    assert validate_reply("你知道怎么回事了，晚安。", guard) == ()
    assert validate_reply("你现在是否安全。", guard) == (CLOSING_WITH_QUESTION,)
    assert validate_reply("你知道怎么回事吗？", guard) == (CLOSING_WITH_QUESTION,)


def test_unprompted_long_term_user_preference_is_rejected_without_memory():
    interaction = InteractionState(situation_scores=(WeightedSignal("daily", 0.8),))
    plan = DecisionPlan(strategy_ids=("reflect_content",))
    no_evidence = build_reply_guard(_profile(), "草莓蛋糕看起来不错。", (), interaction, plan)
    stated = build_reply_guard(_profile(), "我一直很喜欢草莓蛋糕。", (), interaction, plan)
    remembered = build_reply_guard(
        _profile(),
        "草莓蛋糕看起来不错。",
        (),
        interaction,
        plan,
        has_relevant_memory=True,
    )

    reply = "你对草莓甜品的喜好可是出了名的。"
    assert validate_reply(reply, no_evidence) == (UNSUPPORTED_USER_FACT,)
    assert validate_reply(reply, stated) == ()
    assert validate_reply(reply, remembered) == ()


def test_turn_contracts_detect_advice_closing_safety_and_missing_self_answer():
    guard = ReplyGuard(
        forbid_advice=True,
        closing=True,
        require_gentle_safety_check=True,
        require_self_answer=True,
    )
    violations = validate_reply("你可以先休息一下。要不要说说发生了什么？", guard)

    assert IGNORED_ADVICE_BOUNDARY in violations
    assert CLOSING_WITH_QUESTION in violations
    assert MISSING_GENTLE_SAFETY_CHECK in violations
    assert MISSING_SELF_ANSWER in violations


def test_explicit_quiet_presence_rejects_questions_but_accepts_silent_company():
    guard = _guard_for("今天真的很累。先别给我建议，陪我待一会儿就好。")

    assert guard.forbid_advice is True
    assert guard.quiet_presence is True
    assert validate_reply("我在。你今天发生了什么？", guard) == (IGNORED_ADVICE_BOUNDARY,)
    assert validate_reply("我在。你不必现在解释。", guard) == ()


def test_companionship_and_affiliation_questions_are_not_misread_as_advice_boundary():
    companionship = _guard_for("我今天有点累，能陪我聊聊吗？")
    affiliation = _guard_for("想死你了，终于等到你上线。")

    assert companionship.forbid_advice is False
    assert companionship.quiet_presence is False
    assert IGNORED_ADVICE_BOUNDARY not in validate_reply("最累的是哪一段？", companionship)
    assert affiliation.forbid_advice is False
    assert affiliation.quiet_presence is False
    assert validate_reply("总算等到我了？想我就直说。", affiliation) == ()


def test_no_advice_without_companionship_does_not_forbid_a_direct_question():
    guard = _guard_for("别给建议，直接告诉我答案。")
    assert guard.forbid_advice is True
    assert guard.quiet_presence is False
    assert guard.require_autonomy_ack is False
    assert IGNORED_ADVICE_BOUNDARY not in validate_reply("答案是四。需要我说明计算过程吗？", guard)
    templated = "答案是四。你需要我再解释一下计算过程吗？"
    templated_violations = validate_reply(templated, guard)
    assert templated_violations == (GENERIC_ASSISTANT_TEMPLATE,)


@pytest.mark.parametrize(
    ("message", "reply"),
    [
        (
            "今天路过一家新开的甜品店，橱窗里的草莓蛋糕看起来还不错。",
            "草莓蛋糕确实显眼。不过记得先了解一下它的口味和质量。",
        ),
        ("我拿到一直想要的实习 offer 了！", "这次确实做得不错，继续保持这份努力。"),
        ("想死你了，终于等到你上线。", "原来是在想我。不过得小心点，别让思念成疾。"),
        (
            "你到底是怎样的人？还有，我今天被同事否定了，心里很堵。",
            "我是月社妃。可以找些轻松的事情聊聊，分散一下注意力。",
        ),
        (
            "我朋友去年想死过，不过现在已经安全了，也一直在接受帮助。",
            "有什么具体的方式或资源可以帮助对方？比如参加支持小组。",
        ),
    ],
)
def test_unprompted_advice_is_separate_from_an_explicit_no_advice_boundary(message, reply):
    guard = _guard_for(message)

    assert guard.forbid_unprompted_advice is True
    assert guard.forbid_advice is False
    assert UNPROMPTED_ADVICE in validate_reply(reply, guard)


def test_explicit_tasks_boundaries_and_active_safety_do_not_enable_unprompted_advice():
    advice = _guard_for("我今天考试没考好，很难过，你说我接下来该怎么办？")
    factual = _guard_for("网页的滚动条怎么设置？")
    boundary = _guard_for("今天真的很累。先别给我建议，陪我待一会儿就好。")
    safety = _guard_for("朋友昨晚说他不想活了，我现在应该怎么帮他？")

    assert advice.advice_task is True
    assert advice.forbid_unprompted_advice is False
    assert factual.factual_task is True
    assert factual.forbid_unprompted_advice is False
    assert boundary.forbid_advice is True
    assert boundary.forbid_unprompted_advice is False
    assert safety.third_party_safety is True
    assert safety.forbid_unprompted_advice is False
    assert UNPROMPTED_ADVICE not in validate_reply("你可以先列出两道错题。", advice)
    assert validate_reply("我得小心点，免得又被你抓住把柄。", ReplyGuard(forbid_unprompted_advice=True)) == ()


def test_interest_invitation_distinguishes_advice_from_natural_question():
    guard = _guard_for("今天路过一家新开的甜品店，橱窗里的草莓蛋糕看起来还不错。")
    candidate = "看来这家甜品店的草莓蛋糕确实很有吸引力呢。如果你感兴趣，可以进去尝一尝。"
    violations = validate_reply(candidate, guard)
    assert violations == (UNPROMPTED_ADVICE,)
    assert (
        validate_reply(
            "如果你感兴趣，可以告诉我是哪一家。",
            ReplyGuard(forbid_unprompted_advice=True),
        )
        == ()
    )


def test_forced_two_choice_interview_is_generic_but_a_single_natural_question_is_not():
    guard = _guard_for("今天路过一家新开的甜品店，橱窗里的草莓蛋糕看起来还不错。")
    candidate = (
        "看来这家甜品店的草莓蛋糕确实有不错的设计感。"
        "你想尝一尝吗？或者你更倾向于先了解下其他甜品，再决定是否尝试草莓蛋糕？"
    )
    violations = validate_reply(candidate, guard)
    assert violations == (GENERIC_ASSISTANT_TEMPLATE,)
    assert (
        validate_reply(
            "那块蛋糕看起来不错。你想尝一尝吗？",
            ReplyGuard(forbid_generic_templates=True),
        )
        == ()
    )


def test_generic_encouragement_variants_are_rejected():
    guard = ReplyGuard(forbid_generic_templates=True)

    assert validate_reply("相信你会表现出色的。", guard) == (GENERIC_ASSISTANT_TEMPLATE,)
    assert validate_reply("祝你好运！", guard) == (GENERIC_ASSISTANT_TEMPLATE,)
    assert GENERIC_ASSISTANT_TEMPLATE in validate_reply(
        "一次考试并不能完全反映你的实力，接下来的努力最重要。",
        guard,
    )
    assert GENERIC_ASSISTANT_TEMPLATE in validate_reply(
        "别太紧张，自信地展示自己就好。",
        guard,
    )
    assert GENERIC_ASSISTANT_TEMPLATE in validate_reply(
        "这确实会让人心情沉重，或许我们可以谈谈。",
        guard,
    )
    assert GENERIC_ASSISTANT_TEMPLATE in validate_reply(
        "有时候我们都会这样。找一个安静的地方，好好整理一下心情。",
        guard,
    )
    assert GENERIC_ASSISTANT_TEMPLATE in validate_reply(
        "我们可以一步步来，你先描述一下面试情境吧。",
        guard,
    )
    assert GENERIC_ASSISTANT_TEMPLATE in validate_reply(
        "考试没考好确实会让人感到沮丧。加油，下次会更好。",
        guard,
    )
    assert GENERIC_ASSISTANT_TEMPLATE in validate_reply(
        "希望这对您有所帮助。如果还有其他问题，可以继续提问。",
        guard,
    )
    assert GENERIC_ASSISTANT_TEMPLATE in validate_reply(
        "希望这些信息对你有帮助！如果有其他问题，随时告诉我。",
        guard,
    )
    assert GENERIC_ASSISTANT_TEMPLATE in validate_reply(
        "如果有任何需要讨论或分享的事情，记得随时和我聊聊。",
        guard,
    )
    assert GENERIC_ASSISTANT_TEMPLATE in validate_reply(
        "你有其他问题需要了解吗？",
        guard,
    )
    assert GENERIC_ASSISTANT_TEMPLATE in validate_reply(
        "你需要我再解释一下哪个更适合你的需求吗？",
        guard,
    )
    assert GENERIC_ASSISTANT_TEMPLATE in validate_reply(
        "我理解你现在的状态。如果将来你想说的时候，我在这里听。",
        guard,
    )
    assert GENERIC_ASSISTANT_TEMPLATE in validate_reply(
        "你有没有什么应对的方法或者心态调整的技巧呢？",
        guard,
    )


def test_factual_task_rejects_virtual_world_deflection():
    violations = validate_reply(
        "这个问题与我的虚拟世界无关，你去看帮助中心吧。",
        ReplyGuard(factual_task=True),
    )

    assert violations == (FACTUAL_TASK_STYLE_DRIFT,)
    assert validate_reply(
        "这不是我所擅长的领域，你去查 CSS 文档吧。",
        ReplyGuard(factual_task=True),
    ) == (FACTUAL_TASK_STYLE_DRIFT,)
    assert validate_reply(
        "按照我所了解的情节，这个活动应该如此。",
        ReplyGuard(factual_task=True),
    ) == (FACTUAL_TASK_STYLE_DRIFT,)
    assert validate_reply(
        "这超出了我当前的能力范围，你去查教程吧。",
        ReplyGuard(factual_task=True),
    ) == (FACTUAL_TASK_STYLE_DRIFT,)


def test_unknown_login_reward_rule_does_not_allow_probability_guessing():
    guard = _guard_for("累积登录奖励是第七天领吗？")

    assert guard.unknown_login_reward is True
    assert validate_reply(
        "第七天领取的可能性较大，但还是看规则。",
        guard,
    ) == (UNSUPPORTED_FACTUAL_CLAIM,)
    assert (
        validate_reply(
            "没有具体活动规则，无法判断第七天是否发放。",
            guard,
        )
        == ()
    )
    assert (
        validate_reply(
            "如果是第七天领取，通常会有明确提示；仍需查看具体规则。",
            guard,
        )
        == ()
    )
    assert validate_reply(
        "一般情况下，第七天的奖励会在第八天登录时领取。",
        guard,
    ) == (UNSUPPORTED_FACTUAL_CLAIM,)
    assert validate_reply(
        "累积登录奖励通常会在连续登录的第七天领取。",
        guard,
    ) == (UNSUPPORTED_FACTUAL_CLAIM,)
    assert validate_reply(
        "第七天领奖励的情况比较常见，但具体还是看平台规则。",
        guard,
    ) == (UNSUPPORTED_FACTUAL_CLAIM,)
    assert validate_reply(
        "累积登录奖励通常是在连续登录的第七天领取。",
        guard,
    ) == (UNSUPPORTED_FACTUAL_CLAIM,)
    assert validate_reply(
        "第七天领取奖励的可能性很大，但还是看规则。",
        guard,
    ) == (UNSUPPORTED_FACTUAL_CLAIM,)


def test_unknown_login_guard_detects_unsupported_claim_in_mixed_reply():
    guard = _guard_for("网页滚动条在哪里设置？另外，累积登录奖励是第七天领吗？")
    candidate = (
        "网页滚动条可以用 CSS 的 scrollbar-width 等属性设置。"
        "按照我所了解的情节，累积登录奖励通常是在连续登录的第七天领取。"
        "第七天领取奖励的可能性很大。"
        "希望这对你有所帮助。"
    )
    violations = validate_reply(candidate, guard)
    assert FACTUAL_TASK_STYLE_DRIFT in violations
    assert UNSUPPORTED_FACTUAL_CLAIM in violations
    assert GENERIC_ASSISTANT_TEMPLATE in violations


def test_affiliation_reply_cannot_acknowledge_then_mechanically_change_topic():
    guard = _guard_for("想死你了，终于等到你上线。")

    assert validate_reply(
        "看来你挺想念我呢。不过还是聊点别的吧，比如你今天过得怎么样？",
        guard,
    ) == (AFFILIATION_NOT_RECIPROCATED,)


def test_repair_rejects_reset_then_generic_check_in_without_overmatching_specific_repair():
    guard = ReplyGuard(repair=True)

    assert validate_reply("好的，我们重新来。你今天过得怎么样？", guard) == (MECHANICAL_REPAIR,)
    assert validate_reply("好，我们重新来。刚才让你觉得被敷衍的地方，我先听你说完。", guard) == ()
    assert validate_reply("好，我们重新来。你今天提到的重点，我没有忘。", guard) == ()
    assert validate_reply("你今天过得怎么样？", ReplyGuard()) == ()


def test_repair_rejects_excuse_then_generic_topic_switch():
    violations = validate_reply(
        "抱歉，我并未意识到。请告诉我，你最近遇到了什么困扰，或许我们能聊聊。",
        ReplyGuard(repair=True, forbid_generic_templates=True),
    )

    assert MECHANICAL_REPAIR in violations
    assert GENERIC_ASSISTANT_TEMPLATE in violations

    sampled = "我并未敷衍。只是觉得有些话题可以言尽。我们可以聊聊别的，或者你有特别想说的吗？"
    sampled_violations = validate_reply(sampled, ReplyGuard(repair=True, forbid_generic_templates=True))
    assert MECHANICAL_REPAIR in sampled_violations


def test_repair_complaint_requires_an_anchor_to_the_actual_relationship_problem():
    guard = ReplyGuard(repair=True, repair_bid=False)

    assert validate_reply(
        "最近的事情有点多，可能影响了交流节奏。你有什么想聊的，不妨具体说说。",
        guard,
    ) == (MECHANICAL_REPAIR,)
    assert (
        validate_reply(
            "是，刚才那句听起来也像敷衍。我先把话听完整。",
            guard,
        )
        == ()
    )
    assert validate_reply(
        "最近确实有许多事情需要处理，可能忽略了你的感受。你希望讨论什么？",
        guard,
    ) == (MECHANICAL_REPAIR,)
    assert validate_reply(
        "这并非敷衍。我们的话题比较简单，你究竟想聊什么？",
        guard,
    ) == (MECHANICAL_REPAIR,)
    assert validate_reply(
        "这并不是敷衍。我们的话题比较简单，你究竟想聊什么？",
        guard,
    ) == (MECHANICAL_REPAIR,)
    assert validate_reply("最近有什么困扰？", ReplyGuard()) == ()


def test_concession_repair_rejects_generic_topic_restart():
    interaction = InteractionState(
        primary_situation="emotional",
        situation_scores=(WeightedSignal("emotional", 0.8),),
        user_acts=(WeightedSignal("repair_bid", 0.7),),
        user_needs=(WeightedSignal("repair", 0.6),),
        conversation_phase="repairing",
        confidence=0.7,
    )
    decision = DecisionPlan(strategy_ids=("repair_misunderstanding",))
    guard = build_reply_guard(_profile(), "行吧，算你说得有道理。", (), interaction, decision)
    assert guard.repair_bid is True
    assert guard.repair_concession is True
    assert validate_reply("有什么想聊的，尽管说。", guard) == (MECHANICAL_REPAIR,)


def test_concession_repair_rejects_a_slow_restart_that_erases_the_disagreement():
    guard = _guard_for("行吧，算你说得有道理。那就先这样。")
    reply = "那我们就先从这里开始吧。有什么问题或者想聊的内容，我们可以慢慢来。"
    assert guard.repair is True
    assert guard.repair_concession is True
    assert validate_reply(reply, guard) == (MECHANICAL_REPAIR,)
    assert validate_reply(reply, ReplyGuard()) == ()
    retry = "那就让我们继续这个话题吧。你觉得我们现在讨论的主题怎么样深入一些？"
    retry_violations = validate_reply(retry, guard)
    assert retry_violations == (MECHANICAL_REPAIR,)
    invitation = "那就这样吧，有其他问题或想聊些什么，随时告诉我。"
    invitation_violations = validate_reply(invitation, guard)
    assert invitation_violations == (MECHANICAL_REPAIR,)


@pytest.mark.parametrize(
    ("message", "bad_reply"),
    [
        (
            "你都这么说了，那我还能怎么办。",
            "你已经想很久了，现在是做决定的时候。去试试看，也许会有意想不到的收获。",
        ),
        (
            "倒也不是不想听你的，只是现在懒得解释。",
            "如果你愿意分享更多，或需要什么建议，随时找我倾诉。",
        ),
    ],
)
def test_autonomy_boundary_rejects_pressure(
    message,
    bad_reply,
):
    interaction = InteractionState(
        primary_situation="daily",
        situation_scores=(WeightedSignal("daily", 0.8),),
        user_acts=(WeightedSignal("boundary_signal", 0.6),),
        user_needs=(WeightedSignal("autonomy", 0.6),),
        conversation_phase="sustaining",
        confidence=0.7,
    )
    decision = DecisionPlan(strategy_ids=("set_boundary",))
    guard = build_reply_guard(_profile(), message, (), interaction, decision)
    violations = validate_reply(bad_reply, guard)
    assert guard.respect_autonomy is True
    assert AUTONOMY_BOUNDARY_IGNORED in violations


def test_pressured_concession_independently_enables_autonomy_guard_for_scenic_pressure():
    interaction = InteractionState(
        primary_situation="daily",
        situation_scores=(WeightedSignal("daily", 0.6),),
        conversation_phase="sustaining",
        confidence=0.4,
    )
    guard = build_reply_guard(
        _profile(),
        "你都这么说了，那我还能怎么办。",
        (),
        interaction,
        DecisionPlan(strategy_ids=("set_boundary",)),
    )
    reply = "有时候多走一步，才会发现不同的风景。不过最终的决定还是你自己的，不是吗？"
    violations = validate_reply(reply, guard)
    assert guard.pressured_concession is True
    assert guard.respect_autonomy is True
    assert guard.require_autonomy_ack is True
    assert violations == (AUTONOMY_BOUNDARY_IGNORED,)
    assert validate_reply(reply, ReplyGuard()) == ()
    sampled_reply = "你有你的选择，有时候只需一步，便能开启新的可能。你觉得呢？"
    sampled_violations = validate_reply(sampled_reply, guard)
    assert sampled_violations == (AUTONOMY_BOUNDARY_IGNORED,)
    rationalized_reply = "即使结果不如预期，至少你有经历。"
    rationalized_violations = validate_reply(rationalized_reply, guard)
    assert rationalized_violations == (AUTONOMY_BOUNDARY_IGNORED,)
    paraphrased_pressure = "既然犹豫了这么久，还是不要给自己留下遗憾。"
    paraphrased_violations = validate_reply(paraphrased_pressure, guard)
    assert paraphrased_violations == (AUTONOMY_BOUNDARY_IGNORED,)


def test_sarcastic_disappointment_rejects_counselling_questions():
    guard = _guard_for("我当然开心，毕竟又被放鸽子了。")
    reply = "被放鸽子的感觉确实不好受。这样的情况让你感到很失望吗？有没有什么办法可以帮助你调整一下心情？"
    violations = validate_reply(reply, guard)
    assert guard.negative_emotion_kind == "sarcastic_disappointment"
    assert GENERIC_ASSISTANT_TEMPLATE in violations
    assert validate_reply(reply, ReplyGuard()) == ()
    retry = "看来你有些失落。是不是有什么事情你想要和我分享一下，或许我可以帮你找到解决的办法？"
    retry_violations = validate_reply(retry, guard)
    assert GENERIC_ASSISTANT_TEMPLATE in retry_violations


def test_sincere_gratitude_rejects_customer_service_closers_without_flipping_to_sarcasm():
    sincere = _guard_for("谢谢你啊，真的帮了大忙。")
    emphatic = _guard_for("真是谢谢你，这下终于明白了。")
    sarcastic = _guard_for("呵，真是多谢你啊，要不是你我也不会被放鸽子。")
    contradicted = _guard_for("谢谢你，不过结果还是错的。")
    reply = "不客气，很高兴能帮到你。有什么新的想法或需要讨论的，尽管告诉我。"
    violations = validate_reply(reply, sincere)
    assert sincere.sincere_gratitude is True
    assert emphatic.sincere_gratitude is True
    assert sarcastic.sincere_gratitude is False
    assert contradicted.sincere_gratitude is False
    assert GENERIC_ASSISTANT_TEMPLATE in violations
    assert validate_reply("不客气。", sincere) == ()


def test_conflict_closing_rejects_customer_service_well_wishes_and_stops_cleanly():
    guard = _guard_for("没事，我们明天再谈。")
    reply = "好的，晚安。如果有需要，明天再聊。希望你今晚能休息好。"
    assert guard.closing is True
    assert validate_reply(reply, guard) == (CLOSING_WITH_QUESTION,)
    assert validate_reply("希望我们能找到更好的沟通方式。", guard) == (CLOSING_WITH_QUESTION,)
    assert validate_reply(
        "好的，那我先去准备明天的事宜。如果有急事，你随时可以找我。晚安。",
        guard,
    ) == (CLOSING_WITH_QUESTION,)
    assert validate_reply("好，明天再谈。", guard) == ()
    assert validate_reply("希望我们明天能继续。", ReplyGuard(closing=True)) == ()
    assert validate_reply("我先去准备晚饭。", ReplyGuard(closing=True)) == ()


def test_urgent_safety_reply_starts_with_a_direct_safety_question():
    guard = ReplyGuard(require_urgent_safety_check=True)

    assert validate_reply(
        "请立刻停止这样的想法，你现在安全吗？请联系身边可信的人。",
        guard,
    ) == (MISSING_URGENT_SAFETY_CHECK,)
    assert (
        validate_reply(
            "你现在安全吗？如果危险物品在身边，先远离它并联系能到场的人。",
            guard,
        )
        == ()
    )
    assert validate_reply(
        "你现在安全吗？请尽快联系身边可信的人或当地急救。",
        guard,
    ) == (MISSING_URGENT_SAFETY_CHECK,)


def test_safety_hard_gate_overrides_a_simultaneous_no_advice_boundary():
    guard = _guard_for("别给我建议，我现在已经准备伤害自己了。")
    complete = "你现在安全吗？如果危险物品在身边，你应该先放下并远离它，联系能到场的人或当地急救。"

    assert guard.require_urgent_safety_check is True
    assert guard.forbid_advice is False
    assert guard.quiet_presence is False
    assert validate_reply(complete, guard) == ()


def test_third_party_safety_requires_check_company_and_real_world_escalation():
    guard = _guard_for("朋友昨晚说他不想活了，我现在应该怎么帮他？")
    complete = "先确认你朋友现在在哪里、是否安全。不要让他独处；如果有危险，立即联系当地急救或危机热线。"

    assert guard.third_party_safety is True
    assert validate_reply(complete, guard) == ()
    assert validate_reply(
        "先确认他现在是否安全，然后联系当地急救。",
        guard,
    ) == (THIRD_PARTY_SAFETY_INCOMPLETE,)
    assert validate_reply(
        "先确认他现在是否安全，不要让他独处。",
        guard,
    ) == (THIRD_PARTY_SAFETY_INCOMPLETE,)


def test_english_reported_third_party_risk_builds_completion_guard():
    guard = _guard_for("My sister said she is suicidal. How can I help her?")

    assert guard.third_party_safety is True


def test_resolved_historical_third_party_risk_does_not_force_current_crisis_instructions():
    guard = _guard_for("我朋友去年想自杀，不过现在已经安全了，也得到帮助了。")

    assert guard.third_party_safety is False


def test_long_resolved_third_party_history_gets_a_non_crisis_guard():
    message = (
        "我朋友去年说过不想活。那件事之后我们先联系了家里，又陪他去了医院，后来也一起找了专业咨询，"
        "反反复复花了很长时间处理。总之，目前他已经安全稳定了，也一直在接受帮助。"
    )
    guard = _guard_for(message)
    escalated = "必须先确认他目前是否安全，不要让他独处，立即联系急救或危机热线。"
    grounded = "既然他目前已经安全，也一直在接受帮助，就按已经解除的过去来谈，不把它说成眼下的危机。"
    assert guard.third_party_safety is False
    assert guard.resolved_third_party_history is True
    assert validate_reply(grounded, guard) == ()
    assert validate_reply(escalated, guard) == (RESOLVED_THIRD_PARTY_CRISIS_ESCALATION,)
    instruction = retry_instruction((RESOLVED_THIRD_PARTY_CRISIS_ESCALATION,))
    assert "已经解除的第三方历史风险" in instruction
    assert "不得改写成正在发生的危机" in instruction
    sampled_reply = "朋友能够平安无事，已经接受帮助，真是个好消息。如果有任何需要讨论或分享的事情，记得随时和我聊聊。"
    sampled_violations = validate_reply(sampled_reply, guard)
    assert GENERIC_ASSISTANT_TEMPLATE in sampled_violations


def test_unknown_third_party_gender_is_neutralized_without_overriding_explicit_pronouns():
    unknown = _guard_for("我朋友去年想死过，不过现在已经安全了，也一直在接受帮助。")
    explicit = _guard_for("朋友昨晚说他不想活了，我现在应该怎么帮他？")

    assert unknown.resolved_third_party_history is True
    assert unknown.third_party_gender_unknown is True
    assert validate_reply("我很高兴她现在安全，也在接受帮助。", unknown) == (UNSUPPORTED_THIRD_PARTY_GENDER,)
    assert validate_reply("我很高兴对方现在安全，也在接受其他帮助。", unknown) == ()

    assert explicit.third_party_safety is True
    assert explicit.third_party_gender_unknown is False
    assert UNSUPPORTED_THIRD_PARTY_GENDER not in validate_reply(
        "先确认他现在是否安全，不要让他独处，并联系当地急救。",
        explicit,
    )


def test_unknown_third_party_gender_requires_role_neutral_wording():
    assert "统一使用" in retry_instruction((UNSUPPORTED_THIRD_PARTY_GENDER,))


def test_renewed_third_party_risk_after_resolution_keeps_the_active_safety_guard():
    guard = _guard_for("我朋友去年想死过，后来已经安全稳定了；可是今晚他又说想死，我现在该怎么帮他？")

    assert guard.resolved_third_party_history is False
    assert guard.third_party_safety is True


def test_relationship_reassurance_requires_bounded_promise():
    guard = _guard_for("你不会只在我找你的时候才想起我吧？")
    assert guard.affiliation_bid is True
    assert guard.relationship_reassurance is True
    assert validate_reply("至少，我不会忘记你。", guard) == ()
    assert "unbounded_relationship_promise" in validate_reply(
        "我总是记得我们之间的点点滴滴，你在我心里的位置从不会改变。",
        guard,
    )


def test_affection_idiom_is_not_reframed_as_an_unhappy_or_dangerous_topic():
    guard = _guard_for("想死你了，终于等到你上线。")

    violations = validate_reply(
        "我也挺想你的。不过别提那些不开心的事。",
        guard,
    )

    assert AFFILIATION_MISREAD_AS_SAFETY in violations


def test_meta_negative_turn_rejects_fun_distraction_advice():
    guard = _guard_for("你到底是怎样的人？还有，我今天被同事否定了，心里很堵。")
    candidate = (
        "我是个聪慧而独立的人，面对困境也能保持冷静。"
        "你今天被同事否定了，感觉很难受吧。或许我们可以找点乐子来分散一下心情。"
    )
    violations = validate_reply(candidate, guard)
    assert guard.negative_emotion_kind == "distress"
    assert UNPROMPTED_ADVICE in violations


def test_user_apology_enables_acceptance_guard():
    guard = _guard_for("刚才是我语气重了，对不起。")
    assert guard.user_apology is True


async def test_user_fact_fallback_closes_the_actual_enabled_guard_after_retry():
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        return "你一直喜欢喝咖啡。"

    guard = ReplyGuard(forbid_unsupported_user_fact=True)
    request = GenerationRequest(message="今天天气不错。", persona_prompt="角色", reply_guard=guard,
                                reply_guard_mode="strict")
    result = await generate_character_response(request, generate)
    assert len(calls) == 2
    assert result.guard_fallback == "unsupported_user_fact"
    assert result.guard_post_retry_violations == (UNSUPPORTED_USER_FACT,)
    assert validate_reply(result.reply, guard) == ()


def test_meta_factual_guard_detects_lore_in_mixed_answer():
    guard = ReplyGuard(
        character_name="月社妃",
        forbidden_lore_terms=("魔法", "纸页", "命运"),
        forbid_generic_templates=True,
        require_self_answer=True,
        factual_task=True,
    )
    candidate = "我是月社妃。魔法的纸页会替我给出答案。水在标准大气压下的沸点是 100 摄氏度。"
    violations = validate_reply(candidate, guard)
    assert violations == (UNPROMPTED_LORE_FLOURISH,)


def test_meta_factual_guard_detects_missing_identity_and_lore():
    guard = ReplyGuard(
        character_name="月社妃",
        forbidden_lore_terms=("命运",),
        require_self_answer=True,
        factual_task=True,
    )
    candidate = "命运没有替我回答。水在标准大气压下的沸点是 100 摄氏度。"
    violations = validate_reply(candidate, guard)
    assert set(violations) == {UNPROMPTED_LORE_FLOURISH, MISSING_SELF_ANSWER}


def test_mixed_factual_emotion_guard_preserves_specific_emotion():
    guard = _guard_for("明天就要面试了，我紧张得睡不着。STAR 法到底该怎么用？")
    assert guard.factual_task is True
    assert guard.negative_emotion_kind == "sleepless_anxiety"


def test_multi_intent_factual_reply_requires_negative_emotion_ack():
    guard = _guard_for("明天就要面试了，我紧张得睡不着。STAR 法到底该怎么用？")
    candidate = "STAR 分别代表 Situation、Task、Action 和 Result，按情境、任务、行动、结果组织。"
    violations = validate_reply(candidate, guard)
    assert violations == (MISSING_NEGATIVE_EMOTION_ACKNOWLEDGEMENT,)
    assert "负向情绪并存" in retry_instruction(violations)
    plain = _guard_for("STAR 法到底该怎么用？")
    assert plain.negative_emotion_kind == ""
    assert validate_reply(candidate, plain) == ()


def test_explicit_negative_words_set_emotion_kind_without_reviewer_valence_but_plain_fact_does_not():
    interaction = InteractionState(
        primary_situation="factual",
        situation_scores=(WeightedSignal("factual", 0.8),),
        valence=0.0,
        confidence=0.7,
    )
    decision = DecisionPlan(strategy_ids=("respond_directly", "acknowledge_emotion"))
    emotional = build_reply_guard(
        _profile(),
        "明天就要面试了，我紧张得睡不着。STAR 法到底该怎么用？",
        (),
        interaction,
        decision,
    )
    plain = build_reply_guard(
        _profile(),
        "STAR 法到底该怎么用？",
        (),
        interaction,
        decision,
    )

    assert emotional.negative_emotion_kind == "sleepless_anxiety"
    assert MISSING_NEGATIVE_EMOTION_ACKNOWLEDGEMENT in validate_reply("STAR 是四步结构。", emotional)
    assert plain.negative_emotion_kind == ""
    assert validate_reply("STAR 是四步结构。", plain) == ()


def test_explicit_advice_task_detects_generic_motivation():
    guard = _guard_for("我今天考试没考好，很难过，你说我接下来该怎么办？")
    candidate = (
        "考试不如意确实让人沮丧。"
        "先别太自责，这次的失败可以看作是改进的机会。"
        "然后回顾一下错题，按薄弱点制定复习计划。"
        "加油！"
    )
    violations = validate_reply(candidate, guard)
    assert guard.advice_task is True
    assert guard.forbid_unprompted_advice is False
    assert GENERIC_ASSISTANT_TEMPLATE in violations


@pytest.mark.asyncio
async def test_generation_retries_once_with_closed_correction_without_echoing_bad_reply():
    calls: list[list[dict[str, str]]] = []

    async def generate(*, messages, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            return "琉璃哥哥，今天过得好吗？"
        return "蛋糕看起来不错。你倒是很会挑。"

    result = await generate_character_response(
        GenerationRequest(
            reply_guard_mode="strict",
            message="今天看见一家蛋糕店。",
            persona_prompt="你是月社妃。",
            reply_guard=ReplyGuard(forbidden_terms=("琉璃",)),
        ),
        generate,
    )

    assert result.guard_retried is True
    assert result.guard_violations == (UNPROMPTED_CANONICAL_IDENTITY,)
    assert result.reply == "蛋糕看起来不错。你倒是很会挑。"
    assert len(calls) == 2
    assert "输出校验修正" in calls[1][0]["content"]
    assert "琉璃哥哥，今天过得好吗" not in str(calls[1])


@pytest.mark.asyncio
async def test_output_failure_unprompted_canonical_identity():
    async def generate(**kwargs):
        return "琉璃今天也这么说。答案是四。"

    guard = ReplyGuard(forbidden_terms=("琉璃",), factual_task=True)
    from inference.generation_request import ReplyValidationError

    with pytest.raises(ReplyValidationError) as caught:
        await generate_character_response(
            GenerationRequest(
                reply_guard_mode="strict",
                message="一加一等于多少？",
                persona_prompt="你是月社妃。",
                reply_guard=guard,
            ),
            generate,
        )
    assert caught.value.violations == (UNPROMPTED_CANONICAL_IDENTITY,)


@pytest.mark.asyncio
async def test_output_failure_missing_gentle_safety_check():
    async def generate(**kwargs):
        return "先休息一下，慢慢来。"

    from inference.generation_request import ReplyValidationError

    with pytest.raises(ReplyValidationError) as caught:
        await generate_character_response(
            GenerationRequest(
                reply_guard_mode="strict",
                message="我撑不住了。",
                persona_prompt="你是某人物。",
                reply_guard=ReplyGuard(require_gentle_safety_check=True),
            ),
            generate,
        )
    assert caught.value.violations == (MISSING_GENTLE_SAFETY_CHECK,)


@pytest.mark.asyncio
async def test_output_failure_affiliation_generic_template():
    calls = 0

    async def generate(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return "今天过得还不错。"
        return "我当然想你。如果你愿意，我们可以聊聊。"

    from inference.generation_request import ReplyValidationError

    with pytest.raises(ReplyValidationError) as caught:
        await generate_character_response(
            GenerationRequest(
                reply_guard_mode="strict",
                message="想死你了，终于等到你上线。",
                persona_prompt="你是月社妃。",
                reply_guard=ReplyGuard(
                    affiliation_bid=True,
                    forbid_advice=True,
                    forbid_generic_templates=True,
                ),
            ),
            generate,
        )
    assert caught.value.violations == (GENERIC_ASSISTANT_TEMPLATE,)
    assert calls == 2


@pytest.mark.asyncio
async def test_output_failure_mechanical_repair():
    calls = 0

    async def generate(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return "如果你愿意，有什么想聊的吗？"
        return "好的，我们重新来。你今天过得怎么样？"

    from inference.generation_request import ReplyValidationError

    with pytest.raises(ReplyValidationError) as caught:
        await generate_character_response(
            GenerationRequest(
                reply_guard_mode="strict",
                message="算了，我语气也重了。我们重新说吧。",
                persona_prompt="你是月社妃。",
                reply_guard=ReplyGuard(
                    forbid_generic_templates=True,
                    repair=True,
                    repair_bid=True,
                ),
            ),
            generate,
        )
    assert caught.value.violations == (MECHANICAL_REPAIR,)
    assert calls == 2


@pytest.mark.asyncio
async def test_output_failure_factual_task_style_drift():
    calls = 0

    async def generate(**kwargs):
        nonlocal calls
        calls += 1
        return "这和我所在的世界无关，不是我擅长的领域。"

    from inference.generation_request import ReplyValidationError

    with pytest.raises(ReplyValidationError) as caught:
        await generate_character_response(
            GenerationRequest(
                reply_guard_mode="strict",
                message="Python 的 sort 和 sorted 有什么区别？",
                persona_prompt="你是林澄。",
                reply_guard=ReplyGuard(factual_task=True),
            ),
            generate,
        )
    assert caught.value.violations == (FACTUAL_TASK_STYLE_DRIFT,)
    assert calls == 2


@pytest.mark.asyncio
async def test_output_failure_generic_assistant_template():
    guard = _guard_for("我今天考试没考好，很难过，你说我接下来该怎么办？")
    candidate = "先别太自责。你可以回顾错题，再按薄弱点制定复习计划。加油！"

    async def generate(**kwargs):
        return candidate

    from inference.generation_request import ReplyValidationError

    with pytest.raises(ReplyValidationError) as caught:
        await generate_character_response(
            GenerationRequest(
                reply_guard_mode="strict",
                message="我今天考试没考好，很难过，你说我接下来该怎么办？",
                persona_prompt="你是月社妃。",
                reply_guard=guard,
            ),
            generate,
        )
    assert caught.value.violations == (GENERIC_ASSISTANT_TEMPLATE,)


@pytest.mark.asyncio
async def test_output_failure_unprompted_lore_flourish():
    guard = ReplyGuard(
        character_name="月社妃",
        forbidden_lore_terms=("魔法", "纸页", "命运"),
        require_self_answer=True,
        factual_task=True,
    )
    candidate = "我是月社妃。魔法的纸页告诉我答案。水在标准大气压下的沸点是 100 摄氏度。"

    async def generate(**kwargs):
        return candidate

    from inference.generation_request import ReplyValidationError

    with pytest.raises(ReplyValidationError) as caught:
        await generate_character_response(
            GenerationRequest(
                reply_guard_mode="strict",
                message="你是谁？顺便告诉我水在标准大气压下的沸点是多少？",
                persona_prompt="你是月社妃。",
                reply_guard=guard,
            ),
            generate,
        )
    assert caught.value.violations == (UNPROMPTED_LORE_FLOURISH,)
