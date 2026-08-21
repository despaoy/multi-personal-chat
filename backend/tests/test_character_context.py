"""角色上下文模块（character）的单元测试。

只验证第一步的约定：
- 用户范围隔离（私聊/群聊、平台、群成员）；
- 人物画像进入 system_context，用户记忆只进入 reference_context；
- 记忆数量与总长度限制；
- 编译过程不修改原始 CharacterContext。
"""

from __future__ import annotations

import dataclasses

import pytest

from character import (
    MAX_MEMORY_TOTAL_CHARS,
    MEMORY_REFERENCE_DISCLAIMER,
    CharacterContext,
    CharacterProfile,
    DecisionPlan,
    MemoryItem,
    RelationshipState,
    SituationState,
    UserScope,
    build_user_scope,
    compile_character_context,
    compile_dynamic_context,
    compile_profile_context,
    compile_reference_context,
)


def _profile() -> CharacterProfile:
    return CharacterProfile(
        character_id="tsukiyashiro_kisaki",
        display_name="月社妃",
        identity="《纸上的魔法使》中的人物",
        traits=("毒舌", "自尊心强"),
        values=("秩序", "公平竞争"),
        speaking_style=("句子简短", "常带讥讽"),
        boundaries=("不透露真实身份", "不讨论敏感话题"),
    )


def _scope(
    sender: str = "10001",
    conversation: str = "20001",
    conv_type: str = "group",
    platform: str = "qq",
) -> UserScope:
    return build_user_scope(
        platform=platform,
        adapter="onebot",
        sender_id=sender,
        conversation_id=conversation,
        conversation_type=conv_type,
    )


def _context(
    memories: tuple[MemoryItem, ...] = (),
    sender: str = "10001",
    conversation: str = "20001",
    conv_type: str = "group",
) -> CharacterContext:
    return CharacterContext(
        profile=_profile(),
        user_scope=_scope(sender=sender, conversation=conversation, conv_type=conv_type),
        relationship=RelationshipState(
            stage="familiar",
            preferred_address="同学",
            summary="同班同学，经常斗嘴",
        ),
        situation=SituationState(topic="期中考试", emotion_hint="不耐烦", response_goal="简短回应"),
        memories=memories,
        decision=DecisionPlan(intent="回应提问", tone="冷淡", action="简短回答", avoid="长篇大论"),
    )


def _memory(index: int, content: str, memory_type: str = "user_fact") -> MemoryItem:
    return MemoryItem(
        memory_id=f"mem_{index:03d}",
        memory_type=memory_type,  # type: ignore[arg-type]
        content=content,
        importance=0.5,
    )


# ============================================
# 用户范围隔离
# ============================================


def test_private_and_group_scopes_differ():
    private_scope = build_user_scope(
        platform="qq", adapter="onebot", sender_id="10001",
        conversation_id="", conversation_type="private",
    )
    group_scope = build_user_scope(
        platform="qq", adapter="onebot", sender_id="10001",
        conversation_id="20001", conversation_type="group",
    )
    assert private_scope.memory_scope_key != group_scope.memory_scope_key
    # 元组结构：私聊不含会话ID，群聊含群ID
    assert private_scope.memory_scope_key == ("qq", "onebot", "private", "10001")
    assert group_scope.memory_scope_key == ("qq", "onebot", "group", "20001", "10001")
    # 私聊缺会话ID时按用户ID兜底
    assert private_scope.conversation_id == "10001"


def test_qq_and_wechat_scopes_differ():
    qq_scope = build_user_scope(
        platform="QQ", adapter="onebot", sender_id="10001",
        conversation_id="", conversation_type="private",
    )
    wechat_scope = build_user_scope(
        platform="WeChat", adapter="wechat-pad", sender_id="10001",
        conversation_id="", conversation_type="private",
    )
    assert qq_scope.platform == "qq"
    assert wechat_scope.platform == "wechat"
    assert qq_scope.memory_scope_key != wechat_scope.memory_scope_key


def test_same_platform_user_different_adapter_scopes_differ():
    """同一平台不同适配器（如 NapCat 与 OneBot）的用户ID可能互不通用，必须隔离。"""
    onebot_scope = build_user_scope(
        platform="qq", adapter="OneBot", sender_id="10001",
        conversation_id="", conversation_type="private",
    )
    napcat_scope = build_user_scope(
        platform="qq", adapter="NapCat", sender_id="10001",
        conversation_id="", conversation_type="private",
    )
    assert onebot_scope.adapter == "onebot"
    assert napcat_scope.adapter == "napcat"
    assert onebot_scope.memory_scope_key != napcat_scope.memory_scope_key


def test_private_and_group_with_similar_fields_do_not_collide():
    """私聊用户ID本身含分隔符（如复合ID）时，不得与群聊范围碰撞。"""
    private_scope = build_user_scope(
        platform="qq", adapter="onebot", sender_id="20001:10001",
        conversation_id="", conversation_type="private",
    )
    group_scope = build_user_scope(
        platform="qq", adapter="onebot", sender_id="10001",
        conversation_id="20001", conversation_type="group",
    )
    assert private_scope.memory_scope_key != group_scope.memory_scope_key
    assert private_scope.memory_scope_key == ("qq", "onebot", "private", "20001:10001")
    assert group_scope.memory_scope_key == ("qq", "onebot", "group", "20001", "10001")


def test_channel_scope_includes_channel_id():
    channel_scope = build_user_scope(
        platform="telegram", adapter="telegram", sender_id="10001",
        conversation_id="ch_90001", conversation_type="channel",
    )
    assert channel_scope.memory_scope_key == (
        "telegram", "telegram", "channel", "ch_90001", "10001",
    )


def test_channel_scope_differs_from_group_scope():
    """相同会话ID下，频道与群聊必须隔离。"""
    channel_scope = build_user_scope(
        platform="telegram", adapter="telegram", sender_id="10001",
        conversation_id="90001", conversation_type="channel",
    )
    group_scope = build_user_scope(
        platform="telegram", adapter="telegram", sender_id="10001",
        conversation_id="90001", conversation_type="group",
    )
    assert channel_scope.memory_scope_key != group_scope.memory_scope_key


def test_different_group_members_have_different_scopes():
    member_a = _scope(sender="10001", conversation="20001", conv_type="group")
    member_b = _scope(sender="10002", conversation="20001", conv_type="group")
    assert member_a.memory_scope_key != member_b.memory_scope_key


def test_empty_sender_is_rejected():
    with pytest.raises(ValueError):
        build_user_scope(
            platform="qq", adapter="onebot", sender_id="  ",
            conversation_id="20001", conversation_type="group",
        )


def test_group_without_conversation_id_is_rejected():
    with pytest.raises(ValueError):
        build_user_scope(
            platform="qq", adapter="onebot", sender_id="10001",
            conversation_id="", conversation_type="group",
        )


def test_channel_without_conversation_id_is_rejected():
    with pytest.raises(ValueError):
        build_user_scope(
            platform="telegram", adapter="telegram", sender_id="10001",
            conversation_id="", conversation_type="channel",
        )


# ============================================
# 编译结果的内容分配
# ============================================


def test_profile_enters_profile_context():
    compiled = compile_character_context(_context())
    profile = compiled.profile_context
    assert "月社妃" in profile
    assert "《纸上的魔法使》中的人物" in profile
    assert "毒舌" in profile
    assert "公平竞争" in profile
    assert "句子简短" in profile
    assert "不透露真实身份" in profile
    # 人物画像不包含关系、情景和行为决策（属于 dynamic_context）
    assert "familiar" not in profile
    assert "期中考试" not in profile
    assert "回应提问" not in profile


def test_dynamic_context_holds_relationship_situation_decision():
    compiled = compile_character_context(_context())
    dynamic = compiled.dynamic_context
    # 关系、情景、行为决策进入 dynamic_context
    assert "familiar" in dynamic
    assert "期中考试" in dynamic
    assert "回应提问" in dynamic
    # 人物画像内容不进入 dynamic_context
    assert "毒舌" not in dynamic
    assert "不透露真实身份" not in dynamic
    assert "月社妃" not in dynamic


def test_memories_only_enter_reference_context():
    memories = (
        _memory(1, "用户喜欢喝咖啡"),
        _memory(2, "上周一起参加了学园祭", memory_type="shared_event"),
    )
    context = _context(memories=memories)
    compiled = compile_character_context(context)

    assert "用户喜欢喝咖啡" in compiled.reference_context
    assert "上周一起参加了学园祭" in compiled.reference_context
    assert MEMORY_REFERENCE_DISCLAIMER in compiled.reference_context
    assert compiled.reference_context.startswith(MEMORY_REFERENCE_DISCLAIMER)

    # 用户记忆和声明不得进入任何系统提示词区域
    for trusted_area in (compiled.profile_context, compiled.dynamic_context):
        assert "用户喜欢喝咖啡" not in trusted_area
        assert "上周一起参加了学园祭" not in trusted_area
        assert MEMORY_REFERENCE_DISCLAIMER not in trusted_area
    assert compiled.used_memory_ids == ("mem_001", "mem_002")


def test_preferred_address_only_enters_reference_context():
    """用户自述称呼偏好属于用户控制内容，只进不可信参考区。"""
    # 使用唯一称呼值，避免与关系摘要中的常见词撞车
    context = dataclasses.replace(
        _context(),
        relationship=RelationshipState(
            stage="familiar",
            preferred_address="小刺猬骑士",
            summary="同班同学，经常斗嘴",
        ),
    )
    compiled = compile_character_context(context)

    # 称呼出现在参考区，并标注为用户自述偏好
    assert "小刺猬骑士" in compiled.reference_context
    assert "用户希望被称为" in compiled.reference_context

    # 称呼绝不进入系统提示词区域（画像区、动态区）
    for trusted_area in (compiled.profile_context, compiled.dynamic_context):
        assert "小刺猬骑士" not in trusted_area
        assert "用户希望被称为" not in trusted_area


def test_no_memories_yields_empty_reference_context():
    # 无记忆且无称呼偏好 → 参考区为空
    context = dataclasses.replace(
        _context(), relationship=RelationshipState(stage="familiar")
    )
    compiled = compile_character_context(context)
    assert compiled.reference_context == ""
    assert compiled.used_memory_ids == ()


# ============================================
# 画像与动态上下文长度限制
# ============================================


def _section_lines(context_text: str, title: str) -> list[str]:
    """从编译结果中提取【title】段的条目行（仅测试用）。"""
    lines = context_text.splitlines()
    start = lines.index(f"【{title}】")
    result = []
    for line in lines[start + 1 :]:
        if line.startswith("【") or not line.strip():
            break
        result.append(line)
    return result


def test_profile_lists_are_limited_to_eight_items():
    profile = dataclasses.replace(
        _profile(), traits=tuple(f"性格点{i:02d}" for i in range(1, 13))
    )
    context = dataclasses.replace(_context(), profile=profile)
    compiled = compile_character_context(context)
    trait_lines = _section_lines(compiled.profile_context, "人物核心性格")
    assert len(trait_lines) == 8
    for i in range(1, 9):
        assert f"性格点{i:02d}" in compiled.profile_context
    for i in range(9, 13):
        assert f"性格点{i:02d}" not in compiled.profile_context


def test_profile_items_are_truncated():
    long_trait = "超长性格描述" * 30  # 180 字符 > 150
    profile = dataclasses.replace(_profile(), traits=(long_trait,))
    context = dataclasses.replace(_context(), profile=profile)
    compiled = compile_character_context(context)
    trait_lines = _section_lines(compiled.profile_context, "人物核心性格")
    assert len(trait_lines) == 1
    # "- " + 149字符 + "…" = 152
    assert len(trait_lines[0]) <= 152
    assert trait_lines[0].endswith("…")


def test_relationship_summary_is_truncated():
    long_summary = "关系摘要内容" * 80  # 400 字符 > 300
    relationship = RelationshipState(stage="familiar", summary=long_summary)
    context = dataclasses.replace(_context(), relationship=relationship)
    compiled = compile_character_context(context)
    for line in _section_lines(compiled.dynamic_context, "当前关系"):
        if line.startswith("关系摘要："):
            assert len(line) <= len("关系摘要：") + 300
            assert line.endswith("…")


def test_situation_and_decision_fields_are_truncated():
    long_text = "情景描述" * 80  # 320 字符 > 200
    situation = SituationState(
        topic=long_text, emotion_hint=long_text, response_goal=long_text
    )
    decision = DecisionPlan(
        intent=long_text, tone=long_text, action=long_text, avoid=long_text
    )
    context = dataclasses.replace(
        _context(), situation=situation, decision=decision
    )
    compiled = compile_character_context(context)
    for title in ("当前情景", "本轮行为决策"):
        lines = _section_lines(compiled.dynamic_context, title)
        assert lines
        for line in lines:
            _, _, value = line.partition("：")
            assert len(value) <= 200
            assert value.endswith("…")


def test_boundaries_and_decision_survive_oversized_descriptions():
    """无关描述（性格/价值观/语言习惯/关系摘要/情景）超限截断时，
    行为边界和本轮决策必须完整保留，不被挤掉。"""
    long_text = "无关描述" * 60  # 240 字符
    profile = dataclasses.replace(
        _profile(),
        traits=tuple(f"性格点{i:02d}{long_text}" for i in range(1, 13)),
        values=tuple(f"价值观{i:02d}{long_text}" for i in range(1, 13)),
        speaking_style=tuple(f"语言习惯{i:02d}{long_text}" for i in range(1, 13)),
    )
    context = dataclasses.replace(
        _context(),
        profile=profile,
        relationship=RelationshipState(stage="close", summary=long_text * 2),
        situation=SituationState(
            topic=long_text, emotion_hint=long_text, response_goal=long_text
        ),
    )
    compiled = compile_character_context(context)
    profile_area = compiled.profile_context
    dynamic_area = compiled.dynamic_context
    # 行为边界完整保留在画像区
    assert "不透露真实身份" in profile_area
    assert "不讨论敏感话题" in profile_area
    # 本轮行为决策完整保留在动态区
    assert "回应提问" in dynamic_area
    assert "简短回答" in dynamic_area
    assert "长篇大论" in dynamic_area
    # 边界和决策段没有被截断标记污染
    for line in _section_lines(profile_area, "人物行为边界") + _section_lines(
        dynamic_area, "本轮行为决策"
    ):
        assert not line.endswith("…")


# ============================================
# 记忆效率限制
# ============================================


def test_at_most_five_memories_are_used():
    memories = tuple(_memory(i, f"记忆内容{i}") for i in range(1, 8))
    compiled = compile_character_context(_context(memories=memories))
    assert len(compiled.used_memory_ids) == 5
    # 保留调用方提供的相关度顺序（前5条）
    assert compiled.used_memory_ids == ("mem_001", "mem_002", "mem_003", "mem_004", "mem_005")


def test_memory_total_length_is_limited():
    # 5条长记忆，单条均低于单条上限，但总长度必然超过1000
    # （不带称呼偏好，专注验证记忆本身的预算）
    long_content = "很长的记忆" * 50  # 250字符
    memories = tuple(_memory(i, long_content) for i in range(1, 6))
    context = dataclasses.replace(
        _context(memories=memories),
        relationship=RelationshipState(stage="familiar"),
    )
    compiled = compile_character_context(context)

    memory_body = compiled.reference_context[len(MEMORY_REFERENCE_DISCLAIMER) + 1 :]
    assert len(memory_body) <= MAX_MEMORY_TOTAL_CHARS
    # 超预算导致部分记忆被放弃或截断
    assert len(compiled.used_memory_ids) < 5


def test_single_memory_is_truncated():
    memories = (_memory(1, "超长记忆" * 100),)  # 400字符，超过单条上限
    context = dataclasses.replace(
        _context(memories=memories),
        relationship=RelationshipState(stage="familiar"),
    )
    compiled = compile_character_context(context)
    assert len(compiled.used_memory_ids) == 1
    memory_body = compiled.reference_context[len(MEMORY_REFERENCE_DISCLAIMER) + 1 :]
    # "- "前缀 + 截断内容 + 省略号
    assert len(memory_body) <= 301


# ============================================
# 不可变性
# ============================================


def test_original_context_is_not_modified():
    memories = (
        _memory(1, "用户喜欢喝咖啡"),
        _memory(2, "超长记忆" * 100),
    )
    context = _context(memories=memories)
    snapshot = dataclasses.asdict(context)

    compile_character_context(context)

    assert dataclasses.asdict(context) == snapshot


def test_context_objects_are_frozen():
    context = _context()
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.situation.topic = "被篡改的话题"  # type: ignore[misc]


# ============================================
# 三个编译函数可独立调用
# ============================================


def test_compile_functions_are_independently_callable():
    context = _context(
        memories=(_memory(1, "用户喜欢喝咖啡"),)
    )

    profile_ctx = compile_profile_context(context.profile)
    dynamic_ctx = compile_dynamic_context(
        context.relationship, context.situation, context.decision
    )
    reference_ctx, used_ids = compile_reference_context(
        context.memories,
        preferred_address=context.relationship.preferred_address,
    )

    assert "月社妃" in profile_ctx
    assert "familiar" in dynamic_ctx
    assert "用户喜欢喝咖啡" in reference_ctx
    assert used_ids == ("mem_001",)

    # 与组合入口的结果一致
    compiled = compile_character_context(context)
    assert compiled.profile_context == profile_ctx
    assert compiled.dynamic_context == dynamic_ctx
    assert compiled.reference_context == reference_ctx
    assert compiled.used_memory_ids == used_ids


# ============================================
# 行为决策：关系阶段策略
# ============================================


def test_stage_action_modifies_decision():
    """关系阶段的行动策略必须叠加进决策 action，而不是只改语气。"""
    from character.decision_policy import DecisionPolicy

    policy = DecisionPolicy()
    stranger = policy.decide(_profile(), RelationshipState(stage="stranger"), "daily")
    close = policy.decide(_profile(), RelationshipState(stage="close"), "daily")

    # 陌生阶段：行动包含"点到为止"的约束
    assert "不主动打听对方私事" in stranger.action
    # 亲近阶段：行动包含主动关心与使用记忆
    assert "主动关心近况" in close.action
    # 不同阶段的行动策略不同（策略表真正生效）
    assert stranger.action != close.action
    # 情景基础行动保留
    assert "接住话题" in stranger.action


def test_safety_situation_skips_stage_modification():
    """安全情景是硬性规则：决策不做任何关系化修饰。"""
    from character.decision_policy import DecisionPolicy
    from character.situation_analyzer import SITUATION_SAFETY

    policy = DecisionPolicy()
    base = policy.decide(_profile(), RelationshipState(stage="stranger"), SITUATION_SAFETY)
    close = policy.decide(_profile(), RelationshipState(stage="close"), SITUATION_SAFETY)

    # 安全情景下各阶段决策一致，且不含阶段修饰
    assert base == close
    assert "不主动打听对方私事" not in base.tone + base.action + base.avoid
    assert "主动关心近况" not in close.tone + close.action + close.avoid


# ============================================
# 长期记忆检索：结构化意图兜底
# ============================================


class _FakeMemoryRepo:
    """只实现 CharacterMemoryService 依赖的 list_memory_records。"""

    def __init__(self, records: list[dict]) -> None:
        self._records = records

    async def list_memory_records(self, character_id, user_scope, limit=30):
        return self._records


def _memory_record(
    index: int,
    memory_key: str,
    content: str,
    memory_type: str = "user_fact",
    importance: float = 0.5,
) -> dict:
    from datetime import datetime, timezone

    return {
        "id": index,
        "memory_type": memory_type,
        "memory_key": memory_key,
        "content": content,
        "importance": importance,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _typical_records() -> list[dict]:
    """名字 + 偏好 + 约定 + 无关摘要的典型记忆集合。"""
    return [
        _memory_record(1, "user_name", "用户说自己叫小明", importance=0.9),
        _memory_record(2, "preference_咖啡", "用户说喜欢咖啡", importance=0.6),
        _memory_record(
            3,
            "promise_周五见面",
            "用户提到：周五一起看电影",
            memory_type="promise",
            importance=0.8,
        ),
        _memory_record(
            4,
            "summary_001",
            "聊过期中考试的复习安排",
            memory_type="conversation_summary",
            importance=0.4,
        ),
    ]


async def _recall(query: str, records: list[dict]):
    from character.memory_service import CharacterMemoryService

    service = CharacterMemoryService(_FakeMemoryRepo(records))
    return await service.load_relevant_memories("kisaki", _scope(), query)


async def test_name_question_recalls_user_name_memory():
    """“我叫什么名字”与“用户说自己叫小明”无公共 bigram，必须靠意图兜底召回。"""
    selected, _total = await _recall("我叫什么名字", _typical_records())
    contents = [item.content for item in selected]
    assert "用户说自己叫小明" in contents


async def test_name_question_variant_still_recalls():
    selected, _total = await _recall("你还记得我的名字吗", _typical_records())
    assert any(item.content == "用户说自己叫小明" for item in selected)


async def test_promise_question_prioritizes_promise_memory():
    selected, _total = await _recall("我们约好了什么", _typical_records())
    contents = [item.content for item in selected]
    assert "用户提到：周五一起看电影" in contents
    if len(contents) > 1:
        # 约定类问题优先召回 promise 记忆
        assert contents[0] == "用户提到：周五一起看电影"


async def test_preference_question_recalls_preference_memory():
    selected, _total = await _recall("我喜欢什么来着", _typical_records())
    assert any(item.content == "用户说喜欢咖啡" for item in selected)


async def test_unrelated_question_still_filters_irrelevant_memories():
    """无关问题（无意图、无词面匹配）不得注入任何记忆：门槛逻辑保持不变。"""
    selected, _total = await _recall("今天天气怎么样", _typical_records())
    assert selected == ()


async def test_intent_floor_keeps_strong_lexical_match_first():
    """意图兜底只保底：词面强匹配（相关度 1.0）仍优先于意图保底分。"""
    records = [
        _memory_record(1, "user_name", "用户说自己叫小明", importance=0.5),
        # 与查询逐字相同的摘要记忆：词面相关度 1.0
        _memory_record(
            2,
            "summary_002",
            "我叫什么名字",
            memory_type="conversation_summary",
            importance=0.5,
        ),
    ]
    selected, _total = await _recall("我叫什么名字", records)
    assert len(selected) == 2
    # 词面完全匹配的记忆第一，意图兜底的名字记忆第二
    assert selected[0].content == "我叫什么名字"
    assert selected[1].content == "用户说自己叫小明"


async def test_character_self_questions_do_not_recall_user_private_memories():
    """询问角色自身/第三方/抽象概念不得召回用户私人记忆。

    意图兜底必须绑定"我/我们"等用户记忆主体：简单子串匹配
    （"名字""喜欢什么""承诺"）会把"你叫什么名字""你喜欢什么"
    "什么是承诺"也判定为用户记忆意图，错误注入私人记忆。
    """
    for query in ("你叫什么名字", "她叫什么名字", "他叫什么名字"):
        selected, _total = await _recall(query, _typical_records())
        assert selected == (), (
            f"{query!r} 询问角色/第三方名字，不应召回用户私人记忆，"
            f"实际召回 {[item.content for item in selected]}"
        )
    for query in ("你喜欢什么", "你讨厌什么", "你的爱好是什么", "她喜欢什么"):
        selected, _total = await _recall(query, _typical_records())
        assert selected == (), (
            f"{query!r} 询问角色/第三方喜好，不应召回用户私人记忆，"
            f"实际召回 {[item.content for item in selected]}"
        )
    # 主体与话题间隔超过固定窗口的稍长语句也必须抑制：
    # "平时最"这类状语不能成为绕过窗口的通道
    for query in ("你平时最喜欢什么", "她平时最喜欢什么", "你平时最爱聊什么"):
        selected, _total = await _recall(query, _typical_records())
        assert selected == (), (
            f"{query!r} 主体与话题间隔较长，不应召回用户私人记忆，"
            f"实际召回 {[item.content for item in selected]}"
        )
    # 空白不是子句边界："你 平时最喜欢什么"若按空白拆分，话题子句
    # 失去主体代词，抑制失效
    for query in ("你 平时最喜欢什么", "她 平时 最 喜欢什么"):
        selected, _total = await _recall(query, _typical_records())
        assert selected == (), (
            f"{query!r} 用空白分割主体与话题，不应召回用户私人记忆，"
            f"实际召回 {[item.content for item in selected]}"
        )
    selected, _total = await _recall("什么是承诺", _typical_records())
    assert selected == (), "抽象概念提问不应召回私人约定记忆"


async def test_whitespace_and_conjunction_do_not_break_subject_detection():
    """空白与连接词不能破坏主体识别。

    - "你知道我 叫什么名字吗"：空白割裂"我"与"叫什么"，若按空白
      拆分子句则用户主体丢失，无法召回名字记忆；
    - "你叫什么名字以及我叫什么名字"：单一子句含两个话题词，
      search 只取首个（主体"你"）会误判为非用户主体而抑制，
      finditer 遍历全部话题才能识别第二个话题的用户主体。
    """
    # 空白割裂主体与话题词：仍必须召回
    selected, _total = await _recall("你知道我 叫什么名字吗", _typical_records())
    assert any(
        item.content == "用户说自己叫小明" for item in selected
    ), "空白割裂主体与话题词，名字记忆未被召回"

    # 连接词长句：两个话题词（你→非用户、我→用户），不抑制且可召回
    selected, _total = await _recall(
        "你叫什么名字以及我叫什么名字", _typical_records()
    )
    assert any(
        item.content == "用户说自己叫小明" for item in selected
    ), "连接词长句只分析首个话题词，用户名字记忆未被召回"


async def test_lexical_preference_overlap_suppressed_for_character_questions():
    """"你喜欢什么"与"用户说喜欢咖啡"共享"喜欢"bigram（Jaccard≈0.11），
    词面匹配本可通过 MIN_RELEVANCE 门槛；非用户主体抑制必须挡住这种泄漏，
    包括主体与话题间隔较长的"你平时最喜欢什么"。
    """
    records = [
        _memory_record(1, "preference_咖啡", "用户说喜欢咖啡", importance=0.9),
        _memory_record(2, "user_name", "用户的名字是小明", importance=0.9),
    ]
    # "你喜欢什么" 与偏好记忆共享"喜欢"；"你叫什么名字"与名字记忆共享"名字"
    # "你平时最喜欢什么"同样共享"喜欢"，且"平时最"超出旧的固定窗口
    for query in ("你喜欢什么", "你叫什么名字", "你平时最喜欢什么"):
        selected, _total = await _recall(query, records)
        assert selected == (), (
            f"{query!r} 的词面匹配泄漏了用户私人记忆："
            f"{[item.content for item in selected]}"
        )


async def test_user_subject_questions_still_recall_despite_other_pronouns():
    """问题中出现"你"但记忆主体仍是用户时，兜底召回必须继续生效。"""
    # "你知道我叫什么名字吗"：主体是我，"你"只是询问对象
    selected, _total = await _recall("你知道我叫什么名字吗", _typical_records())
    assert any(item.content == "用户说自己叫小明" for item in selected)

    # "你答应过我什么"：约定是双方共同记忆，"你"作主体也触发
    selected, _total = await _recall("你答应过我什么", _typical_records())
    assert any(item.content == "用户提到：周五一起看电影" for item in selected)

    # "我们之前说好了什么"：我们主体 + 说好话题
    selected, _total = await _recall("我们之前说好了什么", _typical_records())
    assert any(item.content == "用户提到：周五一起看电影" for item in selected)

    # 主体取话题词前最近的代词："你问我喜欢什么"主体是我，"我问你"主体是你
    selected, _total = await _recall("你问我喜欢什么", _typical_records())
    assert any(item.content == "用户说喜欢咖啡" for item in selected)
    selected, _total = await _recall("我问你喜欢什么", _typical_records())
    assert selected == (), "'我问你喜欢什么'询问角色喜好，不应召回用户偏好"

    # 子句拆分：同类问题里存在用户主体问句时不抑制
    selected, _total = await _recall("你叫什么名字？我叫什么名字？", _typical_records())
    assert any(item.content == "用户说自己叫小明" for item in selected)
