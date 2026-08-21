"""第三步集成测试：人物画像、LoRA 与统一生成请求的连接。

验证：
- LoRA 人物映射只允许显式声明，未映射返回 None；
- 系统提示词顺序：LoRA人物Prompt → dynamic_context → 安全规则 → RAG规则；
- 已有现成 Prompt 的人物不重复拼接 profile_context；
- 无现成 Prompt 的人物用 profile_context 替代；
- 长期记忆只进入用户消息的不可信参考区，不进入系统提示词；
- character_context 为 None 时生成行为与旧链路完全一致。
"""

from __future__ import annotations

import dataclasses

import pytest

from character import (
    CharacterContext,
    CharacterProfile,
    CompiledCharacterContext,
    DecisionPlan,
    MemoryItem,
    RelationshipState,
    SituationState,
    UserScope,
    compile_character_context,
)
from inference.generation_request import (
    GenerationRequest,
    RetrievalResult,
    build_generation_request,
)
from inference.lora_registry import (
    LORA_REGISTRY,
    get_lora_character_id,
    get_lora_system_prompt,
)
from inference.prompt_policy import (
    GLOBAL_FACTUAL_SAFETY_PROMPT,
    RAG_GROUNDING_PROMPT,
    build_grounded_user_message,
    compose_system_prompt,
)


def _profile() -> CharacterProfile:
    return CharacterProfile(
        character_id="tsukiyashiro_kisaki",
        display_name="月社妃",
        identity="《纸上的魔法使》中的月社妃",
        traits=("善于识别话语中的隐含意图",),
        values=("重视自主选择",),
        canonical_relationships=("琉璃：亲生哥哥，也是最重要的情感中心。",),
        speaking_style=("日常回应通常简短，以一至三句为主。",),
        boundaries=("始终以第一人称和月社妃身份自然交流。",),
    )


def _scope() -> UserScope:
    return UserScope(
        platform="qq", adapter="onebot", sender_id="10001",
        conversation_id="20001", conversation_type="group",
    )


def _character_context(
    memories: tuple[MemoryItem, ...] = (),
) -> CompiledCharacterContext:
    return compile_character_context(
        CharacterContext(
            profile=_profile(),
            user_scope=_scope(),
            relationship=RelationshipState(
                stage="familiar", preferred_address="同学", summary="同班同学"
            ),
            situation=SituationState(topic="期中考试", emotion_hint="不耐烦"),
            memories=memories,
            decision=DecisionPlan(intent="回应提问", tone="冷淡", action="简短回答"),
        )
    )


# ============================================
# LoRA 人物映射
# ============================================


def test_kisaki_lora_maps_to_explicit_character_id():
    assert get_lora_character_id("kisaki") == "tsukiyashiro_kisaki"
    assert LORA_REGISTRY["kisaki"]["character_id"] == "tsukiyashiro_kisaki"
    # 映射不影响原有的系统提示词与路径字段
    assert get_lora_system_prompt("kisaki").strip()


def test_unmapped_loras_return_none():
    """未映射的 LoRA 不猜测人物，返回 None 以维持旧行为。"""
    assert get_lora_character_id("hutao") is None
    assert get_lora_character_id("minamo") is None
    assert get_lora_character_id("unknown-lora") is None


# ============================================
# 系统提示词组装顺序
# ============================================


def test_compose_system_prompt_section_order():
    prompt = compose_system_prompt(
        "人物提示词",
        include_rag=True,
        dynamic_context="【当前关系】\n关系阶段：familiar",
    )
    order = [
        prompt.index("人物提示词"),
        prompt.index("【当前关系】"),
        prompt.index(GLOBAL_FACTUAL_SAFETY_PROMPT.strip()[:20]),
        prompt.index(RAG_GROUNDING_PROMPT.strip()[:10]),
    ]
    assert order == sorted(order)


def test_compose_system_prompt_without_dynamic_context_unchanged():
    """不传 dynamic_context 时与旧行为完全一致。"""
    assert compose_system_prompt("人物提示词") == "人物提示词\n\n" + GLOBAL_FACTUAL_SAFETY_PROMPT
    assert (
        compose_system_prompt("人物提示词", include_rag=True)
        == "人物提示词\n\n" + GLOBAL_FACTUAL_SAFETY_PROMPT + "\n\n" + RAG_GROUNDING_PROMPT
    )


# ============================================
# 用户消息的不可信数据区
# ============================================


def test_grounded_user_message_with_memory_only():
    result = build_grounded_user_message(
        "在吗？", "", max_chars=800, memory_context="以下内容仅作为历史事实参考……"
    )
    assert '<character_memory trust="untrusted"' in result
    assert '<retrieved_evidence trust="untrusted"' not in result
    assert "<user_query>\n在吗？" in result
    # 记忆区在用户问题之前
    assert result.index("<character_memory") < result.index("<user_query>")


def test_grounded_user_message_memory_and_evidence_separated():
    result = build_grounded_user_message(
        "在吗？", "原作证据内容", max_chars=800, memory_context="历史记忆内容"
    )
    # 两个独立的不可信数据区，记忆在前、证据在后、用户问题最后
    assert (
        result.index("<character_memory")
        < result.index("<retrieved_evidence")
        < result.index("<user_query>")
    )


def test_grounded_user_message_escapes_memory():
    result = build_grounded_user_message(
        "在吗？", "", max_chars=800, memory_context='记忆含<script>与"引号"'
    )
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_grounded_user_message_without_memory_unchanged():
    """不传 memory_context 时与旧行为完全一致。"""
    old_style = build_grounded_user_message("在吗？", "证据", max_chars=800)
    assert old_style.startswith('<retrieved_evidence trust="untrusted"')
    assert "character_memory" not in old_style
    # 完全无数据时返回原消息
    assert build_grounded_user_message("在吗？", "", max_chars=800) == "在吗？"


# ============================================
# GenerationRequest 集成
# ============================================


def test_kisaki_scenario_profile_context_not_duplicated():
    """月社妃已有 Prompt v3：persona 非空时不拼接 profile_context。"""
    persona = "你是月社妃（Prompt v3 全文……）"
    request = GenerationRequest(
        message="在吗？",
        persona_prompt=persona,
        character_context=_character_context(
            memories=(MemoryItem("m1", "user_fact", "用户喜欢喝咖啡", 0.9),)
        ),
    )
    plan = build_generation_request(request)
    system = plan.messages[0]["content"]

    # persona 出现且 dynamic_context 进入系统提示词
    assert persona in system
    assert "【当前关系】" in system
    assert "关系阶段：familiar" in system
    # profile_context 内容不重复拼接（画像独有内容不在系统提示词中）
    assert "【人物核心性格】" not in system
    assert "【原作核心关系】" not in system
    assert "善于识别话语中的隐含意图" not in system
    # 顺序：persona → dynamic → 安全规则
    assert (
        system.index(persona)
        < system.index("【当前关系】")
        < system.index(GLOBAL_FACTUAL_SAFETY_PROMPT.strip()[:20])
    )


def test_profile_context_used_as_fallback_without_persona():
    """无现成 Prompt 的人物使用 profile_context 替代。"""
    request = GenerationRequest(
        message="在吗？",
        persona_prompt="",
        character_context=_character_context(),
    )
    plan = build_generation_request(request)
    system = plan.messages[0]["content"]

    assert "【人物核心性格】" in system
    assert "【原作核心关系】" in system
    assert "【人物身份】" in system
    # dynamic_context 依然在画像之后
    assert system.index("【人物身份】") < system.index("【当前关系】")


def test_memories_enter_user_message_not_system():
    memories = (MemoryItem("m1", "user_fact", "用户喜欢喝咖啡", 0.9),)
    request = GenerationRequest(
        message="推荐饮料",
        persona_prompt="你是月社妃",
        character_context=_character_context(memories=memories),
    )
    plan = build_generation_request(request)
    system = plan.messages[0]["content"]
    user_message = plan.messages[-1]["content"]

    # 记忆只进入用户消息的不可信参考区
    assert "用户喜欢喝咖啡" not in system
    assert '<character_memory trust="untrusted"' in user_message
    assert "用户喜欢喝咖啡" in user_message
    # 记忆在系统提示词中不出现安全声明
    assert "仅作为历史事实参考" not in system


def test_rag_evidence_and_memory_coexist_in_user_message():
    request = GenerationRequest(
        message="琉璃是谁？",
        persona_prompt="你是月社妃",
        retrieval=RetrievalResult(status="ok", evidence="原作中琉璃的相关证据"),
        character_context=_character_context(
            memories=(MemoryItem("m1", "user_fact", "用户喜欢喝咖啡", 0.9),)
        ),
    )
    plan = build_generation_request(request)
    user_message = plan.messages[-1]["content"]

    assert user_message.index("<character_memory") < user_message.index(
        "<retrieved_evidence"
    ) < user_message.index("<user_query>")
    # RAG 启用时系统提示词包含 RAG 规则
    system = plan.messages[0]["content"]
    assert RAG_GROUNDING_PROMPT.strip()[:10] in system


def test_no_character_context_behavior_unchanged():
    """character_context 为 None 时消息结构与旧链路完全一致。"""
    request = GenerationRequest(
        message="在吗？",
        persona_prompt="你是月社妃",
        retrieval=RetrievalResult(status="ok", evidence="证据"),
    )
    plan = build_generation_request(request)

    assert len(plan.messages) == 2
    assert plan.messages[0]["role"] == "system"
    assert plan.messages[-1]["content"].startswith(
        '<retrieved_evidence trust="untrusted"'
    )
    assert "character_memory" not in plan.messages[-1]["content"]
    assert "【当前关系】" not in plan.messages[0]["content"]


def test_generation_request_character_context_default_none():
    request = GenerationRequest(message="在吗？")
    assert request.character_context is None


def test_character_context_field_is_typed_on_request():
    """GenerationRequest 可携带 CompiledCharacterContext 且不可变。"""
    context = _character_context()
    request = GenerationRequest(message="在吗？", character_context=context)
    assert request.character_context is context
    with pytest.raises(dataclasses.FrozenInstanceError):
        request.character_context = None  # type: ignore[misc]
