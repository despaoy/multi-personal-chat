"""角色上下文的数据模型。

只定义人物相关的不可变数据结构。本模块不依赖 FastAPI、数据库、
RAG 和 vLLM，保证可以独立测试。

所有对象均为 frozen dataclass：并发请求处理中不会被意外修改，
这是低成本的并发安全措施。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RelationshipStage = Literal["stranger", "acquaintance", "familiar", "close"]
MemoryType = Literal["user_fact", "shared_event", "promise", "conversation_summary"]
ConversationType = Literal["private", "group", "channel"]


@dataclass(frozen=True)
class CharacterProfile:
    """固定人物画像，只保存人物稳定特征。

    不保存用户信息和当前情绪。

    - canonical_relationships 保存与原作人物（琉璃、彼方、夜子、理央等）
      的稳定关系；与当前聊天用户的动态关系（RelationshipState）严格分开。
    - version 用于画像确实变化时区分版本，第一版为 "v1"。
    """

    character_id: str
    display_name: str
    identity: str = ""
    version: str = "v1"
    traits: tuple[str, ...] = ()
    values: tuple[str, ...] = ()
    canonical_relationships: tuple[str, ...] = ()
    speaking_style: tuple[str, ...] = ()
    boundaries: tuple[str, ...] = ()


@dataclass(frozen=True)
class UserScope:
    """当前用户范围，用于隔离 QQ、微信、私聊、群聊和频道记忆。

    只保存规范化后的身份标识，不保存用户昵称——昵称不作为身份依据。
    """

    platform: str
    adapter: str
    sender_id: str
    conversation_id: str
    conversation_type: ConversationType

    @property
    def memory_scope_key(self) -> tuple[str, ...]:
        """长期记忆的隔离键（元组形式：可比较、可哈希，无分隔符碰撞）。

        - 私聊：(平台, 适配器, "private", 用户ID)
        - 群聊：(平台, 适配器, "group", 群ID, 用户ID)
        - 频道：(平台, 适配器, "channel", 频道ID, 用户ID)

        必须包含适配器：同一平台下 NapCat / OneBot 等不同适配器的
        用户ID可能互不通用，直接拼接会串用记忆。
        """
        if self.conversation_type == "private":
            return (self.platform, self.adapter, "private", self.sender_id)
        return (
            self.platform,
            self.adapter,
            self.conversation_type,
            self.conversation_id,
            self.sender_id,
        )


@dataclass(frozen=True)
class RelationshipState:
    """人物与当前用户的关系。

    关系阶段使用简单枚举，不设计精确好感度。
    """

    stage: RelationshipStage = "stranger"
    preferred_address: str = ""
    summary: str = ""


@dataclass(frozen=True)
class SituationState:
    """当前对话情景。

    topic 只允许存放固定分类标签（见 situation_analyzer.SITUATION_LABELS，
    如"日常闲聊"/"事实询问"），严禁写入用户消息原文——本字段会进入
    系统提示词，用户原文属于提示词注入向量。
    emotion_hint 是系统推测（固定标签），不是确定事实，不得当成事实保存。
    """

    topic: str = ""
    emotion_hint: str = ""
    response_goal: str = ""


@dataclass(frozen=True)
class MemoryItem:
    """已经被选中的一条相关记忆（由调用方完成检索与排序）。"""

    memory_id: str
    memory_type: MemoryType
    content: str
    importance: float = 0.0


@dataclass(frozen=True)
class DecisionPlan:
    """本轮行为决策。

    表达"人物准备怎么回应"，不保存最终回答。
    """

    intent: str = ""
    tone: str = ""
    action: str = ""
    avoid: str = ""


@dataclass(frozen=True)
class CharacterContext:
    """组合一轮对话需要的全部信息。"""

    profile: CharacterProfile
    user_scope: UserScope
    relationship: RelationshipState = field(default_factory=RelationshipState)
    situation: SituationState = field(default_factory=SituationState)
    memories: tuple[MemoryItem, ...] = ()
    decision: DecisionPlan = field(default_factory=DecisionPlan)


@dataclass(frozen=True)
class CompiledCharacterContext:
    """最终整理结果，按信任级别拆分。

    - profile_context：结构化人物画像（稳定人物规则）。人物已有现成
      系统提示词（如月社妃 Prompt v3）时不拼接，避免规则重复；
      只作为无现成 Prompt 时的替代。
    - dynamic_context：当前关系、情景和行为决策（每轮变化）。
    - reference_context：长期记忆，单独保存且只进入用户消息的
      不可信参考区，防止记忆中的恶意内容被当成系统指令。
    - used_memory_ids：实际选中的记忆ID。
    """

    profile_context: str
    dynamic_context: str
    reference_context: str
    used_memory_ids: tuple[str, ...] = ()
