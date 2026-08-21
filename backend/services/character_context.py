"""角色上下文编排服务：串起画像、关系、记忆、历史、情景与决策。

一轮对话的完整流程：
1. prepare_turn：生成前加载全部上下文并编译成模型输入；
2. 模型生成回复（调用方负责）；
3. complete_turn：生成后写入新记忆、更新关系。

服务本身不调用 LLM、不访问 vLLM；所有可变数据经仓储读写，
所有规则计算委托给 character 包内的纯函数模块。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from character.context_builder import (
    build_user_scope,
    compile_character_context,
)
from character.decision_policy import DecisionPolicy
from character.memory_extractor import (
    extract_memories,
    extract_preferred_address,
    next_relationship_stage,
)
from character.memory_service import CharacterMemoryService
from character.models import (
    CharacterContext,
    CompiledCharacterContext,
    MemoryItem,
    RelationshipState,
    SituationState,
    UserScope,
)
from character.profile_registry import CharacterProfileRegistry
from character.situation_analyzer import SITUATION_DAILY, SITUATION_LABELS, SituationAnalyzer
from repositories.character_memory import CharacterMemoryRepository
from repositories.messages import MessageRepository

logger = logging.getLogger(__name__)

# prepare_turn 并发加载时历史读取的参数
HISTORY_LIMIT = 8
HISTORY_MAX_CHARS = 6000


@dataclass(frozen=True)
class TurnInput:
    """一轮对话的输入侧信息（由 API 层从请求中组装）。"""

    message: str
    platform: str
    adapter: str
    sender_id: str
    conversation_id: str
    conversation_type: str
    # 调用方（bot/前端）自带的现场历史；非空时优先于数据库历史
    history: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True)
class PreparedCharacterTurn:
    """prepare_turn 的结果：编译后的上下文 + 生成所需附加信息。

    注意：有角色状态的对话是"每轮都有副作用"的有状态流程（回写
    记忆与关系），不能进入回复缓存——缓存命中会跳过回写导致状态
    与对话脱节，因此调用方应整体绕过响应缓存，而不是为角色状态
    构造缓存指纹。
    """

    character_id: str
    user_scope: UserScope
    compiled: CompiledCharacterContext
    history: tuple[dict[str, str], ...]
    relationship: RelationshipState
    memory_candidates: int
    interaction_count: int


@dataclass
class _TurnOutcome:
    """complete_turn 的执行结果（用于日志与测试断言）。"""

    new_memories: int = 0
    interaction_count: int = 0
    stage: str = "stranger"
    preferred_address: str = ""


class CharacterContextService:
    """角色上下文编排：生成前编译、生成后回写。"""

    def __init__(
        self,
        profile_registry: CharacterProfileRegistry,
        memory_repository: CharacterMemoryRepository,
        message_repository: MessageRepository,
        *,
        memory_service: CharacterMemoryService | None = None,
        situation_analyzer: SituationAnalyzer | None = None,
        decision_policy: DecisionPolicy | None = None,
    ) -> None:
        self._profiles = profile_registry
        self._memory_repo = memory_repository
        self._message_repo = message_repository
        self._memory_service = memory_service or CharacterMemoryService(memory_repository)
        self._situation_analyzer = situation_analyzer or SituationAnalyzer()
        self._decision_policy = decision_policy or DecisionPolicy()

    async def prepare_turn(self, turn: TurnInput, character_id: str) -> PreparedCharacterTurn:
        """加载本轮全部上下文并编译成模型输入。

        任何用户范围字段非法都会抛 ValueError（调用方应降级为
        无角色上下文的旧行为，而不是让整条消息失败）。
        """
        user_scope = build_user_scope(
            platform=turn.platform,
            adapter=turn.adapter,
            sender_id=turn.sender_id,
            conversation_id=turn.conversation_id,
            conversation_type=turn.conversation_type,
        )

        profile, relationship, memories, history = await asyncio.gather(
            asyncio.to_thread(self._profiles.get_profile, character_id),
            self._memory_repo.get_relationship(character_id, user_scope),
            self._memory_service.load_relevant_memories(
                character_id, user_scope, turn.message
            ),
            self._load_history(turn, user_scope),
        )
        memories_items, memory_candidates = memories

        situation_type, response_goal = self._situation_analyzer.analyze(turn.message)
        situation = SituationState(
            # 系统提示词中只放固定分类标签，用户消息原文绝不进入
            # 系统提示词（提示词注入防护）
            topic=SITUATION_LABELS.get(situation_type, SITUATION_LABELS[SITUATION_DAILY]),
            emotion_hint=self._situation_analyzer.detect_emotion(turn.message),
            response_goal=response_goal,
        )
        decision = self._decision_policy.decide(profile, relationship, situation_type)

        context = CharacterContext(
            profile=profile,
            user_scope=user_scope,
            relationship=relationship,
            situation=situation,
            memories=memories_items,
            decision=decision,
        )
        compiled = compile_character_context(context)

        relationship_record = await self._memory_repo.get_relationship_record(
            character_id, user_scope
        )
        interaction_count = int(
            (relationship_record or {}).get("interaction_count") or 0
        )

        return PreparedCharacterTurn(
            character_id=character_id,
            user_scope=user_scope,
            compiled=compiled,
            history=tuple(turn.history) or tuple(history),
            relationship=relationship,
            memory_candidates=memory_candidates,
            interaction_count=interaction_count,
        )

    async def complete_turn(
        self,
        prepared: PreparedCharacterTurn,
        turn: TurnInput,
        reply: str,
        *,
        source_message_id: str = "",
    ) -> _TurnOutcome:
        """生成成功后回写：交互计数、新记忆、关系推进。

        任何单条写入失败只记日志，不影响其余写入（记忆是增强项，
        不允许让已完成生成的消息在调用方表现为失败）。
        """
        outcome = _TurnOutcome()

        # 1. 交互计数 +1
        try:
            outcome.interaction_count = await self._memory_repo.increment_interaction(
                prepared.character_id, prepared.user_scope
            )
        except Exception:
            logger.warning(
                "角色交互计数更新失败 character=%s error=%s",
                prepared.character_id,
                exc_info=True,
            )

        # 2. 提取并写入新记忆
        try:
            extracted = extract_memories(turn.message)
            for item in extracted[:4]:
                await self._memory_repo.add_or_update_memory(
                    prepared.character_id,
                    prepared.user_scope,
                    MemoryItem(
                        memory_id="",
                        memory_type=item.memory_type,  # type: ignore[arg-type]
                        content=item.content,
                        importance=item.importance,
                    ),
                    memory_key=item.memory_key,
                    source_message_id=source_message_id or None,
                )
                outcome.new_memories += 1
        except Exception:
            logger.warning(
                "角色长期记忆写入失败 character=%s error=%s",
                prepared.character_id,
                exc_info=True,
            )

        # 3. 关系阶段推进与称呼偏好
        try:
            stage = next_relationship_stage(
                prepared.relationship.stage, outcome.interaction_count
            )
            address = extract_preferred_address(turn.message)
            if stage != prepared.relationship.stage or address:
                await self._memory_repo.upsert_relationship(
                    prepared.character_id,
                    prepared.user_scope,
                    RelationshipState(
                        stage=stage,  # type: ignore[arg-type]
                        preferred_address=address or prepared.relationship.preferred_address,
                        summary=prepared.relationship.summary,
                    ),
                )
            outcome.stage = stage
            outcome.preferred_address = address or prepared.relationship.preferred_address
        except Exception:
            logger.warning(
                "角色关系更新失败 character=%s error=%s",
                prepared.character_id,
                exc_info=True,
            )

        return outcome

    async def _load_history(
        self, turn: TurnInput, user_scope: UserScope
    ) -> list[dict[str, str]]:
        """调用方带现场历史时直接使用，否则从数据库读取。"""
        if turn.history:
            return list(turn.history)
        try:
            return await self._message_repo.list_recent_conversation_history(
                user_scope, limit=HISTORY_LIMIT, max_chars=HISTORY_MAX_CHARS
            )
        except Exception:
            logger.warning("角色历史读取失败，按空历史继续", exc_info=True)
            return []


def build_character_context_service(database) -> CharacterContextService:
    """基于指定数据库构建编排服务。

    create_app(custom_container) 的应用实例必须用容器数据库构建服务，
    而不是全局单例——否则多应用实例/测试注入会读写到错误的数据库。
    """
    from character.profile_registry import get_default_profile_registry
    from repositories.character_memory import DatabaseCharacterMemoryRepository
    from repositories.messages import DatabaseMessageRepository

    return CharacterContextService(
        profile_registry=get_default_profile_registry(),
        memory_repository=DatabaseCharacterMemoryRepository(database),
        message_repository=DatabaseMessageRepository(database),
    )


_default_service: CharacterContextService | None = None


def get_default_character_context_service() -> CharacterContextService:
    """返回基于全局单例的默认编排服务（进程内单例）。

    仅供非 HTTP 兼容调用方（bot 直连、旧测试）使用；HTTP 路径
    应经 build_character_context_service(container.db) 按应用构建。
    """
    global _default_service
    if _default_service is None:
        from db.adapter import db as _db

        _default_service = build_character_context_service(_db)
    return _default_service
