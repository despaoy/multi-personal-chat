"""角色上下文编排服务：串起画像、关系、记忆、历史、情景与决策。

一轮对话的完整流程：
1. prepare_turn：生成前加载全部上下文并编译成模型输入；
2. 模型生成回复（调用方负责）；
3. complete_turn：生成后写入新记忆、更新关系。

服务本身不直接访问 vLLM；启用后台记忆判断时只提交一个有界任务，
不等待第二次模型调用。所有可变数据经仓储读写，规则计算与 LLM
候选校验委托给 character 包内的独立模块。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
    DecisionPlan,
    InteractionState,
    MemoryItem,
    RelationshipState,
    SituationState,
    UserScope,
)
from character.output_guard import ReplyGuard, build_reply_guard
from character.semantic_state_estimator import SemanticReviewOutcome, SemanticStateEstimator
from character.situation_analyzer import (
    RESPONSE_GOALS,
    SITUATION_DAILY,
    SITUATION_LABELS,
    SituationAnalyzer,
    affect_label,
)

if TYPE_CHECKING:
    from character.profile_registry import CharacterProfileRegistry
    from repositories.character_memory import CharacterMemoryRepository
    from repositories.messages import MessageRepository

logger = logging.getLogger(__name__)

# prepare_turn 并发加载时历史读取的参数
HISTORY_LIMIT = 24
# 约对应 8K 中文/混合 token，和 24K 模型窗口的三分之一历史预算对齐。
HISTORY_MAX_CHARS = 16000


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
    reply_guard: ReplyGuard
    # Expose the trusted, post-review state and bounded diagnostics so
    # evaluation/observability never has to re-run the analyzer and silently
    # report a different state from the one actually used for generation.
    interaction: InteractionState = field(default_factory=InteractionState)
    decision: DecisionPlan = field(default_factory=DecisionPlan)
    semantic_review_status: str = "disabled"
    semantic_review_reasons: tuple[str, ...] = ()
    semantic_review_latency_ms: float = 0.0
    semantic_review_history_count: int = 0
    semantic_review_rule_confidence: float = 0.0
    semantic_review_confidence: float | None = None
    semantic_review_fallback_reason: str = ""


@dataclass
class _TurnOutcome:
    """complete_turn 的执行结果（用于日志与测试断言）。"""

    new_memories: int = 0
    memory_enrichment_scheduled: bool = False
    memory_enrichment_status: str = "skipped"
    memory_enrichment_mode: str = "none"
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
        semantic_estimator: SemanticStateEstimator | None = None,
    ) -> None:
        self._profiles = profile_registry
        self._memory_repo = memory_repository
        self._message_repo = message_repository
        self._memory_service = memory_service or CharacterMemoryService(memory_repository)
        self._situation_analyzer = situation_analyzer or SituationAnalyzer()
        self._decision_policy = decision_policy or DecisionPolicy()
        self._semantic_estimator = semantic_estimator

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
            self._memory_service.load_relevant_memories(character_id, user_scope, turn.message),
            self._load_history(turn, user_scope),
        )
        memories_items, memory_candidates = memories

        # Reuse the already-loaded history: no extra database/model call. The
        # caller-provided live history wins over the persisted fallback.
        effective_history = tuple(turn.history) or tuple(history)
        analysis_ok = True
        try:
            interaction = self._situation_analyzer.estimate(
                turn.message,
                effective_history,
            )
        except Exception:
            # Analysis is an enhancement. Preserve the character path with a
            # neutral legacy plan if a custom analyzer or a future rule fails.
            logger.warning("互动状态分析失败，按中性角色策略继续", exc_info=True)
            interaction = InteractionState()
            analysis_ok = False

        semantic_outcome: SemanticReviewOutcome | None = None
        semantic_review_status = "analysis_failed" if not analysis_ok else "disabled"
        semantic_review_reasons: tuple[str, ...] = ()
        semantic_review_latency_ms = 0.0
        semantic_review_history_count = 0
        semantic_review_rule_confidence = 0.0
        semantic_review_confidence: float | None = None
        semantic_review_fallback_reason = ""
        if analysis_ok and self._semantic_estimator is not None:
            # The estimator itself owns timeout, validation and fail-closed
            # recovery.  Its reviewer calls only the low-level base model and
            # cannot re-enter this character-context pipeline.
            try:
                semantic_outcome = await self._semantic_estimator.refine_with_diagnostics(
                    turn.message,
                    effective_history,
                    interaction,
                )
            except Exception:
                # A future custom estimator must obey the same fail-closed
                # contract as the built-in implementation.  Cancellation is
                # not an Exception on supported Python versions and still
                # propagates to the request owner.
                logger.warning("语义复核编排失败，保留规则互动状态", exc_info=True)
                semantic_review_status = "fallback"
                semantic_review_fallback_reason = "error"
            else:
                interaction = semantic_outcome.state
                semantic_review_status = semantic_outcome.status
                semantic_review_reasons = semantic_outcome.reasons
                semantic_review_latency_ms = semantic_outcome.latency_ms
                semantic_review_history_count = semantic_outcome.history_count
                semantic_review_rule_confidence = semantic_outcome.rule_confidence
                semantic_review_confidence = semantic_outcome.review_confidence
                semantic_review_fallback_reason = semantic_outcome.fallback_reason
                _observe_semantic_review(semantic_outcome)
        situation_type = (
            interaction.primary_situation if interaction.primary_situation in SITUATION_LABELS else SITUATION_DAILY
        )

        situation = SituationState(
            # 系统提示词中只放固定分类标签，用户消息原文绝不进入
            # 系统提示词（提示词注入防护）
            topic=SITUATION_LABELS.get(situation_type, SITUATION_LABELS[SITUATION_DAILY]),
            emotion_hint=(
                affect_label(interaction.valence, interaction.arousal) if interaction.has_soft_context else ""
            ),
            response_goal=(
                self._situation_analyzer.response_goal(interaction)
                if interaction.has_soft_context
                else RESPONSE_GOALS[SITUATION_DAILY]
            ),
        )
        try:
            decision = self._decision_policy.decide(
                profile,
                relationship,
                situation_type,
                interaction=interaction if interaction.has_soft_context else None,
                has_relevant_memory=bool(memories_items),
            )
        except Exception:
            logger.warning("互动策略评分失败，按旧版角色策略继续", exc_info=True)
            decision = self._decision_policy.decide(profile, relationship, situation_type)

        context = CharacterContext(
            profile=profile,
            user_scope=user_scope,
            relationship=relationship,
            situation=situation,
            interaction=interaction,
            memories=memories_items,
            decision=decision,
        )
        compiled = compile_character_context(context)

        relationship_record = await self._memory_repo.get_relationship_record(character_id, user_scope)
        interaction_count = int((relationship_record or {}).get("interaction_count") or 0)

        return PreparedCharacterTurn(
            character_id=character_id,
            user_scope=user_scope,
            compiled=compiled,
            history=effective_history,
            relationship=relationship,
            memory_candidates=memory_candidates,
            interaction_count=interaction_count,
            reply_guard=build_reply_guard(
                profile,
                turn.message,
                effective_history,
                interaction,
                decision,
                has_relevant_memory=bool(memories_items),
            ),
            interaction=interaction,
            decision=decision,
            semantic_review_status=semantic_review_status,
            semantic_review_reasons=semantic_review_reasons,
            semantic_review_latency_ms=semantic_review_latency_ms,
            semantic_review_history_count=semantic_review_history_count,
            semantic_review_rule_confidence=semantic_review_rule_confidence,
            semantic_review_confidence=semantic_review_confidence,
            semantic_review_fallback_reason=semantic_review_fallback_reason,
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

        # 2. 提取记忆候选。启用 LLM 时只提交后台复核任务；未启用时
        # 保留原规则写入，方便离线开发和向后兼容。
        try:
            extracted = extract_memories(turn.message)
            from character.memory_llm import (
                classify_memory_write_mode,
                get_memory_enrichment_scheduler,
            )

            scheduler = get_memory_enrichment_scheduler()
            if scheduler.enabled:
                outcome.memory_enrichment_mode = classify_memory_write_mode(turn.message)
                outcome.memory_enrichment_scheduled = scheduler.schedule(
                    repository=self._memory_repo,
                    character_id=prepared.character_id,
                    user_scope=prepared.user_scope,
                    message=turn.message,
                    rule_hints=extracted,
                    history=prepared.history[-4:],
                    source_message_id=source_message_id or None,
                    # 只传递本轮真正注入回复上下文的 IDs。“刚才那条说错了”
                    # 可据此定向纠错；reply 本身绝不进入记忆证据。
                    feedback_target_ids=prepared.compiled.used_memory_ids,
                    source_type="user",
                )
                outcome.memory_enrichment_status = (
                    "queued_hot"
                    if outcome.memory_enrichment_scheduled and outcome.memory_enrichment_mode == "hot"
                    else "buffered_idle"
                    if outcome.memory_enrichment_scheduled
                    else scheduler.status.last_outcome
                )
            else:
                outcome.memory_enrichment_mode = "rules"
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
                outcome.memory_enrichment_status = "saved" if outcome.new_memories else "no_change"
        except Exception:
            logger.warning(
                "角色长期记忆写入失败 character=%s error=%s",
                prepared.character_id,
                exc_info=True,
            )

        # 3. 关系阶段推进与称呼偏好
        try:
            stage = next_relationship_stage(prepared.relationship.stage, outcome.interaction_count)
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

    async def _load_history(self, turn: TurnInput, user_scope: UserScope) -> list[dict[str, str]]:
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
    from character.semantic_review_adapter import create_default_semantic_review_runtime
    from repositories.character_memory import DatabaseCharacterMemoryRepository
    from repositories.messages import DatabaseMessageRepository

    semantic_runtime = create_default_semantic_review_runtime()

    return CharacterContextService(
        profile_registry=get_default_profile_registry(),
        memory_repository=DatabaseCharacterMemoryRepository(database),
        message_repository=DatabaseMessageRepository(database),
        semantic_estimator=SemanticStateEstimator(
            semantic_runtime.reviewer,
            timeout_seconds=semantic_runtime.timeout_seconds,
        ),
    )


def _observe_semantic_review(outcome: SemanticReviewOutcome) -> None:
    """Record text-free semantic-review metrics without risking the turn."""

    if outcome.status not in {"applied", "fallback", "recursive_skip"}:
        return
    try:
        from infra.observability import increment, log_event

        increment(f"dynamic_context_semantic_review_{outcome.status}")
        if outcome.fallback_reason:
            increment(f"dynamic_context_semantic_review_fallback_{outcome.fallback_reason}")
        log_event(
            "dynamic_context_semantic_review",
            status=outcome.status,
            reasons=list(outcome.reasons),
            latencyMs=round(outcome.latency_ms, 3),
            historyCount=outcome.history_count,
            ruleConfidence=outcome.rule_confidence,
            reviewConfidence=outcome.review_confidence,
            fallbackReason=outcome.fallback_reason,
        )
    except Exception:
        # Diagnostics must never turn an optional review into a request failure.
        logger.debug("语义复核诊断记录失败", exc_info=True)


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
