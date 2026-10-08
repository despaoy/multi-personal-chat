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
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from character.context_builder import (
    build_user_scope,
    compile_dynamic_context,
    compile_profile_context,
    compile_reference_context,
    compile_relationship_style,
)
from character.contextual_policy import ContextualDecisionPolicy, PolicyOutcome, create_contextual_policy
from character.conversation_flow import compile_continuity, compile_rhythm
from character.decision_policy import DecisionPolicy
from character.evidence_selector import ContextualEvidenceSelector, SelectionOutcome, create_evidence_selector
from character.memory_extractor import (
    extract_memories,
    extract_preferred_address,
    memory_write_allowed,
)
from character.memory_service import CharacterMemoryService
from character.models import (
    CompiledCharacterContext,
    DecisionPlan,
    InteractionState,
    MemoryItem,
    RelationshipState,
    SituationState,
    UserScope,
)
from character.natural_relationship import (
    compile_notes,
    parse_command,
    relationship_write_blocked,
    save_note,
)
from character.output_guard import ReplyGuard, build_reply_guard
from character.rule_memory_writer import write_rule_memory
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
    from services.delivery_memory import CompletionSnapshot

def compile_user_recall_context(history):
    """Use complete available user history; assistant guesses are not query facts."""
    return "\n".join(item["content"] for item in history
                     if item.get("role")=="user" and isinstance(item.get("content"),str))


logger = logging.getLogger(__name__)


class ContextAfterMemoryOperationError(RuntimeError):
    """Context failed after an explicit operation returned its execution receipt."""

    def __init__(self, receipt):
        from character.memory_operation import operation_receipt_context

        super().__init__("context preparation failed after a memory operation")
        self.operation_summary = operation_receipt_context(receipt)


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
    received_at: datetime | None = None


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
    memory_selection_status: str = "disabled"
    memory_selection_candidate_count: int = 0
    memory_selection_latency_ms: float = 0.0
    contextual_policy_status: str = "disabled"
    contextual_policy_reason: str = ""
    contextual_policy_latency_ms: float = 0.0
    memory_budget: dict[str, int] = field(default_factory=dict)
    memory_recall: dict[str, object] = field(default_factory=dict)
    # Server preparation time, retained across generation and background writes.
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Only explicit interactive operations populate this; prepare_turn stays read-only.
    memory_operation_receipt: dict[str, object] | None = None


@dataclass
class _TurnOutcome:
    """complete_turn 的执行结果（用于日志与测试断言）。"""

    new_memories: int = 0
    memory_enrichment_scheduled: bool = False
    source_capture: str = ""
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
        memory_selector: ContextualEvidenceSelector | None = None,
        contextual_policy: ContextualDecisionPolicy | None = None,
        source_recall_enabled: bool | None = None,
        source_window_radius: int = 0,
        history_limit: int = HISTORY_LIMIT,
        history_max_chars: int = HISTORY_MAX_CHARS,
        source_max_chars: int = 2400,
        defer_source_budget: bool = False,
        reference_max_chars: int | None = None,
        reference_observation_semantics: bool = False,
    ) -> None:
        if min(history_limit, history_max_chars, source_max_chars) <= 0:
            raise ValueError('Context budgets must be positive')
        if reference_max_chars is not None and (type(reference_max_chars) is not int or reference_max_chars <= 0):
            raise ValueError("Reference context budget must be a positive integer")
        self._reference_max_chars = reference_max_chars
        self._reference_observation_semantics = reference_observation_semantics
        self._history_limit = history_limit
        self._history_max_chars = history_max_chars
        self._profiles = profile_registry
        self._memory_repo = memory_repository
        self._message_repo = message_repository
        self._memory_service = memory_service or CharacterMemoryService(memory_repository)
        self._situation_analyzer = situation_analyzer or SituationAnalyzer()
        self._decision_policy = decision_policy or DecisionPolicy()
        self._semantic_estimator = semantic_estimator
        self._memory_selector = memory_selector or create_evidence_selector()
        self._contextual_policy = contextual_policy or create_contextual_policy()
        from character.source_memory import SourceMemoryService

        self._source_memory = SourceMemoryService(memory_repository, window_radius=source_window_radius,
                                                 max_chars=source_max_chars, defer_budget=defer_source_budget)
        self._source_recall_enabled = (source_recall_enabled if source_recall_enabled is not None else
            os.getenv("MEMORY_SOURCE_RECALL_ENABLED", "false").lower() in {"true", "1", "yes", "on"})

    async def prepare_interactive_turn(
        self, turn: TurnInput, character_id: str, *, source_message_id: str = '',
    ) -> PreparedCharacterTurn:
        """Execute explicit erasure before the response snapshot, once per turn.

        Callers must opt in only for an accepted interactive request, never for
        internal inference, hypothetical branches or delivery-gated previews.
        Ordinary fact writes remain asynchronous in complete_turn.
        """
        from character.memory_llm import get_memory_enrichment_scheduler, is_memory_erasure_request
        from character.memory_operation import operation_receipt_context

        if not is_memory_erasure_request(turn.message):
            return await self.prepare_turn(turn, character_id)
        # Validate identity and profile before any mutation.
        user_scope = build_user_scope(platform=turn.platform, adapter=turn.adapter,
            sender_id=turn.sender_id, conversation_id=turn.conversation_id,
            conversation_type=turn.conversation_type)
        await asyncio.to_thread(self._profiles.get_profile, character_id)
        received_at = turn.received_at or datetime.now(timezone.utc)
        history = tuple(turn.history) or tuple(await self._load_history(turn, user_scope, character_id))
        receipt = await get_memory_enrichment_scheduler().schedule_and_wait(
            repository=self._memory_repo, character_id=character_id, user_scope=user_scope,
            message=turn.message, rule_hints=(), history=history,
            source_message_id=source_message_id or None, observed_at=received_at,
        )
        # The operation may already be committed; preserve its receipt on failure.
        try:
            prepared = await self.prepare_turn(turn, character_id)
        except Exception as exc:
            raise ContextAfterMemoryOperationError(receipt) from exc
        return replace(prepared, memory_operation_receipt=receipt,
            compiled=replace(prepared.compiled,
                memory_operation_receipt={key: receipt[key] for key in ('status', 'persisted', 'source_erased') if key in receipt},
                dynamic_context='\n\n'.join(filter(None, (
                prepared.compiled.dynamic_context, operation_receipt_context(receipt),
            )))))

    async def prepare_turn(self, turn: TurnInput, character_id: str) -> PreparedCharacterTurn:
        """加载本轮全部上下文并编译成模型输入。

        配置、读取、分析与已启用审核的失败交给调用方报告，不替换角色或规则状态。
        """
        received_at = turn.received_at or datetime.now(timezone.utc)
        user_scope = build_user_scope(
            platform=turn.platform,
            adapter=turn.adapter,
            sender_id=turn.sender_id,
            conversation_id=turn.conversation_id,
            conversation_type=turn.conversation_type,
        )

        if self._memory_selector is not None:
            profile, relationship_record, history, relationship_notes = await asyncio.gather(
                asyncio.to_thread(self._profiles.get_profile, character_id),
                self._memory_repo.get_relationship_record(character_id, user_scope),
                self._load_history(turn, user_scope, character_id),
                self._memory_repo.list_relationship_notes(character_id, user_scope),
            )
            # Context is needed before recall, not only after an arbitrary top-k.
            # Assistant guesses are excluded from query expansion; both speakers
            # remain available to the final semantic selector as untrusted data.
            user_topic_context = compile_user_recall_context(history)
            memories = await self._load_memory_candidates(
                character_id, user_scope, turn.message, retrieval_context=user_topic_context, reference_time=received_at,
            )
        else:
            profile, relationship_record, memories, history, relationship_notes = await asyncio.gather(
                asyncio.to_thread(self._profiles.get_profile, character_id),
                self._memory_repo.get_relationship_record(character_id, user_scope),
                self._load_memory_candidates(character_id, user_scope, turn.message),
                self._load_history(turn, user_scope, character_id),
                self._memory_repo.list_relationship_notes(character_id, user_scope),
            )
        memories_items, memory_candidates, memory_recall = memories
        from repositories.character_memory import relationship_from_record

        relationship = relationship_from_record(relationship_record)

        # Reuse the already-loaded history: no extra database/model call. The
        # caller-provided live history wins over the persisted fallback.
        effective_history = tuple(turn.history) or tuple(history)
        interaction = self._situation_analyzer.estimate(turn.message, effective_history)
        semantic_outcome: SemanticReviewOutcome | None = None
        if self._semantic_estimator is not None:
            semantic_outcome = await self._semantic_estimator.refine_with_diagnostics(
                turn.message, effective_history, interaction,
            )
            interaction = semantic_outcome.state
            _observe_semantic_review(semantic_outcome)
        memory_selection = SelectionOutcome()
        if self._memory_selector is not None:
            memory_selection = await self._memory_selector.select(
                turn.message, memories_items, history=effective_history,
                profile=profile, interaction=interaction,
                reference_time=received_at,
            )
            memories_items = memory_selection.memories
            logger.info(
                "Contextual memory selection status=%s candidates=%d selected=%d latency_ms=%.1f",
                memory_selection.status, memory_selection.candidate_count,
                len(memories_items), memory_selection.latency_ms,
            )
        situation_type = (
            interaction.primary_situation if interaction.primary_situation in SITUATION_LABELS else SITUATION_DAILY
        )

        # Budget the evidence before selecting a recall action. Selection is
        # not injection: complete packets can be too large for the final prompt.
        # Compile once so policy and generation see exactly the same evidence.
        memory_budget: dict[str, int] = {}
        reference_context, injected_memory_ids = compile_reference_context(
            tuple(memories_items), preferred_address=relationship.preferred_address,
            complete_evidence=self._memory_selector is not None,
            diagnostics=memory_budget,
            max_chars=self._reference_max_chars,
            observation_semantics=self._reference_observation_semantics,
        )

        note_context = compile_notes(relationship_notes, turn.message, received_at)
        continuity = compile_continuity(effective_history, turn.message)
        conversation_reference = "\n\n".join(filter(None, (note_context, continuity)))

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
        decision = self._decision_policy.decide(
            profile,
            relationship,
            situation_type,
            interaction=interaction if interaction.has_soft_context else None,
            has_relevant_memory=bool(injected_memory_ids),
        )

        policy_outcome = PolicyOutcome(decision)
        if self._contextual_policy is not None:
            policy_outcome = await self._contextual_policy.refine(
                decision, query=turn.message, history=effective_history, profile=profile,
                relationship=relationship, interaction=interaction, has_relevant_memory=bool(injected_memory_ids),
            )
            decision = policy_outcome.plan
            logger.info(
                "Contextual policy status=%s reason=%s latency_ms=%.1f",
                policy_outcome.status, policy_outcome.reason, policy_outcome.latency_ms,
            )
        compiled = CompiledCharacterContext(
            profile_context=compile_profile_context(profile),
            dynamic_context="\n\n".join(filter(None, (
                compile_dynamic_context(relationship, situation, decision, interaction),
                compile_relationship_style(profile),
                compile_rhythm(effective_history, turn.message, interaction),
            ))),
            reference_context=reference_context,
            conversation_reference_context=conversation_reference,
            used_memory_ids=injected_memory_ids,
            memory_packets=tuple(item for item in memories_items if item.memory_id in injected_memory_ids),
            memory_review_query=str(memory_recall.get('mention_query') or ''),
            memory_review_text=str(memory_recall.get('mention_review') or ''),
            memory_field_presence=tuple((key, value) for key, value in
                                        (memory_recall.get('field_presence') or {}).items()
                                        if value is None or isinstance(value, bool)),
            memory_status=("available" if injected_memory_ids else
                           "retrieval_error" if memory_recall.get('status') == 'retrieval_error' else "no_match"),
        )

        if self._source_recall_enabled:
            from character.source_memory import attach_sources

            sources = await self._source_memory.recall(character_id, user_scope, turn.message,
                                                       memories=compiled.memory_packets,
                                                       retrieval_context=compile_user_recall_context(effective_history))
            compiled = attach_sources(compiled, sources, preferred_address=relationship.preferred_address,
                                      complete_evidence=self._memory_selector is not None)
            memory_recall["sources"] = sources.diagnostics

        interaction_count = int((relationship_record or {}).get("interaction_count") or 0)

        return PreparedCharacterTurn(
            received_at=received_at,
            character_id=character_id,
            user_scope=user_scope,
            compiled=compiled,
            history=effective_history,
            relationship=relationship,
            memory_candidates=memory_candidates,
            memory_budget=memory_budget,
            memory_recall=memory_recall,
            memory_selection_status=memory_selection.status,
            memory_selection_candidate_count=memory_selection.candidate_count,
            memory_selection_latency_ms=memory_selection.latency_ms,
            contextual_policy_status=policy_outcome.status,
            contextual_policy_reason=policy_outcome.reason,
            contextual_policy_latency_ms=policy_outcome.latency_ms,
            interaction_count=interaction_count,
            reply_guard=build_reply_guard(
                profile,
                turn.message,
                effective_history,
                interaction,
                decision,
                has_relevant_memory=bool(injected_memory_ids),
            ),
            interaction=interaction,
            decision=decision,
            semantic_review_status=semantic_outcome.status if semantic_outcome else "disabled",
            semantic_review_reasons=semantic_outcome.reasons if semantic_outcome else (),
            semantic_review_latency_ms=semantic_outcome.latency_ms if semantic_outcome else 0.0,
            semantic_review_history_count=semantic_outcome.history_count if semantic_outcome else 0,
            semantic_review_rule_confidence=semantic_outcome.rule_confidence if semantic_outcome else 0.0,
            semantic_review_confidence=semantic_outcome.review_confidence if semantic_outcome else None,
        )

    async def complete_turn(
        self,
        prepared: PreparedCharacterTurn | CompletionSnapshot,
        turn: TurnInput,
        reply: str,
        *,
        source_message_id: str = "",
        memory_receipt: asyncio.Future | None = None,
        memory_retry_only: bool = False,
    ) -> _TurnOutcome:
        """生成成功后回写：交互计数、新记忆、关系推进。

        任何单条写入失败只记日志，不影响其余写入（记忆是增强项，
        不允许让已完成生成的消息在调用方表现为失败）。
        """
        outcome = _TurnOutcome()

        # 1. 交互计数 +1
        try:
            if not memory_retry_only:
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
            note_command = parse_command(turn.message)
            extracted = () if note_command else extract_memories(turn.message, reference_time=prepared.received_at)
            from character.memory_llm import (
                classify_memory_write_mode,
                get_memory_enrichment_scheduler,
                is_memory_erasure_request,
            )

            scheduler = get_memory_enrichment_scheduler()
            from character.natural_relationship import hypothetical_source_only, quoted_source_only

            ordinary_quote = quoted_source_only(turn.message)
            erasure_request = is_memory_erasure_request(turn.message)
            source_only = (hypothetical_source_only(turn.message) or ordinary_quote) and not extracted and not erasure_request
            if prepared.memory_operation_receipt is not None:
                # Already submitted before generation, including pending and
                # failed results. Never duplicate it after saving the reply.
                outcome.memory_enrichment_mode = 'explicit_operation'
                outcome.memory_enrichment_status = str(prepared.memory_operation_receipt.get('status', 'unknown'))
            elif note_command:
                saved = await save_note(self._memory_repo, prepared.character_id, prepared.user_scope,
                                        note_command, source_message_id or None)
                outcome.memory_enrichment_status = "relationship_note_saved" if saved else "no_change"
            elif relationship_write_blocked(turn.message) and not extracted and not source_only and not (ordinary_quote and erasure_request):
                outcome.memory_enrichment_status = "skipped_fiction_or_note"
            elif scheduler.enabled:
                outcome.memory_enrichment_mode = classify_memory_write_mode(turn.message)
                # Keep authorized complete speech durable even when semantic
                # enrichment is full. Pending identity metadata is not speech.
                capture = getattr(self._memory_repo, "capture_source", None)
                if (source_message_id and callable(capture) and memory_write_allowed(turn.message)
                        and not is_memory_erasure_request(turn.message)):
                    outcome.source_capture = "failed"
                    outcome.memory_enrichment_status = "source_capture_failed"
                    outcome.source_capture = await capture(
                        prepared.character_id, prepared.user_scope,
                        source_message_id=source_message_id, body=turn.message,
                        observed_at=prepared.received_at,
                    )
                if outcome.source_capture in {"stale", "revoked", "conflict"}:
                    outcome.memory_enrichment_status = "source_" + outcome.source_capture
                else:
                    outcome.memory_enrichment_scheduled = scheduler.schedule(
                        repository=self._memory_repo,
                        character_id=prepared.character_id,
                        user_scope=prepared.user_scope,
                        message=turn.message,
                        rule_hints=extracted,
                        history=prepared.history,
                        source_message_id=source_message_id or None,
                        # 只传递本轮真正注入回复上下文的 IDs。“刚才那条说错了”
                        # 可据此定向纠错；reply 本身绝不进入记忆证据。
                        feedback_target_ids=prepared.compiled.used_memory_ids,
                        source_type="user",
                        observed_at=prepared.received_at,
                        source_only=source_only,
                        **({"receipt": memory_receipt} if memory_receipt is not None else {}),
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
                    saved = await write_rule_memory(
                        self._memory_repo, prepared.character_id, prepared.user_scope,
                        item, source_message_id or None,
                        observed_at=prepared.received_at,
                    )
                    outcome.new_memories += int(saved)
                outcome.memory_enrichment_status = "saved" if outcome.new_memories else "no_change"
        except Exception:
            logger.warning(
                "角色长期记忆写入失败 character=%s",
                prepared.character_id,
                exc_info=True,
            )

        if memory_retry_only or outcome.source_capture in {"stale", "revoked", "conflict"}:
            return outcome

        # 3. 关系起点仅手动设置；次数只是统计，不自动升温。
        try:
            stage = prepared.relationship.stage
            address = (extract_preferred_address(turn.message)
                       if memory_write_allowed(turn.message) and not relationship_write_blocked(turn.message) else None)
            if address:
                guarded = getattr(self._memory_repo, "set_address_from_turn", None)
                if source_message_id and callable(guarded):
                    result = await guarded(
                        prepared.character_id, prepared.user_scope,
                        source_message_id=source_message_id, observed_at=prepared.received_at,
                        address=address,
                    )
                    if result["status"] in {"stale", "revoked", "conflict"}:
                        outcome.source_capture = result["status"]
                        return outcome
                    stage = result["relationship"]["relationship_stage"]
                else:
                    from repositories.character_memory import relationship_from_record

                    current = relationship_from_record(await self._memory_repo.get_relationship_record(
                        prepared.character_id, prepared.user_scope,
                    ))
                    stage = current.stage
                    await self._memory_repo.upsert_relationship(
                        prepared.character_id,
                        prepared.user_scope,
                        RelationshipState(
                            stage=stage,  # type: ignore[arg-type]
                            preferred_address=address or prepared.relationship.preferred_address,
                            summary=current.summary,
                        ),
                    )
            outcome.stage = stage
            outcome.preferred_address = address or prepared.relationship.preferred_address
        except Exception:
            logger.warning(
                "角色关系更新失败 character=%s",
                prepared.character_id,
                exc_info=True,
            )

        return outcome

    async def _load_memory_candidates(
        self, character_id: str, user_scope: UserScope, query: str,
        *, retrieval_context: str = "", reference_time: datetime | None = None,
    ) -> tuple[tuple[MemoryItem, ...], int, dict[str, object]]:
        recall = getattr(self._memory_service, 'recall_with_diagnostics', None)
        kwargs = {}
        if self._memory_selector is not None:
            kwargs = dict(for_contextual_selection=True, retrieval_context=retrieval_context,
                reference_time=reference_time,
            )
        if recall is not None:
            return await recall(character_id, user_scope, query, **kwargs)
        items, count = await self._memory_service.load_relevant_memories(character_id, user_scope, query, **kwargs)
        return items, count, {'status': 'unavailable'}

    async def _load_history(self, turn: TurnInput, user_scope: UserScope, character_id: str) -> list[dict[str, str]]:
        """调用方带现场历史时直接使用，否则从数据库读取。"""
        if turn.history:
            return list(turn.history)
        return await self._message_repo.list_recent_conversation_history(
            user_scope, limit=self._history_limit, max_chars=self._history_max_chars, character_id=character_id
        )


def build_character_context_service(database, *, source_recall_enabled: bool | None = None) -> CharacterContextService:
    """基于指定数据库构建编排服务。

    create_app(custom_container) 的应用实例必须用容器数据库构建服务，
    而不是全局单例——否则多应用实例/测试注入会读写到错误的数据库。
    """
    from character.profile_registry import get_default_profile_registry
    from character.semantic_review_adapter import create_default_semantic_review_runtime
    from repositories.character_memory import DatabaseCharacterMemoryRepository
    from repositories.messages import DatabaseMessageRepository

    from inference.provider_context import get_provider_context_budget

    budget = get_provider_context_budget()
    semantic_runtime = create_default_semantic_review_runtime()

    return CharacterContextService(
        profile_registry=get_default_profile_registry(),
        memory_repository=DatabaseCharacterMemoryRepository(database),
        message_repository=DatabaseMessageRepository(database),
        source_recall_enabled=source_recall_enabled,
        history_limit=budget.history_limit,
        history_max_chars=budget.history_max_chars,
        source_max_chars=budget.source_max_chars,
        defer_source_budget=budget.defer_source_budget,
        reference_max_chars=budget.reference_max_chars,
        reference_observation_semantics=budget.reference_observation_semantics,
        memory_selector=create_evidence_selector(context_budget=budget.review),
        contextual_policy=create_contextual_policy(context_budget=budget.review),
        semantic_estimator=SemanticStateEstimator(
            semantic_runtime.reviewer,
            timeout_seconds=semantic_runtime.timeout_seconds,
            context_budget=budget.review,
        ),
    )


def _observe_semantic_review(outcome: SemanticReviewOutcome) -> None:
    """Record text-free semantic-review metrics without risking the turn."""

    if outcome.status != "applied":
        return
    try:
        from infra.observability import increment, log_event

        increment(f"dynamic_context_semantic_review_{outcome.status}")
        log_event(
            "dynamic_context_semantic_review",
            status=outcome.status,
            reasons=list(outcome.reasons),
            latencyMs=round(outcome.latency_ms, 3),
            historyCount=outcome.history_count,
            ruleConfidence=outcome.rule_confidence,
            reviewConfidence=outcome.review_confidence,
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
