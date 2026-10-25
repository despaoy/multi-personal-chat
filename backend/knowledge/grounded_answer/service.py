"""Grounded-answer 服务编排。

链路：结构化检索 bundle → 模式识别 → EvidencePacket → grounded prompt
    → 生成（注入式 Provider）→ citation 解析/校验/绑定 → 统一结果

职责边界：
- 不绕过现有 Provider / 流式输出 / 业务服务：生成函数由调用方注入
  （vLLM 客户端 / ModelManager 适配器 / 评估用 DeepSeek 适配器）
- abstention 模式不调用模型（结构性防绕过）
- 检索/模型失败向上传播，不以弃答掩盖系统错误
- 流式与非流式共用同一 citation 绑定逻辑，最终结构语义一致
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from functools import partial
from math import isfinite
from typing import Any

from inference.context_budget import ReviewContextBudget
from inference.history import validate_history
from inference.provider_context import ProviderContextBudget, get_provider_context_budget
from knowledge.curated_sources import CuratedSourceChangedError, curated_source_receipts, validate_curated_source
from knowledge.dialogue_query import resolve_retrieval_query

from .cache import AnswerCache, answer_cache_key
from .models import (  # noqa: E402
    AnswerMode,
    AnswerStreamEvent,
    AnswerTimings,
    EvidencePacket,
    FailureKind,
    GroundedAnswerResult,
)
from .modes import AnswerModeDecider, ModeDecision
from .packet import EvidencePacketBuilder, is_structured_retrieval_bundle
from .prompt import GroundedPromptBuilder
from .validator import CitationValidator

logger = logging.getLogger(__name__)

# 生成函数协议：与 vLLM 客户端 / ModelManager 适配器同构
GenerateFn = Callable[..., Awaitable[str]]
GenerateStreamFn = Callable[..., AsyncIterator[str]]
RetrieveFn = Callable[..., Any]

DEFAULT_ABSTENTION_REPLY = "关于这个问题，我目前掌握的资料还不足以给出可靠回答。"
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_DEFAULT_GENERATION_TIMEOUT = 60.0


def _strip_think_blocks(text: str) -> str:
    return _THINK_BLOCK_RE.sub("", text).strip()


async def _validate_sources(receipts):
    if not receipts:
        return

    def validate():
        for source in receipts:
            validate_curated_source(source)

    await asyncio.to_thread(validate)


class GroundedAnswerService:
    """通用 grounded-answer 编排服务（多知识域、多业务入口共用）。"""

    def __init__(
        self,
        *,
        packet_builder: EvidencePacketBuilder | None = None,
        mode_decider: AnswerModeDecider | None = None,
        prompt_builder: GroundedPromptBuilder | None = None,
        validator: CitationValidator | None = None,
        cache: AnswerCache | None = None,
        retriever: RetrieveFn | None = None,
        domain_supplements: Callable[[str], str] | None = None,
        index_version_resolver: Callable[[str | None], str] | None = None,
        abstention_reply: str = DEFAULT_ABSTENTION_REPLY,
        generation_timeout: float = _DEFAULT_GENERATION_TIMEOUT,
        answer_max_chars: int = 4000,
    ):
        self.packet_builder = packet_builder or EvidencePacketBuilder()
        self.mode_decider = mode_decider or AnswerModeDecider()
        self.prompt_builder = prompt_builder or GroundedPromptBuilder()
        self.validator = validator or CitationValidator()
        self.cache = cache if cache is not None else AnswerCache()
        self.retriever = retriever
        self._domain_supplements = domain_supplements
        self._index_version_resolver = index_version_resolver
        self.abstention_reply = abstention_reply or DEFAULT_ABSTENTION_REPLY
        self.generation_timeout = float(generation_timeout)
        if (
            isinstance(generation_timeout, bool)
            or not isfinite(self.generation_timeout)
            or self.generation_timeout <= 0
        ):
            raise ValueError("generation_timeout must be a finite positive number")
        if type(answer_max_chars) is not int or answer_max_chars < 0:
            raise ValueError("answer_max_chars must be a non-negative integer")
        self.answer_max_chars = answer_max_chars

    # ------------------------------------------------------------------
    # 对外入口 1：知识问答（自检索；/api/ask 等）
    # ------------------------------------------------------------------
    async def answer(
        self,
        query: str,
        *,
        generate: GenerateFn,
        domain_id: str | None = None,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
        persona_prompt: str = "",
        speaker: str = "",
        history: list[dict[str, str]] | None = None,
        model_id: str = "",
        cache_identity: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        want_citations: bool = True,
        before_publish: Callable[[], Awaitable[None]] | None = None,
    ) -> GroundedAnswerResult:
        if history is not None:
            validate_history(history)
        timings = AnswerTimings()
        start = time.perf_counter()

        # Configured resolvers own index identity; failures must not bypass cache policy.
        index_version = ""
        if self._index_version_resolver is not None:
            index_version = self._index_version_resolver(domain_id)

        if self.cache.ttl > 0 and index_version:
            key = answer_cache_key(
                domain_id=domain_id,
                domains=[domain_id] if domain_id else [],
                query=query,
                filters=filters,
                top_k=top_k,
                index_version=index_version,
                model_id=model_id,
                cache_identity=cache_identity,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                persona_prompt=persona_prompt,
                speaker=speaker,
                want_citations=want_citations,
                history=history,
            )
            cached = self.cache.get(key)
            if cached is not None:
                try:
                    await _validate_sources(cached.curated_source_receipts)
                except CuratedSourceChangedError:
                    # A valid source revision is an ordinary cache miss.
                    cached = None
            if cached is not None:
                cached.cache_hit = True
                cached.model_invoked = False
                cached.timings = AnswerTimings(total_ms=(time.perf_counter() - start) * 1000)
                return cached

        # 角色知识检索是同步 CPU 工作，放入线程池避免阻塞事件循环。
        retrieval_start = time.perf_counter()
        bundle = await self._retrieve(
            query,
            history=history,
            domain_id=domain_id,
            top_k=top_k,
            filters=filters,
        )
        timings.retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        result = await self._answer_from_bundle(
            bundle,
            query,
            generate=generate,
            persona_prompt=persona_prompt,
            speaker=speaker,
            history=history,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            want_citations=want_citations,
            timings=timings,
        )
        timings.total_ms = (time.perf_counter() - start) * 1000

        if before_publish is not None:
            await before_publish()
            await _validate_sources(result.curated_source_receipts)

        if (
            self.cache.ttl > 0
            and index_version
            and result.model_invoked
            and not result.abstained
            and result.failure_kind == ""
        ):
            self.cache.set(key, result)
        return result

    # ------------------------------------------------------------------
    # 对外入口 2：流式（SSE；正文完成后统一发送 citations）
    # ------------------------------------------------------------------
    async def answer_stream(
        self,
        query: str,
        *,
        generate: GenerateFn,
        generate_stream: GenerateStreamFn | None = None,
        domain_id: str | None = None,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
        persona_prompt: str = "",
        speaker: str = "",
        history: list[dict[str, str]] | None = None,
        model_id: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        want_citations: bool = True,
    ) -> AsyncIterator[AnswerStreamEvent]:
        if history is not None:
            validate_history(history)
        if self._index_version_resolver is not None:
            self._index_version_resolver(domain_id)
        timings = AnswerTimings()
        start = time.perf_counter()

        retrieval_start = time.perf_counter()
        bundle = await self._retrieve(
            query,
            history=history,
            domain_id=domain_id,
            top_k=top_k,
            filters=filters,
        )
        timings.retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        decision = self.mode_decider.decide(bundle, persona=bool(persona_prompt))
        packet_start = time.perf_counter()
        packet = self._build_packet(query, bundle, decision)
        timings.packet_ms = (time.perf_counter() - packet_start) * 1000

        # meta 事件先行：客户端尽早得知模式与域
        yield AnswerStreamEvent(
            type="meta",
            data={
                "answer_mode": (
                    AnswerMode.ABSTENTION.value if packet.answer_mode == AnswerMode.NO_RAG else packet.answer_mode.value
                ),
                "domain_id": packet.domain_id,
            },
        )

        if packet.answer_mode == AnswerMode.NO_RAG:
            yield AnswerStreamEvent(type="delta", data={"text": self.abstention_reply})
            yield AnswerStreamEvent(type="citations", data={"citations": []})
            yield AnswerStreamEvent(
                type="done",
                data=self._done_payload(
                    answer=self.abstention_reply,
                    packet=packet,
                    decision=decision,
                    warnings=["abstention:no_domain"],
                    timings=timings,
                    start=start,
                    model_invoked=False,
                    model_id=model_id,
                ),
            )
            return

        if packet.answer_mode == AnswerMode.ABSTENTION:
            yield AnswerStreamEvent(type="delta", data={"text": self.abstention_reply})
            yield AnswerStreamEvent(type="citations", data={"citations": []})
            yield AnswerStreamEvent(
                type="done",
                data=self._done_payload(
                    answer=self.abstention_reply,
                    packet=packet,
                    decision=decision,
                    warnings=[
                        *packet.warnings,
                        f"abstention:{packet.abstention_reason}",
                    ],
                    timings=timings,
                    start=start,
                    model_invoked=False,
                    model_id=model_id,
                ),
            )
            return

        # 生成阶段（流式）
        prompt_start = time.perf_counter()
        messages = self.prompt_builder.build_messages(
            packet,
            persona_prompt=persona_prompt,
            domain_supplement=self._supplement_for(packet),
            speaker=speaker,
            history=history,
            max_tokens=max_tokens,
        )
        timings.prompt_ms = (time.perf_counter() - prompt_start) * 1000

        receipts = curated_source_receipts(bundle)
        await _validate_sources(receipts)
        generation_start = time.perf_counter()
        chunks: list[str] = []
        stream_iter: AsyncIterator[str] | None = None
        try:
            stream_fn = generate_stream or self._wrap_as_stream(generate)
            first_token = True
            stream_iter = stream_fn(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            deadline = time.monotonic() + self.generation_timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                try:
                    chunk = await asyncio.wait_for(anext(stream_iter), timeout=remaining)
                except StopAsyncIteration:
                    break
                if not isinstance(chunk, str):
                    raise TypeError("Grounded model stream chunk must be text")
                if not chunk:
                    continue
                if first_token:
                    timings.first_token_ms = (time.perf_counter() - generation_start) * 1000
                    first_token = False
                chunks.append(chunk)
        finally:
            if stream_iter is not None:
                close = getattr(stream_iter, "aclose", None)
                if close is not None:
                    await close()

        timings.generation_ms = (time.perf_counter() - generation_start) * 1000
        raw_answer = _strip_think_blocks("".join(chunks))
        result_citations, answer_text, warnings = self._finalize_answer(
            raw_answer, packet, want_citations=want_citations
        )

        await _validate_sources(receipts)

        # 模型原始流可能含 think 块或引用标记；只有通过与非流式相同的
        # 校验和清理后才把正文交给客户端，避免先泄漏、后纠正。
        yield AnswerStreamEvent(type="delta", data={"text": answer_text})
        yield AnswerStreamEvent(type="citations", data={"citations": result_citations})
        yield AnswerStreamEvent(
            type="done",
            data=self._done_payload(
                answer=answer_text,
                packet=packet,
                decision=decision,
                warnings=warnings,
                timings=timings,
                start=start,
                model_invoked=True,
                model_id=model_id,
            ),
        )

    # ------------------------------------------------------------------
    # 内部编排
    # ------------------------------------------------------------------
    async def _retrieve(
        self,
        query: str,
        *,
        history: list[dict[str, str]] | None,
        domain_id: str | None,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if self.retriever is None:
            raise RuntimeError("Grounded answer retriever is not configured")
        resolved = await resolve_retrieval_query(query, history, domain_id=domain_id)
        if resolved is None:
            return None
        scope = {"question_binding": resolved.question_binding} if resolved.question_binding is not None else {}
        return await asyncio.to_thread(
            self.retriever,
            resolved.query,
            top_k=top_k,
            filters=filters,
            domain_id=domain_id,
            **scope,
        )

    async def _answer_from_bundle(
        self,
        bundle: dict[str, Any] | None,
        query: str,
        *,
        generate: GenerateFn,
        persona_prompt: str,
        speaker: str,
        history: list[dict[str, str]] | None,
        model_id: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        want_citations: bool,
        timings: AnswerTimings,
    ) -> GroundedAnswerResult:
        packet_start = time.perf_counter()
        decision = self.mode_decider.decide(bundle, persona=bool(persona_prompt))
        packet = self._build_packet(query, bundle, decision)
        timings.packet_ms = (time.perf_counter() - packet_start) * 1000

        metadata = self._retrieval_metadata(packet, decision, model_id=model_id)

        if packet.answer_mode == AnswerMode.NO_RAG:
            metadata["abstention_reason"] = FailureKind.NO_DOMAIN.value
            return GroundedAnswerResult(
                answer=self.abstention_reply,
                answer_mode=AnswerMode.ABSTENTION,
                domain_id=None,
                citations=[],
                confidence=None,
                abstained=True,
                warnings=["abstention:no_domain"],
                failure_kind=FailureKind.NO_DOMAIN.value,
                used_rag=False,
                model_invoked=False,
                model_id=model_id,
                retrieval_metadata=metadata,
                timings=timings,
            )

        if packet.answer_mode == AnswerMode.ABSTENTION:
            # abstention：结构上不调用模型（防绕过）；无答案也返回正常结构
            reason = packet.abstention_reason
            return GroundedAnswerResult(
                answer=self.abstention_reply,
                answer_mode=AnswerMode.ABSTENTION,
                domain_id=packet.domain_id,
                citations=[],
                confidence=packet.retrieval_confidence,
                abstained=True,
                warnings=[*packet.warnings, f"abstention:{reason}"],
                failure_kind="",
                used_rag=True,
                model_invoked=False,
                model_id=model_id,
                retrieval_metadata=metadata,
                timings=timings,
            )

        # grounded 生成
        prompt_start = time.perf_counter()
        messages = self.prompt_builder.build_messages(
            packet,
            persona_prompt=persona_prompt,
            domain_supplement=self._supplement_for(packet),
            speaker=speaker,
            history=history,
            max_tokens=max_tokens,
        )
        timings.prompt_ms = (time.perf_counter() - prompt_start) * 1000

        receipts = curated_source_receipts(bundle)
        await _validate_sources(receipts)
        generation_start = time.perf_counter()
        raw_answer = await asyncio.wait_for(
            generate(messages=messages, temperature=temperature, max_tokens=max_tokens, top_p=top_p),
            timeout=self.generation_timeout,
        )
        timings.generation_ms = (time.perf_counter() - generation_start) * 1000
        if not isinstance(raw_answer, str):
            raise TypeError("Grounded model output must be text")
        raw_answer = _strip_think_blocks(raw_answer)
        citations, answer_text, warnings = self._finalize_answer(raw_answer, packet, want_citations=want_citations)

        await _validate_sources(receipts)
        return GroundedAnswerResult(
            answer=answer_text,
            answer_mode=packet.answer_mode,
            domain_id=packet.domain_id,
            citations=citations,
            confidence=decision.effective_confidence,
            abstained=False,
            warnings=warnings,
            failure_kind="",
            used_rag=True,
            model_invoked=True,
            model_id=model_id,
            curated_source_receipts=receipts,
            retrieval_metadata=metadata,
            timings=timings,
        )

    def _finalize_answer(
        self,
        raw_answer: str,
        packet: EvidencePacket,
        *,
        want_citations: bool,
    ) -> tuple[list[dict[str, Any]], str, list[str]]:
        """citation 解析 → 校验 → 绑定 → 回答清理 → 后置检查。

        answer 一律移除 [S#] 标记（正文纯文本）；引用信息由结构化
        citations 携带（key 顺序 = 首次引用顺序）。
        """
        warnings = list(packet.warnings)
        validation = self.validator.validate(raw_answer, packet)
        warnings.extend(validation.warnings)

        answer_text = self.validator.sanitize_answer(raw_answer, validation)
        if not answer_text:
            raise RuntimeError("Grounded model returned empty answer")
        if not validation.valid_keys or validation.invalid_keys:
            raise RuntimeError("Grounded model answer failed citation validation")
        citations = self.validator.bind(validation, packet) if want_citations else []

        post = self.validator.post_check(answer_text, packet, answer_max_chars=self.answer_max_chars)
        warnings.extend(post)
        return citations, answer_text, warnings

    def _build_packet(
        self,
        query: str,
        bundle: dict[str, Any] | None,
        decision: ModeDecision,
    ) -> EvidencePacket:
        if bundle is None:
            return EvidencePacket(
                query=query,
                domain_id=None,
                answer_mode=AnswerMode.NO_RAG,
                abstention_reason=decision.abstention_reason,
                retrieval_confidence=0.0,
            )
        if not is_structured_retrieval_bundle(bundle):
            raise ValueError("Grounded answer requires a structured retrieval bundle with domains and query_analysis")
        packet = self.packet_builder.build(
            query,
            bundle,
            decision.mode,
            warnings=decision.warnings,
        )
        reason = decision.abstention_reason
        if not packet.documents:
            packet.answer_mode = AnswerMode.ABSTENTION
            if not reason:
                reason = (
                    FailureKind.CONTEXT_BUDGET_EXCEEDED.value if packet.truncated else FailureKind.NO_RETRIEVAL.value
                )
        if packet.answer_mode == AnswerMode.ABSTENTION:
            packet.abstention_reason = packet.abstention_reason or reason
            if not packet.abstention_reason:
                raise ValueError("Abstention requires an explicit evidence or decision reason")
        return packet

    def _supplement_for(self, packet: EvidencePacket) -> str:
        if self._domain_supplements is None or not packet.domain_id:
            return ""
        return self._domain_supplements(packet.domain_id) or ""

    def _retrieval_metadata(
        self,
        packet: EvidencePacket,
        decision: ModeDecision,
        *,
        model_id: str,
    ) -> dict[str, Any]:
        analysis = packet.query_analysis or {}
        return {
            "domain_id": packet.domain_id,
            "domains": list(analysis.get("matched_domains") or []),
            "confidence": round(decision.effective_confidence, 4),
            "abstention_reason": packet.abstention_reason,
            "retrieval_confidence": round(packet.retrieval_confidence, 4),
            "evidence_count": len(packet.documents),
            "evidence_chars": len(packet.context_text),
            "evidence_budget": packet.evidence_budget,
            "truncated": packet.truncated,
            "query_entities": list(analysis.get("entities") or []),
            "index_version": packet.index_version,
            "prompt_version": _prompt_version(),
            "model_id": model_id,
        }

    def _done_payload(
        self,
        *,
        answer: str,
        packet: EvidencePacket,
        decision: ModeDecision,
        warnings: list[str],
        timings: AnswerTimings,
        start: float,
        model_invoked: bool,
        model_id: str = "",
    ) -> dict[str, Any]:
        timings.total_ms = (time.perf_counter() - start) * 1000
        no_domain = packet.answer_mode == AnswerMode.NO_RAG
        effective_mode = AnswerMode.ABSTENTION if no_domain else packet.answer_mode
        return {
            "answer_mode": effective_mode.value,
            "domain_id": packet.domain_id,
            "answer": answer,
            "abstained": not model_invoked,
            "confidence": (
                None if no_domain else decision.effective_confidence if model_invoked else packet.retrieval_confidence
            ),
            "warnings": warnings,
            "model_invoked": model_invoked,
            "model_id": model_id,
            "failure_kind": FailureKind.NO_DOMAIN.value if no_domain else "",
            "used_rag": not no_domain,
            "costTime": round(timings.total_ms / 1000, 2),
            "retrieval_metadata": self._retrieval_metadata(packet, decision, model_id=model_id),
            "timings": timings.to_dict(),
        }

    @staticmethod
    def _wrap_as_stream(generate: GenerateFn) -> GenerateStreamFn:
        """将注入的一次性生成接口适配为流事件，不切换模型。"""

        async def _single_shot(**kwargs: Any) -> AsyncIterator[str]:
            text = await generate(**kwargs)
            yield text

        return _single_shot


def _prompt_version() -> str:
    from .prompt import GROUNDED_PROMPT_VERSION

    return GROUNDED_PROMPT_VERSION


_service: GroundedAnswerService | None = None
_service_budget: ProviderContextBudget | None = None
_service_lock = asyncio.Lock()


def build_default_grounded_answer_service(
    *, context_budget: ProviderContextBudget | None = None
) -> GroundedAnswerService:
    """默认服务实例：角色知识检索 + 进程内缓存 + env 配置。"""
    from knowledge.multiscale_rag.constants import INDEX_FORMAT_VERSION
    from knowledge.multiscale_rag.runtime import get_multiscale_rag_service

    context_budget = get_provider_context_budget() if context_budget is None else context_budget
    evidence_chars = context_budget.evidence_max_chars or None
    character_rag = get_multiscale_rag_service()

    def _retriever(query: str, *, top_k: int, filters: dict | None, domain_id: str | None, question_binding=None):
        from knowledge.public_domains import (
            all_bound_objects_generic,
            domain_plan,
            requires_independent_domains,
            retrieve_curated_scope,
        )

        if domain_id is not None and domain_id != character_rag.config.domain_id:
            raise ValueError(f"Unknown knowledge answer domain: {domain_id}")
        if question_binding is not None:
            plan = domain_plan(question_binding, character_rag.config)
            if requires_independent_domains(plan):
                raise ValueError("Independent knowledge answers do not support mixed source authorities")
            if all_bound_objects_generic(plan):
                return None
            if any(obj["authority"] == "curated_character" for obj in plan["objects"]):
                return retrieve_curated_scope(
                    plan, query, top_k, curated=partial(character_rag.retrieve_with_citations, filters=filters)
                )
        return character_rag.retrieve_with_citations(query, domain_id=domain_id, top_k=top_k, filters=filters)

    def _domain_supplement(domain_id: str) -> str:
        if domain_id != character_rag.config.domain_id:
            return ""
        return getattr(character_rag.config, "prompt_supplement", "") or ""

    def _index_version_resolver(domain_id: str | None) -> str:
        if domain_id is not None and domain_id != character_rag.config.domain_id:
            raise ValueError(f"Unknown knowledge answer domain: {domain_id}")
        return f"{character_rag.config.domain_id}:{INDEX_FORMAT_VERSION}"

    return GroundedAnswerService(
        packet_builder=EvidencePacketBuilder(evidence_chars),
        prompt_builder=GroundedPromptBuilder(
            evidence_chars, context_budget=ReviewContextBudget(context_budget.window_tokens)
        ),
        retriever=_retriever,
        domain_supplements=_domain_supplement,
        index_version_resolver=_index_version_resolver,
        abstention_reply=(os.getenv("GROUNDED_ABSTENTION_REPLY", "").strip() or DEFAULT_ABSTENTION_REPLY),
        generation_timeout=float(os.getenv("GROUNDED_GENERATION_TIMEOUT", "60")),
        answer_max_chars=int(os.getenv("GROUNDED_ANSWER_MAX_CHARS", "4000")),
    )


async def get_grounded_answer_service() -> GroundedAnswerService:
    """全局单例（异步锁，惰性构建）。"""
    global _service, _service_budget
    budget = get_provider_context_budget()
    if _service is None or _service_budget != budget:
        async with _service_lock:
            if _service is None or _service_budget != budget:
                _service = build_default_grounded_answer_service(context_budget=budget)
                _service_budget = budget
    return _service
