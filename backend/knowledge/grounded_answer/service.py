"""P7 grounded-answer 服务编排。

链路：P6 检索 bundle → 模式识别 → EvidencePacket → grounded prompt
    → 生成（注入式 Provider）→ citation 解析/校验/绑定 → 统一结果

职责边界：
- 不绕过现有 Provider / 流式输出 / 业务服务：生成函数由调用方注入
  （vLLM 客户端 / ModelManager 适配器 / 评估用 DeepSeek 适配器）
- abstention 模式不调用模型（结构性防绕过）
- 模型失败时保留检索 evidence，降级为明确 abstention，
  不返回未经验证的半截结构
- 流式与非流式共用同一 citation 绑定逻辑，最终结构语义一致
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from .cache import AnswerCache, answer_cache_key
from .corrective import CorrectiveRetrievalAdapter
from .models import (  # noqa: E402
    AnswerMode,
    AnswerStreamEvent,
    AnswerTimings,
    EvidencePacket,
    FailureKind,
    GroundedAnswerResult,
)
from .modes import AnswerModeDecider, ModeDecision
from .packet import EvidencePacketBuilder, is_p6_bundle
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
# 与 inference.generation_request 的既有策略一致：有证据时温度收紧
_GROUNDED_TEMPERATURE_CAP = 0.5


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _strip_think_blocks(text: str) -> str:
    return _THINK_BLOCK_RE.sub("", text).strip()


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
        corrective_enabled: bool | None = None,
        corrective_max_retries: int = 1,
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
        self.corrective_enabled = (
            _env_flag("GROUNDED_ANSWER_CORRECTIVE", "true") if corrective_enabled is None else corrective_enabled
        )
        self.corrective_max_retries = corrective_max_retries
        self.abstention_reply = abstention_reply or DEFAULT_ABSTENTION_REPLY
        self.generation_timeout = float(generation_timeout)
        self.answer_max_chars = int(answer_max_chars)

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
        temperature: float = 0.3,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        want_citations: bool = True,
    ) -> GroundedAnswerResult:
        timings = AnswerTimings()
        start = time.perf_counter()

        # 缓存键需要 index_version：由注入的 resolver 提供（域配置快照，
        # 不触发检索）；未知时用 "unknown"，缓存仍与 prompt/模型绑定
        index_version = ""
        if self._index_version_resolver is not None:
            try:
                index_version = self._index_version_resolver(domain_id)
            except Exception:  # noqa: BLE001 - 缓存键成分失败不影响主链路
                index_version = ""

        if self.cache.ttl > 0 and index_version:
            key = answer_cache_key(
                domain_id=domain_id,
                domains=[domain_id] if domain_id else [],
                query=query,
                filters=filters,
                top_k=top_k,
                index_version=index_version,
                model_id=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                persona_prompt=persona_prompt,
                speaker=speaker,
                want_citations=want_citations,
            )
            cached = self.cache.get(key)
            if cached is not None:
                cached.cache_hit = True
                cached.timings.total_ms = (time.perf_counter() - start) * 1000
                return cached

        # 检索（P6 同步 CPU 检索放线程池）
        retrieval_start = time.perf_counter()
        try:
            bundle, corrective_info = await self._retrieve_with_optional_correction(
                query, domain_id=domain_id, top_k=top_k, filters=filters
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 检索故障转为明确降级结果
            timings.retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
            timings.total_ms = (time.perf_counter() - start) * 1000
            logger.warning("P7 检索失败（降级 abstention）: %s", type(exc).__name__)
            return GroundedAnswerResult(
                answer=self.abstention_reply,
                answer_mode=AnswerMode.ABSTENTION,
                domain_id=domain_id,
                citations=[],
                confidence=None,
                abstained=True,
                warnings=["degraded:retrieval_unavailable"],
                failure_kind=FailureKind.RETRIEVAL_UNAVAILABLE.value,
                used_rag=False,
                model_invoked=False,
                model_id=model_id,
                retrieval_metadata={"domain_id": domain_id, "evidence_count": 0},
                timings=timings,
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
            corrective_info=corrective_info,
            timings=timings,
        )
        # 独立知识问答没有 legacy 聊天链路可回退。未命中域时应返回
        # 明确弃答，而不是一个 answer 为空且 abstained=false 的结果。
        if result.answer_mode == AnswerMode.NO_RAG:
            result = GroundedAnswerResult(
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
                retrieval_metadata={
                    **result.retrieval_metadata,
                    "abstention_reason": FailureKind.NO_DOMAIN.value,
                },
                timings=timings,
            )
        timings.total_ms = (time.perf_counter() - start) * 1000

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
        top_p: float = 0.9,
        want_citations: bool = True,
    ) -> AsyncIterator[AnswerStreamEvent]:
        timings = AnswerTimings()
        start = time.perf_counter()

        retrieval_start = time.perf_counter()
        try:
            bundle, corrective_info = await self._retrieve_with_optional_correction(
                query, domain_id=domain_id, top_k=top_k, filters=filters
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - SSE 内返回结构化降级序列
            timings.retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
            logger.warning("P7 流式检索失败（降级 abstention）: %s", type(exc).__name__)
            packet = EvidencePacket(
                query=query,
                domain_id=domain_id,
                answer_mode=AnswerMode.ABSTENTION,
                retrieval_confidence=0.0,
                warnings=["degraded:retrieval_unavailable"],
            )
            yield AnswerStreamEvent(
                type="meta",
                data={"answer_mode": AnswerMode.ABSTENTION.value, "domain_id": domain_id},
            )
            yield AnswerStreamEvent(
                type="error",
                data={
                    "kind": FailureKind.RETRIEVAL_UNAVAILABLE.value,
                    "message": "知识检索暂不可用",
                },
            )
            yield AnswerStreamEvent(type="delta", data={"text": self.abstention_reply})
            yield AnswerStreamEvent(type="citations", data={"citations": []})
            yield AnswerStreamEvent(
                type="done",
                data=self._done_payload(
                    answer=self.abstention_reply,
                    packet=packet,
                    citations=[],
                    warnings=list(packet.warnings),
                    timings=timings,
                    start=start,
                    model_invoked=False,
                    corrective_info=None,
                    model_id=model_id,
                    abstained=True,
                    failure_kind=FailureKind.RETRIEVAL_UNAVAILABLE.value,
                    used_rag=False,
                ),
            )
            return
        timings.retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        decision = self.mode_decider.decide(bundle, persona=bool(persona_prompt))
        packet_start = time.perf_counter()
        packet = self._build_packet(query, bundle, decision)
        timings.packet_ms = (time.perf_counter() - packet_start) * 1000

        # meta 事件先行：客户端尽早得知模式与域
        yield AnswerStreamEvent(
            type="meta",
            data={
                "answer_mode": packet.answer_mode.value,
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
                    citations=[],
                    warnings=["abstention:no_domain"],
                    timings=timings,
                    start=start,
                    model_invoked=False,
                    corrective_info=corrective_info,
                    model_id=model_id,
                    abstained=True,
                    failure_kind=FailureKind.NO_DOMAIN.value,
                    answer_mode=AnswerMode.ABSTENTION,
                ),
            )
            return

        if packet.answer_mode == AnswerMode.ABSTENTION or not packet.documents:
            yield AnswerStreamEvent(type="delta", data={"text": self.abstention_reply})
            yield AnswerStreamEvent(type="citations", data={"citations": []})
            yield AnswerStreamEvent(
                type="done",
                data=self._done_payload(
                    answer=self.abstention_reply,
                    packet=packet,
                    citations=[],
                    warnings=[
                        *packet.warnings,
                        f"abstention:{packet.abstention_reason or FailureKind.LOW_CONFIDENCE.value}",
                    ],
                    timings=timings,
                    start=start,
                    model_invoked=False,
                    corrective_info=corrective_info,
                    model_id=model_id,
                    abstained=True,
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
        )
        timings.prompt_ms = (time.perf_counter() - prompt_start) * 1000

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
                if not chunk:
                    continue
                if first_token:
                    timings.first_token_ms = (time.perf_counter() - generation_start) * 1000
                    first_token = False
                chunks.append(str(chunk))
        except asyncio.CancelledError:
            logger.info("P7 流式生成被客户端取消 query_len=%d", len(query))
            raise
        except TimeoutError:
            logger.warning("P7 流式生成超时（降级 abstention）query_len=%d", len(query))
            yield AnswerStreamEvent(
                type="error",
                data={
                    "kind": FailureKind.GENERATION_TIMEOUT.value,
                    "message": "生成服务暂不可用",
                },
            )
            yield AnswerStreamEvent(type="citations", data={"citations": []})
            yield AnswerStreamEvent(
                type="done",
                data=self._done_payload(
                    answer=self.abstention_reply,
                    packet=packet,
                    citations=[],
                    warnings=[*packet.warnings, "generation_degraded"],
                    timings=timings,
                    start=start,
                    model_invoked=True,
                    corrective_info=corrective_info,
                    model_id=model_id,
                    abstained=True,
                    failure_kind=FailureKind.GENERATION_TIMEOUT.value,
                ),
            )
            return
        except Exception as e:  # noqa: BLE001 - 中途失败不得返回伪完整 citations
            logger.warning("P7 流式生成失败（降级 abstention）: %s", e)
            yield AnswerStreamEvent(
                type="error",
                data={
                    "kind": self._classify_generation_error(e).value,
                    "message": "生成服务暂不可用",
                },
            )
            yield AnswerStreamEvent(type="citations", data={"citations": []})
            yield AnswerStreamEvent(
                type="done",
                data=self._done_payload(
                    answer=self.abstention_reply,
                    packet=packet,
                    citations=[],
                    warnings=[*packet.warnings, "generation_degraded"],
                    timings=timings,
                    start=start,
                    model_invoked=True,
                    corrective_info=corrective_info,
                    model_id=model_id,
                    abstained=True,
                    failure_kind=self._classify_generation_error(e).value,
                ),
            )
            return
        finally:
            if stream_iter is not None:
                close = getattr(stream_iter, "aclose", None)
                if close is not None:
                    await close()

        timings.generation_ms = (time.perf_counter() - generation_start) * 1000
        raw_answer = _strip_think_blocks("".join(chunks))
        result_citations, answer_text, warnings, citation_valid = self._finalize_answer(
            raw_answer, packet, want_citations=want_citations
        )

        if not citation_valid:
            yield AnswerStreamEvent(
                type="error",
                data={
                    "kind": FailureKind.INVALID_CITATION.value,
                    "message": "回答未能绑定有效资料来源",
                },
            )
            yield AnswerStreamEvent(type="delta", data={"text": self.abstention_reply})
            yield AnswerStreamEvent(type="citations", data={"citations": []})
            yield AnswerStreamEvent(
                type="done",
                data=self._done_payload(
                    answer=self.abstention_reply,
                    packet=packet,
                    citations=[],
                    warnings=warnings,
                    timings=timings,
                    start=start,
                    model_invoked=True,
                    corrective_info=corrective_info,
                    model_id=model_id,
                    abstained=True,
                    failure_kind=FailureKind.INVALID_CITATION.value,
                ),
            )
            return

        # 模型原始流可能含 think 块或引用标记；只有通过与非流式相同的
        # 校验和清理后才把正文交给客户端，避免先泄漏、后纠正。
        yield AnswerStreamEvent(type="delta", data={"text": answer_text})
        yield AnswerStreamEvent(type="citations", data={"citations": result_citations})
        yield AnswerStreamEvent(
            type="done",
            data=self._done_payload(
                answer=answer_text,
                packet=packet,
                citations=result_citations,
                warnings=warnings,
                timings=timings,
                start=start,
                model_invoked=True,
                corrective_info=corrective_info,
                model_id=model_id,
                abstained=False,
            ),
        )

    # ------------------------------------------------------------------
    # 内部编排
    # ------------------------------------------------------------------
    async def _retrieve_with_optional_correction(
        self,
        query: str,
        *,
        domain_id: str | None,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        if self.retriever is None:
            return None, {"reformulated": False, "reformulated_query": None, "rounds": []}

        def _retrieve(q: str, *, top_k: int, filters: dict | None):
            return self.retriever(q, top_k=top_k, filters=filters, domain_id=domain_id)

        if not self.corrective_enabled:
            bundle = await asyncio.to_thread(_retrieve, query, top_k=top_k, filters=filters)
            return bundle, {"reformulated": False, "reformulated_query": None, "rounds": []}

        adapter = CorrectiveRetrievalAdapter(
            lambda q, top_k, filters: self.retriever(q, top_k=top_k, filters=filters, domain_id=domain_id),
            max_retries=self.corrective_max_retries,
        )
        bundle, info = await asyncio.to_thread(adapter.retrieve_with_correction, query, top_k=top_k, filters=filters)
        return bundle, info

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
        corrective_info: dict[str, Any] | None,
        timings: AnswerTimings,
    ) -> GroundedAnswerResult:
        packet_start = time.perf_counter()
        decision = self.mode_decider.decide(bundle, persona=bool(persona_prompt))
        packet = self._build_packet(query, bundle, decision)
        timings.packet_ms = (time.perf_counter() - packet_start) * 1000

        metadata = self._retrieval_metadata(packet, decision, corrective_info, model_id=model_id)

        if packet.answer_mode == AnswerMode.NO_RAG:
            # 未命中域：调用方（聊天链路）应回退既有检索；此处返回
            # no_rag 结果，answer 为空、不调用模型
            return GroundedAnswerResult(
                answer="",
                answer_mode=AnswerMode.NO_RAG,
                domain_id=None,
                citations=[],
                confidence=None,
                abstained=False,
                warnings=[],
                failure_kind=FailureKind.NO_DOMAIN.value,
                used_rag=False,
                model_invoked=False,
                model_id=model_id,
                retrieval_metadata=metadata,
                timings=timings,
            )

        if packet.answer_mode == AnswerMode.ABSTENTION or not packet.documents:
            # abstention：结构上不调用模型（防绕过）；无答案也返回正常结构
            reason = packet.abstention_reason or FailureKind.LOW_CONFIDENCE.value
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
                retrieval_metadata={**metadata, "abstention_reason": reason},
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
        )
        timings.prompt_ms = (time.perf_counter() - prompt_start) * 1000

        generation_start = time.perf_counter()
        try:
            raw_answer = await asyncio.wait_for(
                generate(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                ),
                timeout=self.generation_timeout,
            )
        except TimeoutError:
            timings.generation_ms = (time.perf_counter() - generation_start) * 1000
            logger.warning("P7 生成超时（降级 abstention）query_len=%d", len(query))
            return self._degraded_result(
                packet,
                decision,
                corrective_info,
                model_id,
                FailureKind.GENERATION_TIMEOUT,
                timings,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - Provider 故障保留 evidence、明确降级
            timings.generation_ms = (time.perf_counter() - generation_start) * 1000
            logger.warning("P7 生成失败（降级 abstention）: %s", type(e).__name__)
            return self._degraded_result(
                packet,
                decision,
                corrective_info,
                model_id,
                self._classify_generation_error(e),
                timings,
            )
        timings.generation_ms = (time.perf_counter() - generation_start) * 1000

        raw_answer = _strip_think_blocks(str(raw_answer or ""))
        if not raw_answer.strip():
            return self._degraded_result(
                packet,
                decision,
                corrective_info,
                model_id,
                FailureKind.INVALID_MODEL_OUTPUT,
                timings,
            )

        citations, answer_text, warnings, citation_valid = self._finalize_answer(
            raw_answer, packet, want_citations=want_citations
        )
        if not citation_valid:
            return self._degraded_result(
                packet,
                decision,
                corrective_info,
                model_id,
                FailureKind.INVALID_CITATION,
                timings,
                extra_warnings=warnings,
            )

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
            retrieval_metadata=metadata,
            timings=timings,
        )

    def _finalize_answer(
        self,
        raw_answer: str,
        packet: EvidencePacket,
        *,
        want_citations: bool,
    ) -> tuple[list[dict[str, Any]], str, list[str], bool]:
        """citation 解析 → 校验 → 绑定 → 回答清理 → 后置检查。

        answer 一律移除 [S#] 标记（正文纯文本）；引用信息由结构化
        citations 携带（key 顺序 = 首次引用顺序）。
        """
        warnings = list(packet.warnings)
        validation = self.validator.validate(raw_answer, packet)
        warnings.extend(validation.warnings)

        citations = self.validator.bind(validation, packet) if want_citations else []
        answer_text = self.validator.sanitize_answer(raw_answer, validation)

        post = self.validator.post_check(answer_text, packet, answer_max_chars=self.answer_max_chars)
        warnings.extend(post)
        citation_valid = bool(validation.valid_keys) and not validation.invalid_keys
        return citations, answer_text, warnings, citation_valid

    def _build_packet(
        self,
        query: str,
        bundle: dict[str, Any] | None,
        decision: ModeDecision,
    ) -> EvidencePacket:
        if bundle is None or not is_p6_bundle(bundle):
            mode = AnswerMode.NO_RAG if bundle is None else decision.mode
            return EvidencePacket(
                query=query,
                domain_id=None,
                answer_mode=mode,
                retrieval_confidence=0.0,
                warnings=["non_p6_bundle"] if bundle is not None else [],
            )
        return self.packet_builder.build(
            query,
            bundle,
            decision.mode,
            warnings=decision.warnings,
        )

    def _supplement_for(self, packet: EvidencePacket) -> str:
        if self._domain_supplements is None or not packet.domain_id:
            return ""
        try:
            return self._domain_supplements(packet.domain_id) or ""
        except Exception:  # noqa: BLE001 - 域补充规则失败不影响主链路
            return ""

    def _degraded_result(
        self,
        packet: EvidencePacket,
        decision: ModeDecision,
        corrective_info: dict[str, Any] | None,
        model_id: str,
        failure: FailureKind,
        timings: AnswerTimings,
        extra_warnings: list[str] | None = None,
    ) -> GroundedAnswerResult:
        """模型失败降级：保留检索 evidence（metadata），明确 abstention。"""
        metadata = self._retrieval_metadata(packet, decision, corrective_info, model_id=model_id)
        return GroundedAnswerResult(
            answer=self.abstention_reply,
            answer_mode=AnswerMode.ABSTENTION,
            domain_id=packet.domain_id,
            citations=[],
            confidence=packet.retrieval_confidence,
            abstained=True,
            warnings=list(
                dict.fromkeys(
                    [
                        *packet.warnings,
                        *(extra_warnings or []),
                        f"degraded:{failure.value}",
                    ]
                )
            ),
            failure_kind=failure.value,
            used_rag=True,
            model_invoked=True,
            model_id=model_id,
            retrieval_metadata={**metadata, "degraded_reason": failure.value},
            timings=timings,
        )

    def _retrieval_metadata(
        self,
        packet: EvidencePacket,
        decision: ModeDecision,
        corrective_info: dict[str, Any] | None,
        *,
        model_id: str,
    ) -> dict[str, Any]:
        analysis = packet.query_analysis or {}
        metadata: dict[str, Any] = {
            "domain_id": packet.domain_id,
            "domains": list(analysis.get("matched_domains") or []),
            "confidence": round(decision.effective_confidence, 4),
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
        if corrective_info:
            metadata["corrective"] = {
                "reformulated": bool(corrective_info.get("reformulated")),
                "reformulated_query": corrective_info.get("reformulated_query"),
                "rounds": len(corrective_info.get("rounds") or []),
            }
        return metadata

    def _done_payload(
        self,
        *,
        answer: str,
        packet: EvidencePacket,
        citations: list[dict[str, Any]],
        warnings: list[str],
        timings: AnswerTimings,
        start: float,
        model_invoked: bool,
        corrective_info: dict[str, Any] | None,
        model_id: str = "",
        abstained: bool | None = None,
        failure_kind: str = "",
        answer_mode: AnswerMode | None = None,
        used_rag: bool | None = None,
    ) -> dict[str, Any]:
        timings.total_ms = (time.perf_counter() - start) * 1000
        effective_mode = answer_mode or packet.answer_mode
        decision = ModeDecision(mode=effective_mode)
        return {
            "answer_mode": effective_mode.value,
            "domain_id": packet.domain_id,
            "answer": answer,
            "abstained": (
                abstained
                if abstained is not None
                else not model_invoked and packet.answer_mode == AnswerMode.ABSTENTION
            ),
            "confidence": packet.retrieval_confidence,
            "warnings": warnings,
            "model_invoked": model_invoked,
            "model_id": model_id,
            "failure_kind": failure_kind,
            "used_rag": (used_rag if used_rag is not None else packet.answer_mode != AnswerMode.NO_RAG),
            "costTime": round(timings.total_ms / 1000, 2),
            "retrieval_metadata": self._retrieval_metadata(packet, decision, corrective_info, model_id=model_id),
            "timings": timings.to_dict(),
        }

    @staticmethod
    def _wrap_as_stream(generate: GenerateFn) -> GenerateStreamFn:
        """无流式后端时降级：一次性生成后整体产出（保持事件契约一致）。"""

        async def _single_shot(**kwargs: Any) -> AsyncIterator[str]:
            text = await generate(**kwargs)
            if text:
                yield text

        return _single_shot

    @staticmethod
    def _classify_generation_error(exc: Exception) -> FailureKind:
        name = type(exc).__name__.lower()
        if any(token in name for token in ("connect", "timeout", "remote", "protocol")):
            return FailureKind.PROVIDER_UNAVAILABLE
        return FailureKind.INTERNAL_ERROR


def _prompt_version() -> str:
    from .prompt import GROUNDED_PROMPT_VERSION

    return GROUNDED_PROMPT_VERSION


_service: GroundedAnswerService | None = None
_service_lock = asyncio.Lock()


def build_default_grounded_answer_service() -> GroundedAnswerService:
    """默认服务实例：P6 检索 + 进程内缓存 + env 配置。"""
    from knowledge.rag_pipeline.service import get_rag_pipeline_service

    p6 = get_rag_pipeline_service()
    registry = p6.pipeline.registry

    def _retriever(query: str, *, top_k: int, filters: dict | None, domain_id: str | None):
        return p6.pipeline.retrieve(
            query,
            domain_id=domain_id,
            top_k=top_k,
            filters=filters,
            mode="hybrid",
            use_rerank=True,
            use_context=True,
        )

    def _domain_supplement(domain_id: str) -> str:
        config = registry.get(domain_id)
        if config is None:
            return ""
        return getattr(config, "prompt_supplement", "") or ""

    def _index_version_resolver(domain_id: str | None) -> str:
        if domain_id:
            config = registry.get(domain_id)
            if config is None:
                return "unknown"
            return f"{config.domain_id}:{config.index_version}"
        return ",".join(f"{c.domain_id}:{c.index_version}" for c in registry.list_domains(enabled_only=True))

    return GroundedAnswerService(
        retriever=_retriever,
        domain_supplements=_domain_supplement,
        index_version_resolver=_index_version_resolver,
        abstention_reply=(os.getenv("GROUNDED_ABSTENTION_REPLY", "").strip() or DEFAULT_ABSTENTION_REPLY),
        generation_timeout=float(os.getenv("GROUNDED_GENERATION_TIMEOUT", "60")),
        answer_max_chars=int(os.getenv("GROUNDED_ANSWER_MAX_CHARS", "4000")),
    )


async def get_grounded_answer_service() -> GroundedAnswerService:
    """全局单例（异步锁，惰性构建）。"""
    global _service
    if _service is None:
        async with _service_lock:
            if _service is None:
                _service = build_default_grounded_answer_service()
    return _service


def reset_grounded_answer_service() -> None:
    """重置单例（测试用）。"""
    global _service
    _service = None
