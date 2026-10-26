"""知识问答 API（Grounded Answer 业务入口）。

- POST /api/ask        非流式：answer + 绑定 citations + retrieval metadata
- POST /api/ask/stream 流式（SSE）：meta → delta* → citations → done/error

设计：
- 不绕过现有 Provider 抽象：生成函数使用当前选中的提供方；
  云端保持原始消息契约，本地选择才复用共享 vLLM 客户端
- 域门控：未命中知识域时返回 abstention（no_domain）
- 客户端断开：取消本地等待并关闭已打开的上游流
- 用户可见错误简洁；内部日志不含密钥/完整系统提示词
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from contextlib import aclosing
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.dependencies import get_current_user, get_request_database
from db.schemas import AskRequest, AskResponse
from inference.context_budget import InputBudgetError
from inference.openai_protocol import ProviderResponseError
from inference.request_context import InferenceRequestContext, resolve_request_context, use_request_context
from inference.usage import begin_completion
from knowledge.grounded_answer.service import get_grounded_answer_service
from knowledge.grounded_answer.validator import public_citation_view
from services.model_usage import UsageRecordingError, account_usage

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

router = APIRouter()


async def _resolve_generate_adapters(context: InferenceRequestContext) -> tuple[Any, Any, str, str]:
    """返回所选提供方的惰性生成适配器，保留编排好的消息与参数。

    这里只构造闭包，不连接或初始化生成后端。证据不足、未命中域等
    不需要模型的请求因此不会被模型服务状态拖垮。

    Returns:
        (generate, generate_stream, model_id, cache_identity)
    """
    selected = context.provider_name
    provider = context.provider
    status = provider.get_status()
    model_id = f"{selected}/{status.get('modelName') or selected}"
    identity = {"provider": selected, "status": status}
    if selected == "openai_compat":
        identity.update(
            endpoint=provider.base_url, credential_sha256=hashlib.sha256(provider.api_key.encode()).hexdigest()
        )
    cache_identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if selected == "openai_compat":

        async def cloud_generate(*, messages, temperature, max_tokens, top_p):
            reply, _cost = await provider.async_complete(
                messages, temperature=temperature, max_tokens=max_tokens, top_p=top_p
            )
            return reply

        # The answer service owns one-shot-to-SSE adaptation and full validation.
        return cloud_generate, None, model_id, cache_identity

    client = None
    resolve_lock = asyncio.Lock()

    async def resolve_client():
        nonlocal client
        if client is None:
            async with resolve_lock:
                if client is None:
                    if selected != "vllm":
                        raise ValueError(
                            f"Knowledge answers require openai_compat or vllm; selected provider: {selected}"
                        )
                    from app.providers import get_vllm_client

                    client = await get_vllm_client()
                    if client is None:
                        raise RuntimeError(
                            "Selected vLLM client is unavailable; check VLLM_ENABLED and vLLM configuration"
                        )
        return client

    async def generate(**kwargs: Any) -> str:
        selected_client = await resolve_client()
        begin_completion()
        return await selected_client.generate(**kwargs)

    async def generate_stream(**kwargs: Any) -> AsyncIterator[str]:
        selected_client = await resolve_client()
        begin_completion()
        stream = await selected_client.generate(**kwargs, stream=True)
        try:
            async for chunk in stream:
                yield chunk
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()

    return generate, generate_stream, model_id, cache_identity


def _safe_citations(citations: list[dict[str, Any]], *, debug: bool) -> list[dict[str, Any]]:
    return [public_citation_view(c, debug=debug) for c in citations]


def _usage_metadata(model_id, current_user):
    identity = current_user["user_id"] if current_user["user_id"] is not None else current_user["username"]
    return {
        "traceId": "ask:" + uuid.uuid4().hex,
        "platform": "web",
        "sessionId": f"ask:{identity}",
        "purpose": "knowledge_answer",
        "modelName": model_id,
        "usedRag": True,
    }


@router.post("/api/ask")
async def ask(
    request: AskRequest, current_user: dict = Depends(get_current_user), database=Depends(get_request_database)
):
    """知识问答（非流式）。"""
    from inference.model_manager import get_model_manager

    context = await resolve_request_context(get_model_manager(database=database))
    generate, _stream, model_id, cache_identity = await _resolve_generate_adapters(context)

    start = time.perf_counter()
    with use_request_context(context):
        async with account_usage(database, _usage_metadata(model_id, current_user)) as confirm_usage:
            service = await get_grounded_answer_service()
            result = await service.answer(
                request.message,
                generate=generate,
                before_publish=confirm_usage,
                domain_id=request.domainId,
                top_k=request.topK,
                history=request.history or None,
                model_id=model_id,
                cache_identity=cache_identity,
                temperature=request.temperature,
                max_tokens=request.maxTokens,
                top_p=request.topP,
                want_citations=request.wantCitations,
            )
    cost_time = time.perf_counter() - start
    logger.info(
        "grounded_answer domain=%s mode=%s abstained=%s citations=%d cost=%.2fs",
        result.domain_id,
        result.answer_mode.value,
        result.abstained,
        len(result.citations),
        cost_time,
    )
    return AskResponse(
        answer=result.answer,
        answerMode=result.answer_mode.value,
        domainId=result.domain_id,
        citations=result.api_citations(debug=request.debug and current_user.get("role") == "admin"),
        confidence=(round(result.confidence, 4) if result.confidence is not None else None),
        abstained=result.abstained,
        warnings=list(result.warnings),
        model=model_id,
        cacheHit=result.cache_hit,
        modelInvoked=result.model_invoked,
        costTime=round(cost_time, 2),
        retrievalMetadata=result.retrieval_metadata,
    )


@router.post("/api/ask/stream")
async def ask_stream(
    request: AskRequest,
    http_request: Request,
    current_user: dict = Depends(get_current_user),
    database=Depends(get_request_database),
):
    """知识问答（SSE 流式）。

    事件序列：meta → delta* → citations → done；失败以 error 终止。
    citation metadata 在正文完成后
    统一发送，避免流式过程中引用映射变化。
    """
    from inference.model_manager import get_model_manager

    context = await resolve_request_context(get_model_manager(database=database))
    generate, generate_stream, model_id, _cache_identity = await _resolve_generate_adapters(context)
    debug = request.debug and current_user.get("role") == "admin"

    async def event_stream() -> AsyncIterator[bytes]:
        with use_request_context(context):
            try:
                terminal_event = None
                async with account_usage(database, _usage_metadata(model_id, current_user)):
                    service = await get_grounded_answer_service()
                    async with aclosing(
                        service.answer_stream(
                            request.message,
                            generate=generate,
                            generate_stream=generate_stream,
                            domain_id=request.domainId,
                            top_k=request.topK,
                            history=request.history or None,
                            model_id=model_id,
                            temperature=request.temperature,
                            max_tokens=request.maxTokens,
                            top_p=request.topP,
                            want_citations=request.wantCitations,
                        )
                    ) as stream:
                        async for event in stream:
                            # 客户端断开：终止生成器（取消传播到底层 httpx 流）
                            if await http_request.is_disconnected():
                                logger.info("grounded_answer_stream client disconnected")
                                break
                            payload = dict(event.data)
                            if event.type == "citations":
                                payload["citations"] = _safe_citations(payload.get("citations") or [], debug=debug)
                            encoded = "data: {}\n\n".format(
                                json.dumps({"type": event.type, **payload}, ensure_ascii=False)
                            ).encode("utf-8")
                            if event.type == "done":
                                terminal_event = encoded
                            else:
                                yield encoded
                if terminal_event is not None:
                    yield terminal_event
            except Exception as e:  # noqa: BLE001 - SSE 中途异常以 error 事件收尾
                detail = (
                    e.public_detail()
                    if isinstance(e, (ProviderResponseError, InputBudgetError, UsageRecordingError))
                    else {"kind": "internal_error", "message": "服务内部错误"}
                )
                logger.warning(
                    "grounded_answer_stream failed: %s upstream_status=%s", detail["kind"], detail.get("upstreamStatus")
                )
                yield "data: {}\n\n".format(
                    json.dumps(
                        {"type": "error", **detail},
                        ensure_ascii=False,
                    )
                ).encode("utf-8")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
