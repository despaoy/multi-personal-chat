"""知识问答 API（Grounded Answer 业务入口）。

- POST /api/ask        非流式：answer + 绑定 citations + retrieval metadata
- POST /api/ask/stream 流式（SSE）：meta → delta* → citations → done/error

设计：
- 不绕过现有 Provider 抽象：生成函数使用当前选中的提供方；
  云端保持原始消息契约，本地选择才复用共享 vLLM 客户端
- 域门控：未命中知识域时返回 abstention（no_domain）
- 客户端断开：SSE 生成器随请求取消而终止，不继续模型工作
- 用户可见错误简洁；内部日志不含密钥/完整系统提示词
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.dependencies import get_current_user
from db.schemas import AskRequest, AskResponse
from inference.openai_protocol import ProviderResponseError
from inference.request_context import InferenceRequestContext, resolve_request_context, use_request_context
from knowledge.grounded_answer.service import get_grounded_answer_service
from knowledge.grounded_answer.validator import public_citation_view

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
    from inference.model_manager import get_model_manager

    manager = get_model_manager()
    selected = context.provider_name
    provider = context.provider if selected == "openai_compat" else manager.get_current_provider()
    status = provider.get_status()
    model_id = f"{selected}/{status.get('modelName') or selected}"
    identity = {"provider": selected, "status": status}
    if selected == "openai_compat":
        identity.update(endpoint=provider.base_url,
                        credential_sha256=hashlib.sha256(provider.api_key.encode()).hexdigest())
    cache_identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    resolved: tuple[Any, Any | None] | None = None
    resolve_lock = asyncio.Lock()

    async def resolve() -> tuple[Any, Any | None]:
        nonlocal resolved
        if resolved is not None:
            return resolved
        async with resolve_lock:
            if resolved is not None:
                return resolved

            from api.generate import get_vllm_client

            if selected == "openai_compat":

                async def cloud_generate(*, messages, temperature, max_tokens, top_p):
                    reply, _cost = await provider.async_complete(
                        messages, temperature=temperature, max_tokens=max_tokens, top_p=top_p
                    )
                    return reply

                resolved = (cloud_generate, None)
                return resolved

            if selected != "vllm":
                raise ValueError(f"Knowledge answers require openai_compat or vllm; selected provider: {selected}")
            client = await get_vllm_client()
            if client is None:
                raise RuntimeError("Selected vLLM client is unavailable; check VLLM_ENABLED and vLLM configuration")

            async def vllm_generate(**kwargs: Any) -> str:
                return await client.generate(**kwargs)

            async def vllm_stream(**kwargs: Any) -> AsyncIterator[str]:
                stream = await client.generate(**kwargs, stream=True)
                try:
                    async for chunk in stream:
                        yield chunk
                finally:
                    close = getattr(stream, "aclose", None)
                    if close is not None:
                        await close()

            resolved = (vllm_generate, vllm_stream)
            return resolved

    def check_provider():
        if manager._current_provider.value != selected:
            raise RuntimeError("Selected provider changed before knowledge generation; retry the request")

    async def generate(**kwargs: Any) -> str:
        generate_fn, _ = await resolve()
        check_provider()
        return await generate_fn(**kwargs)

    async def generate_stream(**kwargs: Any) -> AsyncIterator[str]:
        generate_fn, stream_fn = await resolve()
        check_provider()
        if stream_fn is None:
            text = await generate_fn(**kwargs)
            yield text
            return
        async for chunk in stream_fn(**kwargs):
            yield chunk

    return generate, generate_stream, model_id, cache_identity


def _safe_citations(citations: list[dict[str, Any]], *, debug: bool) -> list[dict[str, Any]]:
    return [public_citation_view(c, debug=debug) for c in citations]


@router.post("/api/ask")
async def ask(request: AskRequest, current_user: dict = Depends(get_current_user)):
    """知识问答（非流式）。"""
    from inference.model_manager import get_model_manager

    service = await get_grounded_answer_service()
    context = await resolve_request_context(get_model_manager())
    generate, _stream, model_id, cache_identity = await _resolve_generate_adapters(context)

    start = time.perf_counter()
    with use_request_context(context):
        result = await service.answer(
            request.message,
            generate=generate,
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
):
    """知识问答（SSE 流式）。

    事件序列：meta → delta* → citations → done；失败以 error 终止。
    citation metadata 在正文完成后
    统一发送，避免流式过程中引用映射变化。
    """
    from inference.model_manager import get_model_manager

    service = await get_grounded_answer_service()
    context = await resolve_request_context(get_model_manager())
    generate, generate_stream, model_id, _cache_identity = await _resolve_generate_adapters(context)
    debug = request.debug and current_user.get("role") == "admin"

    async def event_stream() -> AsyncIterator[bytes]:
        with use_request_context(context):
            stream = service.answer_stream(
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
            try:
                async for event in stream:
                    # 客户端断开：终止生成器（取消传播到底层 httpx 流）
                    if await http_request.is_disconnected():
                        logger.info("grounded_answer_stream client disconnected")
                        break
                    payload = dict(event.data)
                    if event.type == "citations":
                        payload["citations"] = _safe_citations(payload.get("citations") or [], debug=debug)
                    yield "data: {}\n\n".format(json.dumps({"type": event.type, **payload}, ensure_ascii=False)).encode(
                        "utf-8"
                    )
            except Exception as e:  # noqa: BLE001 - SSE 中途异常以 error 事件收尾
                detail = (e.public_detail() if isinstance(e, ProviderResponseError)
                          else {"kind": "internal_error", "message": "服务内部错误"})
                logger.warning("grounded_answer_stream failed: %s upstream_status=%s",
                               detail["kind"], detail.get("upstreamStatus"))
                yield "data: {}\n\n".format(
                    json.dumps(
                        {"type": "error", **detail},
                        ensure_ascii=False,
                    )
                ).encode("utf-8")
            finally:
                await stream.aclose()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
