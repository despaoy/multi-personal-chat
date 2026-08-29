"""知识问答 API（Grounded Answer 业务入口）。

- POST /api/ask        非流式：answer + 绑定 citations + retrieval metadata
- POST /api/ask/stream 流式（SSE）：meta → delta* → citations → done/error

设计：
- 不绕过现有 Provider 抽象：生成函数优先复用共享 vLLM 客户端，
  不可用时回退 ModelManager（与 /api/generate 相同的优先级）
- 域门控：未命中知识域时返回 abstention（no_domain）
- 客户端断开：SSE 生成器随请求取消而终止，不继续模型工作
- 用户可见错误简洁；内部日志不含密钥/完整系统提示词
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.dependencies import get_current_user
from db.schemas import AskRequest, AskResponse
from knowledge.grounded_answer.service import get_grounded_answer_service
from knowledge.grounded_answer.validator import public_citation_view

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

router = APIRouter()


async def _resolve_generate_adapters() -> tuple[Any, Any, str]:
    """返回惰性生成适配器：vLLM 优先，ModelManager 回退。

    这里只构造闭包，不连接或初始化生成后端。证据不足、未命中域等
    不需要模型的请求因此不会被模型服务状态拖垮。

    Returns:
        (generate, generate_stream | None, model_id)
    """
    from app.config import get_vllm_served_model_name, is_vllm_enabled

    model_id = f"vllm/{get_vllm_served_model_name()}" if is_vllm_enabled() else "auto"
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

            client = await get_vllm_client()
            if client is not None:

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

            # 回退：ModelManager（无流式后端时整体产出一次）
            from inference.model_manager import get_model_manager

            manager = get_model_manager()

            async def manager_generate(
                *,
                messages: list[dict[str, str]],
                temperature: float,
                max_tokens: int,
                top_p: float,
            ) -> str:
                del temperature, top_p
                prompt = _flatten_messages(messages)
                reply, _cost = await manager.async_generate(
                    prompt,
                    session_history=None,
                    rag_docs=None,
                    max_tokens_override=max_tokens,
                )
                return reply

            resolved = (manager_generate, None)
            return resolved

    async def generate(**kwargs: Any) -> str:
        generate_fn, _ = await resolve()
        return await generate_fn(**kwargs)

    async def generate_stream(**kwargs: Any) -> AsyncIterator[str]:
        generate_fn, stream_fn = await resolve()
        if stream_fn is None:
            text = await generate_fn(**kwargs)
            if text:
                yield text
            return
        async for chunk in stream_fn(**kwargs):
            yield chunk

    return generate, generate_stream, model_id


def _flatten_messages(messages: list[dict[str, str]]) -> str:
    """ModelManager 回退路径：messages → 平铺 prompt（system 在前）。"""
    parts: list[str] = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role == "system":
            parts.append(content)
        elif role == "user":
            parts.append(f"用户：{content}")
        elif role == "assistant":
            parts.append(f"助手：{content}")
    return "\n\n".join(parts)


def _safe_citations(citations: list[dict[str, Any]], *, debug: bool) -> list[dict[str, Any]]:
    return [public_citation_view(c, debug=debug) for c in citations]


@router.post("/api/ask")
async def ask(request: AskRequest, current_user: dict = Depends(get_current_user)):
    """知识问答（非流式）。"""
    service = await get_grounded_answer_service()
    generate, _stream, model_id = await _resolve_generate_adapters()

    start = time.perf_counter()
    result = await service.answer(
        request.message,
        generate=generate,
        domain_id=request.domainId,
        top_k=request.topK,
        history=request.history or None,
        model_id=model_id,
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

    事件序列：meta → delta* → citations → done（中途失败为 error →
    citations(空) → done(degraded)）。citation metadata 在正文完成后
    统一发送，避免流式过程中引用映射变化。
    """
    service = await get_grounded_answer_service()
    generate, generate_stream, model_id = await _resolve_generate_adapters()
    debug = request.debug and current_user.get("role") == "admin"

    async def event_stream() -> AsyncIterator[bytes]:
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
            logger.warning("grounded_answer_stream failed: %s", type(e).__name__)
            yield "data: {}\n\n".format(
                json.dumps(
                    {"type": "error", "kind": "internal_error", "message": "服务内部错误"},
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
