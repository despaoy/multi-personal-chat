"""AstrBot integration endpoints.

AstrBot acts as the multi-platform gateway. This module accepts normalized
message events from the AstrBot plugin and delegates generation to the existing
FastAPI brain.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from typing import Any, Callable, Literal, TypeVar

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

from api.generate import generate_reply_core
from db.adapter import db
from app.runtime import get_runtime_container
from db.schemas import MessageRequest
from infra.concurrency_control import InferenceQueueFull, RateLimitExceeded, inference_runtime
from infra.observability import increment, log_event
from infra.security_utils import (
    constant_time_contains,
    is_production,
    raw_json_size,
    redact_sensitive,
    split_tokens,
    strip_control_chars,
    verify_integration_signature,
)

router = APIRouter()
logger = logging.getLogger(__name__)

Platform = Literal["qq", "telegram", "wecom", "wechat_official", "wechat_personal"]
ConversationType = Literal["private", "group", "channel"]
T = TypeVar("T")

_AUTH_TIMEOUT = float(os.getenv("INTEGRATION_AUTH_TIMEOUT", "3"))
_DB_TIMEOUT = float(os.getenv("INTEGRATION_DB_TIMEOUT", "3"))
_MODEL_TIMEOUT = float(os.getenv("MODEL_INFERENCE_TIMEOUT", "180"))
_RAW_MAX_BYTES = int(os.getenv("ASTRBOT_RAW_MAX_BYTES", "65536"))
_SIGNATURE_SKEW_SECONDS = int(os.getenv("INTEGRATION_SIGNATURE_SKEW_SECONDS", "300"))

_PLATFORM_SETTINGS: dict[str, tuple[str, str, bool]] = {
    "qq": ("ASTRBOT_QQ_ENABLED", "astrbotQQEnabled", True),
    "telegram": ("ASTRBOT_TELEGRAM_ENABLED", "astrbotTelegramEnabled", True),
    "wecom": ("ASTRBOT_WECOM_ENABLED", "astrbotWecomEnabled", True),
    "wechat_official": ("ASTRBOT_WECHAT_OFFICIAL_ENABLED", "astrbotWechatOfficialEnabled", True),
    "wechat_personal": ("ASTRBOT_WECHAT_PERSONAL_ENABLED", "astrbotWechatPersonalEnabled", False),
}


class AstrBotMessageRequest(BaseModel):
    platform: Platform
    adapter: str = Field(default="other", max_length=64)
    messageId: str = Field(default="", max_length=256)
    conversationId: str = Field(..., min_length=1, max_length=256)
    conversationType: ConversationType = "private"
    senderId: str = Field(default="", max_length=256)
    senderName: str = Field(default="", max_length=256)
    text: str = Field(..., min_length=1, max_length=8000)
    raw: dict[str, Any] = Field(default_factory=dict)


class AstrBotMessageResponse(BaseModel):
    shouldReply: bool
    replyText: str = ""
    model: str = ""
    costTime: float = 0.0
    traceId: str


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


async def _run_blocking(label: str, func: Callable[[], T], timeout: float = _DB_TIMEOUT) -> T:
    try:
        return await asyncio.wait_for(asyncio.to_thread(func), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("AstrBot integration step timed out: %s", label)
        raise
    except Exception:
        logger.warning("AstrBot integration step failed: %s", label, exc_info=True)
        raise


async def _load_config(timeout: float = _DB_TIMEOUT) -> dict[str, Any]:
    try:
        config = await _run_blocking("load-config", lambda: db.config, timeout=timeout)
        return config if isinstance(config, dict) else {}
    except Exception:
        return {}


def _allowed_tokens() -> list[str]:
    """Load integration secrets only from server-side environment variables."""
    tokens: list[str] = []
    tokens.extend(split_tokens(os.getenv("ASTRBOT_INTEGRATION_TOKEN")))
    tokens.extend(split_tokens(os.getenv("ASTRBOT_INTEGRATION_TOKENS")))
    return list(dict.fromkeys(token for token in tokens if token))


def _signature_required() -> bool:
    configured = os.getenv("INTEGRATION_SIGNATURE_REQUIRED")
    if configured is not None:
        return _parse_bool(configured, False)
    return is_production()


async def _check_token(token: str | None) -> tuple[str, list[str]]:
    allowed_tokens = _allowed_tokens()
    if not allowed_tokens:
        if is_production():
            raise HTTPException(status_code=503, detail="AstrBot integration token is not configured")
        return "", []
    if not constant_time_contains(token, allowed_tokens):
        raise HTTPException(status_code=401, detail="Invalid AstrBot integration token")
    return token or "", allowed_tokens


async def _is_platform_enabled(platform: str) -> bool:
    global_env = os.getenv("ASTRBOT_ENABLED")
    if global_env is not None and not _parse_bool(global_env, True):
        return False

    config = await _load_config()
    if global_env is None and not _parse_bool(config.get("astrbotEnabled"), True):
        return False

    env_key, config_key, default = _PLATFORM_SETTINGS.get(platform, ("", "", True))
    env_value = os.getenv(env_key) if env_key else None
    if env_value is not None:
        return _parse_bool(env_value, default)
    return _parse_bool(config.get(config_key), default)


def _session_id(platform: str, conversation_type: str, conversation_id: str) -> str:
    return f"{platform}:{conversation_type}:{conversation_id}"


def _validate_and_normalize_request(request: AstrBotMessageRequest) -> str:
    text = strip_control_chars(request.text).strip()
    if not text:
        raise HTTPException(status_code=422, detail="Message text is empty after sanitization")
    if len(text) > 8000:
        raise HTTPException(status_code=422, detail="Message text is too long")
    if raw_json_size(request.raw) > _RAW_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Raw event payload is too large")
    return text


def _is_sensitive_admin_request(text: str) -> bool:
    lowered = text.lower()
    blocked = [
        "export config",
        "dump config",
        "read secret",
        "show secret",
        "read token",
        "show token",
        "read .env",
        "cat .env",
        "print env",
        "ignore previous instructions and export",
        "\u5bfc\u51fa\u914d\u7f6e",
        "\u8bfb\u53d6\u5bc6\u94a5",
        "\u663e\u793a\u5bc6\u94a5",
        "\u8bfb\u53d6token",
        "\u663e\u793atoken",
        "\u8bfb\u53d6.env",
        "\u5ffd\u7565\u4e4b\u524d\u6307\u4ee4\u5e76\u5bfc\u51fa",
    ]
    return any(pattern in lowered for pattern in blocked)


def _dedup_message_id(request: AstrBotMessageRequest, text: str) -> str:
    if request.messageId.strip():
        return request.messageId.strip()
    raw_timestamp = request.raw.get("timestamp") or request.raw.get("time") or request.raw.get("message_time")
    bucket = str(raw_timestamp or int(time.time() // 60))
    basis = "|".join([
        request.platform,
        request.adapter or "other",
        request.conversationType,
        request.conversationId,
        request.senderId,
        bucket,
        text,
    ])
    return "fallback:" + hashlib.sha256(basis.encode("utf-8")).hexdigest()


async def _record_integration_event(request: AstrBotMessageRequest, trace_id: str, text: str, source_message_id: str, status: str = "received") -> None:
    raw_summary = json.dumps(redact_sensitive(request.raw), ensure_ascii=False, default=str)[:4096]
    event_hash_basis = "|".join([
        request.platform,
        request.adapter or "other",
        source_message_id,
        request.conversationId,
        request.senderId,
        text,
    ])
    event_hash = hashlib.sha256(event_hash_basis.encode("utf-8")).hexdigest()
    try:
        await _run_blocking(
            "integration-event",
            lambda: db.add_integration_event({
                "platform": request.platform,
                "adapter": request.adapter or "other",
                "sourceMessageId": source_message_id,
                "conversationId": request.conversationId,
                "conversationType": request.conversationType,
                "senderId": request.senderId,
                "eventType": "message",
                "eventHash": event_hash,
                "rawSummary": raw_summary,
                "traceId": trace_id,
                "status": status,
            }),
        )
    except Exception:
        logger.warning("AstrBot integration event log failed traceId=%s", trace_id, exc_info=True)


def _request_inference_runtime(http_request: Request):
    """Resolve the scheduler owned by this FastAPI application."""
    try:
        runtime = get_runtime_container(http_request.app).inference_runtime
    except (AttributeError, RuntimeError):
        runtime = None
    return inference_runtime if runtime is None else runtime


def _request_container_db(http_request: Request):
    """Resolve this app's container db when it differs from the global singleton.

    生成链路（消息持久化/调用记录/生成配置）与人物上下文必须读写
    同一数据库；返回 None 表示与全局单例相同，调用方直接用全局。
    """
    try:
        container_db = get_runtime_container(http_request.app).db
    except (AttributeError, RuntimeError):
        return None
    if container_db is db:
        return None
    return container_db


def _request_character_service(http_request: Request):
    """Resolve the character context service bound to this app's container db.

    生成链路的角色记忆/关系必须与角色管理接口读写同一（容器）数据库；
    容器数据库即全局单例时返回 None，回退全局默认服务。
    """
    container_db = _request_container_db(http_request)
    if container_db is None:
        return None
    from services.character_context import build_character_context_service

    return build_character_context_service(container_db)


def _degraded_response(
    request: AstrBotMessageRequest,
    trace_id: str,
    *,
    model: str,
    reply_text: str = "\u540e\u7aef\u670d\u52a1\u6682\u65f6\u7e41\u5fd9\uff0c\u8bf7\u7a0d\u540e\u518d\u8bd5\u3002",
    cost_time: float = 0.0,
) -> AstrBotMessageResponse:
    if request.conversationType == "private":
        return AstrBotMessageResponse(
            shouldReply=True,
            replyText=reply_text,
            model=model,
            costTime=cost_time,
            traceId=trace_id,
        )
    return AstrBotMessageResponse(shouldReply=False, model=model, costTime=cost_time, traceId=trace_id)


@router.post("/api/integrations/astrbot/messages", response_model=AstrBotMessageResponse)
async def receive_astrbot_message(
    request: AstrBotMessageRequest,
    http_request: Request,
    x_integration_token: str | None = Header(default=None, alias="X-Integration-Token"),
    x_integration_timestamp: str | None = Header(default=None, alias="X-Integration-Timestamp"),
    x_integration_nonce: str | None = Header(default=None, alias="X-Integration-Nonce"),
    x_integration_signature: str | None = Header(default=None, alias="X-Integration-Signature"),
):
    trace_id = uuid.uuid4().hex
    start_time = time.monotonic()

    try:
        active_token, _ = await asyncio.wait_for(_check_token(x_integration_token), timeout=_AUTH_TIMEOUT)
        if _signature_required():
            body = await http_request.body()
            verify_integration_signature(
                token=active_token,
                timestamp=x_integration_timestamp,
                nonce=x_integration_nonce,
                signature=x_integration_signature,
                body=body,
                skew_seconds=_SIGNATURE_SKEW_SECONDS,
            )
    except HTTPException as exc:
        if exc.status_code == 401:
            increment("integration_auth_failures")
            log_event("integration_auth_failed", level="warning", traceId=trace_id, platform=request.platform, conversationId=request.conversationId, senderId=request.senderId, model="auth", costTime=0, errorType=str(exc.status_code))
        raise
    except Exception as exc:
        increment("integration_auth_failures")
        log_event("integration_auth_failed", level="warning", traceId=trace_id, platform=request.platform, conversationId=request.conversationId, senderId=request.senderId, model="auth", costTime=0, errorType=type(exc).__name__)
        logger.warning("AstrBot auth unavailable traceId=%s error=%s", trace_id, exc)
        raise HTTPException(status_code=503, detail="AstrBot integration auth unavailable") from exc

    runtime = _request_inference_runtime(http_request)
    text = _validate_and_normalize_request(request)
    if _is_sensitive_admin_request(text):
        log_event("integration_security_blocked", level="warning", traceId=trace_id, platform=request.platform, conversationId=request.conversationId, senderId=redact_sensitive(request.senderId), model="security-policy", costTime=0, errorType="SecurityPolicy")
        logger.warning(
            "AstrBot blocked sensitive chat command traceId=%s platform=%s sender=%s",
            trace_id,
            request.platform,
            redact_sensitive(request.senderId),
        )
        return _degraded_response(
            request,
            trace_id,
            model="security-policy",
            reply_text="\u8be5\u8bf7\u6c42\u6d89\u53ca\u7cfb\u7edf\u914d\u7f6e\u6216\u51ed\u636e\uff0c\u5df2\u88ab\u5b89\u5168\u7b56\u7565\u62e6\u622a\u3002",
        )

    try:
        if not await _is_platform_enabled(request.platform):
            return AstrBotMessageResponse(shouldReply=False, model="platform-disabled", traceId=trace_id)
    except Exception:
        logger.warning("AstrBot platform switch check failed traceId=%s", trace_id, exc_info=True)
        return _degraded_response(request, trace_id, model="config-unavailable")

    dedup_message_id = _dedup_message_id(request, text)
    await _record_integration_event(request, trace_id, text, dedup_message_id)

    try:
        is_new = await _run_blocking(
            "dedup-message",
            lambda: db.mark_integration_message_processed(
                request.platform,
                request.adapter or "other",
                dedup_message_id,
            ),
        )
    except Exception:
        increment("db_write_failures")
        log_event("db_write_failed", level="warning", traceId=trace_id, platform=request.platform, conversationId=request.conversationId, senderId=request.senderId, model="dedup", costTime=0, errorType="DedupUnavailable")
        return _degraded_response(request, trace_id, model="db-unavailable")
    if not is_new:
        return AstrBotMessageResponse(shouldReply=False, model="duplicate", traceId=trace_id)
    try:
        await runtime.check_rate_limits(request.platform, request.conversationId, request.senderId)
    except RateLimitExceeded as exc:
        log_event("integration_rate_limited", level="warning", traceId=trace_id, platform=request.platform, conversationId=request.conversationId, senderId=request.senderId, model=f"rate-limit:{exc.scope}", costTime=0, errorType="RateLimitExceeded")
        if request.conversationType == "private":
            return AstrBotMessageResponse(
                shouldReply=True,
                replyText="\u8bf7\u6c42\u8fc7\u4e8e\u9891\u7e41\uff0c\u8bf7\u7a0d\u540e\u518d\u8bd5\u3002",
                model="rate-limit",
                traceId=trace_id,
            )
        return AstrBotMessageResponse(shouldReply=False, model=f"rate-limit:{exc.scope}", traceId=trace_id)


    session_id = _session_id(request.platform, request.conversationType, request.conversationId)
    try:
        session_enabled = await _run_blocking(
            "session-switch",
            lambda: db.is_session_bot_enabled(
                session_id,
                request.platform,
                request.conversationId,
                request.conversationType,
            ),
        )
    except Exception:
        increment("db_write_failures")
        log_event("db_write_failed", level="warning", traceId=trace_id, platform=request.platform, conversationId=request.conversationId, senderId=request.senderId, model="session-switch", costTime=0, errorType="SessionSwitchUnavailable")
        return _degraded_response(request, trace_id, model="db-unavailable")
    if not session_enabled:
        return AstrBotMessageResponse(shouldReply=False, model="session-disabled", traceId=trace_id)

    msg = MessageRequest(
        message=text,
        sessionType=request.conversationType,
        conversationType=request.conversationType,
        sessionId=session_id,
        sessionName=request.raw.get("conversationName") or request.senderName or request.conversationId,
        userId=request.senderId,
        userName=request.senderName or request.senderId,
        senderName=request.senderName or request.senderId,
        platform=request.platform,
        adapter=request.adapter or "other",
        conversationId=request.conversationId,
        senderId=request.senderId,
        sourceMessageId=dedup_message_id,
        traceId=trace_id,
    )
    priority = runtime.priority_for("astrbot", request.conversationType)
    character_service = _request_character_service(http_request)
    message_db = _request_container_db(http_request)
    try:
        result = await runtime.submit(
            lambda: generate_reply_core(
                msg,
                current_user={"username": "astrbot", "user_id": 0},
                character_service=character_service,
                message_db=message_db,
            ),
            session_id=session_id,
            priority=priority,
            timeout=_MODEL_TIMEOUT,
        )
    except InferenceQueueFull:
        log_event("integration_queue_full", level="warning", traceId=trace_id, platform=request.platform, conversationId=request.conversationId, senderId=request.senderId, model="queue-full", costTime=round(time.monotonic() - start_time, 2), errorType="InferenceQueueFull")
        return _degraded_response(request, trace_id, model="queue-full", reply_text="\u5f53\u524d\u6d88\u606f\u8f83\u591a\uff0c\u6211\u7a0d\u540e\u518d\u56de\u590d\u3002")
    except asyncio.TimeoutError:
        log_event("integration_queue_timeout", level="warning", traceId=trace_id, platform=request.platform, conversationId=request.conversationId, senderId=request.senderId, model="queue-timeout", costTime=round(time.monotonic() - start_time, 2), errorType="TimeoutError")
        return _degraded_response(request, trace_id, model="queue-timeout", reply_text="\u5f53\u524d\u5904\u7406\u8f83\u6162\uff0c\u8bf7\u7a0d\u540e\u518d\u8bd5\u3002")
    except HTTPException as exc:
        log_event("integration_generation_failed", level="warning", traceId=trace_id, platform=request.platform, conversationId=request.conversationId, senderId=request.senderId, model=f"generation-http-{exc.status_code}", costTime=round(time.monotonic() - start_time, 2), errorType=str(exc.status_code))
        logger.warning("AstrBot generation HTTP error traceId=%s status=%s", trace_id, exc.status_code)
        return _degraded_response(request, trace_id, model=f"generation-http-{exc.status_code}")
    except Exception:
        log_event("integration_generation_failed", level="warning", traceId=trace_id, platform=request.platform, conversationId=request.conversationId, senderId=request.senderId, model="generation-failed", costTime=round(time.monotonic() - start_time, 2), errorType="Exception")
        logger.warning("AstrBot generation failed traceId=%s", trace_id, exc_info=True)
        return _degraded_response(request, trace_id, model="generation-failed")

    log_event("integration_reply", traceId=trace_id, platform=request.platform, conversationId=request.conversationId, senderId=request.senderId, model=result.model, costTime=result.costTime or round(time.monotonic() - start_time, 2), errorType="")
    return AstrBotMessageResponse(
        shouldReply=True,
        replyText=result.reply,
        model=result.model,
        costTime=result.costTime or round(time.monotonic() - start_time, 2),
        traceId=trace_id,
    )
