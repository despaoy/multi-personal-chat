"""AstrBot gateway plugin for MultiPersonal Chat System.

Install this directory as an AstrBot plugin. It normalizes platform events and
forwards text messages to the MultiPersonal Chat System FastAPI backend.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register


DEFAULT_TIMEOUT = 60


def _env(name: str, legacy_name: str, default: str) -> str:
    """Prefer the MULTIPERSONAL_* variable and fall back to the legacy QQCHAT_* alias."""
    value = os.getenv(name)
    if value is not None:
        return value
    return os.getenv(legacy_name, default)


@register(
    "multipersonal_gateway",
    "multi-personal-chat",
    "Forward AstrBot multi-platform messages to MultiPersonal Chat System FastAPI.",
    "0.1.0",
)
class MultiPersonalGatewayPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self.backend_url = _env("MULTIPERSONAL_BACKEND_URL", "QQCHAT_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
        self.integration_token = os.getenv("ASTRBOT_INTEGRATION_TOKEN", "")
        self.command_prefixes = tuple(
            p.strip() for p in _env("MULTIPERSONAL_TRIGGER_PREFIXES", "QQCHAT_TRIGGER_PREFIXES", "/ai,/chat,@bot").split(",") if p.strip()
        )
        self.reply_on_group_all = _env("MULTIPERSONAL_REPLY_GROUP_ALL", "QQCHAT_REPLY_GROUP_ALL", "false").lower() == "true"
        self.timeout = float(_env("MULTIPERSONAL_BACKEND_TIMEOUT", "QQCHAT_BACKEND_TIMEOUT", str(DEFAULT_TIMEOUT)))
        self.dedup_ttl = float(_env("MULTIPERSONAL_DEDUP_TTL", "QQCHAT_DEDUP_TTL", "300"))
        self._seen_events: dict[str, float] = {}

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        text = self._message_text(event)
        if not text:
            return
        if not self._should_forward(event, text):
            return

        payload = self._build_payload(event, text)
        dedup_key = self._dedup_key(payload)
        if self._is_duplicate(dedup_key):
            logger.info("multipersonal gateway skipped duplicate event: %s", dedup_key)
            event.stop_event()
            return

        try:
            response = await asyncio.to_thread(self._post_message, payload)
        except Exception as exc:
            logger.warning(
                "multipersonal gateway request failed: platform=%s conversation=%s messageId=%s error=%s",
                payload["platform"],
                payload["conversationId"],
                payload["messageId"],
                exc,
            )
            if payload["conversationType"] == "private":
                yield event.plain_result("[\u7cfb\u7edf\u63d0\u793a] \u540e\u7aef\u670d\u52a1\u6682\u65f6\u4e0d\u53ef\u7528\uff0c\u8bf7\u7a0d\u540e\u518d\u8bd5\u3002")
            event.stop_event()
            return

        logger.info(
            "qqchat gateway traceId=%s platform=%s conversation=%s shouldReply=%s",
            response.get("traceId", ""),
            payload["platform"],
            payload["conversationId"],
            response.get("shouldReply"),
        )
        if response.get("shouldReply") and response.get("replyText"):
            yield event.plain_result(str(response["replyText"]))
        event.stop_event()

    def _message_text(self, event: AstrMessageEvent) -> str:
        text = getattr(event, "message_str", "") or ""
        return str(text).strip()

    def _should_forward(self, event: AstrMessageEvent, text: str) -> bool:
        conversation_type = self._conversation_type(event)
        if conversation_type == "private":
            return True
        if self.reply_on_group_all:
            return True
        if any(text.startswith(prefix) for prefix in self.command_prefixes):
            return True
        is_at = getattr(event, "is_at_or_wake_command", None)
        if callable(is_at):
            try:
                return bool(is_at())
            except Exception:
                return False
        return False

    def _build_payload(self, event: AstrMessageEvent, text: str) -> dict[str, Any]:
        platform = self._platform(event)
        conversation_type = self._conversation_type(event)
        conversation_id = self._conversation_id(event)
        sender_id = self._sender_id(event)
        raw = self._raw_event(event)
        return {
            "platform": platform,
            "adapter": self._adapter(platform),
            "messageId": self._message_id(event),
            "conversationId": conversation_id,
            "conversationType": conversation_type,
            "senderId": sender_id,
            "senderName": self._sender_name(event) or sender_id,
            "text": self._strip_prefix(text),
            "raw": raw,
        }

    def _post_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.backend_url}/api/integrations/astrbot/messages"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.integration_token:
            headers["X-Integration-Token"] = self.integration_token
            timestamp = str(int(time.time()))
            nonce = uuid.uuid4().hex
            headers["X-Integration-Timestamp"] = timestamp
            headers["X-Integration-Nonce"] = nonce
            headers["X-Integration-Signature"] = self._signature(timestamp, nonce, body)
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # 仅记录状态码与短预览，避免将后端完整错误 body（可能含堆栈/数据库错误）
            # 拼入异常消息后被 catch 写入运维日志，造成信息泄露
            detail = exc.read().decode("utf-8", errors="replace")
            preview = detail[:200] if detail else ""
            raise RuntimeError(f"backend returned HTTP {exc.code}: {preview}") from exc

    def _signature(self, timestamp: str, nonce: str, body: bytes) -> str:
        body_hash = hashlib.sha256(body).hexdigest()
        payload = f"{timestamp}.{nonce}.{body_hash}".encode("utf-8")
        digest = hmac.new(self.integration_token.encode("utf-8"), payload, hashlib.sha256).hexdigest()
        return "sha256=" + digest

    def _dedup_key(self, payload: dict[str, Any]) -> str:
        message_id = str(payload.get("messageId") or "").strip()
        if message_id:
            return f"{payload['platform']}:{payload['adapter']}:{message_id}"
        basis = "|".join([
            str(payload.get("platform") or ""),
            str(payload.get("adapter") or ""),
            str(payload.get("conversationId") or ""),
            str(payload.get("senderId") or ""),
            str(payload.get("text") or ""),
        ])
        return "fallback:" + hashlib.sha256(basis.encode("utf-8")).hexdigest()

    def _is_duplicate(self, key: str) -> bool:
        now = time.monotonic()
        expires_at = self._seen_events.get(key)
        if expires_at and expires_at > now:
            return True
        self._seen_events[key] = now + self.dedup_ttl
        if len(self._seen_events) > 2048:
            expired = [k for k, deadline in self._seen_events.items() if deadline <= now]
            for old_key in expired[:1024]:
                self._seen_events.pop(old_key, None)
        return False

    def _platform(self, event: AstrMessageEvent) -> str:
        candidates: list[str] = []
        getter = getattr(event, "get_platform_name", None)
        if callable(getter):
            try:
                candidates.append(str(getter()))
            except Exception:
                pass
        for attr in ("platform", "platform_id", "platform_name"):
            value = getattr(event, attr, None)
            if value:
                candidates.append(str(value))
        raw = self._raw_event(event)
        for key in ("platform", "platform_id", "platform_name", "adapter", "type"):
            value = raw.get(key)
            if value:
                candidates.append(str(value))
        name = " ".join(candidates).lower()
        if "telegram" in name:
            return "telegram"
        if "wecom" in name or "enterprise" in name:
            return "wecom"
        if "weixin_personal" in name or "personal_weixin" in name:
            return "wechat_personal"
        if "weixin_oc" in name:
            return "wechat_personal"
        if "official" in name or "mp" in name:
            return "wechat_official"
        if "wechat" in name or "gewechat" in name or "weixin" in name:
            return "wechat_personal"
        return "qq"

    def _adapter(self, platform: str) -> str:
        if platform == "qq":
            return _env("MULTIPERSONAL_QQ_ADAPTER", "QQCHAT_QQ_ADAPTER", "napcat")
        if platform == "telegram":
            return "telegram"
        if platform == "wecom":
            return "wecom"
        if platform == "wechat_official":
            return "official_account"
        if platform == "wechat_personal":
            return _env("MULTIPERSONAL_WECHAT_ADAPTER", "QQCHAT_WECHAT_ADAPTER", "gewechat")
        return "other"

    def _conversation_type(self, event: AstrMessageEvent) -> str:
        raw = self._raw_event(event)
        if raw.get("group_id") or raw.get("chat_type") in {"group", "supergroup"}:
            return "group"
        if raw.get("channel_id"):
            return "channel"
        return "private"

    def _conversation_id(self, event: AstrMessageEvent) -> str:
        raw = self._raw_event(event)
        value = raw.get("group_id") or raw.get("channel_id") or raw.get("session_id")
        if value:
            return str(value)
        getter = getattr(event, "get_session_id", None)
        if callable(getter):
            try:
                return str(getter())
            except Exception:
                pass
        return self._sender_id(event)

    def _sender_id(self, event: AstrMessageEvent) -> str:
        raw = self._raw_event(event)
        value = raw.get("user_id") or raw.get("sender_id") or raw.get("from_id")
        if value:
            return str(value)
        getter = getattr(event, "get_sender_id", None)
        if callable(getter):
            try:
                return str(getter())
            except Exception:
                pass
        return "unknown"

    def _sender_name(self, event: AstrMessageEvent) -> str:
        raw = self._raw_event(event)
        sender = raw.get("sender") if isinstance(raw.get("sender"), dict) else {}
        return str(sender.get("nickname") or sender.get("name") or raw.get("sender_name") or "")

    def _message_id(self, event: AstrMessageEvent) -> str:
        raw = self._raw_event(event)
        value = raw.get("message_id") or raw.get("id") or raw.get("msg_id")
        return str(value or "")

    def _raw_event(self, event: AstrMessageEvent) -> dict[str, Any]:
        raw = getattr(event, "raw_message", None) or getattr(event, "raw", None) or {}
        if isinstance(raw, dict):
            return raw
        if hasattr(raw, "dict"):
            try:
                return raw.dict()
            except Exception:
                return {}
        return {}

    def _strip_prefix(self, text: str) -> str:
        stripped = text.strip()
        for prefix in self.command_prefixes:
            if stripped.startswith(prefix):
                return stripped[len(prefix):].strip() or stripped
        return stripped
