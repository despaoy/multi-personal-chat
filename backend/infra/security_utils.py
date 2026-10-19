"""Security helpers shared by external integrations and admin APIs."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time
from collections.abc import Iterable
from typing import Any

from fastapi import HTTPException

SENSITIVE_KEY_RE = re.compile(
    r"(authorization|cookie|token|secret|password|api[_-]?key|credential|openid|unionid|phone|mobile)",
    re.IGNORECASE,
)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?86[-\s]?)?1[3-9]\d{9}(?!\d)")
_OPENID_LIKE_RE = re.compile(r"\b(?:openid|unionid)[=:]\s*([A-Za-z0-9_\-]{10,})", re.IGNORECASE)

_NONCE_TTL_SECONDS = int(os.getenv("INTEGRATION_NONCE_TTL_SECONDS", "300"))
_MAX_NONCES = int(os.getenv("INTEGRATION_NONCE_CACHE_MAX", "10000"))
_seen_nonces: dict[str, float] = {}


def is_production() -> bool:
    return os.getenv("ENVIRONMENT", "development").strip().lower() == "production"


def split_tokens(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def constant_time_contains(candidate: str | None, allowed_tokens: Iterable[str]) -> bool:
    if not candidate:
        return False
    return any(hmac.compare_digest(candidate, token) for token in allowed_tokens if token)


def raw_json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8"))


def mask_secret(value: Any, visible: int = 4) -> str:
    text = str(value or "")
    if not text:
        return ""
    if len(text) <= visible * 2:
        return "***"
    return f"{text[:visible]}***{text[-visible:]}"


def redact_text(text: str) -> str:
    redacted = PHONE_RE.sub(lambda m: mask_secret(m.group(0), 3), text)
    redacted = _OPENID_LIKE_RE.sub(lambda m: m.group(0).replace(m.group(1), mask_secret(m.group(1), 4)), redacted)
    return redacted


def redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if SENSITIVE_KEY_RE.search(str(key)):
                redacted[key] = mask_secret(item)
            else:
                redacted[key] = redact_sensitive(item)
        return redacted
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def _cleanup_nonces(now: float) -> None:
    expired = [key for key, expires_at in _seen_nonces.items() if expires_at <= now]
    for key in expired:
        _seen_nonces.pop(key, None)
    if len(_seen_nonces) > _MAX_NONCES:
        for key, _ in sorted(_seen_nonces.items(), key=lambda item: item[1])[: len(_seen_nonces) // 2]:
            _seen_nonces.pop(key, None)


def remember_nonce(nonce_key: str, ttl: int = _NONCE_TTL_SECONDS) -> bool:
    now = time.time()
    _cleanup_nonces(now)
    if not nonce_key:
        return False
    expires_at = _seen_nonces.get(nonce_key)
    if expires_at and expires_at > now:
        return False
    _seen_nonces[nonce_key] = now + ttl
    return True


def integration_signature(token: str, timestamp: str, nonce: str, body: bytes) -> str:
    body_hash = hashlib.sha256(body).hexdigest()
    payload = f"{timestamp}.{nonce}.{body_hash}".encode("utf-8")
    return "sha256=" + hmac.new(token.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def verify_integration_signature(
    *,
    token: str,
    timestamp: str | None,
    nonce: str | None,
    signature: str | None,
    body: bytes,
    now: float | None = None,
    skew_seconds: int = 300,
) -> None:
    if not timestamp or not nonce or not signature:
        raise HTTPException(status_code=401, detail="Missing integration signature headers")
    try:
        ts = int(timestamp)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Invalid integration timestamp") from exc
    current = int(now if now is not None else time.time())
    if abs(current - ts) > skew_seconds:
        raise HTTPException(status_code=401, detail="Integration signature expired")
    expected = integration_signature(token, timestamp, nonce, body)
    if not hmac.compare_digest(signature, expected):
        raise HTTPException(status_code=401, detail="Invalid integration signature")
    nonce_key = hashlib.sha256(f"{timestamp}:{nonce}:{signature}".encode("utf-8")).hexdigest()
    if not remember_nonce(nonce_key, ttl=skew_seconds):
        raise HTTPException(status_code=409, detail="Duplicate integration request")
